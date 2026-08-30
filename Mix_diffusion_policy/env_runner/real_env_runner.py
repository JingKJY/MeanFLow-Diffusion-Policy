import os
import math
import time
import pathlib
import collections
from typing import Optional, List, Dict, Any, Union

import numpy as np
import torch
import tqdm
import wandb
import dill

from Mix_diffusion_policy.policy.base_pcd_policy import BasePcdPolicy
from Mix_diffusion_policy.common.pytorch_util import dict_apply
from Mix_diffusion_policy.env_runner.base_pcd_runner import BasePcdRunner
from Mix_diffusion_policy.common.replay_buffer import ReplayBuffer
from Mix_diffusion_policy.common.transformation import transPts_tq_npbatch
from Mix_diffusion_policy.model.common.rotation_transformer import RotationTransformer

# 假设已有模块
from Mix_diffusion_policy.env.real.real_pcd_wrapper_new import RealEnvWrapper
from Mix_diffusion_policy.env.real.real_env_new import RealEnv
from scipy.spatial.transform import Rotation as R, Slerp

class RealEnvRunner():
    """
    真实机器人环境 Runner，支持：
    - 单环境或多环境（实际多环境需要多个真实机器人，通常为1）
    - 使用 replay_buffer 提供的静态场景/物体点云（可选）
    - 子目标预测（可选）
    - 视频录制（从相机读取 RGB 图像）
    - 动作限幅、安全保护
    """
    def __init__(
        self,
        env: RealEnv,
        action_key: str = 'actions_joint',
        obs_keys: List[str] = None,
        abs_action: bool = True,
        rotation_rep: str = 'rotation_6d',
        observation_history_num: int = 2,
        use_subgoal: bool = False,
        camera_names: List[str] = ['base', 'wrist'],
        ik_solver: Optional[Any] = None,
        control_hz: float = 100.0,
        gripper_max_open: float = 0.085,
        reset_target_joints: Optional[np.ndarray] = None,
        reset_steps: int = 25,
        reset_max_step_delta: float = 0.05,
        check_action_delta_limit: bool = True,
        action_delta_limit: float = 0.8,
        max_steps: int = 200,
        n_action_steps: int = 8,
        n_latency_steps: int = 0,
        visualize_pcd_subgoal: bool = False,
        visualize_every: int = 1,
        visualize_point_size: float = 2.0,
        visualize_subgoal_radius: float = 0.015,
        use_rgb: bool = False,
    ):

        # 保存关键参数
        self.action_key = action_key
        self.abs_action = abs_action
        self.max_steps = max_steps
        self.n_action_steps = n_action_steps
        self.n_latency_steps = n_latency_steps
        self.observation_history_num = observation_history_num
        self.use_subgoal = use_subgoal
        self.rot_transformer = None
        self.visualize_pcd_subgoal = visualize_pcd_subgoal
        self.visualize_every = max(1, int(visualize_every))
        self.visualize_point_size = visualize_point_size
        self.visualize_subgoal_radius = visualize_subgoal_radius
        self.use_rgb = use_rgb
        self._vis = None
        self._vis_enabled = visualize_pcd_subgoal

        # 创建 Wrapper
        if obs_keys is None:
            obs_keys = ['state_joint', 'state_ee', 'gripper', 'force_torque']
        self.wrapper = RealEnvWrapper(
            env=env,
            action_key=action_key,
            obs_keys=obs_keys,
            abs_action=abs_action,
            rotation_rep=rotation_rep,
            observation_history_num=observation_history_num,
            use_subgoal=use_subgoal,
            camera_names=camera_names,
            ik_solver=ik_solver,
            control_hz=control_hz,
            gripper_max_open=gripper_max_open,
            reset_target_joints=reset_target_joints,
            reset_steps=reset_steps,
            reset_max_step_delta=reset_max_step_delta,
            check_action_delta_limit=check_action_delta_limit,
            action_delta_limit=action_delta_limit,
            use_rgb=use_rgb,
        )

        # 若使用绝对末端执行器动作，需要旋转转换器
        if abs_action and action_key == 'actions_ee':
            self.rot_transformer = RotationTransformer(from_rep='quaternion', to_rep=rotation_rep)

        # 打印初始化信息
        print("[RealEnvRunner] init")
        print(f"  action_key: {action_key}, abs_action: {abs_action}")
        print(f"  max_steps: {max_steps}, n_action_steps: {n_action_steps}, n_latency_steps: {n_latency_steps}")
        print(f"  use_subgoal: {use_subgoal}, observation_history_num: {observation_history_num}")
        print(f"  use_rgb: {use_rgb}")
        print(f"  visualize_pcd_subgoal: {visualize_pcd_subgoal}, visualize_every: {self.visualize_every}")

    def run(self, policy: BasePcdPolicy) -> Dict[str, Any]:
        """
        执行评估 rollout。
        """
        device = policy.device
        dtype = policy.dtype

        print("[RealEnvRunner] starting policy rollout")
        # 重置环境（会移动到初始关节位置）
        np_obs_dict = self.wrapper.reset()
        print("  -env reset done")
        policy.reset()
        steps = 0
        self.prev_action = None
        while steps < self.max_steps:

            print(f"  --- step {steps}/{self.max_steps} ---")

            Tinput_dict = dict_apply(np_obs_dict, lambda x: torch.from_numpy(x).to(device=device, dtype=dtype))
            for key in Tinput_dict.keys():
                Tinput_dict[key] = Tinput_dict[key].unsqueeze(0)  # (1, ...)
                
            if self.use_subgoal:
                # 使用策略的子目标生成器预测子目标
                with torch.no_grad():
                    subgoal = policy.guider_predict_target(Tinput_dict).detach().cpu().numpy()
                    #print(f"predict subgoal: {subgoal}")
                    
                Tinput_dict['subgoal'] = torch.from_numpy(subgoal).to(device=device, dtype=dtype) # (1, subgoal_dim)
                #print(f"    predict subgoal shape: {subgoal.shape}")
                self._visualize_inference(np_obs_dict, subgoal, steps)


            # 策略推理动作
            with torch.no_grad():
                action_dict = policy.actor_predict_action(Tinput_dict)  # dict{'action', ...}
            np_action_dict = dict_apply(action_dict,
                                    lambda x: x.detach().cpu().numpy())

            # 处理延迟步数（丢弃前 n_latency_steps 个动作）
            action_seq = np_action_dict['action']  # (1, n_action_steps, action_dim)
            if self.n_latency_steps > 0:
                action_seq = action_seq[:, self.n_latency_steps:, :]

            # 如果需要绝对动作变换（将 rotation_6d 转回四元数等）
            env_action = action_seq
            if self.abs_action and self.action_key == 'actions_ee':
                env_action = self.undo_transform_action(action_seq)

            # 执行动作序列（wrapper 会依次执行所有动作）
            env_action=env_action.squeeze(0)  # (n_action_steps, action_dim)
            env_action = env_action[:self.n_action_steps]
            #print(env_action)
            env_action = self.smooth_action_chunk(env_action)
            #print(env_action)
            start_time = time.time()
            np_obs_dict = self.wrapper.step(env_action, return_intermediate=False)
            self.prev_action = env_action[-1].copy()
            print(f"step {steps} done, time: {time.time() - start_time:.3f}s")

            # 更新步数
            steps += self.n_action_steps

        print(f"[RealEnvRunner] finished policy rollout after {steps}/{self.max_steps} steps")
        self._close_visualizer()
        # 返回最后一次的观测字典（可自行扩展）
        return np_obs_dict

    def _make_sphere(self, center: np.ndarray, color: List[float], radius: Optional[float] = None):
        import open3d as o3d

        radius = self.visualize_subgoal_radius if radius is None else radius
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
        sphere.translate(np.asarray(center, dtype=np.float64))
        sphere.paint_uniform_color(color)
        return sphere

    def _visualize_inference(self, obs_dict: Dict[str, np.ndarray], subgoal: Optional[np.ndarray], steps: int) -> None:
        if not self._vis_enabled or (steps % self.visualize_every) != 0:
            return

        try:
            import open3d as o3d

            pcd_frame = np.asarray(obs_dict["pcd"][-1], dtype=np.float64)
            pcd_frame = pcd_frame.reshape(-1, pcd_frame.shape[-1])
            pcd_np = pcd_frame[:, :3]
            finite_mask = np.isfinite(pcd_np).all(axis=1)
            pcd_np = pcd_np[finite_mask]

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pcd_np)
            if pcd_frame.shape[1] >= 6:
                colors = pcd_frame[:, 3:6][finite_mask]
                if colors.size > 0 and np.nanmax(colors) <= 1.0:
                    colors = colors * 255.0
                colors = np.clip(colors, 0.0, 255.0) / 255.0
                pcd.colors = o3d.utility.Vector3dVector(colors)
            else:
                pcd.paint_uniform_color([0.25, 0.25, 0.25])

            geometries = [pcd, o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.08)]

            subgoal_np = None if subgoal is None else np.asarray(subgoal).reshape(-1)
            if subgoal_np is not None and subgoal_np.shape[0] >= 6:
                left_goal = subgoal_np[:3]
                right_goal = subgoal_np[3:6]
                if subgoal_np.shape[0] >= 8:
                    left_active = bool(np.round(subgoal_np[6]).astype(np.int32))
                    right_active = bool(np.round(subgoal_np[7]).astype(np.int32))
                else:
                    left_active = True
                    right_active = True

                if left_active:
                    geometries.append(self._make_sphere(left_goal, [1.0, 0.0, 0.0]))
                if right_active:
                    geometries.append(self._make_sphere(right_goal, [0.13, 0.55, 0.13]))

                if left_active and right_active:
                    line_set = o3d.geometry.LineSet()
                    line_set.points = o3d.utility.Vector3dVector(np.stack([left_goal, right_goal], axis=0))
                    line_set.lines = o3d.utility.Vector2iVector(np.array([[0, 1]], dtype=np.int32))
                    line_set.colors = o3d.utility.Vector3dVector(np.array([[1.0, 0.8, 0.0]], dtype=np.float64))
                    geometries.append(line_set)
            else:
                left_active = False
                right_active = False

            state_np = np.asarray(obs_dict["state"][-1]).reshape(-1)
            if state_np.shape[0] >= 6:
                left_finger = state_np[-6:-3]
                right_finger = state_np[-3:]
                geometries.append(self._make_sphere(left_finger, [0.0, 0.0, 0.0], radius=self.visualize_subgoal_radius * 0.7))
                geometries.append(self._make_sphere(right_finger, [0.0, 0.0, 0.0], radius=self.visualize_subgoal_radius * 0.7))

                finger_line = o3d.geometry.LineSet()
                finger_line.points = o3d.utility.Vector3dVector(np.stack([left_finger, right_finger], axis=0))
                finger_line.lines = o3d.utility.Vector2iVector(np.array([[0, 1]], dtype=np.int32))
                finger_line.colors = o3d.utility.Vector3dVector(np.array([[0.0, 0.0, 0.0]], dtype=np.float64))
                geometries.append(finger_line)

                if subgoal_np is not None and subgoal_np.shape[0] >= 6:
                    left_dist = np.linalg.norm(left_goal - left_finger)
                    right_dist = np.linalg.norm(right_goal - right_finger)
                    print(
                        "[RealEnvRunner] visual "
                        f"step={steps} "
                        f"left_goal={np.round(left_goal, 4)} right_goal={np.round(right_goal, 4)} "
                        f"active=({int(left_active)}, {int(right_active)}) "
                        f"left_finger={np.round(left_finger, 4)} right_finger={np.round(right_finger, 4)} "
                        f"dist=({left_dist:.4f}, {right_dist:.4f})"
                    )

            if self._vis is None:
                self._vis = o3d.visualization.Visualizer()
                self._vis.create_window(window_name="test_run pcd/subgoal", width=1280, height=720)
                render_opt = self._vis.get_render_option()
                render_opt.point_size = self.visualize_point_size
                render_opt.background_color = np.array([1.0, 1.0, 1.0])

            self._vis.clear_geometries()
            for geometry_idx, geometry in enumerate(geometries):
                self._vis.add_geometry(geometry, reset_bounding_box=(geometry_idx == 0))
            self._vis.poll_events()
            self._vis.update_renderer()
        except Exception as exc:
            print(f"[RealEnvRunner] disable pcd/subgoal visualization after error: {exc}")
            self._vis_enabled = False
            self._close_visualizer()

    def _close_visualizer(self) -> None:
        if self._vis is not None:
            try:
                self._vis.destroy_window()
            except Exception:
                pass
            self._vis = None

    def _safe_quat(self, quat: np.ndarray, fallback: Optional[np.ndarray] = None) -> np.ndarray:
        quat = np.asarray(quat, dtype=np.float32).copy()
        norm = np.linalg.norm(quat)
        if np.isfinite(norm) and norm > 1e-6:
            return quat / norm

        if fallback is not None:
            fallback = np.asarray(fallback, dtype=np.float32).copy()
            fallback_norm = np.linalg.norm(fallback)
            if np.isfinite(fallback_norm) and fallback_norm > 1e-6:
                return fallback / fallback_norm

        return np.array([0, 0, 0, 1], dtype=np.float32)

    def smooth_action_chunk(
        self,
        actions: np.ndarray,
        pos_alpha: float = 0.6,
        rot_alpha: float = 0.45,
        gripper_alpha: float = 0.9,
        max_pos_delta: float = 0.03,
        max_rot_delta: float = 0.08,
        max_gripper_delta: float = 0.25,
        blend_steps: int = 1,
        has_gripper: bool = True,
    ) -> np.ndarray:
        """
        平滑 diffusion policy 输出的 action chunk。

        默认 action 格式:
            [x, y, z, qx, qy, qz, qw, gripper]

        核心处理:
            1. 当前 chunk 开头与上一 chunk 末尾做过渡融合
            2. 限制相邻 action 的最大位置/姿态变化
            3. 对位置、姿态、夹爪分别做低通滤波
        """

        actions = np.asarray(actions, dtype=np.float32)

        if actions.ndim != 2:
            raise ValueError(f"actions should be (T, action_dim), got {actions.shape}")

        T, action_dim = actions.shape
        smoothed = actions.copy()

        if hasattr(self, "prev_action") and self.prev_action is not None:
            prev = self.prev_action.astype(np.float32).copy()
        else:
            prev = actions[0].astype(np.float32).copy()

        prev[:7] = self._normalize_pose_quat(prev[:7])

        # -----------------------------
        # 1. chunk 头部融合
        # -----------------------------
        # diffusion policy 每次都会重新采样一段动作，
        # 当前 chunk 的第一个动作可能和上一个执行动作差很多。
        # 因此对前 blend_steps 步做 warm-start blending。
        if self.prev_action is not None and blend_steps > 0:
            n_blend = min(blend_steps, T)

            for i in range(n_blend):
                # w 从小到大，越靠后越相信当前模型输出
                w = float(i + 1) / float(n_blend + 1)

                # 位置线性融合
                smoothed[i, :3] = (1.0 - w) * prev[:3] + w * actions[i, :3]

                # 姿态 slerp 融合
                q_prev = prev[3:7]
                q_raw = actions[i, 3:7]

                q_prev = self._safe_quat(q_prev)
                q_raw = self._safe_quat(q_raw, fallback=q_prev)

                if np.dot(q_prev, q_raw) < 0:
                    q_raw = -q_raw

                key_rots = R.from_quat([q_prev, q_raw])
                slerp = Slerp([0, 1], key_rots)
                smoothed[i, 3:7] = slerp([w]).as_quat()[0]

                # 夹爪融合
                if has_gripper:
                    smoothed[i, 7] = (1.0 - w) * prev[7] + w * actions[i, 7]

        # -----------------------------
        # 2. 对整个 chunk 做逐步限速 + 低通
        # -----------------------------
        output = smoothed.copy()
        running_prev = prev.copy()

        for t in range(T):
            raw = smoothed[t].astype(np.float32).copy()
            raw[:7] = self._normalize_pose_quat(raw[:7])

            out = raw.copy()

            # 2.1 位置限速 + EMA
            pos_delta = raw[:3] - running_prev[:3]
            pos_norm = np.linalg.norm(pos_delta)

            if pos_norm > max_pos_delta:
                pos_delta = pos_delta / (pos_norm + 1e-8) * max_pos_delta

            limited_pos = running_prev[:3] + pos_delta
            out[:3] = pos_alpha * limited_pos + (1.0 - pos_alpha) * running_prev[:3]

            # 2.2 姿态限速 + Slerp
            q_prev = running_prev[3:7]
            q_raw = raw[3:7]

            q_prev = self._safe_quat(q_prev)
            q_raw = self._safe_quat(q_raw, fallback=q_prev)

            if np.dot(q_prev, q_raw) < 0:
                q_raw = -q_raw

            r_prev = R.from_quat(q_prev)
            r_raw = R.from_quat(q_raw)

            r_rel = r_prev.inv() * r_raw
            rotvec = r_rel.as_rotvec()
            angle = np.linalg.norm(rotvec)

            if angle > max_rot_delta:
                rotvec = rotvec / (angle + 1e-8) * max_rot_delta
                r_limited = r_prev * R.from_rotvec(rotvec)
            else:
                r_limited = r_raw

            key_rots = R.from_quat([
                r_prev.as_quat(),
                r_limited.as_quat(),
            ])
            slerp = Slerp([0, 1], key_rots)
            out[3:7] = slerp([rot_alpha]).as_quat()[0]

            # 2.3 夹爪单独限速 + EMA
            if has_gripper:
                g_delta = np.clip(
                    raw[7] - running_prev[7],
                    -max_gripper_delta,
                    max_gripper_delta,
                )
                limited_g = running_prev[7] + g_delta
                out[7] = gripper_alpha * limited_g + (1.0 - gripper_alpha) * running_prev[7]

            out[:7] = self._normalize_pose_quat(out[:7])
            output[t] = out
            running_prev = out

        return output


    def _normalize_pose_quat(self, pose: np.ndarray) -> np.ndarray:
        """
        归一化 [x, y, z, qx, qy, qz, qw] 中的四元数。
        """

        pose = pose.copy()
        q = pose[3:7]
        pose[3:7] = self._safe_quat(q)

        return pose
    def undo_transform_action(self, action: np.ndarray) -> np.ndarray:
        """
        将策略输出的绝对动作（pos + rotation_6d + gripper）转换为环境需要的格式。
        环境期望：若是 actions_joint 则直接关节角度；若是 actions_ee 则 pos+quat+gripper。
        这里假设 action_key='actions_ee' 且 abs_action=True。
        """
        # action shape: (B, horizon, 3+6+1) = (B, horizon, 10)
        # 转换为 (B, horizon, 7+1) = (pos(3), quat(4), gripper(1))
        pos = action[..., :3]
        rot6d = action[..., 3:9]
        gripper = action[..., 9:10]
        quat = self.rot_transformer.inverse(rot6d)  # 转换为四元数 (B, horizon, 4)
        uaction = np.concatenate([pos, quat, gripper], axis=-1)
        #print(f"    [undo_transform] 将 rotation_6d 转换为四元数，最终动作 shape: {uaction.shape}")
        return uaction
