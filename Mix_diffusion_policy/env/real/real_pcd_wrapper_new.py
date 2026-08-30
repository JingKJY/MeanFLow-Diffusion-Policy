import time
import numpy as np
from collections import deque
from typing import Dict, Optional, Union, List, Any
from scipy.spatial.transform import Rotation

from Mix_diffusion_policy.model.common.rotation_transformer import RotationTransformer
from Mix_diffusion_policy.env.real.real_env import RealEnv
from Mix_diffusion_policy.env.real.env_clients import ZMQClientRobot, ZMQClientCamera, ZMQClientFT300
from ur_analytic_ik import ur5


def calc_two_fingers_all(obs_state_ee, obs_gripper):
    """同数据集中的实现（保持不变）"""
    TOOL2GRIPPER_Z = 0.20
    W_MAX_HALF = 0.0425
    N = obs_state_ee.shape[0]
    left_finger_list = []
    right_finger_list = []
    for t in range(N):
        pos = obs_state_ee[t, :3]
        qx, qy, qz, qw = obs_state_ee[t, 3:7]
        g_q = obs_gripper[t, 0] if obs_gripper.ndim > 1 else obs_gripper[t]
        R_mat = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
        T_tool0 = np.eye(4)
        T_tool0[:3, :3] = R_mat
        T_tool0[:3, 3] = pos
        T_gripper = T_tool0 @ np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, TOOL2GRIPPER_Z],
            [0, 0, 0, 1]
        ])
        half_w = W_MAX_HALF * g_q
        left_pt = np.array([-half_w, 0.0, 0.0, 1.0])
        right_pt = np.array([ half_w, 0.0, 0.0, 1.0])
        left_world = (T_gripper @ left_pt)[:3]
        right_world = (T_gripper @ right_pt)[:3]
        left_finger_list.append(left_world)
        right_finger_list.append(right_world)
    return np.array(left_finger_list), np.array(right_finger_list)


class RealEnvWrapper:
    def __init__(
        self,
        env: Any,
        action_key: str = 'actions_joint',
        obs_keys: List[str] = ['state_joint', 'state_ee', 'gripper', 'force_torque'],
        abs_action: bool = True,
        rotation_rep: str = 'rotation_6d',
        observation_history_num: int = 2,
        use_subgoal: bool = False,
        camera_names: List[str] = ['base', 'wrist'],
        ik_solver: Optional[Any] = None,
        control_hz: float = 25.0,
        gripper_max_open: float = 0.085,
        reset_target_joints: Optional[np.ndarray] = None,
        reset_steps: int = 25,
        reset_max_step_delta: float = 0.05,
        check_action_delta_limit: bool = True,
        action_delta_limit: float = 0.8,
        # ----- 新增：点云变换参数（可选，带默认值，不影响原接口）-----
        left_rot_quat: Optional[np.ndarray] = None,      # 左侧相机固定外参旋转四元数(x,y,z,w)
        left_trans: Optional[np.ndarray] = None,         # 左侧相机固定外参平移
        right_rot_quat: Optional[np.ndarray] = None, # 右侧相机固定外参旋转四元数(x,y,z,w)
        right_trans: Optional[np.ndarray] = None,    # 右侧相机固定外参平移
        pcd_npoints: int = 1024,
        use_rgb: bool = False,
    ):
        self.env = env
        self.action_key = action_key
        self.obs_keys = obs_keys
        self.abs_action = abs_action
        self.observation_history_num = observation_history_num
        self.use_subgoal = use_subgoal
        self.ik_solver = ur5 if ik_solver is None else ik_solver
        self.gripper_max_open = gripper_max_open
        self.check_action_delta_limit = check_action_delta_limit
        self.action_delta_limit = action_delta_limit
        self.camera_names = camera_names
        self.reset_target_joints = reset_target_joints
        self.reset_steps = reset_steps
        self.reset_max_step_delta = reset_max_step_delta
        self.pcd_npoints = pcd_npoints
        self.use_rgb = use_rgb

        self.rot_transformer = RotationTransformer('axis_angle', 'rotation_6d')
        self._state_history = deque(maxlen=observation_history_num)
        self._pcd_history = deque(maxlen=observation_history_num)
        if use_subgoal:
            self._subgoal_history = deque(maxlen=1)

        self._control_dt = 1.0 / control_hz
        self._last_step_time = None
        self.dof = 7
        self.gripper_joint_indices = slice(-2, None) if self.dof >= 8 else slice(-1, None)
        self.current_joints = np.zeros(self.dof-1)
        if action_key == 'actions_ee':
            self.action_dim = 3 + 6 + 1
        elif action_key == 'actions_joint':
            self.action_dim = self.dof
        else:
            raise ValueError(f"Unsupported action_key: {action_key}")

        # ----- 点云变换参数（默认使用标定值，与数据集处理一致）-----
        # base相机→世界（固定）
        if left_rot_quat is None:
            # 四元数 (w,x,y,z)
            left_rot_quat = np.array([-0.91950325629, 0.0866294472802, -0.217637317384, 0.315662950975])
        if left_trans is None:
            left_trans = np.array([-0.263618073856, -0.806245531009, 0.458467688855])
        # wrist相机→末端（固定）
        if right_rot_quat is None:
            right_rot_quat = np.array([-0.900016690794, -0.0954383787414, 0.227065195379, 0.35958708153])
        if right_trans is None:
            right_trans = np.array([0.302589964549, -0.822491517206, 0.474515465596])

        self.left_rot = Rotation.from_quat(left_rot_quat)   
        self.left_trans = left_trans
        self.right_rot = Rotation.from_quat(right_rot_quat)
        self.right_trans = right_trans

        # 打印初始化信息
        print("[RealEnvWrapper] initialized with point cloud transformation (base fixed + wrist via ee)")

    def _fix_pointcloud_shape(self, points: np.ndarray) -> np.ndarray:
        """
        将每帧真实点云固定为 (pcd_npoints, 3/6)。

        真实相机每帧有效点数可能变化；如果不固定点数，_pcd_history 中不同帧
        shape 不一致，np.stack 会报 all input arrays must have the same shape。
        """
        feature_dim = 6 if self.use_rgb else 3
        points = np.asarray(points, dtype=np.float32)
        points = points.reshape(-1, points.shape[-1])[:, :feature_dim]
        points = points[np.isfinite(points).all(axis=1)]

        if points.shape[0] == 0:
            return np.zeros((self.pcd_npoints, feature_dim), dtype=np.float32)

        if points.shape[0] >= self.pcd_npoints:
            # 等间隔采样保持确定性，避免实时 rollout 中随机采样引入额外抖动。
            idx = np.linspace(0, points.shape[0] - 1, self.pcd_npoints).astype(np.int64)
            points = points[idx]
        else:
            pad_num = self.pcd_npoints - points.shape[0]
            pad_idx = np.arange(pad_num) % points.shape[0]
            points = np.concatenate([points, points[pad_idx]], axis=0)

        return points.astype(np.float32)

    # ----------------------------- 辅助方法 -----------------------------
    def _compute_ee_pose_from_joints(self, joints: np.ndarray) -> np.ndarray:
        """
        通过正运动学从关节角度 (6维) 计算末端位姿 (7: x,y,z, qx,qy,qz,qw)
        """
        if len(joints) < 6:
            raise ValueError(f"Need at least 6 joint angles, got {len(joints)}")
        matrix = self.ik_solver.forward_kinematics(*joints[:6])
        pos = matrix[:3, 3]
        quat = Rotation.from_matrix(matrix[:3, :3]).as_quat()  # (x,y,z,w)
        # 转换为 (qx, qy, qz, qw)
        quat = np.array([quat[0], quat[1], quat[2], quat[3]])
        return np.concatenate([pos, quat])

    def _transform_pointcloud_base(self, points: np.ndarray) -> np.ndarray:
        """将 base 相机点云转换到世界坐标系"""
        flat = points.reshape(-1, 3)
        transformed = self.base_rot.apply(flat) + self.base_trans
        return transformed.reshape(points.shape)

    def _transform_pointcloud_wrist(self, points: np.ndarray, ee_pose: np.ndarray) -> np.ndarray:
        """
        将 wrist 相机点云转换到世界坐标系
        points: (N, 3)  wrist相机坐标系下的点云
        ee_pose: (7,)  世界坐标系下的末端位姿 [x,y,z, qx,qy,qz,qw]
        """
        # 相机 -> 末端
        points_ee = self.wrist_cam_rot.apply(points) + self.wrist_cam_trans
        # 末端 -> 世界
        ee_pos = ee_pose[:3]
        ee_rot = Rotation.from_quat(ee_pose[3:7])  # (qx,qy,qz,qw)
        points_world = ee_rot.apply(points_ee) + ee_pos
        return points_world

    def _get_pointcloud(self, raw_obs: Dict) -> np.ndarray:
        """
        获取并转换点云：
        - base_pointcloud 使用固定外参转世界
        - wrist_pointcloud 根据当前末端位姿（正运动学）转世界
        """
        def extract_pointcloud(pc_data: Any) -> np.ndarray:
            if isinstance(pc_data, dict):
                points = pc_data.get("points", pc_data.get("pointcloud", None))
                colors = pc_data.get("colors", None)
            else:
                points = pc_data
                colors = None

            if points is None:
                return np.empty((0, 6 if self.use_rgb else 3), dtype=np.float32)

            points = np.asarray(points, dtype=np.float32).reshape(-1, np.asarray(points).shape[-1])
            xyz = points[:, :3]
            if not self.use_rgb:
                return xyz[np.isfinite(xyz).all(axis=1)]

            if points.shape[1] >= 6:
                rgb = points[:, 3:6]
            elif colors is not None:
                rgb = np.asarray(colors, dtype=np.float32).reshape(-1, np.asarray(colors).shape[-1])[:, :3]
                if rgb.shape[0] != xyz.shape[0]:
                    rgb = np.zeros_like(xyz, dtype=np.float32)
            else:
                rgb = np.zeros_like(xyz, dtype=np.float32)
            if rgb.size > 0 and np.nanmax(rgb) > 1.0:
                rgb = rgb / 255.0
            rgb = np.clip(rgb, 0.0, 1.0)
            pcd = np.concatenate([xyz, rgb], axis=-1)
            return pcd[np.isfinite(pcd).all(axis=1)]

        left_pcd = extract_pointcloud(raw_obs["left_pointcloud"])
        right_pcd = extract_pointcloud(raw_obs["right_pointcloud"])



        from scipy.spatial.transform import Rotation as R
        def transform_pointcloud(points, rotation, translation):
            """
            points: (..., 3) 任意形状，最后一维是 xyz
            rotation: scipy Rotation 对象
            translation: (3,) ndarray
            返回: 与 points 相同形状的变换后点云
            """
            xyz = points[:, :3]
            transformed_xyz = rotation.apply(xyz) + translation
            if points.shape[1] > 3:
                return np.concatenate([transformed_xyz, points[:, 3:]], axis=-1)
            return transformed_xyz
        # 变换
        left_pcd_world = transform_pointcloud(left_pcd, self.left_rot, self.left_trans)
        right_pcd_world = transform_pointcloud(right_pcd, self.right_rot, self.right_trans)
        # 拼接
        #combined = np.concatenate([base_pcd_world, wrist_pcd_world], axis=0)
        combined = np.concatenate([left_pcd_world, right_pcd_world], axis=0)
        return self._fix_pointcloud_shape(combined)

    def _build_state(self, raw_obs: Dict) -> np.ndarray:
        """
        构建状态向量：末端位姿（正运动学计算）+ 关节位置 + 夹爪开度 + 力/力矩 + 左右指尖
        """
        # 获取关节位置（全 DOF）
        joints = raw_obs['joint_positions'].astype(np.float32)
        # 用正运动学计算末端位姿（世界坐标系）
        ee_pose = self._compute_ee_pose_from_joints(joints)   # (7,)

        # 夹爪开度（归一化到 [0,1]）
        if 'gripper_position' in raw_obs:
            gripper = raw_obs['gripper_position'].astype(np.float32)
            if gripper.ndim > 1:
                gripper = gripper[0:1]
        else:
            # 从 joint_positions 最后几个提取（根据 DOF）
            gripper_vals = joints[self.gripper_joint_indices]
            # 假设夹爪开度范围 [0, 0.085] 映射到 [0,1]
            gripper = (gripper_vals.mean() / self.gripper_max_open).astype(np.float32).reshape(1)
            gripper = np.clip(gripper, 0.0, 1.0)

        # 力/力矩
        force_torque = np.asarray(raw_obs.get('ee_force_torque', np.zeros(6)), dtype=np.float32)

        # 计算指尖位置（基于正运动学得到的末端位姿和夹爪开度）
        left_f, right_f = calc_two_fingers_all(
            obs_state_ee=ee_pose.reshape(1, -1),
            obs_gripper=gripper.reshape(1, -1)
        )
        left_f = left_f[0]    # (3,)
        right_f = right_f[0]

        # 按 obs_keys 顺序拼接
        state_parts = []
        for k in self.obs_keys:
            if k == 'state_joint':
                state_parts.append(joints)
            elif k == 'state_ee':
                state_parts.append(ee_pose)          # 使用正运动学计算的末端位姿
            elif k == 'state_gripper':
                state_parts.append(gripper)
            elif k == 'force_torque':
                state_parts.append(force_torque)
            else:
                if k in raw_obs:
                    state_parts.append(raw_obs[k].astype(np.float32))
                else:
                    raise KeyError(f"Unknown obs_key {k}")
        state_parts.append(left_f)
        state_parts.append(right_f)
        self.current_joints = joints[:6]  # 更新当前关节位置
        state = np.concatenate(state_parts).astype(np.float32)
        return state

    def _move_to_joints(self, target_joints: np.ndarray) -> None:
        """平滑移动到目标关节位置（同原实现）"""
        target_joints = np.asarray(target_joints)
        print(f"[RealEnvWrapper] start moving to joints: {target_joints}")
        current = self.env.get_obs()['joint_positions']
        print(f"[RealEnvWrapper] current joints: {current}")
        final_obs = self.env.get_obs()
        for step in range(self.reset_steps):
            alpha = (step + 1) / self.reset_steps
            desired = current + (target_joints - current) * alpha
            delta = desired - current
            if self.reset_max_step_delta is not None:
                max_delta = np.abs(delta).max()
                if max_delta > self.reset_max_step_delta:
                    delta = delta / max_delta * self.reset_max_step_delta
            next_cmd = current + delta
            final_obs=self.env.step(next_cmd)
            current = self.env.get_obs()['joint_positions']
            self.current_joints = current[:6]
        if not np.allclose(final_obs['joint_positions'], target_joints, atol=0.01):
            self.env.step(target_joints)

    # ----------------------------- 核心执行 -----------------------------
    def _execute_one_step(self, action: np.ndarray, current_joints: np.ndarray) -> Dict[str, np.ndarray]:
        #time_1 = time.time()
        # 1. 根据动作模式计算目标关节命令
        if self.action_key == 'actions_ee':
            target_ee_pose = action[:7]
            gripper_val = action[7]
            if self.ik_solver is None:
                raise RuntimeError("IK solver required for 'actions_ee' mode")
            target_ee_matrix = self._pose_to_matrix(pos=target_ee_pose[:3], quat_xyzw=target_ee_pose[3:])
            solutions = self.ik_solver.inverse_kinematics(target_ee_matrix)
            if not solutions:
                # 可选：使用上一次的解或保持当前位置（根据你的容错策略）
                raise RuntimeError("IK solution not found")
            
            solutions_arr = np.array(solutions)
            distances = np.linalg.norm(solutions_arr - current_joints, axis=1)
            best_idx = np.argmin(distances)
            target_joints = solutions_arr[best_idx]
            target_joints = np.concatenate([target_joints, [gripper_val]])
        else:  # actions_joint
            target_joints = action.copy()
            if len(target_joints) != self.dof:
                raise ValueError(f"Joint action dimension mismatch: expected {self.dof}, got {len(target_joints)}")
        #print(f"time compute target joints: {time.time() - time_1:.3f}s")

        # 2. 动作限幅保护
        delta_joints = target_joints[:6] - current_joints[:6]
        if self.check_action_delta_limit:
            if np.abs(delta_joints).max() > self.action_delta_limit:
                idx = np.where(np.abs(delta_joints) > self.action_delta_limit)[0]
                for i in idx:
                    print(f"Warning: joint[{i}] command {delta_joints[i]:.3f} exceeds limit {self.action_delta_limit}")
                delta_clipped = np.clip(delta_joints, -self.action_delta_limit, self.action_delta_limit)
                target_joints[:6] = current_joints[:6] + delta_clipped

        # 3. 频率控制
        now = time.time()
        if self._last_step_time is not None:
            elapsed = now - self._last_step_time
            if elapsed < self._control_dt:
                time.sleep(self._control_dt - elapsed)
        self._last_step_time = time.time()

        # 4. 执行命令
        #time2 = time.time()
        raw_obs = self.env.step(target_joints)
        #print(f"time execute step: {time.time() - time2:.3f}s")  
        

        # 5. 构造观测并更新历史
        #time3 = time.time()
        new_state = self._build_state(raw_obs)
        new_pcd = self._get_pointcloud(raw_obs)
        self._state_history.append(new_state)
        self._pcd_history.append(new_pcd)
        #print(f"time build state and pcd: {time.time() - time3:.3f}s")  

        obs = {
            'state': np.stack(list(self._state_history), axis=0),
            'pcd': np.stack(list(self._pcd_history), axis=0),
        }
        return obs

    # ----------------------------- 对外接口 -----------------------------
    def step(self, action: Union[np.ndarray, List[np.ndarray]], return_intermediate: bool = False) -> Union[Dict, List[Dict]]:
        if not isinstance(action, np.ndarray):
            action = np.array(action)
        if action.ndim == 1:
            return self._execute_one_step(action)
        elif action.ndim == 2:
            horizon = action.shape[0]
            obs_list = []
            obs = None
            for i in range(horizon):
                #print(f"[RealEnvWrapper] action[{i}]: {action[i]}")
                obs = self._execute_one_step(action[i],self.current_joints)
                if return_intermediate:
                    obs_list.append(obs)
            if return_intermediate:
                return obs_list
            else:
                return obs
        else:
            raise ValueError(f"Unsupported action shape: {action.shape}")

    def reset(self) -> Dict[str, np.ndarray]:
        print("[RealEnvWrapper] start reset")
        target = self.reset_target_joints
        if target is not None:
            if len(target) != self.dof:
                raise ValueError(f"Target joints length {len(target)} != dof {self.dof}")
            self._move_to_joints(target)
            time.sleep(2)

        print("[RealEnvWrapper] start get_obs")
        time1 = time.time()
        raw_obs = self.env.get_obs()
        print(raw_obs)
        time2 = time.time()
        print(f"[RealEnvWrapper] get_obs time: {time2 - time1:.3f}s")

        state = self._build_state(raw_obs)
        pcd = self._get_pointcloud(raw_obs)

        self._state_history.clear()
        self._pcd_history.clear()
        for _ in range(self.observation_history_num):
            self._state_history.append(state)
            self._pcd_history.append(pcd)

        obs = {
            'state': np.stack(list(self._state_history), axis=0),
            'pcd': np.stack(list(self._pcd_history), axis=0),
        }
        return obs

    def get_obs(self) -> Dict[str, np.ndarray]:
        obs = {
            'state': np.stack(list(self._state_history), axis=0),
            'pcd': np.stack(list(self._pcd_history), axis=0),
        }
        return obs

    def _pose_to_matrix(self, pos, quat_xyzw):
        """Convert (x,y,z) position and (w,x,y,z) quaternion to 4x4 homogeneous matrix."""
        from scipy.spatial.transform import Rotation as R
        rot = R.from_quat([quat_xyzw[0], quat_xyzw[1], quat_xyzw[2], quat_xyzw[3]]).as_matrix()
        mat = np.eye(4)
        mat[:3, :3] = rot
        mat[:3, 3] = pos
        return mat
