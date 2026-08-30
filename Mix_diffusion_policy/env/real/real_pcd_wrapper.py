import time
import numpy as np
from collections import deque
from typing import Dict, Optional, Union, List, Any
from scipy.spatial.transform import Rotation

# 假设已有模块

from Mix_diffusion_policy.model.common.rotation_transformer import RotationTransformer
from Mix_diffusion_policy.env.real.real_env import RealEnv
from Mix_diffusion_policy.env.real.env_clients import ZMQClientRobot,ZMQClientCamera,ZMQClientFT300
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
        R = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
        T_tool0 = np.eye(4)
        T_tool0[:3, :3] = R
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


class RealEnvWrapper():
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
        # 新增：姿态初始化参数
        reset_target_joints: Optional[np.ndarray] = None,      # 直接指定目标关节位置
        reset_steps: int = 25,                                 # 平滑移动步数
        reset_max_step_delta: float = 0.05,                    # 每步最大关节变化（rad）
        # 新增：动作限幅参数             # 运行时单步最大允许关节变化（rad）
        check_action_delta_limit: bool = True,                   # 是否检查绝对动作过大
        action_delta_limit: float = 0.8,                         # 绝对动作上限（关节角度 rad）
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
        # 姿态初始化相关
        self.reset_target_joints = reset_target_joints
        self.reset_steps = reset_steps
        self.reset_max_step_delta = reset_max_step_delta

        self.rot_transformer = RotationTransformer('axis_angle', 'rotation_6d')

        self._state_history = deque(maxlen=observation_history_num)
        self._pcd_history = deque(maxlen=observation_history_num)
        if use_subgoal:
            self._subgoal_history = deque(maxlen=1)

        self._control_dt = 1.0 / control_hz
        self._last_step_time = None

        self.dof = self.env.robot().num_dofs()
        self.gripper_joint_indices = slice(-2, None) if self.dof >= 8 else slice(-1, None)

        if action_key == 'actions_ee':
            self.action_dim = 3 + 6 + 1
        elif action_key == 'actions_joint':
            self.action_dim = self.dof
        else:
            raise ValueError(f"Unsupported action_key: {action_key}")
        # 打印初始化信息
        print("[RealEnvWrapper] initialized with:")
        print(f"  action_key: {action_key}, abs_action: {abs_action}")
        print(f"  dof: {self.dof}, action_dim: {self.action_dim}")
        print(f"  observation_history_num: {observation_history_num}, use_subgoal: {use_subgoal}")
        print(f"  gripper_max_open: {gripper_max_open}")
        print(f"  check_action_delta_limit: {check_action_delta_limit}, action_delta_limit: {action_delta_limit}")
        print(f"  reset_target_joints: {reset_target_joints}")
        print(f"  reset_steps: {reset_steps}, reset_max_step_delta: {reset_max_step_delta}")
    # ----------------------------- 辅助方法 -----------------------------
    def _transform_pointcloud(self, pts: np.ndarray) -> np.ndarray:
        flat = pts.reshape(-1, 3)
        transformed = self.pcd_rot.apply(flat) + self.pcd_trans
        return transformed.reshape(pts.shape)

    def _get_pointcloud(self, raw_obs: Dict) -> np.ndarray:
        if self.camera_names is None:
            return np.concatenate([raw_obs[f"{name}_pointcloud"] for name in self.camera_names], axis=0).astype(np.float32)
        base_key = "base_pointcloud"
        wrist_key = "wrist_pointcloud"
        base_pcd = raw_obs[base_key]
        wrist_pcd = raw_obs[wrist_key]
        #print(f"[RealEnvWrapper] base_pcd:{base_pcd['points']}")
        #print(f"[RealEnvWrapper] wrist_pcd:{wrist_pcd['points']}")  
        combined = np.concatenate([base_pcd['points'], wrist_pcd['points']], axis=0)
        return combined.astype(np.float32)

    def _build_state(self, raw_obs: Dict) -> np.ndarray:
        joint = raw_obs['joint_positions'].astype(np.float32)
        ee = raw_obs['ee_pos_quat'].astype(np.float32)
        gripper = raw_obs.get('gripper_position', np.zeros(1)).astype(np.float32)
        if gripper.ndim > 1:
            gripper = gripper[0:1]
        force_torque = np.asarray(raw_obs.get('ee_force_torque', np.zeros(6)), dtype=np.float32)
        left_f, right_f = calc_two_fingers_all(ee.reshape(1, -1), gripper.reshape(1, -1))
        left_f = left_f[0]; right_f = right_f[0]
        state_parts = []
        for k in self.obs_keys:
            if k == 'state_joint':
                state_parts.append(joint)
            elif k == 'state_ee':
                state_parts.append(ee)
            elif k == 'gripper':
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
        '''
        print(f"[RealEnvWrapper] joint:{joint.shape}")
        print(f"[RealEnvWrapper] ee:{ee.shape}")
        print(f"[RealEnvWrapper] gripper:{gripper.shape}")
        print(f"[RealEnvWrapper] force_torque:{force_torque.shape}")
        print(f"[RealEnvWrapper] left_f:{left_f.shape}")
        print(f"[RealEnvWrapper] right_f:{right_f.shape}")'''
        state = np.concatenate(state_parts).astype(np.float32)
        return state

    def _move_to_joints(self, target_joints: np.ndarray) -> None:
        """
        平滑移动到目标关节位置，使用步进插值和限幅。
        """
        print(f"[RealEnvWrapper] start moving to joints: {target_joints}")
        current = self.env.get_obs()['joint_positions']
        print(f"[RealEnvWrapper] current joints: {current}")
        for step in range(self.reset_steps):
            alpha = (step + 1) / self.reset_steps
            desired = current + (target_joints - current) * alpha
            # 限制单步变化量
            delta = desired - current
            if self.reset_max_step_delta is not None:
                max_delta = np.abs(delta).max()
                if max_delta > self.reset_max_step_delta:
                    delta = delta / max_delta * self.reset_max_step_delta
            next_cmd = current + delta
            self.env.step(next_cmd)
            current = self.env.get_obs()['joint_positions']
            # 控制频率（已在 env.step 中 sleep，这里无需额外等待）
        # 最后确保达到目标（如果还有微小误差，直接设置）
        final_obs = self.env.get_obs()
        if not np.allclose(final_obs['joint_positions'], target_joints, atol=0.01):
            self.env.step(target_joints)

    # ----------------------------- 核心执行 -----------------------------
    def _execute_one_step(self, action: np.ndarray) -> Dict[str, np.ndarray]:
        # 1. 根据动作模式计算目标关节命令
        if self.action_key == 'actions_ee':
            target_ee_pose = action[:7]
            gripper_val = action[7]
            current_joints = self.env.get_obs()['joint_positions'][:6]
            if self.ik_solver is None:
                raise RuntimeError("IK solver required for 'actions_ee' mode")
            target_ee_matrix=self._pose_to_matrix(pos=target_ee_pose[:3],quat_xyzw=target_ee_pose[3:])
            solutions = self.ik_solver.inverse_kinematics(target_ee_matrix)
            #solutions = self.ik_solver.inverse_kinematics(target_ee_matrix, current_joints)
            if solutions is None:# 处理失败：保持原关节、报错或重新规划
                raise RuntimeError("IK solution not found")
            else:
                distances = [np.linalg.norm(sol - current_joints) for sol in solutions]
                best_idx = np.argmin(distances)
                target_joints = solutions[best_idx]
                target_joints = np.concatenate([target_joints, [gripper_val]])
                #target_joints = np.concatenate([solutions, [gripper_val]])
        else:  # actions_joint
            target_joints = action.copy()
            if len(target_joints) != self.dof:
                raise ValueError(f"Joint action dimension mismatch: expected {self.dof}, got {len(target_joints)}")

        # 2. 动作限幅保护
        current_joints = self.env.get_obs()['joint_positions']
        delta_joints = target_joints[:6] - current_joints[:6]
        # 3. 可选：检查绝对命令是否过大
        if self.check_action_delta_limit:
            # 只检查目标关节是否超出合理范围（例如绝对值 > limit）
            # 这里假设关节限位对称，可根据实际情况修改
            if np.abs(delta_joints).max() > self.action_delta_limit:
                # 打印警告或抛出异常
                idx = np.where(np.abs(delta_joints) > self.action_delta_limit)[0]
                for i in idx:
                    print(f"Warning: joint[{i}] command {delta_joints[i]:.3f} exceeds limit {self.action_delta_limit}")
                # 可选择裁剪
                delta_clipped = np.clip(delta_joints, -self.action_delta_limit, self.action_delta_limit)
                target_joints[:6] = current_joints[:6] + delta_clipped
        now = time.time()
        if self._last_step_time is not None:
            elapsed = now - self._last_step_time
            if elapsed < self._control_dt:
                time.sleep(self._control_dt - elapsed)
        self._last_step_time = time.time()
        # 4. 执行命令
        raw_obs=self.env.step(target_joints)

        # 5. 构造观测并更新历史
        new_state = self._build_state(raw_obs)
        new_pcd = self._get_pointcloud(raw_obs)
        self._state_history.append(new_state)
        self._pcd_history.append(new_pcd)

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
            obs=None
            for i in range(horizon):
                print(f"[RealEnvWrapper] action[{i}]: {action[i]}")
                obs = self._execute_one_step(action[i])
                if return_intermediate:
                    obs_list.append(obs)
            if return_intermediate:
                return obs_list
            else:
                return obs
        else:
            raise ValueError(f"Unsupported action shape: {action.shape}")

    def reset(self) -> Dict[str, np.ndarray]:
        """
        重置环境，并可选择将机器人移动到起始姿态。
        """
        # 1. 确定目标起始关节位置
        print("[RealEnvWrapper] start reset")
        target = self.reset_target_joints
        if target is not None:
            if len(target) != self.dof:
                raise ValueError(f"Target joints length {len(target)} != dof {self.dof}")
            self._move_to_joints(target)
            time.sleep(2)  # 等待末端稳定

        # 2. 获取最终观测并填充历史队列
        print("[RealEnvWrapper] start get_obs")
        time1 = time.time()
        raw_obs = self.env.get_obs()
        time2 = time.time()
        print(f"[RealEnvWrapper] get_obs time: {time2-time1:.3f}s")
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
'''
# 创建包装器，带姿态初始化和动作限幅
wrapper = RealEnvWrapper(
    env=real_env,
    action_key='actions_joint',
    reset_target_joints=np.array([0, -1.57, 1.57, -1.57, -1.57, 0, 0, 0]),  # 起始关节角度
    reset_steps=30,
    reset_max_step_delta=0.03,
    max_joint_step=0.1,          # 运行时单步最大变化 0.1 rad
    check_action_abs_limit=True,
    action_abs_limit=2.5,        # 关节绝对角度限幅
)

# 重置（会先移动到起始位置）
obs = wrapper.reset()

# 执行动作（若动作超出限幅会被自动裁剪）
action = policy(obs)   # 假设输出关节角度
next_obs = wrapper.step(action)'''
import argparse   
def parse_args():
    parser = argparse.ArgumentParser(description="Test realenv.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Server IP address")
    parser.add_argument("--robot_port", type=int, default=6001, help="Robot server port")
    parser.add_argument("--camera_base_port", type=int, default=5001, help="Camera server port")
    parser.add_argument("--camera_wrist_port", type=int, default=5000, help="Camera server port")
    parser.add_argument("--ft300_port", type=int, default=5002, help="FT300 server port")
    return parser.parse_args()    

def main():
    args = parse_args()
    robot_client = ZMQClientRobot(port=args.robot_port, host=args.host)
    camera_client_base = ZMQClientCamera(port=args.camera_base_port, host=args.host)
    camera_client_wrist = ZMQClientCamera(port=args.camera_wrist_port, host=args.host)
    ft300_client = ZMQClientFT300(port=args.ft300_port, host=args.host)


    real_env = RealEnv(robot=robot_client, camera_dict={"base": camera_client_base, "wrist": camera_client_wrist}, ft300=ft300_client)
    