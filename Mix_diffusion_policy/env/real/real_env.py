import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import zmq

import numpy as np
from Mix_diffusion_policy.env.real.env_clients import ZMQClientRobot,ZMQClientCamera,ZMQClientFT300
from scipy.spatial.transform import Rotation as R
# 可选：若需要手指位置计算，可参考原 updateState 中的 getFingersPos
# 此处假设已有工具函数 getFingersPos，若没有可以留空或实现简单近似
try:
    from Mix_diffusion_policy.common.visual import getFingersPos
except ImportError:
    # 如果无法导入，提供一个占位实现（实际使用时需替换为正确实现）
    def getFingersPos(eef_pos, eef_quat, gripper_width_left, gripper_width_right):
        # 简单返回两个手指位置近似为 eef_pos
        return np.array(eef_pos), np.array(eef_pos)

class Rate:
    """简单的频率控制器"""
    def __init__(self, rate_hz: float):
        self.rate_hz = rate_hz
        self.last_time = time.time()

    def sleep(self):
        dt = 1.0 / self.rate_hz
        elapsed = time.time() - self.last_time
        if elapsed < dt:
            time.sleep(dt - elapsed)
        self.last_time = time.time()

'''
class RealEnv():
    """
    真实机器人环境，整合机器人状态、相机数据（点云/RGB/深度）和力/扭矩传感器。
    提供 Gym 风格的 reset / step 接口，观测可配置为平铺向量或字典。
    """
    def __init__(
        self,
        robot: ZMQClientRobot,
        control_rate_hz: float = 100.0,
        camera_dict: Optional[Dict[str, ZMQClientCamera]] = None,
	ft300: Optional[ZMQClientFT300] = None,
    ) -> None:
        self._robot = robot
        self._rate = Rate(control_rate_hz)
        self._camera_dict = {} if camera_dict is None else camera_dict
        self._ft300 = ft300
        print("RealEnv initialized")

    def robot(self) -> ZMQClientRobot:
        """Get the robot object.

        Returns:
            robot: the robot object.
        """
        return self._robot

    def __len__(self):
        return 0

    def step(self, joints: np.ndarray) -> Dict[str, Any]:
        """Step the environment forward.

        Args:
            joints: joint angles command to step the environment with.

        Returns:
            obs: observation from the environment.
        """
        assert len(joints) == (
            self._robot.num_dofs()
        ), f"input:{len(joints)}, robot:{self._robot.num_dofs()}"
        assert self._robot.num_dofs() == len(joints)
        self._robot.command_joint_state(joints)
        self._rate.sleep()
        return self.get_obs()

    def get_obs(self) -> Dict[str, Any]:
        """Get observation from the environment.

        Returns:
            obs: observation from the environment.
        """
        observations = {}
        for name, camera in self._camera_dict.items():
            #image, depth, pc = camera.read()
            pc = camera.read()
            #print(pc)
            observations[f"{name}_rgb"] = image
            observations[f"{name}_depth"] = depth
            observations[f"{name}_pointcloud"] = pc
        robot_obs = self._robot.get_observations()
        assert "joint_positions" in robot_obs
        assert "joint_velocities" in robot_obs
        assert "ee_pos_quat" in robot_obs
        observations["joint_positions"] = robot_obs["joint_positions"]
        observations["joint_velocities"] = robot_obs["joint_velocities"]
        observations["ee_pos_quat"] = robot_obs["ee_pos_quat"]
        observations["gripper_position"] = robot_obs["gripper_position"]
        force_torque = self._ft300.get_force_torque()
        observations["ee_force_torque"] = force_torque
        return observations
'''
def rotate_pose_by_z180(pose: np.ndarray) -> np.ndarray:
    """
    将位姿 [x, y, z, qx, qy, qz, qw] 绕 Z 轴旋转 180 度。
    四元数顺序: (qx, qy, qz, qw) 与 scipy 一致。
    """
    if pose.shape[-1] != 7:
        raise ValueError("Pose must be a 7D array (x,y,z,qx,qy,qz,qw)")
    pos = pose[:3]
    quat = pose[3:7]   # (qx, qy, qz, qw)
    
    # 位置变换: x,y 取反
    pos_new = np.array([-pos[0], -pos[1], pos[2]])
    
    # 姿态变换: 左乘绕 Z 轴 180° 的四元数
    R_z180 = R.from_quat([0, 0, 1, 0])   # (qx,qy,qz,qw) = (0,0,1,0) 对应绕 Z 轴 180°
    R_ee = R.from_quat(quat)
    R_new = R_z180 * R_ee
    quat_new = R_new.as_quat()           # 返回 (qx, qy, qz, qw)
    
    return np.concatenate([pos_new, quat_new])

import threading
import time
from typing import Any, Dict, Optional

class RealEnv:
    def __init__(
        self,
        robot: ZMQClientRobot,
        control_rate_hz: float = 100.0,
        camera_dict: Optional[Dict[str, ZMQClientCamera]] = None,
        ft300: Optional[ZMQClientFT300] = None,
        obs_update_hz: float = 25.0,   # 后台观测更新频率，默认等于控制频率
    ) -> None:
        self._robot = robot
        self._control_rate_hz = control_rate_hz
        self._rate = Rate(control_rate_hz)
        self._camera_dict = {} if camera_dict is None else camera_dict
        self._ft300 = ft300

        # 观测缓存和线程安全锁
        self._obs_cache: Dict[str, Any] = {}
        self._obs_lock = threading.Lock()
        self._stop_thread = False
        self._update_thread: Optional[threading.Thread] = None

        # 启动后台观测更新线程
        update_hz = obs_update_hz 
        self._start_obs_updater(update_hz)

        # 等待首次观测成功（避免 KeyError）
        timeout = 5.0
        start_time = time.time()
        while not self._obs_cache and (time.time() - start_time) < timeout:
            time.sleep(0.01)
        if not self._obs_cache:
            raise RuntimeError(
                "Failed to retrieve initial observation. Check robot/camera/FT sensor connection."
            )

    def _start_obs_updater(self, hz: float) -> None:
        """启动后台线程，以指定频率更新观测数据"""
        def updater():
            interval = 1.0 / hz
            while not self._stop_thread:
                start = time.perf_counter()
                # 获取最新观测（可能阻塞）
                new_obs = self._blocking_get_obs()
                # 更新缓存
                with self._obs_lock:
                    self._obs_cache = new_obs
                elapsed = time.perf_counter() - start
                if elapsed < interval:
                    time.sleep(interval - elapsed)

        self._update_thread = threading.Thread(target=updater, daemon=True)
        self._update_thread.start()

    def _blocking_get_obs(self) -> Dict[str, Any]:
        """
        原始阻塞式观测获取，在后台线程中调用。
        包含相机、机器人状态、力传感器等可能耗时的 I/O 操作。
        """
        observations = {}

        # 读取相机点云（可能耗时）
        for name, camera in self._camera_dict.items():
            pc = camera.read()
            observations[f"{name}_pointcloud"] = pc

        # 读取机器人状态（可能涉及网络）
        robot_obs = self._robot.get_observations()
        assert "joint_positions" in robot_obs
        assert "joint_velocities" in robot_obs
        assert "ee_pos_quat" in robot_obs
        observations["joint_positions"] = robot_obs["joint_positions"]
        observations["joint_velocities"] = robot_obs["joint_velocities"]
        observations["ee_pos_quat"] = rotate_pose_by_z180(robot_obs["ee_pos_quat"])
        observations["gripper_position"] = robot_obs["gripper_position"]

        # 读取力传感器（可能涉及网络）
        if self._ft300 is not None:
            try:
                force_torque = self._ft300.get_force_torque()
                observations["ee_force_torque"] = force_torque
            except Exception as e:
                # 保留上一次的力数据（如果缓存中有），否则设为 None
                with self._obs_lock:
                    prev_ft = self._obs_cache.get("ee_force_torque", None)
                observations["ee_force_torque"] = prev_ft

        return observations

    def get_obs(self) -> Dict[str, Any]:
        """非阻塞返回最新观测（深拷贝一份，避免主线程意外修改）"""
        with self._obs_lock:
            # 返回字典的浅拷贝已足够，因为内部数据应为不可变或独立对象
            return dict(self._obs_cache)

    def step(self, joints: np.ndarray) -> Dict[str, Any]:
        """
        发送控制指令，并维持控制频率。
        返回最新的观测数据（非阻塞）。
        """
        assert len(joints) == self._robot.num_dofs()
        self._robot.command_joint_state(joints)
        self._rate.sleep()          # 保证控制频率
        return self.get_obs()       # 立即返回最新缓存观测

    def robot(self) -> ZMQClientRobot:
        """返回底层机器人对象"""
        return self._robot

    def __len__(self) -> int:
        return 0

    def __del__(self) -> None:
        """析构时通知后台线程停止"""
        self._stop_thread = True
        if self._update_thread is not None:
            self._update_thread.join(timeout=1.0)

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
    start_time = time.time()
    obs = real_env.get_obs()
    end_time = time.time()
    print(f"Time elapsed: {end_time - start_time:.3f}s")
    print(obs.keys())