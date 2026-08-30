import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import zmq

import numpy as np
from scipy.spatial.transform import Rotation as R
import pickle
import threading

DEFAULT_ENV_PORT = 5555


class ZMQEnvClient:
    """
    统一环境客户端：
    - 发送动作指令并获得完整观测
    - 获取全部或指定观测
    """

    def __init__(
        self,
        port: int = DEFAULT_ENV_PORT,
        host: str = "127.0.0.1",
        timeout_ms: int = 5000,
    ):
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.REQ)
        self._socket.connect(f"tcp://{host}:{port}")
        self._socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
        self._timeout_ms = timeout_ms

    def step(self, joint_state: np.ndarray) -> Dict[str, Any]:
        """
        发送关节指令，机器人执行后返回所有观测。
        joint_state: np.ndarray 目标关节位置
        返回: {'status': 'ok', 'obs': {...}} 或 {'error': ...}
        """
        req = {
            "type": "action",
            "args": {"joint_state": joint_state.tolist()},
        }
        self._socket.send(pickle.dumps(req))
        try:
            resp = pickle.loads(self._socket.recv())
        except zmq.Again:
            raise TimeoutError("Env step timed out")
        if "error" in resp:
            raise RuntimeError(resp["error"])
        return resp["obs"]

    def get_obs(self) -> Dict[str, Any]:
        """获取所有当前观测（不执行动作）"""
        req = {"type": "get_obs", "args": {}}
        self._socket.send(pickle.dumps(req))
        try:
            resp = pickle.loads(self._socket.recv())
        except zmq.Again:
            raise TimeoutError("Get obs timed out")
        if "error" in resp:
            raise RuntimeError(resp["error"])
        return resp["obs"]

    def get_obs_key(self, key: str) -> Any:
        """获取指定观测值，例如 'robot_joint_state', 'ft_force_torque', 'cam1', 'cam2'"""
        req = {"type": "get_obs_key", "args": {"key": key}}
        self._socket.send(pickle.dumps(req))
        try:
            resp = pickle.loads(self._socket.recv())
        except zmq.Again:
            raise TimeoutError(f"Get obs key '{key}' timed out")
        if "error" in resp:
            raise RuntimeError(resp["error"])
        return resp[key]

    def close(self) -> None:
        """关闭连接"""
        self._socket.close()
        self._context.term()

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

class RealEnv:
    """
    远程机器人环境封装，通过统一环境客户端操作机器人、力传感器和相机。
    动作执行后立即返回包含关节状态、力传感器和多个相机点云的观测。
    """

    def __init__(
        self,
        env_client: ZMQEnvClient,
        control_rate_hz: float = 100.0,
        camera_key_mapping: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Args:
            env_client: 统一环境客户端（已连接服务器）
            control_rate_hz: 控制步进频率，用于 rate.sleep()
            camera_key_mapping: 服务器返回的相机名到用户期望名的映射
                例: {"cam1": "base", "cam2": "wrist"}
                若不提供，则使用 {"cam1": "cam1", "cam2": "cam2"}
        """
        self._client = env_client
        self._rate = Rate(control_rate_hz)
        self._cam_mapping = camera_key_mapping or {"cam1": "left", "cam2": "right"}

    def robot(self):
        """返回客户端对象（可获取 num_dofs 等）"""
        return self._client

    def __len__(self):
        return 0

    def step(self, joints: np.ndarray) -> Dict[str, Any]:
        """
        执行关节命令，等待动作完成，然后返回全部最新观测。

        Args:
            joints: 目标关节角度 (numpy array)

        Returns:
            obs: 映射为原格式的观测字典，包含：
                - joint_positions
                - ee_force_torque
                - {camera_name}_pointcloud (根据映射)
        """
        # 通过客户端发送动作并获取观测（服务器先执行动作再采集）
        #time1=time.time()
        obs_raw = self._client.step(joints)
        #print(f"[env] step+obs:{time.time()-time1}")
        # 维持设定频率
        return self._process_obs(obs_raw)

    def get_obs(self) -> Dict[str, Any]:
        """获取当前观测（不执行动作）"""
        obs_raw = self._client.get_obs()
        return self._process_obs(obs_raw)

    def _process_obs(self, obs_raw: Dict[str, Any]) -> Dict[str, Any]:
        """
        将服务端原始观测键转换为与本机直接操作时一致的键名。
        保持原有代码兼容性。
        """
        processed = {}

        # 机器人关节位置
        if "robot_joint_state" in obs_raw:
            processed["joint_positions"] = np.array(obs_raw["robot_joint_state"])

        # 力传感器
        if "ft_force_torque" in obs_raw:
            processed["ee_force_torque"] = np.array(obs_raw["ft_force_torque"])

        # 相机点云（根据映射字典）
        # 相机点云（根据映射字典）
        for server_cam, user_cam in self._cam_mapping.items():
            pc_key = f"{server_cam}_pointcloud"
            if pc_key in obs_raw:
                processed[f"{user_cam}_pointcloud"] = obs_raw[pc_key]


        # 如需扩展其他字段（如关节速度、末端位姿等），在此处添加映射即可
        return processed

import argparse   
def parse_args():
    parser = argparse.ArgumentParser(description="Test realenv.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Server IP address")
    parser.add_argument("--env_port", type=int, default=5555, help="Robot server port")
    return parser.parse_args()    

def main():
    args = parse_args()
    env_client = ZMQEnvClient(port=args.env_port, host=args.host)

    real_env = RealEnv(env_client)
    times=[]
    for i in range(100):
        start_time = time.time()
        obs = real_env.get_obs()
        #for key in obs.keys():
            #print(key)
        end_time = time.time()
    times.append(end_time - start_time)
    print(f"Time elapsed: {np.mean(times)}s.3f")
    print(obs.keys())

if __name__ == "__main__":
    main()
