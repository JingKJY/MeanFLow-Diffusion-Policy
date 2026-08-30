#!/usr/bin/env python3
"""
测试 ZMQ 客户端与服务器的连接
用法：
    python test_clients.py --host <服务器IP> 
        --robot_port 6001 --camera_port 5000 --ft300_port 5002
"""

import argparse
import numpy as np
from typing import Optional,Tuple,List,Dict
import time
# 假设 client 类定义在 client_modules 模块中，这里直接复制类定义以便独立运行
# 实际使用时请根据项目结构调整 import
# from your_client_module import ZMQClientRobot, ZMQClientCamera, ZMQClientFT300

# ---------- 客户端类定义（为独立运行而复制，实际可导入） ----------
import pickle
import zmq

DEFAULT_CAMERA_PORT = 5000
DEFAULT_ROBOT_PORT = 6000

import threading
import queue
import pickle
import zmq

import open3d as o3d
DEFAULT_CAMERA_PORT = 5555

class ZMQClientCamera():
    def __init__(self, port: int = DEFAULT_CAMERA_PORT, host: str = "127.0.0.1"):
        # 原有的命令 socket（用于保留将来可能的写操作，目前无）
        self._context = zmq.Context()
        self._control_socket = self._context.socket(zmq.REQ)
        self._control_socket.connect(f"tcp://{host}:{port}")

        # 专门用于接收数据流的 socket（后台线程专用）
        self._data_socket = self._context.socket(zmq.REQ)
        self._data_socket.connect(f"tcp://{host}:{port}")

        # 缓存与线程控制
        self._cache_lock = threading.Lock()
        self._cached_data = None  # (image, depth, pointcloud)
        self._stop_event = threading.Event()
        self._refresh_rate_hz = 25  # 30 Hz 后台刷新频率，可根据需要调整
        self._requested_img_size = None  # 最近一次 read 传人的尺寸
        self._refresh_thread = threading.Thread(target=self._refresh_loop, daemon=True)
        self._refresh_thread.start()

    def _refresh_loop(self):
        """后台线程：持续请求最新图像/点云并更新缓存"""
        interval = 1.0 / self._refresh_rate_hz
        while not self._stop_event.is_set():
            start = time.time()
            try:
                # 使用当前请求的 img_size（可能为 None）
                with self._cache_lock:
                    img_size = self._requested_img_size
                send_message = pickle.dumps(img_size)
                self._data_socket.send(send_message)
                # 注意：服务器端返回的是 (image, depth, pointcloud) 整体
                recv_data = pickle.loads(self._data_socket.recv())
                # 服务器端可能返回 None 表示数据未就绪
                if recv_data is not None:
                    with self._cache_lock:
                        self._cached_data = recv_data
            except Exception as e:
                # 发生错误时保留原有缓存
                # print(f"[Camera] refresh error: {e}")
                pass
            elapsed = time.time() - start
            sleep_time = interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def read(self, img_size: Optional[Tuple[int, int]] = None) -> Tuple[np.ndarray, np.ndarray, Optional[dict]]:
        # 更新请求尺寸，供后台线程使用
        with self._cache_lock:
            self._requested_img_size = img_size
            data = self._cached_data
        if data is None:
            # 如果还没有任何缓存，返回空数据（可以按需抛出异常或返回 None）
            return (np.array([]), np.array([]), None)
        return data

    def close(self):
        self._stop_event.set()
        if self._refresh_thread.is_alive():
            self._refresh_thread.join(timeout=1.0)
        self._control_socket.close()
        self._data_socket.close()
        self._context.term()


class ZMQClientFT300:
    def __init__(self, port=5001, host="127.0.0.1"):
        self.context = zmq.Context()
        # 控制 socket（用于写操作，如 reset_zero）
        self.control_socket = self.context.socket(zmq.REQ)
        self.control_socket.connect(f"tcp://{host}:{port}")
        self.control_socket.setsockopt(zmq.RCVTIMEO, 5000)

        # 读取专用 socket（后台线程使用）
        self.data_socket = self.context.socket(zmq.REQ)
        self.data_socket.connect(f"tcp://{host}:{port}")
        self.data_socket.setsockopt(zmq.RCVTIMEO, 100)  # 短超时保证线程及时退出

        # 缓存数据
        self._lock = threading.Lock()
        self._cached_force_torque: Optional[List[float]] = None
        self._cached_acceleration: Optional[List[float]] = None

        self._stop_event = threading.Event()
        self._refresh_rate_hz = 50  # 50 Hz
        self._refresh_thread = threading.Thread(target=self._refresh_loop, daemon=True)
        self._refresh_thread.start()

    def _refresh_loop(self):
        interval = 1.0 / self._refresh_rate_hz
        while not self._stop_event.is_set():
            start = time.time()
            # 依次请求两个高频数据
            try:
                # 请求力/力矩
                msg = pickle.dumps(("get_force_torque", None))
                self.data_socket.send(msg)
                ft = pickle.loads(self.data_socket.recv())
                if not isinstance(ft, Exception):
                    with self._lock:
                        self._cached_force_torque = ft

                # 请求加速度
                msg = pickle.dumps(("get_acceleration", None))
                self.data_socket.send(msg)
                acc = pickle.loads(self.data_socket.recv())
                if not isinstance(acc, Exception):
                    with self._lock:
                        self._cached_acceleration = acc
            except zmq.Again:
                # 超时，继续循环
                pass
            except Exception:
                pass
            elapsed = time.time() - start
            sleep_time = interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _send_command(self, cmd, args=None):
        # 写操作仍然使用 control_socket，实时发送并等待应答
        msg = pickle.dumps((cmd, args))
        self.control_socket.send(msg)
        resp = pickle.loads(self.control_socket.recv())
        if isinstance(resp, Exception):
            raise resp
        return resp

    def get_force_torque(self):
        with self._lock:
            if self._cached_force_torque is None:
                # 如果第一次调用还没有缓存，可以同步请求一次
                return self._send_command("get_force_torque")
            return self._cached_force_torque

    def get_acceleration(self):
        with self._lock:
            if self._cached_acceleration is None:
                return self._send_command("get_acceleration")
            return self._cached_acceleration

    # 其他写操作（如 reset_zero）保持不变，直接使用 _send_command
    def reset_zero(self):
        return self._send_command("reset_zero")

    def disconnect(self):
        return self._send_command("disconnect")

    def close(self):
        self._stop_event.set()
        if self._refresh_thread.is_alive():
            self._refresh_thread.join(timeout=1.0)
        self.control_socket.close()
        self.data_socket.close()
        self.context.term()

class ZMQClientRobot():
    def __init__(self, port: int = DEFAULT_ROBOT_PORT, host: str = "127.0.0.1"):
        self._context = zmq.Context()
        # 控制指令 socket（用于 command_joint_state 等写操作）
        self._control_socket = self._context.socket(zmq.REQ)
        self._control_socket.connect(f"tcp://{host}:{port}")

        # 数据读取专用 socket（后台线程）
        self._data_socket = self._context.socket(zmq.REQ)
        self._data_socket.connect(f"tcp://{host}:{port}")

        # 缓存
        self._lock = threading.Lock()
        self._cached_num_dofs = None
        self._cached_joint_state = None
        self._cached_observations = None

        self._stop_event = threading.Event()
        self._refresh_rate_hz = 50
        self._refresh_thread = threading.Thread(target=self._refresh_loop, daemon=True)
        self._refresh_thread.start()

    def _refresh_loop(self):
        interval = 1.0 / self._refresh_rate_hz
        while not self._stop_event.is_set():
            start = time.time()
            try:
                # 请求 num_dofs（通常不变，但定期刷新也无妨）
                req = pickle.dumps({"method": "num_dofs"})
                self._data_socket.send(req)
                dofs = pickle.loads(self._data_socket.recv())
                if not (isinstance(dofs, dict) and "error" in dofs):
                    with self._lock:
                        self._cached_num_dofs = dofs

                # 请求 joint_state
                req = pickle.dumps({"method": "get_joint_state"})
                self._data_socket.send(req)
                js = pickle.loads(self._data_socket.recv())
                if not (isinstance(js, dict) and "error" in js):
                    with self._lock:
                        self._cached_joint_state = js

                # 请求 observations
                req = pickle.dumps({"method": "get_observations"})
                self._data_socket.send(req)
                obs = pickle.loads(self._data_socket.recv())
                if not (isinstance(obs, dict) and "error" in obs):
                    with self._lock:
                        self._cached_observations = obs
            except zmq.Again:
                pass
            except Exception:
                pass
            elapsed = time.time() - start
            sleep_time = interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def num_dofs(self) -> int:
        with self._lock:
            if self._cached_num_dofs is None:
                # 同步请求
                request = {"method": "num_dofs"}
                self._control_socket.send(pickle.dumps(request))
                return pickle.loads(self._control_socket.recv())
            return self._cached_num_dofs

    def get_joint_state(self) -> np.ndarray:
        with self._lock:
            if self._cached_joint_state is None:
                request = {"method": "get_joint_state"}
                self._control_socket.send(pickle.dumps(request))
                result = pickle.loads(self._control_socket.recv())
                if isinstance(result, dict) and "error" in result:
                    raise RuntimeError(result["error"])
                return result
            return self._cached_joint_state

    def command_joint_state(self, joint_state: np.ndarray) -> None:
        # 写操作直接发送，不经过缓存，且加锁防止与后台线程干扰
        with self._lock:
            request = {
                "method": "command_joint_state",
                "args": {"joint_state": joint_state},
            }

            self._control_socket.send(pickle.dumps(request))
            result = pickle.loads(self._control_socket.recv())
            if isinstance(result, dict) and "error" in result:
                raise RuntimeError(result["error"])
            return result

    def get_observations(self) -> Dict[str, np.ndarray]:
        with self._lock:
            if self._cached_observations is None:
                request = {"method": "get_observations"}
                self._control_socket.send(pickle.dumps(request))
                result = pickle.loads(self._control_socket.recv())
                if isinstance(result, dict) and "error" in result:
                    raise RuntimeError(result["error"])
                return result
            return self._cached_observations

    def close(self) -> None:
        self._stop_event.set()
        if self._refresh_thread.is_alive():
            self._refresh_thread.join(timeout=1.0)
        self._control_socket.close()
        self._data_socket.close()
        self._context.term()
# ---------- 类定义结束 ----------


def parse_args():
    parser = argparse.ArgumentParser(description="Test ZMQ clients connection.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Server IP address")
    parser.add_argument("--robot_port", type=int, default=6001, help="Robot server port")
    parser.add_argument("--camera_base_port", type=int, default=5001, help="Camera server port")
    parser.add_argument("--camera_wrist_port", type=int, default=5000, help="Camera server port")
    parser.add_argument("--ft300_port", type=int, default=5002, help="FT300 server port")
    return parser.parse_args()


def test_camera(host, port):
    print(f"\n[Camera] Connecting to {host}:{port} ...")
    try:
        cam = ZMQClientCamera(port=port, host=host)
        times = []
        count = 0
        
        # 创建 Open3D 窗口和几何体
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="PointCloud Stream", width=800, height=600)
        pcd = o3d.geometry.PointCloud()
        # 添加一个坐标系以便观察方向（可选）
        coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
        vis.add_geometry(pcd)
        vis.add_geometry(coord)
        
        while True:
            time1 = time.time()
            pc = cam.read()   # 正确的解包方式
            time2 = time.time()
            times.append(time2 - time1)
            print(f"Frame {count}: {time2-time1:.4f}s")
            
            # 可视化点云
            if pc is not None:
                if isinstance(pc, dict):
                    points = pc.get('points')
                    colors = pc.get('colors')  # 可选
                elif isinstance(pc, np.ndarray):
                    points = pc
                    colors = None
                else:
                    points = None
                
                if points is not None and len(points) > 0:
                    pcd.points = o3d.utility.Vector3dVector(points)
                    if colors is not None:
                        pcd.colors = o3d.utility.Vector3dVector(colors)
                    # 更新几何体
                    vis.update_geometry(pcd)
                    vis.poll_events()
                    vis.update_renderer()
            count += 1
        
        print(f"平均时间: {np.mean(times):.4f}s")
        # 保持窗口打开，按 Q 或关闭按钮退出
        vis.run()
        vis.destroy_window()
        return True
    except Exception as e:
        print(f"[Camera] ERROR: {e}")
        return False

def test_ft300(host, port):
    print(f"\n[FT300] Connecting to {host}:{port} ...")
    try:
        ft = ZMQClientFT300(port=port, host=host)
        times=[]
        count=0
        while count<100:
            time1 = time.time()
            ft_data = ft.get_force_torque()
            time2 = time.time()
            times.append(time2 - time1)
            print(f"Frame {count}: {time2-time1:.4f}s")
            count += 1
            print(f"[FT300] Force/Torque: {ft_data}")
            acc = ft.get_acceleration()
            print(f"[FT300] Acceleration: {acc}")
        print(f"平均时间: {np.mean(times):.4f}s")
        return True
    except Exception as e:
        print(f"[FT300] ERROR: {e}")
        return False


def test_robot(host, port):
    print(f"\n[Robot] Connecting to {host}:{port} ...")
    try:
        robot = ZMQClientRobot(port=port, host=host)
        times1=[]
        times2=[]
        times3=[]
        count=0
        for i in range(100):
            time1 = time.time()
            dof = robot.num_dofs()
            time2 = time.time()
            times1.append(time2 - time1)
            print(f"Frame {count}: {time2-time1:.4f}s")
            # 获取关节状态
            time3 = time.time()
            joint_state = robot.get_joint_state()
            time4 = time.time()
            times2.append(time4 - time3)
            print(f"Frame {count}: {time4-time3:.4f}s")
            # 获取完整观测
            time5 = time.time()
            obs = robot.get_observations()
            time6 = time.time()
            times3.append(time6 - time5)
            print(f"Frame {count}: {time6-time5:.4f}s")
            count += 1
            robot.close()
        return True
    except Exception as e:
        print(f"[Robot] ERROR: {e}")
        return False


def main():
    args = parse_args()
    print(f"Testing connection to server at {args.host}")

    success = True
    # 测试机器人
    #success &= test_robot(args.host, args.robot_port)
    # 测试相机
    #success &= test_camera(args.host, args.camera_base_port)
    #success &= test_camera(args.host, args.camera_wrist_port)
    # 测试 FT300
    success &= test_ft300(args.host, args.ft300_port)

    if success:
        print("\n✅ All clients connected and responded successfully.")
    else:
        print("\n❌ Some clients failed. Please check server status and network.")
        exit(1)


if __name__ == "__main__":
    main()