import pickle
import threading
from typing import Optional, Tuple, Dict

import numpy as np
import zmq


DEFAULT_CAMERA_PORT = 5000
DEFAULT_ROBOT_PORT = 6000

class ZMQClientCamera():
    def __init__(self, port: int = DEFAULT_CAMERA_PORT, host: str = "127.0.0.1"):
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.REQ)
        self._socket.connect(f"tcp://{host}:{port}")

    def read(
        self,
        img_size: Optional[Tuple[int, int]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[dict]]:
        send_message = pickle.dumps(img_size)
        self._socket.send(send_message)
        pointcloud = pickle.loads(self._socket.recv())
        return pointcloud

class ZMQClientFT300:
    def __init__(self, port=5001, host="127.0.0.1"):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(f"tcp://{host}:{port}")
        self.socket.setsockopt(zmq.RCVTIMEO, 5000)  # 5 second timeout

    def _send_command(self, cmd, args=None):
        msg = pickle.dumps((cmd, args))
        self.socket.send(msg)
        resp = pickle.loads(self.socket.recv())
        if isinstance(resp, Exception):
            raise resp
        return resp

    def get_force_torque(self):
        return self._send_command("get_force_torque")

    def get_acceleration(self):
        return self._send_command("get_acceleration")

class ZMQClientRobot():
    """A class representing a ZMQ client for a leader robot."""

    def __init__(self, port: int = DEFAULT_ROBOT_PORT, host: str = "127.0.0.1"):
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.REQ)
        self._socket.connect(f"tcp://{host}:{port}")

    def num_dofs(self) -> int:
        """Get the number of joints in the robot.

        Returns:
            int: The number of joints in the robot.
        """
        request = {"method": "num_dofs"}
        send_message = pickle.dumps(request)
        self._socket.send(send_message)
        result = pickle.loads(self._socket.recv())
        return result

    def get_joint_state(self) -> np.ndarray:
        """Get the current state of the leader robot.

        Returns:
            T: The current state of the leader robot.
        """
        request = {"method": "get_joint_state"}
        send_message = pickle.dumps(request)
        try:
            self._socket.send(send_message)
            result = pickle.loads(self._socket.recv())
            if isinstance(result, dict) and "error" in result:
                raise RuntimeError(result["error"])
            return result
        except zmq.Again:
            raise RuntimeError("ZMQ timeout - robot may be disconnected")

    def command_joint_state(self, joint_state: np.ndarray) -> None:
        """Command the leader robot to the given state.

        Args:
            joint_state (T): The state to command the leader robot to.
        """
        request = {
            "method": "command_joint_state",
            "args": {"joint_state": joint_state},
        }
        send_message = pickle.dumps(request)
        self._socket.send(send_message)
        result = pickle.loads(self._socket.recv())
        return result

    def get_observations(self) -> Dict[str, np.ndarray]:
        """Get the current observations of the leader robot.

        Returns:
            Dict[str, np.ndarray]: The current observations of the leader robot.
        """
        request = {"method": "get_observations"}
        send_message = pickle.dumps(request)
        try:
            self._socket.send(send_message)
            result = pickle.loads(self._socket.recv())
            if isinstance(result, dict) and "error" in result:
                raise RuntimeError(result["error"])
            return result
        except zmq.Again:
            raise RuntimeError("ZMQ timeout - robot may be disconnected")

    def close(self) -> None:
        """Close the ZMQ socket and context."""
        self._socket.close()
        self._context.term()