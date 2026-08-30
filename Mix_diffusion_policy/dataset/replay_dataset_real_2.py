from typing import Dict, List
import torch
import numpy as np
import h5py
from tqdm import tqdm
import copy
from scipy.spatial.transform import Rotation
from Mix_diffusion_policy.common.robot import get_subgoals_from_gripper, get_ring_insertion_event_subgoals, get_block_stack_event_subgoals
from Mix_diffusion_policy.common.pytorch_util import dict_apply
from Mix_diffusion_policy.dataset.base_dataset import BasePcdDataset
from Mix_diffusion_policy.model.common.normalizer import Normalizer
from Mix_diffusion_policy.model.common.rotation_transformer import RotationTransformer
from Mix_diffusion_policy.common.replay_buffer import ReplayBuffer
from Mix_diffusion_policy.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from Mix_diffusion_policy.common.normalize_util import (
    robomimic_abs_action_only_normalizer_from_stat,
    array_to_stats
)
from Mix_diffusion_policy.common.visual import segmentation_to_rgb, getFingersPos, show_rgb_seg_dep, show_pcd_loop_new_ee,show_pcd_tf,show_poses_from_obs,show_pcd_loop

from ur_analytic_ik import ur5

class CustomReplayDataset(BasePcdDataset):
    def __init__(self,
            dataset_path: str,
            observation_history_num=2,
            use_subgoal=False,
            horizon=16,
            pad_before=1,
            pad_after=7,
            max_demo=0,
            action_key='actions_ee',
            subgoal_extractor='gripper',
            drop_return_reset=False,
            return_reset_keep_after_release=10,
            return_reset_min_episode_len=None,
            binary_gripper_action=False,
            gripper_action_open_threshold=0.7,
            obs_keys: List[str]=[
                'state_joint',
                'force_torque'],
            max_train_episodes=None,
            abs_action=True,
            rotation_rep='rotation_6d',
            seed=42,
            Tr=1,
            val_ratio=0.02
        ):
        obs_keys = list(obs_keys)
        rotation_transformer = RotationTransformer(
            from_rep='quaternion', to_rep=rotation_rep)
        self.use_subgoal = use_subgoal
        self.subgoal_extractor = subgoal_extractor
        self.drop_return_reset = drop_return_reset
        self.return_reset_keep_after_release = return_reset_keep_after_release
        self.return_reset_min_episode_len = return_reset_min_episode_len
        self.binary_gripper_action = binary_gripper_action
        self.gripper_action_open_threshold = float(gripper_action_open_threshold)
        replay_buffer = ReplayBuffer.create_empty_numpy()
        with h5py.File(dataset_path, 'r') as file:
            demos = file['trajectories']
            for i in tqdm(range(len(demos)), desc="Loading hdf5"):
                demo = demos[f'demo_{i}']
                data = _data_to_obs(
                    raw_obs=demo['obs'],
                    raw_actions=demo['actions_joint'][:].astype(np.float32),
                    obs_keys=obs_keys,
                    action_key=action_key,
                    abs_action=abs_action,
                    rotation_transformer=rotation_transformer,
                    binary_gripper_action=binary_gripper_action,
                    gripper_action_open_threshold=self.gripper_action_open_threshold,
                    Tr=Tr)
                # ******** 新增加：子目标生成 ********
                states_ee = []
                for j in range(demo['obs']['state_joint'].shape[0]):
                    matrix = ur5.forward_kinematics(*demo['obs']['state_joint'][j, :6])
                    state_ee = np.concatenate([matrix[:3, 3], Rotation.from_matrix(matrix[:3, :3]).as_quat()])
                    states_ee.append(state_ee)
                states_ee = np.stack(states_ee, axis=0)    # 得到末端位姿
                states_ee = states_ee.astype(np.float32)

                if self.drop_return_reset:
                    keep_len = find_episode_len_without_return_reset(
                        gripper=demo['obs']['state_joint'][:, 6:],
                        Tr=Tr,
                        keep_after_release=self.return_reset_keep_after_release,
                        min_episode_len=self.return_reset_min_episode_len,
                    )
                    if keep_len < data['action'].shape[0]:
                        data = slice_episode_data(data, keep_len)
                        states_ee = states_ee[:keep_len + Tr]

                if self.use_subgoal:
                    if self.subgoal_extractor == 'gripper':
                        subgoals = get_subgoals_from_gripper(
                            states_ee,        # 直接传原始观测
                            demo['obs']['state_joint'][:states_ee.shape[0], 6:],
                            Tr=Tr,
                            # 可传入超参数，或使用默认值
                        )
                    elif self.subgoal_extractor == 'ring_event':
                        subgoals = get_ring_insertion_event_subgoals(
                            states_ee,
                            demo['obs']['state_joint'][:states_ee.shape[0], 6:],
                            Tr=Tr,
                            include_return_closed=not self.drop_return_reset,
                        )
                    elif self.subgoal_extractor == 'block_stack_event':
                        subgoals = get_block_stack_event_subgoals(
                            states_ee,
                            demo['obs']['state_joint'][:states_ee.shape[0], 6:],
                            Tr=Tr,
                            include_return_closed=not self.drop_return_reset,
                        )
                    else:
                        raise ValueError(f"Unknown subgoal_extractor: {self.subgoal_extractor}")
                    data.update(subgoals)   # 合并到 data 字典
                replay_buffer.add_episode(data)
                if i >= max_demo and max_demo > 0:
                    break
                
        val_mask = get_val_mask(
            n_episodes=replay_buffer.n_episodes,
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask

        train_mask = downsample_mask(
            mask=train_mask, 
            max_n=max_train_episodes, 
            seed=seed)

        sampler = SequenceSampler(
            replay_buffer=replay_buffer, 
            abs_action=abs_action,
            sequence_length=horizon,
            pad_before=pad_before, 
            pad_after=pad_after,
            episode_mask=train_mask)
        
        self.replay_buffer = replay_buffer
        
        self.observation_history_num = observation_history_num
        self.sampler = sampler
        self.abs_action = abs_action
        self.train_mask = train_mask
        self.val_mask = val_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.Tr = Tr

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            abs_action=self.abs_action,
            sequence_length=self.horizon,
            pad_before=self.pad_before, 
            pad_after=self.pad_after,
            episode_mask=self.val_mask
            )
        val_set.train_mask = self.val_mask
        return val_set

    def get_normalizer(self) -> Normalizer:
        normalizer = Normalizer()
        action_stat = array_to_stats(self.replay_buffer['action'])
        action_params = robomimic_abs_action_only_normalizer_from_stat(action_stat)
        normalizer.params_dict['action'] = action_params
        
        state_stat = array_to_stats(self.replay_buffer['state'])
        normalizer.params_dict['state'] = normalizer_from_stat(state_stat)
        return normalizer
    
    def __len__(self):
        return len(self.sampler)

    def _sample_to_data(self, sample, i):
        data = {
            'id': np.array([i,]),
            'pcd': sample['data']['pcd'][:self.observation_history_num],
            'state': sample['data']['state'][:self.observation_history_num],
            'action': sample['data']['action'],
            'next_pcd': sample['data']['next_pcd'][:self.observation_history_num],
            'next_state': sample['data']['next_state'][:self.observation_history_num],
            'next_action': sample['data']['next_action'],
        }
        if self.use_subgoal:
            data['subgoal'] = sample['data']['subgoal'][self.observation_history_num-1]      # 取观测历史帧
            data['next_subgoal'] = sample['data']['next_subgoal'][self.observation_history_num-1]
            data['reward'] = sample['data']['reward'][self.observation_history_num-1:self.observation_history_num]
        return data

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample, idx)
        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data


def slice_episode_data(data: Dict[str, np.ndarray], keep_len: int) -> Dict[str, np.ndarray]:
    """Keep the first keep_len transitions from one processed episode."""
    keep_len = int(keep_len)
    return {key: value[:keep_len] for key, value in data.items()}


def find_episode_len_without_return_reset(
        gripper: np.ndarray,
        Tr: int = 1,
        keep_after_release: int = 10,
        min_episode_len=None,
        close_change_ratio: float = 0.25,
        open_change_ratio: float = 0.25,
        min_event_gap: int = 8,
    ) -> int:
    """
    Return transition length after dropping the final return-to-reset segment.

    The real ring task uses high gripper values for open and low values for
    closed. We keep data until the release event, plus a small stable tail,
    and remove the later reset motion where the arm returns near the initial
    pose and closes again.
    """
    g = np.asarray(gripper, dtype=np.float32).reshape(-1)
    T = g.shape[0]
    if T <= Tr + 1:
        return max(0, T - Tr)

    g_min = float(np.nanmin(g))
    g_max = float(np.nanmax(g))
    g_range = max(g_max - g_min, 1e-6)
    closed_th = g_min + 0.35 * g_range
    open_th = g_min + 0.65 * g_range
    close_change_th = close_change_ratio * g_range
    open_change_th = open_change_ratio * g_range

    dg = np.diff(g, prepend=g[0])
    frame_ids = np.arange(T)
    closed = g <= closed_th
    opened = g >= open_th

    close_candidates = np.where((dg < -close_change_th) | closed)[0]
    if len(close_candidates) == 0:
        return T - Tr
    grasp_close = int(close_candidates[0])

    release_candidates = np.where(
        ((dg > open_change_th) | opened) &
        (frame_ids >= grasp_close + min_event_gap)
    )[0]
    if len(release_candidates) == 0:
        return T - Tr

    release_start = int(release_candidates[-1])
    keep_obs_len = min(T, release_start + int(keep_after_release) + 1)
    keep_transition_len = max(0, keep_obs_len - Tr)

    if min_episode_len is not None:
        keep_transition_len = max(keep_transition_len, int(min_episode_len))
    return min(keep_transition_len, T - Tr)


def normalizer_from_stat(stat):
    max_abs = np.maximum(stat['max'].max(), np.abs(stat['min']).max())
    scale = np.full_like(stat['max'], fill_value=1/max_abs)
    offset = np.zeros_like(stat['max'])
    return Normalizer.create_manual(
        scale=scale,
        offset=offset,
        input_stats_dict=stat
    )


# UR5 + FT300 + Robotiq 2F-85 双指尖计算
def calc_two_fingers_all(obs_state_ee, obs_gripper):
    # 硬件标定参数
    FT300_HEIGHT = 0.035
    GRIPPER_BASE_H = 0.025
    TOOL2GRIPPER_Z = 0.20# 0.060m
    W_MAX_HALF = 0.0425  # 2F-85 单侧最大半宽

    N = obs_state_ee.shape[0]
    left_finger_list = []
    right_finger_list = []

    for t in range(N):
        # 位置
        pos = obs_state_ee[t, :3]
        # 四元数: 正确顺序 (qx, qy, qz, qw)
        qx, qy, qz, qw = obs_state_ee[t, 3:7]

        # 夹爪归一化开度 [0,1]
        g_q = obs_gripper[t, 0]

        # 构造 tool0 齐次矩阵
        R = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
        T_tool0 = np.eye(4)
        T_tool0[:3, :3] = R
        T_tool0[:3, 3] = pos

        # tool0 -> 夹爪基座 沿 +Z 偏移
        T_gripper = T_tool0 @ np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, TOOL2GRIPPER_Z],
            [0, 0, 0, 1]
        ])

        half_w = W_MAX_HALF * g_q
        # 指尖在夹爪坐标系中的位置（假设 Z=0，根据实际情况可能需要调整）
        left_pt  = np.array([-half_w, 0.0, 0.0, 1.0])
        right_pt = np.array([ half_w, 0.0, 0.0, 1.0])

        # 转换到世界坐标系
        left_world  = (T_gripper @ left_pt)[:3]
        right_world = (T_gripper @ right_pt)[:3]

        left_finger_list.append(left_world)
        right_finger_list.append(right_world)

    left_arr = np.array(left_finger_list, dtype=np.float32)
    right_arr = np.array(right_finger_list, dtype=np.float32)
    return left_arr, right_arr


def binarize_gripper_action(
        actions: np.ndarray,
        open_threshold: float = 0.7,
    ) -> np.ndarray:
    actions = np.asarray(actions, dtype=np.float32).copy()
    if actions.ndim != 2 or actions.shape[1] < 1:
        return actions
    actions[:, -1:] = (actions[:, -1:] >= float(open_threshold)).astype(np.float32)
    return actions


def _data_to_obs(
        raw_obs,
        raw_actions,
        obs_keys,
        action_key,
        abs_action,
        rotation_transformer,
        binary_gripper_action=False,
        gripper_action_open_threshold=0.7,
        Tr=1):
        # 1. 计算 两个独立指尖 (N,3)
    if action_key == 'actions_ee':
        actions_ee = []
        for i in range(raw_actions.shape[0]):
            matrix = ur5.forward_kinematics(*raw_actions[i, :6])
            action_ee = np.concatenate([matrix[:3, 3], Rotation.from_matrix(matrix[:3, :3]).as_quat()])
            actions_ee.append(action_ee)
        actions_ee = np.stack(actions_ee, axis=0)
        actions_ee = actions_ee.astype(np.float32)
        
        actions_joint = raw_actions[:, 6:]
        state_joint = raw_obs['state_joint']
        state_ee = []     
        for i in range(state_joint.shape[0]):
            matrix = ur5.forward_kinematics(*state_joint[i, :6])
            state_ee.append(np.concatenate([matrix[:3, 3], Rotation.from_matrix(matrix[:3, :3]).as_quat()]))
        state_ee = np.stack(state_ee, axis=0)
        state_ee = state_ee.astype(np.float32)
        #print(actions_ee.shape, actions_joint.shape, state_ee.shape, state_joint.shape)
        left_finger, right_finger = calc_two_fingers_all(
            obs_state_ee=state_ee,
            obs_gripper=raw_obs['state_joint'][:, 6:]
        )
        obs = []
        for key in obs_keys:
            if key == 'state_ee':
                obs.append(state_ee)
            elif key == 'state_joint':
                obs.append(state_joint)  
            elif key == 'state_gripper':
                obs.append(raw_obs['state_joint'][:, 6:])
            elif key == 'force_torque':
                obs.append(raw_obs['force_torque'])
        obs.append(left_finger)
        obs.append(right_finger)
        obs = np.concatenate(obs, axis=-1)
        #obs = np.concatenate([state_ee, state_joint, left_finger, right_finger], axis=-1)
        #obs = np.concatenate([state_ee, left_finger, right_finger], axis=-1)
        actions = np.concatenate([actions_ee, actions_joint], axis=-1)
    elif action_key == 'actions_joint':
        state_joint = raw_obs['state_joint']
        state_ee = []
        for i in range(state_joint.shape[0]):
            matrix = ur5.forward_kinematics(*state_joint[i, :6])
            state_ee.append(np.concatenate([matrix[:3, 3], Rotation.from_matrix(matrix[:3, :3]).as_quat()]))
        state_ee = np.stack(state_ee, axis=0)
        state_ee = state_ee.astype(np.float32)
        
        left_finger, right_finger = calc_two_fingers_all(
            obs_state_ee=state_ee,
            obs_gripper=raw_obs['state_joint'][:, 6:]
        )
        obs = []
        for key in obs_keys:
            if key == 'state_ee':
                obs.append(state_ee)
            elif key == 'state_joint':
                obs.append(state_joint)
            elif key == 'state_gripper':
                obs.append(raw_obs['state_joint'][:, 6:])
            elif key == 'force_torque':
                obs.append(raw_obs['force_torque'])
        obs.append(left_finger)
        obs.append(right_finger)
        obs = np.concatenate(obs, axis=-1)
        #os = np.concatenate([state_ee, state_joint, left_finger, right_finger], axis=-1)
        actions = np.concatenate([actions_ee, actions_joint], axis=-1)
        #bs = np.concatenate([state_ee, state_joint, left_finger, right_finger], axis=-1)

    # 3. 处理 action_ee: pos3 + quat4 + gri1 -> 转6d旋转
    if action_key == 'actions_ee':
        if abs_action:
            pos = actions[:, :3]
            quat = actions[:, 3:7]
            gripper = actions[:, 7:]
            rot_6d = rotation_transformer.forward(quat)
            raw_actions = np.concatenate([pos, rot_6d, gripper], axis=-1).astype(np.float32)
    elif action_key == 'actions_joint':
        raw_actions = raw_actions.astype(np.float32)
    else:
        raise ValueError("Invalid action_key")

    if binary_gripper_action:
        raw_actions = binarize_gripper_action(
            raw_actions,
            open_threshold=gripper_action_open_threshold,
        )

    # 4. 拼接 left + right 双点云
    left_pcd = raw_obs['left_pointcloud'][:].astype(np.float32)
    right_pcd = raw_obs['right_pointcloud'][:].astype(np.float32)

    # 变换参数
    from scipy.spatial.transform import Rotation as R
    '''
    rotation_left = R.from_quat([-0.877778754555, 0.0997887390624, -0.212844131958, 0.417425491674])
    translation_left = np.array([-0.295412506243, -0.821509264833, 0.487239743002])
    
    rotation_right = R.from_quat([-0.899403981288, -0.10832539495, 0.223448896709, 0.359734176597])
    translation_right = np.array([0.308320971175, -0.843742843137, 0.499726139978])'''

    rotation_left = R.from_quat([-0.91950325629, 0.0866294472802, -0.217637317384, 0.315662950975])
    translation_left = np.array([-0.263618073856, -0.806245531009, 0.458467688855])
    
    rotation_right = R.from_quat([-0.900016690794, -0.0954383787414, 0.227065195379, 0.35958708153])
    translation_right = np.array([0.302589964549, -0.822491517206, 0.474515465596])
    
    def transform_pointcloud(points, rotation, translation):
        """
        points: (..., 3+) 任意形状，最后一维前三个通道是 xyz，后续通道如 rgb 原样保留
        rotation: scipy Rotation 对象
        translation: (3,) ndarray
        返回: 与 points 相同形状的变换后点云
        """
        original_shape = points.shape
        feature_dim = original_shape[-1]
        flat = points.reshape(-1, feature_dim)
        xyz_world = rotation.apply(flat[:, :3]) + translation
        if feature_dim > 3:
            transformed = np.concatenate([xyz_world, flat[:, 3:]], axis=-1)
        else:
            transformed = xyz_world
        return transformed.reshape(original_shape)
    def transform_pointcloud_cam_to_world(points_cam, cam_to_ee_rot, cam_to_ee_trans, ee_pose):
        """
        points_cam: (N, 3)  wrist相机坐标系下的点云
        cam_to_ee_rot: scipy Rotation 对象，表示相机→末端的旋转
        cam_to_ee_trans: (3,) ndarray，相机→末端的平移
        ee_pose: (7,) ndarray，末端在世界坐标系下的位姿 [x,y,z, qx,qy,qz,qw]
        返回: (N, 3) 世界坐标系下的点云
        """
        # 1. 相机坐标系 -> 末端坐标系
        points_ee = cam_to_ee_rot.apply(points_cam) + cam_to_ee_trans
        
        # 2. 末端坐标系 -> 世界坐标系
        ee_pos = ee_pose[:3]
        ee_quat = ee_pose[3:7]   # [qx,qy,qz,qw]
        ee_rot = R.from_quat(ee_quat)
        points_world = ee_rot.apply(points_ee) + ee_pos
        return points_world
    left_pcd_world = transform_pointcloud(left_pcd, rotation_left, translation_left)
    right_pcd_world = transform_pointcloud(right_pcd, rotation_right, translation_right)
    pcd = np.concatenate([left_pcd_world, right_pcd_world], axis=1)
    #show_pcd_loop(wrist_pcd)
    #show_pcd_loop(base_pcd)
    n_frames = pcd.shape[0]
    '''
    wrist_pcd_world = []
    for i in range(n_frames):
        ee_pose = raw_obs['state_ee'][i]   # (7,)
        wrist_cam_frame = wrist_pcd[i]      # (N_w, 3)
        wrist_world_frame = transform_pointcloud_cam_to_world(
            wrist_cam_frame, cam_to_ee_rot, cam_to_ee_trans, ee_pose
        )
        wrist_pcd_world.append(wrist_world_frame)
    wrist_pcd_world = np.stack(wrist_pcd_world, axis=0)
    pcd_world=np.concatenate([base_pcd_world, wrist_pcd_world], axis=1)'''
    point1_list = [left_finger[i].reshape(1, 3) for i in range(n_frames)]       # 每个元素 (1,3)
    point2_list = [right_finger[i].reshape(1, 3) for i in range(n_frames)]      # 每个元素 (1,3)
    #show_pcd_loop_new_ee(pcd, point1_list, point2_list,state_ee)    
    # 5. 时序截断
    data = {
        'pcd': pcd[:-Tr],
        'state': obs[:-Tr],
        'action': raw_actions[:-Tr],
        'next_pcd': pcd[Tr:],
        'next_state': obs[Tr:],
        'next_action': raw_actions[Tr:],
    }
    return data


if __name__ == '__main__': 
    dataset_path = "data/real/cylinder_short_insert_only/gello_dataset.h5"
    dataset = CustomReplayDataset(
        dataset_path=dataset_path,
        observation_history_num=2,
        use_subgoal=False,
        horizon=16,
        pad_before=1,
        pad_after=7,
        max_demo=0,
        obs_keys=['state_joint','state_ee','gripper','force_torque'],
        abs_action=True,
        rotation_rep='rotation_6d',
        seed=42,
        Tr=1,
        val_ratio=0.02
    )
