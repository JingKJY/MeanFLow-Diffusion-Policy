from typing import Dict, List
import torch
import numpy as np
import h5py
from tqdm import tqdm
import copy
from scipy.spatial.transform import Rotation
from Mix_diffusion_policy.common.robot import get_subgoals_from_gripper
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
            obs_keys: List[str]=[
                'state_joint',
                'state_ee',
                'gripper',
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
        replay_buffer = ReplayBuffer.create_empty_numpy()
        with h5py.File(dataset_path, 'r') as file:
            demos = file['trajectories']
            for i in tqdm(range(len(demos)), desc="Loading hdf5"):
                demo = demos[f'demo_{i}']
                data = _data_to_obs(
                    raw_obs=demo['obs'],
                    raw_actions=demo[action_key][:].astype(np.float32),
                    obs_keys=obs_keys,
                    action_key=action_key,
                    abs_action=abs_action,
                    rotation_transformer=rotation_transformer,
                    Tr=Tr)
                # ******** 新增加：子目标生成 ********
                if self.use_subgoal:
                    subgoals = get_subgoals_from_gripper(
                        demo['obs']['state_ee'],        # 直接传原始观测
                        demo['obs']['gripper'],
                        Tr=Tr,
                        # 可传入超参数，或使用默认值
                    )
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

        # tool0 -> 夹爪基座 沿 -Z 偏移
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


def _data_to_obs(raw_obs, raw_actions, obs_keys, action_key, abs_action, rotation_transformer, Tr):
    # 1. 计算 两个独立指尖 (N,3)
    left_finger, right_finger = calc_two_fingers_all(
        obs_state_ee=raw_obs['state_ee'][:],
        obs_gripper=raw_obs['gripper'][:]
    )
    state_ee=raw_obs['state_ee'][:]
    # 2. 拼接状态: 原有观测 + 左指尖 + 右指尖
    state_parts = []
    for k in obs_keys:
        state_parts.append(raw_obs[k][:].astype(np.float32))
    # 加入两个指尖
    state_parts.append(left_finger)
    state_parts.append(right_finger)
    obs = np.concatenate(state_parts, axis=-1)

    # 3. 处理 action_ee: pos3 + quat4 + gri1 -> 转6d旋转
    if action_key == 'actions_ee':
        if abs_action:
            pos = raw_actions[..., :3]
            quat = raw_actions[..., 3:7]
            gripper = raw_actions[..., 7:]
            rot_6d = rotation_transformer.forward(quat)
            raw_actions = np.concatenate([pos, rot_6d, gripper], axis=-1).astype(np.float32)
    elif action_key == 'actions_joint':
        raw_actions = raw_actions.astype(np.float32)
    else:
        raise ValueError("Invalid action_key")


    # 4. 拼接 base + wrist 双点云
    base_pcd = raw_obs['base_pointcloud'][:].astype(np.float32)
    wrist_pcd = raw_obs['wrist_pointcloud'][:].astype(np.float32)
    pcd = np.concatenate([base_pcd, wrist_pcd], axis=1)
    # 变换参数
    from scipy.spatial.transform import Rotation as R
    rotation = R.from_quat([-0.640141814416, 0.641962273626, -0.308035560623, 0.288473551766])
    translation = np.array([-0.377129, -0.564256, 0.573252])
    cam_to_ee_rot = R.from_quat([-0.237599428984, -0.968350417038, -0.0399076896506, 0.0652024345794])
    cam_to_ee_trans = np.array([0.0237718822114, -0.0934458077043, 0.169067206708])
    def transform_pointcloud(points, rotation, translation):
        """
        points: (..., 3) 任意形状，最后一维是 xyz
        rotation: scipy Rotation 对象
        translation: (3,) ndarray
        返回: 与 points 相同形状的变换后点云
        """
        original_shape = points.shape
        flat = points.reshape(-1, 3)
        transformed = rotation.apply(flat) + translation
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
    base_pcd_world = transform_pointcloud(base_pcd, rotation, translation)
    #show_pcd_loop(wrist_pcd)
    #show_pcd_loop(base_pcd)
    n_frames = base_pcd_world.shape[0]
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
    #show_pcd_loop_new_ee(base_pcd_world, point1_list, point2_list,state_ee)    
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