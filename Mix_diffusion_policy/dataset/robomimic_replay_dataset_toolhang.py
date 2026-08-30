from typing import Dict, List
import torch
import numpy as np
import h5py
from tqdm import tqdm
import json
import copy
import Mix_diffusion_policy.common.transformation as tf
from Mix_diffusion_policy.common.robot import get_subgoals_stage_robomimic, get_subgoals_realtime_robomimic,get_subgoals_assembly_contact,get_subgoals_assembly_contact_new,get_subgoals_assembly_contact_keyframe
from Mix_diffusion_policy.common.visual import visual_subgoals_v6, visual_pcd, getFingersPos, getGripperPos
from Mix_diffusion_policy.common.pytorch_util import dict_apply
from Mix_diffusion_policy.dataset.base_dataset import BasePcdDataset
from Mix_diffusion_policy.model.common.normalizer import Normalizer
from Mix_diffusion_policy.model.common.rotation_transformer import RotationTransformer
from Mix_diffusion_policy.common.replay_buffer import ReplayBuffer
from Mix_diffusion_policy.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from Mix_diffusion_policy.common.normalize_util import (
    robomimic_abs_action_only_normalizer_from_stat,
    robomimic_abs_action_only_dual_arm_normalizer_from_stat,
    get_identity_normalizer_from_stat,
    array_to_stats
)
from Mix_diffusion_policy.common.visual import segmentation_to_rgb, getFingersPos, show_rgb_seg_dep, show_pcd_loop_new,show_pcd_tf,show_poses_from_obs
import robomimic.utils.file_utils as FileUtils
import open3d as o3d
import random


class RobomimicReplayDataset(BasePcdDataset):
    def __init__(self,
            dataset_path: str,
            observation_history_num=2,
            use_subgoal=True,
            horizon=1,
            pad_before=0,
            pad_after=0,
            max_demo=0,
            obs_keys: List[str]=[
                'object', 
                'robot0_joint_pos',
                'robot0_eef_pos', 
                'robot0_eef_quat', 
                'robot0_gripper_qpos'],
            max_train_episodes=None,
            abs_action=False,
            rotation_rep='rotation_6d',
            seed=42,
            Tr=1,
            val_ratio=0.02
        ):
        obs_keys = list(obs_keys)
        rotation_transformer = RotationTransformer(
            from_rep='axis_angle', to_rep=rotation_rep)

        replay_buffer = ReplayBuffer.create_empty_numpy()
        with h5py.File(dataset_path) as file:
            demos = file['data']

            if abs_action and 'absactions' in demos['demo_0']:
                action_key = 'absactions'
            else:
                action_key = 'actions'

            # 读取三个局部点云（所有 demo 共享，从第一个 demo 读取）
            tool_pc_local = demos['demo_0']['tool_pc'][:].astype(np.float32)
            frame_pc_local = demos['demo_0']['frame_pc'][:].astype(np.float32)
            stand_pc_local = demos['demo_0']['stand_pc'][:].astype(np.float32)
            scene_pcd = demos['demo_0']['scene_pcd'][:].astype(np.float32)

            # 遍历轨迹
            for i in tqdm(range(len(demos)), desc="Loading hdf5 to ReplayBuffer"):
                demo = demos[f'demo_{i}']
                data = _data_to_obs(
                    raw_obs=demo['obs'],
                    tool_pc_local=tool_pc_local,
                    frame_pc_local=frame_pc_local,
                    stand_pc_local=stand_pc_local,
                    raw_actions=demo[action_key][:].astype(np.float32),
                    obs_keys=obs_keys,
                    abs_action=abs_action,
                    rotation_transformer=rotation_transformer,
                    Tr=Tr
                )

                if use_subgoal:
                    # 子目标函数使用合并后的局部点云
                    subgoals = get_subgoals_assembly_contact_keyframe(
                        demo['obs'],
                        tool_pc_local,
                        frame_pc_local,
                        fin_rad=0.008,
                        sim_thresh=[0.02, 10./180*np.pi],
                        max_reward=10,
                        reward_mode='only_success',
                        Tr=Tr)
                    data.update(subgoals)

                # 存储场景点云和合并后的局部点云到 replay_buffer（全局共享）
                replay_buffer.scene_pcd = scene_pcd
                replay_buffer.tool_pcd = tool_pc_local
                replay_buffer.frame_pcd = frame_pc_local
                replay_buffer.stand_pcd = stand_pc_local
                replay_buffer.add_episode(data)
                if i >= max_demo:
                    break

        # 后续 train/val mask, sampler 等保持不变
        val_mask = get_val_mask(
            n_episodes=replay_buffer.n_episodes,
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask

        if max_train_episodes is not None:
            print(f'Use {max_train_episodes} demos to train!')
        else:
            print('Use all demos to train!')
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
        self.use_subgoal = use_subgoal
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
            'scene_pcd': self.replay_buffer.scene_pcd,
            'tool_pcd': self.replay_buffer.tool_pcd,
            'frame_pcd': self.replay_buffer.frame_pcd,
            'stand_pcd': self.replay_buffer.stand_pcd,
            'pcd': sample['data']['pcd'][:self.observation_history_num],
            'state': sample['data']['state'][:self.observation_history_num],
            'action': sample['data']['action'],
            'next_pcd': sample['data']['next_pcd'][:self.observation_history_num],
            'next_state': sample['data']['next_state'][:self.observation_history_num],
            'next_action': sample['data']['next_action'],
        }
        if self.use_subgoal:
            subgoal_data = {
                'subgoal': sample['data']['subgoal'][self.observation_history_num-1],
                'next_subgoal': sample['data']['next_subgoal'][self.observation_history_num-1],
                'reward': sample['data']['reward'][self.observation_history_num-1:self.observation_history_num],
            }
            data.update(subgoal_data)
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


def farthest_point_sample_batch(points_batch, npoint):
    """
    Batch farthest point sampling.
    points_batch: (B, N, 3)
    npoint: int
    returns: (B, npoint, 3)
    """
    B, N, _ = points_batch.shape
    sampled = np.zeros((B, npoint, 3), dtype=points_batch.dtype)
    for b in range(B):
        sampled[b] = tf.farthest_point_sample(points_batch[b], npoint=npoint)
    return sampled


def _data_to_obs(raw_obs, 
                 tool_pc_local,   # (1024, 3)
                 frame_pc_local,  # (1024, 3)
                 stand_pc_local,  # (1024, 3)
                 raw_actions, 
                 obs_keys, 
                 abs_action, 
                 rotation_transformer, 
                 Tr):
    """
    Args:
        raw_obs: dict with keys: 'object' (44,), 'robot0_eef_pos', 'robot0_eef_quat', 
                 'robot0_gripper_qpos', 'robot0_joint_pos'
        tool_pc_local, frame_pc_local, stand_pc_local: local point clouds (each 1024,3)
        raw_actions: (N, A)
        obs_keys: list of keys to concatenate for state
        abs_action: bool
        rotation_transformer: RotationTransformer
        Tr: step offset for next data

    Returns:
        dict with keys: pcd, state, action, next_pcd, next_state, next_action
    """
    # 计算手指位置
    fs_pos = []
    for step in range(raw_obs['object'].shape[0]):
        fl_pos, fr_pos = getFingersPos(
            raw_obs['robot0_eef_pos'][step], 
            raw_obs['robot0_eef_quat'][step], 
            raw_obs['robot0_gripper_qpos'][step, 0] + 0.0145/2,
            raw_obs['robot0_gripper_qpos'][step, 1] - 0.0145/2,
        )
        fs_pos.append(np.concatenate((fl_pos, fr_pos), axis=0))
    
    # 拼接状态 (N, S)
    # 然后下面 concat 时，不再使用 raw_obs[obs_keys[0]]，而是使用 obs0
    # 修改 concat 部分：
    obs = np.concatenate(
        [raw_obs[key] for key in obs_keys[:-1]] + [np.array(fs_pos)],
        axis=-1
    ).astype(np.float32)

    # 绝对动作处理
    if abs_action:
        is_dual_arm = False
        if raw_actions.shape[-1] == 14:
            raw_actions = raw_actions.reshape(-1, 2, 7)
            is_dual_arm = True
        pos = raw_actions[..., :3]
        rot = raw_actions[..., 3:6]
        gripper = raw_actions[..., 6:]
        rot = rotation_transformer.forward(rot)
        raw_actions = np.concatenate([pos, rot, gripper], axis=-1).astype(np.float32)
        if is_dual_arm:
            raw_actions = raw_actions.reshape(-1, 20)

    # 从 obs 中提取三个物体的位姿（前44维为 object 观测）
    #show_poses_from_obs(obj_obs, scene_pcd=None, frame_scale=0.2, colors=None, point_size=0.5)
    stand_pos = obs[:, 0:3]
    stand_quat = obs[:, 3:7]
    frame_pos = obs[:, 14:17]
    frame_quat = obs[:, 17:21]
    tool_pos = obs[:, 28:31]
    tool_quat = obs[:, 31:35]

    N = obs.shape[0]
    # 扩展局部点云到 batch 维度
    tool_local_batch = np.expand_dims(tool_pc_local, axis=0).repeat(N, axis=0)   # (N, 1024, 3)
    frame_local_batch = np.expand_dims(frame_pc_local, axis=0).repeat(N, axis=0)
    stand_local_batch = np.expand_dims(stand_pc_local, axis=0).repeat(N, axis=0)

    # 正向变换：局部 -> 世界
    tool_world = tf.transPts_tq_npbatch(tool_local_batch, tool_pos, tool_quat)     # (N, 1024, 3)
    frame_world = tf.transPts_tq_npbatch(frame_local_batch, frame_pos, frame_quat)
    stand_world = tf.transPts_tq_npbatch(stand_local_batch, stand_pos, stand_quat)

    # 合并三个点云 (N, 3072, 3)
    obj_pcd_world = np.concatenate([tool_world, frame_world, stand_world], axis=1)
    #print(f'obj_pcd_world shape: {obj_pcd_world.shape}')
    # 采样到 1024 点
    #obj_pcd_world = farthest_point_sample_batch(obj_pcd_world, npoint=1024)
    fs_pos_np = np.array(fs_pos)  # 形状 (n_frames, 6)
    point1_list = [p[:3].reshape(1,3) for p in fs_pos_np]
    point2_list = [p[3:].reshape(1,3) for p in fs_pos_np]
    #show_pcd_loop_new(obj_pcd_world,point1_list,point2_list)
    data = {
        'pcd': obj_pcd_world[:-Tr],
        'state': obs[:-Tr],
        'action': raw_actions[:-Tr],
        'next_pcd': obj_pcd_world[Tr:],
        'next_state': obs[Tr:],
        'next_action': raw_actions[Tr:],
    }
    return data


if __name__ == '__main__': 
    dataset_path = 'data/robomimic/datasets/tool_hang/ph/low_dim_abs_pcd.hdf5'
    dataset = RobomimicReplayDataset(
        dataset_path=dataset_path,
        observation_history_num=2,
        use_subgoal=True,
        horizon=16,
        pad_before=1,
        pad_after=7,
        max_demo=1,
        obs_keys=[
            'object', 
            'robot0_joint_pos',
            'robot0_eef_pos', 
            'robot0_eef_quat', 
            'robot0_gripper_qpos'],
        abs_action=True,
        rotation_rep='rotation_6d',
        seed=42,
        Tr=1,
        val_ratio=0.02
    )
