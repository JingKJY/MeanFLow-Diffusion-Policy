from typing import List, Dict, Optional
import numpy as np
import gym
from gym.spaces import Box
from robomimic.envs.env_robosuite import EnvRobosuite
from Mix_diffusion_policy.common.visual import getFingersPos
class RobomimicLowdimWrapper(gym.Env):
    def __init__(self, 
        env: EnvRobosuite,
        obs_keys: List[str]=[
            'object', 
            'robot0_eef_pos', 
            'robot0_eef_quat', 
            'robot0_gripper_qpos'],
        init_state: Optional[np.ndarray]=None,
        render_hw=(256,256),
        render_camera_name='agentview'
        ):

        self.env = env
        self.obs_keys = obs_keys
        self.init_state = init_state
        self.render_hw = render_hw
        self.render_camera_name = render_camera_name
        self.seed_state_map = dict()
        self._seed = None
        
        # setup spaces
        low = np.full(env.action_dimension, fill_value=-1)
        high = np.full(env.action_dimension, fill_value=1)
        self.action_space = Box(
            low=low,
            high=high,
            shape=low.shape,
            dtype=low.dtype
        )
        obs_example = self.get_observation()
        low = np.full_like(obs_example, fill_value=-1)
        high = np.full_like(obs_example, fill_value=1)
        self.observation_space = Box(
            low=low,
            high=high,
            shape=low.shape,
            dtype=low.dtype
        )

    def get_observation(self):
        raw_obs = self.env.get_observation()
        # obs = np.concatenate([
        #     raw_obs[key] for key in self.obs_keys
        # ], axis=0)
        fs_pos = []   # 将存储每个时间步的手指位置
        for step in range(raw_obs['object'].shape[0]):   # 遍历时间步
            fl_pos, fr_pos = getFingersPos(
                raw_obs['robot0_eef_pos'][step], 
                raw_obs['robot0_eef_quat'][step], 
                raw_obs['robot0_gripper_qpos'][step, 0] + 0.0145/2,
                raw_obs['robot0_gripper_qpos'][step, 1] - 0.0145/2,
            )
            fs_pos.append(np.concatenate((fl_pos, fr_pos), axis=0))

        obs = np.concatenate(
            [raw_obs[key] for key in self.obs_keys[:-1]] + [np.array(fs_pos)],
            axis=-1
        ).astype(np.float32)
        return obs

    def seed(self, seed=None):
        np.random.seed(seed=seed)
        self._seed = seed
    
    def reset(self):
        if self.init_state is not None:
            # always reset to the same state
            # to be compatible with gym
            self.env.reset_to({'states': self.init_state})
        elif self._seed is not None:
            # reset to a specific seed
            seed = self._seed
            if seed in self.seed_state_map:
                # env.reset is expensive, use cache
                self.env.reset_to({'states': self.seed_state_map[seed]})
            else:
                # robosuite's initializes all use numpy global random state
                np.random.seed(seed=seed)
                self.env.reset()
                state = self.env.get_state()['states']
                self.seed_state_map[seed] = state
            self._seed = None
        else:
            # random reset
            self.env.reset()

        # return obs
        obs = self.get_observation()
        return obs
    
    def step(self, action):
        raw_obs, reward, done, info = self.env.step(action)
        # obs = np.concatenate([
        #     raw_obs[key] for key in self.obs_keys
        # ], axis=0)

        obs = np.concatenate([
            raw_obs[key][:7] for key in self.obs_keys
        ], axis=0)
        return obs, reward, done, info
    
    def render(self, mode='rgb_array'):
        h, w = self.render_hw
        return self.env.render(mode=mode, 
            height=h, width=w, 
            camera_name=self.render_camera_name)


def test():
    import robomimic.utils.file_utils as FileUtils
    import robomimic.utils.env_utils as EnvUtils
    import robomimic.utils.obs_utils as ObsUtils
    from matplotlib import pyplot as plt

    dataset_path = 'data/robomimic/datasets/tool_hang/ph/low_dim_abs_pcd.hdf5'
    env_meta = FileUtils.get_env_metadata_from_dataset(
        dataset_path)
    if 'env_kwargs' in env_meta:
        env_meta['env_kwargs'].pop('lite_physics', None)
        
        # 转换控制器配置：从旧版 body_parts 嵌套格式转为新版扁平格式
        if 'controller_configs' in env_meta['env_kwargs']:
            cfg = env_meta['env_kwargs']['controller_configs']
            # 检查是否为旧版格式（存在 body_parts 字段）
            if 'body_parts' in cfg:
                # 获取第一个机械臂的配置（通常是 'right'）
                arm_name = list(cfg['body_parts'].keys())[0]
                arm_cfg = cfg['body_parts'][arm_name]
                # 如果 arm_cfg 中还有 type 字段，则使用该 type，否则从顶层继承 type
                if 'type' in arm_cfg:
                    controller_type = arm_cfg['type']
                else:
                    controller_type = cfg.get('type', 'OSC_POSE')
                # 构建新版扁平配置
                new_cfg = {
                    'type': controller_type,
                    **arm_cfg,   # 合并手臂特定参数
                }
                # 移除可能遗留的不兼容字段
                new_cfg.pop('body_parts', None)
                # 确保必需字段存在
                if 'interpolation' not in new_cfg:
                    new_cfg['interpolation'] = None
                if 'ramp_ratio' not in new_cfg:
                    new_cfg['ramp_ratio'] = 0.2
                # 替换原配置
                env_meta['env_kwargs']['controller_configs'] = new_cfg
                print("[INFO] Converted controller config from old format to new flat format.")
    obs_keys=[
            'object', 
            'robot0_eef_pos', 
            'robot0_eef_quat', 
            'robot0_gripper_qpos']

    ObsUtils.initialize_obs_modality_mapping_from_dict(
        {'low_dim': obs_keys})
    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        render=False, 
        render_offscreen=True,
        use_image_obs=False, 
    )
    wrapper = RobomimicLowdimWrapper(
        env=env,
        obs_keys=obs_keys
    )

    states = list()
    for _ in range(2):
        wrapper.seed(0)
        wrapper.reset()
        states.append(wrapper.env.get_state()['states'])
    assert np.allclose(states[0], states[1])

    img = wrapper.render()
    print('img =', img)
    plt.imshow(img)
    plt.show()
    # wrapper.seed()
    # states.append(wrapper.env.get_state()['states'])


if __name__ == '__main__':
    test()