import os
import wandb
import numpy as np
import torch
import collections
import pathlib
import tqdm
import h5py
import dill
import math
import wandb.sdk.data_types.video as wv
from Mix_diffusion_policy.gym_util.async_vector_env import AsyncVectorEnv
# from Mix_diffusion_policy.gym_util.sync_vector_env import SyncVectorEnv
from Mix_diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper
from Mix_diffusion_policy.gym_util.video_recording_wrapper import VideoRecordingWrapper, VideoRecorder
from Mix_diffusion_policy.model.common.rotation_transformer import RotationTransformer

from Mix_diffusion_policy.policy.base_pcd_policy import BasePcdPolicy
from Mix_diffusion_policy.common.pytorch_util import dict_apply
from Mix_diffusion_policy.env_runner.base_pcd_runner import BasePcdRunner
# from Mix_diffusion_policy.env.robomimic.robomimic_lowdim_wrapper import RobomimicLowdimWrapper
from Mix_diffusion_policy.env.robomimic.robomimic_pcd_wrapper import RobomimicPcdWrapper
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.obs_utils as ObsUtils
from Mix_diffusion_policy.common.replay_buffer import ReplayBuffer
import Mix_diffusion_policy.common.transformation as tf
from Mix_diffusion_policy.common.visual import visual_subgoals_tilt_v44_1, visual_subgoals_tilt_v44_2, visual_pcd,show_pcd_loop
import cv2

def create_env(env_meta, obs_keys, enable_render=True):
    ObsUtils.initialize_obs_modality_mapping_from_dict(
        {'low_dim': obs_keys})
    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        render=False, 
        # only way to not show collision geometry
        # is to enable render_offscreen
        # which uses a lot of RAM.
        render_offscreen=enable_render, # 原始为False
        use_image_obs=False, 
    )
    return env


class RobomimicRunner(BasePcdRunner):
    def __init__(self, 
            output_dir,
            dataset_path,
            replay_buffer: ReplayBuffer,
            obs_keys,
            n_train=10,
            n_train_vis=3,
            train_start_idx=0,
            n_test=22,
            n_test_vis=6,
            test_start_seed=10000,
            max_steps=400,
            use_subgoal=True,
            use_pcd=True,
            observation_history_num=2,
            n_action_steps=8,
            n_latency_steps=0,
            # 渲染参数
            render_hw=(256,256),
            render_camera_name='agentview',
            fps=10,
            crf=22,
            past_action=False,
            abs_action=False,   # true
            tqdm_interval_sec=5.0,
            n_envs=None,
            test_run=False
        ):
        """
        Assuming:
        observation_history_num=2
        n_latency_steps=3
        n_action_steps=4
        o: obs
        i: inference
        a: action
        Batch t:
        |o|o| | | | | | |
        | |i|i|i| | | | |
        | | | | |a|a|a|a|
        Batch t+1
        | | | | |o|o| | | | | | |
        | | | | | |i|i|i| | | | |
        | | | | | | | | |a|a|a|a|
        """

        super().__init__(output_dir)

        if n_envs is None:
            n_envs = n_train + n_test

        # handle latency step
        # to mimic latency, we request n_latency_steps additional steps 
        # of past observations, and the discard the last n_latency_steps
        env_n_obs_steps = observation_history_num + n_latency_steps
        env_n_action_steps = n_action_steps

        # assert n_obs_steps <= n_action_steps
        dataset_path = os.path.expanduser(dataset_path)
        robosuite_fps = 20
        steps_per_render = max(robosuite_fps // fps, 1)

        # read from dataset
        env_meta = FileUtils.get_env_metadata_from_dataset(
            dataset_path)

        rotation_transformer = None
        if abs_action:
            try:
                env_meta['env_kwargs']['controller_configs']['control_delta'] = False
            except:
                env_meta['controller_configs']['control_delta'] = False
            rotation_transformer = RotationTransformer('axis_angle', 'rotation_6d')

        def env_fn():
            robomimic_env = create_env(
                    env_meta=env_meta, 
                    obs_keys=obs_keys
                )
            # hard reset doesn't influence lowdim env
            # robomimic_env.env.hard_reset = False
            return MultiStepWrapper(
                    VideoRecordingWrapper(
                        RobomimicPcdWrapper(
                            env=robomimic_env,
                            obs_keys=obs_keys,
                            init_state=None,
                            render_hw=render_hw,
                            render_camera_name=render_camera_name
                        ),
                        video_recoder=VideoRecorder.create_h264(
                            fps=fps,
                            codec='h264',
                            input_pix_fmt='rgb24',
                            crf=crf,
                            thread_type='FRAME',
                            thread_count=1
                        ),
                        file_path=None,
                        steps_per_render=steps_per_render
                    ),
                    n_obs_steps=env_n_obs_steps,
                    n_action_steps=env_n_action_steps,
                    max_episode_steps=max_steps
                )

        # For each process the OpenGL context can only be initialized once
        # Since AsyncVectorEnv uses fork to create worker process,
        # a separate env_fn that does not create OpenGL context (enable_render=False)
        # is needed to initialize spaces.
        def dummy_env_fn():
            robomimic_env = create_env(
                    env_meta=env_meta, 
                    obs_keys=obs_keys,
                    enable_render=False
                )
            return MultiStepWrapper(
                    VideoRecordingWrapper(
                        RobomimicPcdWrapper(
                            env=robomimic_env,
                            obs_keys=obs_keys,
                            init_state=None,
                            render_hw=render_hw,
                            render_camera_name=render_camera_name
                        ),
                        video_recoder=VideoRecorder.create_h264(
                            fps=fps,
                            codec='h264',
                            input_pix_fmt='rgb24',
                            crf=crf,
                            thread_type='FRAME',
                            thread_count=1
                        ),
                        file_path=None,
                        steps_per_render=steps_per_render
                    ),
                    n_obs_steps=env_n_obs_steps,
                    n_action_steps=env_n_action_steps,
                    max_episode_steps=max_steps
                )

        env_fns = [env_fn] * n_envs
        env_seeds = list()
        env_prefixs = list()
        env_init_fn_dills = list()

        # train
        with h5py.File(dataset_path, 'r') as f:
            for i in range(n_train):
                train_idx = train_start_idx + i
                enable_render = i < n_train_vis
                # init_state = f[f'data/demo_{train_idx}/states'][0]

                # def init_fn(env, init_state=init_state, 
                def init_fn(env, 
                    enable_render=enable_render):
                    # setup rendering
                    # video_wrapper
                    assert isinstance(env.env, VideoRecordingWrapper)
                    env.env.video_recoder.stop()
                    env.env.file_path = None
                    if enable_render:
                        filename = pathlib.Path(output_dir).joinpath(
                            'media', wv.util.generate_id() + ".mp4")
                        filename.parent.mkdir(parents=False, exist_ok=True)
                        filename = str(filename)
                        env.env.file_path = filename

                    # switch to init_state reset
                    assert isinstance(env.env.env, RobomimicPcdWrapper)
                    # env.env.env.init_state = init_state

                env_seeds.append(train_idx)
                env_prefixs.append('train/')
                env_init_fn_dills.append(dill.dumps(init_fn))
        
        # test
        for i in range(n_test):
            seed = test_start_seed + i
            enable_render = i < n_test_vis

            def init_fn(env, seed=seed, 
                enable_render=enable_render):
                # setup rendering
                # video_wrapper
                assert isinstance(env.env, VideoRecordingWrapper)
                env.env.video_recoder.stop()
                env.env.file_path = None
                if enable_render:
                    filename = pathlib.Path(output_dir).joinpath(
                        'media', wv.util.generate_id() + ".mp4")
                    filename.parent.mkdir(parents=False, exist_ok=True)
                    filename = str(filename)
                    env.env.file_path = filename

                # switch to seed reset
                assert isinstance(env.env.env, RobomimicPcdWrapper)
                env.env.env.init_state = None
                env.seed(seed)

            env_seeds.append(seed)
            env_prefixs.append('test/')
            env_init_fn_dills.append(dill.dumps(init_fn))
        
        env = AsyncVectorEnv(env_fns, dummy_env_fn=dummy_env_fn)
        # env = SyncVectorEnv(env_fns)

        self.env_meta = env_meta
        self.env = env
        self.env_fns = env_fns
        self.env_seeds = env_seeds
        self.env_prefixs = env_prefixs
        self.env_init_fn_dills = env_init_fn_dills
        self.fps = fps
        self.crf = crf
        self.use_subgoal = use_subgoal
        self.use_pcd = use_pcd
        self.observation_history_num = observation_history_num
        self.n_action_steps = n_action_steps
        self.n_latency_steps = n_latency_steps
        self.env_n_obs_steps = env_n_obs_steps
        self.env_n_action_steps = env_n_action_steps
        self.past_action = past_action
        self.max_steps = max_steps
        self.rotation_transformer = rotation_transformer
        self.abs_action = abs_action
        self.tqdm_interval_sec = tqdm_interval_sec
        self.replay_buffer = replay_buffer
        self.test_run = test_run


    def run(self, policy: BasePcdPolicy, first=False):
        device = policy.device
        dtype = policy.dtype
        env = self.env
        
        # plan for rollout
        n_envs = len(self.env_fns)
        n_inits = len(self.env_init_fn_dills)
        n_chunks = math.ceil(n_inits / n_envs)

        all_video_paths = [None] * n_inits
        all_rewards = [None] * n_inits

        # 辅助函数：批处理最远点采样
        def farthest_point_sample_batch(points_batch, npoint):
            B, N, _ = points_batch.shape
            sampled = np.zeros((B, npoint, 3), dtype=points_batch.dtype)
            for b in range(B):
                sampled[b] = tf.farthest_point_sample(points_batch[b], npoint=npoint)
            return sampled

        for chunk_idx in range(n_chunks):
            start = chunk_idx * n_envs
            end = min(n_inits, start + n_envs)
            this_global_slice = slice(start, end)
            this_n_active_envs = end - start
            this_local_slice = slice(0, this_n_active_envs)
            
            this_init_fns = self.env_init_fn_dills[this_global_slice]
            n_diff = n_envs - len(this_init_fns)
            if n_diff > 0:
                this_init_fns.extend([self.env_init_fn_dills[0]]*n_diff)
            assert len(this_init_fns) == n_envs

            env.call_each('run_dill_function', args_list=[(x,) for x in this_init_fns])

            obs = env.reset()
            policy.reset()
            B = n_envs
            if first:
                return
            step = 0
            # 从 replay_buffer 获取静态点云（局部坐标系）
            tool_pcd_local = self.replay_buffer.tool_pcd      # (1024,3)
            frame_pcd_local = self.replay_buffer.frame_pcd    # (1024,3)
            stand_pcd_local = self.replay_buffer.stand_pcd    # (1024,3)
            scene_pcd = self.replay_buffer.scene_pcd          # (1024,3) 用于可视化

            pbar = tqdm.tqdm(total=self.max_steps, desc=f"Eval {self.env_meta['env_name']}Pcd {chunk_idx+1}/{n_chunks}", 
                             leave=False, mininterval=self.tqdm_interval_sec)
            
            done = False
            while not done:
                # 原始观测字典
                np_obs_dict = {
                    'state': obs[:, :self.observation_history_num].astype(np.float32)
                }
                state = np_obs_dict['state']   # (B, H, state_dim)
                if self.use_pcd:
                    obj_pcd_history = []
                    for h in range(self.observation_history_num):
                        # 提取该历史步的物体观测（前44维）
                        obj_obs_h = state[:, h, :44]   # (B,44)
                        # 各部件位姿（偏移7）
                        stand_pos = obj_obs_h[:, 0:3]   
                        stand_quat = obj_obs_h[:, 3:7]  
                        frame_pos = obj_obs_h[:, 14:17]  
                        frame_quat = obj_obs_h[:, 17:21] 
                        tool_pos = obj_obs_h[:, 28:31] 
                        tool_quat = obj_obs_h[:, 31:35]

                        # 扩展局部点云到 batch 维度
                        tool_local_batch = np.expand_dims(tool_pcd_local, axis=0).repeat(B, axis=0)
                        frame_local_batch = np.expand_dims(frame_pcd_local, axis=0).repeat(B, axis=0)
                        stand_local_batch = np.expand_dims(stand_pcd_local, axis=0).repeat(B, axis=0)

                        # 变换到世界坐标系
                        tool_world = tf.transPts_tq_npbatch(tool_local_batch, tool_pos, tool_quat)
                        frame_world = tf.transPts_tq_npbatch(frame_local_batch, frame_pos, frame_quat)
                        stand_world = tf.transPts_tq_npbatch(stand_local_batch, stand_pos, stand_quat)

                        # 合并三个点云并采样到 1024 点
                        merged = np.concatenate([tool_world, frame_world, stand_world], axis=1)  # (B,3072,3)
                        #merged = farthest_point_sample_batch(merged, npoint=1024)     # (B,1024,3) 
                        
                        obj_pcd_history.append(merged)
                    obj_pcd = np.stack(obj_pcd_history, axis=1)
                    np_obs_dict['pcd'] = obj_pcd

                if self.use_subgoal:
                    Tinput_dict = dict_apply(np_obs_dict, lambda x: torch.from_numpy(x).to(device=device))
                    with torch.no_grad():
                        subgoal = subgoal = policy.guider_predict_target(Tinput_dict).detach().to('cpu').numpy()
                    np_obs_dict['subgoal'] = subgoal

                Tinput_dict = dict_apply(np_obs_dict, lambda x: torch.from_numpy(x).to(device=device))
                with torch.no_grad():
                    action_dict = policy.actor_predict_action(Tinput_dict)
                np_action_dict = dict_apply(action_dict, lambda x: x.detach().to('cpu').numpy())

                action = np_action_dict['action'][:, self.n_latency_steps:]
                if not np.all(np.isfinite(action)):
                    print(action)
                    raise RuntimeError("Nan or Inf action")
                
                env_action = action
                if self.abs_action:
                    env_action = self.undo_transform_action(action)

                obs, reward, done, info = env.step(env_action)
                done = np.all(done)
                pbar.update(action.shape[1])
                step += self.n_action_steps

            pbar.close()

            all_video_paths[this_global_slice] = env.render()[this_local_slice]
            all_rewards[this_global_slice] = env.call('get_attr', 'reward')[this_local_slice]

        # 日志记录部分与原代码相同 ...
        max_rewards = collections.defaultdict(list)
        log_data = dict()
        for i in range(n_inits):
            seed = self.env_seeds[i]
            prefix = self.env_prefixs[i]
            max_reward = np.max(all_rewards[i])
            max_rewards[prefix].append(max_reward)
            log_data[prefix+f'sim_max_reward_{seed}'] = max_reward
            video_path = all_video_paths[i]
            if video_path is not None:
                sim_video = wandb.Video(video_path)
                log_data[prefix+f'sim_video_{seed}'] = sim_video

        for prefix, value in max_rewards.items():
            name = prefix+'mean_score'
            value = np.mean(value)
            log_data[name] = value

        return log_data
    

    def undo_transform_action(self, action):
        # raw_shape = action.shape
        # if raw_shape[-1] == 20:
        #     # dual arm
        #     action = action.reshape(-1,2,10)

        d_rot = action.shape[-1] - 4    # 6
        pos = action[...,:3]
        rot = action[...,3:3+d_rot]
        gripper = action[...,[-1]]
        rot = self.rotation_transformer.inverse(rot)
        uaction = np.concatenate([
            pos, rot, gripper
        ], axis=-1)

        # if raw_shape[-1] == 20:
        #     # dual arm
        #     uaction = uaction.reshape(*raw_shape[:-1], 14)

        return uaction
