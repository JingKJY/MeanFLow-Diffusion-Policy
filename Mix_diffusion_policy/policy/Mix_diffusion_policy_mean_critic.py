from typing import Dict
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from termcolor import cprint
import copy
import time
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from functools import partial

from Mix_diffusion_policy.model.flow.sde_lib import ConsistencyFM
from Mix_diffusion_policy.model.common.normalizer import Normalizer
from Mix_diffusion_policy.policy.base_pcd_policy import BasePcdPolicy
from Mix_diffusion_policy.model.diffusion.guider_transform import Guider
from Mix_diffusion_policy.model.diffusion.guider_mlp import Guider_mlp
#from Mix_diffusion_policy.model.flow.actor_mean import Actor
from Mix_diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
from Mix_diffusion_policy.common.pytorch_util import dict_apply 
from Mix_diffusion_policy.model.encoder.condition_encoder import PCDStateSubgoalEncoder
from Mix_diffusion_policy.model.encoder.pointnet_extractor import FlowPolicyEncoder
from Mix_diffusion_policy.model.diffusion.pointcloud_encoder import PointNetEncoder
import Mix_diffusion_policy.common.transformation as tf
from Mix_diffusion_policy.common.visual import Color, draw_pcl
from Mix_diffusion_policy.model.flow.u_dit_mean import RMDiT
#from Mix_diffusion_policy.model.UMamba.conditional_umamba1d_mean import ConditionalMambaUnet1D
from Mix_diffusion_policy.model.UMamba.conditional_umamba1d_new import ConditionalMambaUnet1D
#from Mix_diffusion_policy.model.flow.actor_mean import Actor
#from Mix_diffusion_policy.model.flow.actor_mean_ssl import Actor
#from Mix_diffusion_policy.model.flow.actor_mean_ssl_t_a import Actor
from Mix_diffusion_policy.model.flow.dispersive_loss import DispersiveLoss
from Mix_diffusion_policy.model.diffusion.critic import Critic2net
from Mix_diffusion_policy.model.diffusion.pointcloud_encoder import PointNetEncoder
class MixPolicy(BasePcdPolicy):
    def __init__(self, 
            guider_noise_scheduler:DDPMScheduler,
            guider_horizon, 
            actor_horizon,
            guider_n_action_steps, 
            actor_n_action_steps,
            guider_n_obs_steps,
            actor_n_obs_steps,
            guider_mode,
            actor_mode,
            mean_mode,
            guider_model_cfg,
            actor_model_cfg,
            guider_pcd_encoder,
            actor_pcd_encoder,
            pcd_dim,
            state_dim,
            subgoal_dim,
            film_dim,
            action_dim,
            subgoal_dim_nocont,
            guider_hidden_dim,
            actor_hidden_dim,
            guider_cond_dim,
            actor_cond_dim,
            ssl_weight,
            MeanFlow,
            use_subgoal,
            dispersive_loss_cfg,
            apply_dispersive_to_embeddings,
            apply_dispersive_to_latents,
            # critic 参数
            critic_n_action_steps: int = 8,          # 评估多少个连续动作 (Tr)
            discount: float = 0.95,                  # TD 折扣因子
            eta: float = 0.001,                        # Actor 中 Q 损失的权重（默认0，不启用）
            fin_rad: float = 0.008, 
            single_step_flow: bool = True,
            **kwargs):
        super().__init__()
            
        guider_obs_encoder = PCDStateSubgoalEncoder(
            pcd_dim, 
            state_dim, 
            subgoal_dim, 
            obs_dim=guider_n_obs_steps,
            is_use_subgoal=False, 
            is_use_film=False,
            hidden_dim=guider_hidden_dim, 
            film_dim=film_dim,
            output_dim=guider_cond_dim,
            pcd_encoder=guider_pcd_encoder,
            type="total"
        )
        if guider_mode == "transformer":
            guider_model = Guider(
                input_dim=subgoal_dim,
                output_dim=subgoal_dim,
                horizon=guider_horizon,
                n_obs_steps=guider_model_cfg['n_obs_steps'],
                cond_dim=guider_cond_dim,
                n_layer=guider_model_cfg['n_layer'],
                n_head=guider_model_cfg['n_head'],
                n_emb=guider_model_cfg['n_emb'],
                p_drop_emb=guider_model_cfg['p_drop_emb'],
                p_drop_attn=guider_model_cfg['p_drop_attn'],
                causal_attn=guider_model_cfg['causal_attn'],
                time_as_cond=guider_model_cfg['time_as_cond'],
                obs_as_cond=guider_model_cfg['obs_as_cond'],
                n_cond_layers=guider_model_cfg['n_cond_layers']
            )
        elif guider_mode == "mlp":
            guider_model = Guider_mlp(
                subgoal_dim=subgoal_dim,
                cond_dim=guider_cond_dim,
                n_emb=guider_model_cfg['n_emb'],
                mlp_dims=[1024, 512, 256]
            )        
        self.guider_mode=guider_mode
        self.actor_mode=actor_mode
        self.mean_mode=mean_mode
        actor_obs_encoder = PCDStateSubgoalEncoder(
            pcd_dim, 
            state_dim, 
            subgoal_dim, 
            obs_dim=actor_n_obs_steps,
            is_use_subgoal=use_subgoal, 
            is_use_film=False,
            hidden_dim=actor_hidden_dim, 
            film_dim=film_dim,
            output_dim=actor_cond_dim,
            pcd_encoder=actor_pcd_encoder,
            type="crossattn"
        )
        self.critic_pcd_encoder = PointNetEncoder(
            input_dim=pcd_dim*actor_n_obs_steps,
            mlp_dims=[64, 128, 256])
        '''  
        actor_obs_encoder = FlowPolicyEncoder(
            state_dim=state_dim,
            pcd_dim=pcd_dim,
            subgoal_dim=subgoal_dim,
            out_channel=64,
            state_mlp_size=(64, 64), state_mlp_activation_fn=nn.ReLU,
            pointcloud_encoder_cfg=actor_pcd_encoder,#64
            use_pc_color=False,
            is_use_subgoal=True,
            pointnet_type='mlp',
        )#output_dim=state_mlp_size[-1]+ pointcloud_encoder_cfg['output_dim']+subgoal_dim)
        '''
        if actor_mode == "UDit":
            actor_model= RMDiT(
                input_dim=action_dim,
                output_dim=action_dim,
                horizon=actor_horizon,
                n_obs_steps=actor_n_obs_steps,
                cond_dim=actor_cond_dim,
                depth=actor_model_cfg['n_layer'],
                n_head=actor_model_cfg['n_head'],
                side_layer=actor_model_cfg['side_layer'],
                mask_ratio=actor_model_cfg['mask_ratio'],
                n_emb=actor_model_cfg['n_emb'],
                decode_layer=actor_model_cfg['decode_layer'],
                attn_drop=actor_model_cfg['attn_drop'],
            )
        elif actor_mode == "Umamba_new":
            actor_model = ConditionalMambaUnet1D(
                input_dim=action_dim,
                global_cond_dim=actor_cond_dim,
                diffusion_step_embed_dim=actor_model_cfg['diffusion_step_embed_dim'],
                hide_dim=actor_model_cfg['hide_dim'],
                kernel_size=actor_model_cfg['kernel_size'],
                n_groups=actor_model_cfg['n_groups'],
                condition_type=actor_model_cfg['condition_type'],
                mamba_version= actor_model_cfg['mamba_version'],
                decode_layer=actor_model_cfg['decode_layer'],
                side_layer=actor_model_cfg['side_layer'],
                mask_ratio=actor_model_cfg['mask_ratio'],
                depth=actor_model_cfg['n_layer'],
                horizon=actor_n_obs_steps,
            )        
        elif actor_mode == "Umamba1d":           
            actor_model = ConditionalMambaUnet1D(
                input_dim=action_dim,
                global_cond_dim=actor_cond_dim,
                diffusion_step_embed_dim=actor_model_cfg['diffusion_step_embed_dim'],
                down_dims=actor_model_cfg['down_dims'],
                kernel_size=actor_model_cfg['kernel_size'],
                n_groups=actor_model_cfg['n_groups'],
                condition_type=actor_model_cfg['condition_type'],
                use_down_condition=True,
                use_mid_condition=True,
                use_up_condition=True,
                mamba_version= actor_model_cfg['mamba_version'],
                horizon=actor_n_obs_steps,
            )  
        elif actor_mode == "UNet":
            if self.mean_mode == "MF_ssl_ta":
                from Mix_diffusion_policy.model.flow.actor_mean_ssl_t_a import Actor
                #from Mix_diffusion_policy.model.flow.actor_mean_ssl import Actor
            else:
                from Mix_diffusion_policy.model.flow.actor_mean_ssl import Actor
            actor_model = Actor(
                input_dim=action_dim,
                local_cond_dim=None,
                global_cond_dim=actor_cond_dim,
                diffusion_step_embed_dim=actor_model_cfg['diffusion_step_embed_dim'],
                down_dims=actor_model_cfg['down_dims'],
                kernel_size=actor_model_cfg['kernel_size'],
                n_groups=actor_model_cfg['n_groups'],
                condition_type=actor_model_cfg['condition_type'],
                mask_ratio=actor_model_cfg['mask_ratio'],
                cat_ratio=actor_model_cfg['cat_ratio'],
                use_down_condition=True,
                use_mid_condition=True,
                use_up_condition=True,
                horizon=actor_n_obs_steps,)
            
        self.critic_model = Critic2net(
            pcd_encoder=self.critic_pcd_encoder,
            state_dim=state_dim*actor_n_obs_steps,
            subgoal_dim=subgoal_dim,
            action_dim=action_dim*critic_n_action_steps,
            mlp_dims=[512, 256, 128],
        )
        self.critic_target = copy.deepcopy(self.critic_model)
        self.critic_target.eval()

        self.guider_obs_encoder = guider_obs_encoder
        self.actor_obs_encoder = actor_obs_encoder
        self.guider_model = guider_model
        self.actor_model = actor_model
        self.guider_noise_scheduler = guider_noise_scheduler
        
        
        self.guider_noise_scheduler_pc = copy.deepcopy(guider_noise_scheduler)


        self.guider_mask_generator = LowdimMaskGenerator(
            action_dim=subgoal_dim,
            obs_dim=guider_cond_dim ,
            max_n_obs_steps=guider_n_obs_steps,
            fix_obs_steps=True,
            action_visible=False
        )
        self.actor_mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0,
            max_n_obs_steps=actor_n_obs_steps,  
            fix_obs_steps=True,
            action_visible=False
        )
        # create normalizer
        self.normalizer = Normalizer()

        self.guider_horizon = guider_horizon
        self.actor_horizon = actor_horizon
        self.guider_obs_feature_dim = guider_cond_dim
        self.actor_obs_feature_dim = actor_cond_dim
        self.guider_action_dim = subgoal_dim
        self.actor_action_dim = action_dim

        self.guider_n_action_steps = guider_n_action_steps
        self.actor_n_action_steps = actor_n_action_steps
        self.guider_n_obs_steps = guider_n_obs_steps
        self.actor_n_obs_steps = actor_n_obs_steps
        self.subgoal_dim_nocont=subgoal_dim_nocont
        self.subgoal_dim=subgoal_dim
        self.kwargs = kwargs
        self.state_dim = state_dim
        self.ssl_weight = ssl_weight
        self.use_subgoal = use_subgoal
        self.pcd_dim = pcd_dim
        #MeanFLowConfig
        self.time_dist = MeanFlow['time_dist']
        self.flow_ratio = MeanFlow['flow_ratio']
        self.sample_type = MeanFlow['sample_type']
        self.jvp_api = MeanFlow['jvp_api']
        self.num_inference_step = MeanFlow['num_inference_step']
        self.cfg_ratio = MeanFlow['cfg_ratio']
        self.w = MeanFlow['cfg_scale']
        self.cfg_uncond = MeanFlow['cfg_uncond']
        self.cond_ratio = MeanFlow['cond_ratio']
        assert self.jvp_api in ['funtorch', 'autograd'], "jvp_api must be 'funtorch' or 'autograd'"
        if self.jvp_api == 'funtorch':
            self.jvp_fn = torch.func.jvp
            self.create_graph = False
        elif self.jvp_api == 'autograd':
            self.jvp_fn = torch.autograd.functional.jvp
            self.create_graph = True 
        print(f"time_dist:{self.time_dist}, flow_ratio:{self.flow_ratio}, jvp_api:{self.jvp_api}, num_inference_step:{self.num_inference_step}, cfg_ratio:{self.cfg_ratio}, w:{self.w}")
        
        self.dispersive_loss = DispersiveLoss(
            loss_type=dispersive_loss_cfg['dispersive_loss_type'],
            temperature=dispersive_loss_cfg['dispersive_temperature'],
            margin=dispersive_loss_cfg['dispersive_margin'],
            weight=dispersive_loss_cfg['dispersive_loss_weight'],
        )
        print(dispersive_loss_cfg)
        self.apply_dispersive_to_embeddings = apply_dispersive_to_embeddings
        self.apply_dispersive_to_latents = apply_dispersive_to_latents
        self.subgoal_token = nn.Parameter(torch.randn(1, self.subgoal_dim))

        self.eta = eta
        self.discount = discount
        self.fin_rad = fin_rad
        self.critic_n_action_steps = critic_n_action_steps
        self.single_step_flow = single_step_flow

    def predict_next_Q(self, batch):
        nbatch = self.normalizer.normalize(batch, self.subgoal_dim_nocont)
        B = nbatch['next_state'].shape[0]

        # 当前状态
        next_pcd = None
        next_pcd = nbatch['next_pcd'].transpose(1, 2).reshape(
                (B, -1, self.pcd_dim*self.actor_n_obs_steps))  # (B, 1024, 3n)
        next_state = nbatch['next_state'].reshape((B, -1))
        next_subgoal = nbatch['next_subgoal'] # (B, 8)
        next_action = nbatch['next_action'][:, self.actor_n_obs_steps-1:
                                            self.actor_n_obs_steps-1+self.critic_n_action_steps] # (B, A)
        next_action = next_action.reshape((B, -1))
        with torch.no_grad():
            current_q1, current_q2 = self.critic_target(
                next_pcd, next_state, next_subgoal, next_action)
        return torch.min(current_q1, current_q2)
    
        
    def guider_predict_target(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "key_state" key
        """
        nbatch = self.normalizer.normalize(obs_dict, self.subgoal_dim_nocont)
        B = nbatch['state'].shape[0]

        # 初始化子目标
        target_pred = torch.randn(size=(B,self.guider_horizon, self.subgoal_dim),
                         dtype=self.dtype, device=self.device)  # (B, n*8+n)
        nbatch['state']=nbatch['state'][:,:,-self.state_dim:]
        global_cond = self.guider_obs_encoder(nbatch)
        
        
        scheduler = self.guider_noise_scheduler     
        for t in scheduler.timesteps:
            # 1. apply conditioning
            #target_pred[cond_mask] = cond_data[cond_mask]
            #Transform
            if self.guider_mode == "transformer":
                model_output = self.guider_model(sample=target_pred,
                                    timestep=t, 
                                    cond=global_cond)
            # MLP
            elif self.guider_mode == "mlp":
                model_output = self.guider_model(subgoal=target_pred,
                                    timestep=t, 
                                    cond=global_cond)
            
            target_pred = scheduler.step(
                model_output, t, target_pred, ).prev_sample          
            # 3. compute previous image: x_t -> x_t-1
        #target_pred[cond_mask] = cond_data[cond_mask]
        # unnormalize prediction
        target_pred = target_pred.reshape(B, -1)
        target_pred[:, :6] = self.normalizer.unnormalize(nposition=target_pred[:, :6])
        target_pred[:, 6:] = torch.round(target_pred[:, 6:])
        target_pred[:, :3] *= target_pred[:, 6:7]
        target_pred[:, 3:6] *= target_pred[:, 7:]
        
        
        return target_pred
    # ========= Actor inference  ============
    def actor_predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict: 必须包含"obs"键
        result: 必须包含"action"键
        """
        # 归一化输入观测数据

        # 准备条件数据
        nobs = self.normalizer.normalize(obs_dict, self.subgoal_dim_nocont)
        B , T, _ = nobs['state'].shape
        Da = self.actor_action_dim
        To = self.actor_n_obs_steps
        nobs['state']=nobs['state'][:,:,-self.state_dim:]
        # 准备条件数据
        obs=self.actor_obs_encoder(nobs).reshape(B, T, -1)
        shape = (B, self.actor_horizon, self.actor_action_dim)
        # 运行采样过程
        noise = torch.randn(size=shape, dtype=self.dtype, device=self.device)
        z = noise.detach().clone() # 初始噪声

        t_vals = torch.linspace(1.0, 0.0, self.num_inference_step + 1, device=self.device)
        # 循环采样步骤
        for i in range(self.num_inference_step):
            t = torch.ones(z.shape[0], device=noise.device) * t_vals[i]
            r = torch.ones(z.shape[0], device=noise.device) * t_vals[i+1]
            if self.sample_type == 't_r':
                pred_v = self.actor_model(z, t, r, local_cond=None, global_cond=obs)
            elif self.sample_type == 't_c':
                pred_v = self.actor_model(z, t, t-r, local_cond=None, global_cond=obs)
            
            t_expanded = t.view(-1, 1, 1)  # [8, 1, 1]
            r_expanded = r.view(-1, 1, 1)  # [8, 1, 1]
            z = z - (t_expanded - r_expanded) * pred_v


        # 反归一化预测的动作
        naction_pred = z[...,:Da]
        action_pred = self.normalizer.unnormalize(naction_pred)
        # 提取预测的动作
        start = To - 1
        end = start + self.actor_n_action_steps
        action = action_pred[:,start:end]
        result = {
            'action': action,
            'action_pred': action_pred,
        }
        return result
    # ========= Guider_training  ============
    def set_normalizer(self, normalizer: Normalizer):#设置归一化器
        self.normalizer.load_state_dict(normalizer.state_dict())

    def guider_compute_loss(self, batch):
        # normalize input
        #print(f"batch:{batch.keys()}")
        nbatch = self.normalizer.normalize(batch, self.subgoal_dim_nocont)
        target = nbatch['subgoal']
        target = target.unsqueeze(1).repeat(1, self.guider_horizon, 1)
        B = target.shape[0]

        #condition_mask = torch.zeros_like(target, dtype=torch.bool)
        timesteps = torch.randint(
            0, self.guider_noise_scheduler.config.num_train_timesteps, # 100
            (B,), device=self.device
        ).long()    
        noise = torch.randn(size=(B,self.guider_horizon, self.subgoal_dim), device=self.device)  # 采样噪声
        #cur_state=nbatch('tcp_gr_pose')[:,0,:]
        #noisy_sg = self.guider_noise_scheduler.add_noise(cur_state, noise, timesteps)
        noisy_sg = self.guider_noise_scheduler.add_noise(target, noise, timesteps)
        # 预测和损失计算
        #loss_mask = ~condition_mask
        # 应用条件数据
        #noisy_sg[condition_mask] = target[condition_mask]
        obs=self.guider_obs_encoder(nbatch)
        if self.guider_mode == "transformer":
            pred = self.guider_model(noisy_sg, timesteps, obs)
        elif self.guider_mode == "mlp":
            pred = self.guider_model(obs, noisy_sg, timesteps)#mlp
        assert self.guider_noise_scheduler.config.prediction_type == 'epsilon'
        loss = F.mse_loss(pred, noise)
        #loss = F.l1_loss(pred, noise)  #MAEloss
        #loss = loss * loss_mask.type(loss.dtype)
        #loss = reduce(loss, 'b ... -> b (...)', 'mean')
        #loss = loss.mean()
        return loss

        # print(f"t2-t1: {t2-t1:.3f}")
        # print(f"t3-t2: {t3-t2:.3f}")
        # print(f"t4-t3: {t4-t3:.3f}")
        # print(f"t5-t4: {t5-t4:.3f}")
        # print(f"t6-t5: {t6-t5:.3f}")
    # ========= actor_training  ============
    def critic_compute_loss(self, batch):
        # normalize input
        nbatch = self.normalizer.normalize(batch, self.subgoal_dim_nocont)
        B = nbatch['state'].shape[0]

        # 当前状态
        pcd = None
        
        pcd = nbatch['pcd'].transpose(1, 2).reshape(
                (B, -1, self.pcd_dim*self.actor_n_obs_steps))  # (B, 1024, 3n)
        state = nbatch['state'].reshape((B, -1))
        action = nbatch['action'][:, self.actor_n_obs_steps-1:
                                      self.actor_n_obs_steps-1+self.critic_n_action_steps]    # (B, A)
        # action = action.reshape((B, -1))
        subgoal = nbatch['subgoal'] # (B, 8)
        reward = nbatch['reward']   # (B, 1)
        dones = torch.zeros((B, 1), device=self.device)
        dones[reward==10] = 1

        # 添加平移噪声，旋转噪声设为0
        # 噪声逻辑：先生成原始尺度噪声，再乘scale
        # 小噪声：最终action不加噪声, done不变
        # 大噪声: r=0, done=1
        if np.random.uniform() > 0.5:
            if np.random.uniform() > 0.5:
                # 小噪声
                noise = torch.randn(action.shape, device=self.device)*0.1
                scale = self.normalizer.params_dict['action']['scale']
                nscale = scale.expand_as(action)
                noise = torch.clip(noise, -self.fin_rad/2*nscale, self.fin_rad/2*nscale)
                action[:, :-1] += noise[:, :-1]
            else:
                # 大噪声
                noise = torch.randn(action.shape, device=self.device)

                action += noise
                reward = torch.zeros((B, 1), device=self.device)
                dones = torch.ones((B, 1), device=self.device)

        action = action.reshape((B, -1))
        current_q1, current_q2 = self.critic_model(
            pcd, state, subgoal, action)

        target_q = self.predict_next_Q(batch)

        target_q = (reward + (1-dones) * self.discount * target_q).detach()
        critic_loss = F.mse_loss(current_q1, target_q) + \
                      F.mse_loss(current_q2, target_q)
        return critic_loss


    def run_ema_critic(self):
        # ** critic ema **
        tau = 0.005
        for param, target_param in zip(self.critic_model.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def sample_t_r(self, batch_size):
        if self.time_dist[0] == 'uniform':
            samples = np.random.rand(batch_size, 2).astype(np.float32)

        elif self.time_dist[0] == 'lognorm':
            mu, sigma = self.time_dist[-2], self.time_dist[-1]
            normal_samples = np.random.randn(batch_size, 2).astype(np.float32) * sigma + mu
            samples = 1 / (1 + np.exp(-normal_samples))  # Apply sigmoid

        elif self.time_dist[0] == 'marginal':
            alpha = self.time_dist[1] if len(self.time_dist) > 1 else 0.8
            samples = np.random.beta(alpha, alpha, size=(batch_size, 2)).astype(np.float32)
        # Assign t = max, r = min, for each pair

        '''
        num_selected = int(self.flow_ratio * batch_size) 
        indices = np.random.permutation(batch_size)[:num_selected]
        samples[indices, 0] = samples[indices, 1]
        '''
        t_np = np.maximum(samples[:, 0], samples[:, 1])
        r_np = np.minimum(samples[:, 0], samples[:, 1])

        
        num_selected = int(self.flow_ratio * batch_size)    
        indices = np.random.permutation(batch_size)[:num_selected]
        r_np[indices] = t_np[indices]
        '''
        indices1 = indices[:num_selected//2]
        indices2 = indices[num_selected//2:]
        r_np[indices1] = t_np[indices1]
        t_np[indices2] = r_np[indices2]'''

        
        t = torch.tensor(t_np, device=self.device)
        r = torch.tensor(r_np, device=self.device)
        return t, r   
    def sample_t_c(self, batch_size):

        t_samples = np.random.rand(batch_size, 1).astype(np.float32)
        uniform_samples = np.random.rand(batch_size, 1).astype(np.float32)
        c_samples = t_samples * uniform_samples

        
        num_selected = int(self.cfg_ratio * batch_size)
        indices = np.random.permutation(batch_size)[:num_selected]
        c_samples[indices] = 0

        t = torch.tensor(t_samples, device=self.device).squeeze(1)
        c = torch.tensor(c_samples, device=self.device).squeeze(1)
        
        return t, c

    def actor_compute_loss(self, batch):
        nbatch = self.normalizer.normalize(batch, self.subgoal_dim_nocont)
        # 根据观测条件类型处理观测数据
        local_cond = None
        global_cond = None
        # 准备目标数据
        target = nbatch['action']
        B, T, _ = nbatch['state'].shape
        nbatch['state']=nbatch['state'][:,:,-self.state_dim:]
        global_cond = self.actor_obs_encoder(nbatch)
        global_cond = global_cond.reshape(B, T, -1)
        if self.mean_mode=="MF_subgoal_guider":
            mask = torch.bernoulli(torch.full((B, 1), 1 - self.cond_ratio)).to(global_cond.device)
            nbatch['subgoal']=nbatch['subgoal']*mask
            global_cond = self.actor_obs_encoder(nbatch)
            global_cond = global_cond.reshape(B, T, -1)
        cond_data = target.clone()
        # 生成遮罩
        condition_mask = self.actor_mask_generator(target.shape)
        loss_mask = ~condition_mask
        # 初始化目标和噪声
        a0 = torch.randn(target.shape, device=target.device)
        t,r = self.sample_t_r(B)

        t_=t.view(-1, 1, 1).repeat(1, target.shape[1], target.shape[2])
        r_=r.view(-1, 1, 1).repeat(1, target.shape[1], target.shape[2]) 

        xt = (1-t_)*target+t_*a0
        xt[condition_mask] = cond_data[condition_mask]
        loss=0
        if self.mean_mode == "MF":
            v = a0-target
            if self.apply_dispersive_to_embeddings:
                t_embed, r_embed ,c_embed= self.actor_model.get_embedding(xt, t, r, global_cond)
                c_loss=self.dispersive_loss(c_embed)
                t_loss = self.dispersive_loss(t_embed)
                r_loss = self.dispersive_loss(r_embed)
                dispersiveLoss=c_loss+t_loss+r_loss
            # 无条件引导                    
            
            v_hat=v
            #v_hat_maa=v
            model_partial = partial(self.actor_model, global_cond=global_cond)
            if self.sample_type == 't_r':
                jvp_args = (
                    lambda xt, t, r: model_partial(xt, t, r),  
                    (xt, t, r),
                    (v_hat, torch.ones_like(t), torch.zeros_like(r)),
                )
            elif self.sample_type == 't_c':
                jvp_args = (
                    lambda xt, t, r: model_partial(xt, t, t-r),  
                    (xt, t, r),
                    (v_hat, torch.ones_like(t), torch.zeros_like(r)),
                )
            if self.create_graph:
                u, dudt = self.jvp_fn(*jvp_args, create_graph=True)
            else:
                u, dudt = self.jvp_fn(*jvp_args)
            if self.sample_type == 't_r':     
                u_tgt = v_hat - (t_ - r_) * dudt
            elif self.sample_type == 't_c':
                u_tgt = v_hat - (t_ - r_) * dudt 
            error = (u - stopgrad(u_tgt))#*loss_mask
            #error=torch.nan_to_num(error)
            loss = adaptive_l2_loss(error)#+adaptive_l2_loss(error_1)
            #loss = F.mse_loss(u, stopgrad(u_tgt)) 
           # loss = loss + self.ssl_weight*loss_maa
            # loss = F.mse_loss(u, stopgrad(u_tgt))
            if self.apply_dispersive_to_embeddings:
                loss = loss + dispersiveLoss
            dispersiveLoss=0
            if self.apply_dispersive_to_latents:
                for down_latent in self.actor_model.down_latents_tensor:
                    dispersiveLoss = dispersiveLoss + self.dispersive_loss(down_latent)  
                loss = loss + dispersiveLoss
            #mse_val = torch.mean(error ** 2)  # 所有元素的平均
        if self.mean_mode == "MF_subgoal_guider":
            v = a0-target
            if self.apply_dispersive_to_embeddings:
                t_embed, r_embed ,c_embed= self.actor_model.get_embedding(xt, t, r, global_cond)
                c_loss=self.dispersive_loss(c_embed)
                t_loss = self.dispersive_loss(t_embed)
                r_loss = self.dispersive_loss(r_embed)
                dispersiveLoss=c_loss+t_loss+r_loss
            # 无条件引导                    
            v_hat=v         
            model_partial = partial(self.actor_model, global_cond=global_cond)
            if self.sample_type == 't_r':
                jvp_args = (
                    lambda xt, t, r: model_partial(xt, t, r),  
                    (xt, t, r),
                    (v_hat, torch.ones_like(t), torch.zeros_like(r)),
                )
            elif self.sample_type == 't_c':
                jvp_args = (
                    lambda xt, t, r: model_partial(xt, t, t-r),  
                    (xt, t, r),
                    (v_hat, torch.ones_like(t), torch.zeros_like(r)),
                )
            if self.create_graph:
                u, dudt = self.jvp_fn(*jvp_args, create_graph=True)
            else:
                u, dudt = self.jvp_fn(*jvp_args)
            if self.sample_type == 't_r':     
                u_tgt = v_hat - (t_ - r_) * dudt
            elif self.sample_type == 't_c':
                u_tgt = v_hat - (t_ - r_) * dudt 
            error = (u - stopgrad(u_tgt))#*loss_mask
            #error=torch.nan_to_num(error)
            loss = adaptive_l2_loss(error)#+adaptive_l2_loss(error_1)
            #loss = F.mse_loss(u, stopgrad(u_tgt)) 
           # loss = loss + self.ssl_weight*loss_maa
            # loss = F.mse_loss(u, stopgrad(u_tgt))
            if self.apply_dispersive_to_embeddings:
                loss = loss + dispersiveLoss
            dispersiveLoss=0
            if self.apply_dispersive_to_latents:
                for down_latent in self.actor_model.down_latents_tensor:
                    dispersiveLoss = dispersiveLoss + self.dispersive_loss(down_latent)  
                loss = loss + dispersiveLoss
            #mse_val = torch.mean(error ** 2)  # 所有元素的平均
        elif self.mean_mode == "MF_ssl_t":
            v = a0-target
            if self.apply_dispersive_to_embeddings:
                t_embed, r_embed ,c_embed= self.actor_model.get_embedding(xt, t, r, global_cond)
                c_loss=self.dispersive_loss(c_embed)
                t_loss = self.dispersive_loss(t_embed)
                r_loss = self.dispersive_loss(r_embed)
                dispersiveLoss=c_loss+t_loss+r_loss
            # 无条件引导
            if self.apply_dispersive_to_latents:
                for down_latent in self.actor_model.down_latents_tensor:
                    dispersiveLoss = dispersiveLoss + self.dispersive_loss(down_latent)
            if self.w is not None:
                with torch.no_grad():
                    u_t=self.actor_model(xt, t, t, global_cond=global_cond, local_cond=local_cond)
                    u_t_maa=self.actor_model(xt, t, t, global_cond=global_cond, local_cond=local_cond,enable_mask=True,mask_type='t')
                v_hat=(1-self.w)*u_t+v*self.w
                v_hat_maa=(1-self.w)*u_t_maa+v*self.w
            else:
                v_hat=v
                v_hat_maa=v                         
            
            #v_hat=v
            #v_hat_maa=v
            model_partial = partial(self.actor_model, global_cond=global_cond)
            model_partial_maa = partial(self.actor_model, global_cond=global_cond, enable_mask=True,mask_type='t')
            if self.sample_type == 't_r':
                jvp_args = (
                    lambda xt, t, r: model_partial(xt, t, r),  
                    (xt, t, r),
                    (v_hat, torch.ones_like(t), torch.zeros_like(r)),
                )
                
                jvp_args_maa = (
                    lambda xt, t, r: model_partial_maa(xt, t, r),  
                    (xt, t, r),
                    (v_hat_maa, torch.ones_like(t), torch.zeros_like(r)),
                )
            elif self.sample_type == 't_c':
                jvp_args = (
                    lambda xt, t, r: model_partial(xt, t, t-r),  
                    (xt, t, r),
                    (v_hat, torch.ones_like(t), torch.zeros_like(r)),
                )
                
                jvp_args_maa = (
                    lambda xt, t, r: model_partial_maa(xt, t, t-r),  
                    (xt, t, r),
                    (v_hat_maa, torch.ones_like(t), torch.zeros_like(r)),
                )
            if self.create_graph:
                u, dudt = self.jvp_fn(*jvp_args, create_graph=True)
                u_maa, dudt_maa = self.jvp_fn(*jvp_args_maa, create_graph=True)
            else:
                u, dudt = self.jvp_fn(*jvp_args)
                u_maa, dudt_maa = self.jvp_fn(*jvp_args_maa)   
            if self.sample_type == 't_r':     
                u_tgt = v_hat - (t_ - r_) * dudt
                u_tgt_maa = v_hat - (t_ - r_) * dudt_maa
            elif self.sample_type == 't_c':
                u_tgt = v_hat - (t_ - r_) * dudt 
                u_tgt_maa = v_hat - (t_ - r_) * dudt_maa
        
            error = (u - stopgrad(u_tgt))#*loss_mask
            error_maa = (u_maa - stopgrad(u_tgt_maa))*loss_mask
            #error=torch.nan_to_num(error)
            loss = adaptive_l2_loss(error)#+adaptive_l2_loss(error_1)
            #loss = F.mse_loss(u, stopgrad(u_tgt)) 
            loss_maa = adaptive_l2_loss(error_maa)
            loss = loss + self.ssl_weight*loss_maa
            # loss = F.mse_loss(u, stopgrad(u_tgt))
            if self.apply_dispersive_to_embeddings:
                loss = loss + dispersiveLoss
            #mse_val = torch.mean(error ** 2)  # 所有元素的平均
        elif self.mean_mode == "MF_ssl_a":
            v = a0-target
            if self.apply_dispersive_to_embeddings:
                t_embed, r_embed ,c_embed= self.actor_model.get_embedding(xt, t, r, global_cond)
                c_loss=self.dispersive_loss(c_embed)
                t_loss = self.dispersive_loss(t_embed)
                r_loss = self.dispersive_loss(r_embed)
                dispersiveLoss=c_loss+t_loss+r_loss
            # 无条件引导
            if self.apply_dispersive_to_latents:
                for down_latent in self.actor_model.down_latents_tensor:
                    dispersiveLoss = dispersiveLoss + self.dispersive_loss(down_latent)
            if self.w is not None:
                with torch.no_grad():
                    u_t=self.actor_model(xt, t, t, global_cond=global_cond, local_cond=local_cond)
                    u_t_maa=self.actor_model(xt, t, t, global_cond=global_cond, local_cond=local_cond,enable_mask=True,mask_type='a')
                v_hat=(1-self.w)*u_t+v*self.w
                v_hat_maa=(1-self.w)*u_t_maa+v*self.w
            else:
                v_hat=v
                v_hat_maa=v                         
            
            #v_hat=v
            #v_hat_maa=v
            model_partial = partial(self.actor_model, global_cond=global_cond)
            model_partial_maa = partial(self.actor_model, global_cond=global_cond, enable_mask=True,mask_type='a')
            if self.sample_type == 't_r':
                jvp_args = (
                    lambda xt, t, r: model_partial(xt, t, r),  
                    (xt, t, r),
                    (v_hat, torch.ones_like(t), torch.zeros_like(r)),
                )
                
                jvp_args_maa = (
                    lambda xt, t, r: model_partial_maa(xt, t, r),  
                    (xt, t, r),
                    (v_hat_maa, torch.ones_like(t), torch.zeros_like(r)),
                )
            elif self.sample_type == 't_c':
                jvp_args = (
                    lambda xt, t, r: model_partial(xt, t, t-r),  
                    (xt, t, r),
                    (v_hat, torch.ones_like(t), torch.zeros_like(r)),
                )
                
                jvp_args_maa = (
                    lambda xt, t, r: model_partial_maa(xt, t, t-r),  
                    (xt, t, r),
                    (v_hat_maa, torch.ones_like(t), torch.zeros_like(r)),
                )
            if self.create_graph:
                u, dudt = self.jvp_fn(*jvp_args, create_graph=True)
                u_maa, dudt_maa = self.jvp_fn(*jvp_args_maa, create_graph=True)
            else:
                u, dudt = self.jvp_fn(*jvp_args)
                u_maa, dudt_maa = self.jvp_fn(*jvp_args_maa)   
            if self.sample_type == 't_r':     
                u_tgt = v_hat - (t_ - r_) * dudt
                u_tgt_maa = v_hat - (t_ - r_) * dudt_maa
            elif self.sample_type == 't_c':
                u_tgt = v_hat - (t_ - r_) * dudt 
                u_tgt_maa = v_hat - (t_ - r_) * dudt_maa
        
            error = (u - stopgrad(u_tgt))#*loss_mask
            error_maa = (u_maa - stopgrad(u_tgt_maa))*loss_mask
            #error=torch.nan_to_num(error)
            loss = adaptive_l2_loss(error)#+adaptive_l2_loss(error_1)
            #loss = F.mse_loss(u, stopgrad(u_tgt)) 
            loss_maa = adaptive_l2_loss(error_maa)
            loss = loss + self.ssl_weight*loss_maa
            # loss = F.mse_loss(u, stopgrad(u_tgt))
            if self.apply_dispersive_to_embeddings:
                loss = loss + dispersiveLoss
            #mse_val = torch.mean(error ** 2)  # 所有元素的平均
        elif self.mean_mode == "MF_ssl_ta":
            v = a0-target
            if self.apply_dispersive_to_embeddings:
                t_embed, r_embed ,c_embed= self.actor_model.get_embedding(xt, t, r, global_cond)
                c_loss=self.dispersive_loss(c_embed)
                t_loss = self.dispersive_loss(t_embed)
                r_loss = self.dispersive_loss(r_embed)
                dispersiveLoss=c_loss+t_loss+r_loss
            # 无条件引导
            if self.apply_dispersive_to_latents:
                for down_latent in self.actor_model.down_latents_tensor:
                    dispersiveLoss = dispersiveLoss + self.dispersive_loss(down_latent)
            if self.w is not None:
                with torch.no_grad():
                    u_t=self.actor_model(xt, t, t, global_cond=global_cond, local_cond=local_cond)
                    u_t_maa=self.actor_model(xt, t, t, global_cond=global_cond, local_cond=local_cond,enable_mask=True,mask_type='t_a')
                v_hat=(1-self.w)*u_t+v*self.w
                v_hat_maa=(1-self.w)*u_t_maa+v*self.w
            else:
                v_hat=v
                v_hat_maa=v                         
            
            #v_hat=v
            #v_hat_maa=v
            model_partial = partial(self.actor_model, global_cond=global_cond)
            model_partial_maa = partial(self.actor_model, global_cond=global_cond, enable_mask=True,mask_type='t_a')
            if self.sample_type == 't_r':
                jvp_args = (
                    lambda xt, t, r: model_partial(xt, t, r),  
                    (xt, t, r),
                    (v_hat, torch.ones_like(t), torch.zeros_like(r)),
                )
                
                jvp_args_maa = (
                    lambda xt, t, r: model_partial_maa(xt, t, r),  
                    (xt, t, r),
                    (v_hat_maa, torch.ones_like(t), torch.zeros_like(r)),
                )
            elif self.sample_type == 't_c':
                jvp_args = (
                    lambda xt, t, r: model_partial(xt, t, t-r),  
                    (xt, t, r),
                    (v_hat, torch.ones_like(t), torch.zeros_like(r)),
                )
                
                jvp_args_maa = (
                    lambda xt, t, r: model_partial_maa(xt, t, t-r),  
                    (xt, t, r),
                    (v_hat_maa, torch.ones_like(t), torch.zeros_like(r)),
                )
            if self.create_graph:
                u, dudt = self.jvp_fn(*jvp_args, create_graph=True)
                u_maa, dudt_maa = self.jvp_fn(*jvp_args_maa, create_graph=True)
            else:
                u, dudt = self.jvp_fn(*jvp_args)
                u_maa, dudt_maa = self.jvp_fn(*jvp_args_maa)   
            if self.sample_type == 't_r':     
                u_tgt = v_hat - (t_ - r_) * dudt
                u_tgt_maa = v_hat - (t_ - r_) * dudt_maa
            elif self.sample_type == 't_c':
                u_tgt = v_hat - (t_ - r_) * dudt 
                u_tgt_maa = v_hat - (t_ - r_) * dudt_maa
        
            error = (u - stopgrad(u_tgt))#*loss_mask
            error_maa = (u_maa - stopgrad(u_tgt_maa))*loss_mask
            #error=torch.nan_to_num(error)
            loss = adaptive_l2_loss(error)#+adaptive_l2_loss(error_1)
            #loss = F.mse_loss(u, stopgrad(u_tgt)) 
            loss_maa = adaptive_l2_loss(error_maa)
            loss = loss + self.ssl_weight*loss_maa
            # loss = F.mse_loss(u, stopgrad(u_tgt))
            if self.apply_dispersive_to_embeddings:
                loss = loss + dispersiveLoss
            #mse_val = torch.mean(error ** 2)  # 所有元素的平均
        elif self.mean_mode == "MF_ssl_t_a":
            v = a0-target
            if self.apply_dispersive_to_embeddings:
                t_embed, r_embed ,c_embed= self.actor_model.get_embedding(xt, t, r, global_cond)
                c_loss=self.dispersive_loss(c_embed)
                t_loss = self.dispersive_loss(t_embed)
                r_loss = self.dispersive_loss(r_embed)
                dispersiveLoss=c_loss+t_loss+r_loss
            # 无条件引导
            if self.apply_dispersive_to_latents:
                for down_latent in self.actor_model.down_latents_tensor:
                    dispersiveLoss = dispersiveLoss + self.dispersive_loss(down_latent)
            if self.w is not None:
                with torch.no_grad():
                    u_t=self.actor_model(xt, t, t, global_cond=global_cond, local_cond=local_cond)
                    u_t_maa_t=self.actor_model(xt, t, t, global_cond=global_cond, local_cond=local_cond,enable_mask=True,mask_type='t')
                    u_t_maa_a=self.actor_model(xt, t, t, global_cond=global_cond, local_cond=local_cond,enable_mask=True,mask_type='a')
                v_hat=(1-self.w)*u_t+v*self.w
                v_hat_maa_t=(1-self.w)*u_t_maa_t+v*self.w
                v_hat_maa_a=(1-self.w)*u_t_maa_a+v*self.w
            else:
                v_hat=v
                v_hat_maa_t=v 
                v_hat_maa_a=v                         
            
            #v_hat=v
            #v_hat_maa=v
            model_partial = partial(self.actor_model, global_cond=global_cond)
            model_partial_maa_t = partial(self.actor_model, global_cond=global_cond, enable_mask=True,mask_type='t')
            model_partial_maa_a = partial(self.actor_model, global_cond=global_cond, enable_mask=True,mask_type='a')
            if self.sample_type == 't_r':
                jvp_args = (
                    lambda xt, t, r: model_partial(xt, t, r),  
                    (xt, t, r),
                    (v_hat, torch.ones_like(t), torch.zeros_like(r)),
                )
                
                jvp_args_maa_t = (
                    lambda xt, t, r: model_partial_maa_t(xt, t, r),  
                    (xt, t, r),
                    (v_hat_maa_t, torch.ones_like(t), torch.zeros_like(r)),
                )

                jvp_args_maa_a = (
                    lambda xt, t, r: model_partial_maa_a(xt, t, r),  
                    (xt, t, r),
                    (v_hat_maa_a, torch.ones_like(t), torch.zeros_like(r)),
                )
            elif self.sample_type == 't_c':
                jvp_args = (
                    lambda xt, t, r: model_partial(xt, t, t-r),  
                    (xt, t, r),
                    (v_hat, torch.ones_like(t), torch.zeros_like(r)),
                )
                
                jvp_args_maa_t = (
                    lambda xt, t, r: model_partial_maa_t(xt, t, t-r),  
                    (xt, t, r),
                    (v_hat_maa_t, torch.ones_like(t), torch.zeros_like(r)),
                )

                jvp_args_maa_a = (
                    lambda xt, t, r: model_partial_maa_a(xt, t, t-r),  
                    (xt, t, r),
                    (v_hat_maa_a, torch.ones_like(t), torch.zeros_like(r)),
                )
            if self.create_graph:
                u, dudt = self.jvp_fn(*jvp_args, create_graph=True)
                u_maa_t, dudt_maa_t = self.jvp_fn(*jvp_args_maa_t, create_graph=True)
                u_maa_a, dudt_maa_a = self.jvp_fn(*jvp_args_maa_a, create_graph=True)
            else:
                u, dudt = self.jvp_fn(*jvp_args)
                u_maa_t, dudt_maa_t = self.jvp_fn(*jvp_args_maa_t)   
                u_maa_a, dudt_maa_a = self.jvp_fn(*jvp_args_maa_a)
            if self.sample_type == 't_r':     
                u_tgt = v_hat - (t_ - r_) * dudt
                u_tgt_maa_t = v_hat - (t_ - r_) * dudt_maa_t
                u_tgt_maa_a = v_hat - (t_ - r_) * dudt_maa_a
            elif self.sample_type == 't_c':
                u_tgt = v_hat - (t_ - r_) * dudt 
                u_tgt_maa_t = v_hat - (t_ - r_) * dudt_maa_t
                u_tgt_maa_a = v_hat - (t_ - r_) * dudt_maa_a
        
            error = (u - stopgrad(u_tgt))#*loss_mask
            error_maa_t = (u_maa_t - stopgrad(u_tgt_maa_t))*loss_mask
            error_maa_a = (u_maa_a - stopgrad(u_tgt_maa_a))*loss_mask
            #error=torch.nan_to_num(error)
            loss = adaptive_l2_loss(error)#+adaptive_l2_loss(error_1)
            #loss = F.mse_loss(u, stopgrad(u_tgt)) 
            loss_maa_t = adaptive_l2_loss(error_maa_t)
            loss_maa_a = adaptive_l2_loss(error_maa_a)

            loss = loss + (self.ssl_weight*loss_maa_t + self.ssl_weight*loss_maa_a)/2
            # loss = F.mse_loss(u, stopgrad(u_tgt))
            if self.apply_dispersive_to_embeddings:
                loss = loss + dispersiveLoss
            #mse_val = torch.mean(error ** 2)  # 所有元素的平均
        elif self.mean_mode == "IMF":
            v = self.actor_model(xt, t, t, global_cond=global_cond, local_cond=local_cond)
            v_hat=v
            v_maa = self.actor_model(xt, t, t, global_cond=global_cond, local_cond=local_cond,enable_mask=True)
            v_hat_maa = v_maa
            model_partial = partial(self.actor_model, global_cond=global_cond)
            model_partial_maa = partial(self.actor_model, global_cond=global_cond, enable_mask=True)
            jvp_args = (
                lambda xt, t, r: model_partial(xt, t, r),  
                (xt, t, r),
                (v_hat, torch.ones_like(t), torch.zeros_like(r)),
            )
            
            jvp_args_maa = (
                lambda xt, t, r: model_partial_maa(xt, t, r),  
                (xt, t, r),
                (v_hat_maa, torch.ones_like(t), torch.zeros_like(r)),
            )
            if self.create_graph:
                u, dudt = self.jvp_fn(*jvp_args, create_graph=True)
                u_maa, dudt_maa = self.jvp_fn(*jvp_args_maa, create_graph=True)
            else:
                u, dudt = self.jvp_fn(*jvp_args)    
                u_maa, dudt_maa = self.jvp_fn(*jvp_args_maa)
            if self.sample_type == 't_r':     
                V = u + (t_ - r_) * stopgrad(dudt)
                V_maa = u_maa + (t_ - r_) * stopgrad(dudt_maa)
            elif self.sample_type == 't_c':
                V = u + (r_) * stopgrad(dudt)
                V_maa = u_maa + (r_) * stopgrad(dudt_maa)
            
            error = (V - (a0-target))*loss_mask
            error_maa = (V_maa - (a0-target))*loss_mask
            loss = adaptive_l2_loss(error)
            loss_maa = adaptive_l2_loss(error_maa)
            loss = loss + self.ssl_weight*loss_maa
        
        elif self.mean_mode == "IMF_w":
            v = self.actor_model(xt, t, t, global_cond=global_cond, local_cond=local_cond)
            v_g = self.w*(a0-target) + (1-self.w)*v
            v_hat=v
            model_partial = partial(self.actor_model, global_cond=global_cond)
            jvp_args = (
                lambda xt, t, r: model_partial(xt, t, r),
                (xt, t, r),
                (v_hat, torch.ones_like(t), torch.zeros_like(r)),
            )
            if self.create_graph:
                u, dudt = self.jvp_fn(*jvp_args, create_graph=True)
            else:
                u, dudt = self.jvp_fn(*jvp_args)    
            if self.sample_type == 't_r':     
                V = u + (t_ - r_) * stopgrad(dudt)
            elif self.sample_type == 't_c':
                V = u + (r_) * stopgrad(dudt)
            
            error = V - v_g
            loss = adaptive_l2_loss(error)
            loss = loss
            #loss = torch.mean(error ** 2)
        if self.eta != 0 and self.use_subgoal:
            if self.single_step_flow:
                # 单次逆扩散，由Xt直接生成X0
                pred_action_seq = self._sample_actions_flow(nbatch, global_cond, fast=True)
            else:
                # 完整逆扩散
                pred_action_seq = self._sample_actions_flow(nbatch, global_cond, fast=False)
            new_action = pred_action_seq[:, self.actor_n_obs_steps-1:
                                        self.actor_n_obs_steps-1+self.critic_n_action_steps]
            new_action = new_action.reshape((B, -1))
            pcd = nbatch['pcd'].transpose(1, 2).reshape(
                (B, -1, self.pcd_dim*self.actor_n_obs_steps))  # (B, 1024, 3n)
            subgoal = nbatch['subgoal'] # (B, 8)
            state = nbatch['state'].reshape((B, -1))
            q1_new_action, q2_new_action = self.critic_model(
                pcd, state, subgoal, new_action)

            if np.random.uniform() > 0.5:
                q_loss = - q1_new_action.mean() / q2_new_action.abs().mean().detach()
            else:
                q_loss = - q2_new_action.mean() / q1_new_action.abs().mean().detach()

            actor_loss = loss + self.eta*q_loss

        else:
            q_loss = torch.tensor(-1, device=self.device)
            actor_loss = loss   
         
        return actor_loss
    def _sample_actions_flow(self, nbatch, global_cond, fast=False):

        B , T, _ = nbatch['state'].shape
    
        shape = (B, self.actor_horizon, self.actor_action_dim)
        # 运行采样过程
        noise = torch.randn(size=shape, dtype=self.dtype, device=self.device)
        z = noise.detach().clone() # 初始噪声
        n_steps = self.num_inference_step if not fast else 1
        t_vals = torch.linspace(1.0, 0.0, n_steps + 1, device=self.device)
        # 循环采样步骤
        for i in range(n_steps):
            t = torch.ones(z.shape[0], device=noise.device) * t_vals[i]
            r = torch.ones(z.shape[0], device=noise.device) * t_vals[i+1]
            if self.sample_type == 't_r':
                pred_v = self.actor_model(z, t, r, local_cond=None, global_cond=global_cond)
            elif self.sample_type == 't_c':
                pred_v = self.actor_model(z, t, t-r, local_cond=None, global_cond=global_cond)
            
            t_expanded = t.view(-1, 1, 1)  # [8, 1, 1]
            r_expanded = r.view(-1, 1, 1)  # [8, 1, 1]
            z = z - (t_expanded - r_expanded) * pred_v

        # 反归一化预测的动作
        naction_pred = z[...,:self.actor_action_dim]
        
        return naction_pred
    def test_guider(self, batch):
        """ """
        from Mix_diffusion_policy.common.pytorch_util import dict_apply

        Tbatch = dict_apply(batch, lambda x: x.to(self.device, non_blocking=True))
        subgoal = self.guider_predict_target(Tbatch).detach().to('cpu').numpy()
        B = batch['state'].shape[0]
        reward = np.zeros((B,))
        subgoal_gt = batch['subgoal'].numpy()
        '''
        visual_pred_subgoals(
            state=batch['state'][:, -1],
            subgoal=subgoal_gt,
            reward=reward,
            object_pcd=batch['pcd'][:, -1], 
            scene_pcd=batch['scene_pcd'][0]
        )
        '''
        visual_pred_and_gt_subgoals(
            state=batch['state'][:, -1],
            subgoal_pred=subgoal,
            subgoal_gt=subgoal_gt,
            reward=reward,
            object_pcd=batch['pcd'][:, -1], 
            scene_pcd=batch['scene_pcd'][0])
    def test_critic(self, batch):
        """
        测试critic
        """
        nbatch = self.normalizer.normalize(batch, self.subgoal_dim_nocont)
        B = nbatch['state'].shape[0]

        # 当前状态
        pcd = None
        pcd = nbatch['pcd'].transpose(1, 2).reshape(
                (B, -1, self.pcd_dim*self.actor_n_obs_steps))  # (B, 1024, 3n)
        state = nbatch['state'].reshape((B, -1))
        subgoal = nbatch['subgoal'] # (B, 8)
        action = nbatch['action'][:, self.actor_n_obs_steps-1:
                                            self.actor_n_obs_steps-1+self.critic_n_action_steps] # (B, A)
        action = action.reshape((B, -1))
        
        reward = batch['reward']   # (B, 1)
        dones = torch.zeros((B, 1), device=self.device)
        dones[reward==10] = 1

        current_q1, current_q2 = self.critic_model(
            pcd, state, subgoal, action)
        
        # print('cur_action =\n', cur_action[0])
        # print('q =', current_q1[0], current_q2[0])

        for i in range(B):
            # print('cur_action =\n', cur_action[i])
            print('r =', reward[i])
            print('q =', current_q1[i], current_q2[i])    

def stopgrad(x):
    return x.detach()


def adaptive_l2_loss(error, gamma=0.5, c=1e-3):
    """
    Adaptive L2 loss for trajectory-shaped tensors
    Args:
        error: Tensor of shape (B, horizon, action_dim) or similar
        gamma: Power used in original ||Δ||^{2γ} loss
        c: Small constant for stability
    Returns:
        Scalar loss
    """
    # 计算每个样本的均方误差
    if error.dim() == 3:  # (B, H, D)
        delta_sq = torch.mean(error ** 2, dim=(1, 2), keepdim=False)  # (B,)
    elif error.dim() == 2:  # (B, D)
        delta_sq = torch.mean(error ** 2, dim=1, keepdim=False)  # (B,)
    else:
        delta_sq = torch.mean(error ** 2, dim=tuple(range(1, error.dim())), keepdim=False)  # (B,)
    
    p = 1.0 - gamma
    w = 1.0 / (delta_sq + c).pow(p)
    loss_per_sample = delta_sq  # ||Δ||^2 for each sample
    
    return (stopgrad(w) * loss_per_sample).mean()




def visual_pred_subgoals(state, subgoal, reward, scene_pcd, object_pcd):
    """可视化subgoal"""
    # 构建手指点云
    finger_radius = 0.008
    ft_mesh = o3d.geometry.TriangleMesh.create_sphere(radius=finger_radius, resolution=5)
    finger_pcd = np.asarray(ft_mesh.vertices)

    for step in range(state.shape[0])[::20]:
        print('='*20)
        print('step:', step, 'reward:', reward[step])
        print('subgoal =', subgoal[step])    
        fig = plt.figure(figsize=(15, 15))
        ax = fig.add_subplot(projection='3d')

        # 可视化场景
        ax.scatter(*tuple(scene_pcd.transpose(1, 0)), color=Color.color('black'))

        # 可视化当前对象
        obj_color = np.array([[139, 105, 20]]).repeat(object_pcd[step].shape[0], axis=0)/255.
        ax.scatter(*tuple(object_pcd[step].transpose(1, 0)), color=obj_color)

        # 可视化当前手指
        fl_pos = state[step, -6:-3]
        fr_pos = state[step, -3:]
        fl_pcd = tf.transPts_tq(finger_pcd, fl_pos, [0, 0, 0, 1])
        fr_pcd = tf.transPts_tq(finger_pcd, fr_pos, [0, 0, 0, 1])
        fl_pcd_color = np.array([[0, 0, 0]]).repeat(fl_pcd.shape[0], axis=0)/255.
        fr_pcd_color = np.array([[0, 0, 0]]).repeat(fr_pcd.shape[0], axis=0)/255.
        ax.scatter(*tuple(fl_pcd.transpose(1, 0)), color=fl_pcd_color)
        ax.scatter(*tuple(fr_pcd.transpose(1, 0)), color=fr_pcd_color)

        # 可视化subgoals
        # 左手指
        fl_sg_pcd = tf.transPts_tq(finger_pcd, subgoal[step, :3], (0, 0, 0, 1))
        fl_sg_color = np.array([[255, 0, 0]]).repeat(fl_sg_pcd.shape[0], axis=0)/255.
        ax.scatter(*tuple(fl_sg_pcd.transpose(1, 0)), color=fl_sg_color)
        # 右手指
        fr_sg_pcd = tf.transPts_tq(finger_pcd, subgoal[step, 3:6], (0, 0, 0, 1))
        fr_sg_color = np.array([[34, 139, 34]]).repeat(fr_sg_pcd.shape[0], axis=0)/255.
        ax.scatter(*tuple(fr_sg_pcd.transpose(1, 0)), color=fr_sg_color)
        
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')

        ax.set_zlim(0.79, 1.05)
        plt.xticks(np.arange(-0.3, 0.3, 0.05))
        plt.yticks(np.arange(-0.3, 0.3, 0.05))
        plt.show()


def visual_pred_and_gt_subgoals(state, subgoal_pred, subgoal_gt, reward, scene_pcd, object_pcd):
    """
    可视化预测子目标和真实子目标（Ground Truth）

    参数:
        state: (T, ...) 状态序列，最后6维为左右手指位置 (fl_pos, fr_pos)
        subgoal_pred: (T, 8) 预测子目标 [fl_x,fl_y,fl_z, fr_x,fr_y,fr_z, active_fl, active_fr]
        subgoal_gt:   (T, 8) 真实子目标，格式同上
        reward: (T,) 奖励序列
        scene_pcd: (N,3) 场景固定点云
        object_pcd: (T, M, 3) 每个时间步的物体点云
    """
    # 构建手指模型点云（球体顶点）
    finger_radius = 0.008
    ft_mesh = o3d.geometry.TriangleMesh.create_sphere(radius=finger_radius, resolution=5)
    finger_pcd = np.asarray(ft_mesh.vertices)

    # 每隔20步显示一帧
    for step in range(0, state.shape[0], 20):
        print('=' * 20)
        print('step:', step, 'reward:', reward[step])
        print('subgoal_pred =', subgoal_pred[step])
        print('subgoal_gt   =', subgoal_gt[step])

        fig = plt.figure(figsize=(15, 15))
        ax = fig.add_subplot(projection='3d')

        # 1. 场景点云（黑色）
        ax.scatter(*tuple(scene_pcd.transpose(1, 0)), color=Color.color('black'))

        # 2. 物体点云（棕色）
        obj_color = np.array([[139, 105, 20]]).repeat(object_pcd[step].shape[0], axis=0) / 255.
        ax.scatter(*tuple(object_pcd[step].transpose(1, 0)), color=obj_color)

        # 3. 当前手指位置（黑色球体）
        fl_pos = state[step, -6:-3]
        fr_pos = state[step, -3:]
        fl_pcd = tf.transPts_tq(finger_pcd, fl_pos, [0, 0, 0, 1])
        fr_pcd = tf.transPts_tq(finger_pcd, fr_pos, [0, 0, 0, 1])
        fl_pcd_color = np.array([[0, 0, 0]]).repeat(fl_pcd.shape[0], axis=0) / 255.
        fr_pcd_color = np.array([[0, 0, 0]]).repeat(fr_pcd.shape[0], axis=0) / 255.
        ax.scatter(*tuple(fl_pcd.transpose(1, 0)), color=fl_pcd_color)
        ax.scatter(*tuple(fr_pcd.transpose(1, 0)), color=fr_pcd_color)

        # 4. 预测子目标（红色左手指，绿色右手指）
        pred = subgoal_pred[step]
        if pred[6] == 1:
            fl_sg_pcd = tf.transPts_tq(finger_pcd, pred[:3], [0, 0, 0, 1])
            fl_sg_color = np.array([[255, 0, 0]]).repeat(fl_sg_pcd.shape[0], axis=0) / 255.
            ax.scatter(*tuple(fl_sg_pcd.transpose(1, 0)), color=fl_sg_color, label='Pred Left')
        if pred[7] == 1:
            fr_sg_pcd = tf.transPts_tq(finger_pcd, pred[3:6], [0, 0, 0, 1])
            fr_sg_color = np.array([[34, 139, 34]]).repeat(fr_sg_pcd.shape[0], axis=0) / 255.
            ax.scatter(*tuple(fr_sg_pcd.transpose(1, 0)), color=fr_sg_color, label='Pred Right')

        # 5. 真实子目标（蓝色左手指，橙色右手指）
        gt = subgoal_gt[step]
        if gt[6] == 1:
            fl_gt_pcd = tf.transPts_tq(finger_pcd, gt[:3], [0, 0, 0, 1])
            fl_gt_color = np.array([[0, 0, 255]]).repeat(fl_gt_pcd.shape[0], axis=0) / 255.
            ax.scatter(*tuple(fl_gt_pcd.transpose(1, 0)), color=fl_gt_color, marker='*', s=80, label='GT Left')
        if gt[7] == 1:
            fr_gt_pcd = tf.transPts_tq(finger_pcd, gt[3:6], [0, 0, 0, 1])
            fr_gt_color = np.array([[255, 165, 0]]).repeat(fr_gt_pcd.shape[0], axis=0) / 255.
            ax.scatter(*tuple(fr_gt_pcd.transpose(1, 0)), color=fr_gt_color, marker='*', s=80, label='GT Right')

        # 坐标轴设置
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        ax.set_zlim(0.79, 1.05)
        plt.xticks(np.arange(-0.3, 0.3, 0.05))
        plt.yticks(np.arange(-0.3, 0.3, 0.05))
        ax.legend()
        plt.show()