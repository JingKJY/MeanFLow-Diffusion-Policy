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
#from Mix_diffusion_policy.model.diffusion.guider_transform import Guider
from Mix_diffusion_policy.model.diffusion.guider_mlp import Guider
from Mix_diffusion_policy.model.flow.actor_mean import Actor
from Mix_diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
from Mix_diffusion_policy.common.pytorch_util import dict_apply 
from Mix_diffusion_policy.model.encoder.condition_encoder import PCDStateSubgoalEncoder
from Mix_diffusion_policy.model.encoder.pointnet_extractor import FlowPolicyEncoder
from Mix_diffusion_policy.model.diffusion.pointcloud_encoder import PointNetEncoder
import Mix_diffusion_policy.common.transformation as tf
from Mix_diffusion_policy.common.visual import Color, draw_pcl
from Mix_diffusion_policy.model.flow.u_dit_mean import RMDiT
from Mix_diffusion_policy.model.flow.dispersive_loss import DispersiveLoss

class MixPolicy(BasePcdPolicy):
    def __init__(self, 
            guider_noise_scheduler:DDPMScheduler,
            guider_horizon, 
            actor_horizon,
            guider_n_action_steps, 
            actor_n_action_steps,
            guider_n_obs_steps,
            actor_n_obs_steps,
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
            # parameters passed to step
            **kwargs):
        super().__init__()
            
        guider_obs_encoder = PCDStateSubgoalEncoder(
            pcd_dim, 
            state_dim, 
            subgoal_dim, 
            obs_dim=guider_n_obs_steps,
            is_use_subgoal=False, 
            is_use_film=True,
            hidden_dim=guider_hidden_dim, 
            film_dim=film_dim,
            output_dim=guider_cond_dim,
            pcd_encoder=guider_pcd_encoder,
            type="total"
        )
        '''
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
        )'''
        guider_model = Guider(
            input_dim=subgoal_dim,
            cond_dim=guider_cond_dim,
            mlp_dims=[1024, 512, 256]
        )
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
        #MeanFLowConfig
        self.time_dist = MeanFlow['time_dist']
        self.flow_ratio = MeanFlow['flow_ratio']
        self.jvp_api = MeanFlow['jvp_api']
        self.num_inference_step = MeanFlow['num_inference_step']
        self.cfg_ratio = MeanFlow['cfg_ratio']
        self.w = MeanFlow['cfg_scale']
        self.cfg_uncond = MeanFlow['cfg_uncond']
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
            '''model_output = self.guider_model(sample=target_pred,
                                timestep=t, 
                                cond=global_cond)'''
            model_output = self.guider_model(subgoal=target_pred,
                                timestep=t, 
                                cond=global_cond)
            
            # 3. compute previous image: x_t -> x_t-1
            target_pred = scheduler.step(
                model_output, t, target_pred, ).prev_sample
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
            pred_v = self.actor_model(z, t, r, local_cond=None, global_cond=obs)
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
        #pred = self.guider_model(noisy_sg, timesteps, obs)
        pred = self.guider_model(obs, noisy_sg, timesteps)
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
        t_np = np.maximum(samples[:, 0], samples[:, 1])
        r_np = np.minimum(samples[:, 0], samples[:, 1])

        num_selected = int(self.flow_ratio * batch_size)
        indices = np.random.permutation(batch_size)[:num_selected]
        r_np[indices] = t_np[indices]
 
        t = torch.tensor(t_np, device=self.device)
        r = torch.tensor(r_np, device=self.device)
        return t, r   

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
        v = a0-target
        if self.apply_dispersive_to_embeddings:
            t_embed, r_embed ,c_embed= self.actor_model.get_embedding(xt, t, r, global_cond)
            c_loss=self.dispersive_loss(c_embed)
            t_loss = self.dispersive_loss(t_embed)
            r_loss = self.dispersive_loss(r_embed)
            dispersiveLoss=c_loss+t_loss+r_loss
        # 无条件引导
        '''
        if self.w is not None:
            with torch.no_grad():
                u_t=self.actor_model(xt, t, t, global_cond=global_cond, local_cond=local_cond)
            v_hat=(1-self.w)*u_t+v*self.w
        else:
            v_hat=v                         
        '''
        v_hat=v
        model_partial = partial(self.actor_model, global_cond=global_cond)
        #model_partial_maa = partial(self.actor_model, global_cond=global_cond, enable_mask=True)
        jvp_args = (
            lambda xt, t, r: model_partial(xt, t, r),  
            (xt, t, r),
            (v_hat, torch.ones_like(t), torch.zeros_like(r)),
        )
        '''
        jvp_args_maa = (
            lambda xt, t, r: model_partial_maa(xt, t*99, r*99),  
            (xt, t, r),
            (v_hat, torch.ones_like(t), torch.zeros_like(r)),
        )'''
        if self.create_graph:
            u, dudt = self.jvp_fn(*jvp_args, create_graph=True)
            #u_maa, dudt_maa = self.jvp_fn(*jvp_args_maa, create_graph=True)
        else:
            u, dudt = self.jvp_fn(*jvp_args)
            #u_maa, dudt_maa = self.jvp_fn(*jvp_args_maa)        
        u_tgt = v_hat - (t_ - r_) * dudt
        #u_tgt_maa = v_hat - (t_ - r_) * dudt_maa
        #print(f"u_tgt:{u_tgt.shape}, u_tgt_maa:{u_tgt_maa.shape}")
        #print(f"loss_mask:{loss_mask.shape}")
        error = (u - stopgrad(u_tgt))#*loss_mask
        ##error_maa = (u_maa - stopgrad(u_tgt_maa))*loss_mask
        loss = adaptive_l2_loss(error)
        #loss = F.mse_loss(u, stopgrad(u_tgt)) 
        ##loss_maa = adaptive_l2_loss(error_maa)
        #loss = loss + self.ssl_weight*loss_maa
        # loss = F.mse_loss(u, stopgrad(u_tgt))
        if self.apply_dispersive_to_embeddings:
            loss = loss + dispersiveLoss
        #mse_val = torch.mean(error ** 2)  # 所有元素的平均
        return loss  #, mse_val

    def test_guider(self, batch):
        """ """
        from Mix_diffusion_policy.common.pytorch_util import dict_apply

        Tbatch = dict_apply(batch, lambda x: x.to(self.device, non_blocking=True))
        subgoal = self.guider_predict_target(Tbatch).detach().to('cpu').numpy()
        B = batch['state'].shape[0]
        reward = np.zeros((B,))

        visual_pred_subgoals(
            state=batch['state'][:, -1],
            subgoal=subgoal,
            reward=reward,
            object_pcd=batch['pcd'][:, -1], 
            scene_pcd=batch['scene_pcd'][0])
        
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
        if subgoal[step, 6] == 1:
            fl_sg_pcd = tf.transPts_tq(finger_pcd, subgoal[step, :3], (0, 0, 0, 1))
            fl_sg_color = np.array([[255, 0, 0]]).repeat(fl_sg_pcd.shape[0], axis=0)/255.
            ax.scatter(*tuple(fl_sg_pcd.transpose(1, 0)), color=fl_sg_color)
        # 右手指
        if subgoal[step, 7] == 1:
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