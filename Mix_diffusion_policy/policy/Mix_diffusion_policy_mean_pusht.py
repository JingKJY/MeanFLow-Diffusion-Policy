from typing import Dict
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from diffusers import DDPMScheduler
import copy
import numpy as np
from functools import partial

from Mix_diffusion_policy.model.flow.sde_lib import ConsistencyFM
from Mix_diffusion_policy.model.common.normalizer_pusht import LinearNormalizer
from Mix_diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy
from Mix_diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
from Mix_diffusion_policy.model.flow.actor_mean_ssl_t_a_small import Actor
from Mix_diffusion_policy.model.flow.dispersive_loss import DispersiveLoss
from Mix_diffusion_policy.model.flow.u_dit_mean import RMDiT
from Mix_diffusion_policy.common.visual import visual_pushT_pred_subgoals

class MixPolicy(BaseLowdimPolicy):
    def __init__(self,
                 # 基本参数
                 horizon: int,
                 action_dim: int,
                 subgoal_dim: int,
                 subgoal_dim_nocont: int,
                 state_dim: int,
                 observation_history_num: int = 2,
                 # guider 参数
                 guider_horizon: int = 16,
                 guider_n_action_steps: int = 1,
                 guider_n_obs_steps: int = 2,
                 guider_mode: str = "transformer",
                 guider_model_cfg: dict = None,
                 guider_hidden_dim: int = 256,
                 guider_cond_dim: int = 128,
                 # actor 参数
                 actor_horizon: int = 16,
                 actor_n_action_steps: int = 8,
                 actor_n_obs_steps: int = 2,
                 actor_mode: str = "UNet",
                 actor_model_cfg: dict = None,
                 actor_hidden_dim: int = 256,
                 actor_cond_dim: int = 128,
                 # 流匹配参数
                 mean_mode: str = "MF_ssl_t_a",
                 MeanFlow: dict = None,
                 # 其他
                 ssl_weight: float = 1.0,
                 use_subgoal: bool = True,
                 dispersive_loss_cfg: dict = None,
                 apply_dispersive_to_embeddings: bool = False,
                 apply_dispersive_to_latents: bool = False,
                 # 噪声调度器 (可选，内部创建)
                 guider_noise_scheduler: DDPMScheduler = None,
                 **kwargs):
        super().__init__()

        # 保存参数
        self.horizon = horizon
        self.action_dim = action_dim
        self.obs_dim = state_dim
        self.subgoal_dim = subgoal_dim
        self.subgoal_dim_nocont = subgoal_dim_nocont
        self.observation_history_num = observation_history_num

        self.guider_horizon = guider_horizon
        self.guider_n_action_steps = guider_n_action_steps
        self.guider_n_obs_steps = guider_n_obs_steps
        self.guider_mode = guider_mode
        self.guider_hidden_dim = guider_hidden_dim
        self.guider_cond_dim = guider_cond_dim

        self.actor_horizon = actor_horizon
        self.actor_n_action_steps = actor_n_action_steps
        self.actor_n_obs_steps = actor_n_obs_steps
        self.actor_mode = actor_mode
        self.actor_hidden_dim = actor_hidden_dim
        self.actor_cond_dim = actor_cond_dim

        self.mean_mode = mean_mode
        self.ssl_weight = ssl_weight
        self.use_subgoal = use_subgoal
        self.apply_dispersive_to_embeddings = apply_dispersive_to_embeddings
        self.apply_dispersive_to_latents = apply_dispersive_to_latents


        # 创建编码器（低维观测版本）
        self.guider_obs_encoder = self._build_obs_encoder(
            obs_dim=self.obs_dim,
            subgoal_dim=subgoal_dim,
            n_obs_steps=guider_n_obs_steps,
            hidden_dim=guider_hidden_dim,
            output_dim=guider_cond_dim,
            use_subgoal=False
        )
        self.actor_obs_encoder = self._build_obs_encoder_actor(
            obs_dim=self.obs_dim,
            subgoal_dim=subgoal_dim,
            n_obs_steps=actor_n_obs_steps,
            hidden_dim=actor_hidden_dim,
            output_dim=actor_cond_dim,
            use_subgoal=use_subgoal
        )

        # 创建 guider 模型
        if guider_mode == "transformer":
            from Mix_diffusion_policy.model.diffusion.guider_transform import Guider
            self.guider_model = Guider(
                input_dim=subgoal_dim,
                output_dim=subgoal_dim,
                horizon=guider_horizon,
                n_obs_steps=guider_n_obs_steps,
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
            from Mix_diffusion_policy.model.diffusion.guider_lowdim import Guider
            from Mix_diffusion_policy.model.diffusion.positional_embedding import TimestepEncoder
            diffusion_step_encoder=TimestepEncoder(diffusion_step_embed_dim=guider_model_cfg['n_emb'])
            self.guider_model = Guider(
                diffusion_step_encoder,
                state_dim=self.obs_dim*self.observation_history_num,
                subgoal_dim=self.subgoal_dim,
                mlp_dims=[1024, 512, 256]
            )
        else:
            raise ValueError(f"Unknown guider_mode: {guider_mode}")

        # 创建 actor 模型
        if actor_mode == "UDit":
            self.actor_model = RMDiT(
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
        elif actor_mode == "UNet":
            self.actor_model = Actor(
                input_dim=action_dim+state_dim,
                local_cond_dim=None,
                global_cond_dim=self.subgoal_dim,#self.subgoal_dim/None
                diffusion_step_embed_dim=actor_model_cfg['diffusion_step_embed_dim'],
                down_dims=actor_model_cfg['down_dims'],
                kernel_size=actor_model_cfg['kernel_size'],
                n_groups=actor_model_cfg['n_groups'],
                condition_type=actor_model_cfg['condition_type'],
                type_ssl=actor_model_cfg.get('type_ssl', 'none'),
                mask_ratio=actor_model_cfg['mask_ratio'],
                use_down_condition=True,
                use_mid_condition=True,
                use_up_condition=True,
                use_decoder=actor_model_cfg['use_decoder'],
                use_subgoal_as_cond=True,
                horizon=1,
            )
        else:
            raise ValueError(f"Unknown actor_mode: {actor_mode}")

        # 噪声调度器
        if guider_noise_scheduler is None:
            self.guider_noise_scheduler = DDPMScheduler(
                num_train_timesteps=100,
                beta_start=0.0001,
                beta_end=0.02,
                prediction_type='epsilon'
            )
        else:
            self.guider_noise_scheduler = guider_noise_scheduler
        self.guider_noise_scheduler_pc = copy.deepcopy(self.guider_noise_scheduler)

        # Mask 生成器
        self.guider_mask_generator = LowdimMaskGenerator(
            action_dim=subgoal_dim,
            obs_dim=guider_cond_dim,
            max_n_obs_steps=guider_n_obs_steps,
            fix_obs_steps=True,
            action_visible=False
        )
        self.actor_mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=self.obs_dim,
            max_n_obs_steps=actor_n_obs_steps,
            fix_obs_steps=True,
            action_visible=False
        )

        # 归一化器
        self.normalizer = LinearNormalizer()

        # 流匹配参数
        self.time_dist = MeanFlow['time_dist']
        self.flow_ratio = MeanFlow['flow_ratio']
        self.sample_type = MeanFlow['sample_type']
        self.jvp_api = MeanFlow['jvp_api']
        self.num_inference_step = MeanFlow['num_inference_step']
        self.cfg_ratio = MeanFlow['cfg_ratio']
        self.w = MeanFlow['cfg_scale']
        self.cfg_uncond = MeanFlow['cfg_uncond']
        self.Mask_ratio = MeanFlow['cond_ratio']
        self.subgoal_token = nn.Parameter(torch.randn(1, self.subgoal_dim))
        assert self.jvp_api in ['funtorch', 'autograd']
        if self.jvp_api == 'funtorch':
            self.jvp_fn = torch.func.jvp
            self.create_graph = False
        else:
            self.jvp_fn = torch.autograd.functional.jvp
            self.create_graph = True

        # 分散损失
        self.dispersive_loss = DispersiveLoss(
            loss_type=dispersive_loss_cfg['dispersive_loss_type'],
            temperature=dispersive_loss_cfg['dispersive_temperature'],
            margin=dispersive_loss_cfg['dispersive_margin'],
            weight=dispersive_loss_cfg['dispersive_loss_weight'],
        )

    # ========= 辅助函数 =========
    def _build_obs_encoder(self, obs_dim, subgoal_dim, n_obs_steps, hidden_dim, output_dim, use_subgoal):
        """构建低维观测编码器（MLP）"""
        input_dim = obs_dim * n_obs_steps
        if use_subgoal:
            input_dim += subgoal_dim  # 子目标作为条件拼接
        layers = []
        cur_dim = input_dim
        for h in [hidden_dim, hidden_dim]:
            layers.append(nn.Linear(cur_dim, h))
            layers.append(nn.ReLU())
            cur_dim = h
        layers.append(nn.Linear(cur_dim, output_dim))
        return nn.Sequential(*layers)

    def _build_obs_encoder_actor(self, obs_dim, subgoal_dim, n_obs_steps, hidden_dim, output_dim, use_subgoal):
        """构建低维观测编码器（MLP）"""
        input_dim = subgoal_dim
        layers = []
        cur_dim = input_dim
        for h in [hidden_dim, hidden_dim]:
            layers.append(nn.Linear(cur_dim, h))
            layers.append(nn.ReLU())
            cur_dim = h
        layers.append(nn.Linear(cur_dim, output_dim))
        return nn.Sequential(*layers)
    def _prepare_obs_cond(self, nobs, subgoal=None, encoder=None, n_obs_steps=None):
        """准备条件向量：展平观测历史，可选拼接子目标"""
        B, T, _ = nobs.shape
        if n_obs_steps is None:
            n_obs_steps = self.observation_history_num
        # 取最近 n_obs_steps 步
        obs_flat = nobs[:, -n_obs_steps:].reshape(B, -1)  # (B, obs_dim * n_obs_steps)
        if subgoal is not None and self.use_subgoal:
            cond = torch.cat([obs_flat, subgoal], dim=-1)
        else:
            cond = obs_flat
        if encoder is not None:
            cond = encoder(cond)
        return cond
    
    def _prepare_obs_cond_actor(self, nobs, subgoal=None, encoder=None, n_obs_steps=None):
        """准备条件向量：展平观测历史，可选拼接子目标"""
        if encoder is not None:
            cond = encoder(subgoal)
        return cond
    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype
    def guider_predict_target(self, obs_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """预测子目标（流匹配采样）"""
        nobs = self.normalizer['obs'].normalize(obs_dict['obs'])
        B = nobs.shape[0]
        nobs = nobs[:, :self.guider_n_obs_steps].reshape((B, -1))
        # 初始化子目标（高斯噪声）
        assert self.subgoal_dim == 3
        target_pred = torch.randn(size=(B, self.subgoal_dim),
                                  dtype=self.dtype, device=self.device)

        # 准备条件
        '''global_cond = self._prepare_obs_cond(
            nobs, subgoal=None, encoder=self.guider_obs_encoder,
            n_obs_steps=self.guider_n_obs_steps
        ).float()'''
        
        # 扩散去噪
        scheduler = self.guider_noise_scheduler
        with torch.no_grad():
            for t in scheduler.timesteps:
                if self.guider_mode == "transformer":
                    model_output = self.guider_model(target_pred, t, nobs)
                else:  # mlp
                    model_output = self.guider_model(nobs,target_pred, t)
                target_pred = scheduler.step(model_output, t, target_pred, generator=None).prev_sample

        # 后处理：反归一化 + 接触掩码
        #target_pred = target_pred.reshape(B, -1)
        target_pred = self.normalizer['subgoal'].unnormalize(target_pred)
        target_pred[:, 2:] = torch.round(target_pred[:, 2:])
        target_pred[:, :2] *= target_pred[:, 2:3]
        return target_pred  # (B, subgoal_dim)
    
    def actor_predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        预测动作序列（流匹配采样，支持条件掩码强制）
        """
        nobs = self.normalizer['obs'].normalize(obs_dict['obs'])
        B, T_obs, Do = nobs.shape
        subgoal = self.normalizer['subgoal'].normalize(obs_dict['subgoal']) if 'subgoal' in obs_dict else None
        #subgoal = self._prepare_obs_cond_actor(nobs, subgoal=subgoal, encoder=self.actor_obs_encoder, n_obs_steps=self.actor_n_obs_steps)
        horizon = self.actor_horizon
        Da = self.action_dim
        shape = (B, horizon, Da + Do)
        cond_data = torch.zeros(size=shape, device=self.device, dtype=self.dtype)
        cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)

        # 已知观测位置：前 observation_history_num 步的 obs 维度
        cond_data[:, :self.observation_history_num, Da:] = nobs[:, :self.observation_history_num]
        cond_mask[:, :self.observation_history_num, Da:] = True

        # 初始轨迹（全噪声）
        trajectory = torch.randn(size=shape, device=self.device, dtype=self.dtype)

        # 欧拉采样
        t_vals = torch.linspace(1.0, 0.0, self.num_inference_step + 1, device=self.device).float()
        for i in range(self.num_inference_step):
            t = torch.ones(B, device=self.device) * t_vals[i]
            r = torch.ones(B, device=self.device) * t_vals[i + 1]

            # 强制已知观测位置
            trajectory[cond_mask] = cond_data[cond_mask]

            # 预测速度场
            if self.sample_type == 't_r':
                pred_v = self.actor_model(trajectory, t, r, local_cond=None, global_cond=subgoal)
            else:
                pred_v = self.actor_model(trajectory, t, t - r, local_cond=None, global_cond=subgoal)

            # 欧拉更新
            t_exp = t.view(-1, 1, 1)
            r_exp = r.view(-1, 1, 1)
            trajectory = trajectory - (t_exp - r_exp) * pred_v

        # 最终强制已知观测位置（防止数值漂移）
        trajectory[cond_mask] = cond_data[cond_mask]

        # 提取动作并反归一化
        naction_pred = trajectory[..., :Da]
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        start = self.observation_history_num - 1
        end = start + self.actor_n_action_steps
        action = action_pred[:, start:end]

        return {'action': action, 'action_pred': action_pred}
    '''
    def actor_predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        预测动作序列（流匹配采样，支持 CFG 引导）
        """
        nobs = self.normalizer['obs'].normalize(obs_dict['obs'])
        B, T_obs, Do = nobs.shape
        # 有条件条件：正常编码子目标
        subgoal = self.normalizer['subgoal'].normalize(obs_dict['subgoal']) if 'subgoal' in obs_dict else None
        global_cond_cond = subgoal
        
        # 无条件条件：全零（匹配训练时的 mask=0）
        global_cond_uncond = torch.zeros_like(global_cond_cond)

        horizon = self.actor_horizon
        Da = self.action_dim
        shape = (B, horizon, Da + Do)
        cond_data = torch.zeros(size=shape, device=self.device, dtype=self.dtype)
        cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)

        # 已知观测位置（前 observation_history_num 步的 obs 部分）
        cond_data[:, :self.observation_history_num, Da:] = nobs[:, :self.observation_history_num]
        cond_mask[:, :self.observation_history_num, Da:] = True

        # 初始轨迹（全噪声）
        trajectory = torch.randn(size=shape, device=self.device, dtype=self.dtype)

        # 欧拉采样
        t_vals = torch.linspace(1.0, 0.0, self.num_inference_step + 1, device=self.device).float()

        for i in range(self.num_inference_step):
            t = torch.ones(B, device=self.device) * t_vals[i]
            r = torch.ones(B, device=self.device) * t_vals[i + 1]

            # 强制已知观测位置
            trajectory[cond_mask] = cond_data[cond_mask]

            # 有条件预测
            if self.sample_type == 't_r':
                v_cond = self.actor_model(trajectory, t, r, local_cond=None, global_cond=global_cond_cond)
            else:
                v_cond = self.actor_model(trajectory, t, t - r, local_cond=None, global_cond=global_cond_cond)

            # 无条件预测（使用零条件）
            with torch.no_grad():
                if self.sample_type == 't_r':
                    v_uncond = self.actor_model(trajectory, t, r, local_cond=None, global_cond=global_cond_uncond)
                else:
                    v_uncond = self.actor_model(trajectory, t, t - r, local_cond=None, global_cond=global_cond_uncond)

            # CFG 组合：v_guided = v_uncond + cfg_scale * (v_cond - v_uncond)
            v_guided = v_uncond + self.w * (v_cond - v_uncond)

            # 欧拉更新
            t_exp = t.view(-1, 1, 1)
            r_exp = r.view(-1, 1, 1)
            trajectory = trajectory - (t_exp - r_exp) * v_guided

        # 最终强制已知观测位置
        trajectory[cond_mask] = cond_data[cond_mask]

        # 提取动作并反归一化
        naction_pred = trajectory[..., :Da]
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        start = self.observation_history_num - 1
        end = start + self.actor_n_action_steps
        action = action_pred[:, start:end]

        return {'action': action, 'action_pred': action_pred}'''
    # ========= 训练损失 =========
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def guider_compute_loss(self, batch):
        """Guider 损失（DDPM epsilon 预测）"""
        B = batch['obs'].shape[0]
        # Sample random timesteps
        timesteps = torch.randint(
            0, self.guider_noise_scheduler.config.num_train_timesteps, # 100
            (B,), device=self.device
        ).long()

        # add noise
        subgoal = self.normalizer['subgoal'].normalize(batch['subgoal'])
        noise = torch.randn(subgoal.shape, device=self.device)  # Sample noise
        noisy_sg = self.guider_noise_scheduler.add_noise(subgoal, noise, timesteps)

        # ** pred and loss **
        obs = self.normalizer['obs'].normalize(batch['obs'])
        obs = obs[:, :self.observation_history_num].reshape((B, -1))
        #print('obs', obs.shape, 'noisy_sg', noisy_sg.shape, 'timesteps', timesteps.shape)
        if self.guider_mode == "transformer":
            pred = self.guider_model(noisy_sg, timesteps, obs)
        else:
            pred = self.guider_model(obs, noisy_sg, timesteps)
        assert self.guider_noise_scheduler.config.prediction_type == 'epsilon'
        loss = F.mse_loss(pred, noise)
        return loss
    
    def actor_compute_loss(self, batch):
        """Actor 损失（流匹配，支持多种 mean_mode）"""
        action = self.normalizer['action'].normalize(batch['action'])
        B, T, Da = action.shape
        subgoal = self.normalizer['subgoal'].normalize(batch['subgoal']) if 'subgoal' in batch else None
        #subgoal =self.normalizer['subgoal'].normalize(self.guider_predict_target(batch))
        #subgoal = self._prepare_obs_cond_actor()
        subgoal_nomask = subgoal
        nobs = self.normalizer['obs'].normalize(batch['obs'])
        target = torch.cat([action, nobs], dim=-1)
        mask = torch.rand(B, device=self.device) < self.Mask_ratio
        mask_expanded = mask.view(B, 1)                 # (B,1)
        subgoal = torch.where(
            mask_expanded,
            self.subgoal_token.expand(B, -1),              # (B,D)
            subgoal
        )
        # 生成条件掩码（已知观测位置为 True）
        condition_mask = self.actor_mask_generator(target.shape)
        loss_mask = ~condition_mask   # 需要计算损失的位置

        # 采样 t, r
        a0 = torch.randn_like(target)
        t, r = self.sample_t_r(B)
        t_ = t.view(-1, 1, 1).expand_as(target)
        r_ = r.view(-1, 1, 1).expand_as(target)
        xt = (1 - t_) * target + t_ * a0
        xt[condition_mask] = target[condition_mask]   # 条件观测位置强制为真值

        # 根据 mean_mode 分发
        if self.mean_mode == "MF":
            return self._loss_mf(xt, target, a0, t, r, t_, r_, subgoal, loss_mask)
        elif self.mean_mode == "MF_nocond":
            return self._loss_mf_nocond(xt, target, a0, t, r, t_, r_, subgoal, loss_mask)
        elif self.mean_mode == "MF_ssl_t":
            return self._loss_mf_ssl(xt, target, a0, t, r, t_, r_, subgoal,subgoal_nomask, loss_mask, mask_type='t')
        elif self.mean_mode == "MF_ssl_a":
            return self._loss_mf_ssl(xt, target, a0, t, r, t_, r_, subgoal, loss_mask, mask_type='a')
        elif self.mean_mode == "MF_ssl_ta":
            return self._loss_mf_ssl(xt, target, a0, t, r, t_, r_, subgoal, loss_mask, mask_type='t_a')
        elif self.mean_mode == "MF_ssl_t_a":
            return self._loss_mf_ssl_t_a(xt, target, a0, t, r, t_, r_, subgoal, loss_mask)
        elif self.mean_mode == "IMF":
            return self._loss_imf(xt, target, a0, t, r, t_, r_, subgoal, loss_mask)
        elif self.mean_mode == "IMF_w":
            return self._loss_imf_w(xt, target, a0, t, r, t_, r_, subgoal, loss_mask)
        else:
            raise ValueError(f"Unknown mean_mode: {self.mean_mode}")

    # ---------- 损失子函数（已加入 loss_mask 应用）----------
    def _loss_mf(self, xt, target, a0, t, r, t_, r_, global_cond, loss_mask):
        v = a0 - target
        dispersive_loss = self._compute_dispersive_loss(xt, t, r, global_cond)

        model_partial = partial(self.actor_model, global_cond=global_cond)
        jvp_args = (
            lambda x, t, r: model_partial(x, t, r),
            (xt, t, r),
            (v, torch.ones_like(t), torch.zeros_like(r))
        )
        u, dudt = self.jvp_fn(*jvp_args, create_graph=True)
        u_tgt = v - (t_ - r_) * dudt
        error = u - u_tgt.detach()

        loss = adaptive_l2_loss(error, mask=loss_mask)
        if dispersive_loss is not None:
            if isinstance(dispersive_loss, torch.Tensor) and dispersive_loss.dim() > 0:
                dispersive_loss = adaptive_l2_loss(dispersive_loss, mask=loss_mask)
            loss = loss + dispersive_loss
        return loss

    def _loss_mf_nocond(self, xt, target, a0, t, r, t_, r_, global_cond, loss_mask):
        v = a0 - target
        dispersive_loss = self._compute_dispersive_loss(xt, t, r, global_cond)
        model_partial = partial(self.actor_model, global_cond=global_cond)
        jvp_args = (
            lambda x, t, r: model_partial(x, t, r),
            (xt, t, r),
            (v, torch.ones_like(t), torch.zeros_like(r))
        )
        u, dudt = self.jvp_fn(*jvp_args, create_graph=True)

        u_tgt = v - (t_ - r_) * dudt
        error = u - u_tgt.detach()

        loss = adaptive_l2_loss(error, mask=loss_mask)
        if dispersive_loss is not None:
            if isinstance(dispersive_loss, torch.Tensor) and dispersive_loss.dim() > 0:
                dispersive_loss = adaptive_l2_loss(dispersive_loss, mask=loss_mask)
            loss = loss + dispersive_loss
        return loss
    # ---------- MF_ssl 变体（单掩码类型） ----------
    def _loss_mf_ssl(self, xt, target, a0, t, r, t_, r_, global_cond,global_cond_nomask, loss_mask, mask_type):
        v = a0 - target
        dispersive_loss = self._compute_dispersive_loss(xt, t, r, global_cond)

        if self.w is not None:
            with torch.no_grad():
                u_t = self.actor_model(xt, t, t, global_cond=global_cond)
                u_t_maa = self.actor_model(xt, t, t, global_cond=global_cond_nomask, enable_mask=True, mask_type=mask_type)
            v_hat = (1 - self.w) * u_t + v * self.w
            v_hat_maa = (1 - self.w) * u_t_maa + v * self.w
        else:
            v_hat = v_hat_maa = v

        model_partial = partial(self.actor_model, global_cond=global_cond)
        model_partial_maa = partial(self.actor_model, global_cond=global_cond_nomask, enable_mask=True, mask_type=mask_type)

        jvp_args = (
            lambda x, t, r: model_partial(x, t, r),
            (xt, t, r),
            (v_hat, torch.ones_like(t), torch.zeros_like(r))
        )
        jvp_args_maa = (
            lambda x, t, r: model_partial_maa(x, t, r),
            (xt, t, r),
            (v_hat_maa, torch.ones_like(t), torch.zeros_like(r))
        )

        u, dudt = self.jvp_fn(*jvp_args, create_graph=True)
        u_maa, dudt_maa = self.jvp_fn(*jvp_args_maa, create_graph=True)

        u_tgt = v_hat - (t_ - r_) * dudt
        u_tgt_maa = v_hat - (t_ - r_) * dudt_maa

        error = u - u_tgt.detach()
        error_maa = u_maa - u_tgt_maa.detach()

        loss = adaptive_l2_loss(error, mask=loss_mask) + self.ssl_weight * adaptive_l2_loss(error_maa, mask=loss_mask)
        if dispersive_loss is not None:
            if isinstance(dispersive_loss, torch.Tensor) and dispersive_loss.dim() > 0:
                dispersive_loss = adaptive_l2_loss(dispersive_loss, mask=loss_mask)
            loss = loss + dispersive_loss
        return loss

    # ---------- MF_ssl_t_a 变体（双掩码类型） ----------
    def _loss_mf_ssl_t_a(self, xt, target, a0, t, r, t_, r_, global_cond, loss_mask):
        v = a0 - target
        dispersive_loss = self._compute_dispersive_loss(xt, t, r, global_cond)

        if self.w is not None:
            with torch.no_grad():
                u_t = self.actor_model(xt, t, t, global_cond=global_cond)
                u_t_maa_t = self.actor_model(xt, t, t, global_cond=global_cond, enable_mask=True, mask_type='t')
                u_t_maa_a = self.actor_model(xt, t, t, global_cond=global_cond, enable_mask=True, mask_type='a')
            v_hat = (1 - self.w) * u_t + v * self.w
            v_hat_maa_t = (1 - self.w) * u_t_maa_t + v * self.w
            v_hat_maa_a = (1 - self.w) * u_t_maa_a + v * self.w
        else:
            v_hat = v_hat_maa_t = v_hat_maa_a = v

        model_partial = partial(self.actor_model, global_cond=global_cond)
        model_partial_t = partial(self.actor_model, global_cond=global_cond, enable_mask=True, mask_type='t')
        model_partial_a = partial(self.actor_model, global_cond=global_cond, enable_mask=True, mask_type='a')

        jvp_args = (lambda x, t, r: model_partial(x, t, r), (xt, t, r), (v_hat, torch.ones_like(t), torch.zeros_like(r)))
        jvp_args_t = (lambda x, t, r: model_partial_t(x, t, r), (xt, t, r), (v_hat_maa_t, torch.ones_like(t), torch.zeros_like(r)))
        jvp_args_a = (lambda x, t, r: model_partial_a(x, t, r), (xt, t, r), (v_hat_maa_a, torch.ones_like(t), torch.zeros_like(r)))

        u, dudt = self.jvp_fn(*jvp_args, create_graph=True)
        u_t, dudt_t = self.jvp_fn(*jvp_args_t, create_graph=True)
        u_a, dudt_a = self.jvp_fn(*jvp_args_a, create_graph=True)

        u_tgt = v_hat - (t_ - r_) * dudt
        u_tgt_t = v_hat - (t_ - r_) * dudt_t
        u_tgt_a = v_hat - (t_ - r_) * dudt_a

        error = u - u_tgt.detach()
        error_t = u_t - u_tgt_t.detach()
        error_a = u_a - u_tgt_a.detach()

        loss = adaptive_l2_loss(error, mask=loss_mask)
        loss += self.ssl_weight * adaptive_l2_loss(error_t, mask=loss_mask)
        loss += self.ssl_weight * adaptive_l2_loss(error_a, mask=loss_mask)

        if dispersive_loss is not None:
            if isinstance(dispersive_loss, torch.Tensor) and dispersive_loss.dim() > 0:
                dispersive_loss = adaptive_l2_loss(dispersive_loss, mask=loss_mask)
            loss += dispersive_loss
        return loss

    # ---------- IMF 损失 ----------
    def _loss_imf(self, xt, target, a0, t, r, t_, r_, global_cond, loss_mask):
        v = self.actor_model(xt, t, t, global_cond=global_cond)
        v_maa = self.actor_model(xt, t, t, global_cond=global_cond, enable_mask=True)

        model_partial = partial(self.actor_model, global_cond=global_cond)
        model_partial_maa = partial(self.actor_model, global_cond=global_cond, enable_mask=True)

        jvp_args = (lambda x, t, r: model_partial(x, t, r), (xt, t, r), (v, torch.ones_like(t), torch.zeros_like(r)))
        jvp_args_maa = (lambda x, t, r: model_partial_maa(x, t, r), (xt, t, r), (v_maa, torch.ones_like(t), torch.zeros_like(r)))

        u, dudt = self.jvp_fn(*jvp_args, create_graph=True)
        u_maa, dudt_maa = self.jvp_fn(*jvp_args_maa, create_graph=True)

        V = u + (t_ - r_) * dudt.detach()
        V_maa = u_maa + (t_ - r_) * dudt_maa.detach()

        error = V - (a0 - target)
        error_maa = V_maa - (a0 - target)

        loss = adaptive_l2_loss(error, mask=loss_mask) + self.ssl_weight * adaptive_l2_loss(error_maa, mask=loss_mask)
        return loss

    # ---------- IMF_w 损失 ----------
    def _loss_imf_w(self, xt, target, a0, t, r, t_, r_, global_cond, loss_mask):
        v = self.actor_model(xt, t, t, global_cond=global_cond)
        v_g = self.w * (a0 - target) + (1 - self.w) * v

        model_partial = partial(self.actor_model, global_cond=global_cond)
        jvp_args = (lambda x, t, r: model_partial(x, t, r), (xt, t, r), (v_g, torch.ones_like(t), torch.zeros_like(r)))
        u, dudt = self.jvp_fn(*jvp_args, create_graph=True)
        V = u + (t_ - r_) * dudt.detach()

        error = V - v_g
        loss = adaptive_l2_loss(error, mask=loss_mask)
        return loss
    def _compute_dispersive_loss(self, xt, t, r, global_cond):
        if not self.apply_dispersive_to_embeddings and not self.apply_dispersive_to_latents:
            return None
        loss = 0.0
        if self.apply_dispersive_to_embeddings:
            t_embed, r_embed, c_embed = self.actor_model.get_embedding(xt, t, r, global_cond)
            loss += self.dispersive_loss(c_embed)
            loss += self.dispersive_loss(t_embed)
            loss += self.dispersive_loss(r_embed)
        if self.apply_dispersive_to_latents:
            for latent in getattr(self.actor_model, 'down_latents_tensor', []):
                loss += self.dispersive_loss(latent)
        return loss

    # ========= 辅助 =========
    def sample_t_r(self, batch_size):
        """采样 (t, r) 对"""
        if self.time_dist[0] == 'uniform':
            samples = np.random.rand(batch_size, 2).astype(np.float32)
        elif self.time_dist[0] == 'lognorm':
            mu, sigma = self.time_dist[-2], self.time_dist[-1]
            normal_samples = np.random.randn(batch_size, 2).astype(np.float32) * sigma + mu
            samples = 1 / (1 + np.exp(-normal_samples))
        elif self.time_dist[0] == 'marginal':
            alpha = self.time_dist[1] if len(self.time_dist) > 1 else 0.8
            samples = np.random.beta(alpha, alpha, size=(batch_size, 2)).astype(np.float32)
        else:
            raise ValueError(f"Unknown time_dist: {self.time_dist[0]}")
        t_np = np.maximum(samples[:, 0], samples[:, 1])
        r_np = np.minimum(samples[:, 0], samples[:, 1])
        num_selected = int(self.flow_ratio * batch_size)
        indices = np.random.permutation(batch_size)[:num_selected]
        r_np[indices] = t_np[indices]
        return torch.tensor(t_np, device=self.device), torch.tensor(r_np, device=self.device)
    def test_guider(self, batch):
        """ """
        from Mix_diffusion_policy.common.pytorch_util import dict_apply

        Tbatch = dict_apply(batch, lambda x: x.to(self.device, non_blocking=True))
        subgoal = self.guider_predict_target(Tbatch).detach().to('cpu').numpy()
        visual_pushT_pred_subgoals(batch['state'][:, self.observation_history_num-1], subgoal, batch['subgoal'])

def stopgrad(x):
    return x.detach()

def adaptive_l2_loss(error, mask=None, gamma=0.5, c=1e-3):
    """
    自适应 L2 损失，若提供 mask（bool 或 float），则只计算 mask==True 的区域。
    """
    if mask is not None:
        mask = mask.to(error.dtype)
        valid = mask.sum()
        if valid == 0:
            return torch.tensor(0.0, device=error.device, requires_grad=True)
        sq_error = (error ** 2) * mask
        loss = sq_error.sum() / valid
        return loss
    else:
        # 原始自适应逻辑（无掩码）
        if error.dim() == 3:
            delta_sq = torch.mean(error ** 2, dim=(1, 2))
        elif error.dim() == 2:
            delta_sq = torch.mean(error ** 2, dim=1)
        else:
            delta_sq = torch.mean(error ** 2, dim=tuple(range(1, error.dim())))
        p = 1.0 - gamma
        w = 1.0 / (delta_sq + c).pow(p)
        loss_per_sample = delta_sq
        return (w.detach() * loss_per_sample).mean()
