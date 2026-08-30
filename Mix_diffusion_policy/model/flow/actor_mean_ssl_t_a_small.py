from typing import Union
import logging
import torch
import torch.nn as nn 
import torch.nn.functional as F
import einops
from einops.layers.torch import Rearrange
from termcolor import cprint
from Mix_diffusion_policy.model.diffusion.conv1d_components import (
    Downsample1d, Upsample1d, Conv1dBlock)
from Mix_diffusion_policy.model.diffusion.positional_embedding import SinusoidalPosEmb



logger = logging.getLogger(__name__)

class CrossAttention(nn.Module):
    def __init__(self, in_dim, cond_dim, out_dim):
        super().__init__()
        self.query_proj = nn.Linear(in_dim, out_dim)
        self.key_proj = nn.Linear(cond_dim, out_dim)
        self.value_proj = nn.Linear(cond_dim, out_dim)

    def forward(self, x, cond):
        # x: [batch_size, t_act, in_dim]
        # cond: [batch_size, t_obs, cond_dim]

        # Project x and cond to query, key, and value
        query = self.query_proj(x)  # [batch_size, horizon, out_dim]
        key = self.key_proj(cond)   # [batch_size, horizon, out_dim]
        value = self.value_proj(cond)  # [batch_size, horizon, out_dim]


        # Compute attention
        attn_weights = torch.matmul(query, key.transpose(-2, -1))  # [batch_size, horizon, horizon]
        attn_weights = F.softmax(attn_weights, dim=-1)

        # Apply attention
        attn_output = torch.matmul(attn_weights, value)  # [batch_size, horizon, out_dim]
        
        return attn_output
    

class ConditionalResidualBlock1D(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 cond_dim,
                 kernel_size=3,
                 n_groups=8,
                 condition_type='film'):
        super().__init__()

        self.blocks = nn.ModuleList([
            Conv1dBlock(in_channels,
                        out_channels,
                        kernel_size,
                        n_groups=n_groups),
            Conv1dBlock(out_channels,
                        out_channels,
                        kernel_size,
                        n_groups=n_groups),
        ])

        
        self.condition_type = condition_type

        cond_channels = out_channels
        if condition_type == 'film': # FiLM modulation https://arxiv.org/abs/1709.07871
            # predicts per-channel scale and bias
            cond_channels = out_channels * 2
            self.cond_encoder = nn.Sequential(
                nn.Mish(),
                nn.Linear(cond_dim, cond_channels),
                Rearrange('batch t -> batch t 1'),
            )
        elif condition_type == 'add':
            self.cond_encoder = nn.Sequential(
                nn.Mish(),
                nn.Linear(cond_dim, out_channels),
                Rearrange('batch t -> batch t 1'),
            )
        elif condition_type == 'cross_attention_add':
            self.cond_encoder = CrossAttention(in_channels, cond_dim, out_channels)
        elif condition_type == 'cross_attention_film':
            cond_channels = out_channels * 2
            self.cond_encoder = CrossAttention(in_channels, cond_dim, cond_channels)
        elif condition_type == 'mlp_film':
            cond_channels = out_channels * 2
            self.cond_encoder = nn.Sequential(
                nn.Mish(),
                nn.Linear(cond_dim, cond_dim),
                nn.Mish(),
                nn.Linear(cond_dim, cond_channels),
                Rearrange('batch t -> batch t 1'),
            )
        else:
            raise NotImplementedError(f"condition_type {condition_type} not implemented")
        
        self.out_channels = out_channels
        # make sure dimensions compatible
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) \
            if in_channels != out_channels else nn.Identity()
        self.skip_conv = nn.Sequential(
            nn.SiLU(),
            nn.Conv1d(in_channels, in_channels, kernel_size=1)
        )
    def forward(self, x, cond=None, skip=None):
        '''
            x : [ batch_size x in_channels x horizon ]
            cond : [ batch_size x cond_dim]

            returns:
            out : [ batch_size x out_channels x horizon ]
        '''
        if self.skip_conv and skip is not None:
            x = x+self.skip_conv(skip)
        
        out = self.blocks[0](x)  
        if cond is not None:      
            if self.condition_type == 'film':
                embed = self.cond_encoder(cond)
                embed = embed.reshape(embed.shape[0], 2, self.out_channels, 1)
                scale = embed[:, 0, ...]
                bias = embed[:, 1, ...]
                out = scale * out + bias
            elif self.condition_type == 'add':
                embed = self.cond_encoder(cond)
                out = out + embed
            elif self.condition_type == 'cross_attention_add':
                embed = self.cond_encoder(x.permute(0, 2, 1), cond)
                embed = embed.permute(0, 2, 1) # [batch_size, out_channels, horizon]
                out = out + embed
            elif self.condition_type == 'cross_attention_film':
                embed = self.cond_encoder(x.permute(0, 2, 1), cond)
                embed = embed.permute(0, 2, 1)
                embed = embed.reshape(embed.shape[0], 2, self.out_channels, -1)
                scale = embed[:, 0, ...]
                bias = embed[:, 1, ...]
                out = scale * out + bias
            elif self.condition_type == 'mlp_film':
                embed = self.cond_encoder(cond)
                embed = embed.reshape(embed.shape[0], 2, self.out_channels, -1)
                scale = embed[:, 0, ...]
                bias = embed[:, 1, ...]
                out = scale * out + bias
            else:
                raise NotImplementedError(f"condition_type {self.condition_type} not implemented")
        out = self.blocks[1](out)
        out = out + self.residual_conv(x)
        return out


class Actor(nn.Module):
    def __init__(self, 
        input_dim,
        local_cond_dim=None,
        global_cond_dim=None,
        diffusion_step_embed_dim=256,
        down_dims=[256,512,1024],
        kernel_size=3,
        n_groups=8,
        condition_type='film',
        use_down_condition=True,
        use_mid_condition=True,
        use_up_condition=True,
        use_decoder=False,
        use_subgoal_as_cond=False,
        horizon=2,
        side_layers=2,  
        decoder_layers=2,
        mask_ratio=[0.1,0.2,0.2,0.2],
        type_ssl="t_a",
        ):
        super().__init__()
        self.condition_type = condition_type
        
        start_dim = down_dims[0]
        all_dims = [start_dim] + list(down_dims)
        

        dsed = diffusion_step_embed_dim
        diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )
        cond_dim = dsed
        if global_cond_dim is not None:
            cond_dim += global_cond_dim*horizon

        in_out = list(zip(all_dims[:-1], all_dims[1:]))


        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList([
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups,
                condition_type=condition_type
            ),
        ])

        down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            down_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_in, dim_out, cond_dim=cond_dim, 
                    kernel_size=kernel_size, n_groups=n_groups,
                    condition_type=condition_type),
                # ConditionalResidualBlock1D(
                #     dim_in, dim_in, cond_dim=cond_dim,
                #     kernel_size=kernel_size, n_groups=n_groups,
                #     cond_predict_scale=cond_predict_scale),
                Downsample1d(dim_out) if not is_last else nn.Identity()
            ]))

        up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            up_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_out*2, dim_in, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups,
                    condition_type=condition_type),
                Upsample1d(dim_in) if not is_last else nn.Identity()
            ]))
        
        self.side_layers_a = nn.ModuleList(ConditionalResidualBlock1D(
                    start_dim, start_dim, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups,
                    condition_type=condition_type) for _ in range(side_layers))
        
        self.side_layers_t = nn.ModuleList(ConditionalResidualBlock1D(
                    start_dim, start_dim, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups,
                    condition_type=condition_type) for _ in range(side_layers))
        
        self.decoder_layers = nn.ModuleList(ConditionalResidualBlock1D(
                    start_dim, start_dim, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups,
                    condition_type=condition_type) for _ in range(decoder_layers))
        
        start_conv = nn.Sequential(
            Conv1dBlock(input_dim, start_dim, kernel_size=kernel_size),
            nn.Conv1d(start_dim, start_dim, 1),
        )
        final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size),
            nn.Conv1d(start_dim, input_dim, 1),
        )
        

        self.diffusion_step_encoder = diffusion_step_encoder
        self.up_modules = up_modules
        self.down_modules = down_modules
        self.final_conv = final_conv
        self.start_conv = start_conv
        self.mask_ratio = mask_ratio
        self.mask_token = nn.Parameter(torch.randn(1, start_dim, 1))
        self.type_ssl = type_ssl
        self.use_decoder = use_decoder
        self.use_subgoal_as_cond = use_subgoal_as_cond
        logger.info(
            "number of parameters: %e", sum(p.numel() for p in self.parameters())
        )

    def random_masking_t(self, x, mask_ratio):
        """
        对输入 x 进行随机掩码，将掩码位置替换为可学习的 mask_token。
        x: (N, D, L)
        mask_ratio: 掩码比例
        返回:
            x_masked: (N, D, L) 掩码后的张量（掩码位置为 mask_token）
            mask: (N, L) 浮点掩码，1 表示掩码，0 表示保留
        """
        N, D, L = x.shape
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]

        mask = torch.ones(N, L, device=x.device, dtype=torch.float32)
        mask.scatter_(dim=1, index=ids_keep, value=0.0)  # 1: mask, 0: keep

        # 生成 mask_token 并广播到与 x 相同形状
        # mask_token 形状为 (1, D, 1)，可广播到 (N, D, L)
        mask_token = self.mask_token.expand(N, -1, L)  # (N, D, L)
        
        # 保留位置保持原值，掩码位置替换为 mask_token
        x_masked = torch.where(mask.unsqueeze(1) == 1, mask_token, x)
        
        return x_masked, mask
    
    def random_masking_a(self, x, mask_ratio):
        """
        在特征维度 D 上对输入 x 进行随机掩码。
        x: (N, D, L)
        mask_ratio: 掩码比例
        返回:
            x_masked: (N, D, L) 掩码后的张量（掩码位置为 mask_token）
            mask: (N, D) 浮点掩码，1 表示掩码，0 表示保留
        """
        N, D, L = x.shape
        len_keep = int(D * (1 - mask_ratio))          # 保留的特征维度数

        # 生成随机噪声，按特征维度排序
        noise = torch.rand(N, D, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)     # (N, D) 排序后的索引
        ids_keep = ids_shuffle[:, :len_keep]           # 保留的索引

        # 创建掩码 (N, D)，1 表示掩码，0 表示保留
        mask = torch.ones(N, D, device=x.device, dtype=torch.float32)
        mask.scatter_(dim=1, index=ids_keep, value=0.0)

        # mask_token 形状为 (1, D, 1)，已包含完整的 D 维信息
        # 通过广播机制与 x 对齐，掩码位置替换为 mask_token
        x_masked = torch.where(mask.unsqueeze(2) == 1, self.mask_token, x)

        return x_masked, mask
            
    def forward_side_interpolater_t(self, x, c, mask):
        # x: (N, D, L) 原始长度，掩码位置已置零
        # c: 条件
        # mask: (N, L) 布尔或浮点，1表示掩码，0表示保留
        x_before = x  # 保存原始输入
        for sideblock in self.side_layers_t:
            x = sideblock(x, c)   # 通过 side_modules 处理
        # 混合：掩码位置使用 sideblock 输出，非掩码位置保持原始值
        mask = mask.unsqueeze(1)  # (N, 1, L)
        x = x * mask + (1 - mask) * x_before
        return x
    def forward_side_interpolater_a(self, x, c, mask):
        """
        x: (N, D, L)
        c: 条件
        mask: (N, D) - 1 表示掩码，0 表示保留
        """
        x_before = x
        for sideblock in self.side_layers_a:
            x = sideblock(x, c)
        mask = mask.unsqueeze(2)          # (N, D, 1)
        x = x * mask + (1 - mask) * x_before
        return x
    def forward_side_interpolater(self, x, c, mask):
        """
        x: (N, D, L)
        c: 条件
        mask: (N, D) - 1 表示掩码，0 表示保留
        """
        x_before = x
        for sideblock in self.side_layers:
            x = sideblock(x, c)
        x = x * mask + (1 - mask) * x_before
        return x    
    def forward(self, 
            sample: torch.Tensor, 
            timestep_t: Union[torch.Tensor, float, int],
            timestep_r: Union[torch.Tensor, float, int],
            local_cond=None, global_cond=None, enable_mask=False, mask_type=None, **kwargs):
        """
        x: (B,T,input_dim)
        timestep: (B,) or int, diffusion step
        local_cond: (B,T,local_cond_dim)
        global_cond: (B,global_cond_dim)
        output: (B,T,input_dim)
        """
        sample = einops.rearrange(sample, 'b h t -> b t h')
        # print (f"sample: {sample.shape}")
        # 1. time
        timesteps_t = timestep_t
        timesteps_r = timestep_r
        #print(f"timestep: {timestep.shape}")
        if not torch.is_tensor(timesteps_t) and not torch.is_tensor(timesteps_r):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            timesteps_r = torch.tensor([timesteps_r], dtype=torch.float, device=sample.device)
            timesteps_t = torch.tensor([timesteps_t], dtype=torch.float, device=sample.device)
        elif torch.is_tensor(timesteps_t) and len(timesteps_t.shape) == 0 and not torch.is_tensor(timesteps_r) and not torch.is_tensor(timesteps_t):
            timesteps_r = timesteps_r[None].to(sample.device)
            timesteps_t = timesteps_t[None].to(sample.device)
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps_t = timesteps_t.expand(sample.shape[0]).to(sample.device)
        timesteps_r = timesteps_r.expand(sample.shape[0]).to(sample.device)
        timestep_embed_t = self.diffusion_step_encoder(timesteps_t)
        timestep_embed_r = self.diffusion_step_encoder(timesteps_r)
        #timestep_embed = torch.cat([timestep_embed_t, timestep_embed_r], dim=-1)
        timestep_embed = timestep_embed_t + timestep_embed_r
        #print(f"timestep_embed: {timestep_embed.shape}")
        if global_cond is not None:
            if self.condition_type == 'cross_attention_film':
                timestep_embed = timestep_embed.unsqueeze(1).expand(-1, global_cond.shape[1], -1)
            if self.use_subgoal_as_cond:
                global_cond = global_cond.view(global_cond.size(0), -1)
                global_feature = torch.cat([timestep_embed, global_cond], axis=-1)
            else:
                global_feature = timestep_embed

        
        x = self.start_conv(sample)
        x_skip = x
        h = []

        masked_stage = False
        if self.mask_ratio is not None and enable_mask:
            if mask_type == "t_a":
                rand_mask_ratio = torch.rand(1, device=x.device)  # noise in [0, 1]
                rand_mask_ratio_t = rand_mask_ratio * self.mask_ratio[1] + self.mask_ratio[0] # mask_ratio, mask_ratio + 0.2 
                rand_mask_ratio = torch.rand(1, device=x.device)  # noise in [0, 1]
                rand_mask_ratio_a = rand_mask_ratio * self.mask_ratio[3] + self.mask_ratio[2] # mask_ratio, mask_ratio + 0.2 
                x, mask_t = self.random_masking_t(x, rand_mask_ratio_t)
                x, mask_a = self.random_masking_a(x, rand_mask_ratio_a)
            elif mask_type == "t":
                rand_mask_ratio = torch.rand(1, device=x.device)  # noise in [0, 1]
                rand_mask_ratio_t = rand_mask_ratio * self.mask_ratio[1] + self.mask_ratio[0]
                x, mask_t = self.random_masking_t(x, rand_mask_ratio_t)
            elif mask_type == "a":
                rand_mask_ratio = torch.rand(1, device=x.device)  # noise in [0, 1]
                rand_mask_ratio_a = rand_mask_ratio * self.mask_ratio[3] + self.mask_ratio[2]
                x, mask_a = self.random_masking_a(x, rand_mask_ratio_a)

            #combined_mask = mask_t.unsqueeze(1) * mask_a.unsqueeze(2)
            #combined_mask = torch.logical_or(mask_t.unsqueeze(1), mask_a.unsqueeze(2)).float()
            masked_stage = True      
      
        for idx, (resnet, downsample) in enumerate(self.down_modules):
            x = resnet(x, global_feature)
            h.append(x)
            x = downsample(x)


        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)


        for idx, (resnet, upsample) in enumerate(self.up_modules):
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, global_feature)
            x = upsample(x)

        if self.mask_ratio is not None and enable_mask:
            #x = self.forward_side_interpolater_a(x, global_feature, mask)
            if mask_type=="t_a":
                x_t = self.forward_side_interpolater_t(x, global_feature, mask_t)
                x_a = self.forward_side_interpolater_a(x, global_feature, mask_a)
                x = (x_t + x_a)/2
            elif mask_type=="t":
                x = self.forward_side_interpolater_t(x, global_feature, mask_t)
            elif mask_type=="a":
                x = self.forward_side_interpolater_a(x, global_feature, mask_a)
            masked_stage = False
        if self.use_decoder:
            for decode_module in self.decoder_layers:
                x= decode_module(x, global_feature, x_skip)
        x = self.final_conv(x)

        x = einops.rearrange(x, 'b t h -> b h t')

        return x
    
