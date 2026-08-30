from typing import Union
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from einops.layers.torch import Rearrange
from termcolor import cprint
from Mix_diffusion_policy.model.UMamba.conv1d_components import (
    Downsample1d, Upsample1d, Conv1dBlock)
from Mix_diffusion_policy.model.diffusion.positional_embedding import SinusoidalPosEmb

# from mamba_ssm import Mamba, Mamba2

from mamba_ssm.modules.mamba_simple import Mamba
try:
    from Mix_diffusion_policy.model.UMamba.hydra_ssm import Hydra
except ImportError:
    Hydra = None

try:
    from mamba_ssm.modules.mamba2 import Mamba2
except ImportError:
    Mamba2 = None
# from mamba_ssm.ops.triton.layernorm import RMSNorm
from timm.models.layers import DropPath

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


class Attention(nn.Module):

    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_norm=False,
            attn_drop=0.,
            proj_drop=0.,
            norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = False

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        
        if self.fused_attn:
            x = F.scaled_dot_product_attention(
             q, k, v,
                dropout_p=self.attn_drop.p,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x 


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class MambaVisionBlock(nn.Module):
    def __init__(self, 
                 dim, 
                 mlp_ratio=4., 
                 drop=0., 
                 drop_path=0.2, 
                 act_layer=nn.GELU, 
                 norm_layer=nn.LayerNorm, 
                 Mlp_block=Mlp,
                 layer_scale=None,
                 mamba_type='v1',  ## ['v1', 'v2', 'bi', 'hydra']
                 ):
        super().__init__()

        cprint(f'Using Mamba {mamba_type} version', 'yellow')
        self.norm1 = norm_layer(dim)
        if mamba_type == 'v1':
            self.mixer = Mamba(d_model=dim)
        elif mamba_type == 'v2':
            self.mixer = Mamba2(d_model=dim, headdim=32)
        elif mamba_type == 'bi':
            self.mixer = Mamba(d_model=dim, bimamba=True)
        elif mamba_type == 'hydra':
            self.mixer = Hydra(d_model=dim)
        else:
            NotImplementedError(f"mamba_type {mamba_type} not implemented")
                 
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_block(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        use_layer_scale = layer_scale is not None and type(layer_scale) in [int, float]
        self.gamma_1 = nn.Parameter(layer_scale * torch.ones(dim))  if use_layer_scale else 1
        self.gamma_2 = nn.Parameter(layer_scale * torch.ones(dim))  if use_layer_scale else 1

    def forward(self, x):
        '''
        x : [ batch_size x in_channels x horizon ]
        '''
        x = x.permute(0, 2, 1)
        x = x + self.drop_path(self.gamma_1 * self.mixer(self.norm1(x)))
        x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        x = x.permute(0, 2, 1)

        return x

class AttnVisionBlock(nn.Module):
    def __init__(self, 
                 dim, 
                 num_heads=8,
                 qkv_bias=False, 
                 qk_scale=False, 
                 attn_drop=0.,
                 mlp_ratio=4., 
                 drop=0., 
                 drop_path=0.2, 
                 act_layer=nn.GELU, 
                 norm_layer=nn.LayerNorm, 
                 Mlp_block=Mlp,
                 layer_scale=None,
                 ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.mixer = Attention(
                        dim,
                        num_heads=num_heads,
                        qkv_bias=qkv_bias,
                        qk_norm=qk_scale,
                        attn_drop=attn_drop,
                        proj_drop=drop,
                        norm_layer=norm_layer,
                    )
                 
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_block(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        use_layer_scale = layer_scale is not None and type(layer_scale) in [int, float]
        self.gamma_1 = nn.Parameter(layer_scale * torch.ones(dim))  if use_layer_scale else 1
        self.gamma_2 = nn.Parameter(layer_scale * torch.ones(dim))  if use_layer_scale else 1

    def forward(self, x):
        '''
        x : [ batch_size x in_channels x horizon ]
        '''
        x = x.permute(0, 2, 1)
        x = x + self.drop_path(self.gamma_1 * self.mixer(self.norm1(x)))
        x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        x = x.permute(0, 2, 1)

        return x

class ConditionalMambaResidualBlock1D(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 cond_dim,
                 kernel_size=3,
                 n_groups=8,
                 condition_type='film',
                 mamba_version='mambavision_v1',
                 skip=False,
                 ):
        super().__init__()

        if 'mambavision' in mamba_version:
            if 'v1' in mamba_version:
                mamba_type = 'v1'
            elif 'v2' in mamba_version:
                mamba_type = 'v2'
            elif 'bi' in mamba_version:
                mamba_type = 'bi'
            elif 'hydra' in mamba_version:
                mamba_type = 'hydra'
            else:
                NotImplementedError(f"mamba_version {mamba_version} not implemented")
            self.blocks = nn.ModuleList([
                Conv1dBlock(in_channels,
                            out_channels,
                            kernel_size,
                            n_groups=n_groups),
                MambaVisionBlock(out_channels, mamba_type=mamba_type),
                AttnVisionBlock(out_channels,),
            ])
        else:
            raise NotImplementedError(f"mamba_version {mamba_version} not implemented")
        
        self.mamba_version = mamba_version
        
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
        self.skip_linear = nn.Sequential(
            nn.SiLU(),
            nn.Conv1d(in_channels, in_channels, kernel_size=1)
        ) if skip else None
    def forward(self, x, cond=None, skip=None, time_cond=None):
        '''
            x : [ batch_size x in_channels x horizon ]
            cond : [ batch_size x cond_dim]

            returns:
            out : [ batch_size x out_channels x horizon ]
        '''
        if self.skip_linear and skip is not None:
            x = x+self.skip_linear(skip)
        
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

        if 'mambavision' in self.mamba_version:
            out = self.blocks[2](out)
        out = out + self.residual_conv(x)

        return out

class ConditionalMambaUnet1D(nn.Module):
    def __init__(self, 
        input_dim,
        global_cond_dim=None,
        diffusion_step_embed_dim=256,
        hide_dim=256,
        kernel_size=3,
        n_groups=8,
        condition_type='film',
        use_down_condition=True,
        use_mid_condition=True,
        use_up_condition=True,
        mamba_version='mambavision_v1',
        decode_layer=3,
        side_layer=1,
        mask_ratio=0.5,
        depth:int =28,
        horizon=2,
        ):
        super().__init__()
        self.condition_type = condition_type
        
        self.use_down_condition = use_down_condition
        self.use_mid_condition = use_mid_condition
        self.use_up_condition = use_up_condition

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

        half_depth = depth // 2

        self.en_inmodules = nn.ModuleList([
            ConditionalMambaResidualBlock1D(
                    in_channels=hide_dim, out_channels=hide_dim, 
                    cond_dim=cond_dim, kernel_size=kernel_size, n_groups=n_groups,
                    condition_type=condition_type, mamba_version=mamba_version) for i in range(half_depth)
        ])
        self.en_outmodules = nn.ModuleList([
            ConditionalMambaResidualBlock1D(
                    in_channels=hide_dim, out_channels=hide_dim, 
                    cond_dim=cond_dim, kernel_size=kernel_size, n_groups=n_groups,
                    condition_type=condition_type, mamba_version=mamba_version,skip=True) for i in range(half_depth)
        ])
        side_modules = nn.ModuleList([])
        decode_modules = nn.ModuleList([])

        for i in range(side_layer):
            side_modules.append(ConditionalMambaResidualBlock1D(
                    in_channels=hide_dim, out_channels=hide_dim, 
                    cond_dim=cond_dim, kernel_size=kernel_size, n_groups=n_groups,
                    condition_type=condition_type, mamba_version=mamba_version))
        for _ in range(decode_layer):
            decode_modules.append(ConditionalMambaResidualBlock1D(
                    in_channels=hide_dim, out_channels=hide_dim, 
                    cond_dim=cond_dim, kernel_size=kernel_size, n_groups=n_groups,
                    condition_type=condition_type, mamba_version=mamba_version,skip=True)) 
        final_conv = nn.Sequential(
            Conv1dBlock(hide_dim, hide_dim, kernel_size=kernel_size),
            nn.Conv1d(hide_dim, input_dim, 1),
        )
        self.x_embedder = nn.Conv1d(input_dim, hide_dim, 1)
        self.diffusion_step_encoder = diffusion_step_encoder
        self.side_modules = side_modules
        self.decode_modules = decode_modules
        self.mask_ratio = mask_ratio
        self.final_conv=final_conv

        if mask_ratio is not None:
            self.mask_token = nn.Parameter(torch.zeros(1, hide_dim, 1))
            self.mask_ratio = float(mask_ratio)
        else:
            self.mask_token = nn.Parameter(torch.zeros(
                1, hide_dim, 1), requires_grad=False)
            self.mask_ratio = None
    
        logger.info(
            "number of parameters: %e", sum(p.numel() for p in self.parameters())
        )

    def random_masking(self, x, mask_ratio):
        N, D, L = x.shape
        len_keep = int(L * (1 - mask_ratio))
        noise = torch.rand(N, L, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)  # (N, L)
        ids_restore = torch.argsort(ids_shuffle, dim=1)  # (N, L)
        ids_keep = ids_shuffle[:, :len_keep]  # (N, len_keep)

        # 在长度维度 (dim=2) 上 gather
        # 需要索引形状 (N, D, len_keep)
        index = ids_keep.unsqueeze(1).expand(-1, D, -1)  # (N, D, len_keep)
        x_masked = torch.gather(x, dim=2, index=index)  # (N, D, len_keep)

        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)  # (N, L)

        return x_masked, mask, ids_restore, ids_keep
    

    def forward_side_interpolater(self, x, c, mask, ids_restore):
        # 生成 mask_tokens：形状 [N, D, L - len_keep]
        N, D, L = x.shape
        mask_tokens = self.mask_token.repeat(
            x.shape[0], 1, ids_restore.shape[1] - x.shape[2])   
        # self.mask_token 原为 [1,D,1] → 扩展为 [N, D, L - len_keep]

        # 沿序列维度 (dim=2) 拼接
        print(x.shape)
        print(mask_tokens.shape)
        x_ = torch.cat([x, mask_tokens], dim=2)          # [N, D, L]

        # 沿 dim=2 进行 unshuffle
        index = ids_restore.unsqueeze(1).expand(-1, D, -1)   # [N, D, L]
        x = torch.gather(x_, dim=2, index=index)             # [N, D, L]

        x_before = x
        for sideblock in self.side_modules:
            x = sideblock(x, c)   

        # mask 形状调整：从 [N, L] 变为 [N, 1, L] 以匹配 [N, D, L]
        mask = mask.unsqueeze(1)                  # [N, 1, L]
        x = x * mask + (1 - mask) * x_before
        return x
    def forward(self, 
            sample: torch.Tensor, 
            timestep_1: Union[torch.Tensor, float, int], 
            timestep_2: Union[torch.Tensor, float, int], 
            global_cond=None,enable_mask=False, **kwargs):
        """
        x: (B,T,input_dim)
        timestep: (B,) or int, diffusion step
        local_cond: (B,T,local_cond_dim)
        global_cond: (B,global_cond_dim)
        output: (B,T,input_dim)
        """
        
        sample = einops.rearrange(sample, 'b h t -> b t h')
        
        # 1. time
        timesteps_1 = timestep_1
        timesteps_2 = timestep_2
        if not torch.is_tensor(timesteps_1):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps_1 as tensors if you can
            timesteps_1 = torch.tensor([timesteps_1], dtype=torch.float, device=sample.device)
            timesteps_2 = torch.tensor([timesteps_2], dtype=torch.float, device=sample.device)
        elif torch.is_tensor(timesteps_1) and len(timesteps_1.shape) == 0:
            timesteps_1 = timesteps_1[None].to(sample.device)
            timesteps_2 = timesteps_2[None].to(sample.device)
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps_1 = timesteps_1.expand(sample.shape[0])
        timesteps_2 = timesteps_2.expand(sample.shape[0])   

        timestep_embed_1 = self.diffusion_step_encoder(timesteps_1)  ## [B,time_dim]
        timestep_embed_2 = self.diffusion_step_encoder(timesteps_2)  ## [B,time_dim]
        timestep_embed = timestep_embed_1 + timestep_embed_2  ## [B,time_dim]
        if global_cond is not None:
            if self.condition_type == 'cross_attention':
                timestep_embed = timestep_embed.unsqueeze(1).expand(-1, global_cond.shape[1], -1)
            global_cond = global_cond.view(global_cond.size(0), -1)
            global_feature = torch.cat([timestep_embed, global_cond], axis=-1)


        masked_stage = False
        x = self.x_embedder(sample)
        input_skip = x
        if self.mask_ratio is not None and enable_mask:
            rand_mask_ratio = torch.rand(1, device=x.device)  # noise in [0, 1]
            rand_mask_ratio = rand_mask_ratio * 0.2 + self.mask_ratio # mask_ratio, mask_ratio + 0.2 
            x, mask, ids_restore, ids_keep = self.random_masking(
                input_skip, rand_mask_ratio)
            masked_stage = True 
        
        skips = []
        for module in self.en_inmodules:
            if masked_stage:
                x = module(x, global_feature)
            else:
                x = module(x, global_feature)
            skips.append(x)
        for module in self.en_outmodules:
            if masked_stage:
                x = module(x, global_feature, skip=skips.pop())
            else:
                x = module(x, global_feature, skip=skips.pop())

        if self.mask_ratio is not None and enable_mask:
            x = self.forward_side_interpolater(x, global_feature, mask, ids_restore)
            masked_stage = False
        for decode_module in self.decode_modules:
            x= decode_module(x, global_feature, input_skip)

        x = self.final_conv(x)
        x = einops.rearrange(x, 'b t h -> b h t')

        return x
    



if __name__ == "__main__":

    net_mambaunet = ConditionalMambaUnet1D(
        input_dim=24,
        global_cond_dim=256,
        diffusion_step_embed_dim=128,
        hide_dim=256,
        kernel_size=5,
        n_groups=8,
        condition_type='film',
        use_down_condition=True,
        use_mid_condition=True,
        use_up_condition=True,
        mamba_version= 'mambavision_v1',
        decode_layer=3,
        side_layer=1,
        mask_ratio=0.5,
        depth=8,
        horizon=2,
    ).cuda()
    input = torch.randn(256, 8, 24).cuda()
    cond = torch.randn(256, 2, 256).cuda()
    timestep1 = torch.randn(256).cuda()
    timestep2 = torch.randn(256).cuda()
    output1 = net_mambaunet(sample=input, timestep_1=timestep1, timestep_2=timestep2, global_cond=cond)
    output2 = net_mambaunet(sample=input, timestep_1=timestep1, timestep_2=timestep2, global_cond=cond, enable_mask=True)
    print(f'number of parameters in MambaUNet: {sum(p.numel() for p in net_mambaunet.parameters())} ')

