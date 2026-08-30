from typing import Union

import torch
import torch.nn as nn
import numpy as np
import math
import einops
from timm.models.vision_transformer import Mlp
from timm.models.layers import trunc_normal_

class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class RotaryPositionEmbedding(nn.Module):
    def __init__(self, max_len, dim):
        super().__init__()
        self.max_len = max_len
        self.dim = dim
        self.pos_emb = self.generate_rope_embeddings(max_len, dim)
    @staticmethod
    def generate_rope_embeddings(seq_len, output_dim):
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(-1)
        ids = torch.arange(0, output_dim // 2, dtype=torch.float) 
        theta = torch.pow(10000, -2 * ids / output_dim)
        embeddings = position * theta
        embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        return embeddings
    def forward(self, bacth_size, num_heads, device,ids_keep=None):
        pos_emb = self.pos_emb.repeat((bacth_size, num_heads, *([1] * len(self.pos_emb.shape)))).to(device)  # 在bs维度重复，其他维度都是1不重复
        pos_emb = torch.reshape(pos_emb, (bacth_size, num_heads, self.max_len, self.dim))
        if ids_keep is not None:
            batch_indices = torch.arange(pos_emb.size(0)).unsqueeze(1).unsqueeze(1).expand(-1, pos_emb.size(1), ids_keep.size(1))
            head_indices = torch.arange(pos_emb.size(1)).unsqueeze(0).unsqueeze(2).expand(pos_emb.size(0), -1, ids_keep.size(1))
            pos_emb = pos_emb[batch_indices, head_indices, ids_keep.unsqueeze(1).expand(-1, pos_emb.size(1), -1)]
        return pos_emb
    
def apply_rotary_pos_emb(q, k, rope_embeddings):
    cos_pos = rope_embeddings[...,  1::2].repeat_interleave(2, dim=-1) 
    sin_pos = rope_embeddings[..., ::2].repeat_interleave(2, dim=-1) 
    q2 = torch.stack([-q[..., 1::2], q[..., ::2]], dim=-1)
    q2 = q2.reshape(q.shape)
    q = q * cos_pos + q2 * sin_pos
    k2 = torch.stack([-k[..., 1::2], k[..., ::2]], dim=-1)
    k2 = k2.reshape(k.shape)
    k = k * cos_pos + k2 * sin_pos
    return q, k

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size
    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb

class RelativePositionBias1D(nn.Module):
    def __init__(self, window_size, num_heads):
        super().__init__()
        self.window_size = window_size
        self.num_relative_distance = 2 * window_size - 1
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(self.num_relative_distance, num_heads))
        coords = torch.arange(window_size)
        relative_coords = coords[:, None] - coords[None, :]
        relative_coords += window_size - 1
        self.register_buffer("relative_position_index", relative_coords)

        trunc_normal_(self.relative_position_bias_table, std=.02)

    def forward(self):
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size, self.window_size, -1)  # W, W, nH
        return relative_position_bias.permute(2, 0, 1).contiguous()  # nH, W, W


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, 
                 attn_drop=0., proj_drop=0., horizon=None):
        super().__init__()

        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.wq = nn.Sequential(nn.Linear(dim, dim, bias=qkv_bias),
                                nn.LayerNorm(dim)
                                )
        self.wv = nn.Linear(dim, dim, bias=qkv_bias)

        self.wk = nn.Sequential(nn.Linear(dim, dim, bias=qkv_bias),
                                nn.LayerNorm(dim)
                                )
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rpoe_embed = RotaryPositionEmbedding(horizon, head_dim)
        
    def forward(self, x, ids_keep=None):
        B, N, C = x.shape
        q=self.wq(x).reshape(B, self.num_heads, N , C //self.num_heads)
        k=self.wk(x).reshape(B, self.num_heads, N , C //self.num_heads)
        v=self.wv(x).reshape(B, self.num_heads, N , C //self.num_heads)
        rope_embeddings = self.rpoe_embed(B, self.num_heads, x.device, ids_keep)
        q, k = apply_rotary_pos_emb(q, k, rope_embeddings)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x

class MDTBlock(nn.Module):
    """
    A MDT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, skip=False, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(
            hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        def approx_gelu(): return nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size,
                       hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )
        self.skip_linear = nn.Sequential(nn.SiLU(),
                            nn.Linear(hidden_size, hidden_size)) if skip else None

    def forward(self, x, c, skip=None, ids_keep=None):
        if self.skip_linear is not None:
            x = x+self.skip_linear(skip)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(
            c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa), ids_keep=ids_keep)
        x = x + gate_mlp.unsqueeze(
                1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


def get_1d_sincos_pos_embed_from_grid(embed_dim, grid_size):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)
    pos = np.arange(grid_size, dtype=np.float32)
    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product
    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)
    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

class RMDiT(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        horizon: int,
        n_obs_steps:int,
        attn_drop: float,
        depth:int =28,
        mlp_ratio:int=4.0,
        mask_ratio=None,
        decode_layer:int=4,
        side_layer:int=1,
        cond_dim: int = 0,
        n_head: int = 12,
        n_emb: int = 768,
    ):
        super().__init__()
        decode_layer = int(decode_layer)
        self.in_channels = input_dim
        # print(input_dim)
        # print(output_dim)
        self.out_channels = output_dim 
        self.num_heads = n_head
        self.t_embedder = TimestepEmbedder(n_emb)
        self.x_embedder = nn.Linear(input_dim, n_emb)
        self.c_embedder = nn.Linear(cond_dim*n_obs_steps, n_emb)
        half_depth = (depth - decode_layer)//2
        self.half_depth=half_depth
        self.en_inblocks = nn.ModuleList([
            MDTBlock(n_emb, n_head, mlp_ratio=mlp_ratio, horizon=horizon,attn_drop=attn_drop) for i in range(half_depth)
        ])
        self.en_outblocks = nn.ModuleList([
            MDTBlock(n_emb, n_head, mlp_ratio=mlp_ratio, horizon=horizon, skip=True,attn_drop=attn_drop) for i in range(half_depth)
        ])
        self.de_blocks = nn.ModuleList([
            MDTBlock(n_emb, n_head, mlp_ratio=mlp_ratio, horizon=horizon, skip=True,attn_drop=attn_drop) for i in range(decode_layer)
        ])
        self.sideblocks = nn.ModuleList([
            MDTBlock(n_emb, n_head, mlp_ratio=mlp_ratio, horizon=horizon) for _ in range(side_layer)
        ])
        
        self.final_layer = FinalLayer(
            n_emb, self.out_channels)

        if mask_ratio is not None:
            self.mask_token = nn.Parameter(torch.zeros(1, 1, n_emb))
            self.mask_ratio = float(mask_ratio)
            self.decode_layer = int(decode_layer)
        else:
            self.mask_token = nn.Parameter(torch.zeros(
                1, 1, n_emb), requires_grad=False)
            self.mask_ratio = None
            self.decode_layer = int(decode_layer)
        
        print("mask ratio:", self.mask_ratio, "decode_layer:", self.decode_layer)
        print("Diffusion params: %e" % sum(p.numel() for p in self.parameters()))
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize pos_embed by sin-cos embedding:

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.bias, 0)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        for block in self.en_inblocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        for block in self.en_outblocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
            nn.init.constant_(block.skip_linear[-1].weight, 0)
            nn.init.constant_(block.skip_linear[-1].bias, 0)
        for block in self.de_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
            nn.init.constant_(block.skip_linear[-1].weight, 0)
            nn.init.constant_(block.skip_linear[-1].bias, 0)

        for block in self.sideblocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

        if self.mask_ratio is not None:
            torch.nn.init.normal_(self.mask_token, std=.02)
    def get_optim_groups(self, weight_decay: float=1e-3):
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = "%s.%s" % (mn, pn) if mn else pn  # full param name
                if pn.endswith("bias"):
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    no_decay.add(fpn)
                elif pn.endswith("bias_table"):
                    no_decay.add(fpn)
        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add("mask_token")
        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert (
            len(inter_params) == 0
        ), "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert (
            len(param_dict.keys() - union_params) == 0
        ), "parameters %s were not separated into either decay/no_decay set!" % (
            str(param_dict.keys() - union_params),
        )
        # create the pytorch optimizer object
        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(list(decay))],
                "weight_decay": weight_decay,
            },
            {
                "params": [param_dict[pn] for pn in sorted(list(no_decay))],
                "weight_decay": 0.0,
            },
        ]
        return optim_groups
    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int((L * (1 - mask_ratio)))
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(
            x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return x_masked, mask, ids_restore, ids_keep

    def forward_side_interpolater(self, x, c, mask, ids_restore):
        mask_tokens = self.mask_token.repeat(
            x.shape[0], ids_restore.shape[1] - x.shape[1], 1)
        x_ = torch.cat([x, mask_tokens], dim=1)
        x = torch.gather(
            x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x_before = x
        for sideblock in self.sideblocks:
            x = sideblock(x, c, ids_keep=None)
        mask = mask.unsqueeze(dim=-1)
        x = x*mask + (1-mask)*x_before
        return x
    
    def get_embedding(self, sample: torch.Tensor, 
            timestep1: Union[torch.Tensor, float, int],
            timestep2: Union[torch.Tensor, float], 
            global_cond=None, enable_mask=False, **kwargs):
        B=sample.shape[0]
        if global_cond is not None:
            cond=global_cond.view(B,-1)
        timesteps1 = timestep1
        timesteps2 = timestep2
        if not torch.is_tensor(timesteps1):
            timesteps1 = torch.tensor([timesteps1], dtype=torch.float, device=sample.device)
            timesteps2 = torch.tensor([timesteps2], dtype=torch.float, device=sample.device)
        elif torch.is_tensor(timesteps1) and len(timesteps1.shape) == 0:
            timesteps1 = timesteps1[None].to(sample.device)
            timesteps2 = timesteps2[None].to(sample.device)
        timesteps1 = timesteps1.expand(sample.shape[0])
        timesteps2 = timesteps2.expand(sample.shape[0])
        t = self.t_embedder(timesteps1)        # (N, D)
        r = self.t_embedder(timesteps2)       # (N, D)
        c = self.c_embedder(cond)             # (N, D)
        return t, r , c
    
    def forward(self, sample: torch.Tensor, 
            timestep1: Union[torch.Tensor, float, int],
            timestep2: Union[torch.Tensor, float], 
            global_cond=None, enable_mask=False, **kwargs):
        B=sample.shape[0]
        if global_cond is not None:
            cond=global_cond.view(B,-1)
        timesteps1 = timestep1
        timesteps2 = timestep2
        if not torch.is_tensor(timesteps1):
            timesteps1 = torch.tensor([timesteps1], dtype=torch.float, device=sample.device)
            timesteps2 = torch.tensor([timesteps2], dtype=torch.float, device=sample.device)
        elif torch.is_tensor(timesteps1) and len(timesteps1.shape) == 0:
            timesteps1 = timesteps1[None].to(sample.device)
            timesteps2 = timesteps2[None].to(sample.device)
        timesteps1 = timesteps1.expand(sample.shape[0])
        timesteps2 = timesteps2.expand(sample.shape[0])
        t = self.t_embedder(timesteps1)        # (N, D)
        r = self.t_embedder(timesteps2)       # (N, D)
        x = self.x_embedder(sample)           # (N, T, D), where
        c = self.c_embedder(cond)             # (N, D)
        c = t + r + c  
        input_skip = x
        masked_stage = False
        skips = []
        down_latents = []
        if self.mask_ratio is not None and enable_mask:
            rand_mask_ratio = torch.rand(1, device=x.device)  # noise in [0, 1]
            rand_mask_ratio = rand_mask_ratio * 0.2 + self.mask_ratio # mask_ratio, mask_ratio + 0.2 
            x, mask, ids_restore, ids_keep = self.random_masking(
                x, rand_mask_ratio)
            masked_stage = True
        
        for block in self.en_inblocks:
            if masked_stage:
                x = block(x, c, ids_keep=ids_keep)
            else:
                x = block(x, c, ids_keep=None)
            skips.append(x)
            down_latents.append(x)
        for block in self.en_outblocks:
            if masked_stage:
                x = block(x, c, skip=skips.pop(), ids_keep=ids_keep)
            else:
                x = block(x, c, skip=skips.pop(), ids_keep=None)
        if self.mask_ratio is not None and enable_mask:
            x = self.forward_side_interpolater(x, c, mask, ids_restore)
            masked_stage = False
        for i in range(len(self.de_blocks)):
            block = self.de_blocks[i]
            this_skip = input_skip
            x = block(x, c, skip=this_skip, ids_keep=None)
        x = self.final_layer(x, c)
        self.down_latents_tensor = torch.stack(down_latents, dim=0).mean(dim=2) #(N,T,D)->(N,D)
        return x
    
if __name__=="__main__":
    model=RMDiT(input_dim=4,output_dim=4, horizon=4,n_obs_steps=2,attn_drop=0.1,
              cond_dim=32, n_head = 4, n_emb = 16, depth=8,decode_layer=4,mask_ratio=0.5)
    print("Diffusion params: %e" % sum(p.numel() for p in model.parameters()))
    x=torch.rand(2,4,4)
    c=torch.rand(2,64)
    t=torch.rand(2)
    d=model(x, t, c, enable_mask=True) 
    print(d.shape)