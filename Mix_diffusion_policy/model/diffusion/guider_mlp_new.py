from typing import Union
import logging
import torch
import torch.nn as nn
import einops
import time
from einops.layers.torch import Rearrange
import copy

from Mix_diffusion_policy.model.diffusion.mlps import StateEncoder
from Mix_diffusion_policy.model.diffusion.positional_embedding import SinusoidalPosEmb, TimestepEncoder
logger = logging.getLogger(__name__)



class Guider_mlp(nn.Module):
    # 全连接结构
    def __init__(
            self,               
            cond_dim,
            subgoal_dim,
            n_emb,
            mlp_dims=[1024, 512, 256]
            ):
        super().__init__()

        # ************* condition *************                             
        self.time_emb = TimestepEncoder(n_emb)
        input_dim = n_emb + cond_dim + subgoal_dim  
        last_dim = input_dim
        self.layers = nn.Sequential()
        for i, d in enumerate(mlp_dims):
            self.layers.append(nn.Linear(last_dim, d))
            self.layers.append(nn.ReLU())
            last_dim = d
        
        self.final_layer = nn.Linear(mlp_dims[-1], subgoal_dim)

        logger.info("parameters number of Guider: %e", self.params_num())


    def forward(self, cond, subgoal, timestep):
        """ """
        if not torch.is_tensor(timestep):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            timestep = torch.tensor([timestep], dtype=torch.long, device=cond.device)
        elif torch.is_tensor(timestep) and len(timestep.shape) == 0:
            timestep = timestep[None].to(cond.device)
        timestep = timestep.expand(cond.shape[0])
        
        # ************** encode conditions **************

        timestep_emb = self.time_emb(timestep)
        x = torch.concat((timestep_emb, cond, subgoal), dim=1)

        x = self.layers(x)
        x = self.final_layer(x)
        return x


    def params_num(self):
        return sum(p.numel() for p in self.parameters())

