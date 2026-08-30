import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import time
from Mix_diffusion_policy.model.diffusion.pointcloud_encoder import PointNetEncoder


class PCDStateSubgoalEncoder(nn.Module):
    def __init__(self, 
                 pcd_dim, 
                 state_dim, 
                 subgoal_dim, 
                 obs_dim,
                 is_use_subgoal=True,
                 is_use_film=True, 
                 hidden_dim=256, 
                 film_dim=128,
                 output_dim=256,
                 type='total',
                 pcd_encoder:PointNetEncoder=None):
        """
        点云状态子目标编码器
        
        Args:
            pcd_dim: 点云特征维度 (self.pcd_dim * self.observation_history_num + pcd_id_dim)
            state_dim: 状态维度
            subgoal_dim: 子目标维度
            hidden_dim: 隐藏层维度
            film_dim: FiLM条件维度
        """
        super().__init__()
        self.pcd_dim = pcd_dim
        self.state_dim = state_dim
        self.subgoal_dim = subgoal_dim
        self.obs_dim = obs_dim
        self.is_use_subgoal = is_use_subgoal
        self.is_use_film = is_use_film
        self.pcd_encoder = pcd_encoder
        self.type = type
        self.low_dim_size = 0
        # 状态和子目标融合编码器
        if type == 'total':
            if is_use_subgoal:
                self.low_dim_size = state_dim*self.obs_dim + subgoal_dim
            else:
                self.low_dim_size = state_dim*self.obs_dim     
        elif type == 'crossattn':
            if is_use_subgoal:
                self.low_dim_size = state_dim + subgoal_dim
            else:
                self.low_dim_size = state_dim
        self.state_subgoal_encoder = nn.Sequential(
            nn.Linear(self.low_dim_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, film_dim * 2),  # 输出gamma和beta
        )
        self.pcd_encoder_net = nn.Sequential(
            nn.Linear(self.pcd_encoder.out_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
            
        # FiLM后的投影层
        self.film_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # FiLM输出层
        self.film_output = nn.Linear(hidden_dim, output_dim)

        # 输出层
        cat_dim = self.pcd_encoder.out_dim + self.low_dim_size
        self.output=nn.Linear(cat_dim, output_dim)
        
    def forward(self, nbatch):
        """
        Args:
            nbatch: 包含pcd, state, subgoal等字段的字典
            
        Returns:
            fused_features: 融合后的特征
        """
        output = None
        if  self.type == 'total':  #时序总编码（B，N）
            B = nbatch['pcd'].shape[0]
            
            # 处理点云数据
            pcd = nbatch['pcd'].transpose(1, 2).reshape(
                (B, -1, self.pcd_dim*self.obs_dim))
            if 'pcd_id' in nbatch:
                pcd = torch.cat((pcd, nbatch['pcd_id']), dim=-1)
            # 获取状态和子目标
            state = nbatch['state'].reshape((B, -1))
            subgoal = nbatch['subgoal'] if 'subgoal' in nbatch else None
            state_subgoal = None
            # 拼接状态和子目标
            if self.is_use_subgoal:
                state_subgoal = torch.cat([state, subgoal], dim=-1)
            else:
                state_subgoal = state
            # 编码点云特征
            pcd_features = self.pcd_encoder(pcd)  # [B, N, hidden_dim]
            pcd_features_FiLM = None  # [B, N, hidden_dim]
            # 编码状态和子目标，生成FiLM参数
            if self.is_use_film:    
                film_params = self.state_subgoal_encoder(state_subgoal)
                gamma, beta = torch.chunk(film_params, 2, dim=-1)
                pcd_features_FiLM = self.pcd_encoder_net(pcd_features)
            
            # 应用FiLM调制
            # gamma和beta需要扩展以匹配点云的空间维度
            if self.is_use_film:
                gamma = gamma.unsqueeze(1)  # [B, 1, film_dim]
                beta = beta.unsqueeze(1)    # [B, 1, film_dim]
            
                # 对点云特征进行调制
                modulated_features = pcd_features_FiLM * (1 + gamma) + beta
            
                # 投影调制后的特征
                fused_features = self.film_projection(modulated_features)
            
                # 全局池化得到全局特征
                global_features = fused_features.mean(dim=1)  # [B, hidden_dim]
            
                # 最终输出
                output = self.film_output(global_features)
            else:
                fused_features = torch.cat([pcd_features, state_subgoal], dim=-1)
                output = self.output(fused_features)
        elif self.type == 'crossattn': #时序编码  （B*T，N）
            B, T, N, F = nbatch['pcd'].shape
            # 处理点云数据
            pcd = nbatch['pcd'].reshape(B * T, N, F)
            if 'pcd_id' in nbatch:
                pcd = torch.cat((pcd, nbatch['pcd_id']), dim=-1)
            # 获取状态和子目标
            B, T, state_dim = nbatch['state'].shape
            state = nbatch['state'].reshape(B * T, state_dim)
            subgoal = nbatch['subgoal'] if 'subgoal' in nbatch else None
            B , N = subgoal.shape
            subgoal=subgoal.unsqueeze(1).repeat(1,T,1).reshape(B*T,N)
            state_subgoal = None
            
            # 拼接状态和子目标
            if self.is_use_subgoal:
                state_subgoal = torch.cat([state, subgoal], dim=-1)
            else:
                state_subgoal = state
            # 编码点云特征
            pcd_features = self.pcd_encoder(pcd)  # [B, N, hidden_dim]
            pcd_features_FiLM = self.pcd_encoder_net(pcd_features)  # [B, N, hidden_dim]

            # 编码状态和子目标，生成FiLM参数
            if self.is_use_film:    
                film_params = self.state_subgoal_encoder(state_subgoal)
                gamma, beta = torch.chunk(film_params, 2, dim=-1)
            
            # 应用FiLM调制
            # gamma和beta需要扩展以匹配点云的空间维度
            if self.is_use_film:
                gamma = gamma.unsqueeze(1)  # [B, 1, film_dim]
                beta = beta.unsqueeze(1)    # [B, 1, film_dim]
            
                # 对点云特征进行调制
                modulated_features = pcd_features_FiLM * (1 + gamma) + beta
            
                # 投影调制后的特征
                fused_features = self.film_projection(modulated_features)
            
                # 全局池化得到全局特征
                global_features = fused_features.mean(dim=1)  # [B, hidden_dim]
            
                # 最终输出
                output = self.film_output(global_features)
            else:
                fused_features = torch.cat([pcd_features, state_subgoal], dim=-1)
                output = self.output(fused_features)
        
        return output
