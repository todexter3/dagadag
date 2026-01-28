import torch
import torch.nn as nn
from torch.distributions import Beta

class ActorCritic(nn.Module):
    def __init__(self, state_dim, hidden_dim=128):
        super().__init__()
        # 公共特征提取层
        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Actor: 输出 Beta 分布的 alpha 和 beta 参数
        self.alpha_head = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Softplus())
        self.beta_head = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Softplus())
        
        # Critic: 输出状态价值
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        z = self.feature(state)
        # 加 1.0 是为了防止 alpha/beta 过小导致分布过于集中在边界
        alpha = self.alpha_head(z) + 1.0
        beta = self.beta_head(z) + 1.0
        value = self.value_head(z)
        return alpha, beta, value

    def act(self, state, deterministic=False):
        alpha, beta, value = self.forward(state)
        dist = Beta(alpha, beta)
        
        if deterministic:
            # 确定性模式下取众数或均值
            action = alpha / (alpha + beta)
        else:
            action = dist.sample()
            
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob, value

    def evaluate(self, state, action):
        alpha, beta, value = self.forward(state)
        dist = Beta(alpha, beta)
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy, value
