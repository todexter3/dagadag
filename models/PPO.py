import torch
import torch.nn as nn
from torch.distributions import Beta
import torch.nn.functional as F


class ActorCritic(nn.Module):
    def __init__(self,args, state_dim, hidden_dim=128, n_layers=2):
        super().__init__()

        self.args=args
        
        layers = []
        layers.append(nn.Linear(state_dim, hidden_dim))
        layers.append(nn.ReLU())

        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())

        self.feature = nn.Sequential(*layers)

        # Actor heads
        self.alpha_head = nn.Linear(hidden_dim, 1)
        self.beta_head  = nn.Linear(hidden_dim, 1)

        # Critic head
        self.value_head = nn.Linear(hidden_dim, 1)

        nn.init.zeros_(self.value_head.weight)
        nn.init.zeros_(self.value_head.bias)

    def forward(self, state):
        z = self.feature(state)
        
        alpha = torch.clamp(F.softplus(self.alpha_head(z)) + 1.0, 1.0, self.args.alpha_max)
        beta  = torch.clamp(F.softplus(self.beta_head(z))  + 1.0, 1.0, self.args.beta_max)

        value = self.value_head(z)
        return alpha, beta, value

    def act(self, state, deterministic=False):
        alpha, beta, value = self.forward(state)
        dist = Beta(alpha, beta)

        if deterministic:
            action = alpha / (alpha + beta)
        else:
            action = dist.sample()

        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob, value

    def evaluate(self, state, action):
        alpha, beta, value = self.forward(state)
        dist = Beta(alpha, beta)

        log_prob = dist.log_prob(action).view(action.size(0), -1).sum(dim=-1)
        entropy = dist.entropy().view(action.size(0), -1).sum(dim=-1)

        return log_prob, entropy, value.squeeze(-1)
