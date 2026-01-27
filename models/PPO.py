import torch
import torch.nn as nn
from torch.distributions import Dirichlet


class ActorCritic(nn.Module):
    """
    PPO Actor-Critic for asset allocation
    action: portfolio weights (sum to 1)
    """
    def __init__(self, state_dim, n_assets, hidden_dim=256):
        super().__init__()

        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Actor: concentration parameters for Dirichlet
        self.alpha_head = nn.Sequential(
            nn.Linear(hidden_dim, n_assets),
            nn.Softplus()   # ensure alpha > 0
        )

        # Critic
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        z = self.feature(state)
        alpha = self.alpha_head(z) + 1e-3
        value = self.value_head(z)
        return alpha, value

    def act(self, state, deterministic=False):
        alpha, value = self.forward(state)
        dist = Dirichlet(alpha)

        if deterministic:
            action = alpha / alpha.sum(dim=-1, keepdim=True)
        else:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        return action, log_prob, value

    def evaluate(self, state, action):
        alpha, value = self.forward(state)
        dist = Dirichlet(alpha)
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return log_prob, entropy, value