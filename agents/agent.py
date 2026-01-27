import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from ppo_models import ActorCritic


class PPOAgent:
    def __init__(
        self,
        state_dim,
        n_assets,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_eps=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        update_epochs=10,
        batch_size=256,
        device=None
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.net = ActorCritic(state_dim, n_assets).to(self.device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.update_epochs = update_epochs
        self.batch_size = batch_size

    @torch.no_grad()
    def select_action(self, state):
        state = torch.FloatTensor(state).to(self.device)
        action, logp, value = self.net.act(state)
        return (
            action.cpu().numpy(),
            logp.cpu().numpy(),
            value.cpu().numpy()
        )

    def compute_gae(self, rewards, values, dones, last_value):
        advantages = []
        gae = 0
        values = values + [last_value]

        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t+1] * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)

        returns = [adv + val for adv, val in zip(advantages, values[:-1])]
        return advantages, returns

    def update(self, buffer):
        states, actions, logps_old, returns, advs = buffer

        advs = (advs - advs.mean()) / (advs.std() + 1e-8)

        dataset = TensorDataset(
            states, actions, logps_old, returns, advs
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        for _ in range(self.update_epochs):
            for s, a, logp_old, ret, adv in loader:
                logp, entropy, value = self.net.evaluate(s, a)

                ratio = torch.exp(logp - logp_old)
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv

                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = nn.MSELoss()(value.squeeze(-1), ret)
                entropy_loss = -entropy.mean()

                loss = (
                    policy_loss
                    + self.vf_coef * value_loss
                    + self.ent_coef * entropy_loss
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                self.optimizer.step()