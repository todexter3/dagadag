import torch
import torch.nn as nn
import torch.optim as optim
from models.PPO import ActorCritic
import numpy as np

class PPOAgent:
    def __init__(self, state_dim, args):
        self.args = args
        self.device = args.device
        self.net = ActorCritic(state_dim).to(self.device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=args.lr)
        self.buffer = []

    def select_action(self, state, deterministic=False):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action, logp, value = self.net.act(state, deterministic)
        return action.item(), logp.item(), value.item()

    def store_transition(self, transition):
        self.buffer.append(transition)

    def update(self):
        # 准备数据
        states = torch.FloatTensor(np.array([t[0] for t in self.buffer])).to(self.device)
        actions = torch.FloatTensor(np.array([t[1] for t in self.buffer])).unsqueeze(-1).to(self.device)
        
        
        rewards = [t[2] for t in self.buffer]
        dones = [t[4] for t in self.buffer]
        old_logps = torch.FloatTensor(np.array([t[5] for t in self.buffer])).to(self.device)

        
        # 计算 GAE 和 Returns
        returns = []
        discounted_sum = 0
        for r, d in zip(reversed(rewards), reversed(dones)):
            if d: discounted_sum = 0
            discounted_sum = r + self.args.gamma * discounted_sum
            returns.insert(0, discounted_sum)
        returns = torch.FloatTensor(returns).to(self.device)
        
        # 评估当前值
        _, _, values = self.net.forward(states)
        values = values.squeeze(-1).detach()
        advantages = returns - values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO 更新
        for _ in range(self.args.n_epochs):
            new_logps, entropy, new_values = self.net.evaluate(states, actions)
            ratio = torch.exp(new_logps - old_logps)
            
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.args.clip_eps, 1 + self.args.clip_eps) * advantages
            
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = nn.MSELoss()(new_values.squeeze(-1), returns)
            loss = policy_loss + 0.5 * value_loss - 0.01 * entropy.mean()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
        self.buffer = [] # 清空 buffer
