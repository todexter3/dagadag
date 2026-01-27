import os
import torch
import pandas as pd
from data_loader.data_loader import load_zz500_data
from envs.trading_envs import TimingEnv
from agents.agent import PPOAgent

class Exp_PPO:
    def __init__(self, args):
        self.args = args
        self.data = load_zz500_data(args.data_path)
        
        # 简单划分训练/测试集
        split = int(len(self.data) * 0.8)
        self.train_data = self.data.iloc[:split]
        self.test_data = self.data.iloc[split:]
        
        self.train_env = TimingEnv(self.train_data, args)
        self.test_env = TimingEnv(self.test_data, args)
        
        # 初始化 Agent
        self.agent = PPOAgent(args)

    def train(self):
        best_reward = -float('inf')
        for ep in range(100): # 示例迭代次数
            state = self.train_env.reset()
            ep_reward = 0
            
            for _ in range(len(self.train_data)-1):
                action, logp = self.agent.select_action(state)
                next_state, reward, done, info = self.train_env.step(action)
                
                self.agent.store_transition(state, action, reward, next_state, done, logp)
                state = next_state
                ep_reward += reward
                
                if len(self.agent.buffer) >= self.args.buffer_size:
                    self.agent.update()
                if done: break

            print(f"Epoch: {ep} | Reward: {ep_reward:.4f}")
            
            # 保存最佳模型
            if ep_reward > best_reward:
                best_reward = ep_reward
                torch.save(self.agent.network.state_dict(), os.path.join(self.args.checkpoints, 'best_model.pth'))

    def test(self):
        self.agent.network.load_state_dict(torch.load(os.path.join(self.args.checkpoints, 'best_model.pth')))
        self.agent.network.eval()
        # 执行测试集回测逻辑并记录结果到 results 文件夹