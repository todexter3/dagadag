import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from envs.trading_envs import TimingEnv
from agents.agent import PPOAgent
from data_loader.data_loader import load_zz500_data

class Exp_PPO:
    def __init__(self, args):
        self.args = args
        self.data = load_zz500_data(args.data_path)
        
        split = int(len(self.data) * 0.8)
        self.train_env = TimingEnv(self.data.iloc[:split], args)
        self.test_env = TimingEnv(self.data.iloc[split:], args)
        
        sample_obs = self.train_env.reset()
        self.agent = PPOAgent(len(sample_obs), args)

    def train(self):
        for ep in range(self.args.n_epochs):
            state = self.train_env.reset()
            ep_reward = 0
            done = False
            while not done:
                action, logp, val = self.agent.select_action(state)
                next_state, reward, done, info = self.train_env.step(action)
                self.agent.store_transition((state, action, reward, done, logp,val))
                state = next_state
                ep_reward += reward
                
                if len(self.agent.buffer) >= self.args.buffer_size:
                    self.agent.update()
            
            print(f"Epoch: {ep} | Train Reward: {ep_reward:.4f}")
            torch.save(self.agent.net.state_dict(), os.path.join(self.args.checkpoints, 'latest.pth'))

    def test(self):
        self.agent.net.load_state_dict(torch.load(os.path.join(self.args.checkpoints, 'latest.pth')))
        state = self.test_env.reset()
        done = False
        
        results = []
        while not done:
            action, _, _ = self.agent.select_action(state, deterministic=True)
            next_state, reward, done, info = self.test_env.step(action)
            results.append(info)
            state = next_state
        
        res_df = pd.DataFrame(results)
        res_df['excess_ret'] = res_df['agent_ret'] - res_df['bench_ret']
        res_df['cum_agent'] = (1 + res_df['agent_ret']).cumprod()
        res_df['cum_bench'] = (1 + res_df['bench_ret']).cumprod()
        res_df['cum_excess'] = res_df['excess_ret'].cumsum()
        
        res_df.to_csv(os.path.join(self.args.res_path, 'test_results.csv'))
        print(f"Test Complete. Final Excess PNL: {res_df['cum_excess'].iloc[-1]:.4f}")
        

        plt.figure(figsize=(12, 15))

        plt.subplot(3, 1, 1)
        plt.plot(res_df['cum_agent'], label='Agent Strategy', color='red')
        plt.plot(res_df['cum_bench'], label='Benchmark (Buy & Hold)', color='blue', linestyle='--')
        plt.title('Cumulative Returns Comparison')
        plt.legend()
        plt.grid(True)

        plt.subplot(3, 1, 2)
        plt.fill_between(res_df.index, res_df['cum_excess'], 0, 
                         where=(res_df['cum_excess'] >= 0), facecolor='green', alpha=0.3)
        plt.fill_between(res_df.index, res_df['cum_excess'], 0, 
                         where=(res_df['cum_excess'] < 0), facecolor='red', alpha=0.3)
        plt.plot(res_df['cum_excess'], label='Cumulative Excess Return (Alpha)', color='black')
        plt.title('Alpha Curve')
        plt.legend()
        plt.grid(True)

        plt.subplot(3, 1, 3)
        plt.plot(res_df['pos'], label='Agent Position (Weight)', color='orange', linewidth=1)
        plt.ylim(-0.1, 1.1) 
        plt.title('Portfolio Position Over Time')
        plt.ylabel('Weight')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(self.args.res_path, 'performance_analysis.png'))
        plt.close()
