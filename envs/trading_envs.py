import numpy as np
import pandas as pd

class TimingEnv:
    def __init__(self, df, args):
        self.df = df.reset_index(drop=True)
        self.args = args
        self.current_step = 0
        self.pos = args.max_pos # 初始持仓

    def reset(self):
        self.current_step = 0
        self.pos = self.args.max_pos
        return self._get_obs()

    def _get_obs(self):
        # 观察值：当天特征 + 当前持仓状态
        row = self.df.iloc[self.current_step]
        # 排除非特征列
        feats = row.drop(['date', 'close', 'open', 'high', 'low', 'volume']).values
        obs = np.append(feats, self.pos).astype(np.float32)
        return obs

    def step(self, action):
        # action 是 [0, 1] 连续值，映射到 [min_pos, max_pos]
        new_pos = float(np.clip(action, self.args.min_pos, self.args.max_pos))
        
        prev_row = self.df.iloc[self.current_step]
        curr_row = self.df.iloc[self.current_step + 1]
        
        # 1. 净收益 (Gross Profit)
        # exp(ret) - 1 将对数收益率转为简单收益率
        gross = (np.exp(curr_row['ret']) - 1) * prev_row['close'] * self.pos
        
        # 2. 手续费 (Commission)
        comm = abs(new_pos - self.pos) * curr_row['close'] * self.args.commission
        
        # Reward 归一化：转为相对于前一天收盘价的百分比收益
        reward = (gross - comm) / prev_row['close']
        
        # 记录模型日收益和基线日收益 (简单收益率 * 权重)
        daily_ret = (np.exp(curr_row['ret']) - 1)
        agent_ret = daily_ret * self.pos
        bench_ret = daily_ret * self.args.fix_weight
        
        info = {
            'agent_ret': agent_ret,
            'bench_ret': bench_ret,
            'pos': self.pos
        }
        
        self.pos = new_pos
        self.current_step += 1
        done = self.current_step >= len(self.df) - 2
        
        return self._get_obs(), reward, done, info
