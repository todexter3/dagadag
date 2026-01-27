import numpy as np

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
        # 返回当前行的特征 + 当前持仓
        row = self.df.iloc[self.current_step]
        # 假设特征在 load_data 时已处理好
        obs = row.drop(['date', 'close']).values 
        return np.append(obs, self.pos).astype(np.float32)

    def step(self, action):
        # 将网络输出映射到 [min_pos, max_pos]
        new_pos = np.clip(action, self.args.min_pos, self.args.max_pos)
        
        prev_row = self.df.iloc[self.current_step]
        curr_row = self.df.iloc[self.current_step + 1]
        
        # 你的公式实现:
        # Gross = (exp(ret_t)-1) * close_{t-1} * pos_{t-1}
        gross = (np.exp(curr_row['ret']) - 1) * prev_row['close'] * self.pos
        # comm = abs(pos_t - pos_{t-1}) * close_t * commission
        comm = abs(new_pos - self.pos) * curr_row['close'] * self.args.commission
        
        reward = (gross - comm) / prev_row['close'] # 收益率形式的 reward
        
        self.pos = new_pos
        self.current_step += 1
        done = self.current_step >= len(self.df) - 2
        
        return self._get_obs(), reward, done, {}