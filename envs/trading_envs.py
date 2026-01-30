import numpy as np
import collections

class TimingEnv:
    def __init__(self, df, args):
        self.df = df.reset_index(drop=True)
        self.args = args

        self.max_step = len(self.df) - 1
        self.current_step = 0

        self.window_size=args.window_size
        self.ret_history = collections.deque(maxlen=self.window_size)

        self.pos = args.max_pos       # 当前真实仓位
        self.prev_pos = args.max_pos  # 上一期仓位（用于算 cost）

    def reset(self):
        self.current_step = 0
        self.pos = self.args.max_pos
        self.prev_pos = self.args.max_pos

        self.ret_history.clear()
        init_rets = self.df.iloc[max(0, self.current_step-self.window_size):self.current_step]['ret'].tolist()
        self.ret_history.extend(init_rets if init_rets else [0.0])

        return self._get_obs()
    
    def _get_vol(self):
        # 计算当前窗口的年化波动率
        if len(self.ret_history) < 2:
            return 0.0
        return np.std(self.ret_history) * np.sqrt(252)

    def _get_obs(self):
        row = self.df.iloc[self.current_step]

        feats = row.drop(
            ['date', 'close', 'open', 'high', 'low', 'volume']
        ).values.astype(np.float32)

        # state:feature, pos
        obs = np.concatenate([feats, np.array([self.pos], dtype=np.float32)])
        return obs

    def step(self, action):
        """
        action ∈ [0, 1] → new_pos ∈ [min_pos, max_pos]
        """
        new_pos = float(np.clip(action, self.args.min_pos, self.args.max_pos))

        #  取下一期收益 
        next_row = self.df.iloc[self.current_step + 1]
        daily_ret = np.exp(next_row['ret']) - 1.0

        #历史窗口
        self.ret_history.append(next_row['ret'])

        # 收益 & 成本 
        pnl = new_pos * daily_ret

        bench_pnl = self.args.fix_weight * daily_ret

        excess_ret = pnl - bench_pnl

        cost = self.args.commission * abs(new_pos - self.pos)

        #  risk
        current_vol = self._get_vol()
        risk_free_rate = 0.0 
        risk_penalty = 0.5 * self.args.risk_beta * (new_pos**2) * (current_vol**2)



        reward = pnl - cost - bench_pnl * 0.4 - risk_penalty

        if daily_ret > 0.02:
            reward *= 1.5

        # info
        info = {
            'agent_ret': pnl,
            'bench_ret': daily_ret * self.args.fix_weight,
            'pos': new_pos
        }

        #  状态更新 
        self.prev_pos = self.pos
        self.pos = new_pos
        self.current_step += 1

        done = self.current_step >= self.max_step - 1

        return self._get_obs(), reward, done, info
    
