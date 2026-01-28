import pandas as pd
import numpy as np

def load_zz500_data(file_path):
    # 1. 读取 CSV，指定日期列名为 'time'，并处理千分位逗号
    # thousands=',' 可以直接处理 "8,447.73" 这种带逗号的字符串
    df = pd.read_csv(file_path, parse_dates=['time'], thousands=',')
    
    # 统一列名映射（将 time 转为代码习惯的 date）
    df = df.rename(columns={'time': 'date'})
    df = df.sort_values('date')

    # 2. 预处理：清洗 volume 列 (19.49B -> 19.49 * 1e9)
    def convert_volume(v):
        if not isinstance(v, str): return v
        v = v.upper()
        if 'B' in v:
            return float(v.replace('B', '')) * 1e9
        elif 'M' in v:
            return float(v.replace('M', '')) * 1e6
        elif 'K' in v:
            return float(v.replace('K', '')) * 1e3
        return float(v)

    df['volume'] = df['volume'].apply(convert_volume)

    # 3. 预处理：清洗 changes 列 ("-0.69%" -> -0.0069)
    if 'changes' in df.columns:
        df['changes'] = df['changes'].str.replace('%', '').astype(float) / 100.0

    # 4. 收益率计算 (对数收益率)
    df['ret'] = np.log(df['close'] / df['close'].shift(1))
    
    # 5. 技术指标特征
    # MA5 比例
    df['ma5_ratio'] = df['close'].rolling(5).mean() / df['close']
    
    # 成交量比例 (处理可能为 0 的情况防止报错)
    df['vol_ratio'] = df['volume'] / (df['volume'].rolling(20).mean() + 1e-9)
    
    # 6. 处理涨跌幅 (对应你 CSV 里的 changes)
    df['rise_fall_norm'] = df['changes']
        
    return df.dropna().reset_index(drop=True)
