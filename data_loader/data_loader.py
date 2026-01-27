import pandas as pd
import numpy as np

def load_zz500_data(file_path):
    # 假设 CSV 包含: date, close, high, low, open, volume
    df = pd.read_csv(file_path, parse_dates=['date'])
    df = df.sort_values('date')
    
    # 基础特征
    df['ret'] = np.log(df['close'] / df['close'].shift(1))
    df['ma5'] = df['close'].rolling(5).mean() / df['close']
    df['ma20'] = df['close'].rolling(20).mean() / df['close']
    df['vol_std'] = df['ret'].rolling(20).std()
    df['rsi'] = calculate_rsi(df['close'], 14)
    
    return df.dropna().reset_index(drop=True)

def calculate_rsi(series, period):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))