import argparse
import os
import yaml
import torch
from exp.exp_ppo import Exp_PPO

def main():
    parser = argparse.ArgumentParser(description='PPO for ZZ500 Timing')
    
    # 路径与环境配置
    parser.add_argument('--data_path', type=str, default='./data/zz500_daily.csv')
    parser.add_argument('--save_dir', type=str, default='./experiments/')
    parser.add_argument('--exp_name', type=str, default='timing_v1')
    
    # 从 yaml 加载默认参数
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # 允许通过命令行覆盖 yaml 中的参数
    parser.add_argument('--lr', type=float, default=config['lr'])
    parser.add_argument('--batch_size', type=int, default=config['batch_size'])
    parser.add_argument('--min_pos', type=float, default=config['min_pos'])
    parser.add_argument('--max_pos', type=float, default=config['max_pos'])
    
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 创建实验文件夹结构
    args.res_path = os.path.join(args.save_dir, args.exp_name)
    args.checkpoints = os.path.join(args.res_path, 'checkpoints')
    os.makedirs(args.checkpoints, exist_ok=True)
    os.makedirs(os.path.join(args.res_path, 'results'), exist_ok=True)

    print(f"Starting experiment: {args.exp_name} on {args.device}")
    
    # 启动实验
    exp = Exp_PPO(args)
    exp.train()
    exp.test()

if __name__ == "__main__":
    main()