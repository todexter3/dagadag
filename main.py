import argparse
import os
import yaml
import torch
from exp.exp_ppo import Exp_PPO

def main():
    parser = argparse.ArgumentParser(description='PPO for ZZ500 Timing')
    
    # 1. 基础路径配置
    parser.add_argument('--data_path', type=str, default='./data/zz500_daily.csv', help='数据文件路径')
    parser.add_argument('--save_dir', type=str, default='./experiments/', help='实验结果保存根目录')
    parser.add_argument('--exp_name', type=str, default='timing_v1', help='实验名称')
    
    # 2. 从 yaml 加载默认参数
    config_path = 'config.yaml'
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"未找到配置文件: {config_path}")
        
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 3. 策略与环境参数 (允许命令行覆盖)
    parser.add_argument('--min_pos', type=float, default=config.get('min_pos', 0.0))
    parser.add_argument('--max_pos', type=float, default=config.get('max_pos', 1.0))
    parser.add_argument('--commission', type=float, default=config.get('commission', 0.0003))
    parser.add_argument('--fix_weight', type=float, default=config.get('fix_weight', 1.0))

    # 4. 训练与模型参数 (确保与上一版提供的 PPOAgent 匹配)
    parser.add_argument('--lr', type=float, default=config.get('lr', 0.0001))
    parser.add_argument('--gamma', type=float, default=config.get('gamma', 0.99))
    parser.add_argument('--gae_lambda', type=float, default=config.get('gae_lambda', 0.95))
    parser.add_argument('--clip_eps', type=float, default=config.get('clip_eps', 0.2))
    parser.add_argument('--batch_size', type=int, default=config.get('batch_size', 64))
    parser.add_argument('--n_epochs', type=int, default=config.get('n_epochs', 10))
    parser.add_argument('--buffer_size', type=int, default=config.get('buffer_size', 1024))
    parser.add_argument('--hidden_dim', type=int, default=config.get('hidden_dim', 128))
    
    args = parser.parse_args()
    
    # 5. 设备配置
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 6. 创建实验文件夹结构
    # 最终结果会存放在 ./experiments/timing_v1/ 目录下
    args.res_path = os.path.join(args.save_dir, args.exp_name)
    args.checkpoints = os.path.join(args.res_path, 'checkpoints')
    args.results_dir = os.path.join(args.res_path, 'results')
    
    os.makedirs(args.checkpoints, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    print("="*50)
    print(f"Starting experiment: {args.exp_name}")
    print(f"Device: {args.device}")
    print(f"Data: {args.data_path}")
    print(f"Baseline Weight: {args.fix_weight}")
    print("="*50)
    
    # 7. 启动实验
    try:
        exp = Exp_PPO(args)
        
        print("\n>>> Phase 1: Training")
        exp.train()
        
        print("\n>>> Phase 2: Testing & Evaluation")
        exp.test()
        
        print(f"\nExperiment finished. Results saved in: {args.res_path}")
        
    except Exception as e:
        print(f"An error occurred during experiment: {e}")
        raise e

if __name__ == "__main__":
    main()
