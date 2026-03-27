"""
NILM Seq2Point 训练快速示例。
本脚本演示基础流程。
"""

import os
import sys
import torch
import numpy as np

# 将项目根目录加入 sys.path，便于直接运行脚本
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import Seq2Point, get_model
from utils.data_loader import NILMDataLoader, create_dataloaders
from utils.config_utils import load_config, get_appliance_params
from utils.metrics import compute_metrics, print_metrics


def quick_start_example():
    """快速开始示例。"""
    
    print("=" * 60)
    print("NILM Seq2Point Quick Start Example")
    print("=" * 60)
    
    # 1. 加载配置
    print("\n1. Loading configuration...")
    config_path = 'configs/config.yaml'
    
    if not os.path.exists(config_path):
        print(f"Error: Config file not found at {config_path}")
        print("Please make sure you're running this from the project root directory.")
        return
    
    config = load_config(config_path)
    print(f"   Dataset: {config['data']['dataset']}")
    print(f"   Appliance: {config['data']['appliance']}")
    print(f"   Data path: {config['data']['data_path']}")
    
    # 2. 检查数据文件
    print("\n2. Checking data file...")
    if os.path.exists(config['data']['data_path']):
        print(f"   ✓ Data file exists: {config['data']['data_path']}")
    else:
        print(f"   ✗ Data file not found: {config['data']['data_path']}")
        print("\n   Please prepare your data file in HDF5 format.")
        print("   You can use the preprocessing script:")
        print("   python scripts/preprocess_data.py --input <nilmtk_data> --output data/redd.h5")
        return
    
    # 3. 获取电器参数
    print("\n3. Getting appliance parameters...")
    try:
        appliance_params = get_appliance_params(
            config['data']['appliance'],
            'configs/appliance_params.yaml'
        )
        print(f"   Max power: {appliance_params['max_power']} W")
        print(f"   Threshold: {appliance_params['threshold']} W")
    except Exception as e:
        print(f"   Error: {e}")
        return
    
    # 4. 创建模型
    print("\n4. Creating Seq2Point model...")
    try:
        model = get_model(
            model_name='seq2point',
            input_size=599,
            dropout_rate=0.1
        )
        
        # 统计参数量
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"   ✓ Model created successfully")
        print(f"   Parameters: {num_params:,}")
        
        # 测试前向传播
        dummy_input = torch.randn(8, 599)
        with torch.no_grad():
            output = model(dummy_input)
        print(f"   Input shape: {dummy_input.shape}")
        print(f"   Output shape: {output.shape}")
        
    except Exception as e:
        print(f"   Error creating model: {e}")
        return
    
    # 5. 准备开始训练
    print("\n5. Setup complete! Ready to train.")
    print("\n" + "=" * 60)
    print("Next steps:")
    print("=" * 60)
    print("\nTo start training, run:")
    print("   python train.py --config configs/config.yaml")
    print("\nTo evaluate a trained model, run:")
    print("   python evaluate.py --checkpoint checkpoints/model.pth --save-predictions --plot")
    print("\nTo run inference on new data, run:")
    print("   python inference.py --checkpoint checkpoints/model.pth --input data/test.npy")
    print("\n" + "=" * 60)
    print("\nFor more information, see README.md")
    print("=" * 60)


def test_data_loading():
    """测试数据加载功能。"""
    
    print("\n" + "=" * 60)
    print("Testing Data Loading")
    print("=" * 60)
    
    config = load_config('configs/config.yaml')
    
    if not os.path.exists(config['data']['data_path']):
        print("Data file not found. Skipping data loading test.")
        return
    
    try:
        print("\nCreating data loader...")
        appliance_params = get_appliance_params(
            config['data']['appliance'],
            'configs/appliance_params.yaml'
        )
        
        data_loader = NILMDataLoader(
            data_path=config['data']['data_path'],
            appliance=config['data']['appliance'],
            window_size=config['data']['window_size'],
            window_stride=config['data']['window_stride'],
            max_power=appliance_params['max_power']
        )
        
        print("Loading data from first building...")
        X, y = data_loader.load_data(
            buildings=[config['data']['train_buildings'][0]],
            normalize=True
        )
        
        print(f"\n✓ Data loaded successfully!")
        print(f"  Input shape: {X.shape}")
        print(f"  Target shape: {y.shape}")
        print(f"  Input range: [{X.min():.2f}, {X.max():.2f}]")
        print(f"  Target range: [{y.min():.2f}, {y.max():.2f}]")
        
    except Exception as e:
        print(f"\n✗ Error loading data: {e}")
        print("\nPlease check:")
        print("  1. Data file format is correct")
        print("  2. Building IDs exist in the dataset")
        print("  3. Appliance name is spelled correctly")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Quick start example')
    parser.add_argument(
        '--test-data',
        action='store_true',
        help='Test data loading functionality'
    )
    
    args = parser.parse_args()
    
    if args.test_data:
        test_data_loading()
    else:
        quick_start_example()
