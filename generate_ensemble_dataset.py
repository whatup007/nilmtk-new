"""
生成第二层训练数据的脚本
使用已训练好的两个模型对训练集进行预测，并融合结果
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models import get_model
from utils.data_loader import NILMDataLoader, create_dataloaders
from utils.config_utils import load_config, get_appliance_params
from utils.logger import setup_logger


def load_checkpoint(checkpoint_path: str, device: torch.device) -> dict:
    """加载检查点"""
    if os.path.exists(checkpoint_path):
        return torch.load(checkpoint_path, map_location=device, weights_only=False)
    else:
        raise FileNotFoundError(f"检查点未找到: {checkpoint_path}")


def generate_predictions(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    description: str = "Predicting"
) -> np.ndarray:
    """
    使用模型生成预测

    Args:
        model: 训练好的模型
        dataloader: 数据加载器
        device: 计算设备
        description: 进度条描述

    Returns:
        预测数组，形状 (N,)
    """
    model.eval()
    all_predictions = []

    with torch.no_grad():
        with tqdm(dataloader, desc=description, leave=False) as pbar:
            for inputs, _ in pbar:
                inputs = inputs.to(device)
                outputs = model(inputs)
                all_predictions.append(outputs.cpu().numpy())

    predictions = np.concatenate(all_predictions)
    return predictions.flatten()  # 确保是1D数组


def visualize_predictions(
    pred1: np.ndarray,
    pred2: np.ndarray,
    pred_fused: np.ndarray,
    targets: np.ndarray,
    model1_name: str = 'Model 1',
    model2_name: str = 'Model 2',
    output_dir: str = 'ensemble_data',
    dataset_name: str = 'Dataset'
):
    """
    可视化两个模型的预测和融合结果对比

    Args:
        pred1: 第一个模型的预测 (N,)
        pred2: 第二个模型的预测 (N,)
        pred_fused: 融合的预测 (N,) 或 (N, 2) for stack
        targets: 目标值 (N,)
        model1_name: 模型1名称
        model2_name: 模型2名称
        output_dir: 输出目录
        dataset_name: 数据集名称
    """
    os.makedirs(output_dir, exist_ok=True)

    # 处理 stack 方法的情况（pred_fused 是 (N, 2) 形状）
    if pred_fused.ndim == 2 and pred_fused.shape[1] == 2:
        # 对于 stack 方法，可视化时使用简单平均
        pred_fused_vis = pred_fused.mean(axis=1)
        fusion_label = 'Stacked (avg for visualization)'
    else:
        # 确保是 1D 数组
        pred_fused_vis = pred_fused.flatten()
        fusion_label = 'Fused'

    # 创建4个子图
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Ensemble Prediction Comparison - {dataset_name}', fontsize=16, fontweight='bold')

    # 1. 三个模型的预测值分布对比
    ax = axes[0, 0]
    ax.hist(pred1, bins=50, alpha=0.5, label=model1_name, color='blue')
    ax.hist(pred2, bins=50, alpha=0.5, label=model2_name, color='green')
    ax.hist(pred_fused_vis, bins=50, alpha=0.5, label=fusion_label, color='red')
    ax.set_xlabel('Predicted Power (W)')
    ax.set_ylabel('Frequency')
    ax.set_title('Prediction Distribution Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. 预测值与目标值的散点图对比
    ax = axes[0, 1]
    ax.scatter(targets, pred1, alpha=0.3, s=10, label=model1_name, color='blue')
    ax.scatter(targets, pred2, alpha=0.3, s=10, label=model2_name, color='green')
    ax.scatter(targets, pred_fused_vis, alpha=0.3, s=10, label=fusion_label, color='red')
    ax.plot([targets.min(), targets.max()], [targets.min(), targets.max()], 'k--', lw=2, label='Perfect')
    ax.set_xlabel('Target Power (W)')
    ax.set_ylabel('Predicted Power (W)')
    ax.set_title('Predictions vs Targets')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. 预测误差分布对比
    ax = axes[1, 0]
    error1 = np.abs(pred1 - targets)
    error2 = np.abs(pred2 - targets)
    error_fused = np.abs(pred_fused_vis - targets)

    ax.hist(error1, bins=50, alpha=0.5, label=f'{model1_name} (MAE={error1.mean():.2f})', color='blue')
    ax.hist(error2, bins=50, alpha=0.5, label=f'{model2_name} (MAE={error2.mean():.2f})', color='green')
    ax.hist(error_fused, bins=50, alpha=0.5, label=f'{fusion_label} (MAE={error_fused.mean():.2f})', color='red')
    ax.set_xlabel('Absolute Error (W)')
    ax.set_ylabel('Frequency')
    ax.set_title('Prediction Error Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. 前1000个样本的预测对比
    ax = axes[1, 1]
    n_samples = min(1000, len(targets))
    x = np.arange(n_samples)
    ax.plot(x, targets[:n_samples], 'k-', label='Target', linewidth=2)
    ax.plot(x, pred1[:n_samples], 'b-', label=model1_name, alpha=0.7)
    ax.plot(x, pred2[:n_samples], 'g-', label=model2_name, alpha=0.7)
    ax.plot(x, pred_fused_vis[:n_samples], 'r-', label=fusion_label, alpha=0.7)
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('Power (W)')
    ax.set_title(f'Time Series Comparison (First {n_samples} samples)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # 保存图表
    filename = f'predictions_comparison_{dataset_name.lower().replace(" ", "_")}.png'
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ 可视化图表已保存: {save_path}")
    plt.close()


def fuse_predictions(
    pred1: np.ndarray,
    pred2: np.ndarray,
    fusion_method: str = 'average',
    window_size: int = None
) -> np.ndarray:
    """
    融合两个预测结果

    Args:
        pred1: 第一个模型的预测 (N,)
        pred2: 第二个模型的预测 (N,)
        fusion_method: 融合方法 ('average', 'weighted_average', 'stack')
        window_size: 如果需要expand融合预测到原始维度时指定

    Returns:
        融合后的预测
    """
    if fusion_method == 'average':
        # 简单平均
        fused = (pred1 + pred2) / 2.0
    elif fusion_method == 'weighted_average':
        # 加权平均（你可以根据模型性能调整权重）
        weight1, weight2 = 0.5, 0.5  # 可以改为 0.6, 0.4 等
        fused = weight1 * pred1 + weight2 * pred2
    elif fusion_method == 'stack':
        # 堆叠：返回两个预测作为特征（形状变为 (N, 2)）
        fused = np.column_stack([pred1, pred2])
    else:
        raise ValueError(f"未知的融合方法: {fusion_method}")

    # 如果指定了window_size，将预测expand到该维度（填充）
    # 这样可以用Seq2Point作为网络3，同时支持原始输入(N, window_size)
    if window_size is not None and fusion_method != 'stack':
        # 将(N,)或(N,1)的融合预测expand到(N, window_size)
        if fused.ndim == 1:
            fused = fused.reshape(-1, 1)
        # 通过复制填充到window_size
        fused = np.repeat(fused, window_size, axis=1)

    return fused


def generate_ensemble_dataset(
    exp30_checkpoint: str,
    exp33_checkpoint: str,
    config_path: str,
    appliance: str,
    output_dir: str = 'ensemble_data',
    fusion_method: str = 'average',
    **kwargs
):
    """
    生成第二层的训练数据

    Args:
        exp30_checkpoint: exp30（Seq2Point）的检查点路径
        exp33_checkpoint: exp33（LSTM）的检查点路径
        config_path: 配置文件路径
        appliance: 电器名称
        output_dir: 输出目录
        fusion_method: 融合方法
    """
    # 创建日志记录器
    os.makedirs(output_dir, exist_ok=True)
    logger = setup_logger('ensemble', output_dir)

    # 加载配置
    config = load_config(config_path)
    config['data']['appliance'] = appliance

    # 设置计算设备
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        logger.info("使用 GPU")
    else:
        device = torch.device('cpu')
        logger.info("使用 CPU")

    # 获取电器参数
    appliance_params = get_appliance_params(
        appliance,
        'configs/appliance_params.yaml'
    )
    max_power = appliance_params['max_power']

    logger.info(f"电器: {appliance}, 最大功率: {max_power}W")

    # 创建数据加载器
    logger.info("加载数据...")
    data_loader = NILMDataLoader(
        data_path=config['data']['data_path'],
        appliance=appliance,
        window_size=config['data']['window_size'],
        window_stride=config['data']['window_stride'],
        max_power=max_power
    )

    dataloaders = create_dataloaders(
        data_loader=data_loader,
        train_buildings=config['data']['train_buildings'],
        test_buildings=config['data']['test_buildings'],
        batch_size=config['training']['batch_size'],
        val_split=config['validation']['val_split'],
        num_workers=config['device']['num_workers'],
        shuffle=False,  # 不打乱顺序，保持与原始数据的对应关系
        pin_memory=False  # ⭐ 禁用 pin_memory 以节省显存，因为我们后续主要进行 Numpy 操作
    )

    # ===== 加载第一个模型（Seq2Point） =====
    logger.info("\n加载 Seq2Point 模型...")
    checkpoint1 = load_checkpoint(exp30_checkpoint, device)
    config1 = checkpoint1['config']

    model1 = get_model(
        model_name='seq2point',
        input_size=config1['model']['input_size'],
        dropout_rate=config1['model'].get('dropout_rate', 0.1)
    ).to(device)
    model1.load_state_dict(checkpoint1['model_state_dict'])
    logger.info("✓ Seq2Point 模型加载成功")

    # ===== 加载第二个模型（LSTM） =====
    logger.info("\n加载 LSTM 模型...")
    checkpoint2 = load_checkpoint(exp33_checkpoint, device)
    config2 = checkpoint2['config']

    model2 = get_model(
        model_name='seq2point_lstm',
        input_size=config2['model']['input_size'],
        dropout_rate=config2['model'].get('dropout_rate', 0.1)
    ).to(device)
    model2.load_state_dict(checkpoint2['model_state_dict'])
    logger.info("✓ LSTM 模型加载成功")

    # ===== 对训练集进行预测 =====
    logger.info("\n对训练集进行预测...")
    print("\n[模型1: Seq2Point]")
    train_pred1 = generate_predictions(model1, dataloaders['train'], device, "Seq2Point")

    print("\n[模型2: LSTM]")
    train_pred2 = generate_predictions(model2, dataloaders['train'], device, "LSTM")

    # 对验证集进行预测（用于验证）
    logger.info("\n对验证集进行预测...")
    print("\n[模型1: Seq2Point]")
    val_pred1 = generate_predictions(model1, dataloaders['val'], device, "Seq2Point (Val)")

    print("\n[模型2: LSTM]")
    val_pred2 = generate_predictions(model2, dataloaders['val'], device, "LSTM (Val)")

    # ===== 融合预测结果 =====
    logger.info(f"\n融合预测结果 (方法: {fusion_method})...")
    # 对训练和验证集，融合预测保持原样 (N, 1)
    train_pred_fused = fuse_predictions(train_pred1, train_pred2, fusion_method)
    val_pred_fused = fuse_predictions(val_pred1, val_pred2, fusion_method)

    logger.info(f"训练集融合预测形状: {train_pred_fused.shape}")
    logger.info(f"验证集融合预测形状: {val_pred_fused.shape}")

    # ===== 释放 GPU 内存 =====
    # 在获取原始数据之前，释放不再需要的模型以节省显存
    logger.info("\n释放模型以节省显存...")
    del model1
    del model2
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        import gc
        gc.collect()

    logger.info("\n对测试集使用原始数据（不进行融合）...")
    # 对测试集，直接使用原始聚合功率数据，不用网络预测
    # test_pred_fused = None  # 标记为使用原始数据

    # ===== 获取原始监督数据和原始输入 =====
    logger.info("\n获取原始输入和监督数据...")
    train_inputs_original = []
    train_targets = []
    val_inputs_original = []
    val_targets = []
    test_inputs_original = []
    test_targets = []

    with torch.no_grad():
        for inputs, targets in dataloaders['train']:
            train_inputs_original.append(inputs.numpy())
            train_targets.append(targets.numpy())
        for inputs, targets in dataloaders['val']:
            val_inputs_original.append(inputs.numpy())
            val_targets.append(targets.numpy())
        for inputs, targets in dataloaders['test']:
            test_inputs_original.append(inputs.numpy())
            test_targets.append(targets.numpy())

    train_inputs_original = np.concatenate(train_inputs_original)  # (N, 599)
    train_targets = np.concatenate(train_targets).flatten()
    val_inputs_original = np.concatenate(val_inputs_original)      # (N, 599)
    val_targets = np.concatenate(val_targets).flatten()
    test_inputs_original = np.concatenate(test_inputs_original)    # (N, 599)
    test_targets = np.concatenate(test_targets).flatten()

    logger.info(f"训练集原始输入形状: {train_inputs_original.shape}")
    logger.info(f"训练集目标形状: {train_targets.shape}")
    logger.info(f"验证集原始输入形状: {val_inputs_original.shape}")
    logger.info(f"验证集目标形状: {val_targets.shape}")
    logger.info(f"测试集原始输入形状: {test_inputs_original.shape}")
    logger.info(f"测试集目标形状: {test_targets.shape}")

    # ===== 保存融合数据 =====
    logger.info("\n保存融合数据...")

    # 根据融合方法选择不同的数据格式
    if fusion_method == 'stack':
        # stack 方法：输入是两个模型的预测 (N, 2)，目标是真实值 (N,)
        # 这样第二层模型学习如何组合两个预测
        logger.info("\n使用 stack 方法：输入为两个模型预测，目标为真实值...")

        data_dict = {
            'train_inputs': train_pred_fused,           # 两个预测堆叠 (N, 2)
            'train_targets': train_targets,             # 真实电器功率 (N,)
            'val_inputs': val_pred_fused,               # 两个预测堆叠 (N, 2)
            'val_targets': val_targets,                 # 真实电器功率 (N,)
            'test_inputs': test_inputs_original,        # 原始聚合功率 (N, 599) - 需要先预测
            'test_targets': test_targets,               # 真实电器功率 (N,)
            'fusion_method': fusion_method,
            'model1_name': 'seq2point',
            'model2_name': 'seq2point_lstm',
            'note': 'Stack method: Train/Val inputs are stacked predictions (N, 2); All targets are true appliance power (N,)'
        }
    else:
        # average/weighted_average 方法：输入是原始数据 (N, 599)，目标是融合预测 (N,)
        # 这样第二层模型学习从原始数据预测融合结果
        logger.info("\n使用融合预测作为 Transformer 的目标值...")

        data_dict = {
            'train_inputs': train_inputs_original,      # 原始聚合功率 (N, 599)
            'train_targets': train_pred_fused,          # 融合预测作为监督信号 (N,)
            'val_inputs': val_inputs_original,          # 原始聚合功率 (N, 599)
            'val_targets': val_pred_fused,              # 融合预测作为监督信号 (N,)
            'test_inputs': test_inputs_original,        # 原始聚合功率 (N, 599)
            'test_targets': test_targets,               # 真实电器功率 (N,) - 用于最终评估
            'fusion_method': fusion_method,
            'model1_name': 'seq2point',
            'model2_name': 'seq2point_lstm',
            'note': 'Average method: All inputs are original aggregate power (N, 599); Train/Val targets are fused predictions; Test targets are true appliance power'
        }

    # 保存为 npz 文件
    npz_path = os.path.join(output_dir, f'ensemble_dataset_{appliance}_{fusion_method}.npz')
    np.savez(npz_path, **data_dict)
    logger.info(f"✓ 融合数据已保存: {npz_path}")

    # ===== 生成可视化对比图表 =====
    logger.info("\n生成可视化对比图表...")
    visualize_predictions(
        pred1=train_pred1,
        pred2=train_pred2,
        pred_fused=train_pred_fused,
        targets=train_targets,
        model1_name='Seq2Point (exp30)',
        model2_name='LSTM (exp33)',
        output_dir=output_dir,
        dataset_name='Train Set'
    )

    visualize_predictions(
        pred1=val_pred1,
        pred2=val_pred2,
        pred_fused=val_pred_fused,
        targets=val_targets,
        model1_name='Seq2Point (exp30)',
        model2_name='LSTM (exp33)',
        output_dir=output_dir,
        dataset_name='Validation Set'
    )

    logger.info("✓ 可视化图表已生成")

    # 打印统计信息
    logger.info("\n=== 融合数据统计信息 ===")
    logger.info(f"训练集输入 - 均值: {train_pred_fused.mean():.4f}, 标准差: {train_pred_fused.std():.4f}")
    logger.info(f"训练集目标 - 均值: {train_targets.mean():.4f}, 标准差: {train_targets.std():.4f}")
    logger.info(f"验证集输入 - 均值: {val_pred_fused.mean():.4f}, 标准差: {val_pred_fused.std():.4f}")
    logger.info(f"测试集输入 - 形状: {test_inputs_original.shape} (原始聚合功率)")

    return npz_path


def main():
    parser = argparse.ArgumentParser(
        description='生成第二层集成学习的训练数据'
    )

    # 检查点路径
    parser.add_argument('--exp30-checkpoint', type=str, default='runs/exp30/weights/best_metrics.pth',
                       help='exp30 (Seq2Point) 的检查点路径')
    parser.add_argument('--exp33-checkpoint', type=str, default='runs/exp33/weights/best_metrics.pth',
                       help='exp33 (LSTM) 的检查点路径')

    # 配置参数
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='配置文件路径')
    parser.add_argument('--appliance', type=str, default='kettle',
                       help='电器名称')

    # 输出参数
    parser.add_argument('--output-dir', type=str, default='ensemble_data',
                       help='输出目录')
    parser.add_argument('--fusion-method', type=str, default='average',
                       choices=['average', 'weighted_average', 'stack'],
                       help='融合方法')

    args = parser.parse_args()

    npz_path = generate_ensemble_dataset(
        exp30_checkpoint=args.exp30_checkpoint,
        exp33_checkpoint=args.exp33_checkpoint,
        config_path=args.config,
        appliance=args.appliance,
        output_dir=args.output_dir,
        fusion_method=args.fusion_method
    )

    print(f"\n✅ 融合数据生成完成！\n文件位置: {npz_path}")


if __name__ == '__main__':
    main()
