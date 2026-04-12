#!/usr/bin/env python3
"""
为已训练的实验生成完整的评估指标可视化图表
仿照 train.py 的结构和保存逻辑
"""

import os
import sys
import argparse
import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models import get_model
from utils.data_loader import NILMDataLoader, create_dataloaders
from utils.config_utils import load_config, get_appliance_params
from utils.metrics import compute_metrics, compute_relative_error_metrics
from utils.plot_styles import plot_evaluation_metrics
from utils.logger import setup_logger
from utils.exp_manager import init_experiment


def resolve_exp_path(exp_input):
    """解析实验目录路径"""
    if os.path.isabs(exp_input):
        return exp_input

    # 尝试当前目录
    if os.path.isdir(exp_input):
        return os.path.abspath(exp_input)

    # 尝试 runs/ 目录
    runs_path = os.path.join('runs', exp_input)
    if os.path.isdir(runs_path):
        return os.path.abspath(runs_path)

    return os.path.abspath(exp_input)


def generate_visualization(exp_dir, checkpoint_name='best_metrics.pth', viz_config=None):
    """
    为实验生成完整的评估指标可视化图表

    Args:
        exp_dir: 实验目录（如 runs/exp44 或 exp44）
        checkpoint_name: 检查点文件名
        viz_config: 可视化配置文件路径（如果为None则使用默认）
    """
    exp_dir = resolve_exp_path(exp_dir)

    # 验证路径
    if not os.path.exists(exp_dir):
        print(f"✗ 实验目录不存在: {exp_dir}")
        return False

    config_path = os.path.join(exp_dir, 'config.yaml')
    checkpoint_path = os.path.join(exp_dir, 'weights', checkpoint_name)

    if not os.path.exists(config_path):
        print(f"✗ 配置文件不存在: {config_path}")
        return False

    if not os.path.exists(checkpoint_path):
        print(f"✗ 检查点不存在: {checkpoint_path}")
        return False

    # ===== 设置日志 =====
    logger_path = os.path.join(exp_dir, 'logs')
    os.makedirs(logger_path, exist_ok=True)
    logger = setup_logger('visualization', logger_path)

    # ===== 加载可视化配置（可选） =====
    viz_params = {}
    if viz_config is None:
        viz_config = 'configs/visualization_config.yaml'

    if os.path.exists(viz_config):
        try:
            viz_params = load_config(viz_config)
            logger.info(f"已加载可视化配置: {viz_config}")
        except Exception as e:
            logger.warning(f"无法加载可视化配置: {e}")
    else:
        logger.info(f"可视化配置文件不存在: {viz_config}，使用默认参数")

    print(f"\n{'='*70}")
    logger.info(f"为实验生成评估指标可视化: {exp_dir}")
    if viz_config and os.path.exists(viz_config):
        logger.info(f"使用可视化配置: {viz_config}")
    print(f"{'='*70}\n")

    # ===== 加载配置 =====
    config = load_config(config_path)
    logger.info(f"已加载实验配置: {config_path}")

    # 如果有可视化配置，则合并参数
    if viz_params:
        # 仅合并可视化相关的参数
        if 'evaluation' in viz_params:
            if 'evaluation' not in config:
                config['evaluation'] = {}
            config['evaluation'].update(viz_params['evaluation'])
            logger.info(f"已应用可视化评估配置")

    appliance = config['data']['appliance']
    logger.info(f"电器: {appliance}")

    # ===== 设置设备 =====
    if config['device']['gpu'] and torch.cuda.is_available():
        device = torch.device(f"cuda:{config['device']['gpu_id']}")
        logger.info(f"使用 GPU: {torch.cuda.get_device_name(device)}")
    else:
        device = torch.device('cpu')
        logger.info("使用 CPU")

    # ===== 加载检查点 =====
    logger.info(f"加载检查点: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # ===== 获取电器参数 =====
    appliance_params = get_appliance_params(
        appliance,
        'configs/appliance_params.yaml'
    )
    max_power = appliance_params['max_power']
    threshold = appliance_params['threshold']
    logger.info(f"最大功率: {max_power}W, 阈值: {threshold}W")

    # ===== 创建数据加载器 =====
    logger.info("加载数据标准化参数...")
    data_loader = NILMDataLoader(
        data_path=config['data']['data_path'],
        appliance=appliance,
        window_size=config['data']['window_size'],
        window_stride=config['data']['window_stride'],
        max_power=max_power
    )

    logger.info("创建数据加载器...")
    dataloaders = create_dataloaders(
        data_loader=data_loader,
        train_buildings=config['data']['train_buildings'],
        test_buildings=config['data']['test_buildings'],
        batch_size=config['training']['batch_size'],
        val_split=config['validation']['val_split'],
        num_workers=config['device']['num_workers'],
        shuffle=False  # 不打乱，便于可视化
    )

    # ===== 检测是否为集成模型 =====
    is_ensemble = 'ensemble' in config

    # ===== 创建和加载模型 =====
    logger.info("创建模型...")

    if is_ensemble:
        # 集成学习模型
        fusion_method = config['ensemble']['fusion_method']
        logger.info(f"检测到集成学习模型 (融合方法: {fusion_method})")

        if fusion_method == 'stack':
            # Stack 方法：使用 SimpleEnsembleModel
            from train_ensemble_model import SimpleEnsembleModel
            model = SimpleEnsembleModel(input_size=2, output_size=1).to(device)
            logger.info("使用 SimpleEnsembleModel (输入: 2, 输出: 1)")
        else:
            # Average/Weighted average: 使用 Transformer
            logger.info(f"使用 {fusion_method} 方法的 Transformer 模型")
            transformer_cfg = config['model'].get('transformer', {})
            model = get_model(
                model_name='seq2point_transformer',
                input_size=transformer_cfg.get('input_size', config['model']['input_size']),
                d_model=transformer_cfg.get('d_model', 64),
                nhead=transformer_cfg.get('nhead', 4),
                num_layers=transformer_cfg.get('num_layers', 3),
                dim_feedforward=transformer_cfg.get('dim_feedforward', 128),
                dropout_rate=transformer_cfg.get('dropout_rate', config['model'].get('dropout_rate', 0.1))
            ).to(device)
    else:
        # 普通模型
        # 自动检测实际的模型类型（根据检查点中的键）
        checkpoint_keys = set(checkpoint['model_state_dict'].keys())
        is_transformer = any('encoder' in k or 'input_proj' in k or 'pos_encoder' in k
                             for k in checkpoint_keys)

        model_name = config['model']['name']
        if is_transformer:
            logger.info("检测到 Transformer 模型架构")
            model_name = 'seq2point_transformer'
            transformer_cfg = config['model'].get('transformer', {})
            model = get_model(
                model_name=model_name,
                input_size=transformer_cfg.get('input_size', config['model']['input_size']),
                d_model=transformer_cfg.get('d_model', 64),
                nhead=transformer_cfg.get('nhead', 4),
                num_layers=transformer_cfg.get('num_layers', 3),
                dim_feedforward=transformer_cfg.get('dim_feedforward', 128),
                dropout_rate=transformer_cfg.get('dropout_rate', config['model'].get('dropout_rate', 0.1))
            )
        else:
            logger.info(f"使用配置的模型: {model_name}")
            model = get_model(
                model_name=model_name,
                input_size=config['model']['input_size'],
                dropout_rate=config['model'].get('dropout_rate', 0.1)
            )

        model = model.to(device)

    # 加载模型权重
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info("模型权重已加载")
    else:
        logger.warning("检查点中没有 'model_state_dict'")
        return False

    # ===== 在测试集上评估 =====
    logger.info("\n在测试集上进行评估...")
    model.eval()

    if is_ensemble and fusion_method == 'stack':
        # Stack 方法：需要先用两个基础模型生成预测，然后堆叠
        logger.info("集成模型 (stack 方法) - 加载基础模型并生成堆叠预测...")

        # 加载两个基础模型
        model1_checkpoint_path = config['ensemble']['model1']['checkpoint']
        model2_checkpoint_path = config['ensemble']['model2']['checkpoint']
        model1_name = config['ensemble']['model1']['name']
        model2_name = config['ensemble']['model2']['name']

        logger.info(f"  加载模型1 ({model1_name}): {model1_checkpoint_path}")
        ckpt1 = torch.load(model1_checkpoint_path, map_location=device, weights_only=False)
        model1 = get_model(model1_name, input_size=599).to(device)
        model1.load_state_dict(ckpt1['model_state_dict'])
        model1.eval()

        logger.info(f"  加载模型2 ({model2_name}): {model2_checkpoint_path}")
        ckpt2 = torch.load(model2_checkpoint_path, map_location=device, weights_only=False)
        model2 = get_model(model2_name, input_size=599).to(device)
        model2.load_state_dict(ckpt2['model_state_dict'])
        model2.eval()

        # 从测试集生成两个模型的预测
        logger.info("  生成两个基础模型的预测...")
        pred1_list = []
        pred2_list = []
        test_targets = []

        with torch.no_grad():
            for inputs, targets in dataloaders['test']:
                inputs = inputs.to(device)

                # 两个模型的预测
                out1 = model1(inputs)
                out2 = model2(inputs)

                pred1_list.append(out1.cpu().numpy())
                pred2_list.append(out2.cpu().numpy())
                test_targets.append(targets.cpu().numpy())

        pred1 = np.concatenate(pred1_list).flatten()
        pred2 = np.concatenate(pred2_list).flatten()
        test_targets = np.concatenate(test_targets)

        logger.info(f"  模型1预测形状: {pred1.shape}")
        logger.info(f"  模型2预测形状: {pred2.shape}")

        # 反归一化基础模型的预测
        pred1 = data_loader.denormalize_predictions(pred1.reshape(-1, 1)).flatten()
        pred2 = data_loader.denormalize_predictions(pred2.reshape(-1, 1)).flatten()
        test_targets = data_loader.denormalize_predictions(test_targets)

        # 堆叠预测
        stacked_inputs = np.column_stack([pred1, pred2])
        logger.info(f"  堆叠预测形状: {stacked_inputs.shape}")

        # 使用集成模型预测
        logger.info("  使用集成模型进行最终预测...")
        test_predictions = []
        stacked_tensor = torch.FloatTensor(stacked_inputs)

        with torch.no_grad():
            batch_size = 2048
            for i in range(0, len(stacked_tensor), batch_size):
                batch = stacked_tensor[i:i+batch_size].to(device)
                outputs = model(batch)
                test_predictions.append(outputs.cpu().numpy())

        test_predictions = np.concatenate(test_predictions).flatten()
        logger.info(f"  集成模型预测形状: {test_predictions.shape}")

    else:
        # 普通模型或 average/weighted_average 集成
        test_predictions = []
        test_targets = []

        with torch.no_grad():
            for inputs, targets in dataloaders['test']:
                inputs = inputs.to(device)
                outputs = model(inputs)
                test_predictions.append(outputs.cpu().numpy())
                test_targets.append(targets.cpu().numpy())

        test_predictions = np.concatenate(test_predictions)
        test_targets = np.concatenate(test_targets)

        # 反归一化
        test_predictions = data_loader.denormalize_predictions(test_predictions)
        test_targets = data_loader.denormalize_predictions(test_targets)

    logger.info(f"✓ 评估完成: {len(test_targets)} 个样本")

    # ===== 计算指标 =====
    logger.info("\n计算评估指标...")
    test_metrics = compute_metrics(test_targets, test_predictions, threshold)

    logger.info(f"  MAE: {test_metrics.get('mae', 0):.2f} W")
    logger.info(f"  RMSE: {test_metrics.get('rmse', 0):.2f} W")
    logger.info(f"  F1 Score: {test_metrics.get('f1', 0):.4f}")
    logger.info(f"  R² Score: {test_metrics.get('r2_score', 0):.4f}")
    logger.info(f"  能量准确度: {test_metrics.get('energy_accuracy', 0):.4f}")

    # 计算相对误差指标
    logger.info("计算相对误差指标...")
    relative_metrics = compute_relative_error_metrics(test_targets, test_predictions)
    test_metrics.update(relative_metrics)

    # ===== 生成评估指标图像 =====
    logger.info("\n生成评估指标图像...")
    plot_evaluation_metrics(
        test_metrics,
        output_dir=exp_dir,
        appliance=appliance,
        y_true=test_targets,
        y_pred=test_predictions
    )
    logger.info(f"✓ 评估指标图像已保存到: {exp_dir}")

    # ===== 完成 =====
    print(f"\n{'='*70}")
    logger.info("✓ 可视化生成完成！")
    print(f"{'='*70}\n")
    logger.info(f"生成的图表:")
    logger.info(f"  - 00_{appliance}_comprehensive_evaluation.png (综合评估仪表盘)")
    logger.info(f"  - 01_{appliance}_error_distribution.png (误差分布)")
    logger.info(f"  - 02_{appliance}_state_recognition.png (状态识别性能)")
    logger.info(f"  - 03_{appliance}_predictions_comparison.png (预测对比)")
    logger.info(f"  - 04_{appliance}_prediction_scatter.png (预测散点图)")

    return True


def main():
    parser = argparse.ArgumentParser(
        description='为已训练的实验生成评估指标可视化图表',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        'exp_dir',
        help='实验目录 (如 runs/exp44 或 exp44)'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='best_metrics.pth',
        choices=['best_metrics.pth', 'best.pth', 'last.pth'],
        help='使用的检查点文件名'
    )
    parser.add_argument(
        '--viz-config',
        type=str,
        default='configs/visualization_config.yaml',
        help='可视化配置文件路径'
    )

    args = parser.parse_args()

    success = generate_visualization(args.exp_dir, args.checkpoint, args.viz_config)
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
