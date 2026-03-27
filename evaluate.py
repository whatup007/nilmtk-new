"""
Seq2Point NILM 模型评估脚本。
"""

import os
import sys
import argparse
import random


def _ensure_positive_int_env(var_name: str, default: str = '1') -> None:
    """Ensure thread-related env vars are valid positive integers."""
    value = os.environ.get(var_name)
    if value is None:
        return
    try:
        if int(value) <= 0:
            os.environ[var_name] = default
    except (TypeError, ValueError):
        os.environ[var_name] = default


_ensure_positive_int_env('OMP_NUM_THREADS', '1')

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

# 将项目根目录加入 sys.path，便于直接运行脚本
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import get_model
from utils.data_loader import NILMDataLoader, create_dataloaders
from utils.config_utils import load_config, get_appliance_params
from utils.metrics import compute_metrics, print_metrics, compute_relative_error_metrics, compute_composite_score
from utils.logger import setup_logger, save_predictions, plot_predictions, plot_evaluation_metrics


def evaluate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    data_loader: NILMDataLoader,
    threshold: float = 10.0
) -> tuple:
    """
    评估模型在测试集上的表现。
    
    Args:
        model: 待评估模型
        dataloader: 测试集数据加载器
        device: 评估设备
        data_loader: 用于反归一化的 NILMDataLoader
        threshold: 指标计算阈值
        
    Returns:
        (predictions, targets, metrics)
    """
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc='Evaluating'):
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # 前向推理
            outputs = model(inputs)
            
            # 缓存预测与真实值
            all_predictions.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
    
    # 拼接所有批次结果
    predictions = np.concatenate(all_predictions)
    targets = np.concatenate(all_targets)
    
    # 反归一化回原始功率尺度
    predictions = data_loader.denormalize_predictions(predictions)
    targets = data_loader.denormalize_predictions(targets)
    
    # 确保预测值非负
    predictions = np.maximum(predictions, 0)
    
    # 计算评估指标
    metrics = compute_metrics(targets, predictions, threshold)
    relative_metrics = compute_relative_error_metrics(targets, predictions)
    metrics.update(relative_metrics)
    
    return predictions, targets, metrics


def evaluate_parallel(
    model_seq2point: nn.Module,
    model_transformer: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    data_loader: NILMDataLoader,
    threshold: float = 10.0
) -> tuple:
    """
    并行评估：两模型输出相加后计算指标。

    Returns:
        (pred_sum, targets, metrics, pred_seq2point, pred_transformer)
    """
    model_seq2point.eval()
    model_transformer.eval()
    all_seq2point = []
    all_transformer = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc='Evaluating'):
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs_seq2point = model_seq2point(inputs)
            outputs_transformer = model_transformer(inputs)

            all_seq2point.append(outputs_seq2point.cpu().numpy())
            all_transformer.append(outputs_transformer.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    pred_seq2point = np.concatenate(all_seq2point)
    pred_transformer = np.concatenate(all_transformer)
    targets = np.concatenate(all_targets)

    pred_seq2point = data_loader.denormalize_predictions(pred_seq2point)
    pred_transformer = data_loader.denormalize_predictions(pred_transformer)
    targets = data_loader.denormalize_predictions(targets)

    pred_sum = pred_seq2point + pred_transformer
    pred_sum = np.maximum(pred_sum, 0)

    metrics = compute_metrics(targets, pred_sum, threshold)
    relative_metrics = compute_relative_error_metrics(targets, pred_sum)
    metrics.update(relative_metrics)

    return pred_sum, targets, metrics, pred_seq2point, pred_transformer


def main():
    # 命令行参数解析器：用于配置评估流程
    parser = argparse.ArgumentParser(description='Evaluate Seq2Point NILM model')

    # 兼容参数：模型检查点路径（.pth）
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Path to model checkpoint'
    )

    # 推荐参数：实验目录名称（例如 exp18）
    parser.add_argument(
        '--exp',
        type=str,
        default=None,
        help='Experiment name under runs/ (e.g., exp18)'
    )

    # 可选参数：当使用 --exp 时指定权重文件名
    parser.add_argument(
        '--weight-name',
        type=str,
        default='best_metrics.pth',
        choices=['best_metrics.pth', 'best.pth', 'last.pth'],
        help='Weight filename to load when using --exp'
    )

    # 可选参数：配置文件路径；若不传则尝试从 checkpoint 内读取 config
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to configuration file (if None, auto-detect from experiment/checkpoint)'
    )

    # 可选参数：评估结果输出目录（指标文件、预测文件、图像）
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Directory to save evaluation results (default: results/<exp>_<split>)'
    )

    # 开关参数：是否将预测结果保存为文件（.npz）
    parser.add_argument(
        '--save-predictions',
        action='store_true',
        help='Save predictions to file'
    )

    # 开关参数：是否绘制预测曲线与评估图
    parser.add_argument(
        '--plot',
        action='store_true',
        help='Plot predictions'
    )

    # 可选参数：指定评估数据划分（训练集 / 验证集 / 测试集）
    parser.add_argument(
        '--split',
        type=str,
        default='test',
        choices=['train', 'val', 'test'],
        help='Dataset split to evaluate'
    )
    
    # 解析命令行参数
    args = parser.parse_args()

    # 兼容 --exp 20 与 --exp exp20 两种写法
    original_exp = args.exp
    if args.exp is not None:
        exp_candidates = [args.exp]
        if not args.exp.startswith('exp'):
            exp_candidates.insert(0, f"exp{args.exp}")

        for exp_name in exp_candidates:
            if os.path.isdir(os.path.join('runs', exp_name)):
                args.exp = exp_name
                break

    # 解析检查点路径：优先使用 --checkpoint；否则使用 --exp 自动拼接路径
    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        if args.exp is None:
            raise ValueError("Please provide either --checkpoint or --exp")
        checkpoint_path = os.path.join('runs', args.exp, 'weights', args.weight_name)

    # 若未显式指定输出目录，默认保存到 results/<exp>_<split>
    if args.output_dir is None:
        output_exp = args.exp
        if output_exp is None:
            checkpoint_abs = os.path.abspath(checkpoint_path)
            checkpoint_dir = os.path.dirname(checkpoint_abs)
            if os.path.basename(checkpoint_dir) == 'weights':
                output_exp = os.path.basename(os.path.dirname(checkpoint_dir))
        if output_exp is None:
            output_exp = 'exp'
        args.output_dir = os.path.join('results', f"{output_exp}_{args.split}")

    # 自动推断配置文件路径：
    # 1) 若使用 --exp，则默认 runs/<exp>/config.yaml
    # 2) 若使用 --checkpoint，则尝试从 .../runs/<exp>/weights/*.pth 推断同级 config.yaml
    if args.config is None:
        if args.exp is not None:
            exp_config_path = os.path.join('runs', args.exp, 'config.yaml')
            if os.path.exists(exp_config_path):
                args.config = exp_config_path
        if args.config is None:
            checkpoint_abs = os.path.abspath(checkpoint_path)
            checkpoint_dir = os.path.dirname(checkpoint_abs)
            if os.path.basename(checkpoint_dir) == 'weights':
                inferred_exp_dir = os.path.dirname(checkpoint_dir)
                inferred_config = os.path.join(inferred_exp_dir, 'config.yaml')
                if os.path.exists(inferred_config):
                    args.config = inferred_config
    
    # 加载检查点
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # 加载配置
    if args.config:
        config = load_config(args.config)
    else:
        if 'config' in checkpoint:
            config = checkpoint['config']
        else:
            raise ValueError("No config found in checkpoint. Please provide --config")
    
    # 设置日志
    logger = setup_logger('evaluation', config['logging']['log_dir'])
    if original_exp and args.exp != original_exp:
        logger.info(f"Resolved --exp {original_exp} -> {args.exp}")
    logger.info(f"Loaded checkpoint from {checkpoint_path}")
    logger.info(f"Checkpoint epoch: {checkpoint.get('epoch', 'N/A')}")
    
    # 设置运行设备
    if config['device']['gpu'] and torch.cuda.is_available():
        device = torch.device(f"cuda:{config['device']['gpu_id']}")
        logger.info(f"Using GPU: {torch.cuda.get_device_name(device)}")
    else:
        device = torch.device('cpu')
        logger.info("Using CPU")
    
    # 获取电器参数
    appliance_params = get_appliance_params(
        config['data']['appliance'],
        'configs/appliance_params.yaml'
    )
    max_power = appliance_params['max_power']
    threshold = appliance_params['threshold']
    
    logger.info(f"Evaluating for appliance: {config['data']['appliance']}")
    logger.info(f"Max power: {max_power}W, Threshold: {threshold}W")

    # 与训练阶段保持一致的随机种子，保证 train/val 划分可复现
    seed = config.get('seed', 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # 创建数据加载器
    data_loader = NILMDataLoader(
        data_path=config['data']['data_path'],
        appliance=config['data']['appliance'],
        window_size=config['data']['window_size'],
        window_stride=config['data']['window_stride'],
        max_power=max_power
    )
    
    # 创建训练/验证/测试加载器
    dataloaders = create_dataloaders(
        data_loader=data_loader,
        train_buildings=config['data']['train_buildings'],
        test_buildings=config['data']['test_buildings'],
        batch_size=config['training']['batch_size'],
        val_split=config['validation']['val_split'],
        num_workers=config['device']['num_workers'],
        shuffle=config['validation'].get('shuffle', True)
    )

    eval_split = args.split
    if eval_split not in dataloaders:
        raise ValueError(f"Invalid split '{eval_split}'. Available splits: {list(dataloaders.keys())}")

    eval_loader = dataloaders[eval_split]
    
    is_parallel = (
        config['training'].get('mode', 'single') == 'parallel'
        or config['model'].get('name') == 'parallel'
        or 'seq2point_state_dict' in checkpoint
    )

    if is_parallel:
        seq2point_cfg = config['model'].get('seq2point', {})
        transformer_cfg = config['model'].get('transformer', {})

        model_seq2point = get_model(
            model_name='seq2point',
            input_size=seq2point_cfg.get('input_size', config['model']['input_size']),
            dropout_rate=seq2point_cfg.get('dropout_rate', config['model'].get('dropout_rate', 0.1))
        )
        model_transformer = get_model(
            model_name='seq2point_transformer',
            input_size=transformer_cfg.get('input_size', config['model']['input_size']),
            d_model=transformer_cfg.get('d_model', 128),
            nhead=transformer_cfg.get('nhead', 4),
            num_layers=transformer_cfg.get('num_layers', 4),
            dim_feedforward=transformer_cfg.get('dim_feedforward', 256),
            dropout_rate=transformer_cfg.get('dropout_rate', 0.1)
        )

        model_seq2point.load_state_dict(checkpoint['seq2point_state_dict'])
        model_transformer.load_state_dict(checkpoint['transformer_state_dict'])
        model_seq2point = model_seq2point.to(device)
        model_transformer = model_transformer.to(device)

        logger.info("Parallel models loaded successfully")

        logger.info(f"\nEvaluating on {eval_split} set...")
        predictions, targets, metrics, pred_seq2point, pred_transformer = evaluate_parallel(
            model_seq2point, model_transformer, eval_loader, device, data_loader, threshold
        )
    else:
        # 创建模型
        model = get_model(
            model_name=config['model']['name'],
            input_size=config['model']['input_size'],
            dropout_rate=config['model']['dropout_rate']
        )

        # 加载模型权重
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)

        logger.info("Model loaded successfully")

        # 在指定数据集上评估
        logger.info(f"\nEvaluating on {eval_split} set...")
        predictions, targets, metrics = evaluate(
            model, eval_loader, device, data_loader, threshold
        )
    
    # 打印指标
    print_metrics(metrics, prefix="Test")
    
    # 额外指标
    logger.info(f"\nAdditional Metrics:")
    logger.info(f"RAE:  {metrics['rae']:.4f}")
    logger.info(f"RSE:  {metrics['rse']:.4f}")
    logger.info(f"MAPE: {metrics['mape']:.2f}%")
    
    # 保存结果
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 保存指标到文件
    metrics_file = os.path.join(
        args.output_dir,
        f"{config['data']['appliance']}_{config['data']['dataset']}_{eval_split}_metrics.txt"
    )
    with open(metrics_file, 'w') as f:
        f.write(f"Evaluation Metrics for {config['data']['appliance']}\n")
        f.write("=" * 50 + "\n")
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")
    logger.info(f"Metrics saved to {metrics_file}")
    
    # 保存预测结果
    if args.save_predictions:
        pred_file = os.path.join(
            args.output_dir,
            f"{config['data']['appliance']}_{config['data']['dataset']}_{eval_split}_predictions.npz"
        )
        if is_parallel:
            np.savez(
                pred_file,
                y_true=targets,
                pred_seq2point=pred_seq2point,
                pred_transformer=pred_transformer,
                pred_sum=predictions
            )
            logger.info(f"Predictions saved to {pred_file}")
        else:
            save_predictions(targets, predictions, args.output_dir, os.path.basename(pred_file))
    
    # 绘制预测曲线
    if args.plot:
        plot_file = os.path.join(
            args.output_dir,
            f"{config['data']['appliance']}_{config['data']['dataset']}_{eval_split}_predictions.png"
        )
        plot_predictions(
            targets,
            predictions,
            save_path=plot_file,
            num_samples=2000,
            title=f"{config['data']['appliance']} ({eval_split}) - Predictions vs Ground Truth"
        )
        
        # 生成多个评估指标图表
        plot_evaluation_metrics(
            metrics,
            output_dir=args.output_dir,
            appliance=config['data']['appliance']
        )
    
    logger.info("\nEvaluation completed!")


if __name__ == '__main__':
    main()
