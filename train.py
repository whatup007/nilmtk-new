"""
Seq2Point NILM模型训练脚本
训练脚本用于训练非侵入式负荷监测（NILM）的Seq2Point模型
支持命令行参数覆盖配置文件，提供灵活的训练选项
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
import torch.optim as optim
from tqdm import tqdm

# 将项目根目录加入 sys.path，便于直接运行脚本
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import get_model
from utils.data_loader import NILMDataLoader, create_dataloaders
from utils.config_utils import load_config, get_appliance_params, ConfigValidator
from utils.metrics import compute_metrics, print_metrics, EarlyStopping, compute_composite_score, compute_relative_error_metrics
from utils.logger import setup_logger, TrainingLogger
from utils.plot_styles import plot_evaluation_metrics
from utils.exp_manager import init_experiment


def set_seed(seed: int):
    """
    设置随机种子以确保结果可复现
    为Python、NumPy和PyTorch设置相同的随机种子
    
    Args:
        seed: 随机种子值
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True  # 确保CUDA操作确定性
        torch.backends.cudnn.benchmark = False  # 禁用自动优化以保证可复现


def create_optimizer(
    optimizer_name: str,
    params,
    learning_rate: float
) -> optim.Optimizer:
    """
    根据名称创建优化器。
    
    Args:
        optimizer_name: 优化器名称
        params: 模型参数
        learning_rate: 学习率
    """
    if optimizer_name == 'adam':
        return optim.Adam(params, lr=learning_rate)
    if optimizer_name == 'sgd':
        return optim.SGD(params, lr=learning_rate, momentum=0.9)
    if optimizer_name == 'rmsprop':
        return optim.RMSprop(params, lr=learning_rate)
    raise ValueError(f"未知的优化器: {optimizer_name}")


def resolve_resume_path(resume: str) -> str:
    """Resolve --resume input to a concrete checkpoint file when possible."""
    if not resume:
        return resume

    normalized = os.path.normpath(resume)
    if os.path.isfile(normalized):
        return normalized

    candidates = []

    if os.path.isdir(normalized):
        candidates.extend([
            os.path.join(normalized, 'weights', 'last.pth'),
            os.path.join(normalized, 'weights', 'best.pth'),
            os.path.join(normalized, 'weights', 'best_metrics.pth'),
            os.path.join(normalized, 'last.pth'),
        ])

    base_name = os.path.basename(normalized)
    if base_name.startswith('exp') and base_name[3:].isdigit():
        exp_dir = os.path.join('runs', base_name)
        candidates.extend([
            os.path.join(exp_dir, 'weights', 'last.pth'),
            os.path.join(exp_dir, 'weights', 'best.pth'),
            os.path.join(exp_dir, 'weights', 'best_metrics.pth'),
        ])

    if not normalized.lower().endswith('.pth'):
        candidates.append(normalized + '.pth')

    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate

    return normalized


def infer_config_from_resume(config_path: str, resume: str) -> str:
    """When using default config, infer experiment config next to checkpoint."""
    if not resume:
        return config_path

    normalized_config = os.path.normpath(config_path)
    default_config = os.path.normpath(os.path.join('configs', 'config.yaml'))
    if normalized_config != default_config:
        return config_path

    checkpoint_dir = os.path.dirname(os.path.normpath(resume))
    exp_dir = os.path.dirname(checkpoint_dir)
    candidate = os.path.join(exp_dir, 'config.yaml')
    if os.path.isfile(candidate):
        return candidate

    return config_path


def train_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device
) -> float:
    """
    训练一个epoch（训练周期）
    遍历整个训练数据集，更新模型参数
    
    Args:
        model: 要训练的模型
        dataloader: 训练数据加载器
        criterion: 损失函数
        optimizer: 优化器
        device: 训练设备（CPU或GPU）
        
    Returns:
        平均训练损失
    """
    model.train()  # 设置模型为训练模式
    total_loss = 0.0
    
    with tqdm(dataloader, desc='Training', leave=False) as pbar:
        for inputs, targets in pbar:
            inputs = inputs.to(device)  # 输入数据移至指定设备
            targets = targets.to(device)  # 目标数据移至指定设备
            
            # 前向传播
            optimizer.zero_grad()  # 清空梯度
            outputs = model(inputs)  # 模型预测
            loss = criterion(outputs, targets)  # 计算损失
            
            # 反向传播
            loss.backward()  # 计算梯度
            optimizer.step()  # 更新参数
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})  # 更新进度条显示
    
    return total_loss / len(dataloader)  # 返回平均损失


def train_epoch_parallel(
    model_seq2point: nn.Module,
    model_transformer: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer_seq2point: optim.Optimizer,
    optimizer_transformer: optim.Optimizer,
    device: torch.device
) -> float:
    """
    并行训练一个 epoch（输出相加后计算损失）。
    """
    model_seq2point.train()
    model_transformer.train()
    total_loss = 0.0

    with tqdm(dataloader, desc='Training', leave=False) as pbar:
        for inputs, targets in pbar:
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer_seq2point.zero_grad()
            optimizer_transformer.zero_grad()

            outputs_seq2point = model_seq2point(inputs)
            outputs_transformer = model_transformer(inputs)
            outputs_sum = outputs_seq2point + outputs_transformer

            loss = criterion(outputs_sum, targets)
            loss.backward()

            optimizer_seq2point.step()
            optimizer_transformer.step()

            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})

    return total_loss / len(dataloader)


def validate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    data_loader: NILMDataLoader,
    threshold: float = 10.0
) -> tuple:
    """
    验证模型性能
    在验证集上评估模型，不进行参数更新
    
    Args:
        model: 要验证的模型
        dataloader: 验证数据加载器
        criterion: 损失函数
        device: 验证设备（CPU或GPU）
        data_loader: NILMDataLoader，用于数据反归一化
        threshold: 用于分类指标计算的阈值（功率阈值，单位W）
        
    Returns:
        (损失, 评估指标) 的元组
    """
    model.eval()  # 设置模型为评估模式
    total_loss = 0.0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():  # 禁用梯度计算
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            
            # 存储预测值和目标值
            all_predictions.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
    
    # 合并所有批次的预测值和目标值
    predictions = np.concatenate(all_predictions)
    targets = np.concatenate(all_targets)
    
    # 反归一化（还原到实际功率值）
    predictions = data_loader.denormalize_predictions(predictions)
    targets = data_loader.denormalize_predictions(targets)
    
    # 计算评估指标
    metrics = compute_metrics(targets, predictions, threshold)
    
    avg_loss = total_loss / len(dataloader)
    
    return avg_loss, metrics


def validate_parallel(
    model_seq2point: nn.Module,
    model_transformer: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    data_loader: NILMDataLoader,
    threshold: float = 10.0
) -> tuple:
    """
    并行验证模型性能（两模型输出相加后计算）。
    """
    model_seq2point.eval()
    model_transformer.eval()
    total_loss = 0.0
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs_seq2point = model_seq2point(inputs)
            outputs_transformer = model_transformer(inputs)
            outputs_sum = outputs_seq2point + outputs_transformer

            loss = criterion(outputs_sum, targets)
            total_loss += loss.item()

            all_predictions.append(outputs_sum.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    predictions = np.concatenate(all_predictions)
    targets = np.concatenate(all_targets)

    predictions = data_loader.denormalize_predictions(predictions)
    targets = data_loader.denormalize_predictions(targets)

    metrics = compute_metrics(targets, predictions, threshold)
    avg_loss = total_loss / len(dataloader)

    return avg_loss, metrics


def train(config_path: str, resume: str = None, **kwargs):
    """
    主训练函数
    加载配置、准备数据、初始化模型并执行训练循环
    
    Args:
        config_path: 配置文件路径
        resume: 恢复训练的检查点路径
        **kwargs: 命令行参数，用于覆盖配置文件中的设置
    """
    # 加载配置
    config = load_config(config_path)
    
    # 使用命令行参数覆盖配置
    if kwargs:
        if kwargs.get('mode'):
            config['training']['mode'] = kwargs['mode']
        if kwargs.get('appliance'):
            config['data']['appliance'] = kwargs['appliance']
        if kwargs.get('data_path'):
            config['data']['data_path'] = kwargs['data_path']
        if kwargs.get('epochs'):
            config['training']['epochs'] = kwargs['epochs']
        if kwargs.get('batch_size'):
            config['training']['batch_size'] = kwargs['batch_size']
        if kwargs.get('learning_rate'):
            if config['training'].get('mode', 'single') == 'parallel':
                config.setdefault('training', {}).setdefault('parallel', {}).setdefault('seq2point', {})['learning_rate'] = kwargs['learning_rate']
                config.setdefault('training', {}).setdefault('parallel', {}).setdefault('transformer', {})['learning_rate'] = kwargs['learning_rate']
            else:
                config['training']['learning_rate'] = kwargs['learning_rate']
        if kwargs.get('optimizer'):
            if config['training'].get('mode', 'single') == 'parallel':
                config.setdefault('training', {}).setdefault('parallel', {}).setdefault('seq2point', {})['optimizer'] = kwargs['optimizer']
                config.setdefault('training', {}).setdefault('parallel', {}).setdefault('transformer', {})['optimizer'] = kwargs['optimizer']
            else:
                config['training']['optimizer'] = kwargs['optimizer']
        if kwargs.get('window_size'):
            config['data']['window_size'] = kwargs['window_size']
            config['model']['input_size'] = kwargs['window_size']
            if 'seq2point' in config['model']:
                config['model']['seq2point']['input_size'] = kwargs['window_size']
            if 'transformer' in config['model']:
                config['model']['transformer']['input_size'] = kwargs['window_size']
        if kwargs.get('window_stride'):
            config['data']['window_stride'] = kwargs['window_stride']
        if kwargs.get('dropout'):
            config['model']['dropout_rate'] = kwargs['dropout']
            if 'seq2point' in config['model']:
                config['model']['seq2point']['dropout_rate'] = kwargs['dropout']
            if 'transformer' in config['model']:
                config['model']['transformer']['dropout_rate'] = kwargs['dropout']
        if kwargs.get('patience'):
            config['training']['patience'] = kwargs['patience']
        if kwargs.get('workers'):
            config['device']['num_workers'] = kwargs['workers']
        if kwargs.get('device'):
            config['device']['gpu'] = (kwargs['device'] != 'cpu')
            if kwargs['device'] != 'cpu':
                try:
                    config['device']['gpu_id'] = int(kwargs['device'])
                except ValueError:
                    pass
        if kwargs.get('seed'):
            config['seed'] = kwargs['seed']
        if kwargs.get('save_dir'):
            config['logging']['log_dir'] = kwargs['save_dir']
    
    ConfigValidator.validate(config)  # 验证配置的有效性
    
    # ===== 初始化实验管理器 (YOLO风格的 runs/expN 结构) =====
    exp_manager = init_experiment(base_dir='runs')
    exp_dir = exp_manager.get_new_exp_dir()
    logger_path = os.path.join(exp_dir, 'logs')
    os.makedirs(logger_path, exist_ok=True)
    
    # 设置日志
    logger = setup_logger('training', logger_path)
    logger.info(f"从 {config_path} 加载配置")
    
    # 保存训练配置到实验目录
    exp_manager.save_config(config, 'config.yaml')
    logger.info(f"实验目录: {exp_dir}")
    
    # 设置随机种子
    set_seed(config['seed'])
    logger.info(f"设置随机种子为 {config['seed']}")
    
    # 设置计算设备
    if config['device']['gpu'] and torch.cuda.is_available():
        device = torch.device(f"cuda:{config['device']['gpu_id']}")
        logger.info(f"使用 GPU: {torch.cuda.get_device_name(device)}")
    else:
        device = torch.device('cpu')
        logger.info("使用 CPU")
    
    # 获取电器参数
    appliance_params = get_appliance_params(
        config['data']['appliance'],
        'configs/appliance_params.yaml'
    )
    max_power = appliance_params['max_power']  # 最大功率
    threshold = appliance_params['threshold']  # 开/关状态阈值
    
    logger.info(f"训练电器: {config['data']['appliance']}")
    logger.info(f"最大功率: {max_power}W, 阈值: {threshold}W")
    
    # 创建数据加载器
    data_loader = NILMDataLoader(
        data_path=config['data']['data_path'],
        appliance=config['data']['appliance'],
        window_size=config['data']['window_size'],
        window_stride=config['data']['window_stride'],
        max_power=max_power
    )
    
    # 创建训练/验证/测试数据加载器
    dataloaders = create_dataloaders(
        data_loader=data_loader,
        train_buildings=config['data']['train_buildings'],
        test_buildings=config['data']['test_buildings'],
        batch_size=config['training']['batch_size'],
        val_split=config['validation']['val_split'],
        num_workers=config['device']['num_workers'],
        shuffle=config['validation']['shuffle']
    )

    # 并行训练模式：Seq2Point + Transformer
    mode = config['training'].get('mode', 'single')
    if mode == 'parallel' or config['model'].get('name') == 'parallel':
        seq2point_cfg = config['model'].get('seq2point', {})
        transformer_cfg = config['model'].get('transformer', {})

        model_seq2point = get_model(
            model_name='seq2point',
            input_size=seq2point_cfg.get('input_size', config['model']['input_size']),
            dropout_rate=seq2point_cfg.get('dropout_rate', config['model'].get('dropout_rate', 0.1))
        ).to(device)

        model_transformer = get_model(
            model_name='seq2point_transformer',
            input_size=transformer_cfg.get('input_size', config['model']['input_size']),
            d_model=transformer_cfg.get('d_model', 128),
            nhead=transformer_cfg.get('nhead', 4),
            num_layers=transformer_cfg.get('num_layers', 4),
            dim_feedforward=transformer_cfg.get('dim_feedforward', 256),
            dropout_rate=transformer_cfg.get('dropout_rate', 0.1)
        ).to(device)

        # 统计参数量
        num_params_seq2point = sum(p.numel() for p in model_seq2point.parameters() if p.requires_grad)
        num_params_transformer = sum(p.numel() for p in model_transformer.parameters() if p.requires_grad)
        logger.info(f"Seq2Point 参数数量: {num_params_seq2point:,}")
        logger.info(f"Transformer 参数数量: {num_params_transformer:,}")

        # 损失函数
        if config['training']['loss_function'] == 'mse':
            criterion = nn.MSELoss()
        elif config['training']['loss_function'] == 'mae':
            criterion = nn.L1Loss()
        else:
            raise ValueError(f"未知的损失函数: {config['training']['loss_function']}")

        parallel_cfg = config['training'].get('parallel', {})
        seq2point_train_cfg = parallel_cfg.get('seq2point', {})
        transformer_train_cfg = parallel_cfg.get('transformer', {})

        optimizer_seq2point = create_optimizer(
            seq2point_train_cfg.get('optimizer', config['training'].get('optimizer', 'adam')),
            model_seq2point.parameters(),
            seq2point_train_cfg.get('learning_rate', config['training'].get('learning_rate', 0.001))
        )
        optimizer_transformer = create_optimizer(
            transformer_train_cfg.get('optimizer', config['training'].get('optimizer', 'adam')),
            model_transformer.parameters(),
            transformer_train_cfg.get('learning_rate', config['training'].get('learning_rate', 0.001))
        )

        if config['training']['lr_schedule']:
            scheduler_seq2point = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer_seq2point,
                mode='min',
                factor=config['training']['lr_decay_factor'],
                patience=config['training']['lr_decay_patience']
            )
            scheduler_transformer = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer_transformer,
                mode='min',
                factor=config['training']['lr_decay_factor'],
                patience=config['training']['lr_decay_patience']
            )
        else:
            scheduler_seq2point = None
            scheduler_transformer = None

        if config['training']['early_stopping']:
            early_stopping = EarlyStopping(
                patience=config['training']['patience'],
                mode='min'
            )
        else:
            early_stopping = None

        training_logger = TrainingLogger(
            log_dir=logger_path,
            experiment_name=f"{config['data']['appliance']}_{config['data']['dataset']}_parallel"
        )

        start_epoch = 0
        best_val_loss = float('inf')
        best_metrics_score = -float('inf')  # 最佳综合指标分数（综合评分）

        if resume:
            if os.path.exists(resume):
                checkpoint = torch.load(resume, map_location=device, weights_only=False)
                model_seq2point.load_state_dict(checkpoint['seq2point_state_dict'])
                model_transformer.load_state_dict(checkpoint['transformer_state_dict'])
                optimizer_seq2point.load_state_dict(checkpoint['optimizer_state_dicts']['seq2point'])
                optimizer_transformer.load_state_dict(checkpoint['optimizer_state_dicts']['transformer'])
                start_epoch = checkpoint['epoch'] + 1
                best_val_loss = checkpoint['best_val_loss']
                logger.info(f"从检查点恢复: {resume}")
                logger.info(f"从第 {start_epoch} 轮开始")
            else:
                logger.warning(f"检查点未找到: {resume}")

        logger.info("开始并行训练...")

        for epoch in range(start_epoch, config['training']['epochs']):
            logger.info(f"\nEpoch {epoch + 1}/{config['training']['epochs']}")

            train_loss = train_epoch_parallel(
                model_seq2point,
                model_transformer,
                dataloaders['train'],
                criterion,
                optimizer_seq2point,
                optimizer_transformer,
                device
            )
            logger.info(f"训练损失: {train_loss:.6f}")

            val_loss, val_metrics = validate_parallel(
                model_seq2point,
                model_transformer,
                dataloaders['val'],
                criterion,
                device,
                data_loader,
                threshold
            )
            logger.info(f"验证损失: {val_loss:.6f}")
            print_metrics(val_metrics, prefix="Validation")

            current_lr = optimizer_seq2point.param_groups[0]['lr']

            training_logger.log_epoch(
                epoch=epoch + 1,
                train_loss=train_loss,
                val_loss=val_loss,
                val_metrics=val_metrics,
                learning_rate=current_lr
            )

            if scheduler_seq2point:
                scheduler_seq2point.step(val_loss)
            if scheduler_transformer:
                scheduler_transformer.step(val_loss)

            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
                logger.info(f"新的最佳模型！验证损失: {val_loss:.6f}")
            
            # 综合指标评分（考虑MAE、RMSE、F1、R2、能量准确度）
            current_composite_score = compute_composite_score(val_metrics)
            is_best_metrics = current_composite_score > best_metrics_score
            if is_best_metrics:
                best_metrics_score = current_composite_score
                logger.info(
                    f"最佳综合指标！综合评分: {current_composite_score:.6f} "
                    f"(MAE: {val_metrics['mae']:.4f}, RMSE: {val_metrics['rmse']:.4f}, "
                    f"F1: {val_metrics['f1']:.4f}, R²: {val_metrics['r2_score']:.4f}, "
                    f"能量准确度: {val_metrics['energy_accuracy']:.4f})"
                )
            
            # 创建检查点（并行模式）
            checkpoint = {
                'epoch': epoch,
                'seq2point_state_dict': model_seq2point.state_dict(),
                'transformer_state_dict': model_transformer.state_dict(),
                'optimizer_state_dicts': {
                    'seq2point': optimizer_seq2point.state_dict(),
                    'transformer': optimizer_transformer.state_dict()
                },
                'train_loss': train_loss,
                'val_loss': val_loss,
                'best_val_loss': best_val_loss,
                'val_metrics': val_metrics,
                'config': config
            }

            # 保存三个版本的模型
            # 1. 最后一次模型（每个epoch都保存）
            exp_manager.save_last_weight(checkpoint, 'last.pth')
            
            # 2. 最低验证损失的模型
            if is_best:
                exp_manager.save_best_weight(checkpoint, 'best.pth')
            
            # 3. 最佳综合指标的模型
            if is_best_metrics:
                exp_manager.save_best_metrics_weight(checkpoint, 'best_metrics.pth')

            # 定期保存训练曲线和指标图
            save_frequency = config['training'].get('save_frequency', 0)
            if save_frequency and (epoch + 1) % save_frequency == 0:
                training_logger.save_history()
                training_logger.plot_history()
                training_logger.plot_metrics()

                log_dir = training_logger.log_dir
                for file in os.listdir(log_dir):
                    if file.endswith('.png'):
                        src_path = os.path.join(log_dir, file)
                        exp_manager.save_visualization(src_path, file)

            if early_stopping:
                if early_stopping(val_loss):
                    logger.info(f"在第 {epoch + 1} 轮触发早停")
                    break

        training_logger.save_history()
        training_logger.plot_history()
        training_logger.plot_metrics()

        logger.info("并行训练完成！")
        
        # 复制可视化图像到实验目录
        logger.info("\n保存可视化图像到实验目录...")
        log_dir = training_logger.log_dir
        for file in os.listdir(log_dir):
            if file.endswith('.png'):
                src_path = os.path.join(log_dir, file)
                exp_manager.save_visualization(src_path, file)

        logger.info("\n在测试集上进行评估...")
        test_loss, test_metrics = validate_parallel(
            model_seq2point,
            model_transformer,
            dataloaders['test'],
            criterion,
            device,
            data_loader,
            threshold
        )
        logger.info(f"测试损失: {test_loss:.6f}")
        print_metrics(test_metrics, prefix="Test")
        
        # 计算相对误差指标（用于评估指标可视化）
        logger.info("\n计算相对误差指标...")
        # 需要获取所有测试数据来计算
        test_predictions = []
        test_targets = []
        with torch.no_grad():
            for inputs, targets in dataloaders['test']:
                inputs = inputs.to(device)
                seq2point_out = model_seq2point(inputs)
                transformer_out = model_transformer(inputs)
                outputs = seq2point_out + transformer_out  # 并行模式下两个模型输出相加
                test_predictions.append(outputs.cpu().numpy())
                test_targets.append(targets.cpu().numpy())
        
        test_predictions = np.concatenate(test_predictions)
        test_targets = np.concatenate(test_targets)
        test_predictions = data_loader.denormalize_predictions(test_predictions)
        test_targets = data_loader.denormalize_predictions(test_targets)
        
        # 计算相对误差指标
        relative_metrics = compute_relative_error_metrics(test_targets, test_predictions)
        test_metrics.update(relative_metrics)
        
        # 生成评估指标图像到实验目录
        logger.info("\n生成评估指标图像...")
        plot_evaluation_metrics(
            test_metrics,
            output_dir=exp_manager.exp_dir,
            appliance=config['data']['appliance'],
            y_true=test_targets,
            y_pred=test_predictions
        )
        logger.info(f"✓ 评估指标图像已保存到: {exp_manager.exp_dir}")
        
        # 打印实验目录结构
        logger.info("\n✓ 并行训练完成！")
        exp_manager.print_structure()
        return
    
    # 创建模型
    model_name = config['model']['name']
    if model_name in ('transformer', 'seq2point_transformer'):
        transformer_cfg = config['model'].get('transformer', {})
        model = get_model(
            model_name=model_name,
            input_size=transformer_cfg.get('input_size', config['model']['input_size']),
            d_model=transformer_cfg.get('d_model', 128),
            nhead=transformer_cfg.get('nhead', 4),
            num_layers=transformer_cfg.get('num_layers', 4),
            dim_feedforward=transformer_cfg.get('dim_feedforward', 256),
            dropout_rate=transformer_cfg.get('dropout_rate', config['model'].get('dropout_rate', 0.1))
        )
    elif model_name == 'seq2point_lstm':
        lstm_cfg = config['model'].get('lstm', {})
        model = get_model(
            model_name=model_name,
            input_size=config['model']['input_size'],
            hidden_size=lstm_cfg.get('hidden_size', 128),
            num_layers=lstm_cfg.get('num_layers', 2),
            dropout_rate=lstm_cfg.get('dropout_rate', config['model'].get('dropout_rate', 0.1)),
            bidirectional=lstm_cfg.get('bidirectional', False)
        )
    elif model_name == 'seq2point_bilstm':
        bilstm_cfg = config['model'].get('bilstm', {})
        model = get_model(
            model_name=model_name,
            input_size=config['model']['input_size'],
            hidden_size=bilstm_cfg.get('hidden_size', 128),
            num_layers=bilstm_cfg.get('num_layers', 2),
            dropout_rate=bilstm_cfg.get('dropout_rate', config['model'].get('dropout_rate', 0.1))
        )
    else:
        model = get_model(
            model_name=model_name,
            input_size=config['model']['input_size'],
            dropout_rate=config['model']['dropout_rate']
        )
    model = model.to(device)
    
    # 计算模型参数数量
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"模型参数数量: {num_params:,}")
    
    # 设置损失函数
    if config['training']['loss_function'] == 'mse':
        criterion = nn.MSELoss()  # 均方误差损失
    elif config['training']['loss_function'] == 'mae':
        criterion = nn.L1Loss()  # 平均绝对误差损失
    else:
        raise ValueError(f"未知的损失函数: {config['training']['loss_function']}")
    
    # 设置优化器
    if config['training']['optimizer'] == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=config['training']['learning_rate']
        )
    elif config['training']['optimizer'] == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=config['training']['learning_rate'],
            momentum=0.9
        )
    elif config['training']['optimizer'] == 'rmsprop':
        optimizer = optim.RMSprop(
            model.parameters(),
            lr=config['training']['learning_rate']
        )
    else:
        raise ValueError(f"未知的优化器: {config['training']['optimizer']}")
    
    # 设置学习率调度器
    if config['training']['lr_schedule']:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',  # 当损失不再下降时减小学习率
            factor=config['training']['lr_decay_factor'],
            patience=config['training']['lr_decay_patience']
        )
    else:
        scheduler = None
    
    # 设置早停机制
    if config['training']['early_stopping']:
        early_stopping = EarlyStopping(
            patience=config['training']['patience'],
            mode='min'
        )
    else:
        early_stopping = None
    
    # 设置训练日志记录器
    training_logger = TrainingLogger(
        log_dir=logger_path,
        experiment_name=f"{config['data']['appliance']}_{config['data']['dataset']}"
    )
    
    # 加载检查点（如果需要恢复训练）
    start_epoch = 0  # 起始训练轮次
    best_val_loss = float('inf')  # 最佳验证损失
    best_metrics_score = -float('inf')  # 最佳综合指标分数（综合评分）
    
    if resume:
        if os.path.exists(resume):
            checkpoint = torch.load(resume, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_val_loss = checkpoint['best_val_loss']
            logger.info(f"从检查点恢复: {resume}")
            logger.info(f"从第 {start_epoch} 轮开始")
        else:
            logger.warning(f"检查点未找到: {resume}")
    
    # 训练循环
    logger.info("开始训练...")
    
    for epoch in range(start_epoch, config['training']['epochs']):
        logger.info(f"\nEpoch {epoch + 1}/{config['training']['epochs']}")
        
        # 训练一个epoch
        train_loss = train_epoch(model, dataloaders['train'], criterion, optimizer, device)
        logger.info(f"训练损失: {train_loss:.6f}")
        
        # 验证
        val_loss, val_metrics = validate(
            model, dataloaders['val'], criterion, device, data_loader, threshold
        )
        logger.info(f"验证损失: {val_loss:.6f}")
        print_metrics(val_metrics, prefix="Validation")
        
        # 获取当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        
        # 记录指标
        training_logger.log_epoch(
            epoch=epoch + 1,
            train_loss=train_loss,
            val_loss=val_loss,
            val_metrics=val_metrics,
            learning_rate=current_lr
        )
        
        # 更新学习率
        if scheduler:
            scheduler.step(val_loss)
        
        # 保存检查点
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            logger.info(f"新的最佳模型！验证损失: {val_loss:.6f}")
        
        # 综合指标评分（考虑MAE、RMSE、F1、R2、能量准确度）
        current_composite_score = compute_composite_score(val_metrics)
        is_best_metrics = current_composite_score > best_metrics_score
        if is_best_metrics:
            best_metrics_score = current_composite_score
            logger.info(
                f"最佳综合指标！综合评分: {current_composite_score:.6f} "
                f"(MAE: {val_metrics['mae']:.4f}, RMSE: {val_metrics['rmse']:.4f}, "
                f"RAE: {val_metrics.get('rae', 0.0):.4f}, "
                f"F1: {val_metrics['f1']:.4f}, R²: {val_metrics['r2_score']:.4f}, "
                f"能量准确度: {val_metrics['energy_accuracy']:.4f})"
            )
        
        # 创建检查点
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'best_val_loss': best_val_loss,
            'val_metrics': val_metrics,
            'config': config
        }
        
        # 保存三个版本的模型
        # 1. 最后一次模型（每个epoch都保存）
        exp_manager.save_last_weight(checkpoint, 'last.pth')
        
        # 2. 最低验证损失的模型
        if is_best:
            exp_manager.save_best_weight(checkpoint, 'best.pth')
        
        # 3. 最佳综合指标的模型
        if is_best_metrics:
            exp_manager.save_best_metrics_weight(checkpoint, 'best_metrics.pth')

        # 定期保存训练曲线和指标图
        save_frequency = config['training'].get('save_frequency', 0)
        if save_frequency and (epoch + 1) % save_frequency == 0:
            training_logger.save_history()
            training_logger.plot_history()
            training_logger.plot_metrics()

            log_dir = training_logger.log_dir
            for file in os.listdir(log_dir):
                if file.endswith('.png'):
                    src_path = os.path.join(log_dir, file)
                    exp_manager.save_visualization(src_path, file)
        
        # 早停检查
        if early_stopping:
            if early_stopping(val_loss):
                logger.info(f"在第 {epoch + 1} 轮触发早停")
                break
    
    # 保存训练历史
    training_logger.save_history()
    training_logger.plot_history()
    training_logger.plot_metrics()
    
    logger.info("训练完成！")
    
    # 复制可视化图像到实验目录
    logger.info("\n保存可视化图像到实验目录...")
    log_dir = training_logger.log_dir
    for file in os.listdir(log_dir):
        if file.endswith('.png'):
            src_path = os.path.join(log_dir, file)
            exp_manager.save_visualization(src_path, file)
    
    # 在测试集上进行最终评估
    logger.info("\n在测试集上进行评估...")
    test_loss, test_metrics = validate(
        model, dataloaders['test'], criterion, device, data_loader, threshold
    )
    logger.info(f"测试损失: {test_loss:.6f}")
    print_metrics(test_metrics, prefix="Test")
    
    # 计算相对误差指标（用于评估指标可视化）
    logger.info("\n计算相对误差指标...")
    # 需要获取所有测试数据来计算
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
    test_predictions = data_loader.denormalize_predictions(test_predictions)
    test_targets = data_loader.denormalize_predictions(test_targets)
    
    # 计算相对误差指标
    relative_metrics = compute_relative_error_metrics(test_targets, test_predictions)
    test_metrics.update(relative_metrics)
    
    # 生成评估指标图像到实验目录
    logger.info("\n生成评估指标图像...")
    plot_evaluation_metrics(
        test_metrics,
        output_dir=exp_manager.exp_dir,
        appliance=config['data']['appliance'],
        y_true=test_targets,
        y_pred=test_predictions
    )
    logger.info(f"✓ 评估指标图像已保存到: {exp_manager.exp_dir}")
    
    # 打印实验目录结构
    logger.info("\n✓ 训练完成！")
    exp_manager.print_structure()


def main():
    parser = argparse.ArgumentParser(
        description='训练Seq2Point NILM模型',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 配置参数
    parser.add_argument('--config',type=str,default='configs/config.yaml',help='配置文件路径')
    parser.add_argument('--resume',type=str,default=None,help='要恢复的检查点路径（支持 runs/expN、expN 或具体 .pth 文件）')
    
    # 数据参数
    parser.add_argument('--data-path',type=str,default=None,help='HDF5数据文件路径')
    parser.add_argument('--appliance',type=str,default=None,
        choices=['fridge', 'microwave', 'dishwasher', 'washing_machine', 'kettle', 
                 'washer_dryer', 'lighting', 'electric_heat', 'stove'],
        help='目标电器名称')
    parser.add_argument('--window-size',type=int,default=None,help='输入序列窗口大小')
    parser.add_argument('--window-stride',type=int,default=None,help='滑动窗口步长')
    
    # 训练参数
    parser.add_argument('--mode',type=str,default=None,choices=['single', 'parallel'],help='训练模式')
    parser.add_argument('--epochs',type=int,default=None,help='训练轮次')
    parser.add_argument('--batch-size',type=int,default=None,help='批次大小')
    parser.add_argument('--learning-rate', '--lr',type=float,default=None,help='初始学习率')
    parser.add_argument('--optimizer',type=str,default=None,choices=['adam', 'sgd', 'rmsprop'],help='优化器类型')
    parser.add_argument('--patience',type=int,default=None,help='早停耐心值（轮次）')
    parser.add_argument('--dropout',type=float,default=None,help='Dropout比率')
    
    # 设备参数
    parser.add_argument('--device',type=str,default=None,help='使用的设备: cpu 或 GPU ID (0, 1 等)')
    parser.add_argument('--workers',type=int,default=None,help='数据加载工作进程数')
    
    # 其他参数
    parser.add_argument('--seed',type=int, default=None,help='随机种子')
    parser.add_argument('--save-dir',type=str,default=None,help='保存日志和检查点的目录')
    
    args = parser.parse_args()
    
    # 将args转换为字典并过滤None值
    kwargs = {k: v for k, v in vars(args).items() 
              if v is not None and k not in ['config', 'resume']}
    
    resolved_resume = resolve_resume_path(args.resume) if args.resume else None
    if args.resume and resolved_resume != args.resume:
        print(f"Info: --resume 已解析为检查点文件: {resolved_resume}")

    resolved_config = infer_config_from_resume(args.config, resolved_resume)
    if resolved_config != args.config:
        print(
            f"Info: 检测到恢复训练，自动切换配置为: {resolved_config} "
            f"(原始: {args.config})"
        )

    cleanup_data_path = None
    try:
        preview_config = load_config(resolved_config)
        cleanup_data_path = preview_config.get('data', {}).get('data_path')
    except Exception:
        cleanup_data_path = None

    try:
        train(resolved_config, resolved_resume, **kwargs)
    finally:
        # 防止 Ctrl+C 或异常时遗留 nilmtk 临时 HDF5 句柄
        NILMDataLoader._cleanup_nilmtk_temp_handles(cleanup_data_path)


if __name__ == '__main__':
    main()

