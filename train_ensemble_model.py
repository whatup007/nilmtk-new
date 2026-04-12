"""
使用融合数据训练第三个网络（Transformer）
这是堆叠集成学习的第二层

数据处理说明：
- 输入：原始聚合功率序列 (N, 599)
- 训练/验证目标：融合预测值 (N,)
- 测试目标：真实电器功率 (N,)
所有数据形式统一，保持一致性

改进说明：
- 自动检测并生成融合数据（如果不存在）
- 从配置文件读取模型选取和融合方法
- 支持缓存机制，避免重复生成数据
"""

import os
import sys
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models import get_model
from utils.config_utils import load_config, get_appliance_params
from utils.metrics import compute_metrics, print_metrics, EarlyStopping, compute_composite_score, compute_relative_error_metrics
from utils.logger import setup_logger, TrainingLogger
from utils.plot_styles import plot_evaluation_metrics
from utils.exp_manager import init_experiment
from train import set_seed, create_optimizer

# 导入数据融合功能
from generate_ensemble_dataset import generate_ensemble_dataset


class EnsembleDataset(torch.utils.data.Dataset):
    """融合数据集 - 原始聚合功率序列作为输入"""

    def __init__(self, inputs: np.ndarray, targets: np.ndarray):
        """
        Args:
            inputs: 原始聚合功率序列，形状 (N, 599)
            targets: 目标值，形状 (N,)
        """
        # inputs 应该是 (N, 599)
        self.inputs = torch.FloatTensor(inputs)   # (N, 599)
        self.targets = torch.FloatTensor(targets) # (N,)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


class SimpleEnsembleModel(nn.Module):
    """用于 stack 融合的简单 FC 网络（支持配置化）"""

    def __init__(self, input_size: int = 2, output_size: int = 1,
                 hidden_layers: list = None, dropout_rate: float = 0.2,
                 activation: str = 'relu', use_batchnorm: bool = True):
        """
        Args:
            input_size: 输入维度（默认2，对应两个模型的预测）
            output_size: 输出维度（默认1，最终功率预测）
            hidden_layers: 隐藏层维度列表，例如 [64, 32] 表示两个隐藏层
            dropout_rate: Dropout比率
            activation: 激活函数类型 ('relu', 'tanh', 'sigmoid', 'leaky_relu')
            use_batchnorm: 是否使用 Batch Normalization
        """
        super().__init__()

        # 默认隐藏层配置
        if hidden_layers is None:
            hidden_layers = [64, 32]

        # 选择激活函数
        activation_map = {
            'relu': nn.ReLU,
            'tanh': nn.Tanh,
            'sigmoid': nn.Sigmoid,
            'leaky_relu': lambda: nn.LeakyReLU(0.2)
        }
        if activation.lower() not in activation_map:
            raise ValueError(f"不支持的激活函数: {activation}. 可选: {list(activation_map.keys())}")

        # 动态构建网络层
        layers = []
        in_dim = input_size

        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(in_dim, hidden_dim))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(activation_map[activation.lower()]())  # 创建激活函数实例
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            in_dim = hidden_dim

        # 输出层（不加激活函数、BN 和 Dropout）
        layers.append(nn.Linear(in_dim, output_size))

        self.fc = nn.Sequential(*layers)

    def forward(self, x):
        # x: (batch_size, input_size) -> (batch_size, output_size)
        out = self.fc(x)
        # 确保输出是 (batch_size,) 形状
        if out.shape[-1] == 1:
            out = out.squeeze(-1)
        return out


def load_ensemble_data(npz_path: str) -> dict:
    """加载融合数据 - 根据融合方法自动适配数据格式"""
    data = np.load(npz_path, allow_pickle=True)

    # 获取融合方法
    fusion_method = str(data.get('fusion_method', 'average'))

    result = {
        'train_inputs': data['train_inputs'],
        'train_targets': data['train_targets'],
        'val_inputs': data['val_inputs'],
        'val_targets': data['val_targets'],
        'test_inputs': data['test_inputs'],
        'test_targets': data['test_targets'],
        'fusion_method': fusion_method
    }

    # 验证数据格式
    if fusion_method == 'stack':
        # stack 方法：输入是堆叠的预测 (N, 2)，目标是真实值 (N,)
        assert result['train_inputs'].shape[1] == 2, \
            f"stack 方法的输入应该是 (N, 2)，实际是 {result['train_inputs'].shape}"
        assert result['train_targets'].ndim == 1, \
            f"stack 方法的目标应该是 (N,)，实际形状是 {result['train_targets'].shape}"
    else:
        # average/weighted_average: 输入是原始数据 (N, 599)，目标是融合预测 (N,)
        for key in ['train_inputs', 'val_inputs', 'test_inputs']:
            assert result[key].ndim == 2 and result[key].shape[1] == 599, \
                f"{key} 应该是 (N, 599) 的形状，实际是 {result[key].shape}"

    return result


def generate_predictions_from_original(
    model: nn.Module,
    inputs: torch.Tensor,
    device: torch.device,
    batch_size: int = 2048
) -> np.ndarray:
    """
    用模型对原始输入进行预测

    Args:
        model: 训练好的模型
        inputs: 原始输入，形状 (N, 599)
        device: 计算设备
        batch_size: 批次大小

    Returns:
        预测数组，形状 (N,)
    """
    model.eval()
    predictions = []

    with torch.no_grad():
        for i in range(0, len(inputs), batch_size):
            batch = inputs[i:i+batch_size].to(device)
            outputs = model(batch)
            predictions.append(outputs.cpu().numpy())

    predictions = np.concatenate(predictions)
    return predictions.flatten()


def load_and_predict_with_original_test(
    test_inputs: np.ndarray,
    test_targets: np.ndarray,
    exp30_checkpoint: str,
    exp33_checkpoint: str,
    appliance: str,
    config_path: str,
    device: torch.device,
    fusion_method: str = 'average',
    batch_size: int = 2048,
    logger = None
) -> tuple:
    """
    用 exp30 和 exp33 对原始测试数据进行预测，然后融合

    Args:
        test_inputs: 原始测试数据 (N, 599)
        test_targets: 测试目标值 (N,)
        exp30_checkpoint: exp30 模型检查点路径
        exp33_checkpoint: exp33 模型检查点路径
        appliance: 电器名称
        config_path: 配置文件路径
        device: 计算设备
        fusion_method: 融合方法 ('average', 'weighted_average', 'stack')
        batch_size: 批次大小
        logger: 日志记录器

    Returns:
        融合预测 (N, 1)，目标值 (N,)
    """
    if logger is None:
        import logging
        logger = logging.getLogger('ensemble_training')

    config = load_config(config_path)

    # 加载 exp30 和 exp33 的模型
    logger.info("加载 exp30 和 exp33 的模型进行原始测试数据预测...")

    # exp30: Seq2Point
    logger.info("  加载 exp30 (Seq2Point)...")
    checkpoint30 = torch.load(exp30_checkpoint, map_location=device, weights_only=False)
    model30 = get_model(
        model_name='seq2point',
        input_size=config['model']['input_size']
    ).to(device)
    model30.load_state_dict(checkpoint30['model_state_dict'])

    # exp33: LSTM
    logger.info("  加载 exp33 (LSTM)...")
    checkpoint33 = torch.load(exp33_checkpoint, map_location=device, weights_only=False)
    model33 = get_model(
        model_name='seq2point_lstm',
        input_size=config['model']['input_size']
    ).to(device)
    model33.load_state_dict(checkpoint33['model_state_dict'])

    # 对原始测试数据进行预测
    logger.info("对原始测试数据进行预测...")
    test_inputs_tensor = torch.FloatTensor(test_inputs)

    pred30 = generate_predictions_from_original(model30, test_inputs_tensor, device, batch_size)
    pred33 = generate_predictions_from_original(model33, test_inputs_tensor, device, batch_size)

    logger.info(f"  exp30 预测形状: {pred30.shape}")
    logger.info(f"  exp33 预测形状: {pred33.shape}")

    # 融合预测
    if fusion_method == 'average':
        pred_fused = (pred30 + pred33) / 2.0
    elif fusion_method == 'weighted_average':
        w1, w2 = 0.5, 0.5
        pred_fused = w1 * pred30 + w2 * pred33
    elif fusion_method == 'stack':
        pred_fused = np.column_stack([pred30, pred33])
    else:
        raise ValueError(f"未知的融合方法: {fusion_method}")

    logger.info(f"融合预测形状: {pred_fused.shape}")

    # 确保是 (N, 1) 形状
    if pred_fused.ndim == 1:
        pred_fused = pred_fused.reshape(-1, 1)

    return pred_fused, test_targets


def train_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    scaler: GradScaler = None
) -> float:
    """训练一个 epoch"""
    model.train()
    total_loss = 0.0

    with tqdm(dataloader, desc='Training', leave=False) as pbar:
        for inputs, targets in pbar:
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()

            if scaler is not None:
                # 混合精度训练
                with autocast('cuda'):
                    outputs = model(inputs)
                    if outputs.ndim > targets.ndim:
                        outputs = outputs.squeeze(-1)
                    loss = criterion(outputs, targets)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # 标准训练
                outputs = model(inputs)
                if outputs.ndim > targets.ndim:
                    outputs = outputs.squeeze(-1)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})

    return total_loss / len(dataloader)


def validate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    max_power: float = None,
    threshold: float = None
) -> tuple:
    """
    验证模型

    Args:
        model: 模型
        dataloader: 数据加载器
        criterion: 损失函数
        device: 计算设备
        max_power: 最大功率（用于反归一化）
        threshold: 功率阈值（用于状态识别）
    """
    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)

            # 确保输出和目标形状一致
            if outputs.ndim > targets.ndim:
                outputs = outputs.squeeze(-1)

            loss = criterion(outputs, targets)
            total_loss += loss.item()

            all_predictions.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    predictions = np.concatenate(all_predictions).flatten()
    targets = np.concatenate(all_targets).flatten()

    # 如果提供了 max_power，则反归一化后再计算指标
    if max_power is not None and max_power > 1.0:
        predictions_denorm = predictions * max_power
        targets_denorm = targets * max_power
        # 使用反归一化后的值计算指标
        metrics = compute_metrics(
            targets_denorm,
            predictions_denorm,
            threshold=threshold if threshold is not None else 10.0
        )
    else:
        # 数据未归一化或无法确定，使用默认阈值
        metrics = compute_metrics(
            targets,
            predictions,
            threshold=threshold if threshold is not None else 10.0
        )

    avg_loss = total_loss / len(dataloader)

    return avg_loss, metrics


def generate_stacked_test_predictions(
    test_inputs: np.ndarray,
    model1_checkpoint: str,
    model2_checkpoint: str,
    config: dict,
    device: torch.device,
    logger
) -> np.ndarray:
    """
    为 stack 方法生成测试集的堆叠预测

    Args:
        test_inputs: 原始测试输入 (N, 599)
        model1_checkpoint: 模型1检查点路径
        model2_checkpoint: 模型2检查点路径
        config: 配置字典
        device: 计算设备
        logger: 日志记录器

    Returns:
        堆叠的预测 (N, 2)
    """
    logger.info("  加载模型1进行预测...")
    checkpoint1 = torch.load(model1_checkpoint, map_location=device, weights_only=False)
    model1 = get_model(
        model_name='seq2point',
        input_size=config['model']['input_size']
    ).to(device)
    model1.load_state_dict(checkpoint1['model_state_dict'])

    logger.info("  加载模型2进行预测...")
    checkpoint2 = torch.load(model2_checkpoint, map_location=device, weights_only=False)
    model2 = get_model(
        model_name='seq2point_lstm',
        input_size=config['model']['input_size']
    ).to(device)
    model2.load_state_dict(checkpoint2['model_state_dict'])

    # 生成预测
    test_inputs_tensor = torch.FloatTensor(test_inputs)
    pred1 = generate_predictions_from_original(model1, test_inputs_tensor, device, batch_size=2048)
    pred2 = generate_predictions_from_original(model2, test_inputs_tensor, device, batch_size=2048)

    # 堆叠预测
    stacked_preds = np.column_stack([pred1, pred2])
    logger.info(f"  ✓ 测试集堆叠预测形状: {stacked_preds.shape}")

    return stacked_preds


def prepare_ensemble_data(config: dict, appliance: str, logger) -> str:
    """
    准备融合数据，自动检测并生成（如果不存在）

    Args:
        config: 配置字典
        appliance: 电器名称
        logger: 日志记录器

    Returns:
        融合数据的 npz 文件路径
    """
    ensemble_config = config.get('ensemble', {})

    # 获取融合方法和缓存目录
    fusion_method = ensemble_config.get('fusion_method', 'average')
    cache_dir = ensemble_config.get('cache_dir', 'ensemble_data')
    use_cache = ensemble_config.get('use_cache', True)

    # 获取模型检查点路径
    model1_checkpoint = ensemble_config.get('model1', {}).get('checkpoint', 'runs/exp30/weights/best_metrics.pth')
    model2_checkpoint = ensemble_config.get('model2', {}).get('checkpoint', 'runs/exp33/weights/best_metrics.pth')

    # 构建 npz 文件路径
    npz_filename = f'ensemble_dataset_{appliance}_{fusion_method}.npz'
    npz_path = os.path.join(cache_dir, npz_filename)

    # 检查缓存
    if use_cache and os.path.exists(npz_path):
        logger.info(f"✓ 发现缓存的融合数据: {npz_path}")
        logger.info("  如需重新生成，请设置 ensemble.use_cache = False")
        return npz_path

    # 生成融合数据
    logger.info("\n" + "="*70)
    logger.info("🔄 自动生成融合数据...")
    logger.info("="*70)
    logger.info(f"模型1: {ensemble_config.get('model1', {}).get('alias', 'Model1')} - {model1_checkpoint}")
    logger.info(f"模型2: {ensemble_config.get('model2', {}).get('alias', 'Model2')} - {model2_checkpoint}")
    logger.info(f"融合方法: {fusion_method}")
    logger.info(f"输出目录: {cache_dir}")

    # 检查检查点是否存在
    if not os.path.exists(model1_checkpoint):
        raise FileNotFoundError(f"模型1检查点未找到: {model1_checkpoint}")
    if not os.path.exists(model2_checkpoint):
        raise FileNotFoundError(f"模型2检查点未找到: {model2_checkpoint}")

    # 调用数据融合函数
    try:
        npz_path = generate_ensemble_dataset(
            exp30_checkpoint=model1_checkpoint,
            exp33_checkpoint=model2_checkpoint,
            config_path='configs/config.yaml',  # 使用原始配置文件进行数据加载
            appliance=appliance,
            output_dir=cache_dir,
            fusion_method=fusion_method
        )
        logger.info(f"✓ 融合数据生成完成: {npz_path}")
    except Exception as e:
        logger.error(f"❌ 融合数据生成失败: {str(e)}")
        raise

    logger.info("="*70 + "\n")
    return npz_path


def train_ensemble_model(
    config_path: str,
    appliance: str,
    model_type: str = 'transformer',
    batch_size: int = None,
    use_amp: bool = True,
    seed: int = None,
    npz_path: str = None,  # 可选，手动指定融合数据路径
    **kwargs
):
    """
    训练第二层集成模型

    Args:
        config_path: 配置文件路径
        appliance: 电器名称
        model_type: 模型类型 ('transformer')
        batch_size: 批次大小（可选，覆盖配置文件）
        use_amp: 是否使用混合精度训练
        seed: 随机种子
        npz_path: 融合数据的 npz 文件路径（可选，默认自动生成）
    """
    # 初始化实验管理器
    exp_manager = init_experiment(base_dir='runs')
    exp_dir = exp_manager.get_new_exp_dir()
    logger_path = os.path.join(exp_dir, 'logs')
    os.makedirs(logger_path, exist_ok=True)

    logger = setup_logger('ensemble_training', logger_path)
    logger.info(f"配置文件: {config_path}")
    logger.info(f"电器名称: {appliance}")
    logger.info(f"模型类型: {model_type}")

    # 加载配置
    config = load_config(config_path)
    
    # 确定电器名称：优先级 命令行参数 > 配置文件
    if appliance is None:
        appliance = config['data'].get('appliance')
        if appliance is None:
            raise ValueError("必须通过 --appliance 指定电器名称，或在配置文件中设置 data.appliance")
    
    config['data']['appliance'] = appliance
    logger.info(f"最终确定的电器名称: {appliance}")

    # 自动生成或加载融合数据
    if npz_path is None:
        logger.info("\n未指定融合数据路径，自动检测并生成...")
        npz_path = prepare_ensemble_data(config, appliance, logger)
    else:
        logger.info(f"\n使用指定的融合数据: {npz_path}")
        if not os.path.exists(npz_path):
            raise FileNotFoundError(f"融合数据文件未找到: {npz_path}")

    logger.info(f"✓ 融合数据路径: {npz_path}")

    # 设置随机种子
    if seed is not None:
        config['seed'] = seed
    set_seed(config.get('seed', 42))

    # 覆盖配置参数
    if batch_size is not None:
        config['training']['batch_size'] = batch_size
    else:
        batch_size = config['training']['batch_size']

    # 保存配置文件
    exp_manager.save_config(config, 'config.yaml')
    logger.info(f"配置保存在: {exp_dir}/config.yaml")

    logger.info("加载融合数据...")
    ensemble_data = load_ensemble_data(npz_path)
    fusion_method = ensemble_data['fusion_method']

    logger.info(f"✓ 融合方法: {fusion_method}")

    # 设置设备
    if torch.cuda.is_available() and config['device'].get('gpu', True):
        device = torch.device(f'cuda:{config["device"].get("gpu_id", 0)}')
        logger.info("使用 GPU")
    else:
        device = torch.device('cpu')
        logger.info("使用 CPU")

    # 加载数据
    train_inputs = ensemble_data['train_inputs']  # (N, 599) 原始聚合功率
    train_targets = ensemble_data['train_targets']
    val_inputs = ensemble_data['val_inputs']      # (N, 599) 原始聚合功率
    val_targets = ensemble_data['val_targets']
    test_inputs = ensemble_data['test_inputs']    # (N, 599) 原始聚合功率
    test_targets = ensemble_data['test_targets']

    logger.info(f"\n数据形状:")
    logger.info(f"  训练集: {train_inputs.shape} → {train_targets.shape}")
    logger.info(f"  验证集: {val_inputs.shape} → {val_targets.shape}")
    logger.info(f"  测试集: {test_inputs.shape} → {test_targets.shape}")
    logger.info(f"✓ 所有数据形式一致，都是原始聚合功率序列 (N, 599)")

    # 获取电器参数（用于反归一化和状态识别阈值）
    logger.info("\n加载电器参数...")
    appliance_params = get_appliance_params(appliance, 'configs/appliance_params.yaml')
    max_power = appliance_params['max_power']
    threshold = appliance_params['threshold']
    logger.info(f"  电器: {appliance}")
    logger.info(f"  最大功率: {max_power}W")
    logger.info(f"  状态识别阈值: {threshold}W")

    # 创建数据加载器
    train_dataset = EnsembleDataset(train_inputs, train_targets)
    val_dataset = EnsembleDataset(val_inputs, val_targets)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size,
        shuffle=True, num_workers=config['device']['num_workers']
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size,
        shuffle=False, num_workers=config['device']['num_workers']
    )

    # 获取输入信息
    input_size = train_inputs.shape[1]
    logger.info(f"\n网络参数:")
    logger.info(f"  输入大小: {input_size}")
    logger.info(f"  融合方法: {fusion_method}")

    # 创建模型
    logger.info(f"\n创建模型...")

    # 根据融合方法和输入维度选择模型
    if fusion_method == 'stack' or input_size == 2:
        # stack 方法：使用简单的 FC 网络学习组合两个预测
        logger.info("  使用简单的 FC 网络（适用于 stack 融合）")

        # 从配置文件读取全连接网络参数
        fc_config = config['model'].get('fc_network', {})
        hidden_layers = fc_config.get('hidden_layers', [64, 32])
        dropout_rate = fc_config.get('dropout_rate', 0.2)
        activation = fc_config.get('activation', 'relu')
        use_batchnorm = fc_config.get('use_batchnorm', True)

        model = SimpleEnsembleModel(
            input_size=2,
            output_size=1,
            hidden_layers=hidden_layers,
            dropout_rate=dropout_rate,
            activation=activation,
            use_batchnorm=use_batchnorm
        ).to(device)

        logger.info(f"  网络结构:")
        logger.info(f"    输入维度: 2")
        logger.info(f"    隐藏层: {hidden_layers}")
        logger.info(f"    输出维度: 1")
        logger.info(f"    Dropout: {dropout_rate}")
        logger.info(f"    Batch Normalization: {use_batchnorm}")
        logger.info(f"    激活函数: {activation}")
    elif model_type == 'transformer':
        # 从配置中获取 Transformer 参数
        transformer_config = config['model'].get('transformer', {})
        model = get_model(
            model_name='seq2point_transformer',
            input_size=input_size,
            d_model=transformer_config.get('d_model', 64),
            nhead=transformer_config.get('nhead', 4),
            num_layers=transformer_config.get('num_layers', 3),
            dim_feedforward=transformer_config.get('dim_feedforward', 128),
            dropout_rate=transformer_config.get('dropout_rate', 0.2)
        ).to(device)

        logger.info(f"  Transformer 参数:")
        logger.info(f"    d_model: {transformer_config.get('d_model', 64)}")
        logger.info(f"    nhead: {transformer_config.get('nhead', 4)}")
        logger.info(f"    num_layers: {transformer_config.get('num_layers', 3)}")
        logger.info(f"    dim_feedforward: {transformer_config.get('dim_feedforward', 128)}")
        logger.info(f"    dropout_rate: {transformer_config.get('dropout_rate', 0.2)}")
    else:
        raise ValueError(f"未知的模型类型: {model_type}")

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"  模型参数数量: {num_params:,}")

    # 损失函数和优化器
    criterion = nn.MSELoss()
    optimizer_name = config['training'].get('optimizer', 'adam').lower()
    optimizer = create_optimizer(
        optimizer_name,
        model.parameters(),
        config['training']['learning_rate']
    )

    # 学习率调度器
    scheduler = None
    if config['training'].get('lr_schedule', True):
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=config['training'].get('lr_decay_factor', 0.5),
            patience=config['training'].get('lr_decay_patience', 5)
        )

    # 早停
    early_stopping = None
    if config['training'].get('early_stopping', True):
        early_stopping = EarlyStopping(
            patience=config['training'].get('patience', 10),
            mode='min'
        )

    # 训练日志记录器
    training_logger = TrainingLogger(
        log_dir=logger_path,
        experiment_name=f"{appliance}_ensemble_{model_type}"
    )

    logger.info("\n开始训练...")
    logger.info(f"总轮数: {config['training']['epochs']}")
    logger.info(f"批次大小: {batch_size}")
    logger.info(f"学习率: {config['training']['learning_rate']}")
    logger.info(f"优化器: {optimizer_name}")
    logger.info(f"混合精度训练: {use_amp}")

    start_epoch = 0
    best_val_loss = float('inf')
    best_metrics_score = -float('inf')

    # 初始化混合精度缩放器
    scaler = GradScaler('cuda') if (use_amp and torch.cuda.is_available()) else None

    # 训练循环
    for epoch in range(start_epoch, config['training']['epochs']):
        logger.info(f"\nEpoch {epoch + 1}/{config['training']['epochs']}")

        # 训练
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, device, scaler
        )
        logger.info(f"训练损失: {train_loss:.6f}")

        # 验证
        val_loss, val_metrics = validate(
            model, val_loader, criterion, device, max_power, threshold
        )
        logger.info(f"验证损失: {val_loss:.6f}")
        print_metrics(val_metrics, prefix="Validation")

        # 记录学习率和指标
        current_lr = optimizer.param_groups[0]['lr']
        training_logger.log_epoch(
            epoch=epoch + 1,
            train_loss=train_loss,
            val_loss=val_loss,
            val_metrics=val_metrics,
            learning_rate=current_lr
        )

        # 学习率衰减
        if scheduler:
            scheduler.step(val_loss)

        # 检查最佳验证损失
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            logger.info(f"新的最佳模型！验证损失: {val_loss:.6f}")

        # 检查最佳综合指标
        current_composite_score = compute_composite_score(val_metrics)
        is_best_metrics = current_composite_score > best_metrics_score
        if is_best_metrics:
            best_metrics_score = current_composite_score
            logger.info(
                f"最佳综合指标！综合评分: {current_composite_score:.6f} "
                f"(MAE: {val_metrics['mae']:.4f}, RMSE: {val_metrics['rmse']:.4f}, "
                f"RAE: {val_metrics.get('rae', 0.0):.4f}, "
                f"F1: {val_metrics['f1']:.4f}, R²: {val_metrics['r2_score']:.4f})"
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

        # 定期保存训练曲线
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

    logger.info("\n训练完成！")

    # 保存训练历史和可视化
    logger.info("\n保存训练曲线...")
    training_logger.save_history()
    training_logger.plot_history()
    training_logger.plot_metrics(metrics_to_plot=['mae', 'rmse', 'f1'])

    # 复制可视化图像到实验目录
    logger.info("复制可视化图像到实验目录...")
    log_dir = training_logger.log_dir
    for file in os.listdir(log_dir):
        if file.endswith('.png'):
            src_path = os.path.join(log_dir, file)
            exp_manager.save_visualization(src_path, file)

    logger.info("✓ 训练曲线已保存")

    # 测试集评估和可视化
    logger.info("\n在测试集上进行评估...")

    # 为 stack 方法生成测试集的堆叠预测
    if fusion_method == 'stack':
        logger.info("  为 stack 方法生成测试集预测...")
        # 需要先用两个模型对测试集进行预测，然后堆叠
        ensemble_config = config.get('ensemble', {})
        model1_checkpoint = ensemble_config.get('model1', {}).get('checkpoint')
        model2_checkpoint = ensemble_config.get('model2', {}).get('checkpoint')

        # 加载并预测
        test_inputs_stacked = generate_stacked_test_predictions(
            test_inputs, model1_checkpoint, model2_checkpoint,
            config, device, logger
        )

        # 创建测试数据加载器
        test_dataset = EnsembleDataset(test_inputs_stacked, test_targets)
    else:
        # average/weighted_average: 直接使用原始测试数据
        test_dataset = EnsembleDataset(test_inputs, test_targets)

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size,
        shuffle=False, num_workers=config['device']['num_workers']
    )

    # 获取测试集预测值
    logger.info("\n生成测试集预测结果...")
    test_predictions = []
    with torch.no_grad():
        for inputs, _ in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            test_predictions.append(outputs.cpu().numpy())

    test_predictions = np.concatenate(test_predictions).flatten()
    test_targets_flat = test_targets.flatten()

    # 反归一化预测值和目标值
    logger.info("\n反归一化测试集预测和目标...")
    # 使用已在训练开始时加载的电器参数
    test_max_power = appliance_params['max_power']
    test_threshold = appliance_params['threshold']

    # 反归一化：从 [0, 1] → [0, max_power]
    test_predictions_denorm = test_predictions * test_max_power
    test_targets_denorm = test_targets_flat * test_max_power

    logger.info(f"  max_power: {test_max_power}W")
    logger.info(f"  threshold: {test_threshold}W")
    logger.info(f"  预测值范围（反归一化前）: [{test_predictions.min():.4f}, {test_predictions.max():.4f}]")
    logger.info(f"  预测值范围（反归一化后）: [{test_predictions_denorm.min():.2f}, {test_predictions_denorm.max():.2f}] W")
    logger.info(f"  目标值范围（反归一化后）: [{test_targets_denorm.min():.2f}, {test_targets_denorm.max():.2f}] W")

    # 使用反归一化后的值重新计算指标
    logger.info("\n使用反归一化后的值计算测试集指标...")
    test_metrics = compute_metrics(test_targets_denorm, test_predictions_denorm, threshold=test_threshold)

    logger.info(f"✓ 测试损失: {np.mean((test_targets_denorm - test_predictions_denorm)**2):.2f} (MSE)")
    print_metrics(test_metrics, prefix="Test")

    # 生成评估指标图像（使用反归一化后的值）
    logger.info("\n生成评估指标可视化...")
    plot_evaluation_metrics(
        test_metrics,
        output_dir=exp_manager.exp_dir,
        appliance=appliance,
        y_true=test_targets_denorm,
        y_pred=test_predictions_denorm
    )
    logger.info(f"✓ 评估指标图像已保存到: {exp_manager.exp_dir}")

    # 打印实验目录结构
    logger.info("\n✓ 训练完成！")
    exp_manager.print_structure()

    return exp_dir


def main():
    parser = argparse.ArgumentParser(
        description='训练第二层集成模型（自动生成融合数据）',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="""
使用示例：
  # 基础用法（自动生成融合数据）
  python train_ensemble_model.py --appliance kettle

  # 手动指定融合数据
  python train_ensemble_model.py --appliance kettle --npz-path ensemble_data/ensemble_dataset_kettle_average.npz

  # 自定义配置
  python train_ensemble_model.py --appliance fridge --config configs/config_3.yaml --seed 42
"""
    )

    parser.add_argument('--config', type=str, default='configs/config_3.yaml',
                       help='配置文件路径（集成学习推荐使用 config_3.yaml）')
    parser.add_argument('--appliance', type=str, default=None,
                       help='电器名称')
    parser.add_argument('--npz-path', type=str, default=None,
                       help='融合数据的 npz 文件路径（可选，默认自动生成）')
    parser.add_argument('--model-type', type=str, default='transformer',
                       choices=['transformer'],
                       help='模型类型（仅支持 transformer）')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='批次大小（默认使用配置文件中的值）')
    parser.add_argument('--no-amp', dest='use_amp', action='store_false', default=True,
                       help='禁用混合精度训练')
    parser.add_argument('--seed', type=int, default=None,
                       help='随机种子')

    args = parser.parse_args()

    exp_dir = train_ensemble_model(
        config_path=args.config,
        appliance=args.appliance,
        model_type=args.model_type,
        batch_size=args.batch_size,
        use_amp=args.use_amp,
        seed=args.seed,
        npz_path=args.npz_path
    )

    print(f"\n✅ 训练完成！\n实验目录: {exp_dir}")


if __name__ == '__main__':
    main()
