"""
NILM 评估指标计算。
"""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from typing import Dict


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = 10.0
) -> Dict[str, float]:
    """
    计算 NILM 评估指标。
    
    Args:
        y_true: 真实值
        y_pred: 预测值
        threshold: 开关状态阈值
        
    Returns:
        指标字典
    """
    # 确保预测值非负
    y_pred = np.maximum(y_pred, 0)
    
    # MAE
    mae = mean_absolute_error(y_true, y_pred)
    
    # MSE
    mse = mean_squared_error(y_true, y_pred)
    
    # RMSE
    rmse = np.sqrt(mse)
    
    # 归一化指标
    y_true_mean = np.mean(y_true)
    if y_true_mean > 0:
        nmae = mae / y_true_mean
        nrmse = rmse / y_true_mean
    else:
        nmae = mae
        nrmse = rmse
    
    # 基于开关状态的指标
    true_states = (y_true > threshold).astype(int)
    pred_states = (y_pred > threshold).astype(int)
    
    # 准确率
    accuracy = np.mean(true_states == pred_states)
    
    # 精确率、召回率、F1
    tp = np.sum((true_states == 1) & (pred_states == 1))
    fp = np.sum((true_states == 0) & (pred_states == 1))
    fn = np.sum((true_states == 1) & (pred_states == 0))
    tn = np.sum((true_states == 0) & (pred_states == 0))
    
    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)
    
    # 能量相关指标
    total_energy_true = np.sum(y_true)
    total_energy_pred = np.sum(y_pred)
    
    if total_energy_true > 0:
        energy_accuracy = 1 - abs(total_energy_true - total_energy_pred) / total_energy_true
    else:
        energy_accuracy = 0.0
    
    # R² 得分（决定系数）
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true_mean) ** 2)
    r2_score = 1 - (ss_res / (ss_tot + 1e-10))
    
    metrics = {
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'nmae': nmae,
        'nrmse': nrmse,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'energy_accuracy': energy_accuracy,
        'r2_score': r2_score,
        'total_energy_true': total_energy_true,
        'total_energy_pred': total_energy_pred
    }
    
    return metrics


def compute_composite_score(
    metrics: Dict[str, float],
    mae_weight: float = 0.2,
    rmse_weight: float = 0.2,
    f1_weight: float = 0.3,
    r2_weight: float = 0.15,
    energy_weight: float = 0.15
) -> float:
    """
    计算综合评分，考虑多个指标的加权组合。
    
    综合评分考虑的指标：
    - MAE（越低越好）
    - RMSE（越低越好）
    - F1-Score（越高越好）
    - R²得分（越高越好）
    - 能量准确度（越高越好）
    
    Args:
        metrics: 指标字典
        mae_weight: MAE 的权重
        rmse_weight: RMSE 的权重
        f1_weight: F1-Score 的权重
        r2_weight: R²得分的权重
        energy_weight: 能量准确度的权重
        
    Returns:
        综合评分（0-1之间，越高越好）
    """
    # 标准化 MAE 和 RMSE（使用倒数映射）
    # 对于下降指标，使用 1/(1+指标) 进行标准化
    mae_normalized = 1 / (1 + metrics['mae'])
    rmse_normalized = 1 / (1 + metrics['rmse'])
    
    # 上升指标直接取值，并裁剪到[0,1]
    f1_normalized = np.clip(metrics['f1'], 0, 1)
    r2_normalized = np.clip(metrics['r2_score'], 0, 1)
    energy_normalized = np.clip(metrics['energy_accuracy'], 0, 1)
    
    # 加权组合
    composite_score = (
        mae_weight * mae_normalized +
        rmse_weight * rmse_normalized +
        f1_weight * f1_normalized +
        r2_weight * r2_normalized +
        energy_weight * energy_normalized
    )
    
    return composite_score


def print_metrics(metrics: Dict[str, float], prefix: str = ""):
    """
    以格式化方式打印指标。
    
    Args:
        metrics: 指标字典
        prefix: 输出前缀（如 'Train'/'Val'/'Test'）
    """
    print(f"\n{prefix} Metrics:")
    print("-" * 50)
    print(f"MAE:             {metrics['mae']:.4f}")
    print(f"MSE:             {metrics['mse']:.4f}")
    print(f"RMSE:            {metrics['rmse']:.4f}")
    print(f"NMAE:            {metrics['nmae']:.4f}")
    print(f"NRMSE:           {metrics['nrmse']:.4f}")
    print(f"Accuracy:        {metrics['accuracy']:.4f}")
    print(f"Precision:       {metrics['precision']:.4f}")
    print(f"Recall:          {metrics['recall']:.4f}")
    print(f"F1 Score:        {metrics['f1']:.4f}")
    print(f"R² Score:        {metrics.get('r2_score', 0.0):.4f}")
    print(f"Energy Accuracy: {metrics['energy_accuracy']:.4f}")
    
    # 计算并打印综合评分
    composite = compute_composite_score(metrics)
    print(f"综合评分:        {composite:.4f}")
    print("-" * 50)


def compute_relative_error_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    epsilon: float = 1e-10
) -> Dict[str, float]:
    """
    计算相对误差指标。
    
    Args:
        y_true: 真实值
        y_pred: 预测值
        epsilon: 防止除零的小量
        
    Returns:
        相对误差指标字典
    """
    # Relative Absolute Error (RAE)
    rae = np.sum(np.abs(y_true - y_pred)) / (np.sum(np.abs(y_true - np.mean(y_true))) + epsilon)
    
    # Relative Squared Error (RSE)
    rse = np.sum((y_true - y_pred) ** 2) / (np.sum((y_true - np.mean(y_true)) ** 2) + epsilon)
    
    # Mean Absolute Percentage Error (MAPE)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100
    
    return {
        'rae': rae,
        'rse': rse,
        'mape': mape
    }


class EarlyStopping:
    """早停机制：验证损失不再提升时终止训练。"""
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = 'min'
    ):
        """
        初始化早停器。
        
        Args:
            patience: 等待轮数
            min_delta: 最小改进幅度
            mode: 'min' 或 'max'
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_value = None
        self.early_stop = False
        
    def __call__(self, current_value: float) -> bool:
        """
        判断是否应提前停止训练。
        
        Args:
            current_value: 当前指标值
            
        Returns:
            是否触发早停
        """
        if self.best_value is None:
            self.best_value = current_value
            return False
        
        if self.mode == 'min':
            improved = current_value < (self.best_value - self.min_delta)
        else:
            improved = current_value > (self.best_value + self.min_delta)
        
        if improved:
            self.best_value = current_value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
        
        return False
    
    def reset(self):
        """重置早停状态。"""
        self.counter = 0
        self.best_value = None
        self.early_stop = False
