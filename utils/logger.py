"""
训练日志与可视化工具。
"""

import os
import json
import logging
import warnings
from datetime import datetime
from typing import Dict, Any
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib import font_manager
import numpy as np


def _setup_chinese_font_for_plots():
    """自动配置可用中文字体，避免绘图阶段缺字形告警。"""
    preferred_cjk_fonts = [
        'Noto Sans CJK SC',
        'Noto Sans SC',
        'Source Han Sans SC',
        'Source Han Sans CN',
        'WenQuanYi Zen Hei',
        'WenQuanYi Micro Hei',
        'Microsoft YaHei',
        'PingFang SC',
        'SimHei',
        'Arial Unicode MS',
    ]

    available_fonts = {f.name for f in font_manager.fontManager.ttflist}
    matched_fonts = [f for f in preferred_cjk_fonts if f in available_fonts]

    rcParams['font.sans-serif'] = matched_fonts + ['DejaVu Sans']
    rcParams['axes.unicode_minus'] = False

    if not matched_fonts:
        warnings.filterwarnings(
            'ignore',
            message=r'Glyph .* missing from font\(s\) DejaVu Sans\.',
            category=UserWarning,
        )


_setup_chinese_font_for_plots()


class NumpyEncoder(json.JSONEncoder):
    """自定义 JSON encoder，支持 NumPy 类型"""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def setup_logger(
    name: str,
    log_dir: str = 'logs',
    log_file: str = None
) -> logging.Logger:
    """
    创建并配置日志记录器。
    
    Args:
        name: 日志器名称
        log_dir: 日志目录
        log_file: 日志文件名（为空则使用时间戳）
        
    Returns:
        Logger 实例
    """
    os.makedirs(log_dir, exist_ok=True)
    
    if log_file is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = f'{name}_{timestamp}.log'
    
    log_path = os.path.join(log_dir, log_file)
    
    # 创建 logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # 清理旧的 handler
    logger.handlers = []
    
    # 文件 handler
    file_handler = logging.FileHandler(log_path, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    
    # 控制台 handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # 格式化器
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # 注册 handler
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


class TrainingLogger:
    """训练指标与历史记录器。"""
    
    def __init__(self, log_dir: str = 'logs', experiment_name: str = None):
        """
        初始化训练记录器。
        
        Args:
            log_dir: 日志目录
            experiment_name: 实验名称
        """
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        if experiment_name is None:
            experiment_name = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        self.experiment_name = experiment_name
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_metrics': [],
            'val_metrics': [],
            'learning_rates': []
        }
        
    def log_epoch(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        train_metrics: Dict[str, float] = None,
        val_metrics: Dict[str, float] = None,
        learning_rate: float = None
    ):
        """
        记录单轮训练指标。
        
        Args:
            epoch: 轮次
            train_loss: 训练损失
            val_loss: 验证损失
            train_metrics: 训练指标
            val_metrics: 验证指标
            learning_rate: 当前学习率
        """
        self.history['train_loss'].append(train_loss)
        self.history['val_loss'].append(val_loss)
        
        if train_metrics:
            self.history['train_metrics'].append(train_metrics)
        
        if val_metrics:
            self.history['val_metrics'].append(val_metrics)
        
        if learning_rate:
            self.history['learning_rates'].append(learning_rate)
    
    def save_history(self, filename: str = None):
        """
        将训练历史保存为 JSON 文件。
        
        Args:
            filename: 文件名（为空则使用实验名）
        """
        if filename is None:
            filename = f'{self.experiment_name}_history.json'
        
        save_path = os.path.join(self.log_dir, filename)
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(self.history, f, indent=4, cls=NumpyEncoder)
        
        print(f"Training history saved to {save_path}")
    
    def plot_history(self, save_path: str = None):
        """
        绘制训练曲线。
        
        Args:
            save_path: 保存路径（为空则使用 log_dir）
        """
        if save_path is None:
            save_path = os.path.join(self.log_dir, f'{self.experiment_name}_history.png')
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # 绘制损失曲线
        epochs = range(1, len(self.history['train_loss']) + 1)
        axes[0].plot(epochs, self.history['train_loss'], 'b-', label='Train Loss')
        axes[0].plot(epochs, self.history['val_loss'], 'r-', label='Val Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # 绘制学习率（如果有记录）
        if self.history['learning_rates']:
            axes[1].plot(epochs, self.history['learning_rates'], 'g-')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Learning Rate')
            axes[1].set_title('Learning Rate Schedule')
            axes[1].grid(True)
        else:
            axes[1].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        
        print(f"Training history plot saved to {save_path}")
    
    def plot_metrics(self, metrics_to_plot: list = ['mae', 'rmse'], save_path: str = None):
        """
        绘制指定指标随 epoch 的变化。
        
        Args:
            metrics_to_plot: 需要绘制的指标名称列表
            save_path: 保存路径
        """
        if not self.history['train_metrics'] or not self.history['val_metrics']:
            print("No metrics to plot")
            return
        
        if save_path is None:
            save_path = os.path.join(self.log_dir, f'{self.experiment_name}_metrics.png')
        
        n_metrics = len(metrics_to_plot)
        fig, axes = plt.subplots(1, n_metrics, figsize=(6 * n_metrics, 5))
        
        if n_metrics == 1:
            axes = [axes]
        
        epochs = range(1, len(self.history['train_metrics']) + 1)
        
        for i, metric in enumerate(metrics_to_plot):
            train_values = [m.get(metric, 0) for m in self.history['train_metrics']]
            val_values = [m.get(metric, 0) for m in self.history['val_metrics']]
            
            axes[i].plot(epochs, train_values, 'b-', label=f'Train {metric.upper()}')
            axes[i].plot(epochs, val_values, 'r-', label=f'Val {metric.upper()}')
            axes[i].set_xlabel('Epoch')
            axes[i].set_ylabel(metric.upper())
            axes[i].set_title(f'{metric.upper()} over Epochs')
            axes[i].legend()
            axes[i].grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        
        print(f"Metrics plot saved to {save_path}")


def save_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_dir: str = 'results',
    filename: str = 'predictions.npz'
):
    """
    保存预测值与真实值。
    
    Args:
        y_true: 真实值
        y_pred: 预测值
        save_dir: 保存目录
        filename: 文件名
    """
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)
    
    np.savez(save_path, y_true=y_true, y_pred=y_pred)
    print(f"Predictions saved to {save_path}")


def plot_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: str = 'results/predictions.png',
    num_samples: int = 1000,
    title: str = 'Predictions vs Ground Truth'
):
    """
    绘制预测值与真实值对比图。
    
    Args:
        y_true: 真实值
        y_pred: 预测值
        save_path: 保存路径
        num_samples: 绘制样本数
        title: 图标题
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 样本过多时进行抽样
    if len(y_true) > num_samples:
        indices = np.random.choice(len(y_true), num_samples, replace=False)
        indices = np.sort(indices)
        y_true_sample = y_true[indices]
        y_pred_sample = y_pred[indices]
    else:
        y_true_sample = y_true
        y_pred_sample = y_pred
    
    fig, axes = plt.subplots(2, 1, figsize=(15, 8))
    
    # 时间序列图
    x = np.arange(len(y_true_sample))
    axes[0].plot(x, y_true_sample, 'b-', label='Ground Truth', alpha=0.7)
    axes[0].plot(x, y_pred_sample, 'r-', label='Prediction', alpha=0.7)
    axes[0].set_xlabel('Sample')
    axes[0].set_ylabel('Power (W)')
    axes[0].set_title(title)
    axes[0].legend()
    axes[0].grid(True)
    
    # 散点图
    axes[1].scatter(y_true_sample, y_pred_sample, alpha=0.5, s=1)
    max_val = max(y_true_sample.max(), y_pred_sample.max())
    axes[1].plot([0, max_val], [0, max_val], 'r--', label='Perfect Prediction')
    axes[1].set_xlabel('Ground Truth (W)')
    axes[1].set_ylabel('Prediction (W)')
    axes[1].set_title('Prediction Scatter Plot')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    
    print(f"Predictions plot saved to {save_path}")


def plot_power_error_metrics(
    metrics: dict,
    save_path: str = 'results/01_power_error_metrics.png'
):
    """绘制功率预测误差指标（MAE、RMSE、NMAE）"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 平均绝对误差
    mae = metrics.get('mae', 0)
    ax = axes[0]
    ax.barh(['MAE'], [mae], color='#FF6B6B', edgecolor='black', linewidth=2, height=0.5)
    ax.text(mae/2, 0, f'{mae:.2f} W', ha='center', va='center', fontsize=14, fontweight='bold', color='white')
    ax.set_xlabel('数值 (W)', fontsize=11, fontweight='bold')
    ax.set_title('平均绝对误差\n(Mean Absolute Error)', fontsize=12, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    ax.set_xlim(0, max(mae * 1.2, 100))
    
    # 均方根误差
    rmse = metrics.get('rmse', 0)
    ax = axes[1]
    ax.barh(['RMSE'], [rmse], color='#4ECDC4', edgecolor='black', linewidth=2, height=0.5)
    ax.text(rmse/2, 0, f'{rmse:.2f} W', ha='center', va='center', fontsize=14, fontweight='bold', color='white')
    ax.set_xlabel('数值 (W)', fontsize=11, fontweight='bold')
    ax.set_title('均方根误差\n(Root Mean Square Error)', fontsize=12, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    ax.set_xlim(0, max(rmse * 1.2, 100))
    
    # 归一化平均绝对误差
    nmae = metrics.get('nmae', 0) * 100 if metrics.get('nmae', 0) < 1 else metrics.get('nmae', 0)
    ax = axes[2]
    ax.barh(['NMAE'], [nmae], color='#45B7D1', edgecolor='black', linewidth=2, height=0.5)
    ax.text(nmae/2, 0, f'{nmae:.2f}%', ha='center', va='center', fontsize=14, fontweight='bold', color='white')
    ax.set_xlabel('数值 (%)', fontsize=11, fontweight='bold')
    ax.set_title('归一化平均绝对误差\n(Normalized Mean Absolute Error)', fontsize=12, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    ax.set_xlim(0, max(nmae * 1.2, 50))
    
    plt.suptitle('功率预测误差指标 (Power Prediction Error Metrics)', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Power error metrics plot saved to {save_path}")


def plot_state_recognition_metrics(
    metrics: dict,
    save_path: str = 'results/02_state_recognition_metrics.png'
):
    """绘制状态识别指标（Accuracy、Precision、Recall、F1-Score）"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    metrics_data = [
        ('准确率\n(Accuracy)', metrics.get('accuracy', 0), '#95E1D3', axes[0, 0]),
        ('精确率\n(Precision)', metrics.get('precision', 0), '#F38181', axes[0, 1]),
        ('召回率\n(Recall)', metrics.get('recall', 0), '#AA96DA', axes[1, 0]),
        ('F1 得分\n(F1 Score)', metrics.get('f1', 0), '#FCBAD3', axes[1, 1]),
    ]
    
    for name, value, color, ax in metrics_data:
        # 背景条形
        ax.barh([name], [1], color='lightgray', edgecolor='black', linewidth=1, height=0.3, alpha=0.3)
        # 数值条形
        ax.barh([name], [value], color=color, edgecolor='black', linewidth=2, height=0.3)
        
        # 参考线
        ax.axvline(x=0.9, color='red', linestyle='--', linewidth=2, alpha=0.5, label='优秀 0.9')
        ax.axvline(x=0.8, color='orange', linestyle='--', linewidth=2, alpha=0.5, label='良好 0.8')
        
        # 数值标签
        ax.text(value + 0.02, 0, f'{value:.3f}', ha='left', va='center', fontsize=12, fontweight='bold')
        
        ax.set_xlim(0, 1.15)
        ax.set_xlabel('得分', fontsize=10, fontweight='bold')
        ax.set_title(f'{name}', fontsize=12, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
    
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.98, 0.98), fontsize=9)
    
    plt.suptitle('设备状态识别指标 (On/Off 检测)', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"State recognition metrics plot saved to {save_path}")


def plot_energy_metrics(
    metrics: dict,
    save_path: str = 'results/03_energy_metrics.png'
):
    """绘制能量相关指标"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 能量对比
    ax = axes[0]
    energy_true = metrics.get('total_energy_true', 0)
    energy_pred = metrics.get('total_energy_pred', 0)
    
    x = np.arange(2)
    width = 0.35
    bars1 = ax.bar(x - width/2, [energy_true, energy_pred], width, label=['真实值', '预测值'],
                   color=['#A8DADC', '#FF6B6B'], edgecolor='black', linewidth=2)
    
    ax.set_ylabel('能量 (Wh)', fontsize=11, fontweight='bold')
    ax.set_title('总能量对比', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['真实值', '预测值'])
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    
    # 添加数值
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.0f}',
               ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 能量准确率仪表盘
    ax = axes[1]
    energy_acc = metrics.get('energy_accuracy', 0)
    
    # 仪表盘背景
    theta = np.linspace(0, np.pi, 100)
    r = 1
    x_dash = r * np.cos(theta)
    y_dash = r * np.sin(theta)
    
    ax.fill_between(x_dash[:33], y_dash[:33], 0, alpha=0.3, color='red', label='差')
    ax.fill_between(x_dash[33:66], y_dash[33:66], 0, alpha=0.3, color='yellow', label='一般')
    ax.fill_between(x_dash[66:], y_dash[66:], 0, alpha=0.3, color='green', label='好')
    
    # 指针
    pointer_angle = energy_acc * np.pi
    pointer_x = [0, 0.9 * np.cos(pointer_angle)]
    pointer_y = [0, 0.9 * np.sin(pointer_angle)]
    ax.plot(pointer_x, pointer_y, 'k-', linewidth=4)
    ax.plot(0, 0, 'ko', markersize=15)
    
    ax.plot(x_dash, y_dash, 'k-', linewidth=3)
    ax.text(0, -0.2, f'{energy_acc:.1%}', ha='center', fontsize=16, fontweight='bold')
    
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-0.3, 1.3)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('能量准确率', fontsize=12, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9)
    
    plt.suptitle('能量指标 (Energy Metrics)', fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Energy metrics plot saved to {save_path}")


def plot_relative_error_metrics(
    metrics: dict,
    save_path: str = 'results/04_relative_error_metrics.png'
):
    """绘制相对误差指标"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    rae = metrics.get('rae', 0)
    rse = metrics.get('rse', 0)
    mape = metrics.get('mape', 0)
    
    if rae == 0 and rse == 0 and mape == 0:
        print("No relative error data available")
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 相对绝对误差
    ax = axes[0]
    ax.barh(['RAE'], [rae], color='#E76F51', edgecolor='black', linewidth=2, height=0.5)
    ax.text(rae/2, 0, f'{rae:.4f}', ha='center', va='center', fontsize=12, fontweight='bold', color='white')
    ax.set_xlabel('相对绝对误差', fontsize=11, fontweight='bold')
    ax.set_title('相对绝对误差\n(Relative Absolute Error)', fontsize=12, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    # 相对平方误差
    ax = axes[1]
    ax.barh(['RSE'], [rse], color='#F4A261', edgecolor='black', linewidth=2, height=0.5)
    ax.text(rse/2, 0, f'{rse:.4f}', ha='center', va='center', fontsize=12, fontweight='bold', color='white')
    ax.set_xlabel('相对平方误差', fontsize=11, fontweight='bold')
    ax.set_title('相对平方误差\n(Relative Squared Error)', fontsize=12, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    # 平均绝对百分比误差
    ax = axes[2]
    ax.barh(['MAPE'], [mape], color='#D62828', edgecolor='black', linewidth=2, height=0.5)
    ax.text(mape/2, 0, f'{mape:.2f}%', ha='center', va='center', fontsize=12, fontweight='bold', color='white')
    ax.set_xlabel('平均绝对百分比误差 (%)', fontsize=11, fontweight='bold')
    ax.set_title('平均绝对百分比误差\n(Mean Absolute Percentage Error)', fontsize=12, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    plt.suptitle('相对误差指标 (Relative Error Metrics)', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Relative error metrics plot saved to {save_path}")


def plot_all_metrics_comparison(
    metrics: dict,
    save_path: str = 'results/05_all_metrics_summary.png'
):
    """绘制所有指标汇总表格和总体对比"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    fig = plt.figure(figsize=(14, 10))
    
    # 指标分类汇总
    categories = {
        '误差指标': ['mae', 'mse', 'rmse', 'nmae', 'nrmse'],
        '状态识别': ['accuracy', 'precision', 'recall', 'f1'],
        '能量指标': ['total_energy_true', 'total_energy_pred', 'energy_accuracy'],
        '相对误差': ['rae', 'rse', 'mape'],
    }
    
    row = 0
    for category, metric_keys in categories.items():
        # 类别标题
        ax = plt.subplot(4, 1, row + 1)
        ax.axis('off')
        
        # 收集数据
        row_data = []
        for key in metric_keys:
            if key in metrics:
                value = metrics[key]
                if isinstance(value, float):
                    if key in ['mae', 'mse', 'rmse', 'nrmse', 'rae', 'rse']:
                        val_str = f'{value:.4f}'
                    elif key in ['nmae', 'mape']:
                        val_str = f'{value:.2f}%' if key == 'mape' else f'{value*100:.2f}%'
                    elif key in ['accuracy', 'precision', 'recall', 'f1', 'energy_accuracy']:
                        val_str = f'{value:.4f}'
                    else:
                        val_str = f'{value:.0f}'
                else:
                    val_str = str(value)
                row_data.append([key.upper(), val_str])
        
        if row_data:
            # 创建表格
            table = ax.table(
                cellText=row_data,
                colLabels=['指标', '数值'],
                cellLoc='center',
                loc='center',
                bbox=[0, 0, 1, 1]
            )
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 2.2)
            
            # 样式化
            colors = ['#4ECDC4', '#95E1D3', '#E76F51', '#F4A261']
            header_color = colors[row % 4]
            
            for i in range(len(row_data) + 1):
                if i == 0:
                    table[(i, 0)].set_facecolor(header_color)
                    table[(i, 1)].set_facecolor(header_color)
                    table[(i, 0)].set_text_props(weight='bold', color='white', size=11)
                    table[(i, 1)].set_text_props(weight='bold', color='white', size=11)
                else:
                    color = '#F0F8FF' if i % 2 == 0 else '#FFFFFF'
                    table[(i, 0)].set_facecolor(color)
                    table[(i, 1)].set_facecolor(color)
            
            # 类别标题
            ax.text(0.02, 0.98, category, fontsize=12, fontweight='bold',
                   transform=ax.transAxes, va='top', ha='left',
                   bbox=dict(boxstyle='round', facecolor=header_color, alpha=0.3))
        
        row += 1
    
    plt.suptitle('完整指标汇总 (Complete Metrics Summary)', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"All metrics summary plot saved to {save_path}")


def plot_evaluation_metrics(
    metrics: dict,
    output_dir: str = 'results',
    appliance: str = 'appliance'
):
    """
    生成所有评估指标的多个图表。
    
    Args:
        metrics: 指标字典
        output_dir: 输出目录
        appliance: 电器名称
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "=" * 60)
    print("Generating evaluation metrics visualizations...")
    print("=" * 60)
    
    # 1. 功率误差指标
    plot_power_error_metrics(
        metrics,
        os.path.join(output_dir, f'01_{appliance}_power_error_metrics.png')
    )
    
    # 2. 状态识别指标
    plot_state_recognition_metrics(
        metrics,
        os.path.join(output_dir, f'02_{appliance}_state_recognition_metrics.png')
    )
    
    # 3. 能量指标
    plot_energy_metrics(
        metrics,
        os.path.join(output_dir, f'03_{appliance}_energy_metrics.png')
    )
    
    # 4. 相对误差指标
    if metrics.get('rae', 0) > 0 or metrics.get('rse', 0) > 0 or metrics.get('mape', 0) > 0:
        plot_relative_error_metrics(
            metrics,
            os.path.join(output_dir, f'04_{appliance}_relative_error_metrics.png')
        )
    
    # 5. 完整汇总表格
    plot_all_metrics_comparison(
        metrics,
        os.path.join(output_dir, f'05_{appliance}_metrics_summary.png')
    )
    
    print("=" * 60)
    print("✓ All metrics visualization plots generated successfully!")
    print("=" * 60)
