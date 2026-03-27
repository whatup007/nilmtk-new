"""
Seq2Point NILM 模型推理脚本。
"""

import os
import sys
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

# 将项目根目录加入 sys.path，便于直接运行脚本
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import get_model
from utils.config_utils import load_config, get_appliance_params


def _load_checkpoint_compat(checkpoint_path: str, device: str):
    """兼容 PyTorch 2.6+ 的 checkpoint 加载。"""
    try:
        return torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(checkpoint_path, map_location=device)


class Seq2PointInference:
    """Seq2Point 单模型推理封装类。"""
    
    def __init__(
        self,
        checkpoint_path: str,
        config_path: str = None,
        device: str = 'cpu'
    ):
        """
        初始化推理器。
        
        Args:
            checkpoint_path: 模型检查点路径
            config_path: 配置文件路径（可选）
            device: 推理设备
        """
        # 加载检查点
        checkpoint = _load_checkpoint_compat(checkpoint_path, device)
        
        # 加载配置
        if config_path:
            self.config = load_config(config_path)
        else:
            if 'config' in checkpoint:
                self.config = checkpoint['config']
            else:
                raise ValueError("No config found. Please provide config_path")
        
        # 设置运行设备
        self.device = torch.device(device)
        
        # 获取电器参数（最大功率与阈值）
        appliance_params = get_appliance_params(
            self.config['data']['appliance'],
            'configs/appliance_params.yaml'
        )
        self.max_power = appliance_params['max_power']
        self.threshold = appliance_params['threshold']
        
        # 归一化参数（对聚合功率进行标准化）
        self.mean_aggregate = 0.0
        self.std_aggregate = 1.0
        
        # 窗口长度
        self.window_size = self.config['data']['window_size']
        
        # 创建并加载模型
        self.model = get_model(
            model_name=self.config['model']['name'],
            input_size=self.config['model']['input_size'],
            dropout_rate=self.config['model']['dropout_rate']
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded from {checkpoint_path}")
        print(f"Appliance: {self.config['data']['appliance']}")
        print(f"Device: {self.device}")
    
    def set_normalization_params(self, mean: float, std: float):
        """设置聚合功率的归一化参数。"""
        self.mean_aggregate = mean
        self.std_aggregate = std
    
    def normalize_aggregate(self, aggregate: np.ndarray) -> np.ndarray:
        """对聚合功率进行标准化。"""
        return (aggregate - self.mean_aggregate) / (self.std_aggregate + 1e-8)
    
    def denormalize_predictions(self, predictions: np.ndarray) -> np.ndarray:
        """将预测值反归一化到功率尺度。"""
        return predictions * self.max_power
    
    def predict_sequence(
        self,
        aggregate: np.ndarray,
        batch_size: int = 4096,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        从聚合功率序列预测电器功率。
        
        Args:
            aggregate: 聚合功率序列（未归一化）
            
        Returns:
            预测的电器功率序列
        """
        # 如果尚未设置归一化参数，则基于输入自动计算
        if self.mean_aggregate == 0.0 and self.std_aggregate == 1.0:
            self.mean_aggregate = np.mean(aggregate)
            self.std_aggregate = np.std(aggregate)
            print(f"Auto-calculated normalization: mean={self.mean_aggregate:.2f}, std={self.std_aggregate:.2f}")
        
        # 标准化聚合功率
        aggregate_norm = self.normalize_aggregate(aggregate)
        
        # 按滑动窗口批量推理
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")

        num_windows = len(aggregate) - self.window_size + 1
        if num_windows <= 0:
            raise ValueError(
                f"Input length {len(aggregate)} is smaller than window_size {self.window_size}"
            )

        windows = np.lib.stride_tricks.sliding_window_view(aggregate_norm, self.window_size)
        predictions = []
        offset = self.window_size // 2

        batch_iter = range(0, num_windows, batch_size)
        if show_progress:
            batch_iter = tqdm(batch_iter, desc='Inference', leave=False)

        with torch.no_grad():
            for start_idx in batch_iter:
                end_idx = min(start_idx + batch_size, num_windows)
                window_batch = windows[start_idx:end_idx]

                window_tensor = torch.from_numpy(window_batch.astype(np.float32)).to(self.device)
                pred = self.model(window_tensor)
                predictions.append(pred.detach().cpu().numpy().reshape(-1))
        
        # 反归一化并确保非负
        predictions = np.concatenate(predictions, axis=0)
        predictions = self.denormalize_predictions(predictions)
        predictions = np.maximum(predictions, 0)  # Ensure non-negative
        
        # 对齐长度（中点对齐）
        padded_predictions = np.zeros(len(aggregate))
        padded_predictions[offset:offset + len(predictions)] = predictions
        
        return padded_predictions
    
    def predict_from_file(
        self,
        input_path: str,
        output_path: str = None,
        plot: bool = True
    ):
        """
        从文件读取聚合功率并执行预测。
        
        Args:
            input_path: 输入文件路径（npy 或 csv）
            output_path: 预测结果保存路径
            plot: 是否绘图
        """
        # 读取输入数据
        if input_path.endswith('.npy'):
            aggregate = np.load(input_path)
        elif input_path.endswith('.csv'):
            aggregate = np.loadtxt(input_path, delimiter=',')
        else:
            raise ValueError("Unsupported file format. Use .npy or .csv")
        
        print(f"Loaded aggregate data: {len(aggregate)} samples")
        
        # 执行预测
        predictions = self.predict_sequence(aggregate)
        
        print(f"Prediction complete")
        print(f"Total energy - Aggregate: {np.sum(aggregate):.2f} Wh")
        print(f"Total energy - Predicted: {np.sum(predictions):.2f} Wh")
        
        # 保存预测结果
        if output_path:
            np.save(output_path, predictions)
            print(f"Predictions saved to {output_path}")
        
        # 绘图展示
        if plot:
            self._plot_results(aggregate, predictions, input_path)
        
        return predictions
    
    def _plot_results(
        self,
        aggregate: np.ndarray,
        predictions: np.ndarray,
        title: str = ""
    ):
        """绘制聚合功率与预测结果。"""
        fig, axes = plt.subplots(2, 1, figsize=(15, 8))
        
        time = np.arange(len(aggregate))
        
        # 绘制聚合功率
        axes[0].plot(time, aggregate, 'b-', label='Aggregate Power', alpha=0.7)
        axes[0].set_xlabel('Time (samples)')
        axes[0].set_ylabel('Power (W)')
        axes[0].set_title(f'Aggregate Power - {title}')
        axes[0].legend()
        axes[0].grid(True)
        
        # 绘制预测功率
        axes[1].plot(time, predictions, 'r-', label=f'{self.config["data"]["appliance"]} (Predicted)', alpha=0.7)
        axes[1].axhline(y=self.threshold, color='g', linestyle='--', label='Threshold')
        axes[1].set_xlabel('Time (samples)')
        axes[1].set_ylabel('Power (W)')
        axes[1].set_title(f'{self.config["data"]["appliance"]} Predictions')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='Run inference with Seq2Point model')
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to input data file (.npy or .csv)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Path to save predictions'
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to configuration file'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        choices=['cpu', 'cuda'],
        help='Device to run inference on'
    )
    parser.add_argument(
        '--no-plot',
        action='store_true',
        help='Disable plotting'
    )
    
    args = parser.parse_args()
    
    # 初始化推理器
    inferencer = Seq2PointInference(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        device=args.device
    )
    
    # 执行推理
    predictions = inferencer.predict_from_file(
        input_path=args.input,
        output_path=args.output,
        plot=not args.no_plot
    )


class Seq2PointParallelInference:
    """Seq2Point + Transformer 并行推理封装类。"""

    def __init__(
        self,
        checkpoint_path: str,
        config_path: str = None,
        device: str = 'cpu'
    ):
        """
        初始化并行推理器。
        
        Args:
            checkpoint_path: 模型检查点路径
            config_path: 配置文件路径（可选）
            device: 推理设备
        """
        checkpoint = _load_checkpoint_compat(checkpoint_path, device)

        if config_path:
            self.config = load_config(config_path)
        else:
            if 'config' in checkpoint:
                self.config = checkpoint['config']
            else:
                raise ValueError("No config found. Please provide config_path")

        self.device = torch.device(device)

        appliance_params = get_appliance_params(
            self.config['data']['appliance'],
            'configs/appliance_params.yaml'
        )
        self.max_power = appliance_params['max_power']
        self.threshold = appliance_params['threshold']

        self.mean_aggregate = 0.0
        self.std_aggregate = 1.0
        self.window_size = self.config['data']['window_size']

        seq2point_cfg = self.config['model'].get('seq2point', {})
        transformer_cfg = self.config['model'].get('transformer', {})

        self.model_seq2point = get_model(
            model_name='seq2point',
            input_size=seq2point_cfg.get('input_size', self.config['model']['input_size']),
            dropout_rate=seq2point_cfg.get('dropout_rate', 0.1)
        )
        self.model_transformer = get_model(
            model_name='seq2point_transformer',
            input_size=transformer_cfg.get('input_size', self.config['model']['input_size']),
            d_model=transformer_cfg.get('d_model', 128),
            nhead=transformer_cfg.get('nhead', 4),
            num_layers=transformer_cfg.get('num_layers', 4),
            dim_feedforward=transformer_cfg.get('dim_feedforward', 256),
            dropout_rate=transformer_cfg.get('dropout_rate', 0.1)
        )

        self.model_seq2point.load_state_dict(checkpoint['seq2point_state_dict'])
        self.model_transformer.load_state_dict(checkpoint['transformer_state_dict'])
        self.model_seq2point = self.model_seq2point.to(self.device)
        self.model_transformer = self.model_transformer.to(self.device)
        self.model_seq2point.eval()
        self.model_transformer.eval()

        print(f"Parallel models loaded from {checkpoint_path}")
        print(f"Appliance: {self.config['data']['appliance']}")
        print(f"Device: {self.device}")

    def set_normalization_params(self, mean: float, std: float):
        """设置聚合功率的归一化参数。"""
        self.mean_aggregate = mean
        self.std_aggregate = std

    def normalize_aggregate(self, aggregate: np.ndarray) -> np.ndarray:
        """对聚合功率进行标准化。"""
        return (aggregate - self.mean_aggregate) / (self.std_aggregate + 1e-8)

    def denormalize_predictions(self, predictions: np.ndarray) -> np.ndarray:
        """将预测值反归一化到功率尺度。"""
        return predictions * self.max_power

    def predict_sequence(
        self,
        aggregate: np.ndarray,
        batch_size: int = 4096,
        show_progress: bool = True
    ) -> tuple:
        """
        并行推理聚合功率，返回 (seq2point_pred, transformer_pred, sum_pred)。
        """
        if self.mean_aggregate == 0.0 and self.std_aggregate == 1.0:
            self.mean_aggregate = np.mean(aggregate)
            self.std_aggregate = np.std(aggregate)
            print(f"Auto-calculated normalization: mean={self.mean_aggregate:.2f}, std={self.std_aggregate:.2f}")

        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")

        num_windows = len(aggregate) - self.window_size + 1
        if num_windows <= 0:
            raise ValueError(
                f"Input length {len(aggregate)} is smaller than window_size {self.window_size}"
            )

        aggregate_norm = self.normalize_aggregate(aggregate)
        windows = np.lib.stride_tricks.sliding_window_view(aggregate_norm, self.window_size)
        pred_seq2point = []
        pred_transformer = []
        offset = self.window_size // 2

        batch_iter = range(0, num_windows, batch_size)
        if show_progress:
            batch_iter = tqdm(batch_iter, desc='Parallel inference', leave=False)

        with torch.no_grad():
            for start_idx in batch_iter:
                end_idx = min(start_idx + batch_size, num_windows)
                window_batch = windows[start_idx:end_idx]
                window_tensor = torch.from_numpy(window_batch.astype(np.float32)).to(self.device)

                out_seq2point = self.model_seq2point(window_tensor)
                out_transformer = self.model_transformer(window_tensor)

                pred_seq2point.append(out_seq2point.detach().cpu().numpy().reshape(-1))
                pred_transformer.append(out_transformer.detach().cpu().numpy().reshape(-1))

        pred_seq2point = np.concatenate(pred_seq2point, axis=0)
        pred_transformer = np.concatenate(pred_transformer, axis=0)
        pred_sum = pred_seq2point + pred_transformer

        pred_seq2point = self.denormalize_predictions(pred_seq2point)
        pred_transformer = self.denormalize_predictions(pred_transformer)
        pred_sum = self.denormalize_predictions(pred_sum)

        pred_seq2point = np.maximum(pred_seq2point, 0)
        pred_transformer = np.maximum(pred_transformer, 0)
        pred_sum = np.maximum(pred_sum, 0)

        padded_seq2point = np.zeros(len(aggregate))
        padded_transformer = np.zeros(len(aggregate))
        padded_sum = np.zeros(len(aggregate))
        padded_seq2point[offset:offset + len(pred_seq2point)] = pred_seq2point
        padded_transformer[offset:offset + len(pred_transformer)] = pred_transformer
        padded_sum[offset:offset + len(pred_sum)] = pred_sum

        return padded_seq2point, padded_transformer, padded_sum

    def predict_from_file(
        self,
        input_path: str,
        output_path: str = None,
        plot: bool = True
    ):
        """
        从文件读取聚合功率并执行并行预测。
        """
        if input_path.endswith('.npy'):
            aggregate = np.load(input_path)
        elif input_path.endswith('.csv'):
            aggregate = np.loadtxt(input_path, delimiter=',')
        else:
            raise ValueError("Unsupported file format. Use .npy or .csv")

        print(f"Loaded aggregate data: {len(aggregate)} samples")

        pred_seq2point, pred_transformer, pred_sum = self.predict_sequence(aggregate)

        print(f"Prediction complete")
        print(f"Total energy - Aggregate: {np.sum(aggregate):.2f} Wh")
        print(f"Total energy - Seq2Point: {np.sum(pred_seq2point):.2f} Wh")
        print(f"Total energy - Transformer: {np.sum(pred_transformer):.2f} Wh")
        print(f"Total energy - Sum: {np.sum(pred_sum):.2f} Wh")

        if output_path:
            np.savez(
                output_path,
                seq2point=pred_seq2point,
                transformer=pred_transformer,
                sum=pred_sum
            )
            print(f"Predictions saved to {output_path}")

        if plot:
            self._plot_results(aggregate, pred_seq2point, pred_transformer, pred_sum, input_path)

        return pred_seq2point, pred_transformer, pred_sum

    def _plot_results(
        self,
        aggregate: np.ndarray,
        pred_seq2point: np.ndarray,
        pred_transformer: np.ndarray,
        pred_sum: np.ndarray,
        title: str = ""
    ):
        """绘制并行预测结果。"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        time = np.arange(len(aggregate))

        # 聚合功率
        axes[0, 0].plot(time, aggregate, 'b-', alpha=0.7)
        axes[0, 0].set_xlabel('Time (samples)')
        axes[0, 0].set_ylabel('Power (W)')
        axes[0, 0].set_title(f'Aggregate Power - {title}')
        axes[0, 0].grid(True)

        # Seq2Point 预测
        axes[0, 1].plot(time, pred_seq2point, 'g-', alpha=0.7)
        axes[0, 1].axhline(y=self.threshold, color='r', linestyle='--')
        axes[0, 1].set_xlabel('Time (samples)')
        axes[0, 1].set_ylabel('Power (W)')
        axes[0, 1].set_title('Seq2Point Prediction')
        axes[0, 1].grid(True)

        # Transformer 预测
        axes[1, 0].plot(time, pred_transformer, 'orange', alpha=0.7)
        axes[1, 0].axhline(y=self.threshold, color='r', linestyle='--')
        axes[1, 0].set_xlabel('Time (samples)')
        axes[1, 0].set_ylabel('Power (W)')
        axes[1, 0].set_title('Transformer Prediction')
        axes[1, 0].grid(True)

        # 相加预测
        axes[1, 1].plot(time, pred_sum, 'purple', alpha=0.7)
        axes[1, 1].axhline(y=self.threshold, color='r', linestyle='--')
        axes[1, 1].set_xlabel('Time (samples)')
        axes[1, 1].set_ylabel('Power (W)')
        axes[1, 1].set_title('Sum Prediction (Seq2Point + Transformer)')
        axes[1, 1].grid(True)

        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    main()
