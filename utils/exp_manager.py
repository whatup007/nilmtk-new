"""
实验管理工具 - 类似YOLO的 runs/expN 结构。
"""

import os
import yaml
import shutil
from pathlib import Path


class ExperimentManager:
    """
    管理实验文件夹结构，类似YOLOv5/v8的 runs/ 结构。
    
    结构信息:
    runs/
    ├── exp1/
    │   ├── weights/
    │   │   ├── best.pth              # 最佳模型（最低验证损失）
    │   │   ├── best_metrics.pth      # 最佳综合指标模型
    │   │   └── last.pth              # 最后一epoch的模型
    │   ├── config.yaml               # 训练配置参数
    │   ├── *.png                     # 可视化图像
    │   └── training.log              # 训练日志
    ├── exp2/
    └── exp3/
    """
    
    def __init__(self, base_dir: str = 'runs'):
        """
        初始化实验管理器。
        
        Args:
            base_dir: 基础目录，默认为 runs
        """
        self.base_dir = base_dir
        self.exp_dir = None
        self.weights_dir = None
    
    def get_new_exp_dir(self) -> str:
        """
        获取新的实验目录（自动递增：exp1, exp2, ...）。
        
        Returns:
            新实验目录路径
        """
        os.makedirs(self.base_dir, exist_ok=True)
        
        # 找最大的 exp 编号
        exp_dirs = [d for d in os.listdir(self.base_dir) 
                    if os.path.isdir(os.path.join(self.base_dir, d)) and d.startswith('exp')]
        
        if not exp_dirs:
            exp_num = 1
        else:
            exp_nums = []
            for d in exp_dirs:
                try:
                    num = int(d.replace('exp', ''))
                    exp_nums.append(num)
                except ValueError:
                    pass
            exp_num = max(exp_nums) + 1 if exp_nums else 1
        
        self.exp_dir = os.path.join(self.base_dir, f'exp{exp_num}')
        self.weights_dir = os.path.join(self.exp_dir, 'weights')
        
        # 创建目录结构
        os.makedirs(self.weights_dir, exist_ok=True)
        
        print(f"✓ 实验目录创建: {self.exp_dir}")
        return self.exp_dir
    
    def get_exp_dir(self) -> str:
        """获取当前实验目录"""
        if self.exp_dir is None:
            self.get_new_exp_dir()
        return self.exp_dir
    
    def get_weights_dir(self) -> str:
        """获取权重保存目录"""
        if self.weights_dir is None:
            self.get_new_exp_dir()
        return self.weights_dir
    
    def save_config(self, config: dict, filename: str = 'config.yaml'):
        """
        保存训练配置。
        
        Args:
            config: 配置字典
            filename: 文件名，默认为 config.yaml
        """
        config_path = os.path.join(self.get_exp_dir(), filename)
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        print(f"✓ 配置已保存: {config_path}")
    
    def save_best_weight(self, checkpoint: dict, model_name: str = 'best.pth'):
        """
        保存最佳模型（最低验证损失）。
        
        Args:
            checkpoint: 模型检查点
            model_name: 保存文件名，默认为 best.pth
        """
        import torch
        weights_path = os.path.join(self.get_weights_dir(), model_name)
        torch.save(checkpoint, weights_path)
        print(f"✓ 最佳模型已保存: {weights_path}")
    
    def save_best_metrics_weight(self, checkpoint: dict, model_name: str = 'best_metrics.pth'):
        """
        保存综合指标最佳模型（F1-Score/综合指标优化）。
        
        Args:
            checkpoint: 模型检查点
            model_name: 保存文件名，默认为 best_metrics.pth
        """
        import torch
        weights_path = os.path.join(self.get_weights_dir(), model_name)
        torch.save(checkpoint, weights_path)
        print(f"✓ 综合指标最佳模型已保存: {weights_path}")
    
    def save_last_weight(self, checkpoint: dict, model_name: str = 'last.pth'):
        """
        保存最后一个epoch的模型。
        
        Args:
            checkpoint: 模型检查点
            model_name: 保存文件名，默认为 last.pth
        """
        import torch
        weights_path = os.path.join(self.get_weights_dir(), model_name)
        torch.save(checkpoint, weights_path)
        print(f"✓ 最后模型已保存: {weights_path}")
    
    def save_visualization(self, src_path: str, dest_name: str = None):
        """
        保存可视化图像到实验目录。
        
        Args:
            src_path: 源文件路径
            dest_name: 目标文件名（不指定则使用原文件名）
        """
        if not os.path.exists(src_path):
            print(f"⚠ 文件不存在: {src_path}")
            return
        
        dest_name = dest_name or os.path.basename(src_path)
        dest_path = os.path.join(self.get_exp_dir(), dest_name)
        
        shutil.copy2(src_path, dest_path)
        print(f"✓ 可视化图像已保存: {dest_path}")
    
    def print_structure(self):
        """打印实验目录结构"""
        exp_dir = self.get_exp_dir()
        print("\n" + "=" * 60)
        print(f"实验目录结构: {exp_dir}")
        print("=" * 60)
        
        for root, dirs, files in os.walk(exp_dir):
            level = root.replace(exp_dir, '').count(os.sep)
            indent = ' ' * 2 * level
            print(f"{indent}📁 {os.path.basename(root)}/")
            sub_indent = ' ' * 2 * (level + 1)
            for file in files:
                print(f"{sub_indent}📄 {file}")
        print("=" * 60 + "\n")


# 全局实验管理器实例
_exp_manager = None


def get_exp_manager(base_dir: str = 'runs') -> ExperimentManager:
    """
    获取全局实验管理器实例。
    
    Args:
        base_dir: 基础目录，默认为 runs
        
    Returns:
        ExperimentManager 实例
    """
    global _exp_manager
    if _exp_manager is None:
        _exp_manager = ExperimentManager(base_dir)
    return _exp_manager


def init_experiment(base_dir: str = 'runs') -> ExperimentManager:
    """
    初始化新的实验。
    
    Args:
        base_dir: 基础目录
        
    Returns:
        ExperimentManager 实例
    """
    global _exp_manager
    _exp_manager = ExperimentManager(base_dir)
    return _exp_manager
