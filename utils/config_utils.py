"""
配置加载与校验工具函数。
"""

import yaml
import os
from typing import Dict, Any


def load_config(config_path: str) -> Dict[str, Any]:
    """
    从 YAML 文件加载配置。
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        配置字典
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config


def save_config(config: Dict[str, Any], save_path: str):
    """
    保存配置到 YAML 文件。
    
    Args:
        config: 配置字典
        save_path: 保存路径
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False)


def update_config(config: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    """
    递归更新配置。
    
    Args:
        config: 原始配置
        updates: 更新内容
        
    Returns:
        更新后的配置
    """
    for key, value in updates.items():
        if isinstance(value, dict) and key in config and isinstance(config[key], dict):
            config[key] = update_config(config[key], value)
        else:
            config[key] = value
    
    return config


def get_appliance_params(appliance: str, config_path: str = 'configs/appliance_params.yaml') -> Dict[str, Any]:
    """
    获取电器参数。
    
    Args:
        appliance: 电器名称
        config_path: 电器参数文件路径
        
    Returns:
        电器参数字典
    """
    params = load_config(config_path)
    
    if appliance not in params:
        raise ValueError(f"Appliance '{appliance}' not found in config")
    
    return params[appliance]


class ConfigValidator:
    """配置校验器。"""
    
    @staticmethod
    def validate(config: Dict[str, Any]) -> bool:
        """
        校验配置合法性。
        
        Args:
            config: 配置字典
            
        Returns:
            合法则返回 True，否则抛出异常
        """
        # 检查必须的顶层字段
        required_keys = ['data', 'model', 'training']
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required configuration key: {key}")
        
        # 校验数据配置
        data_config = config['data']
        if 'data_path' not in data_config:
            raise ValueError("Missing 'data_path' in data configuration")
        
        if not os.path.exists(data_config['data_path']):
            raise ValueError(f"Data file not found: {data_config['data_path']}")
        
        # 校验训练配置
        training_config = config['training']
        if training_config.get('batch_size', 0) <= 0:
            raise ValueError("batch_size must be positive")

        if training_config.get('epochs', 0) <= 0:
            raise ValueError("epochs must be positive")

        mode = training_config.get('mode', 'single')
        if mode == 'parallel':
            parallel_cfg = training_config.get('parallel', {})
            seq_cfg = parallel_cfg.get('seq2point', {})
            trans_cfg = parallel_cfg.get('transformer', {})
            if seq_cfg.get('learning_rate', 0) <= 0:
                raise ValueError("parallel.seq2point.learning_rate must be positive")
            if trans_cfg.get('learning_rate', 0) <= 0:
                raise ValueError("parallel.transformer.learning_rate must be positive")
        else:
            if training_config.get('learning_rate', 0) <= 0:
                raise ValueError("learning_rate must be positive")
        
        return True
