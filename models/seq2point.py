"""
用于 NILM 的 Seq2Point 模型实现。
基于论文 "Sequence-to-point learning with neural networks for
nonintrusive load monitoring" (Zhang 等).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Seq2Point(nn.Module):
    """
    Seq2Point 电器功率分解模型。
    
    输入为聚合功率窗口，输出为窗口中点对应电器的功率。
    """
    
    def __init__(
        self,
        input_size: int = 599,
        dropout_rate: float = 0.1
    ):
        """
        初始化 Seq2Point 模型。
        
        Args:
            input_size: 输入窗口长度
            dropout_rate: Dropout 概率
        """
        super(Seq2Point, self).__init__()
        
        self.input_size = input_size
        self.dropout_rate = dropout_rate
        
        # 卷积层
        self.conv1 = nn.Conv1d(
            in_channels=1,
            out_channels=30,
            kernel_size=10,
            stride=1
        )
        
        self.conv2 = nn.Conv1d(
            in_channels=30,
            out_channels=30,
            kernel_size=8,
            stride=1
        )
        
        self.conv3 = nn.Conv1d(
            in_channels=30,
            out_channels=40,
            kernel_size=6,
            stride=1
        )
        
        self.conv4 = nn.Conv1d(
            in_channels=40,
            out_channels=50,
            kernel_size=5,
            stride=1
        )
        
        self.conv5 = nn.Conv1d(
            in_channels=50,
            out_channels=50,
            kernel_size=2,
            stride=1
        )
        
        # Dropout
        self.dropout = nn.Dropout(p=dropout_rate)
        
        # 全连接层
        # 计算卷积层输出尺寸
        self._conv_output_size = self._get_conv_output_size()
        
        self.fc1 = nn.Linear(self._conv_output_size, 1024)
        self.fc2 = nn.Linear(1024, 1)
        
    def _get_conv_output_size(self):
        """计算所有卷积层输出展开后的尺寸。"""
        # 使用虚拟输入推断卷积输出大小
        dummy_input = torch.zeros(1, 1, self.input_size)
        x = F.relu(self.conv1(dummy_input))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        return x.numel()
    
    def forward(self, x):
        """
        前向传播。
        
        Args:
            x: 输入张量，形状 (batch_size, input_size)
            
        Returns:
            输出张量，形状 (batch_size, 1)
        """
        # 重塑输入: (batch_size, input_size) -> (batch_size, 1, input_size)
        x = x.unsqueeze(1)
        
        # 卷积层 + ReLU 激活
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # 在第 4 层卷积后添加 Dropout
        x = F.relu(self.conv4(x))
        x = self.dropout(x)
        
        # 在第 5 层卷积后添加 Dropout
        x = F.relu(self.conv5(x))
        x = self.dropout(x)
        
        # 展平
        x = x.view(x.size(0), -1)
        
        # 全连接层
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        # 压缩输出维度: (batch_size, 1) -> (batch_size,)
        x = x.squeeze(1)
        
        return x


class Seq2PointLSTM(nn.Module):
    """
    带 LSTM 的 Seq2Point 变体。
    """
    
    def __init__(
        self,
        input_size: int = 599,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout_rate: float = 0.1,
        bidirectional: bool = False
    ):
        """
        初始化 Seq2Point-LSTM 模型。

        Args:
            input_size: 输入窗口长度
            hidden_size: LSTM 隐藏层维度
            num_layers: LSTM 层数
            dropout_rate: Dropout 概率
            bidirectional: 是否使用双向 LSTM
        """
        super(Seq2PointLSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        # LSTM 层
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0,
            bidirectional=bidirectional
        )

        # Dropout
        self.dropout = nn.Dropout(p=dropout_rate)

        # 全连接层（双向 LSTM 输出维度是 hidden_size * 2）
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        self.fc1 = nn.Linear(lstm_output_size, 64)
        self.fc2 = nn.Linear(64, 1)
        
    def forward(self, x):
        """
        前向传播。
        
        Args:
            x: 输入张量，形状 (batch_size, input_size)
            
        Returns:
            输出张量，形状 (batch_size, 1)
        """
        # 重塑输入: (batch_size, input_size) -> (batch_size, input_size, 1)
        x = x.unsqueeze(2)
        
        # LSTM 计算
        lstm_out, _ = self.lstm(x)
        
        # 取窗口中点的输出
        middle_idx = self.input_size // 2
        x = lstm_out[:, middle_idx, :]
        
        # 全连接层
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        # 压缩输出维度
        x = x.squeeze(1)
        
        return x


class Seq2PointBiLSTM(nn.Module):
    """
    带双向 LSTM 的 Seq2Point 变体。
    专门用于双向序列建模，同时利用过去和未来的上下文信息。
    """

    def __init__(
        self,
        input_size: int = 599,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout_rate: float = 0.1
    ):
        """
        初始化 Seq2Point-BiLSTM 模型。

        Args:
            input_size: 输入窗口长度
            hidden_size: BiLSTM 隐藏层维度（每个方向）
            num_layers: BiLSTM 层数
            dropout_rate: Dropout 概率
        """
        super(Seq2PointBiLSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # 双向 LSTM 层
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0,
            bidirectional=True  # 固定为双向
        )

        # Dropout
        self.dropout = nn.Dropout(p=dropout_rate)

        # 全连接层（双向 LSTM 输出维度是 hidden_size * 2）
        self.fc1 = nn.Linear(hidden_size * 2, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        """
        前向传播。

        Args:
            x: 输入张量，形状 (batch_size, input_size)

        Returns:
            输出张量，形状 (batch_size,)
        """
        # 重塑输入: (batch_size, input_size) -> (batch_size, input_size, 1)
        x = x.unsqueeze(2)

        # BiLSTM 计算
        lstm_out, _ = self.lstm(x)

        # 取窗口中点的输出
        middle_idx = self.input_size // 2
        x = lstm_out[:, middle_idx, :]

        # 全连接层
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        # 压缩输出维度
        x = x.squeeze(1)

        return x


def get_model(model_name: str = 'seq2point', **kwargs):
    """
    根据名称创建模型。

    Args:
        model_name: 模型名称（'seq2point', 'seq2point_lstm', 'seq2point_bilstm', 'transformer'）
        **kwargs: 模型参数

    Returns:
        模型实例
    """
    if model_name.lower() == 'seq2point':
        return Seq2Point(**kwargs)
    elif model_name.lower() == 'seq2point_lstm':
        return Seq2PointLSTM(**kwargs)
    elif model_name.lower() == 'seq2point_bilstm':
        return Seq2PointBiLSTM(**kwargs)
    elif model_name.lower() in ('seq2point_transformer', 'transformer'):
        from .seq2point_transformer import Seq2PointTransformer
        return Seq2PointTransformer(**kwargs)
    else:
        raise ValueError(f"Unknown model: {model_name}")


if __name__ == "__main__":
    # 简单自测
    model = Seq2Point(input_size=599)
    print(model)
    
    # 测试前向传播
    dummy_input = torch.randn(32, 599)
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    
    # 统计参数量
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {num_params:,}")
