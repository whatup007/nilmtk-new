"""
Seq2Point Transformer 模型实现。
输入为功率序列窗口，输出为窗口中点的电器功率预测。
"""

import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """标准正弦位置编码。"""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class Seq2PointTransformer(nn.Module):
    """基于 Transformer Encoder 的 Seq2Point 模型。"""

    def __init__(
        self,
        input_size: int = 599,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 256,
        dropout_rate: float = 0.1
    ):
        super().__init__()
        self.input_size = input_size
        self.d_model = d_model

        # 输入映射到 d_model 维度
        self.input_proj = nn.Linear(1, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout_rate)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout_rate,
            activation='relu',
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 取窗口中点的表示进行回归
        self.output_proj = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, input_size)
        Returns:
            (batch_size,)
        """
        # (batch, seq_len) -> (batch, seq_len, 1)
        x = x.unsqueeze(-1)
        x = self.input_proj(x)  # (batch, seq_len, d_model)

        x = self.pos_encoder(x)
        x = self.encoder(x)

        middle_idx = self.input_size // 2
        middle = x[:, middle_idx, :]  # (batch, d_model)
        out = self.output_proj(middle)  # (batch, 1)
        return out.squeeze(1)
