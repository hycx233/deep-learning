"""任务一的轻量 CNN：直接读数值矩阵做三分类。

模型刻意保持小：三个卷积模块加全局池化，参数量在几万量级，和 344 条标注的
数据规模相称。类别不均衡通过加权交叉熵处理，不额外引入采样器或焦点损失。
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


def compute_class_weights(labels: np.ndarray, n_classes: int) -> torch.Tensor:
    """balanced 类权重 N / (K * count)，用于处理 68 / 250 / 26 的不均衡。"""
    counts = np.bincount(labels, minlength=n_classes).astype(np.float64)
    if (counts == 0).any():
        missing = np.flatnonzero(counts == 0).tolist()
        raise ValueError(f"类别 {missing} 在训练集中没有样本，无法计算类权重")
    weights = counts.sum() / (n_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


class SmallCNN(nn.Module):
    """三个卷积模块 + 全局平均池化的三分类模型。

    输入 ``(N, 1, 128, 128)``。每个模块是 3x3 卷积、BatchNorm、ReLU、2x2 池化，
    通道数按 16/32/64 递增；最后对特征图做全局平均池化再接线性层，避免把
    BatchNorm 的统计量绑死在固定空间尺寸上。
    """

    def __init__(
        self,
        n_classes: int = 3,
        in_channels: int = 1,
        widths: tuple[int, ...] = (16, 32, 64),
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        if not widths:
            raise ValueError("widths 不能为空")
        blocks: list[nn.Module] = []
        channels = in_channels
        for width in widths:
            blocks += [
                nn.Conv2d(channels, width, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(width),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            ]
            channels = width
        self.features = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(channels, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
