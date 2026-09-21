"""任务二使用的小型卷积自编码器。"""

from __future__ import annotations

import torch
from torch import nn


class ConvAutoencoder(nn.Module):
    """把 128×128 单通道接触窗口压缩为固定长度表征。"""

    def __init__(self, latent_dim: int = 32) -> None:
        super().__init__()
        if latent_dim <= 0:
            raise ValueError(f"latent_dim 必须为正整数，实际为 {latent_dim}")
        self.latent_dim = latent_dim
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.encoder_linear = nn.Linear(64 * 16 * 16, latent_dim)
        self.decoder_linear = nn.Linear(latent_dim, 64 * 16 * 16)
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1),
        )

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.encoder_conv(inputs)
        return self.encoder_linear(features.flatten(start_dim=1))

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        features = self.decoder_linear(latent).reshape(-1, 64, 16, 16)
        return self.decoder_conv(features)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(inputs))


def upper_triangle_mask(
    size: int = 128,
    diagonal_exclusion: int = 1,
    *,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """返回用于形态误差的上三角掩码，默认排除主对角线及相邻一条。"""
    if size <= 0:
        raise ValueError(f"size 必须为正整数，实际为 {size}")
    if diagonal_exclusion < 0:
        raise ValueError("diagonal_exclusion 不能为负数")
    return torch.triu(
        torch.ones((size, size), dtype=torch.bool, device=device),
        diagonal=diagonal_exclusion + 1,
    )


def masked_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    reduction: str = "mean",
) -> torch.Tensor:
    """计算每个窗口的掩码均方误差，避免对称矩阵重复计数。"""
    if prediction.shape != target.shape:
        raise ValueError(
            f"prediction 与 target 形状必须一致，实际为 {prediction.shape} 和 {target.shape}"
        )
    if prediction.ndim != 4 or prediction.shape[1] != 1:
        raise ValueError(f"输入应为 N×1×H×W，实际为 {prediction.shape}")
    if mask.ndim == 2:
        if mask.shape != prediction.shape[-2:]:
            raise ValueError(
                f"mask 形状 {mask.shape} 与矩阵形状 {prediction.shape[-2:]} 不一致"
            )
        window_mask = mask.unsqueeze(0).expand(len(prediction), -1, -1)
    elif mask.ndim == 3:
        if mask.shape != (len(prediction), *prediction.shape[-2:]):
            raise ValueError(
                f"逐窗口 mask 应为 N×H×W，实际为 {mask.shape}，输入为 {prediction.shape}"
            )
        window_mask = mask
    else:
        raise ValueError(f"mask 应为 H×W 或 N×H×W，实际为 {mask.shape}")
    valid_counts = window_mask.sum(dim=(1, 2))
    if torch.any(valid_counts == 0):
        bad_rows = torch.nonzero(valid_counts == 0, as_tuple=False).flatten().tolist()
        raise ValueError(f"mask 在 loss 区域内没有有效像素，批内行号：{bad_rows}")
    squared_error = (prediction - target).square()[:, 0]
    per_window = (squared_error * window_mask).sum(dim=(1, 2)) / valid_counts
    if reduction == "none":
        return per_window
    if reduction == "mean":
        return per_window.mean()
    raise ValueError(f"不支持的 reduction={reduction!r}")
