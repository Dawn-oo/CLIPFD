import torch
import torch.nn as nn


class GlobalAdapter(nn.Module):
    """
    全局特征适配层：
    将 backbone 原始 global_feat 转成更适合当前任务的全局表征，
    并同时供：
    1) 全局辅助二分类头
    2) 全局-局部融合模块
    使用
    """
    def __init__(self, feat_dim: int = 768, dropout: float = 0.1):
        super().__init__()

        if not isinstance(feat_dim, int) or feat_dim <= 0:
            raise ValueError(f"feat_dim must be a positive integer, but got {feat_dim}")
        if dropout < 0:
            raise ValueError(f"dropout must be >= 0, but got {dropout}")

        self.feat_dim = feat_dim
        self.net = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, feat_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"x must be torch.Tensor, but got {type(x)}")
        if x.dim() != 2 or x.size(1) != self.feat_dim:
            raise ValueError(
                f"x must have shape [B, {self.feat_dim}], but got {tuple(x.shape)}"
            )
        return self.net(x)