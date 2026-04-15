import torch
import torch.nn as nn


class BinaryClassifierHead(nn.Module):
    """
    纯二分类头：
    输入一个表征向量，输出一个二分类 logit

    输入:
        x: [B, in_dim]

    输出:
        logits: [B, 1] B为批次大小
    """
    def __init__(self, in_dim: int):
        """
        :param in_dim: 输入为全局分支的表征向量，维度为768维
        """
        super().__init__()

        if not isinstance(in_dim, int) or in_dim <= 0:
            raise ValueError(f"in_dim must be a positive integer, but got {in_dim}")

        self.fc = nn.Linear(in_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2:
            raise ValueError(f"Input x must have shape [B, in_dim], but got {tuple(x.shape)}")

        if x.size(1) != self.fc.in_features:
            raise ValueError(
                f"Input feature dim mismatch: expected {self.fc.in_features}, got {x.size(1)}"
            )

        logits = self.fc(x)
        return logits











