from __future__ import annotations

import torch
import torch.nn as nn


class FeatureTokenAdapter(nn.Module):
    """Project YOLO CNN features to token sequence for physics encoder."""

    def __init__(self, in_ch: int, token_dim: int = 384):
        """
        Args:
            in_ch: P3 neck output channels, read from the built model at runtime.
            token_dim: Output token dimension. 384 to match PhysicsTokenEncoder.
        """
        super().__init__()
        self.in_ch = in_ch
        self.token_dim = token_dim
        self.proj = nn.Conv2d(in_ch, token_dim, kernel_size=1)
        self.norm = nn.LayerNorm(token_dim)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feat: (B, C, H, W) P3 neck features.
        Returns:
            (B, H*W, token_dim) layer-normalised tokens.
        """
        t = self.proj(feat)                  # (B, 384, H, W)
        t = t.flatten(2).transpose(1, 2)     # (B, H*W, 384) — non-contiguous
        return self.norm(t)
