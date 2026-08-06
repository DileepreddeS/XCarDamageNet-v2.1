from __future__ import annotations

import torch
import torch.nn as nn

NUM_CLASSES = 6  # dent, scratch, crack, glass_shatter, lamp_broken, tire_flat


class AttentionMapHead(nn.Module):
    """Produces per-class damage heatmaps from P3 (finest resolution) features.

    Output: (B, NUM_CLASSES, H/8, W/8) — trained via AttentionSupervisionLoss
    to focus on actual damage regions (EU AI Act explainability requirement).
    """

    def __init__(self, in_ch: int, num_classes: int = NUM_CLASSES) -> None:
        """
        Args:
            in_ch: P3 neck output channel count, read from the built model at
                runtime.
            num_classes: Number of damage classes. 6 for CarDD.
        """
        super().__init__()
        self.in_ch = in_ch
        self.num_classes = num_classes
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.SiLU(inplace=True),
            nn.Conv2d(128, num_classes, 1),
            nn.Sigmoid(),
        )

    def forward(self, p3: torch.Tensor) -> torch.Tensor:
        """
        Args:
            p3: (B, in_ch, H/8, W/8) — finest scale features from neck
        Returns:
            attn_maps: (B, 6, H/8, W/8) — per-class heatmaps in [0,1]
        """
        return self.conv(p3)
