"""Fraud head.

Maps physics tokens (B, N, 396) to an image-level fraud score (B, 1) and
physics-implied damage class logits (B, 6).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Tuple

NUM_CLASSES = 6  # dent, scratch, crack, glass_shatter, lamp_broken, tire_flat


class FraudHead(nn.Module):
    """Detects physics-inconsistent damage claims (fraud indicator).

    Compares physics-implied damage type (from physics tokens) with the
    predicted detection label. High disagreement → high fraud probability.
    """

    def __init__(self, physics_dim: int = 396) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(physics_dim, 64),
            nn.SiLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
        # Also produces implied class logits for physics consistency loss
        self.implied_class = nn.Linear(64, NUM_CLASSES)

    def forward(
        self, physics_tokens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            physics_tokens: (B, N, 396) — from physics encoder

        Returns:
            fraud_score:     (B, 1) — fraud probability in [0,1]
            implied_logits:  (B, NUM_CLASSES) — physics-implied damage class
        """
        # Aggregate physics tokens to image-level representation
        pooled = physics_tokens.mean(dim=1)  # (B, 396)

        hidden = self.mlp[0](pooled)  # Linear: (B, 64)
        hidden = self.mlp[1](hidden)  # SiLU
        implied_logits = self.implied_class(hidden)  # (B, NUM_CLASSES)

        fraud_score = self.mlp[2](hidden)  # Linear: (B, 1)
        fraud_score = self.mlp[3](fraud_score)  # Sigmoid

        return fraud_score, implied_logits
