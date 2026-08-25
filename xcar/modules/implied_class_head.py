"""Physics-implied class head.

Pools physics tokens (B, N, 396) to an image-level representation and projects
to damage-class logits (B, 6) — what the physics evidence alone implies the
damage is. `L_physics` compares these against the detector's class logits.

Checkpoint note: this module replaces the two-output head that lived at
`xcar/modules/fraud_head.py` up to commit cf9c6ff. Its second output and the
loss term that consumed it were removed. State-dict keys moved from
`aux.fraud_head.mlp.{0,2}` to `aux.implied_head.proj.0` / `.implied_class`, so
checkpoints written before that commit will not load these weights by name —
the rest of the model is unaffected.
"""

from __future__ import annotations

import torch
import torch.nn as nn

NUM_CLASSES = 6  # dent, scratch, crack, glass_shatter, lamp_broken, tire_flat


class ImpliedClassHead(nn.Module):
    """Physics tokens -> image-level damage-class logits."""

    def __init__(self, physics_dim: int = 396) -> None:
        super().__init__()
        self.physics_dim = physics_dim
        self.proj = nn.Sequential(
            nn.Linear(physics_dim, 64),
            nn.SiLU(inplace=True),
        )
        self.implied_class = nn.Linear(64, NUM_CLASSES)

    def forward(self, physics_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            physics_tokens: (B, N, 396) — from the physics encoder.

        Returns:
            implied_logits: (B, NUM_CLASSES) — physics-implied damage class.
        """
        pooled = physics_tokens.mean(dim=1)  # (B, 396)
        return self.implied_class(self.proj(pooled))
