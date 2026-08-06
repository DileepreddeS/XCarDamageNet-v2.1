from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PhysicsConsistencyLoss(nn.Module):
    """Cross-entropy between physics-implied class and predicted class.

    Uses the FraudHead's `implied_logits` output as the physics prediction.
    Backpropagates through both the physics head and the detector.
    """

    def __init__(self, reduction: str = "mean") -> None:
        """
        Args:
            reduction: 'mean', 'sum', or 'none'.
        """
        super().__init__()
        assert reduction in ("mean", "sum", "none")
        self.reduction = reduction

    def forward(
        self,
        physics_implied_logits: torch.Tensor,
        predicted_class_logits: torch.Tensor,
    ) -> torch.Tensor:
        """Compute physics consistency loss.

        Args:
            physics_implied_logits: (B, num_classes) — logits from FraudHead
                representing what the physics tokens imply the damage class should be
            predicted_class_logits: (B, num_classes) — detection head class logits

        Returns:
            loss: physics consistency cross-entropy loss
        """
        # Not detached: this loss updates the physics head and the detector.
        physics_probs = F.softmax(physics_implied_logits, dim=-1)

        # CE with soft labels: -sum(p_physics * log(p_pred))
        log_pred = F.log_softmax(predicted_class_logits, dim=-1)
        loss = -(physics_probs * log_pred).sum(dim=-1)  # (B,)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss
