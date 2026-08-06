"""Auxiliary losses: attention supervision, contrastive triplet, physics consistency."""

from xcar.losses.attention_loss import AttentionSupervisionLoss
from xcar.losses.contrastive_loss import ContrastiveTripletLoss
from xcar.losses.physics_loss import PhysicsConsistencyLoss

__all__ = [
    "AttentionSupervisionLoss",
    "ContrastiveTripletLoss",
    "PhysicsConsistencyLoss",
]
