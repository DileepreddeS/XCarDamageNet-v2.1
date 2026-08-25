"""Auxiliary damage-understanding modules."""

from xcar.modules.adapter import FeatureTokenAdapter
from xcar.modules.attention_head import AttentionMapHead
from xcar.modules.contrastive import ContrastiveDamageModule
from xcar.modules.implied_class_head import ImpliedClassHead
from xcar.modules.physics_encoder import PhysicsTokenEncoder

__all__ = [
    "FeatureTokenAdapter",
    "AttentionMapHead",
    "ContrastiveDamageModule",
    "ImpliedClassHead",
    "PhysicsTokenEncoder",
]
