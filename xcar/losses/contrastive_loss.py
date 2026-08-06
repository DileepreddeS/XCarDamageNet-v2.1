from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class ContrastiveTripletLoss(nn.Module):
    """Triplet margin loss for damage vs normal surface discrimination.

    Operates on token-level features from the contrastive module.
    Requires triplets: (anchor_damage, positive_damage, negative_normal).
    """

    def __init__(self, margin: float = 1.0, reduction: str = "mean") -> None:
        """
        Args:
            margin: Triplet margin. 1.0 per spec.
            reduction: 'mean', 'sum', or 'none'.
        """
        super().__init__()
        self.margin = margin
        assert reduction in ("mean", "sum", "none")
        self.reduction = reduction
        self.triplet_loss = nn.TripletMarginLoss(
            margin=margin, p=2, reduction=reduction
        )

    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            anchor:   (N, D) damage token features (anchor)
            positive: (N, D) damage tokens of same class (pulled toward anchor)
            negative: (N, D) normal surface tokens (pushed away from anchor)

        Returns:
            loss: scalar triplet margin loss
        """
        return self.triplet_loss(anchor, positive, negative)

    @staticmethod
    def mine_triplets(
        tokens: torch.Tensor,
        damage_mask: torch.Tensor,
        class_ids: torch.Tensor,
        n_triplets: int = 64,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample (anchor, positive, negative) triplets from a batch of tokens.

        Args:
            tokens:      (B, N, D) token features
            damage_mask: (B, N) bool — True = damage token
            class_ids:   (B, N) int — class of each damage token (-1 = normal)
            n_triplets:  Number of triplets to sample

        Returns:
            anchor:   (M, D)
            positive: (M, D)
            negative: (M, D)
            where M ≤ n_triplets
        """
        B, N, D = tokens.shape
        anchors, positives, negatives = [], [], []

        for b in range(B):
            dmask = damage_mask[b]   # (N,)
            nmask = ~dmask           # normal tokens
            damage_tokens = tokens[b][dmask]       # (n_d, D)
            damage_classes = class_ids[b][dmask]   # (n_d,)
            normal_tokens = tokens[b][nmask]       # (n_n, D)

            if damage_tokens.shape[0] < 2 or normal_tokens.shape[0] == 0:
                continue

            unique_classes = damage_classes.unique()
            for cls in unique_classes:
                cls_mask = (damage_classes == cls)
                cls_tokens = damage_tokens[cls_mask]
                if cls_tokens.shape[0] < 2:
                    continue

                n_sample = min(n_triplets // (B * max(1, len(unique_classes))), cls_tokens.shape[0] - 1)
                n_sample = max(1, n_sample)

                for _ in range(n_sample):
                    idx = torch.randperm(cls_tokens.shape[0])[:2]
                    anchors.append(cls_tokens[idx[0]])
                    positives.append(cls_tokens[idx[1]])
                    neg_idx = torch.randint(normal_tokens.shape[0], (1,)).item()
                    negatives.append(normal_tokens[neg_idx])

        if not anchors:
            dummy = tokens.new_zeros(1, tokens.shape[-1])
            return dummy, dummy, dummy

        return (
            torch.stack(anchors),
            torch.stack(positives),
            torch.stack(negatives),
        )
