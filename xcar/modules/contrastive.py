"""Within-image contrastive damage module.

Compares suspicious tokens against the centroid of the normal (non-suspicious)
tokens of the same image, and adds a scaled residual to the suspicious ones:

    normal_centroid = mean(normal_tokens)
    residual        = tokens - normal_centroid
    damage_score    = 1 - cosine_similarity(tokens, normal_centroid)
    output          = tokens + alpha * residual   (suspicious tokens only)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class ContrastiveDamageModule(nn.Module):
    """Within-image contrastive module that amplifies damage deviation signals.

    Input:  (B, N, 396) physics-augmented tokens + (B, N) suspicious mask
    Output: (B, N, 396) tokens with damage residual added for suspicious tokens

    ~0.5M parameters (alpha scalar + optional projection layer).
    """

    def __init__(
        self,
        token_dim: int = 396,
        use_projection: bool = True,
    ) -> None:
        """
        Args:
            token_dim: Token feature dimension. Default 396.
            use_projection: If True, adds a linear projection before computing
                residuals (helps align suspicious/normal tokens in a shared
                comparison space). Adds ~156K params.
        """
        super().__init__()
        self.token_dim = token_dim

        # Learnable scale for residual addition — init 0.5 as specified
        self.alpha = nn.Parameter(torch.tensor(0.5))

        # Optional projection into comparison space
        self.use_projection = use_projection
        if use_projection:
            self.proj = nn.Linear(token_dim, token_dim, bias=False)

    def forward(
        self,
        tokens: torch.Tensor,
        suspicious_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute within-image contrastive damage residuals.

        Args:
            tokens: Physics-augmented tokens (B, N, 396).
            suspicious_mask: Boolean mask (B, N) where True = suspicious token.
                If None, all tokens are treated as suspicious (no normal centroid
                will be available — falls back to global mean).

        Returns:
            output: Residual-augmented tokens (B, N, 396).
            damage_scores: Per-suspicious-token damage score 0-1 (B, N).
                0 = identical to normal surface, 1 = maximally different.
        """
        B, N, D = tokens.shape
        device = tokens.device

        # Project for comparison if configured
        if self.use_projection:
            proj_tokens = self.proj(tokens)  # (B, N, D)
        else:
            proj_tokens = tokens

        # Build default mask: all tokens suspicious if none provided
        if suspicious_mask is None:
            suspicious_mask = torch.ones(B, N, dtype=torch.bool, device=device)

        normal_mask = ~suspicious_mask  # (B, N)

        # Compute per-image normal centroid
        # If no normal tokens in an image, fall back to all-token mean
        normal_centroid = self._compute_normal_centroid(
            proj_tokens, normal_mask
        )  # (B, 1, D)

        # Compute residual for ALL tokens (suspicious tokens get amplified signal)
        residual = proj_tokens - normal_centroid  # (B, N, D)

        # Damage score = 1 - cosine_similarity(token, centroid)
        # Shape: (B, N)
        damage_scores = 1.0 - F.cosine_similarity(
            proj_tokens, normal_centroid.expand_as(proj_tokens), dim=-1
        )  # (B, N), range [0, 2] for cosine, clipped to [0, 1]
        damage_scores = damage_scores.clamp(0.0, 1.0)

        # Apply residual only to suspicious tokens (normal tokens pass through)
        # suspicious_mask: (B, N) → (B, N, 1) for broadcasting
        susp_mask_3d = suspicious_mask.unsqueeze(-1).float()  # (B, N, 1)
        output = tokens + self.alpha * residual * susp_mask_3d  # (B, N, D)

        return output, damage_scores

    def _compute_normal_centroid(
        self, proj_tokens: torch.Tensor, normal_mask: torch.Tensor
    ) -> torch.Tensor:
        """Compute mean of normal (non-suspicious) tokens per image.

        Falls back to global mean when no normal tokens exist (e.g. heavily
        damaged vehicle where most patches are flagged suspicious).

        Args:
            proj_tokens: (B, N, D)
            normal_mask: (B, N) bool — True = normal token

        Returns:
            centroid: (B, 1, D)
        """
        B, N, D = proj_tokens.shape
        centroids = torch.zeros(B, 1, D, device=proj_tokens.device)

        for b in range(B):
            mask = normal_mask[b]  # (N,)
            if mask.sum() > 0:
                normal_tokens = proj_tokens[b][mask]  # (n_normal, D)
                centroids[b, 0] = normal_tokens.mean(dim=0)
            else:
                # Fallback: global mean when all tokens are suspicious
                centroids[b, 0] = proj_tokens[b].mean(dim=0)

        return centroids  # (B, 1, D)
