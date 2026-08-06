from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionSupervisionLoss(nn.Module):
    """BCE supervision for AttentionMapHead outputs against GT box masks.

    Creates binary mask at attention map resolution for each GT box,
    then applies BCE. Only the channel matching the GT class is supervised.
    """

    def __init__(self, reduction: str = "mean") -> None:
        """
        Args:
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        assert reduction in ("mean", "sum", "none")
        self.reduction = reduction

    def forward(
        self,
        attn_maps: torch.Tensor,
        gt_boxes: list[torch.Tensor],
        gt_classes: list[torch.Tensor],
    ) -> torch.Tensor:
        """
        Args:
            attn_maps:  (B, num_classes, H, W) from AttentionMapHead
            gt_boxes:   list[tensor(n_i, 4)] GT boxes per image in [x1,y1,x2,y2]
                        normalised to [0, 1] (relative coords)
            gt_classes: list[tensor(n_i,)] integer class ids per image

        Returns:
            loss: scalar attention supervision loss
        """
        B, C, H, W = attn_maps.shape
        total_loss = attn_maps.new_zeros(1)
        n_targets = 0

        for b in range(B):
            boxes = gt_boxes[b]    # (n_i, 4) normalised
            classes = gt_classes[b]  # (n_i,)

            if boxes.numel() == 0:
                continue

            for i in range(len(boxes)):
                cls_id = int(classes[i].item())
                if cls_id >= C:
                    continue

                # Create binary mask at attention map resolution
                gt_mask = torch.zeros(H, W, device=attn_maps.device)
                x1 = int(boxes[i, 0].item() * W)
                y1 = int(boxes[i, 1].item() * H)
                x2 = int(boxes[i, 2].item() * W)
                y2 = int(boxes[i, 3].item() * H)

                # Clamp to valid range
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(W - 1, x2), min(H - 1, y2)

                if x2 > x1 and y2 > y1:
                    gt_mask[y1:y2, x1:x2] = 1.0

                pred = attn_maps[b, cls_id]  # (H, W) — already in [0,1] from sigmoid
                loss = F.binary_cross_entropy(pred, gt_mask, reduction="mean")
                total_loss = total_loss + loss
                n_targets += 1

        if n_targets == 0:
            return total_loss.squeeze()

        if self.reduction == "mean":
            return (total_loss / n_targets).squeeze()
        return total_loss.squeeze()
