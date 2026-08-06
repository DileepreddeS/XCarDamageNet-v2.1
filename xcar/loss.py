from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.ops import xywh2xyxy

from xcar.losses.attention_loss import AttentionSupervisionLoss
from xcar.losses.contrastive_loss import ContrastiveTripletLoss
from xcar.losses.physics_loss import PhysicsConsistencyLoss

W_ATTN = 0.10
W_CONTRAST = 0.05
W_PHYSICS = 0.02
W_FRAUD = 0.01  # weakest aux weight — a prior on clean data, not supervision

MAX_MINING_TOKENS = 2048  # cap candidates before Python-loop triplet mining
N_TRIPLETS = 64


class XCarLoss(v8DetectionLoss):
    """v8DetectionLoss + attention / contrastive / physics auxiliary terms."""

    def __init__(self, model, tal_topk: int = 10, tal_topk2: int | None = None):
        super().__init__(model, tal_topk=tal_topk, tal_topk2=tal_topk2)
        self.xcar_model = model  # de-paralleled XCarDetectionModel

        self.use_attention = bool(getattr(model, "use_attention", False))
        self.use_contrastive = bool(getattr(model, "use_contrastive", False))
        self.use_physics = bool(getattr(model, "use_physics", False))

        self.attn_loss = AttentionSupervisionLoss(reduction="mean")
        self.contrast_loss = ContrastiveTripletLoss(margin=1.0, reduction="mean")
        self.physics_loss = PhysicsConsistencyLoss(reduction="mean")

        self.w_attn = W_ATTN
        self.w_contrast = W_CONTRAST
        self.w_physics = W_PHYSICS
        self.w_fraud = W_FRAUD

        # Diagnostics filled per call — read by scripts/smoke_test.py and
        # logged each epoch by XCarTrainer.
        self.last_stats: dict[str, Any] = {}

    # ------------------------------------------------------------------
    @property
    def aux_loss_names(self) -> tuple[str, ...]:
        """Delegate to the model so there is ONE definition of term ordering.

        `loss_items` must line up with `trainer.loss_names` column for column.
        Two independent copies of this list drifting apart would silently
        mislabel every logged loss, so the model owns it and __call__ asserts
        the terms it actually produced match.
        """
        return tuple(self.xcar_model.aux_loss_names)

    # ------------------------------------------------------------------
    def __call__(self, preds, batch):
        """Stock YOLO loss plus the active aux terms."""
        parsed = self.parse_output(preds)
        loss, loss_items = super().__call__(preds, batch)  # (3,)*bs , (3,)
        batch_size = int(parsed["scores"].shape[0])

        aux = getattr(self.xcar_model, "_aux", None) or {}
        if not aux:
            raise RuntimeError(
                "XCarLoss found an empty model._aux stash. The aux modules did "
                "not run in this forward pass — check XCarDetectionModel.predict."
            )

        # (name, weight, value) — order here IS the logged column order.
        terms: list[tuple[str, float, torch.Tensor]] = []
        stats: dict[str, Any] = {}

        if self.use_attention:
            terms.append(("attn_loss", self.w_attn, self._attention_term(aux, batch)))

        if self.use_contrastive:
            terms.append(("cont_loss", self.w_contrast, self._contrastive_term(aux, batch, stats)))

        if self.use_physics:
            terms.append(("phys_loss", self.w_physics, self._physics_term(aux, parsed, stats)))
            terms.append(("fraud_loss", self.w_fraud, self._fraud_term(aux)))

        for name, _, value in terms:
            stats[name] = float(value.detach())
        self.last_stats = stats

        if not terms:
            return loss, loss_items

        produced = tuple(n for n, _, _ in terms)
        if produced != self.aux_loss_names:
            raise RuntimeError(
                f"aux term order mismatch: XCarLoss produced {produced} but "
                f"the model declares {self.aux_loss_names}. loss_items would be "
                "mislabelled against trainer.loss_names."
            )

        total = torch.cat([loss, torch.stack([w * v * batch_size for _, w, v in terms])])
        items = torch.cat([loss_items, torch.stack([v.detach() for _, _, v in terms]).reshape(-1)])
        return total, items

    # ------------------------------------------------------------------
    # aux terms
    # ------------------------------------------------------------------
    def _attention_term(self, aux: dict, batch: dict) -> torch.Tensor:
        """L_attn — BCE of per-class heatmaps against GT box masks."""
        attn_maps = aux.get("attn_maps")
        if attn_maps is None:
            raise RuntimeError("use_attention is set but _aux has no 'attn_maps'.")

        gt_boxes, gt_cls = self._per_image_targets(batch, attn_maps.shape[0])
        if sum(int(b.shape[0]) for b in gt_boxes) == 0:
            # Zero that still carries grad_fn.
            return attn_maps.sum() * 0
        return self.attn_loss(attn_maps, gt_boxes, gt_cls)

    def _contrastive_term(self, aux: dict, batch: dict, stats: dict) -> torch.Tensor:
        """L_contrast — triplet margin loss over mined damage/normal tokens."""
        tokens = aux.get("tokens_contrastive")
        if tokens is None:
            raise RuntimeError("use_contrastive is set but _aux has no 'tokens_contrastive'.")

        th, tw = aux["token_hw"]
        damage_mask, class_ids = self._token_damage_targets(batch, tokens.shape[0], th, tw, tokens.device)

        tok, dmask, cids = self._subsample_tokens(tokens, damage_mask, class_ids)
        stats["n_damage_tokens"] = int(dmask.sum())

        if not self._triplets_available(dmask, cids):
            stats["n_triplets"] = 0
            return tokens.sum() * 0

        anchor, positive, negative = ContrastiveTripletLoss.mine_triplets(
            tok, dmask, cids, n_triplets=N_TRIPLETS
        )
        stats["n_triplets"] = int(anchor.shape[0])
        return self.contrast_loss(anchor, positive, negative)

    def _physics_term(self, aux: dict, parsed: dict, stats: dict) -> torch.Tensor:
        """L_physics — soft CE between physics-implied class and detector class.

        Image-level detector class logits are the per-class maximum over all
        anchors, i.e. the most confident detection of each class in the image.
        `parsed["scores"]` is (B, nc, num_anchors) pre-sigmoid.

        This term updates both heads, which makes mutual collapse possible: the
        loss is linear in the physics distribution, so gradient descent pushes
        it onto a one-hot at the detector's argmax while the detector is pulled
        the other way. Two diagnostics are recorded every call:

            phys_entropy  ln(6)=1.792 at uniform -> 0.0 on collapse
            phys_agree    fraction with argmax(physics)==argmax(detector)
                          -> 1.00 on collapse

        Entropy decaying toward 0 with agreement pinned at 1.00 and phys_loss
        near 0 is the collapse signature.
        """
        implied = aux.get("fraud_implied")
        if implied is None:
            raise RuntimeError("use_physics is set but _aux has no 'fraud_implied'.")
        pred_class_logits = parsed["scores"].amax(dim=2)  # (B, nc)

        with torch.no_grad():
            probs = implied.detach().softmax(dim=-1)
            entropy = -(probs * probs.clamp_min(1e-12).log()).sum(dim=-1).mean()
            agree = (implied.detach().argmax(-1) == pred_class_logits.detach().argmax(-1)).float().mean()
            stats["phys_entropy"] = float(entropy)
            stats["phys_agree"] = float(agree)

        return self.physics_loss(implied, pred_class_logits)

    def _fraud_term(self, aux: dict) -> torch.Tensor:
        """L_fraud — weak prior that training images are not fraudulent.

        The dataset carries no fraud labels, so every training image is assumed
        clean and `fraud_score` is pushed toward 0 by a one-sided BCE. Two
        limits on what this buys:

          * It gives FraudHead a gradient path. That is its main purpose.
          * With no positive examples it cannot teach discrimination — a head
            that outputs 0 for every input satisfies it perfectly. Any fraud
            AUC number must come from held-out data with real positives, not
            from this term converging.

        `fraud_score` is already sigmoid-activated in FraudHead, so BCE (not
        BCEWithLogits) is correct here. PyTorch clamps the log at -100, so a
        saturated score cannot produce inf/NaN.
        """
        fraud_score = aux.get("fraud_score")
        if fraud_score is None:
            raise RuntimeError("use_physics is set but _aux has no 'fraud_score'.")
        return F.binary_cross_entropy(fraud_score, torch.zeros_like(fraud_score))

    # ------------------------------------------------------------------
    # target construction
    # ------------------------------------------------------------------
    def _per_image_targets(self, batch: dict, batch_size: int):
        """Split GT into per-image [x1,y1,x2,y2] in [0,1] plus class ids.

        `batch["bboxes"]` is YOLO-format normalised xywh; the conversion to
        xyxy happens here rather than inside the attention loss.
        """
        bi = batch["batch_idx"].reshape(-1).long().to(self.device)
        cls = batch["cls"].reshape(-1).long().to(self.device)
        xywh = batch["bboxes"].to(self.device)
        xyxy = xywh2xyxy(xywh) if xywh.numel() else xywh.reshape(0, 4)

        boxes, classes = [], []
        for b in range(batch_size):
            sel = bi == b
            boxes.append(xyxy[sel])
            classes.append(cls[sel])
        return boxes, classes

    def _token_damage_targets(
        self, batch: dict, batch_size: int, th: int, tw: int, device
    ):
        """Mark tokens whose grid-cell centre falls inside a GT box.

        Returns:
            damage_mask: (B, th*tw) bool
            class_ids:   (B, th*tw) long, -1 for normal tokens
        """
        boxes, classes = self._per_image_targets(batch, batch_size)

        ys = (torch.arange(th, device=device, dtype=torch.float32) + 0.5) / th
        xs = (torch.arange(tw, device=device, dtype=torch.float32) + 0.5) / tw
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")  # non-contiguous -> reshape only
        gx = gx.reshape(-1)
        gy = gy.reshape(-1)

        n_tok = th * tw
        damage_mask = torch.zeros(batch_size, n_tok, dtype=torch.bool, device=device)
        class_ids = torch.full((batch_size, n_tok), -1, dtype=torch.long, device=device)

        for b in range(batch_size):
            bb, cc = boxes[b], classes[b]
            for i in range(bb.shape[0]):
                x1, y1, x2, y2 = (float(v) for v in bb[i])
                inside = (gx >= x1) & (gx <= x2) & (gy >= y1) & (gy <= y2)
                if not bool(inside.any()):
                    continue
                damage_mask[b] |= inside
                class_ids[b][inside] = int(cc[i])
        return damage_mask, class_ids

    @staticmethod
    def _subsample_tokens(
        tokens: torch.Tensor,
        damage_mask: torch.Tensor,
        class_ids: torch.Tensor,
        max_tokens: int = MAX_MINING_TOKENS,
    ):
        """Cap candidate tokens per image before Python-loop triplet mining.

        At 16,384 tokens `mine_triplets` is far too slow. Keeps up to half the
        budget as damage tokens (so triplets stay available when damage is
        rare) and fills the rest with normal tokens. Selected indices are
        always distinct — a duplicated token would make anchor == positive.
        """
        B, N, D = tokens.shape
        if N <= max_tokens:
            return tokens, damage_mask, class_ids

        device = tokens.device
        sel_rows = []
        for b in range(B):
            d_idx = damage_mask[b].nonzero(as_tuple=False).reshape(-1)
            n_idx = (~damage_mask[b]).nonzero(as_tuple=False).reshape(-1)
            d_idx = d_idx[torch.randperm(d_idx.numel(), device=device)]
            n_idx = n_idx[torch.randperm(n_idx.numel(), device=device)]

            d_keep = d_idx[: max_tokens // 2]
            n_keep = n_idx[: max_tokens - d_keep.numel()]
            sel = torch.cat([d_keep, n_keep])
            if sel.numel() < max_tokens:  # damage-dominated image: top up from spare damage tokens
                spare = d_idx[d_keep.numel():][: max_tokens - sel.numel()]
                sel = torch.cat([sel, spare])
            sel_rows.append(sel)

        idx = torch.stack(sel_rows)  # (B, max_tokens)
        assert idx.shape[1] == max_tokens, f"subsample produced {idx.shape[1]} tokens"
        tok = torch.gather(tokens, 1, idx.unsqueeze(-1).expand(-1, -1, D))
        return tok, torch.gather(damage_mask, 1, idx), torch.gather(class_ids, 1, idx)

    @staticmethod
    def _triplets_available(damage_mask: torch.Tensor, class_ids: torch.Tensor) -> bool:
        """Mirror `mine_triplets`' own preconditions before calling it.

        `mine_triplets` returns zero-filled dummies when it finds nothing, and
        those produce a constant `margin` loss with no gradient. Checking first
        lets the caller return a zero-with-grad instead.
        """
        B = damage_mask.shape[0]
        for b in range(B):
            dm = damage_mask[b]
            if int(dm.sum()) < 2 or int((~dm).sum()) == 0:
                continue
            cls_b = class_ids[b][dm]
            counts = torch.bincount(cls_b.clamp(min=0))
            if bool((counts >= 2).any()):
                return True
        return False
