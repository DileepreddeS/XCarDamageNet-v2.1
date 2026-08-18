from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from ultralytics.nn.modules.head import Detect
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import LOGGER

from xcar.modules.adapter import FeatureTokenAdapter
from xcar.modules.attention_head import AttentionMapHead
from xcar.modules.contrastive import ContrastiveDamageModule
from xcar.modules.fraud_head import FraudHead
from xcar.modules.physics_encoder import PhysicsTokenEncoder

TOKEN_DIM = 384      # PhysicsTokenEncoder input dim
PHYSICS_DIM = 396    # PhysicsTokenEncoder output dim (384+3+6+2+1)
NUM_CLASSES = 6      # CarDD


class _FeatureTap:
    """Picklable forward hook that stashes a layer's output into `store[key]`.

    A module-level class rather than a closure: closures are not picklable, so
    `torch.save` of the model would fail, and a closure would capture the
    original model, leaving a `deepcopy` (as ultralytics does for EMA) writing
    its hooked features into the original model's stash.
    """

    def __init__(self, store: dict, key: str) -> None:
        self.store = store
        self.key = key

    def __call__(self, _module, _inputs, output):
        if isinstance(output, torch.Tensor):
            self.store[self.key] = output


# --------------------------------------------------------------------------
# Runtime introspection helpers — in_ch is always read, never hardcoded
# --------------------------------------------------------------------------
def find_detect_module(model: nn.Module) -> Detect:
    """Return the Detect head of a built ultralytics model."""
    m = model.model[-1]
    if not isinstance(m, Detect):
        raise TypeError(f"Expected Detect as last layer, found {type(m).__name__}")
    return m


def get_neck_channels(model: nn.Module) -> tuple[int, ...]:
    """Read (P3, P4, P5) neck output channels from the BUILT model.

    Derived from Detect's box-regression stem input channels, which are
    constructed directly from the neck channel tuple `ch`.
    """
    detect = find_detect_module(model)
    chans = []
    for branch in detect.cv2:
        conv = branch[0].conv  # Conv(x, c2, 3) -> .conv is the nn.Conv2d
        chans.append(int(conv.in_channels))
    return tuple(chans)


def get_p3_layer_index(model: nn.Module) -> int:
    """Index into `model.model` of the layer whose output is Detect's P3 input."""
    detect = find_detect_module(model)
    srcs = detect.f
    if not isinstance(srcs, (list, tuple)) or len(srcs) < 3:
        raise ValueError(f"Unexpected Detect.f: {srcs!r}")
    return int(srcs[0])


def get_p4_layer_index(model: nn.Module) -> int:
    """Index into `model.model` of the layer whose output is Detect's P4 input."""
    return int(find_detect_module(model).f[1])


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
class XCarDetectionModel(DetectionModel):
    """YOLO11m detection model with aux heads hanging off the neck."""

    def __init__(
        self,
        cfg: str = "yolo11m.yaml",
        ch: int = 3,
        nc: int | None = NUM_CLASSES,
        verbose: bool = True,
        *,
        use_attention: bool = False,
        use_physics: bool = False,
        use_contrastive: bool = False,
        attach: str = "p3",
        token_stride: int = 1,
        suspicious_thresh: float = 0.5,
    ) -> None:
        """
        Args:
            use_attention:  enable AttentionMapHead (+ L_attn).
            use_physics:    enable adapter + PhysicsTokenEncoder + FraudHead
                            (+ L_physics, + L_fraud).
            use_contrastive: enable ContrastiveDamageModule (+ L_contrast).
            attach: "p3" (default, 128x128 = 16,384 tokens @1024px) or "p4"
                (64x64 = 4,096 tokens), which trades token resolution for
                memory. The AttentionMapHead always stays on P3 either way.
            token_stride: spatial subsample factor for physics/contrastive
                tokens only (2 -> 4,096 tokens from P3).
            suspicious_thresh: threshold on max-over-class attention used to
                build the contrastive module's suspicious mask. Only used when
                both attention and contrastive are enabled.
        """
        # These must exist BEFORE the parent's stride-init forward, which calls
        # our overridden predict(). Aux modules must NOT exist yet.
        self._aux_ready = False
        self._feats: dict[str, torch.Tensor] = {}
        self._hook_handles: list[Any] = []
        self._aux: dict[str, Any] = {}
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

        if attach not in ("p3", "p4"):
            raise ValueError(f"attach must be 'p3' or 'p4', got {attach!r}")
        if token_stride < 1:
            raise ValueError(f"token_stride must be >= 1, got {token_stride}")

        self.use_attention = bool(use_attention)
        self.use_physics = bool(use_physics)
        self.use_contrastive = bool(use_contrastive)
        self.attach = attach
        self.token_stride = int(token_stride)
        self.suspicious_thresh = float(suspicious_thresh)

        if self.use_contrastive and not self.use_physics:
            raise ValueError(
                "use_contrastive requires use_physics — the contrastive module "
                "consumes 396-dim physics tokens."
            )

        neck_ch = get_neck_channels(self)
        self.p3_ch, self.p4_ch, self.p5_ch = neck_ch[0], neck_ch[1], neck_ch[2]
        self.p3_layer_idx = get_p3_layer_index(self)
        self.p4_layer_idx = get_p4_layer_index(self)
        self.token_src_ch = self.p3_ch if attach == "p3" else self.p4_ch

        # --- aux modules -------------------------------------------------
        aux: dict[str, nn.Module] = {}
        if self.use_attention:
            # Attention head is always on P3, regardless of `attach`.
            aux["attn_head"] = AttentionMapHead(in_ch=self.p3_ch, num_classes=NUM_CLASSES)
        if self.use_physics:
            aux["adapter"] = FeatureTokenAdapter(in_ch=self.token_src_ch, token_dim=TOKEN_DIM)
            aux["physics"] = PhysicsTokenEncoder(in_dim=TOKEN_DIM)
            aux["fraud_head"] = FraudHead(physics_dim=PHYSICS_DIM)
        if self.use_contrastive:
            aux["contrastive"] = ContrastiveDamageModule(token_dim=PHYSICS_DIM, use_projection=True)
        self.aux = nn.ModuleDict(aux)

        # --- forward hooks on neck outputs -------------------------------
        self._register_neck_hooks()
        self._aux_ready = bool(self.aux)

        LOGGER.info(
            f"XCarDetectionModel | neck channels P3/P4/P5 = {neck_ch} | "
            f"P3 layer idx = {self.p3_layer_idx} | attach = {self.attach} | "
            f"token_stride = {self.token_stride} | "
            f"attention={self.use_attention} physics={self.use_physics} "
            f"contrastive={self.use_contrastive}"
        )

    # ------------------------------------------------------------------
    # hooks
    # ------------------------------------------------------------------
    def _register_neck_hooks(self) -> None:
        """Hook the neck layers feeding Detect so aux modules can read them."""
        self._remove_neck_hooks()
        self._purge_taps()  # drop any tap that arrived via deepcopy/unpickle
        self._hook_handles.append(
            self.model[self.p3_layer_idx].register_forward_hook(_FeatureTap(self._feats, "p3"))
        )
        if self.attach == "p4":
            self._hook_handles.append(
                self.model[self.p4_layer_idx].register_forward_hook(_FeatureTap(self._feats, "p4"))
            )

    def _remove_neck_hooks(self) -> None:
        for h in self._hook_handles:
            h.remove()
        self._hook_handles = []

    def _purge_taps(self) -> None:
        """Remove every _FeatureTap from the module tree.

        A deepcopied or unpickled model carries copies of the old taps, and
        those point at a stale `_feats` dict. Purging before re-registering
        keeps exactly one live tap per attach point.
        """
        for m in self.modules():
            hooks = getattr(m, "_forward_hooks", None)
            if not hooks:
                continue
            stale = [k for k, v in hooks.items() if isinstance(v, _FeatureTap)]
            for k in stale:
                hooks.pop(k, None)
                getattr(m, "_forward_hooks_with_kwargs", {}).pop(k, None)
                getattr(m, "_forward_hooks_always_called", {}).pop(k, None)

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------
    def predict(self, x, profile=False, visualize=False, augment=False, embed=None):
        """Stock prediction, then run aux modules on the hooked neck features."""
        # .clear() not rebinding: the taps hold a reference to THIS dict object.
        self._feats.clear()
        out = super().predict(x, profile=profile, visualize=visualize, augment=augment, embed=embed)

        if self._aux_ready and not augment and embed is None:
            self._aux = self._forward_aux()
        else:
            self._aux = {}
        # Never leave a feature map in the stash: it would otherwise be pickled
        # into every checkpoint via the taps.
        self._feats.clear()
        return out

    def _forward_aux(self) -> dict[str, Any]:
        """Run aux modules on hooked neck features. Returns the `_aux` stash."""
        p3 = self._feats.get("p3")
        if p3 is None:
            raise RuntimeError(
                "P3 hook produced no feature — the neck layer index is stale. "
                "Did you reload/rebuild `self.model` without re-registering hooks?"
            )

        aux: dict[str, Any] = {}

        # Aux modules see a DETACHED view of the neck features, so their
        # gradients stop at the aux module boundary and never reach the
        # backbone or neck. The detection path is untouched: Detect consumes
        # the original tensor straight from the layer output, and our forward
        # hooks only observe it. So P3 still gets full detection gradient --
        # what it no longer gets is L_attn / L_physics / L_contrast pulling the
        # shared representation around while the detector is trying to learn.
        p3_for_aux = p3.detach()

        # --- attention maps (always P3, full resolution) -----------------
        attn_maps = None
        if self.use_attention:
            attn_maps = self.aux["attn_head"](p3_for_aux)  # (B, 6, H3, W3)
            aux["attn_maps"] = attn_maps

        if not self.use_physics:
            return aux

        # --- token source ------------------------------------------------
        src = p3_for_aux if self.attach == "p3" else self._feats["p4"].detach()
        if self.token_stride > 1:
            src = src[:, :, :: self.token_stride, :: self.token_stride]  # non-contiguous
        _, _, th, tw = src.shape

        tokens384 = self.aux["adapter"](src)                        # (B, N, 384)
        tokens396, physics_dict = self.aux["physics"](tokens384)    # (B, N, 396)
        fraud_score, fraud_implied = self.aux["fraud_head"](tokens396)

        aux.update(
            token_hw=(th, tw),
            tokens_physics=tokens396,
            physics_dict=physics_dict,
            fraud_score=fraud_score,
            fraud_implied=fraud_implied,
        )

        # --- contrastive --------------------------------------------------
        if self.use_contrastive:
            suspicious = self._suspicious_mask(attn_maps, (th, tw))
            tokens_out, damage_scores = self.aux["contrastive"](tokens396, suspicious)
            aux.update(
                tokens_contrastive=tokens_out,
                damage_scores=damage_scores,
                suspicious_mask=suspicious,
            )
        return aux

    def _suspicious_mask(
        self, attn_maps: torch.Tensor | None, token_hw: tuple[int, int]
    ) -> torch.Tensor | None:
        """Derive the contrastive module's suspicious mask from attention maps.

        Thresholds the max-over-class attention map, which keeps the signal
        ground-truth-free at inference. Returns None (the module then treats
        every token as suspicious) when the attention head is disabled.

        Returns:
            (B, N) bool mask, detached — it is a routing decision, not a
            gradient path.
        """
        if attn_maps is None:
            return None
        th, tw = token_hw
        with torch.no_grad():
            a = attn_maps.detach()
            if a.shape[-2:] != (th, tw):
                a = torch.nn.functional.interpolate(a, size=(th, tw), mode="bilinear", align_corners=False)
            peak = a.amax(dim=1)                                  # (B, th, tw)
            mask = (peak > self.suspicious_thresh).reshape(peak.shape[0], -1)
        return mask  # (B, N) bool

    # ------------------------------------------------------------------
    # loss
    # ------------------------------------------------------------------
    def init_criterion(self):
        """Return XCarLoss when any aux module is active, else stock YOLO loss."""
        from xcar.loss import XCarLoss  # local import avoids a circular import

        if not self._aux_ready:
            return super().init_criterion()
        return XCarLoss(self)

    # ------------------------------------------------------------------
    # bookkeeping
    # ------------------------------------------------------------------
    @property
    def aux_loss_names(self) -> tuple[str, ...]:
        """Names of the aux loss terms actually active, in logging order.

        Single source of truth for term ordering: XCarLoss delegates to this
        and asserts the terms it produced match, so `loss_items` can never
        drift out of alignment with `trainer.loss_names`.
        """
        names = []
        if self.use_attention:
            names.append("attn_loss")
        if self.use_contrastive:
            names.append("cont_loss")
        if self.use_physics:
            names.extend(["phys_loss", "fraud_loss"])
        return tuple(names)

    def config_summary(self) -> dict[str, Any]:
        """Config values describing how the aux modules are attached."""
        return {
            "p3_ch": self.p3_ch,
            "p4_ch": self.p4_ch,
            "p5_ch": self.p5_ch,
            "p3_layer_idx": self.p3_layer_idx,
            "attach": self.attach,
            "token_stride": self.token_stride,
            "use_attention": self.use_attention,
            "use_physics": self.use_physics,
            "use_contrastive": self.use_contrastive,
            "suspicious_thresh": self.suspicious_thresh,
        }

    def __getstate__(self):
        """Drop unpicklable hook handles when checkpointing.

        `RemovableHandle` holds a weakref and cannot be pickled. The taps
        themselves ARE picklable and travel inside the module tree; they are
        purged and re-registered on the way back in (see __setstate__).
        """
        state = self.__dict__.copy()
        state["_hook_handles"] = []
        state["_aux"] = {}
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._hook_handles = []
        self._feats = {}  # fresh dict; _register_neck_hooks binds taps to it
        self._aux = {}
        if getattr(self, "_aux_ready", False):
            self._register_neck_hooks()  # purges the stale copied taps first
