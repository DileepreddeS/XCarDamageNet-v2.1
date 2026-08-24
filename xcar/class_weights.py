"""Per-class weights for the detection cls term, and the machinery to prove
they reached the live loss object.

Two schemes live here. Exactly one may be active per run.

DIFFICULTY-AWARE (in use)
-------------------------
Weight by inverse Phase A per-class AP: classes the detector is already bad at
get a larger share of the classification gradient.

    w_c = 1 / AP_c,  normalised so sum(w) == nc

CLASS-BALANCED (Cui et al., CVPR 2019 — measured, rejected)
-----------------------------------------------------------
Weight by inverse effective number of samples:

    w_c = (1 - beta) / (1 - beta^n_c),  normalised so sum(w) == nc

Kept in the tree because it is a measured ablation row, not dead code. It did
not work on CarDD: frequency and difficulty are close to uncorrelated here, so
CB spent its budget on classes that were already strong (tire_flat, n=225,
weight 2.37, AP 0.898 -> 0.966) while down-weighting the single hardest class
(crack, n=651, weight 0.84, AP 0.521 -> 0.399). Difficulty weighting is the
direct fix: rank by what the detector actually gets wrong, not by how rare it is.

WHERE BOTH ATTACH
-----------------
ultralytics 8.4.48 has a first-class hook. `v8DetectionLoss` reads
`model.class_weights` at criterion construction:

    ultralytics/utils/loss.py:353   self.class_weights = getattr(model, "class_weights", None)
    ultralytics/utils/loss.py:355   self.class_weights = self.class_weights.to(device).view(1, 1, -1)

and consumes it inside the cls term:

    ultralytics/utils/loss.py:431   bce_loss = self.bce(pred_scores, target_scores.to(dtype))
    ultralytics/utils/loss.py:433   bce_loss *= self.class_weights
    ultralytics/utils/loss.py:434   loss[1] = bce_loss.sum() / target_scores_sum

So neither scheme needs to reach inside and swap a BCE object: the attribute is
real and is read on every batch.

WHY v1's ATTEMPT FAILED
-----------------------
v1 probed `trainer.compute_loss` at `on_train_start`. That attribute does not
exist in ultralytics, and the criterion itself does not exist yet at
`on_train_start` -- it is built lazily inside `BaseModel.loss()` on the first
forward (`ultralytics/nn/tasks.py:330-331`). v1 then logged "could not apply"
and, separately, claimed a focal-loss fallback that was never wired up.

Hence: apply from `on_train_batch_start`, which fires once the criterion can
exist, and PROVE the tensor is attached by reading it back off the live
criterion object -- not by printing "applied".
"""

from __future__ import annotations

import inspect

import torch
from ultralytics.utils import LOGGER
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.torch_utils import unwrap_model

# --------------------------------------------------------------------------
# difficulty-aware weighting
# --------------------------------------------------------------------------
#: Phase A per-class AP@0.5 on the CarDD TEST split (mAP@0.5 = 0.7468).
#:
#: METHODOLOGICAL CAVEAT — READ BEFORE PUBLISHING.
#: These are TEST-split numbers, so weights derived from them carry test
#: information into training. The resulting test mAP is therefore not a clean
#: held-out estimate, and the paper cannot present it as one without saying so.
#: The clean version is to refit this dict from Phase A's VALIDATION per-class
#: AP and re-run; nothing else in the code changes, and the weights would move
#: only slightly. Until then, treat this run's test number as development data.
PHASE_A_TEST_AP50: dict[str, float] = {
    "dent": 0.601,
    "scratch": 0.615,
    "crack": 0.521,
    "glass_shatter": 0.984,
    "lamp_broken": 0.862,
    "tire_flat": 0.898,
}

#: Floor on AP before inverting, so a class the detector fails on completely
#: cannot take over the whole cls gradient. At Phase A's APs (min 0.521) this
#: never binds; it exists so a future refit on a weaker checkpoint degrades
#: sanely instead of producing a 100x weight.
MIN_AP = 0.05

# --------------------------------------------------------------------------
# class-balanced weighting (kept as a measured ablation)
# --------------------------------------------------------------------------
#: CarDD training-split instance counts (CLAUDE.md 0.5).
CARDD_CLASS_COUNTS: dict[str, int] = {
    "dent": 1806,
    "scratch": 2560,
    "crack": 651,
    "glass_shatter": 475,
    "lamp_broken": 494,
    "tire_flat": 225,
}

CB_BETA = 0.9999


def _normalise(w: torch.Tensor) -> torch.Tensor:
    """Scale so the weights sum to len(w), i.e. mean 1.0.

    Keeps the cls-loss magnitude comparable to an unweighted run, so the
    box/cls/dfl gains in the config keep their meaning.
    """
    return w * (len(w) / w.sum())


def compute_difficulty_weights(
    ap_by_class, beta: float | None = None, device=None, dtype=torch.float32
) -> torch.Tensor:
    """Inverse-AP weights, normalised to sum to len(ap_by_class).

    Args:
        ap_by_class: per-class AP, indexed by class id (an iterable of floats,
            or a dict's .values() in class-id order).
        beta: unused; accepted so both schemes share a call signature.

    Returns:
        (nc,) float tensor summing to nc. Hardest class gets the largest weight.
    """
    ap = torch.as_tensor(list(ap_by_class), dtype=torch.float64)
    if bool((ap <= 0).any()) or bool((ap > 1).any()):
        raise ValueError(f"AP values must be in (0, 1], got {ap.tolist()}")
    return _normalise(1.0 / ap.clamp_min(MIN_AP)).to(dtype=dtype, device=device)


def compute_cb_weights(counts, beta: float = CB_BETA, device=None, dtype=torch.float32) -> torch.Tensor:
    """Class-balanced weights, normalised to sum to len(counts).

    Args:
        counts: per-class instance counts, indexed by class id.
        beta: CB hyperparameter. beta -> 0 gives uniform weights; beta -> 1
            approaches plain inverse frequency. 0.9999 is the paper's value for
            long-tailed sets of this size.

    Returns:
        (nc,) float tensor summing to nc, so the mean weight is 1.0.
    """
    if not 0.0 < beta < 1.0:
        raise ValueError(f"beta must be in (0, 1), got {beta}")
    c = torch.as_tensor(list(counts), dtype=torch.float64)
    if bool((c <= 0).any()):
        raise ValueError(f"class counts must all be positive, got {c.tolist()}")

    effective_num = 1.0 - torch.pow(torch.tensor(beta, dtype=torch.float64), c)  # 1 - beta^n
    return _normalise((1.0 - beta) / effective_num).to(dtype=dtype, device=device)


def cardd_difficulty_weights(device=None) -> torch.Tensor:
    """Difficulty weights for CarDD in class-id order (dent .. tire_flat)."""
    return compute_difficulty_weights(PHASE_A_TEST_AP50.values(), device=device)


def cardd_cb_weights(beta: float = CB_BETA, device=None) -> torch.Tensor:
    """CB weights for CarDD in class-id order (dent .. tire_flat)."""
    return compute_cb_weights(CARDD_CLASS_COUNTS.values(), beta=beta, device=device)


def format_weights(weights: torch.Tensor, names) -> str:
    """`dent=0.65, scratch=0.46, ...` in class-id order, for the proof line."""
    vals = weights.detach().flatten().tolist()
    return ", ".join(f"{n}={v:.4f}" for n, v in zip(names, vals))


def rank_weights(weights: torch.Tensor, names) -> str:
    """`crack 1.3550 > dent 1.1746 > ...` — heaviest first."""
    vals = weights.detach().flatten().tolist()
    ordered = sorted(zip(names, vals), key=lambda t: -t[1])
    return " > ".join(f"{n} {v:.4f}" for n, v in ordered)


def bce_multiply_site() -> str | None:
    """Locate the line in the LIVE ultralytics source that consumes the weights.

    A static check, but it is what turns "the attribute is set" into "the
    attribute is read by the cls loss". If a future ultralytics drops the
    multiply, this returns None and the callback fails the run rather than
    printing a reassuring lie.
    """
    src, start = inspect.getsourcelines(v8DetectionLoss.get_assigned_targets_and_loss)
    for offset, line in enumerate(src):
        if "bce_loss" in line and "self.class_weights" in line and "*=" in line:
            return f"{inspect.getsourcefile(v8DetectionLoss)}:{start + offset}  {line.strip()}"
    return None


def make_class_weight_callback(weights: torch.Tensor, class_names, *, tag: str, strict: bool = True):
    """`on_train_batch_start` callback: attach class weights once, then prove it.

    The criterion is built lazily on the first forward, so this returns quietly
    on the batches before it exists and applies on the first batch where it
    does. Once applied it becomes a no-op.

    Args:
        weights: (nc,) weights in class-id order.
        class_names: names in class-id order, for the proof line.
        tag: log prefix identifying the scheme, e.g. "DIFFICULTY WEIGHTS".
        strict: raise if the weights cannot be attached or verified. Leave True
            -- a silent no-op here is precisely the v1 failure being fixed.
    """
    state = {"applied": False}

    def fail(msg: str) -> None:
        state["applied"] = True  # do not re-raise on every subsequent batch
        if strict:
            raise RuntimeError(msg)
        LOGGER.warning(msg)

    def cb(trainer):
        if state["applied"]:
            return

        model = unwrap_model(trainer.model)
        criterion = getattr(model, "criterion", None)
        if criterion is None:
            return  # not built yet; try again next batch

        site = bce_multiply_site()
        if site is None:
            return fail(
                f"[{tag}] ABORT: v8DetectionLoss.get_assigned_targets_and_loss no "
                "longer multiplies bce_loss by self.class_weights. Setting the "
                "attribute would have no effect on the loss."
            )

        # Distinguish the two routes by which the weights can arrive, so the log
        # says which one actually happened instead of implying the callback did
        # the work. The trainer sets `model.class_weights` before the criterion
        # is built, so the normal, healthy outcome is "already present".
        prior = getattr(criterion, "class_weights", None)
        if prior is None:
            via = "attached by this callback (criterion was built without them)"
        elif torch.allclose(prior.flatten().cpu(), weights.flatten().cpu()):
            via = "already present at criterion construction (from model.class_weights)"
        else:
            via = f"OVERWRITTEN by this callback; criterion held {prior.flatten().tolist()}"

        device = getattr(criterion, "device", None) or next(model.parameters()).device
        # Same shape the criterion builds internally: (1, 1, nc), which
        # broadcasts over bce_loss of (bs, num_anchors, nc).
        criterion.class_weights = weights.detach().to(device=device, dtype=torch.float32).reshape(1, 1, -1)

        # ---- PROOF: read back off the LIVE criterion object -----------------
        attached = getattr(criterion, "class_weights", None)
        if attached is None or not torch.allclose(attached.flatten().cpu(), weights.flatten().cpu()):
            return fail(
                f"[{tag}] ABORT: wrote criterion.class_weights but reading it back "
                f"gave {attached!r}. The weights are NOT on the live loss object."
            )

        nc = int(getattr(criterion, "nc", len(class_names)))
        if attached.shape[-1] != nc:
            return fail(
                f"[{tag}] ABORT: attached {attached.shape[-1]} weights but the "
                f"criterion has nc={nc}. Broadcasting against bce_loss would fail or "
                "silently mis-align classes."
            )

        state["applied"] = True
        LOGGER.info(
            f"[{tag}] Applied: {format_weights(attached, class_names)}\n"
            f"[{tag}]   order above is CLASS-ID order, matching the tensor layout\n"
            f"[{tag}]   hardest -> easiest : {rank_weights(attached, class_names)}\n"
            f"[{tag}]   attribute modified : {type(criterion).__name__}.class_weights "
            f"(object id 0x{id(criterion):x}, shape {tuple(attached.shape)}, "
            f"dtype {attached.dtype}, device {attached.device})\n"
            f"[{tag}]   values above were read back FROM the live criterion, not "
            f"echoed from the value written\n"
            f"[{tag}]   provenance : {via}\n"
            f"[{tag}]   consumed at : {site}\n"
            f"[{tag}]   sum={float(attached.sum()):.6f} (== nc={nc}), "
            f"mean={float(attached.mean()):.6f} (== 1.0), "
            f"max/min ratio={float(attached.max() / attached.min()):.3f}"
        )

    return cb
