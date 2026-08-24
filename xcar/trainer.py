from __future__ import annotations

from typing import Any

import numpy as np
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils import LOGGER, RANK
from ultralytics.utils.torch_utils import unwrap_model

from xcar.class_weights import (
    CARDD_CLASS_COUNTS,
    CB_BETA,
    PHASE_A_TEST_AP50,
    compute_cb_weights,
    compute_difficulty_weights,
    format_weights,
    make_class_weight_callback,
    rank_weights,
)
from xcar.model import XCarDetectionModel


class XCarTrainer(DetectionTrainer):
    """DetectionTrainer that builds the aux-augmented detection model."""

    #: Set before calling train(). Keys are XCarDetectionModel kwargs:
    #: use_attention / use_physics / use_contrastive / attach / token_stride /
    #: suspicious_thresh.
    xcar_cfg: dict[str, Any] = {}

    #: Difficulty-aware weighting: w_c = 1/AP_c from Phase A. In use.
    difficulty_weights: bool = True
    #: Class-Balanced loss (Cui et al. 2019). Measured and rejected on CarDD;
    #: kept switchable so the ablation row can be reproduced.
    cb_loss: bool = False
    cb_beta: float = CB_BETA

    # ------------------------------------------------------------------
    # cls-term class weighting
    # ------------------------------------------------------------------
    def set_class_weights(self) -> None:
        """Attach per-class cls weights so the criterion picks them up.

        Overrides ultralytics' `cls_pw` inverse-frequency weighting (which is
        disabled by default, cls_pw=0.0). Runs inside `_setup_train`, i.e. after
        the dataloader exists and before the first forward builds the criterion
        — so `v8DetectionLoss.__init__` reads the weights off the model natively
        (ultralytics/utils/loss.py:353).

        The `on_train_batch_start` callback registered here then verifies that
        against the LIVE criterion object and prints the proof. Both paths
        exist on purpose: the model attribute is how it should work, the
        callback is how we know it did.

        Exactly one scheme may be active: both write the same attribute, so
        enabling both would mean whichever ran last silently won.
        """
        if self.difficulty_weights and self.cb_loss:
            raise ValueError(
                "difficulty_weights and cb_loss are both enabled. Both write "
                "model.class_weights, so one would silently overwrite the other "
                "and the run would be unattributable. Pick one."
            )

        names = [self.data["names"][i] for i in range(self.data["nc"])]

        if self.difficulty_weights:
            tag = "DIFFICULTY WEIGHTS"
            ap = self._difficulty_source_ap(names)
            weights = compute_difficulty_weights(ap)
            source = f"1/AP from Phase A: {dict(zip(names, ap))}"
        elif self.cb_loss:
            tag = "CB LOSS"
            counts = self._cb_class_counts(names)
            weights = compute_cb_weights(counts, beta=self.cb_beta)
            source = f"beta={self.cb_beta}  counts={dict(zip(names, counts))}"
        else:
            LOGGER.info(
                "[CLS WEIGHTS] none enabled (difficulty_weights=False, cb_loss=False); "
                "falling back to cls_pw behaviour"
            )
            return super().set_class_weights()

        model = unwrap_model(self.model)
        model.class_weights = weights.to(self.device)
        LOGGER.info(
            f"[{tag}] source: {source}\n"
            f"[{tag}] set on model.class_weights: {format_weights(weights, names)}\n"
            f"[{tag}] hardest -> easiest: {rank_weights(weights, names)}"
        )
        self.add_callback("on_train_batch_start", make_class_weight_callback(weights, names, tag=tag))

    def _difficulty_source_ap(self, names: list[str]) -> list[float]:
        """Phase A per-class AP in class-id order, for the inverse-AP weights.

        Refuses to guess for a class it has no measurement for: a missing entry
        would otherwise silently become an arbitrary weight, which is exactly
        the kind of unattributable change CLAUDE.md 1.3 forbids.
        """
        missing = [n for n in names if n not in PHASE_A_TEST_AP50]
        if missing:
            raise RuntimeError(
                f"[DIFFICULTY WEIGHTS] no Phase A AP recorded for {missing}. "
                "Add the measured values to PHASE_A_TEST_AP50 in "
                "xcar/class_weights.py, or disable difficulty_weights. "
                "Refusing to invent a weight."
            )
        return [PHASE_A_TEST_AP50[n] for n in names]

    def _cb_class_counts(self, names: list[str]) -> list[int]:
        """Per-class instance counts for the CB formula, in class-id order.

        Uses the counts recorded in CLAUDE.md 0.5 when the class names are
        CarDD's, so the published weights are fixed and reproducible, but always
        cross-checks them against what the dataloader actually holds and warns
        loudly on any mismatch. For any other dataset the observed counts are
        the only defensible source.
        """
        observed: list[int] | None = None
        try:
            labels = self.train_loader.dataset.labels
            classes = np.concatenate([lb["cls"].flatten() for lb in labels], 0)
            observed = np.bincount(classes.astype(int), minlength=len(names)).tolist()
        except Exception as e:  # a counting failure must not kill the run
            LOGGER.warning(f"[CB LOSS] could not count labels from the dataloader: {e}")

        if set(names) != set(CARDD_CLASS_COUNTS):
            if observed is None:
                raise RuntimeError(
                    f"[CB LOSS] dataset classes {names} are not CarDD's and the "
                    "dataloader counts could not be read, so there is no source "
                    "for the CB weights. Refusing to guess."
                )
            LOGGER.info("[CB LOSS] non-CarDD classes; using counts observed in the train split")
            return observed

        counts = [CARDD_CLASS_COUNTS[n] for n in names]
        if observed is not None and observed != counts:
            LOGGER.warning(
                "[CB LOSS] COUNT MISMATCH — the train split does not match CLAUDE.md 0.5.\n"
                f"[CB LOSS]   spec (used)  : {dict(zip(names, counts))}\n"
                f"[CB LOSS]   observed     : {dict(zip(names, observed))}\n"
                "[CB LOSS] Using the spec counts so the weights stay reproducible. "
                "If the observed counts are correct, the dataset changed and this "
                "run's CB weights are wrong — fix CARDD_CLASS_COUNTS before trusting it."
            )
        return counts

    def get_model(self, cfg: str | None = None, weights: str | None = None, verbose: bool = True):
        """Return an XCarDetectionModel with the phase's aux modules enabled."""
        model = XCarDetectionModel(
            cfg or "yolo11m.yaml",
            nc=self.data["nc"],
            ch=self.data["channels"],
            verbose=verbose and RANK == -1,
            **self.xcar_cfg,
        )
        if weights:
            model.load(weights)
        LOGGER.info(f"[XCAR] model config: {model.config_summary()}")
        return model

    def get_validator(self):
        """Stock validator; extend loss_names with the active aux terms."""
        validator = super().get_validator()  # sets loss_names = box/cls/dfl
        model = self.model
        aux_names = getattr(model, "aux_loss_names", ())
        if aux_names:
            self.loss_names = tuple(self.loss_names) + tuple(aux_names)
            LOGGER.info(f"[XCAR] loss_names = {self.loss_names}")
        return validator

    def _log_aux_diagnostics(self) -> None:
        """Log the physics collapse diagnostics once per epoch.

        L_physics reads the detector's class logits as a DETACHED target, so it
        no longer moves the detector; the physics head alone chases it. That
        removes mutual collapse, but the physics head can still saturate on one
        class by itself and drive the term toward 0 while carrying no
        information:

            phys_entropy -> 0.00 AND phys_agree -> 1.00 AND phys_loss -> 0.00
            = collapsed. phys_loss near zero is then not convergence.

        Healthy: entropy well above 0 (ln 6 = 1.792 at uniform) with agreement
        rising gradually rather than pinning at 1.00 within a few epochs.

        Unlike v2.1's earlier wiring, a collapse here costs nothing in mAP —
        the whole aux path is gradient-isolated from the backbone — but it does
        mean the physics head's outputs are not usable evidence in a report.
        """
        criterion = getattr(unwrap_model(self.model), "criterion", None)
        stats = getattr(criterion, "last_stats", None)
        if not stats:
            return
        parts = [f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in stats.items()]
        LOGGER.info(f"[XCAR] epoch {self.epoch + 1} aux diagnostics: " + "  ".join(parts))
        if (
            stats.get("phys_entropy", 1.0) < 0.05
            and stats.get("phys_agree", 0.0) > 0.99
        ):
            LOGGER.warning(
                "[XCAR] PHYSICS COLLAPSE SIGNATURE: entropy ~0 with agreement ~1.0. "
                "phys_loss near zero here means the physics head saturated on the "
                "detector's argmax, NOT that physics consistency was learned. "
                "Detection mAP is unaffected (the aux path is gradient-isolated), "
                "but do not report these physics outputs as evidence."
            )

    def __init__(self, *args, **kwargs):
        """Stock trainer init plus the per-epoch aux diagnostic callback.

        Registered here rather than by overriding the private `_do_train`, so
        this does not depend on an ultralytics-internal signature.
        """
        super().__init__(*args, **kwargs)
        self.add_callback("on_train_epoch_end", lambda _trainer: self._log_aux_diagnostics())
