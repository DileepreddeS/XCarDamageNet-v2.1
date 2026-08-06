from __future__ import annotations

from typing import Any

from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils import LOGGER, RANK
from ultralytics.utils.torch_utils import unwrap_model

from xcar.model import XCarDetectionModel


class XCarTrainer(DetectionTrainer):
    """DetectionTrainer that builds the aux-augmented detection model."""

    #: Set before calling train(). Keys are XCarDetectionModel kwargs:
    #: use_attention / use_physics / use_contrastive / attach / token_stride /
    #: suspicious_thresh.
    xcar_cfg: dict[str, Any] = {}

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

        L_physics updates both the physics head and the detector head, and is
        linear in the physics distribution, so both can collapse onto a single
        class and drive the term toward 0 while carrying no information:

            phys_entropy -> 0.00 AND phys_agree -> 1.00 AND phys_loss -> 0.00
            = collapsed. phys_loss near zero is then not convergence.

        Healthy: entropy well above 0 (ln 6 = 1.792 at uniform) with agreement
        rising gradually rather than pinning at 1.00 within a few epochs.
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
                "phys_loss near zero here means the two heads agreed on one class, "
                "NOT that physics consistency was learned."
            )

    def __init__(self, *args, **kwargs):
        """Stock trainer init plus the per-epoch aux diagnostic callback.

        Registered here rather than by overriding the private `_do_train`, so
        this does not depend on an ultralytics-internal signature.
        """
        super().__init__(*args, **kwargs)
        self.add_callback("on_train_epoch_end", lambda _trainer: self._log_aux_diagnostics())
