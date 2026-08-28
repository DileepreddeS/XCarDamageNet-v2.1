"""Final run: Phase A detection weights + in-domain pretrained physics tokens.

Two warm starts, from two different artefacts:

    models/phase_a_best.pt                 -> backbone, neck, Detect head
    models/physics_encoder_yolo_1024.pt    -> FeatureTokenAdapter + PhysicsTokenEncoder

Aux modules (5): AttentionMapHead, FeatureTokenAdapter, PhysicsTokenEncoder,
ImpliedClassHead, ContrastiveDamageModule. No fraud head — removed in 0570c6b.

Six logged loss terms:

    box, cls, dfl            stock YOLO (TAL-assigned, untouched)
    attn                     0.10
    cont                     0.05
    phys                     0.02

Class weighting is OFF — neither CB nor difficulty weights. `cls_pw` stays at
ultralytics' 0.0 default, so the cls term is the standard unweighted BCE.

The aux path is gradient-isolated: neck features and the L_physics target are
both detached, so aux losses train aux modules only and never the backbone.

Run:  python scripts/train_final.py
      python scripts/train_final.py --verify-only     # load + report, no training
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import ultralytics  # noqa: E402
from ultralytics import YOLO  # noqa: E402
from ultralytics.utils.torch_utils import unwrap_model  # noqa: E402

from xcar.loss import W_ATTN, W_CONTRAST, W_PHYSICS  # noqa: E402
from xcar.model import PHYSICS_DIM, TOKEN_DIM  # noqa: E402
from xcar.trainer import XCarTrainer  # noqa: E402

EXPECTED_ULTRALYTICS = "8.4.48"

DEFAULT_DETECTION_WEIGHTS = REPO / "models" / "phase_a_best.pt"
DEFAULT_PHYSICS_WEIGHTS = REPO / "models" / "physics_encoder_yolo_1024.pt"

EXPECTED_LOSS_NAMES = (
    "box_loss", "cls_loss", "dfl_loss",
    "attn_loss", "cont_loss", "phys_loss",
)

#: Checkpoint key -> the `model.aux` module it initialises.
PHYSICS_KEY_MAP = {"adapter": "adapter", "physics_encoder": "physics"}

TRAIN_CFG = dict(
    data="/scratch/ds3424/cardd/yolo/cardd.yaml",
    project="/scratch/ds3424/cardd",
    name="cardd_v21_final",
    imgsz=1024,
    batch=8,
    epochs=200,
    patience=50,
    cos_lr=True,
    amp=False,                 # fp16 produced NaN losses on this setup
    save_period=10,            # HPC checkpoint safety
    workers=8,
    lr0=0.01,
    lrf=0.01,
    warmup_epochs=3.0,
    momentum=0.937,
    weight_decay=0.0005,
    box=7.5,
    cls=0.5,
    dfl=1.5,
    cls_pw=0.0,                # no class weighting: not CB, not difficulty
    mosaic=1.0,
    mixup=0.10,
    copy_paste=0.30,
    close_mosaic=20,
    fliplr=0.5,
    flipud=0.0,
    degrees=0.0,
    translate=0.1,
    scale=0.5,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
)

AUX_CFG = dict(
    use_attention=True,
    use_physics=True,
    use_contrastive=True,
    attach="p3",
    token_stride=1,
    suspicious_thresh=0.5,
)

UNIFORM_ENTROPY = math.log(6)  # 1.7918 — phys_entropy at a uniform distribution


# --------------------------------------------------------------------------
# physics weight loading — verified, not assumed
# --------------------------------------------------------------------------
def load_physics_weights(model, path: str | Path, *, strict: bool = True) -> dict:
    """Load the pretrained adapter + physics encoder into a built model.

    Reports, per module: how many tensors transferred, and any missing,
    unexpected or shape-mismatched keys. The transferred count is verified with
    `torch.equal` AFTER the load — reading the values back off the live module
    rather than trusting `load_state_dict` to have done what it said.

    Returns a report dict; raises if `strict` and anything did not line up.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"pretrained physics weights not found: {path}\n"
            "Run scripts/pretrain_physics_1024.py first, or pass --physics-weights."
        )

    state = torch.load(path, map_location="cpu", weights_only=False)
    meta = state.get("meta", {})

    print("=" * 78)
    print("PHYSICS WEIGHT TRANSFER")
    print("=" * 78)
    print(f"checkpoint : {path}")
    if meta:
        print(f"  pretrained on : {meta.get('source')} | {meta.get('n_images')} images "
              f"| imgsz {meta.get('imgsz')} | epoch {state.get('epoch')}")
        print(f"  dims          : p3_ch {meta.get('p3_ch')} | token_dim "
              f"{meta.get('token_dim')} | physics_dim {meta.get('physics_dim')}")
        print(f"  MAE           : {meta.get('mae_tokens')} of {meta.get('n_tokens')} "
              f"tokens, mask {meta.get('mask_ratio')}")

    # A checkpoint pretrained at a different width would load "successfully" and
    # silently mean something else.
    problems = []
    if meta.get("p3_ch") not in (None, model.p3_ch):
        problems.append(f"checkpoint p3_ch {meta['p3_ch']} != model p3_ch {model.p3_ch}")
    if meta.get("token_dim") not in (None, TOKEN_DIM):
        problems.append(f"checkpoint token_dim {meta['token_dim']} != {TOKEN_DIM}")
    if meta.get("physics_dim") not in (None, PHYSICS_DIM):
        problems.append(f"checkpoint physics_dim {meta['physics_dim']} != {PHYSICS_DIM}")
    if problems and strict:
        raise RuntimeError("[PHYSICS] checkpoint/model mismatch: " + "; ".join(problems))
    for p in problems:
        print(f"  WARNING: {p}")

    report: dict[str, dict] = {}
    total_transferred = total_expected = 0

    for ckpt_key, aux_key in PHYSICS_KEY_MAP.items():
        print(f"\n  {aux_key} <- checkpoint['{ckpt_key}']")
        if ckpt_key not in state:
            msg = f"checkpoint has no '{ckpt_key}' entry (keys: {sorted(state)})"
            if strict:
                raise RuntimeError(f"[PHYSICS] {msg}")
            print(f"    SKIPPED: {msg}")
            continue
        if aux_key not in model.aux:
            msg = f"model.aux has no '{aux_key}' module (has: {sorted(model.aux)})"
            if strict:
                raise RuntimeError(f"[PHYSICS] {msg}")
            print(f"    SKIPPED: {msg}")
            continue

        module = model.aux[aux_key]
        ckpt_sd = state[ckpt_key]
        model_sd = module.state_dict()

        missing = sorted(set(model_sd) - set(ckpt_sd))
        unexpected = sorted(set(ckpt_sd) - set(model_sd))
        mismatched = [
            (k, tuple(ckpt_sd[k].shape), tuple(model_sd[k].shape))
            for k in sorted(set(ckpt_sd) & set(model_sd))
            if tuple(ckpt_sd[k].shape) != tuple(model_sd[k].shape)
        ]

        if (missing or unexpected or mismatched) and strict:
            raise RuntimeError(
                f"[PHYSICS] {aux_key} does not match the checkpoint.\n"
                f"  missing in checkpoint : {missing}\n"
                f"  unexpected in checkpoint: {unexpected}\n"
                f"  shape mismatches      : {mismatched}\n"
                "The pretraining script builds these modules from the same classes, "
                "so a mismatch means the checkpoint is stale or from another config."
            )

        module.load_state_dict(ckpt_sd, strict=strict)

        # Verify by reading back off the live module.
        after = module.state_dict()
        transferred = [k for k in ckpt_sd if k in after
                       and torch.equal(after[k].detach().cpu(), ckpt_sd[k].detach().cpu())]
        n_params = sum(int(ckpt_sd[k].numel()) for k in transferred)

        print(f"    tensors transferred : {len(transferred)}/{len(model_sd)} "
              f"({n_params:,} parameters)")
        print(f"    missing             : {len(missing)}{' ' + str(missing) if missing else ''}")
        print(f"    unexpected          : {len(unexpected)}"
              f"{' ' + str(unexpected) if unexpected else ''}")
        print(f"    shape mismatches    : {len(mismatched)}"
              f"{' ' + str(mismatched) if mismatched else ''}")
        print(f"    verified by torch.equal AFTER load, not by load_state_dict's return")

        report[aux_key] = dict(
            transferred=len(transferred), expected=len(model_sd), parameters=n_params,
            missing=missing, unexpected=unexpected, mismatched=mismatched,
        )
        total_transferred += len(transferred)
        total_expected += len(model_sd)

    ok = total_transferred == total_expected and not any(
        r["missing"] or r["unexpected"] or r["mismatched"] for r in report.values()
    )
    verdict = "0 mismatches" if ok else "WITH MISMATCHES"
    print(f"\n  TOTAL: {total_transferred}/{total_expected} tensors transferred, {verdict}")

    # Stated plainly because it bears directly on the phys diagnostics below.
    print("\n  NOT covered by this checkpoint: ImpliedClassHead (the head that emits")
    print("  the physics-implied class logits) starts from random init. Pretraining")
    print("  covered the adapter and the encoder only, which is all it ever saved.")
    print("=" * 78)

    if not ok and strict:
        raise RuntimeError("[PHYSICS] transfer incomplete — see the report above.")
    return report


# --------------------------------------------------------------------------
# trainer
# --------------------------------------------------------------------------
class XCarFinalTrainer(XCarTrainer):
    """XCarTrainer that also warm-starts the adapter + physics encoder.

    The load happens inside `get_model`, i.e. during `_setup_train` before the
    EMA snapshot and the optimizer are built. Loading later would leave the EMA
    tracking the random init for the first epochs.
    """

    physics_weights: str | None = None

    #: This run uses the standard unweighted cls loss. Declared on the class so
    #: the default is correct by construction — inheriting XCarTrainer's
    #: `difficulty_weights = True` would silently re-enable inverse-AP weighting
    #: for any caller that does not go through main().
    difficulty_weights: bool = False
    cb_loss: bool = False

    def get_model(self, cfg: str | None = None, weights: str | None = None, verbose: bool = True):
        model = super().get_model(cfg, weights, verbose)
        if self.physics_weights:
            load_physics_weights(model, self.physics_weights)
        return model


# --------------------------------------------------------------------------
# callbacks
# --------------------------------------------------------------------------
def make_loss_name_check():
    """Fail before epoch 1 if the 6 loss columns are not wired up."""

    def cb(trainer):
        names = tuple(trainer.loss_names)
        if names != EXPECTED_LOSS_NAMES:
            raise RuntimeError(
                f"loss_names is {names}, expected {EXPECTED_LOSS_NAMES}. "
                "Every per-epoch loss number would be mislabelled."
            )
        print(f"[FINAL] logging {len(names)} loss terms: {', '.join(names)}")

    return cb


def make_physics_diagnostics():
    """Print phys_entropy / phys_agree every epoch, and flag a dead signal.

    phys_entropy is the entropy of softmax(implied_logits), so ln(6)=1.7918 at
    uniform and 0.0 when the head has saturated onto a single class.
    phys_agree is the fraction of images where argmax(physics)==argmax(detector).
    """

    def cb(trainer):
        criterion = getattr(unwrap_model(trainer.model), "criterion", None)
        stats = getattr(criterion, "last_stats", None) or {}
        if "phys_entropy" not in stats:
            print(f"[FINAL] epoch {trainer.epoch + 1}: no phys diagnostics recorded "
                  "— L_physics did not run this epoch")
            return

        entropy = float(stats["phys_entropy"])
        agree = float(stats.get("phys_agree", float("nan")))
        phys = float(stats.get("phys_loss", float("nan")))
        print(f"[FINAL] epoch {trainer.epoch + 1}: phys_entropy={entropy:.4f} "
              f"(uniform={UNIFORM_ENTROPY:.4f})  phys_agree={agree:.4f}  "
              f"phys_loss={phys:.4f}")

        if entropy < 5e-4:
            print("[FINAL]   phys_entropy is ~0.000: ImpliedClassHead has saturated "
                  "onto one class.")
            print("[FINAL]   This is the documented one-sided collapse of L_physics "
                  "(the term is linear")
            print("[FINAL]   in the physics distribution, so its optimum IS a one-hot "
                  "at the detector's")
            print("[FINAL]   argmax). It is NOT evidence that the pretrained adapter/"
                  "encoder failed to")
            print("[FINAL]   load — that is proven separately by the PHYSICS WEIGHT "
                  "TRANSFER report")
            print("[FINAL]   above. Detection mAP is unaffected: the aux path is "
                  "gradient-isolated.")

    return cb


# --------------------------------------------------------------------------
# evaluation
# --------------------------------------------------------------------------
def evaluate(weights: str | Path, data: str, device: str, imgsz: int, batch: int,
             split: str = "test") -> dict:
    """Validate on `split` with plots off; return a plain-dict summary."""
    metrics = YOLO(str(weights)).val(
        data=data, split=split, imgsz=imgsz, batch=batch, device=device, plots=False,
    )
    per_class = {}
    for i, c in enumerate(metrics.box.ap_class_index):
        name = metrics.names[int(c)] if isinstance(metrics.names, dict) else str(int(c))
        per_class[name] = float(metrics.box.ap50[i])
    return {
        "weights": str(weights), "split": split,
        "mAP50": float(metrics.box.map50), "mAP50_95": float(metrics.box.map),
        "per_class_AP50": per_class,
    }


def print_results(res: dict) -> None:
    print("\n" + "=" * 78)
    print(f"FINAL RESULTS — {res['split']} split")
    print("=" * 78)
    print(f"{'class':<18} {'AP@0.5':>10}")
    print("-" * 78)
    for name, ap in sorted(res["per_class_AP50"].items(), key=lambda kv: -kv[1]):
        print(f"{name:<18} {ap:>10.4f}")
    print("-" * 78)
    print(f"{'mAP@0.5':<18} {res['mAP50']:>10.4f}")
    print(f"{'mAP@0.5:0.95':<18} {res['mAP50_95']:>10.4f}")
    print("=" * 78)
    print(f"v1 published baseline: 0.700 mAP@0.5 (0.739 with TTA), batch=16.")
    print(f"This run: batch=8, so the batch-size control is Phase A, not v1.")
    print("=" * 78)


# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(DEFAULT_DETECTION_WEIGHTS),
                    help="Phase A detection checkpoint")
    ap.add_argument("--physics-weights", default=str(DEFAULT_PHYSICS_WEIGHTS),
                    help="pretrained adapter + physics encoder")
    ap.add_argument("--data", default=None, help="override the dataset YAML")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--device", default="0")
    ap.add_argument("--attach", choices=["p3", "p4"], default=None,
                    help="OOM fallback: move physics/contrastive tokens to P4")
    ap.add_argument("--token-stride", type=int, default=None,
                    help="OOM fallback: subsample physics/contrastive tokens")
    ap.add_argument("--verify-only", action="store_true",
                    help="build the model, load both checkpoints, report, and exit")
    args = ap.parse_args()

    cfg = dict(TRAIN_CFG)
    aux_cfg = dict(AUX_CFG)
    if args.data:
        cfg["data"] = args.data
    if args.epochs:
        cfg["epochs"] = args.epochs
    if args.attach:
        aux_cfg["attach"] = args.attach
    if args.token_stride:
        aux_cfg["token_stride"] = args.token_stride
    cfg["device"] = args.device

    print("=" * 78)
    print("FINAL RUN — Phase A detection weights + pretrained physics tokens")
    print("=" * 78)
    print(f"ultralytics : {ultralytics.__version__}  (pin: {EXPECTED_ULTRALYTICS})")
    print(f"torch       : {torch.__version__}")
    print(f"cuda        : {torch.cuda.is_available()}  "
          f"({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'})")

    if ultralytics.__version__ != EXPECTED_ULTRALYTICS:
        print(f"\nERROR: ultralytics {ultralytics.__version__} != pinned "
              f"{EXPECTED_ULTRALYTICS}. Version drift makes results unattributable.")
        return 2

    assert cfg["amp"] is False, "amp must be False"
    assert cfg["batch"] == 8, "batch must be 8"
    assert cfg["save_period"] == 10, "save_period must be 10"
    assert cfg["cls_pw"] == 0.0, "cls_pw must be 0.0 — this run uses no class weighting"

    det_weights = Path(args.weights)
    if not det_weights.exists():
        print(f"\nERROR: Phase A checkpoint not found:\n  {det_weights}\n"
              "Pass --weights, or archive Phase A's best.pt there first.")
        return 2
    phys_weights = Path(args.physics_weights)
    if not phys_weights.exists():
        print(f"\nERROR: pretrained physics weights not found:\n  {phys_weights}\n"
              "Run scripts/pretrain_physics_1024.py first, or pass --physics-weights.")
        return 2

    XCarFinalTrainer.xcar_cfg = aux_cfg
    XCarFinalTrainer.physics_weights = str(phys_weights)
    # Standard detection loss: neither weighting scheme.
    XCarFinalTrainer.difficulty_weights = False
    XCarFinalTrainer.cb_loss = False

    print(f"\ndetection weights : {det_weights}")
    print(f"physics weights   : {phys_weights}")
    print(f"aux modules       : {aux_cfg}")
    print(f"aux (5)           : attention, adapter, physics, implied-class, contrastive "
          "(no fraud head)")
    print(f"loss terms (6)    : {', '.join(EXPECTED_LOSS_NAMES)}")
    print(f"loss weights      : attn={W_ATTN}  contrast={W_CONTRAST}  physics={W_PHYSICS}")
    print("cls weighting     : OFF (no CB, no difficulty weights, cls_pw=0.0)")
    print("aux gradients     : DETACHED at the neck and at the L_physics target")
    print("\ntrain kwargs:")
    for k in sorted(cfg):
        print(f"  {k:<16} {cfg[k]}")
    print()

    if args.verify_only:
        return verify_only(det_weights, phys_weights, aux_cfg, cfg)

    model = YOLO(str(det_weights))
    model.add_callback("on_train_start", make_loss_name_check())
    model.add_callback("on_train_epoch_end", make_physics_diagnostics())

    results = model.train(trainer=XCarFinalTrainer, **cfg)

    save_dir = (Path(results.save_dir) if hasattr(results, "save_dir")
                else Path(cfg["project"]) / cfg["name"])
    print(f"\nTraining complete. Artifacts: {save_dir}")

    best = save_dir / "weights" / "best.pt"
    if not best.exists():
        print(f"WARNING: {best} not found — cannot evaluate.")
        return 0

    print(f"\nEvaluating {best} on the test split (plots=False)...")
    res = evaluate(best, cfg["data"], args.device, cfg["imgsz"], cfg["batch"])
    print_results(res)

    out = save_dir / "final_test_metrics.json"
    out.write_text(json.dumps(
        {"test": res, "aux_cfg": aux_cfg,
         "loss_terms": list(EXPECTED_LOSS_NAMES),
         "loss_weights": {"attn": W_ATTN, "contrast": W_CONTRAST, "physics": W_PHYSICS},
         "cls_class_weights": {"scheme": "none"},
         "detection_init": str(det_weights), "physics_init": str(phys_weights)},
        indent=2) + "\n")
    print(f"\nwrote {out}")
    print("Now record this run in results/runs.csv.")
    return 0


def verify_only(det_weights: Path, phys_weights: Path, aux_cfg: dict, cfg: dict) -> int:
    """Build the model and load both checkpoints without training."""
    from xcar.model import XCarDetectionModel

    print("--verify-only: building the model and loading both checkpoints\n")
    model = XCarDetectionModel("yolo11m.yaml", ch=3, nc=6, verbose=False, **aux_cfg)
    model.load(str(det_weights))
    report = load_physics_weights(model, phys_weights)

    clean = all(not (r["missing"] or r["unexpected"] or r["mismatched"])
                and r["transferred"] == r["expected"] for r in report.values())
    print(f"\naux modules built : {sorted(model.aux)}")
    print(f"loss terms        : {('box_loss', 'cls_loss', 'dfl_loss') + model.aux_loss_names}")
    print(f"\nVERIFY {'PASSED' if clean else 'FAILED'}")
    return 0 if clean else 1


if __name__ == "__main__":
    raise SystemExit(main())
