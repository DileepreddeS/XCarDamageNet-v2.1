"""Train with every auxiliary module enabled at once.

Enables, on top of the stock YOLO11m detection path:

    AttentionMapHead                                 -> L_attn      (0.10)
    FeatureTokenAdapter + PhysicsTokenEncoder
        + FraudHead                                  -> L_physics   (0.02)
                                                     -> L_fraud     (0.01)
    ContrastiveDamageModule                          -> L_contrast  (0.05)

Warm-starts from a Phase A checkpoint: the backbone, neck and detect head
transfer by name; the aux modules start from random init.

Run:  python scripts/train_full.py [--config configs/full.yaml]
                                   [--weights /path/to/phase_a_best.pt]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import torch
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import ultralytics  # noqa: E402
from ultralytics import YOLO  # noqa: E402

from xcar.loss import W_ATTN, W_CONTRAST, W_FRAUD, W_PHYSICS  # noqa: E402
from xcar.trainer import XCarTrainer  # noqa: E402

EXPECTED_ULTRALYTICS = "8.4.48"
DEFAULT_WEIGHTS = "/home/ds3424/XCarDamageNet-v2.1/models/phase_a_best.pt"

# Config keys that are ours, not ultralytics' — strip before train().
NON_TRAIN_KEYS = {
    "phase", "description", "model",
    "use_attention", "use_physics", "use_contrastive",
    "attach", "token_stride", "suspicious_thresh",
    "backup_dir", "backup_every",
}

EXPECTED_LOSS_NAMES = (
    "box_loss", "cls_loss", "dfl_loss",
    "attn_loss", "cont_loss", "phys_loss", "fraud_loss",
)


# --------------------------------------------------------------------------
# callbacks
# --------------------------------------------------------------------------
def make_loss_name_check():
    """Fail fast if the 7 loss columns are not wired up before epoch 1."""

    def cb(trainer):
        names = tuple(trainer.loss_names)
        if names != EXPECTED_LOSS_NAMES:
            raise RuntimeError(
                f"loss_names is {names}, expected {EXPECTED_LOSS_NAMES}. "
                "The aux terms are not wired into the logged columns; every "
                "per-epoch loss number would be mislabelled."
            )
        print(f"[FULL] logging {len(names)} loss terms: {', '.join(names)}")

    return cb


def make_per_class_reporter():
    """Print per-class validation AP50 after every epoch's validation."""

    def cb(trainer):
        metrics = getattr(getattr(trainer, "validator", None), "metrics", None)
        box = getattr(metrics, "box", None)
        if box is None or not len(getattr(box, "ap50", [])):
            return
        names = getattr(metrics, "names", {}) or {}
        try:
            idx = list(metrics.ap_class_index)
        except Exception:
            return

        epoch = trainer.epoch + 1
        parts = []
        for i, c in enumerate(idx):
            label = names.get(int(c), str(int(c))) if isinstance(names, dict) else str(int(c))
            parts.append(f"{label}={float(box.ap50[i]):.3f}")
        print(
            f"[FULL] epoch {epoch} val AP50 | mAP50={float(box.map50):.4f} "
            f"mAP50-95={float(box.map):.4f} | " + "  ".join(parts)
        )

    return cb


def make_backup(backup_dir: Path, every: int):
    """Copy best.pt off volatile scratch every `every` epochs.

    Writes to a temp file and renames, so a job killed mid-copy leaves the
    previous good backup intact rather than a truncated file.
    """
    backup_dir.mkdir(parents=True, exist_ok=True)
    dest = backup_dir / "xcar_full_best.pt"

    def cb(trainer):
        epoch = trainer.epoch + 1
        if epoch % every and epoch != trainer.epochs:
            return
        src = Path(trainer.best)
        if not src.exists():
            return
        tmp = dest.with_suffix(".pt.tmp")
        try:
            shutil.copy2(src, tmp)
            os.replace(tmp, dest)
            print(f"[FULL] epoch {epoch}: backed up best.pt -> {dest}")
        except OSError as e:
            # A failed backup must not kill a multi-hour training run.
            print(f"[FULL] WARNING: backup failed at epoch {epoch}: {e}")
            tmp.unlink(missing_ok=True)

    return cb


# --------------------------------------------------------------------------
# evaluation / comparison
# --------------------------------------------------------------------------
def evaluate(weights: str | Path, cfg: dict, data: str, device: str, split: str = "test"):
    """Run YOLO validation on `split` and return a plain-dict summary."""
    metrics = YOLO(str(weights)).val(
        data=data, split=split, imgsz=cfg["imgsz"], batch=cfg["batch"], device=device,
    )
    per_class = {}
    for i, c in enumerate(metrics.box.ap_class_index):
        name = metrics.names[int(c)] if isinstance(metrics.names, dict) else str(int(c))
        per_class[name] = float(metrics.box.ap50[i])
    return {
        "weights": str(weights),
        "split": split,
        "mAP50": float(metrics.box.map50),
        "mAP50_95": float(metrics.box.map),
        "per_class_AP50": per_class,
    }


def print_comparison(baseline: dict | None, full: dict) -> None:
    """Side-by-side Phase A vs full-module results on the same split."""
    print("\n" + "=" * 78)
    print("COMPARISON — Phase A baseline vs all-modules run (same split, same imgsz)")
    print("=" * 78)

    if baseline is None:
        print("Phase A baseline was not evaluated (weights unavailable), so no")
        print("side-by-side is possible. Full-module results only:\n")
        print(f"  mAP@0.5      {full['mAP50']:.4f}")
        print(f"  mAP@0.5:0.95 {full['mAP50_95']:.4f}")
        for name, ap in sorted(full["per_class_AP50"].items()):
            print(f"  {name:<16} {ap:.4f}")
        print("=" * 78)
        return

    print(f"{'metric':<18} {'phase A':>10} {'full':>10} {'delta':>10}")
    print("-" * 78)

    def row(label, a, b):
        print(f"{label:<18} {a:>10.4f} {b:>10.4f} {b - a:>+10.4f}")

    row("mAP@0.5", baseline["mAP50"], full["mAP50"])
    row("mAP@0.5:0.95", baseline["mAP50_95"], full["mAP50_95"])
    print("-" * 78)
    for name in sorted(set(baseline["per_class_AP50"]) | set(full["per_class_AP50"])):
        a = baseline["per_class_AP50"].get(name)
        b = full["per_class_AP50"].get(name)
        if a is None or b is None:
            print(f"{name:<18} {'n/a' if a is None else f'{a:.4f}':>10} "
                  f"{'n/a' if b is None else f'{b:.4f}':>10} {'':>10}")
        else:
            row(name, a, b)
    print("=" * 78)
    delta = full["mAP50"] - baseline["mAP50"]
    verdict = "improves on" if delta > 0 else "does not beat"
    print(f"All-modules {verdict} Phase A by {delta:+.4f} mAP@0.5 on the "
          f"{full['split']} split.")
    print("This run changed several things at once, so it attributes the total")
    print("effect only — not which module produced it. That needs the ablations.")
    print("=" * 78)


# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO / "configs" / "full.yaml"))
    ap.add_argument("--weights", default=None,
                    help=f"warm-start checkpoint (default: config `model`, else {DEFAULT_WEIGHTS})")
    ap.add_argument("--data", default=None, help="override the dataset YAML path")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--device", default="0")
    ap.add_argument("--attach", choices=["p3", "p4"], default=None,
                    help="OOM fallback: move physics/contrastive tokens to P4")
    ap.add_argument("--token-stride", type=int, default=None,
                    help="OOM fallback: subsample physics/contrastive tokens")
    ap.add_argument("--skip-baseline-eval", action="store_true",
                    help="skip re-evaluating Phase A weights in the final comparison")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    weights = args.weights or cfg.get("model") or DEFAULT_WEIGHTS
    if args.attach:
        cfg["attach"] = args.attach
    if args.token_stride:
        cfg["token_stride"] = args.token_stride

    print("=" * 78)
    print(f"FULL RUN — {cfg['description']}")
    print("=" * 78)
    print(f"ultralytics : {ultralytics.__version__}  (pin: {EXPECTED_ULTRALYTICS})")
    print(f"torch       : {torch.__version__}")
    print(f"cuda        : {torch.cuda.is_available()}  "
          f"({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'})")

    if ultralytics.__version__ != EXPECTED_ULTRALYTICS:
        print(f"\nERROR: ultralytics {ultralytics.__version__} != pinned "
              f"{EXPECTED_ULTRALYTICS}. Version drift makes results "
              "unattributable. Aborting.")
        return 2

    # Fail loudly rather than train with the wrong settings.
    assert cfg["amp"] is False, "amp must be False"
    assert cfg["batch"] == 8, "batch must be 8"
    assert cfg["save_period"] == 10, "save_period must be 10"
    assert cfg["use_attention"] and cfg["use_physics"] and cfg["use_contrastive"], \
        "this script enables ALL aux modules; use configs/phase_a.yaml for the control"

    # A missing warm-start path would otherwise be treated as a model name and
    # trigger a download of something entirely different.
    if not Path(weights).exists():
        print(f"\nERROR: warm-start checkpoint not found:\n  {weights}\n"
              "Pass --weights with the Phase A best.pt, or point the config's "
              "`model` key at it.")
        return 2

    aux_cfg = {
        "use_attention": True,
        "use_physics": True,
        "use_contrastive": True,
        "attach": cfg["attach"],
        "token_stride": cfg["token_stride"],
        "suspicious_thresh": cfg["suspicious_thresh"],
    }
    XCarTrainer.xcar_cfg = aux_cfg

    train_kwargs = {k: v for k, v in cfg.items() if k not in NON_TRAIN_KEYS}
    if args.data:
        train_kwargs["data"] = args.data
    if args.epochs:
        train_kwargs["epochs"] = args.epochs
    train_kwargs["device"] = args.device

    print(f"\nwarm start  : {weights}")
    print(f"aux modules : {aux_cfg}")
    print(f"loss weights: attn={W_ATTN}  contrast={W_CONTRAST}  "
          f"physics={W_PHYSICS}  fraud={W_FRAUD}")
    print("\ntrain kwargs:")
    for k in sorted(train_kwargs):
        print(f"  {k:<16} {train_kwargs[k]}")
    print()

    backup_dir = Path(cfg["backup_dir"])
    model = YOLO(weights)
    model.add_callback("on_train_start", make_loss_name_check())
    model.add_callback("on_fit_epoch_end", make_per_class_reporter())
    model.add_callback("on_fit_epoch_end", make_backup(backup_dir, int(cfg["backup_every"])))

    results = model.train(trainer=XCarTrainer, **train_kwargs)

    save_dir = Path(results.save_dir) if hasattr(results, "save_dir") else Path(cfg["project"]) / cfg["name"]
    print(f"\nTraining complete. Artifacts: {save_dir}")

    best = save_dir / "weights" / "best.pt"
    if not best.exists():
        print(f"WARNING: {best} not found — cannot run the final comparison.")
        return 0

    data = train_kwargs["data"]
    print(f"\nEvaluating {best} on the test split...")
    full = evaluate(best, cfg, data, args.device)

    baseline = None
    if not args.skip_baseline_eval:
        print(f"\nEvaluating Phase A baseline {weights} on the same split...")
        try:
            baseline = evaluate(weights, cfg, data, args.device)
        except Exception as e:
            print(f"WARNING: baseline evaluation failed ({e}); reporting full run only.")

    print_comparison(baseline, full)

    summary = {"phase": cfg["phase"], "full": full, "phase_a_baseline": baseline,
               "aux_cfg": aux_cfg,
               "loss_weights": {"attn": W_ATTN, "contrast": W_CONTRAST,
                                "physics": W_PHYSICS, "fraud": W_FRAUD}}
    out = save_dir / "full_test_metrics.json"
    out.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\nwrote {out}")

    try:
        shutil.copy2(out, backup_dir / "full_test_metrics.json")
    except OSError as e:
        print(f"WARNING: could not copy metrics to {backup_dir}: {e}")

    print("\nNow record this run in results/runs.csv.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
