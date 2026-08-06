from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import ultralytics  # noqa: E402
from ultralytics import YOLO  # noqa: E402

EXPECTED_ULTRALYTICS = "8.4.48"

# Keys in the config that are ours, not ultralytics' — strip before train().
NON_TRAIN_KEYS = {"phase", "description", "model", "use_attention", "use_physics", "use_contrastive"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO / "configs" / "phase_a.yaml"))
    ap.add_argument("--data", default=None, help="override the dataset YAML path")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--device", default="0")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())

    print("=" * 70)
    print(f"PHASE {str(cfg['phase']).upper()} — {cfg['description']}")
    print("=" * 70)
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
    assert not any(cfg[k] for k in ("use_attention", "use_physics", "use_contrastive")), \
        "Phase A is the stock control — no aux modules"

    train_kwargs = {k: v for k, v in cfg.items() if k not in NON_TRAIN_KEYS}
    if args.data:
        train_kwargs["data"] = args.data
    if args.epochs:
        train_kwargs["epochs"] = args.epochs
    train_kwargs["device"] = args.device

    print("\ntrain kwargs:")
    for k in sorted(train_kwargs):
        print(f"  {k:<16} {train_kwargs[k]}")
    print()

    model = YOLO(cfg["model"])  # COCO-pretrained YOLO11m
    results = model.train(**train_kwargs)

    save_dir = Path(results.save_dir) if hasattr(results, "save_dir") else Path(train_kwargs["project"])
    print(f"\nTraining complete. Artifacts: {save_dir}")

    # Validate the best checkpoint on the TEST split for the ablation table.
    best = save_dir / "weights" / "best.pt"
    if best.exists():
        print(f"\nEvaluating {best} on the test split...")
        metrics = YOLO(str(best)).val(
            data=train_kwargs["data"], split="test", imgsz=cfg["imgsz"],
            batch=cfg["batch"], device=args.device,
        )
        summary = {
            "phase": cfg["phase"],
            "test_mAP50": float(metrics.box.map50),
            "test_mAP50_95": float(metrics.box.map),
            "per_class_AP50": {
                metrics.names[int(c)]: float(metrics.box.ap50[i])
                for i, c in enumerate(metrics.box.ap_class_index)
            },
        }
        print(json.dumps(summary, indent=2))
        out = save_dir / "phase_a_test_metrics.json"
        out.write_text(json.dumps(summary, indent=2) + "\n")
        print(f"wrote {out}")
        print("\nNow record this run in results/runs.csv.")
    else:
        print(f"WARNING: {best} not found — record the run manually.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
