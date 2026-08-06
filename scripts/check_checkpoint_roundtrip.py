"""Verify XCarDetectionModel survives deepcopy + save/load.

ultralytics does `deepcopy(de_parallel(model))` for EMA and again when writing
checkpoints. XCarDetectionModel holds forward-hook handles, which are not
picklable, so it overrides __getstate__/__setstate__ to drop and re-register
them. If that were wrong, training would die at the first periodic checkpoint
rather than at startup.

Run:  python scripts/check_checkpoint_roundtrip.py
"""

from __future__ import annotations

import copy
import io
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from xcar.model import _FeatureTap, XCarDetectionModel  # noqa: E402


def count_taps(model) -> int:
    """Live _FeatureTap hooks in the module tree — must be exactly one per attach point."""
    return sum(
        1
        for m in model.modules()
        for v in getattr(m, "_forward_hooks", {}).values()
        if isinstance(v, _FeatureTap)
    )

IMGSZ = 256  # topology check — size-independent
FAILURES: list[str] = []


def check(cond: bool, msg: str) -> None:
    print(f"  {'PASS' if cond else 'FAIL'}  {msg}")
    if not cond:
        FAILURES.append(msg)


def aux_runs(model, label: str) -> None:
    """Confirm the P3 hook still fires and aux modules still produce output."""
    model.eval()
    with torch.no_grad():
        model(torch.rand(1, 3, IMGSZ, IMGSZ))
    aux = model._aux
    check(count_taps(model) == 1, f"{label}: exactly 1 live P3 tap (got {count_taps(model)})")
    check(not model._feats, f"{label}: feature stash cleared after forward (nothing to bloat checkpoints)")
    check(bool(aux), f"{label}: aux stash is populated after forward")
    check("attn_maps" in aux, f"{label}: attn_maps present (P3 hook fired)")
    check("tokens_physics" in aux, f"{label}: tokens_physics present")
    if "tokens_physics" in aux:
        n = (IMGSZ // 8) ** 2
        check(
            tuple(aux["tokens_physics"].shape) == (1, n, 396),
            f"{label}: tokens_physics shape {tuple(aux['tokens_physics'].shape)} == (1, {n}, 396)",
        )


def main() -> int:
    print("=" * 66)
    print("Checkpoint / EMA round-trip check")
    print("=" * 66)

    torch.manual_seed(0)
    model = XCarDetectionModel(
        "yolo11m.yaml", nc=6, verbose=False,
        use_attention=True, use_physics=True, use_contrastive=True,
    )

    print("\noriginal model:")
    aux_runs(model, "original")

    print("\ndeepcopy (what EMA does every step):")
    clone = copy.deepcopy(model)
    aux_runs(clone, "deepcopy")
    check(
        len(clone._hook_handles) > 0,
        f"deepcopy re-registered its P3 hook ({len(clone._hook_handles)} handle(s))",
    )

    print("\ntorch.save -> torch.load (what save_period=10 does):")
    buf = io.BytesIO()
    torch.save({"model": copy.deepcopy(model).half(), "epoch": 0}, buf)
    buf.seek(0)
    ckpt = torch.load(buf, weights_only=False)
    loaded = ckpt["model"].float()
    aux_runs(loaded, "reloaded")

    print("\nweights preserved:")
    a = model.aux["adapter"].proj.weight
    b = loaded.aux["adapter"].proj.weight
    check(torch.allclose(a, b, atol=1e-2), "adapter weights survived the fp16 save/load round trip")

    print("\nconfig preserved:")
    check(loaded.config_summary() == model.config_summary(),
          f"config_summary matches: {loaded.config_summary()}")

    print("\n" + "=" * 66)
    if FAILURES:
        print(f"FAILED — {len(FAILURES)} failure(s)")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print("PASSED — checkpointing and EMA deepcopy are safe")
    print("=" * 66)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
