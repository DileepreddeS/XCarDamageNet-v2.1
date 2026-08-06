"""Build a stock YOLO11m and read its P3 neck channel count.

`in_ch` for FeatureTokenAdapter is read from the built model at runtime rather
than hardcoded. This script prints the value and writes it to
results/p3_channels.json for reference; the smoke test re-derives it
independently and never reads that file.

Run:  python scripts/probe_p3.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import ultralytics
from ultralytics import YOLO

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from xcar.model import find_detect_module, get_neck_channels  # noqa: E402

EXPECTED_ULTRALYTICS = "8.4.48"
IMGSZ = 1024


def main() -> int:
    print(f"ultralytics : {ultralytics.__version__}")
    print(f"torch       : {torch.__version__}")
    if ultralytics.__version__ != EXPECTED_ULTRALYTICS:
        print(
            f"WARNING: pinned version is {EXPECTED_ULTRALYTICS}. "
            f"Found {ultralytics.__version__}."
        )

    # Build architecture from YAML only — no COCO download needed for a channel probe.
    yolo = YOLO("yolo11m.yaml")
    model = yolo.model
    model.eval()

    detect = find_detect_module(model)
    print(f"\nDetect module   : {type(detect).__name__}")
    print(f"Detect.f (srcs) : {detect.f}")

    static_ch = get_neck_channels(model)
    print(f"\nStatic neck channels (from Detect.cv2 in_channels): {static_ch}")

    # Runtime confirmation: hook the layers that feed Detect and read real shapes.
    captured: dict[int, tuple] = {}

    def make_hook(idx: int):
        def hook(_m, _inp, out):
            if isinstance(out, torch.Tensor):
                captured[idx] = tuple(out.shape)

        return hook

    handles = [model.model[i].register_forward_hook(make_hook(i)) for i in detect.f]
    with torch.no_grad():
        model(torch.zeros(1, 3, IMGSZ, IMGSZ))
    for h in handles:
        h.remove()

    print(f"\nRuntime feature shapes at imgsz={IMGSZ}:")
    runtime_ch = []
    for i in detect.f:
        shape = captured[i]
        print(f"  layer {i:>2}  ->  {shape}")
        runtime_ch.append(shape[1])

    assert runtime_ch == list(static_ch), (
        f"Runtime channels {runtime_ch} disagree with static {static_ch}"
    )

    p3_ch = runtime_ch[0]
    p3_hw = captured[detect.f[0]][2]
    n_tokens = p3_hw * p3_hw

    print("\n" + "=" * 62)
    print(f"P3 NECK CHANNELS  : {p3_ch}")
    print(f"P3 SPATIAL @{IMGSZ} : {p3_hw} x {p3_hw}")
    print(f"P3 TOKEN COUNT N  : {n_tokens}")
    print(f"P4 NECK CHANNELS  : {runtime_ch[1]}")
    print(f"P5 NECK CHANNELS  : {runtime_ch[2]}")
    print("=" * 62)

    # Expectation check — informative, not fatal.
    if p3_ch != 256:
        print(f"NOTE: expected 256 P3 channels; actual is {p3_ch}.")
    if n_tokens != 16384:
        print(f"NOTE: expected N=16384 tokens at 1024px; actual is {n_tokens}.")

    out = REPO / "results" / "p3_channels.json"
    out.write_text(
        json.dumps(
            {
                "ultralytics": ultralytics.__version__,
                "torch": torch.__version__,
                "model": "yolo11m.yaml",
                "imgsz": IMGSZ,
                "detect_sources": list(detect.f),
                "p3_ch": p3_ch,
                "p4_ch": runtime_ch[1],
                "p5_ch": runtime_ch[2],
                "p3_hw": p3_hw,
                "n_tokens": n_tokens,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
