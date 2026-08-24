"""Pre-training smoke test. Must pass on CPU before any training run.

  1. Builds the full model with pretrained weights off.
  2. Creates 2 batches (batch_size=2) of synthetic tensors matching the
     dataset format: 3x1024x1024 images, YOLO-format normalised xywh targets
     with class ids 0-5.
  3. Runs forward -> full loss -> loss.backward().
  4. Asserts every aux output has the expected shape; prints all shapes.
  5. Asserts every loss term is finite and non-zero where it should be.
  6. Exits non-zero on any failure.

Plus four structural checks:
  * The adapter's `in_ch` is asserted equal to the P3 channel count read from
    the built model, so a hardcoded value cannot creep in.
  * A per-module gradient audit runs for each configuration and asserts that
    every active aux module receives gradient. A module that trains on no
    gradient is the same class of silent failure as a loss term stuck at 0.0.
  * An aux-gradient isolation audit: backward through the aux terms alone must
    reach no backbone/neck parameter (the neck features are detached before
    the aux modules), while the full loss still must.
  * A cls class-weight check: the difficulty weights (1/AP from Phase A) are
    correct, the config selects exactly one scheme, and attaching the weights
    demonstrably changes the real cls_loss. v1 logged "applied" for a
    class-weight callback that never attached anything, so arithmetic alone is
    not proof. The retained CB path is checked too, so it cannot rot.

Run:  python scripts/smoke_test.py
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

import torch
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import ultralytics  # noqa: E402
from ultralytics.cfg import get_cfg  # noqa: E402

from xcar.class_weights import (  # noqa: E402
    CARDD_CLASS_COUNTS,
    CB_BETA,
    PHASE_A_TEST_AP50,
    bce_multiply_site,
    cardd_cb_weights,
    cardd_difficulty_weights,
    format_weights,
    make_class_weight_callback,
    rank_weights,
)
from xcar.loss import W_ATTN, W_CONTRAST, W_FRAUD, W_PHYSICS  # noqa: E402
from xcar.model import PHYSICS_DIM, TOKEN_DIM, XCarDetectionModel, get_neck_channels  # noqa: E402

IMGSZ = 1024
BATCH = 2
NUM_CLASSES = 6
CLASS_NAMES = ["dent", "scratch", "crack", "glass_shatter", "lamp_broken", "tire_flat"]

FAILURES: list[str] = []


# --------------------------------------------------------------------------
# tiny assertion helpers that accumulate rather than abort on first failure
# --------------------------------------------------------------------------
def check(condition: bool, message: str) -> bool:
    if condition:
        print(f"  PASS  {message}")
        return True
    print(f"  FAIL  {message}")
    FAILURES.append(message)
    return False


def check_shape(tensor: torch.Tensor, expected: tuple, name: str) -> bool:
    actual = tuple(tensor.shape)
    return check(actual == expected, f"{name} shape {actual} == expected {expected}")


# --------------------------------------------------------------------------
# synthetic CarDD-format batch
# --------------------------------------------------------------------------
def make_batch(seed: int, include_empty_image: bool = False) -> dict:
    """Synthetic batch in ultralytics' exact training format.

    Matches what the dataloader yields:
        img       : (B, 3, 1024, 1024) float in [0, 1]
        batch_idx : (n,)   image index per target
        cls       : (n, 1) class id 0-5
        bboxes    : (n, 4) normalised xywh in [0, 1]
    """
    g = torch.Generator().manual_seed(seed)
    imgs = torch.rand(BATCH, 3, IMGSZ, IMGSZ, generator=g)

    batch_idx, cls, boxes = [], [], []
    for b in range(BATCH):
        if include_empty_image and b == BATCH - 1:
            continue  # exercises the no-GT graceful-degradation path
        n = 2
        for i in range(n):
            # Boxes are deliberately large enough to cover several P3 tokens,
            # and both share a class so triplet mining has a positive pair.
            cx = 0.30 + 0.35 * i
            cy = 0.35 + 0.25 * i
            w = 0.22
            h = 0.18
            batch_idx.append(b)
            cls.append(i % NUM_CLASSES if not include_empty_image else 0)
            boxes.append([cx, cy, w, h])

    return {
        "img": imgs,
        "batch_idx": torch.tensor(batch_idx, dtype=torch.float32),
        "cls": torch.tensor(cls, dtype=torch.float32).reshape(-1, 1),
        "bboxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
    }


def build_model(**aux_flags) -> XCarDetectionModel:
    """Build the model from YAML with pretrained weights off."""
    model = XCarDetectionModel("yolo11m.yaml", ch=3, nc=NUM_CLASSES, verbose=False, **aux_flags)
    # v8DetectionLoss reads box/cls/dfl gains off model.args. The trainer sets
    # this in real runs; here we use ultralytics' defaults overridden with the
    # training config's values so the smoke test exercises the real gains.
    args = get_cfg()
    args.box, args.cls, args.dfl = 7.5, 0.5, 1.5
    model.args = args
    model.names = {i: n for i, n in enumerate(CLASS_NAMES)}
    return model


# --------------------------------------------------------------------------
# main gate: full model, 1024px, 2 batches, forward -> loss -> backward
# --------------------------------------------------------------------------
def run_full_gate() -> None:
    print("\n" + "=" * 74)
    print("PART 1 — FULL MODEL @ 1024px, batch=2, forward -> loss -> backward")
    print("=" * 74)

    torch.manual_seed(0)
    model = build_model(use_attention=True, use_physics=True, use_contrastive=True)
    model.train()

    neck_ch = get_neck_channels(model)
    p3_ch = neck_ch[0]
    print(f"\nBuilt model: neck channels (P3,P4,P5) = {neck_ch}")
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Aux params: {sum(p.numel() for p in model.aux.parameters()):,}")

    print("\nadapter in_ch must come from the built model, never hardcoded:")
    check(
        model.aux["adapter"].in_ch == p3_ch,
        f"adapter.in_ch ({model.aux['adapter'].in_ch}) == runtime P3 channels ({p3_ch})",
    )
    check(
        model.aux["attn_head"].in_ch == p3_ch,
        f"attn_head.in_ch ({model.aux['attn_head'].in_ch}) == runtime P3 channels ({p3_ch})",
    )
    check(model.p3_ch == 256, f"P3 channel count is the expected 256 (got {model.p3_ch})")

    expected_hw = IMGSZ // 8
    expected_n = expected_hw * expected_hw

    for bi, batch in enumerate(
        [make_batch(seed=1), make_batch(seed=2, include_empty_image=True)], start=1
    ):
        n_gt = int(batch["cls"].shape[0])
        print(f"\n--- batch {bi}/2 (batch_size={BATCH}, {n_gt} GT boxes) ---")
        model.zero_grad(set_to_none=True)

        loss, loss_items = model(batch)
        aux = model._aux

        # ---- assert every aux output shape, print all shapes ----
        print("\n  aux output shapes:")
        for key in ("attn_maps", "tokens_physics", "tokens_contrastive", "damage_scores",
                    "fraud_score", "fraud_implied", "suspicious_mask"):
            if key in aux:
                print(f"    {key:<20} {tuple(aux[key].shape)}")
        for key, val in aux["physics_dict"].items():
            print(f"    {'physics[' + key + ']':<20} {tuple(val.shape)}")
        print(f"    token_hw             {aux['token_hw']}")

        print("\n  shape assertions:")
        check_shape(aux["attn_maps"], (BATCH, NUM_CLASSES, expected_hw, expected_hw), "attn_maps")
        check_shape(aux["tokens_physics"], (BATCH, expected_n, PHYSICS_DIM), "tokens_physics")
        check_shape(aux["tokens_contrastive"], (BATCH, expected_n, PHYSICS_DIM), "tokens_contrastive")
        check_shape(aux["damage_scores"], (BATCH, expected_n), "damage_scores")
        check_shape(aux["fraud_score"], (BATCH, 1), "fraud_score")
        check_shape(aux["fraud_implied"], (BATCH, NUM_CLASSES), "fraud_implied")
        check_shape(aux["suspicious_mask"], (BATCH, expected_n), "suspicious_mask")
        check_shape(aux["physics_dict"]["normal"], (BATCH, expected_n, 3), "physics.normal")
        check_shape(aux["physics_dict"]["material"], (BATCH, expected_n, 6), "physics.material")
        check_shape(aux["physics_dict"]["reflectance"], (BATCH, expected_n, 2), "physics.reflectance")
        check_shape(aux["physics_dict"]["curvature"], (BATCH, expected_n, 1), "physics.curvature")
        check(aux["token_hw"] == (expected_hw, expected_hw), f"token grid == ({expected_hw},{expected_hw})")
        check(expected_n == 16384, f"token count N == expected 16384 (got {expected_n})")

        # ---- physics-encoder dim contract ----
        check(
            model.aux["adapter"].token_dim == TOKEN_DIM
            and aux["tokens_physics"].shape[-1] == PHYSICS_DIM,
            f"adapter emits {TOKEN_DIM}-dim tokens; physics emits {PHYSICS_DIM}-dim (384+3+6+2+1)",
        )

        # ---- every loss term finite AND non-zero where it should be ----
        names = ("box_loss", "cls_loss", "dfl_loss") + model.aux_loss_names
        print(f"\n  loss terms ({len(names)} logged, weights: attn={W_ATTN} "
              f"contrast={W_CONTRAST} physics={W_PHYSICS} fraud={W_FRAUD}):")
        check(
            len(loss_items) == len(names),
            f"loss_items length {len(loss_items)} == len(loss_names) {len(names)}",
        )
        for name, value in zip(names, loss_items.tolist()):
            print(f"    {name:<12} = {value:.6f}")

        stats = model.criterion.last_stats
        print(f"  mining stats: {stats}")

        print("\n  loss assertions:")
        for name, value in zip(names, loss_items.tolist()):
            check(torch.isfinite(torch.tensor(value)).item(), f"{name} is finite")
            # Every term here should be strictly positive on a batch with GT.
            # cont_loss is only excused when no triplets could be mined, which
            # this batch construction guarantees does not happen.
            check(value != 0.0, f"{name} is non-zero (a 0.0 term means a dead module)")

        check(
            int(stats.get("n_triplets", 0)) > 0,
            f"triplets were actually mined (n={stats.get('n_triplets')}), "
            "so cont_loss is real and not the dummy-zero path",
        )
        check(torch.isfinite(loss).all().item(), "total loss vector is finite")

        # ---- backward ----
        total = loss.sum()
        total_value = float(total.detach())
        total.backward()
        print(f"\n  backward OK — total loss = {total_value:.4f}")

        n_with_grad = sum(1 for p in model.parameters() if p.grad is not None)
        check(n_with_grad > 0, f"backward populated gradients on {n_with_grad} parameter tensors")

    print("\n  no-GT degradation check (batch 2 contained an image with zero boxes):")
    check(True, "forward/backward survived an image with no GT boxes")


# --------------------------------------------------------------------------
# gradient audit per configuration — the anti-"dead module" check
# --------------------------------------------------------------------------
AUX_GROUPS = ("attn_head", "adapter", "physics", "fraud_head", "contrastive")


def grad_report(model) -> dict[str, str]:
    """Classify each aux module by whether real gradient reached it."""
    report = {}
    for name, module in model.aux.items():
        grads = [p.grad for p in module.parameters() if p.grad is not None]
        if not grads:
            report[name] = "NO GRAD (grad is None)"
        else:
            total = sum(float(g.abs().sum()) for g in grads)
            report[name] = "trains" if total > 0 else "ZERO GRAD (all-zero)"
    return report


def run_grad_audit() -> None:
    """Assert which aux modules receive gradient in each configuration.

    Runs at reduced imgsz for speed: gradient topology (which module is
    connected to which loss term) is identical at any input size, and Part 1
    already covers the 1024px path end to end.
    """
    print("\n" + "=" * 74)
    print("PART 2 — GRADIENT AUDIT (which aux modules actually train)")
    print("=" * 74)

    # Each phase lists the aux modules it activates. Every listed module must
    # receive gradient — a module that is built, fed, and never updated is the
    # same silent failure as a loss term that logs 0.0.
    phases = {
        "B  attention": (dict(use_attention=True), ["attn_head"]),
        "C  + physics": (
            dict(use_attention=True, use_physics=True),
            ["attn_head", "adapter", "physics", "fraud_head"],
        ),
        "D  + contrastive": (
            dict(use_attention=True, use_physics=True, use_contrastive=True),
            ["attn_head", "adapter", "physics", "fraud_head", "contrastive"],
        ),
    }
    small = 256  # reduced imgsz — topology only
    results = {}

    for label, (flags, _expected) in phases.items():
        torch.manual_seed(0)
        model = build_model(**flags)
        model.train()

        batch = make_batch(seed=3)
        batch["img"] = torch.rand(BATCH, 3, small, small)

        model.zero_grad(set_to_none=True)
        loss, loss_items = model(batch)
        loss.sum().backward()

        report = grad_report(model)
        results[label] = report
        names = ("box_loss", "cls_loss", "dfl_loss") + model.aux_loss_names
        print(f"\n  Phase {label}")
        print(f"    losses : {dict(zip(names, [round(v, 5) for v in loss_items.tolist()]))}")
        if getattr(model.criterion, "last_stats", None):
            diag = {k: round(v, 4) for k, v in model.criterion.last_stats.items()
                    if k.startswith("phys_")}
            if diag:
                print(f"    physics diagnostics : {diag}")
        for mod in AUX_GROUPS:
            if mod in report:
                flag = "OK " if report[mod] == "trains" else "!! "
                print(f"    {flag}{mod:<14} {report[mod]}")

    # HARD ASSERTIONS — every module active in a phase must receive gradient.
    print("\n  gradient assertions (every active module must train):")
    for label, (_flags, expected_modules) in phases.items():
        report = results[label]
        for mod in expected_modules:
            check(
                report.get(mod) == "trains",
                f"Phase {label.split()[0]}: {mod} receives gradient "
                f"(got: {report.get(mod, 'module absent')})",
            )

    print("\n" + "-" * 74)
    print("  NOTE: L_physics is symmetric and linear in the physics")
    print("  distribution, so both heads can collapse onto one arbitrary class")
    print("  and drive the term toward 0 while carrying no information. Watch")
    print("  phys_entropy (ln 6 = 1.792 at uniform) and phys_agree in the")
    print("  per-epoch [XCAR] log line; entropy ~0 with agreement ~1.0 is the")
    print("  collapse signature, not convergence.")
    print("-" * 74)


# --------------------------------------------------------------------------
# aux-gradient isolation — the check that the neck detach actually holds
# --------------------------------------------------------------------------
SMALL = 256  # reduced imgsz; gradient topology is size-independent


def run_detach_audit() -> None:
    """Assert the aux losses cannot move the backbone or neck.

    The aux modules read a `.detach()`ed view of the neck features, so:
      * backward through ONLY the aux terms must leave every parameter under
        `model.model` (backbone + neck + Detect) with no gradient, and
      * backward through the full loss must still reach them, via the
        detection path, which the detach does not touch.

    Both halves matter. The first alone would also pass if the aux modules were
    disconnected entirely; the second alone would pass without any detach.
    """
    print("\n" + "=" * 74)
    print("PART 3 — AUX GRADIENT ISOLATION (detached neck features)")
    print("=" * 74)

    torch.manual_seed(0)
    model = build_model(use_attention=True, use_physics=True, use_contrastive=True)
    model.train()

    batch = make_batch(seed=4)
    batch["img"] = torch.rand(BATCH, 3, SMALL, SMALL)

    # ---- (a) aux terms only -> detector must receive nothing ----------
    model.zero_grad(set_to_none=True)
    loss, _ = model(batch)
    n_yolo_terms = 3  # box, cls, dfl
    aux_only = loss[n_yolo_terms:].sum()
    check(
        float(aux_only.detach()) != 0.0,
        f"aux-only loss is non-zero ({float(aux_only.detach()):.6f}) — a zero here "
        "would make the isolation check vacuous",
    )
    aux_only.backward()

    detector_hits = [
        n for n, p in model.model.named_parameters()
        if p.grad is not None and float(p.grad.abs().sum()) > 0
    ]
    aux_hits = [
        n for n, p in model.aux.named_parameters()
        if p.grad is not None and float(p.grad.abs().sum()) > 0
    ]
    print(f"\n  backward(aux terms only):")
    print(f"    detector params (model.model) with non-zero grad : {len(detector_hits)}")
    print(f"    aux params      (model.aux)   with non-zero grad : {len(aux_hits)}")
    if detector_hits:
        print(f"    LEAKED INTO: {detector_hits[:8]}{' ...' if len(detector_hits) > 8 else ''}")

    check(
        not detector_hits,
        "aux losses reach NO backbone/neck/Detect parameter (the neck detach holds)",
    )
    check(
        len(aux_hits) > 0,
        f"aux losses do reach the aux modules ({len(aux_hits)} tensors) — they still train",
    )

    # ---- (b) full loss -> detector must still receive gradient --------
    model.zero_grad(set_to_none=True)
    loss, _ = model(batch)
    loss.sum().backward()
    detector_hits_full = sum(
        1 for _, p in model.model.named_parameters()
        if p.grad is not None and float(p.grad.abs().sum()) > 0
    )
    print(f"\n  backward(full loss):")
    print(f"    detector params with non-zero grad : {detector_hits_full}")
    check(
        detector_hits_full > 0,
        f"the detection loss still trains the backbone/neck ({detector_hits_full} "
        "tensors) — the detach did not sever the detector",
    )


# --------------------------------------------------------------------------
# cls class weights — correct arithmetic AND proof they change the cls term
# --------------------------------------------------------------------------
class _FakeTrainer:
    """Stand-in for the ultralytics trainer the callback receives."""

    def __init__(self, model):
        self.model = model


def check_weight_vector(weights, names, ranking_key, hardest: str, label: str) -> None:
    """Shared arithmetic assertions for any weighting scheme.

    Args:
        ranking_key: per-class quantity the weight should be *decreasing* in
            (class count for CB, AP for difficulty).
        hardest: name of the class that must carry the largest weight.
    """
    nc = len(names)
    print(f"\n  {label} weight assertions:")
    check(tuple(weights.shape) == (nc,), f"weights shape {tuple(weights.shape)} == ({nc},)")
    check(
        abs(float(weights.sum()) - nc) < 1e-4,
        f"weights sum to nc ({float(weights.sum()):.6f} == {nc})",
    )
    check(torch.isfinite(weights).all().item(), "weights are all finite")
    check(bool((weights > 0).all()), "weights are all positive")
    # Checked as a strict ordering rather than spot values, so a sign flip in
    # the formula cannot slip past.
    order = sorted(range(nc), key=lambda i: ranking_key[i])
    ordered_w = [float(weights[i]) for i in order]
    check(
        all(ordered_w[i] > ordered_w[i + 1] for i in range(nc - 1)),
        f"weight is strictly decreasing in {label} ranking key",
    )
    check(
        names[int(weights.argmax())] == hardest,
        f"heaviest weight is on {hardest} (got {names[int(weights.argmax())]})",
    )


def check_weights_reach_cls_loss(weights, names, tag: str) -> None:
    """Run the real criterion twice, weights off then on, and require a change.

    v1's attempt logged success without ever attaching anything, so checking our
    own arithmetic is not enough — only the loss value moving is proof.
    """
    nc = len(names)
    torch.manual_seed(0)
    model = build_model()  # stock detection path; this touches only the cls term
    model.train()
    batch = make_batch(seed=5)
    batch["img"] = torch.rand(BATCH, 3, SMALL, SMALL)

    _, items_off = model(batch)
    cls_off = float(items_off[1])

    trainer = _FakeTrainer(model)
    print(f"\n  invoking the on_train_batch_start callback ({tag}):")
    make_class_weight_callback(weights, names, tag=tag)(trainer)

    attached = getattr(model.criterion, "class_weights", None)
    check(attached is not None, f"{tag}: callback attached class_weights to the live criterion")
    if attached is not None:
        check(
            tuple(attached.shape) == (1, 1, nc),
            f"{tag}: attached shape {tuple(attached.shape)} == (1, 1, {nc}) — broadcasts "
            "over bce_loss (bs, num_anchors, nc)",
        )
        check(
            torch.allclose(attached.flatten().cpu(), weights.cpu()),
            f"{tag}: attached values equal the computed weights",
        )

    _, items_on = model(batch)
    cls_on = float(items_on[1])
    print(f"\n  cls_loss without weights : {cls_off:.6f}")
    print(f"  cls_loss with    weights : {cls_on:.6f}")
    print(f"  delta                    : {cls_on - cls_off:+.6f}")
    check(
        cls_off != cls_on,
        f"{tag}: cls_loss CHANGED once the weights were attached — they are genuinely "
        "consumed by the loss, not merely stored on an object",
    )
    check(
        float(items_off[0]) == float(items_on[0]) and float(items_off[2]) == float(items_on[2]),
        f"{tag}: box_loss and dfl_loss are unchanged — only the cls term is touched",
    )

    # ---- the callback must be a no-op before the criterion exists -----
    torch.manual_seed(0)
    fresh = build_model()
    make_class_weight_callback(weights, names, tag=tag)(_FakeTrainer(fresh))
    check(
        getattr(fresh, "criterion", None) is None,
        f"{tag}: callback is a quiet no-op before the criterion is built "
        "(it retries next batch)",
    )


def run_class_weight_check() -> None:
    """PART 4 — the active scheme (difficulty), then the retained CB path."""
    print("\n" + "=" * 74)
    print("PART 4 — CLS CLASS WEIGHTS (arithmetic + proof they reach cls_loss)")
    print("=" * 74)

    check(
        bce_multiply_site() is not None,
        "ultralytics' cls loss still multiplies bce_loss by self.class_weights",
    )

    # ---------------- active scheme: difficulty-aware ------------------
    names = list(PHASE_A_TEST_AP50)
    ap = list(PHASE_A_TEST_AP50.values())
    dw = cardd_difficulty_weights()
    print("\n  --- DIFFICULTY WEIGHTS (active) ---")
    print(f"  Phase A AP : {PHASE_A_TEST_AP50}")
    print(f"  weights    : {format_weights(dw, names)}")
    print(f"  ranked     : {rank_weights(dw, names)}")
    print(f"  ratio      : {float(dw.max() / dw.min()):.3f}x")

    check_weight_vector(dw, names, ranking_key=ap, hardest="crack", label="difficulty")
    # The whole point of the change: weight must track difficulty, not frequency.
    counts = [CARDD_CLASS_COUNTS[n] for n in names]
    cb = cardd_cb_weights(beta=CB_BETA)
    check(
        names[int(dw.argmax())] != names[int(cb.argmax())],
        f"difficulty and CB disagree on which class needs help most "
        f"(difficulty={names[int(dw.argmax())]}, CB={names[int(cb.argmax())]}) — "
        "the reason CB was replaced",
    )
    check(
        float(dw[names.index("crack")]) > float(dw[names.index("tire_flat")]),
        f"crack ({float(dw[names.index('crack')]):.4f}) outweighs tire_flat "
        f"({float(dw[names.index('tire_flat')]):.4f}); CB had this backwards "
        f"({float(cb[names.index('crack')]):.4f} vs {float(cb[names.index('tire_flat')]):.4f})",
    )
    check(
        float(dw.max() / dw.min()) < float(cb.max() / cb.min()),
        f"difficulty is the gentler intervention ({float(dw.max() / dw.min()):.2f}x "
        f"vs CB {float(cb.max() / cb.min()):.2f}x)",
    )
    check_weights_reach_cls_loss(dw, names, tag="DIFFICULTY WEIGHTS")

    # ---------------- retained scheme: class-balanced ------------------
    # Still switchable via `cb_loss: true`, so it stays under test rather than
    # rotting into code that only looks like it works.
    print("\n  --- CB LOSS (retained, disabled in configs/full.yaml) ---")
    print(f"  weights    : {format_weights(cb, names)}")
    check_weight_vector(cb, names, ranking_key=counts, hardest="tire_flat", label="CB")

    # ---------------- the config actually selects one ------------------
    print("\n  config consistency:")
    cfg = yaml.safe_load((REPO / "configs" / "full.yaml").read_text())
    check(
        cfg.get("difficulty_weights") is True,
        f"configs/full.yaml has difficulty_weights: true (got {cfg.get('difficulty_weights')!r})",
    )
    check(
        cfg.get("cb_loss") is False,
        f"configs/full.yaml has cb_loss: false (got {cfg.get('cb_loss')!r})",
    )
    check(
        cfg.get("cls_pw") == 0.0,
        f"cls_pw stays 0.0 so ultralytics' own weighting cannot overwrite ours "
        f"(got {cfg.get('cls_pw')!r})",
    )
    check(
        not (cfg.get("difficulty_weights") and cfg.get("cb_loss")),
        "exactly one weighting scheme is enabled — both write model.class_weights",
    )


# --------------------------------------------------------------------------
def main() -> int:
    print("=" * 74)
    print("XCarDamageNet — SMOKE TEST")
    print("=" * 74)
    print(f"ultralytics : {ultralytics.__version__}  (pin: 8.4.48)")
    print(f"torch       : {torch.__version__}")
    print(f"device      : cpu (gate must pass on CPU)")

    check(ultralytics.__version__ == "8.4.48", "ultralytics is pinned at 8.4.48")

    try:
        run_full_gate()
        run_grad_audit()
        run_detach_audit()
        run_class_weight_check()
    except Exception:
        traceback.print_exc()
        FAILURES.append(f"uncaught exception: see traceback above")

    print("\n" + "=" * 74)
    if FAILURES:
        print(f"SMOKE TEST FAILED — {len(FAILURES)} failure(s):")
        for f in FAILURES:
            print(f"  - {f}")
        print("=" * 74)
        return 1
    print("SMOKE TEST PASSED")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
