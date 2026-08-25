"""MAE pre-training of the adapter + physics encoder on YOLO P3 features @1024px.

WHY
---
`physics_encoder_best.pt` was MAE-pretrained on DINOv2 tokens at 518px. In v2.1
the encoder instead receives YOLO11m P3 features at 1024px — a shift in both
feature source and granularity. This re-pretrains it in-domain (CLAUDE.md 5).

PIPELINE
--------
    image (3, 1024, 1024)
        -> frozen YOLO11m (yolo11m.pt, eval, no_grad, never updated)
        -> P3 neck features (B, 256, 128, 128)          <- reconstruction TARGET
        -> FeatureTokenAdapter        [TRAINED]  (B, N, 384)
        -> PhysicsTokenEncoder        [TRAINED]  (B, N, 396)
        -> mask 75%, MAE decoder      [TRAINED, discarded after]
        -> predict the frozen P3 vector at each masked position

The target is the FROZEN backbone's own features, not the adapter's output.
That matters: with a jointly-trained target the objective is minimised by the
adapter emitting a constant, which reconstructs perfectly and encodes nothing.
Freezing the target makes collapse unavailable.

TOKEN BUDGET — read before changing `--mae-tokens`
--------------------------------------------------
At 1024px, P3 is 128x128 = 16,384 tokens. The adapter and the physics encoder
are per-token (1x1 conv and MLPs, no spatial mixing), so they cost linear time
and run on every token. The DECODER has attention, which is quadratic:

    16,384 tokens -> 16,384^2 x 4 B = 1.07 GB per head per layer
                     x 6 heads x 2 layers x batch 8  = ~103 GB   INFEASIBLE
     1,024 tokens ->  1,024^2 x 4 B = 4.2 MB per head per layer
                     x 6 heads x 2 layers x batch 8  = ~0.4 GB   fine

So the MAE objective is applied to a random subset of `--mae-tokens` (default
1024) positions per image per step, with 75% of that subset masked. Every token
still passes through the adapter and encoder each step — only the decoder sees
a subset, and the subset is redrawn every step, so all 16,384 positions are
trained over. This is a memory constraint, not a modelling choice; it is logged
in the checkpoint metadata.

OUTPUT
------
`models/physics_encoder_yolo_1024.pt` holds the adapter and physics-encoder
state dicts plus metadata. The decoder is training scaffolding and is saved
only in the resumable checkpoint, never in the final artefact.

Run:  python scripts/pretrain_physics_1024.py
      python scripts/pretrain_physics_1024.py --sanity-only
      python scripts/pretrain_physics_1024.py --resume
"""

from __future__ import annotations

import argparse
import math
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from ultralytics import YOLO  # noqa: E402

from xcar.model import (  # noqa: E402
    TOKEN_DIM,
    _FeatureTap,
    get_neck_channels,
    get_p3_layer_index,
)
from xcar.modules.adapter import FeatureTokenAdapter  # noqa: E402
from xcar.modules.physics_encoder import PhysicsTokenEncoder  # noqa: E402

PHYSICS_DIM = 396
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

DEFAULTS = dict(
    corpus="/scratch/ds3424/pretrain_cars/all_cars",
    yolo="yolo11m.pt",
    out=str(REPO / "models" / "physics_encoder_yolo_1024.pt"),
    imgsz=1024,
    batch=8,
    epochs=200,
    lr=1.5e-4,
    min_lr=0.0,
    warmup_epochs=5,
    weight_decay=0.05,
    mask_ratio=0.75,
    mae_tokens=1024,
    workers=8,
    save_every=1,
)


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------
class CarImageDataset(Dataset):
    """Unlabelled car images resized to imgsz x imgsz, scaled to [0, 1].

    Plain resize rather than ultralytics' letterbox: there are no boxes to keep
    consistent here, and the encoder only ever sees whole-image feature maps.
    """

    def __init__(self, root: str | Path, imgsz: int) -> None:
        self.root = Path(root)
        if not self.root.is_dir():
            raise FileNotFoundError(f"corpus directory not found: {self.root}")
        self.paths = sorted(
            p for p in self.root.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES
        )
        if not self.paths:
            raise FileNotFoundError(f"no images under {self.root}")
        self.imgsz = imgsz

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        # A handful of unreadable files in a 40k-image scrape must not kill a
        # multi-hour job; step to the next index instead.
        for offset in range(8):
            path = self.paths[(idx + offset) % len(self.paths)]
            try:
                with Image.open(path) as im:
                    im = im.convert("RGB").resize((self.imgsz, self.imgsz), Image.BILINEAR)
                    arr = torch.frombuffer(bytearray(im.tobytes()), dtype=torch.uint8)
                return (
                    arr.reshape(self.imgsz, self.imgsz, 3)
                    .permute(2, 0, 1)
                    .float()
                    .div_(255.0)
                )
            except Exception:
                continue
        raise RuntimeError(f"8 consecutive unreadable images starting at index {idx}")


# --------------------------------------------------------------------------
# decoder (training scaffolding — not part of the saved artefact)
# --------------------------------------------------------------------------
def sincos_2d_pos_embed(dim: int, h: int, w: int) -> torch.Tensor:
    """Fixed 2D sin-cos position embedding, (h*w, dim). Parameter-free."""
    if dim % 4:
        raise ValueError(f"pos-embed dim must be divisible by 4, got {dim}")
    omega = 1.0 / (10000 ** (torch.arange(dim // 4, dtype=torch.float32) / (dim / 4.0)))
    gy, gx = torch.meshgrid(
        torch.arange(h, dtype=torch.float32), torch.arange(w, dtype=torch.float32), indexing="ij"
    )
    out = []
    for grid in (gy.reshape(-1), gx.reshape(-1)):          # non-contiguous -> reshape
        ang = grid[:, None] * omega[None, :]
        out += [torch.sin(ang), torch.cos(ang)]
    return torch.cat(out, dim=1)                            # (h*w, dim)


class MAEDecoder(nn.Module):
    """Lightweight 2-layer, 384-dim transformer decoder.

    Predicts the frozen P3 feature vector at every masked position from the
    visible physics tokens plus position.
    """

    def __init__(
        self, token_dim: int = PHYSICS_DIM, dec_dim: int = 384, out_dim: int = 256,
        depth: int = 2, heads: int = 6,
    ) -> None:
        super().__init__()
        self.embed = nn.Linear(token_dim, dec_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dec_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=dec_dim, nhead=heads, dim_feedforward=dec_dim * 4,
            dropout=0.0, activation="gelu", batch_first=True, norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(dec_dim)
        self.pred = nn.Linear(dec_dim, out_dim)
        nn.init.normal_(self.mask_token, std=0.02)

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tokens: (B, S, 396) physics tokens at the sampled positions.
            mask:   (B, S) bool, True where the token is masked out.
            pos:    (B, S, dec_dim) position embedding for those positions.
        Returns:
            (B, S, out_dim) prediction at every sampled position.
        """
        x = self.embed(tokens)
        x = torch.where(mask.unsqueeze(-1), self.mask_token.to(x.dtype), x)
        x = self.blocks(x + pos)
        return self.pred(self.norm(x))


# --------------------------------------------------------------------------
# frozen backbone
# --------------------------------------------------------------------------
class FrozenP3Extractor:
    """Frozen YOLO11m with a forward hook on the neck layer feeding Detect P3."""

    def __init__(self, weights: str, device: torch.device) -> None:
        core = YOLO(weights).model
        core.eval().to(device)
        for p in core.parameters():
            p.requires_grad_(False)

        self.model = core
        self.device = device
        self.p3_ch = get_neck_channels(core)[0]
        self.p3_layer_idx = get_p3_layer_index(core)
        self._feats: dict[str, torch.Tensor] = {}
        self._handle = core.model[self.p3_layer_idx].register_forward_hook(
            _FeatureTap(self._feats, "p3")
        )

    @torch.no_grad()
    def __call__(self, imgs: torch.Tensor) -> torch.Tensor:
        """(B, 3, H, W) -> (B, p3_ch, H/8, W/8), detached."""
        self._feats.clear()
        self.model(imgs)
        p3 = self._feats.get("p3")
        if p3 is None:
            raise RuntimeError(
                f"P3 hook produced no feature at layer {self.p3_layer_idx}. "
                "The neck layer index is stale for this checkpoint."
            )
        out = p3.detach()
        self._feats.clear()
        return out

    def close(self) -> None:
        self._handle.remove()


# --------------------------------------------------------------------------
# MAE step
# --------------------------------------------------------------------------
def mae_step(
    p3: torch.Tensor,
    adapter: FeatureTokenAdapter,
    encoder: PhysicsTokenEncoder,
    decoder: MAEDecoder,
    pos_table: torch.Tensor,
    mask_ratio: float,
    n_sample: int,
) -> tuple[torch.Tensor, int]:
    """One masked-reconstruction step. Returns (loss, n_masked_per_image)."""
    b, c, h, w = p3.shape
    n_tokens = h * w

    tokens384 = adapter(p3)                    # (B, N, 384)   trained
    tokens396, _ = encoder(tokens384)          # (B, N, 396)   trained

    # Target: the frozen backbone's own P3 vectors, (B, N, C).
    target_all = p3.flatten(2).transpose(1, 2)  # non-contiguous -> never .view()

    # Redraw the decoder's token subset every step so all N positions train.
    s = min(n_sample, n_tokens)
    idx = torch.stack([torch.randperm(n_tokens, device=p3.device)[:s] for _ in range(b)])

    tokens = torch.gather(tokens396, 1, idx.unsqueeze(-1).expand(-1, -1, tokens396.shape[-1]))
    target = torch.gather(target_all, 1, idx.unsqueeze(-1).expand(-1, -1, c))
    pos = pos_table[idx]                        # (B, S, dec_dim)

    n_mask = max(1, int(round(s * mask_ratio)))
    noise = torch.rand(b, s, device=p3.device)
    mask = noise.argsort(dim=1).argsort(dim=1) < n_mask   # (B, S) bool, exactly n_mask True

    pred = decoder(tokens, mask, pos)

    # Per-token normalised target (MAE's norm_pix_loss): P3 channel magnitudes
    # vary widely across positions, and without this the loss is dominated by a
    # few high-energy tokens.
    mu = target.mean(dim=-1, keepdim=True)
    sd = target.var(dim=-1, keepdim=True, unbiased=False).add(1e-6).sqrt()
    target = (target - mu) / sd

    loss = (pred - target).pow(2).mean(dim=-1)  # (B, S)
    return (loss * mask).sum() / mask.sum().clamp_min(1), n_mask


# --------------------------------------------------------------------------
# sanity check — mandatory, runs before any training
# --------------------------------------------------------------------------
def sanity_check(args, device: torch.device) -> tuple[FrozenP3Extractor, int]:
    """Two real images end to end. Prints every shape; exits non-zero on mismatch."""
    print("=" * 74)
    print("SANITY CHECK — 2 images through frozen YOLO -> adapter -> encoder")
    print("=" * 74)

    failures: list[str] = []

    def check(cond: bool, msg: str) -> None:
        print(f"  {'PASS' if cond else 'FAIL'}  {msg}")
        if not cond:
            failures.append(msg)

    dataset = CarImageDataset(args.corpus, args.imgsz)
    print(f"\ncorpus  : {args.corpus}")
    print(f"images  : {len(dataset):,}")
    print(f"device  : {device}")

    imgs = torch.stack([dataset[0], dataset[1]]).to(device)
    print(f"\n  images            {tuple(imgs.shape)}")

    extractor = FrozenP3Extractor(args.yolo, device)
    print(f"  P3 layer index    {extractor.p3_layer_idx}")
    print(f"  P3 channels       {extractor.p3_ch}")

    p3 = extractor(imgs)
    hw = args.imgsz // 8
    n_tokens = hw * hw
    print(f"  P3 features       {tuple(p3.shape)}")

    check(p3.shape[1] == extractor.p3_ch, f"P3 channels {p3.shape[1]} == {extractor.p3_ch}")
    check(extractor.p3_ch == 256, f"P3 channel count is the expected 256 (got {extractor.p3_ch})")
    check(
        tuple(p3.shape[2:]) == (hw, hw),
        f"P3 spatial {tuple(p3.shape[2:])} == ({hw}, {hw}) at imgsz {args.imgsz}",
    )
    check(not p3.requires_grad, "P3 features carry no gradient — YOLO is frozen")
    check(
        not any(p.requires_grad for p in extractor.model.parameters()),
        "no YOLO parameter requires grad",
    )

    # Built exactly as XCarDetectionModel builds them, so the saved weights load
    # into the training model with no shape surgery.
    adapter = FeatureTokenAdapter(in_ch=extractor.p3_ch, token_dim=TOKEN_DIM).to(device)
    encoder = PhysicsTokenEncoder(in_dim=TOKEN_DIM).to(device)

    tokens384 = adapter(p3)
    print(f"  adapter output    {tuple(tokens384.shape)}")
    check(
        tuple(tokens384.shape) == (2, n_tokens, TOKEN_DIM),
        f"adapter emits (2, {n_tokens}, {TOKEN_DIM})",
    )
    check(adapter.in_ch == extractor.p3_ch, "adapter.in_ch was read from the built model")

    tokens396, physics = encoder(tokens384)
    print(f"  encoder output    {tuple(tokens396.shape)}")
    for k, v in physics.items():
        print(f"    physics[{k}]{'':<{max(0, 8 - len(k))}}  {tuple(v.shape)}")
    check(
        tuple(tokens396.shape) == (2, n_tokens, PHYSICS_DIM),
        f"encoder accepts adapter output and emits (2, {n_tokens}, {PHYSICS_DIM})",
    )
    check(
        encoder.OUTPUT_DIM == PHYSICS_DIM and tokens396.shape[-1] == PHYSICS_DIM,
        f"physics dim is {PHYSICS_DIM} (384 + 3 + 6 + 2 + 1)",
    )

    pos_table = sincos_2d_pos_embed(384, hw, hw).to(device)
    decoder = MAEDecoder(token_dim=PHYSICS_DIM, out_dim=extractor.p3_ch).to(device)
    loss, n_mask = mae_step(
        p3, adapter, encoder, decoder, pos_table, args.mask_ratio, args.mae_tokens
    )
    s = min(args.mae_tokens, n_tokens)
    print(f"  MAE subset        {s} of {n_tokens} tokens, {n_mask} masked "
          f"({n_mask / s:.0%})")
    print(f"  MAE loss          {float(loss.detach()):.6f}")
    check(torch.isfinite(loss).item(), "MAE loss is finite")
    check(float(loss.detach()) > 0, "MAE loss is non-zero")
    check(abs(n_mask / s - args.mask_ratio) < 0.01, f"mask ratio is {args.mask_ratio}")

    loss.backward()
    a_grad = sum(1 for p in adapter.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    e_grad = sum(1 for p in encoder.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    check(a_grad > 0, f"adapter receives gradient ({a_grad} tensors)")
    check(e_grad > 0, f"physics encoder receives gradient ({e_grad} tensors)")

    # The whole point of the artefact: these weights must load into a freshly
    # built pair with no shape mismatch.
    fresh_a = FeatureTokenAdapter(in_ch=extractor.p3_ch, token_dim=TOKEN_DIM)
    fresh_e = PhysicsTokenEncoder(in_dim=TOKEN_DIM)
    try:
        fresh_a.load_state_dict({k: v.cpu() for k, v in adapter.state_dict().items()}, strict=True)
        fresh_e.load_state_dict({k: v.cpu() for k, v in encoder.state_dict().items()}, strict=True)
        round_trip = True
    except RuntimeError as e:
        print(f"    load_state_dict error: {e}")
        round_trip = False
    check(round_trip, "adapter + encoder state dicts round-trip strict=True into fresh modules")

    print()
    if failures:
        print(f"SANITY CHECK FAILED — {len(failures)} failure(s):")
        for f in failures:
            print(f"  - {f}")
        print("=" * 74)
        extractor.close()
        raise SystemExit(1)
    print("SANITY CHECK PASSED")
    print("=" * 74)
    return extractor, n_tokens


# --------------------------------------------------------------------------
# schedule
# --------------------------------------------------------------------------
def lr_at(step: int, steps_per_epoch: int, args) -> float:
    """Linear warmup then cosine decay, computed per step."""
    warmup = args.warmup_epochs * steps_per_epoch
    total = args.epochs * steps_per_epoch
    if step < warmup:
        return args.lr * (step + 1) / max(1, warmup)
    p = (step - warmup) / max(1, total - warmup)
    return args.min_lr + (args.lr - args.min_lr) * 0.5 * (1 + math.cos(math.pi * min(1.0, p)))


# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    for key, val in DEFAULTS.items():
        ap.add_argument(f"--{key.replace('_', '-')}", type=type(val), default=val)
    ap.add_argument("--device", default="0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sanity-only", action="store_true",
                    help="run the shape/gradient check and exit (CLAUDE.md 5 gate)")
    ap.add_argument("--resume", action="store_true",
                    help="continue from the resumable checkpoint next to --out")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device(
        f"cuda:{args.device}" if torch.cuda.is_available() and args.device != "cpu" else "cpu"
    )

    print("=" * 74)
    print("PHYSICS RE-PRETRAINING — MAE on frozen YOLO11m P3 features @1024px")
    print("=" * 74)
    print(f"torch  : {torch.__version__}")
    print(f"cuda   : {torch.cuda.is_available()} "
          f"({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'})")

    extractor, n_tokens = sanity_check(args, device)
    if args.sanity_only:
        extractor.close()
        return 0

    hw = args.imgsz // 8
    adapter = FeatureTokenAdapter(in_ch=extractor.p3_ch, token_dim=TOKEN_DIM).to(device)
    encoder = PhysicsTokenEncoder(in_dim=TOKEN_DIM).to(device)
    decoder = MAEDecoder(token_dim=PHYSICS_DIM, out_dim=extractor.p3_ch).to(device)
    pos_table = sincos_2d_pos_embed(384, hw, hw).to(device)

    trainable = list(adapter.parameters()) + list(encoder.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, betas=(0.9, 0.95),
                                  weight_decay=args.weight_decay)

    dataset = CarImageDataset(args.corpus, args.imgsz)
    loader = DataLoader(
        dataset, batch_size=args.batch, shuffle=True, num_workers=args.workers,
        pin_memory=(device.type == "cuda"), drop_last=True, persistent_workers=args.workers > 0,
    )
    steps_per_epoch = len(loader)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_path.with_name(out_path.stem + "_resume.pt")

    start_epoch = 0
    if args.resume and ckpt_path.exists():
        state = torch.load(ckpt_path, map_location=device, weights_only=False)
        adapter.load_state_dict(state["adapter"])
        encoder.load_state_dict(state["physics_encoder"])
        decoder.load_state_dict(state["decoder"])
        optimizer.load_state_dict(state["optimizer"])
        start_epoch = int(state["epoch"])
        print(f"\nresumed from {ckpt_path} at epoch {start_epoch}")
    elif args.resume:
        print(f"\n--resume given but {ckpt_path} does not exist; starting from scratch")

    print(f"\nimages        : {len(dataset):,}")
    print(f"batch         : {args.batch}   steps/epoch: {steps_per_epoch:,}")
    print(f"epochs        : {args.epochs} (from {start_epoch})")
    print(f"lr            : {args.lr} cosine to {args.min_lr}, {args.warmup_epochs} warmup epochs")
    print(f"mask ratio    : {args.mask_ratio}")
    print(f"MAE tokens    : {min(args.mae_tokens, n_tokens)} of {n_tokens} per image per step")
    print(f"trainable     : adapter {sum(p.numel() for p in adapter.parameters()):,} | "
          f"encoder {sum(p.numel() for p in encoder.parameters()):,} | "
          f"decoder {sum(p.numel() for p in decoder.parameters()):,}")
    print(f"artefact      : {out_path}")
    print(f"resume ckpt   : {ckpt_path}")
    print()

    meta = dict(
        source="yolo11m P3 @1024 MAE",
        yolo_weights=args.yolo, imgsz=args.imgsz, p3_ch=extractor.p3_ch,
        p3_layer_idx=extractor.p3_layer_idx, token_dim=TOKEN_DIM, physics_dim=PHYSICS_DIM,
        n_tokens=n_tokens, mae_tokens=min(args.mae_tokens, n_tokens),
        mask_ratio=args.mask_ratio, epochs=args.epochs, batch=args.batch,
        lr=args.lr, corpus=str(args.corpus), n_images=len(dataset),
    )

    def save(epoch: int, loss: float) -> None:
        torch.save(
            {"adapter": adapter.state_dict(), "physics_encoder": encoder.state_dict(),
             "decoder": decoder.state_dict(), "optimizer": optimizer.state_dict(),
             "epoch": epoch, "loss": loss, "meta": meta},
            ckpt_path,
        )
        # Final artefact: encoder-side weights only. The decoder is scaffolding.
        torch.save(
            {"adapter": {k: v.cpu() for k, v in adapter.state_dict().items()},
             "physics_encoder": {k: v.cpu() for k, v in encoder.state_dict().items()},
             "epoch": epoch, "loss": loss, "meta": meta},
            out_path,
        )

    global_step = start_epoch * steps_per_epoch
    for epoch in range(start_epoch, args.epochs):
        adapter.train(); encoder.train(); decoder.train()
        running, seen, t0 = 0.0, 0, time.time()

        for i, imgs in enumerate(loader):
            lr = lr_at(global_step, steps_per_epoch, args)
            for g in optimizer.param_groups:
                g["lr"] = lr

            imgs = imgs.to(device, non_blocking=True)
            p3 = extractor(imgs)                       # frozen, no grad

            loss, _ = mae_step(p3, adapter, encoder, decoder, pos_table,
                               args.mask_ratio, args.mae_tokens)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite MAE loss at epoch {epoch + 1} step {i}")

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()

            running += float(loss.detach()) * imgs.shape[0]
            seen += imgs.shape[0]
            global_step += 1

            if i % 100 == 0:
                print(f"  epoch {epoch + 1:3d}/{args.epochs}  step {i:5d}/{steps_per_epoch}  "
                      f"loss {running / max(1, seen):.5f}  lr {lr:.2e}", flush=True)

        epoch_loss = running / max(1, seen)
        print(f"epoch {epoch + 1:3d}/{args.epochs}  loss {epoch_loss:.5f}  "
              f"{time.time() - t0:.0f}s", flush=True)

        if (epoch + 1) % args.save_every == 0 or (epoch + 1) == args.epochs:
            save(epoch + 1, epoch_loss)
            print(f"  saved -> {out_path}", flush=True)

    extractor.close()
    print(f"\nDone. Physics encoder + adapter weights: {out_path}")
    print("Load in Phase C with:")
    print("    state = torch.load(path, weights_only=False)")
    print("    model.aux['adapter'].load_state_dict(state['adapter'])")
    print("    model.aux['physics'].load_state_dict(state['physics_encoder'])")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
