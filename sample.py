"""
Run super-resolution sampling with few-step Euler integration.

Usage:
  python sample.py --config config.yaml --ckpt runs/sr_x4/checkpoints/step_10000.pt
  python sample.py --config config.yaml --ckpt <ckpt> --steps 4
  # Optional: choose a different set of HR images to evaluate
  python sample.py --config config.yaml --ckpt <ckpt> --hr_dir "C:\\path\\to\\some_HR_images"

Behavior:
- Reads HR images from config.paths.valid_hr_dir (or --hr_dir if provided)
- Makes LR by bicubic x{scale}
- Produces SR with the chosen number of steps (default from config.fm.sampler_default_steps)
- Saves: bicubic baseline, SR result, and a 3-panel comparison grid to {exp_dir}/samples/
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import re
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode
from torchvision.utils import save_image, make_grid

from model import UNetSR
from flowmatch import euler_sampler
from utils import load_config, get_device


VALID_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def list_images(root: Path) -> List[Path]:
    return [p for p in sorted(root.iterdir()) if p.suffix.lower() in VALID_EXTS]


@torch.no_grad()
def load_model(config, ckpt_path: Path, device: torch.device) -> UNetSR:
    """Instantiate UNetSR and load weights (EMA preferred if available)."""
    model = UNetSR(
        in_channels=6,
        out_channels=3,
        base_channels=48,
        channel_multipliers=(1, 2, 4),
        num_res_blocks=2,
        time_dim=256,
        use_bottleneck_attention=True,
        use_gradient_checkpointing=False,  # not needed for inference
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict) and "ema" in ckpt and isinstance(ckpt["ema"], dict):
        # Merge EMA weights into model (keep missing buffers from current state)
        model.load_state_dict({**model.state_dict(), **ckpt["ema"]}, strict=False)
        print(f"[load] EMA weights loaded from {ckpt_path.name}")
    else:
        model.load_state_dict(ckpt["model"], strict=True)
        print(f"[load] model weights loaded from {ckpt_path.name}")

    model.eval()
    return model


def compute_starts(length: int, tile: int, overlap: int) -> List[int]:
    """Compute start indices for sliding-window tiling that covers the entire dimension."""
    assert tile > 0 and overlap >= 0 and tile > 2 * overlap
    stride = tile - 2 * overlap
    starts: List[int] = []
    pos = 0
    while True:
        if pos + tile >= length:
            starts.append(max(length - tile, 0))
            break
        starts.append(pos)
        pos += stride
    # Deduplicate while preserving order
    seen = set()
    out: List[int] = []
    for s in starts:
        if s not in seen:
            out.append(s)
            seen.add(s)
    return out


@torch.no_grad()
def sr_tiled(
    model: UNetSR,
    lr_img: torch.Tensor,               # (3, h, w) in [0,1]
    steps: int,
    tile_hr: int,
    overlap_hr: int,
    scale: int = 4,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    Tiled SR to avoid OOM on large images.
    - We tile in LR space (tile_lr = tile_hr/scale), run the model per tile, and stitch in HR space.
    - Overlaps are averaged to hide seams.
    Returns (3, H, W) in [0,1].
    """
    assert lr_img.dim() == 3, "Expected (3, h, w)"
    device = device or (lr_img.device if lr_img.is_cuda else torch.device("cpu"))

    x_lr = lr_img.unsqueeze(0).to(device)  # (1,3,h,w)
    _, _, h_lr, w_lr = x_lr.shape

    tile_lr = tile_hr // scale
    overlap_lr = overlap_hr // scale
    starts_y = compute_starts(h_lr, tile_lr, overlap_lr)
    starts_x = compute_starts(w_lr, tile_lr, overlap_lr)

    H_hr, W_hr = h_lr * scale, w_lr * scale
    acc = torch.zeros((1, 3, H_hr, W_hr), device=device)
    acc_w = torch.zeros((1, 1, H_hr, W_hr), device=device)

    for y0 in starts_y:
        for x0 in starts_x:
            y1 = y0 + tile_lr
            x1 = x0 + tile_lr
            lr_tile = x_lr[:, :, y0:y1, x0:x1]  # (1,3,tile_lr,tile_lr)

            # Few-step Euler sampler → HR tile (1,3,tile_hr,tile_hr)
            sr_tile = euler_sampler(model, lr_tile, steps=steps, scale=scale)

            # Accumulate
            Y0, X0 = y0 * scale, x0 * scale
            Y1, X1 = Y0 + tile_hr, X0 + tile_hr
            acc[:, :, Y0:Y1, X0:X1] += sr_tile
            acc_w[:, :, Y0:Y1, X0:X1] += 1.0

    sr_full = acc / acc_w.clamp_min(1.0)
    return sr_full.squeeze(0).clamp(0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint .pt")
    ap.add_argument("--steps", type=int, default=None, help="Override number of Euler steps")
    ap.add_argument("--hr_dir", type=str, default=None,
                    help="Optional directory of HR images; if omitted, uses config.paths.valid_hr_dir")
    ap.add_argument("--limit", type=int, default=5,
                help="How many images to process (default 5). Ignored if --images is provided.")
    ap.add_argument("--images", type=str, default=None,
                    help="Comma-separated image names or filepaths to process (e.g., '0801,0805.png,C:\\img\\a.jpg').")
    
    args = ap.parse_args()

    config = load_config(args.config)
    device = get_device()

    steps = args.steps if args.steps is not None else int(config["fm"]["sampler_default_steps"])
    scale = int(config["data"]["scale"])
    tile_hr = int(config["inference"]["tile"])
    overlap_hr = int(config["inference"]["overlap"])

    # Inputs/outputs
    hr_root = Path(args.hr_dir) if args.hr_dir else Path(config["paths"]["valid_hr_dir"])
    assert hr_root.exists(), f"HR directory not found: {hr_root}"
    out_dir = Path(config["paths"]["exp_dir"]) / "samples"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Model
    ckpt_path = Path(args.ckpt)
    assert ckpt_path.exists(), f"Checkpoint not found: {ckpt_path}"
    model = load_model(config, ckpt_path, device)

    # Process images
    img_paths = list_images(hr_root)
    # Choose which images to process
    if args.images:
        tokens = [t.strip() for t in re.split(r"[;,]", args.images) if t.strip()]
        selected = []
        seen = set()
        for tok in tokens:
            cand = Path(tok)
            if cand.exists() and cand.is_file():
                key = cand.resolve()
                if key not in seen:
                    selected.append(cand)
                    seen.add(key)
                continue
            # match by name or stem within hr_root
            matches = [p for p in img_paths if p.name.lower() == tok.lower() or p.stem.lower() == tok.lower()]
            if matches:
                key = matches[0].resolve()
                if key not in seen:
                    selected.append(matches[0])
                    seen.add(key)
            else:
                print(f"[warn] No match for '{tok}' in {hr_root}. Skipping.")
        img_paths = selected
    else:
        img_paths = img_paths[: args.limit]

    print(f"[sample] {len(img_paths)} images | steps={steps} | tile={tile_hr}, overlap={overlap_hr}")


    for p in img_paths:
        # --- Load HR as tensor in [0,1] ---
        hr_pil = Image.open(p).convert("RGB")
        hr_t = TF.pil_to_tensor(hr_pil).float() / 255.0  # (3, H, W)

        # --- Make LR (bicubic + antialias), then SR (tiled) ---
        h_lr, w_lr = hr_t.shape[1] // scale, hr_t.shape[2] // scale
        lr_t = TF.resize(
            hr_t, [h_lr, w_lr],
            interpolation=InterpolationMode.BICUBIC,
            antialias=True
        )
        sr_t = sr_tiled(
            model, lr_t, steps=steps,
            tile_hr=tile_hr, overlap_hr=overlap_hr,
            scale=scale, device=device
        )

        # --- Bicubic upsample baseline back to HR size ---
        bicubic_t = F.interpolate(
            lr_t.unsqueeze(0), size=hr_t.shape[-2:], mode="bicubic", align_corners=False
        ).squeeze(0)

        # --- Save: bicubic, SR, and a 3-panel grid [bicubic | SR | HR] ---
        stem = p.stem
        grid = make_grid([bicubic_t.cpu(), sr_t.cpu(), hr_t.cpu()], nrow=3, padding=4)
        save_image(grid, out_dir / f"{stem}_grid_x{scale}_s{steps}.png")

        print(f"[write] {stem}: grid saved.")

    print("Done.")


if __name__ == "__main__":
    main()
