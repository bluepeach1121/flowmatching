"""
Run super-resolution sampling with few-step Euler integration.

Usage:
  python sample.py --config config.yaml --ckpt runs/sr_x4/checkpoints/step_10000.pt
  python sample.py --config config.yaml --ckpt <ckpt> --steps 4
  # Optional: choose a different set of HR images to evaluate
  python sample.py --config config.yaml --ckpt <ckpt> --hr_dir "/path/to/some_HR_images"

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
    root = Path(root)
    out: List[Path] = []
    for ext in VALID_EXTS:
        out.extend(root.rglob(f"*{ext}"))
    return sorted(out)


def load_model(config, ckpt_path: Path, device: torch.device) -> UNetSR:
    """Instantiate UNetSR and load weights (EMA preferred if available)."""
    model = UNetSR(
        in_channels=6,
        out_channels=3,
        base_channels=48,
        channel_multipliers=(1, 2, 4),
        # New API (no FiLM/attention):
        num_blocks=int(config.get("model", {}).get("num_blocks", 2)),
        groups=int(config.get("model", {}).get("groups", 8)),
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict) and "ema" in ckpt and isinstance(ckpt["ema"], dict):
        # Merge EMA weights into model (keep missing buffers from current state)
        model.load_state_dict({**model.state_dict(), **ckpt["ema"]}, strict=False)
        print(f"[load] EMA weights loaded from {ckpt_path.name}")
    else:
        # Be tolerant if old checkpoints had extra keys
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        model.load_state_dict(state, strict=False)
        print(f"[load] model weights loaded from {ckpt_path.name} (strict=False)")

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
    lr_img: torch.Tensor,       # (3,h,w) in [0,1]
    steps: int,
    tile_hr: int,
    overlap_hr: int,
    scale: int,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    Few-step SR with sliding-window tiling to fit in memory.
    Returns a (3,H,W) tensor in [0,1].
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

    # Normalize overlaps
    sr = acc / acc_w.clamp_min(1e-8)
    return sr.squeeze(0).clamp(0.0, 1.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint .pt")
    parser.add_argument("--steps", type=int, default=None, help="Few-step sampler steps (overrides config)")
    parser.add_argument("--hr_dir", type=str, default=None, help="Override: folder of HR images to evaluate")
    parser.add_argument("--tile_hr", type=int, default=256, help="HR tile size for tiling inference")
    parser.add_argument("--overlap_hr", type=int, default=64, help="HR overlap between tiles")
    args = parser.parse_args()

    config = load_config(args.config)
    device = get_device()

    scale = int(config["data"]["scale"])
    steps = int(args.steps or config["fm"].get("sampler_default_steps", 4))

    # I/O
    exp_dir = Path(config["paths"]["exp_dir"])
    out_dir = exp_dir / "samples"
    out_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(config, Path(args.ckpt), device)
    model.eval()

    hr_root = Path(args.hr_dir) if args.hr_dir else Path(config["paths"]["valid_hr_dir"])
    img_paths = list_images(hr_root)
    if not img_paths:
        raise FileNotFoundError(f"No images found in {hr_root}")

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
            tile_hr=args.tile_hr, overlap_hr=args.overlap_hr,
            scale=scale, device=device
        )

        # --- Bicubic upsample baseline back to HR size ---
        bicubic_t = F.interpolate(
            lr_t.unsqueeze(0), size=hr_t.shape[-2:], mode="bicubic", align_corners=False
        ).squeeze(0).clamp(0, 1)

        # --- Save: bicubic, SR, and a 3-panel grid [bicubic | SR | HR] ---
        stem = p.stem
        grid = make_grid([bicubic_t.cpu(), sr_t.cpu(), hr_t.cpu()], nrow=3, padding=4)
        save_image(grid, out_dir / f"{stem}_grid_x{scale}_s{steps}.png")

        print(f"[write] {stem}: grid saved.")

    print("Done.")


if __name__ == "__main__":
    main()
