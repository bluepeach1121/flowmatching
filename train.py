"""
Training script for x4 Super-Resolution via Flow Matching.

- Data: on-the-fly LR from HR (crop-based) via data.py
- Model: UNetSR with LR concatenation + FiLM time conditioning (model.py)
- FM: linear noise->data path targets (flowmatch.py)
- AMP: torch.autocast(device_type="cuda", dtype=torch.float16) + torch.amp.GradScaler
- EMA: exponential moving average of weights
- Eval: quick PSNR/SSIM on a few validation batches

Run:
    python train.py --config config.yaml
"""

from __future__ import annotations

import argparse
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler # type: ignore
from tqdm.auto import tqdm

from metrics import psnr, ssim
from data import make_dataloader
from model import UNetSR
from flowmatch import sample_timesteps, fm_training_targets, euler_sampler
from utils import load_config, seed_everything, get_device, prepare_experiment_dir, human_readable_num


# -------------------------
# EMA
# -------------------------

class EMA:
    """Simple exponential moving average over floating-point parameters."""
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = float(decay)
        self.shadow = {
            k: v.detach().clone()
            for k, v in model.state_dict().items()
            if v.dtype.is_floating_point
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for k, v in model.state_dict().items():
            if k in self.shadow and v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)

    @torch.no_grad()
    def copy_to(self, model: nn.Module) -> None:
        for k, v in model.state_dict().items():
            if k in self.shadow and v.dtype.is_floating_point:
                v.copy_(self.shadow[k])


# -------------------------
# I/O helpers
# -------------------------

def save_checkpoint(
    exp_dir: Path,
    step: int,
    model: nn.Module,
    ema: EMA,
    optimizer: optim.Optimizer,
    scaler: GradScaler,
    log_fn=print,
) -> None:
    payload = {
        "step": step,
        "model": model.state_dict(),
        "ema": ema.shadow,
        "optim": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
    }
    path = exp_dir / "checkpoints" / f"step_{step}.pt"
    torch.save(payload, path)
    log_fn(f"[ckpt] saved {path}")


# -------------------------
# Validation
# -------------------------

@torch.no_grad()
def validate(
    model: nn.Module,
    ema: EMA,
    val_loader,
    device: torch.device,
    steps: int,
    max_batches: int = 4,
) -> Dict[str, float]:
    """
    Evaluate PSNR/SSIM using EMA weights and a few-step Euler sampler.
    To keep it fast, we only run on a handful of validation batches.
    """
    was_training = model.training
    model.eval()

    # Swap in EMA weights
    backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
    ema.copy_to(model)

    acc_psnr: list[float] = []
    acc_ssim: list[float] = []

    # Small, quiet bar for validation progress
    val_bar = tqdm(total=max_batches, desc="val", unit="batch", dynamic_ncols=True, leave=False)

    batches_done = 0
    for lr_tensor, hr_tensor in val_loader:
        lr_tensor = lr_tensor.to(device, non_blocking=True)
        hr_tensor = hr_tensor.to(device, non_blocking=True)

        sr_tensor = euler_sampler(model, lr_tensor, steps=steps, scale=4)
        acc_psnr.append(psnr(sr_tensor, hr_tensor))
        acc_ssim.append(ssim(sr_tensor, hr_tensor))

        batches_done += 1
        val_bar.update(1)
        if batches_done >= max_batches:
            break

    val_bar.close()

    # Restore original weights
    model.load_state_dict(backup)
    if was_training:
        model.train()

    out = {
        "psnr": float(sum(acc_psnr) / len(acc_psnr)) if acc_psnr else 0.0,
        "ssim": float(sum(acc_ssim) / len(acc_ssim)) if acc_ssim else 0.0,
    }
    return out


# -------------------------
# Training loop
# -------------------------

def train(config: Dict[str, Any]) -> None:
    device = get_device()
    seed_everything(1337)
    exp_dir = prepare_experiment_dir(config)

    # Data (train: random crops; val: crops for speed — full-image tiling comes in sample.py)
    train_loader = make_dataloader(
        hr_dir=config["paths"]["train_hr_dir"],
        scale=config["data"]["scale"],
        hr_crop=config["data"]["hr_crop"],
        batch_size=config["data"]["batch_size"],
        num_workers=config["data"]["num_workers"],
        augment=config["data"]["augment"],
        shuffle=True,
        persistent_workers=False,
    )
    val_loader = make_dataloader(
        hr_dir=config["paths"]["valid_hr_dir"],
        scale=config["data"]["scale"],
        hr_crop=config["data"]["hr_crop"],
        batch_size=config["data"]["batch_size"],
        num_workers=min(2, config["data"]["num_workers"]),
        augment={"hflip": False, "rot90": False},
        shuffle=False,
        persistent_workers=False,
    )

    # Model
    model = UNetSR(
        in_channels=6,
        out_channels=3,
        base_channels=48,
        channel_multipliers=(1, 2, 4),
        num_res_blocks=2,
        time_dim=256,
        use_bottleneck_attention=True,
        use_gradient_checkpointing=bool(config["train"].get("grad_checkpoint", True)),
    ).to(device)

    # Optimizer
    base_lr = float(config["train"]["lr"])
    optimizer = optim.AdamW(model.parameters(), lr=base_lr)

    # EMA
    ema = EMA(model, decay=float(config["train"]["ema_decay"]))

    # AMP
    use_amp = bool(config["train"].get("amp", True))
    is_cuda = (device.type == "cuda")
    scaler = GradScaler(device="cuda", enabled=(use_amp and is_cuda))
    amp_ctx = torch.autocast(device_type="cuda", dtype=torch.float16) if (use_amp and is_cuda) else nullcontext()

    # Schedule / logging
    max_steps = int(config["train"]["max_steps"])
    grad_accum = int(config["data"]["grad_accum"])
    log_every = int(config["train"].get("log_every", 200))
    eval_every = int(config["eval"].get("every_steps", 10000))
    sampler_steps_default = int(config["fm"]["sampler_default_steps"])
    grad_clip_val = float(config["train"].get("grad_clip", 0.0))

    # Loss
    loss_fn = nn.MSELoss()

    # Progress tracking
    step = 0
    running_loss = 0.0
    tick_time = time.time()

    # Single progress bar across steps
    pbar = tqdm(total=max_steps, initial=step, desc="train", unit="step", dynamic_ncols=True, smoothing=0.1)

    # Data iterator (recycled as needed)
    train_iter = iter(train_loader)

    model.train()
    while step < max_steps:
        optimizer.zero_grad(set_to_none=True)

        # Gradient accumulation to emulate larger effective batch
        for _ in range(grad_accum):
            try:
                lr_patch, hr_patch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                lr_patch, hr_patch = next(train_iter)

            lr_patch = lr_patch.to(device, non_blocking=True)
            hr_patch = hr_patch.to(device, non_blocking=True)

            # Flow Matching training targets (linear path: noise -> data)
            batch_size = hr_patch.size(0)
            t = sample_timesteps(batch_size, device=device)                     # (B,)
            x_t, v_target, _ = fm_training_targets(hr_patch, t, path="linear")  # (B,3,H,W) each

            with amp_ctx:
                v_pred = model(x_t, lr_patch, t)                                # predict velocity
                loss = loss_fn(v_pred, v_target) / grad_accum

            scaler.scale(loss).backward()
            running_loss += float(loss.item())

        # Gradient clipping (after unscale)
        if grad_clip_val > 0.0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_val)

        scaler.step(optimizer)
        scaler.update()
        ema.update(model)

        step += 1
        pbar.update(1)

        # Periodic logging
        if step % log_every == 0 or step == 1:
            elapsed = time.time() - tick_time
            it_per_sec = log_every / max(elapsed, 1e-6)
            avg_loss = running_loss / log_every
            pbar.set_postfix({"loss": f"{avg_loss:.6f}", "it/s": f"{it_per_sec:.2f}", "lr": f"{base_lr:g}"}, refresh=False)
            running_loss = 0.0
            tick_time = time.time()

        # Eval + checkpoint
        if step % eval_every == 0:
            metrics = validate(model, ema, val_loader, device, steps=sampler_steps_default, max_batches=4)
            pbar.write(f"[eval @ {step}] PSNR={metrics['psnr']:.3f} dB | SSIM={metrics['ssim']:.4f}")
            save_checkpoint(exp_dir, step, model, ema, optimizer, scaler, log_fn=pbar.write)

    # Final save
    save_checkpoint(exp_dir, step, model, ema, optimizer, scaler, log_fn=pbar.write)
    pbar.close()
    print("Training complete.")


# -------------------------
# CLI
# -------------------------

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = load_config(args.config)
    train(cfg)
