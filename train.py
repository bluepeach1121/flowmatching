"""
Training script for x4 Super-Resolution via Flow Matching.

- Data: on-the-fly LR from HR (crop-based) via data.py
- Model: UNetSR with LR concatenation (no FiLM/attention) — see model.py
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
from torch.amp import GradScaler  # type: ignore
#for some reason, Im getting an issue that torch.amp.GradScaler is outdated
from tqdm.auto import tqdm

from metrics import psnr, ssim
from data import make_dataloader
from model import UNetSR
from flowmatch import sample_timesteps, fm_training_targets, euler_sampler
from utils import (
    load_config,
    seed_everything,
    get_device,
    prepare_experiment_dir,
    human_readable_num,
)



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
        msd = model.state_dict()
        for k, v in self.shadow.items():
            if k in msd and msd[k].dtype.is_floating_point:
                msd[k].copy_(v)


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
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    log_fn(f"[ckpt] saved {path}")

@torch.no_grad()
def validate(
    model: nn.Module,
    ema: EMA,
    val_loader,
    device: torch.device,
    steps: int,
    scale: int,
    max_batches: int = 10,
) -> Dict[str, float]:
    # Swap in EMA weights for eval
    tmp = UNetSR()
    tmp.load_state_dict(model.state_dict(), strict=False)
    ema.copy_to(tmp)
    tmp.to(device).eval()

    psnrs, ssims = [], []
    it = 0
    for lr_patch, hr_patch in val_loader:
        lr_patch = lr_patch.to(device, non_blocking=True)
        hr_patch = hr_patch.to(device, non_blocking=True)

        sr_patch = euler_sampler(tmp, lr_patch, steps=steps, scale=scale)
        psnrs.append(float(psnr(sr_patch, hr_patch)))
        ssims.append(float(ssim(sr_patch, hr_patch)))

        it += 1
        if it >= max_batches:
            break

    return {
        "psnr": float(sum(psnrs) / max(1, len(psnrs))),
        "ssim": float(sum(ssims) / max(1, len(ssims))),
    }


# Train
def train(config: Dict[str, Any]) -> None:
    seed_everything(int(config["train"].get("seed", 42)))
    device = get_device()
    exp_dir = prepare_experiment_dir(config)

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

    model = UNetSR(
        in_channels=6,
        out_channels=3,
        base_channels=48,
        channel_multipliers=(1, 2, 4),
        num_blocks=int(config.get("model", {}).get("num_blocks", 2)),
        groups=int(config.get("model", {}).get("groups", 8)),
    ).to(device)

    base_lr = float(config["train"]["lr"])
    optimizer = optim.AdamW(model.parameters(), lr=base_lr)

    ema = EMA(model, decay=float(config["train"]["ema_decay"]))

    use_amp = bool(config["train"].get("amp", True))
    is_cuda = (device.type == "cuda")
    amp_ctx = torch.autocast(device_type="cuda", dtype=torch.float16) if (use_amp and is_cuda) else nullcontext()
    scaler = GradScaler(enabled=use_amp and is_cuda)

    loss_fn = nn.MSELoss()

    train_cfg = config.get("train", {})
    eval_cfg  = config.get("eval", {})
    data_cfg  = config.get("data", {})
    max_steps     = int(train_cfg.get("max_steps", train_cfg.get("steps", 10000)))
    val_every     = int(eval_cfg.get("every_steps", train_cfg.get("val_every", 50)))
    save_every    = int(train_cfg.get("save_every", val_every))
    grad_accum    = int(train_cfg.get("grad_accum", data_cfg.get("grad_accum", 1)))
    grad_clip_val = float(train_cfg.get("grad_clip", 0.0))
    sampler_steps = int(config.get("fm", {}).get("sampler_default_steps", 4))
    scale         = int(data_cfg["scale"])

    model.train()
    step = 0
    running_loss = 0.0
    train_iter = iter(train_loader)

    pbar = tqdm(total=max_steps, desc="train", dynamic_ncols=True)
    t0 = time.time()

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

        if grad_clip_val > 0.0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_val)

        scaler.step(optimizer)
        scaler.update()

        ema.update(model)

        step += 1
        pbar.update(1)

        if step % 50 == 0:
            avg_loss = running_loss / 50.0
            running_loss = 0.0
            pbar.set_postfix(loss=f"{avg_loss:.4f}", lr=f"{optimizer.param_groups[0]['lr']:.2e}")

        if step % val_every == 0:
            model.eval()
            metrics = validate(model, ema, val_loader, device, sampler_steps, scale)
            model.train()
            pbar.write(f"[val@{step}] PSNR {metrics['psnr']:.2f}  SSIM {metrics['ssim']:.4f}")

        if step % save_every == 0:
            save_checkpoint(exp_dir, step, model, ema, optimizer, scaler, log_fn=pbar.write)

    dt = time.time() - t0
    pbar.close()
    print(f"Done. Trained {human_readable_num(step)} steps in {dt/60.0:.1f} min.")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = load_config(args.config)
    train(cfg)
