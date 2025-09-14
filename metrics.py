"""
We keep these fast and dependency-light:
- PSNR on [0,1] RGB
- SSIM (lightweight approximation), averaged over RGB

All functions accept tensors in shape (B,3,H,W) and return Python floats.
"""

from __future__ import annotations

import torch


def psnr(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> float:
    x = pred.clamp(0, 1)
    y = target.clamp(0, 1)
    mse = torch.mean((x - y) ** 2, dim=(1, 2, 3))
    val = -10.0 * torch.log10(mse + eps)
    return float(val.mean().item())


def _ssim_per_channel(x: torch.Tensor, y: torch.Tensor, c1: float, c2: float) -> torch.Tensor:
    mu_x = torch.mean(x, dim=(2, 3), keepdim=True)
    mu_y = torch.mean(y, dim=(2, 3), keepdim=True)
    var_x = torch.var(x, dim=(2, 3), unbiased=False, keepdim=True)
    var_y = torch.var(y, dim=(2, 3), unbiased=False, keepdim=True)
    cov_xy = torch.mean((x - mu_x) * (y - mu_y), dim=(2, 3), keepdim=True)
    ssim_map = ((2 * mu_x * mu_y + c1) * (2 * cov_xy + c2)) / ((mu_x ** 2 + mu_y ** 2 + c1) * (var_x + var_y + c2))
    return ssim_map.mean(dim=(2, 3))


def ssim(pred: torch.Tensor, target: torch.Tensor) -> float:
    x = pred.clamp(0, 1)
    y = target.clamp(0, 1)
    c1 = (0.01 ** 2)
    c2 = (0.03 ** 2)
    vals = []
    for ch in range(x.shape[1]):
        vals.append(_ssim_per_channel(x[:, ch:ch+1], y[:, ch:ch+1], c1, c2))
    ssim_batch = torch.cat(vals, dim=1).mean(dim=1)
    return float(ssim_batch.mean().item())
