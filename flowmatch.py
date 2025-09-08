"""
Flow-Matching utilities (paths, targets, samplers) for x4 Super-Resolution.

We start with a simple **linear path** from Gaussian noise -> HR image:
    x_t = (1 - t) * z + t * x_hr        where z ~ N(0, I), t ~ U[0,1]
The target velocity is constant along the path:
    v*(x_t, t) = d/dt x_t = x_hr - z

Why start here?
- It's stable, easy to reason about, and works well with few-step Euler sampling.
- We can add a variance-preserving "diffusion-style" path later without changing the trainer.

API summary:
- sample_timesteps(batch) -> (B,) in [0,1]
- prepare_linear_path(x_hr, t, generator) -> x_t, v_target, z
- fm_training_targets(x_hr, t, path='linear', generator=None) -> x_t, v_target, z
- euler_sampler(model, x_lr, steps=8, scale=4, generator=None) -> x_hat (clamped to [0,1])
"""

from __future__ import annotations

from typing import Tuple, Optional

import torch
import torch.nn.functional as F


# --------------------------
# Timestep sampling
# --------------------------

def sample_timesteps(batch_size: int, device: torch.device, eps: float = 1e-5) -> torch.Tensor:
    """
    Sample t ~ Uniform(eps, 1-eps) to avoid degenerate ends.
    Returns shape (B,).
    """
    t = torch.rand(batch_size, device=device)
    if eps > 0:
        t = t.clamp(min=eps, max=1.0 - eps)
    return t


# --------------------------
# Paths & training targets
# --------------------------

def prepare_linear_path(
    x_hr: torch.Tensor,
    t: torch.Tensor,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Linear path from Gaussian noise -> data.

    Args:
        x_hr: (B, 3, H, W) HR target in [0,1]
        t:    (B,) in [0,1]
        generator: torch.Generator for deterministic noise, optional

    Returns:
        x_t:       (B, 3, H, W) state on the path
        v_target:  (B, 3, H, W) target velocity (constant x_hr - z)
        z_noise:   (B, 3, H, W) the noise used (helpful for debugging)
    """
    assert x_hr.dim() == 4, "x_hr must be (B,C,H,W)"
    device = x_hr.device
    if generator is None:
        z = torch.randn_like(x_hr, device=device)
    else:
        z = torch.randn(x_hr.shape, dtype=x_hr.dtype, device=device, generator=generator)

    # Broadcast t to image shape
    while t.dim() < x_hr.dim():
        t = t[..., None]
    x_t = (1.0 - t) * z + t * x_hr
    v_target = x_hr - z
    return x_t, v_target, z


def fm_training_targets(
    x_hr: torch.Tensor,
    t: torch.Tensor,
    path: str = "linear",
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build (x_t, v_target, z) for the requested path.
    Currently supported: 'linear' (noise -> data).

    Later extension idea (not implemented yet, by design for clarity):
    - 'vp': variance-preserving diffusion-style path x_t = alpha(t) x_hr + sigma(t) eps,
            with v_target = alpha'(t) x_hr + sigma'(t) eps.
    """
    path = path.lower()
    if path == "linear":
        return prepare_linear_path(x_hr, t, generator)
    raise ValueError(f"Unsupported fm path: {path!r}")


# --------------------------
# Few-step Euler sampler
# --------------------------

@torch.no_grad()
def euler_sampler(
    model,
    x_lr: torch.Tensor,
    steps: int = 8,
    scale: int = 4,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """
    Integrate dx/dt = v_theta(x_t, t, lr) with simple forward Euler from t=0 -> 1.

    Args:
        model: UNetSR-like module with forward(x_t, x_lr, t) -> v_hat
        x_lr:  (B, 3, h, w) low-resolution condition in [0,1]
        steps: number of Euler steps (1,2,4,8...). Default 8.
        scale: SR scale factor (x4 here).
        generator: for deterministic noise (optional)

    Returns:
        x_hat: (B, 3, H, W) in [0,1], where H = h*scale, W = w*scale
    """
    assert x_lr.dim() == 4, "x_lr must be (B,C,h,w)"
    device = x_lr.device
    batch, _, h, w = x_lr.shape
    H, W = h * scale, w * scale

    # Start from Gaussian noise at HR size
    x = torch.randn((batch, 3, H, W), device=device, generator=generator)

    # Uniform time grid;
    dt = 1.0 / float(steps)
    for i in range(steps):
        t_val = (i + 0.5) * dt
        t = torch.full((batch,), t_val, device=device)
        v_hat = model(x, x_lr, t)  # predict velocity
        x = x + dt * v_hat

    # Clamp to valid image range
    return x.clamp(0.0, 1.0)


# --------------------------
# Utility for resizing LR
# --------------------------

def downsample_hr_to_lr(x_hr: torch.Tensor, scale: int = 4) -> torch.Tensor:
    """
    Helper for tests/validation: create an LR tensor from an HR tensor using bicubic.
    Input/output in [0,1].
    """
    return F.interpolate(x_hr, scale_factor=1.0 / scale, mode="bicubic", align_corners=False)
