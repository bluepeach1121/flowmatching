"""
UNet backbone for x4 Super-Resolution via Flow Matching.

- Conditioning: LR image is upsampled to HR size and concatenated with the current HR state x_t.
- Time conditioning: sinusoidal t-embedding -> small MLP -> FiLM (scale/shift) inside ResBlocks.
- Attention: lightweight self-attention at the bottleneck only (configurable).
- Output: predicted velocity field v_hat with the same shape as x_t (3, H, W).

Forward signature:
    v_hat = model(x_t, lr, t)

where
    x_t: current HR state on the flow path (B, 3, H, W)
    lr:  LR patch (B, 3, H/4, W/4)  # scale=4; we upsample internally
    t:   continuous timestep in [0, 1], shape (B,) or (B, 1)
"""

from __future__ import annotations

from typing import Callable, List, Iterator, Iterable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as activation_checkpoint


# ---------- Time embedding ----------

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        # Use half sine, half cosine like diffusion embeddings.
        half = dim // 2
        # Frequencies follow 1 / 10^(4 * k / half) style (log-spaced).
        self.register_buffer(
            "freqs",
            torch.exp(torch.linspace(0, -4, half)),
            persistent=False,
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        t: (B,) or (B,1), assumed in [0, 1].
        returns: (B, dim)
        """
        if t.dim() == 1:
            t = t[:, None]
        # (B, half)
        angles = t * self.freqs[None, :]
        emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        if emb.shape[-1] < self.dim:  # in case dim is odd
            emb = F.pad(emb, (0, self.dim - emb.shape[-1]))
        return emb


class TimeMLP(nn.Module):
    """MLP that maps sinusoidal embedding -> FiLM vector per block."""
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        hidden = max(128, in_dim)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, t_embed: torch.Tensor) -> torch.Tensor:
        return self.net(t_embed)


# ---------- Blocks ----------

class FiLMResBlock(nn.Module):
    """
    Residual block with GroupNorm + SiLU and FiLM modulation from time embedding.

    If in_channels != out_channels, a 1x1 shortcut is applied.
    """
    def __init__(self, in_channels: int, out_channels: int, time_dim: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.norm1 = nn.GroupNorm(8, in_channels)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)

        # FiLM produces scale and shift (gamma, beta)
        self.time_to_film = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, 2 * out_channels),
        )

        self.norm2 = nn.GroupNorm(8, out_channels)
        self.act2 = nn.SiLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        self.shortcut = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, t_vec: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, H, W)
        t_vec: (B, time_dim) from TimeMLP
        """
        h = self.conv1(self.act1(self.norm1(x)))
        # FiLM modulation
        gamma, beta = self.time_to_film(t_vec).chunk(2, dim=-1)  # (B, C), (B, C)
        gamma = gamma[..., None, None]
        beta = beta[..., None, None]

        h = self.norm2(h)
        h = h * (1 + gamma) + beta
        h = self.conv2(self.act2(h))
        return h + self.shortcut(x)


class SelfAttention2d(nn.Module):
    """Lightweight 2D self-attention with GroupNorm, for bottleneck use."""
    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        assert channels % num_heads == 0, "channels must be divisible by num_heads"
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
        self.num_heads = num_heads

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        hdim = c // self.num_heads
        y = self.norm(x)
        qkv = self.qkv(y)  # (B, 3C, H, W)
        q, k, v = qkv.chunk(3, dim=1)

        # (B, heads, d, HW)
        q = q.view(b, self.num_heads, hdim, h * w)
        k = k.view(b, self.num_heads, hdim, h * w)
        v = v.view(b, self.num_heads, hdim, h * w)

        attn = torch.softmax((q.transpose(2, 3) @ k) / (hdim ** 0.5), dim=-1)  # (B, heads, HW, HW)
        out = (attn @ v.transpose(2, 3)).transpose(2, 3)  # (B, heads, d, HW)
        out = out.contiguous().view(b, c, h, w)
        return x + self.proj(out)


# ---------- UNet ----------

class UNetSR(nn.Module):
    """
    A compact UNet with LR concatenation and FiLM time conditioning.
    Tuned to be memory-friendly; increase base_channels or add levels if you have more VRAM.
    """
    def __init__(
        self,
        in_channels: int = 6,              # 3 for x_t + 3 for LR (upsampled)
        out_channels: int = 3,
        base_channels: int = 48,
        channel_multipliers: Tuple[int, ...] = (1, 2, 4),  # 3 levels
        num_res_blocks: int = 2,
        time_dim: int = 256,
        use_bottleneck_attention: bool = True,
        use_gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.use_gradient_checkpointing = use_gradient_checkpointing

        # Time embedding
        self.time_embedder = SinusoidalTimeEmbedding(time_dim)
        self.time_mlp = TimeMLP(time_dim, time_dim)

        # Initial conv
        self.in_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        # Down path
        self.down_blocks = nn.ModuleList()
        ch = base_channels
        skips = []
        for mult in channel_multipliers:
            out_ch = base_channels * mult
            block = nn.ModuleList(
                [FiLMResBlock(ch if i == 0 else out_ch, out_ch, time_dim) for i in range(num_res_blocks)]
            )
            self.down_blocks.append(block)
            skips.append(out_ch)
            ch = out_ch
            if mult != channel_multipliers[-1]:
                self.down_blocks.append(nn.Conv2d(ch, ch, 3, stride=2, padding=1))  # downsample

        # Bottleneck
        self.mid1 = FiLMResBlock(ch, ch, time_dim)
        self.mid_attn = SelfAttention2d(ch) if use_bottleneck_attention else nn.Identity()
        self.mid2 = FiLMResBlock(ch, ch, time_dim)

        # Up path
        self.up_blocks = nn.ModuleList()
        for mult in reversed(channel_multipliers):
            out_ch = base_channels * mult
            # Each up level sees concatenated skip features -> double channels
            block = nn.ModuleList(
                [FiLMResBlock(ch + skips.pop(), out_ch, time_dim)]
                + [FiLMResBlock(out_ch, out_ch, time_dim) for _ in range(num_res_blocks - 1)]
            )
            self.up_blocks.append(block)
            ch = out_ch
            if mult != channel_multipliers[0]:
                self.up_blocks.append(nn.Upsample(scale_factor=2, mode="nearest"))
                self.up_blocks.append(nn.Conv2d(ch, ch, 3, padding=1))  # smooth after upsample

        # Final conv
        self.out_norm = nn.GroupNorm(8, ch)
        self.out_act = nn.SiLU()
        self.out_conv = nn.Conv2d(ch, out_channels, 3, padding=1)

    # ---- helpers ----

    def _apply_block(self, block: nn.Module, x: torch.Tensor, t_vec: torch.Tensor) -> torch.Tensor:
        if isinstance(block, FiLMResBlock):
            return block(x, t_vec)
        return block(x)

    def _run_with_ckpt(
        self,
        block: nn.Module,
        x: torch.Tensor,
        t_vec: torch.Tensor,
    ) -> torch.Tensor:
        """Run a block with optional activation checkpointing (tensors-only interface)."""
        def run_block(x_in: torch.Tensor, t_in: torch.Tensor) -> torch.Tensor:
            return self._apply_block(block, x_in, t_in)

        if self.use_gradient_checkpointing and self.training:
            try:
                return activation_checkpoint(run_block, x, t_vec, use_reentrant=False)  # type: ignore[misc]
            except TypeError:
                return activation_checkpoint(run_block, x, t_vec)  # type: ignore[misc]
        return run_block(x, t_vec)


    # ---- forward ----

    def forward(self, x_t: torch.Tensor, lr: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        x_t: (B, 3, H, W)
        lr:  (B, 3, H/4, W/4)  # scale=4
        t:   (B,) in [0, 1]
        """
        # Upsample LR to HR size and concatenate with x_t
        lr_up = F.interpolate(lr, size=x_t.shape[-2:], mode="bicubic", align_corners=False)
        x = torch.cat([x_t, lr_up], dim=1)  # (B, 6, H, W)

        # Time embedding
        t_embed = self.time_mlp(self.time_embedder(t))  # (B, time_dim)

        # In
        h = self.in_conv(x)

        # Down path with skips
        skip_feats: List[torch.Tensor] = [] 
        idx = 0
        while idx < len(self.down_blocks):
            block_or_down = self.down_blocks[idx]
            if isinstance(block_or_down, nn.ModuleList):
                for rb in block_or_down:
                    h = self._run_with_ckpt(rb, h, t_embed)
                # save ONE skip feature for this resolution
                skip_feats.append(h)
                idx += 1
            else:
                # downsample to the next resolution
                h = block_or_down(h)
                idx += 1

        # Bottleneck
        h = self._run_with_ckpt(self.mid1, h, t_embed)
        h = self.mid_attn(h)
        h = self._run_with_ckpt(self.mid2, h, t_embed)

        # Up path (consume skips in reverse)
        skip_iter: Iterator[torch.Tensor] = iter(reversed(skip_feats))
        idx = 0
        while idx < len(self.up_blocks):
            block = self.up_blocks[idx]
            if isinstance(block, nn.ModuleList):
                skip = next(skip_iter)           
                h = torch.cat([h, skip], dim=1)  
                for rb in block:
                    h = self._run_with_ckpt(rb, h, t_embed)
                idx += 1
            else:
                h = block(h)  # upsample or conv
                idx += 1


        # Out
        h = self.out_act(self.out_norm(h))
        v_hat = self.out_conv(h)
        return v_hat
