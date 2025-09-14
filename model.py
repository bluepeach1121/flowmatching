import torch
import torch.nn as nn
import torch.nn.functional as F

def _block_channels(base: int, mult: int) -> int:
    return base * mult

class ResBlock(nn.Module):
    """
    the original design was to add an attention head to it but it became to complex
    No-norm residual block.
    - conv -> SiLU -> conv
    - residual scaling:
        * set res_scale=1.0 to disable
        * or set learnable_scale=True to use a learnable gamma initialized to 0
    - conv2 is zero-initialized so the block starts as identity.
    """
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        *,
        res_scale: float = 0.1,
        learnable_scale: bool = False,
    ):
        super().__init__()
        self.act = nn.SiLU(inplace=True)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, 1)

        self.learnable_scale = bool(learnable_scale)
        if self.learnable_scale:
            # starts at 0 -> identity at init, grows as needed
            self.gamma = nn.Parameter(torch.zeros(1))
        else:
            self.res_scale = float(res_scale)

        nn.init.zeros_(self.conv2.weight)
        if self.conv2.bias is not None:
            nn.init.zeros_(self.conv2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r = self.skip(x)
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)
        if self.learnable_scale:
            x = self.gamma * x
        else:
            x = self.res_scale * x
        return r + x


class Downsample(nn.Module):
    """Strided 3x3 conv (no checkerboard)."""
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    """Nearest x2 + 3x3 smoothing conv."""
    def __init__(self, channels: int):
        super().__init__()
        self.smooth = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.smooth(x)


class UNetSR(nn.Module):
    """
    Super-resolution UNet
    removed all normalization 
    """
    def __init__(
        self,
        in_channels: int = 6,
        out_channels: int = 3,
        base_channels: int = 48,
        channel_multipliers=(1, 2, 4),
        num_blocks: int = 2,
        groups: int | None = None,     # kept for config compatibility; unused
        res_scale: float = 0.1,        # set to 1.0 to disable scaling
        learnable_scale: bool = False, # set True to use gamma param (init 0)
        **_ignored,
    ):
        super().__init__()

        # Stem
        ch0 = _block_channels(base_channels, channel_multipliers[0])
        self.stem = nn.Conv2d(in_channels, ch0, kernel_size=3, padding=1)

        # Encoder
        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        self._enc_channels = []
        ch = ch0
        for i, mult in enumerate(channel_multipliers):
            out_ch = _block_channels(base_channels, mult)
            blocks = nn.Sequential(*[
                ResBlock(
                    ch if j == 0 else out_ch,
                    out_ch,
                    res_scale=res_scale,
                    learnable_scale=learnable_scale,
                )
                for j in range(num_blocks)
            ])
            self.down_blocks.append(blocks)
            ch = out_ch
            self._enc_channels.append(ch)
            self.downsamples.append(Downsample(ch) if i < len(channel_multipliers) - 1 else nn.Identity())

        # Bottleneck
        self.bot_block1 = ResBlock(ch, ch, res_scale=res_scale, learnable_scale=learnable_scale)
        self.bot_block2 = ResBlock(ch, ch, res_scale=res_scale, learnable_scale=learnable_scale)

        # Decoder
        self.up_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        for i in reversed(range(len(channel_multipliers))):
            skip_ch = self._enc_channels[i]
            in_ch = ch + skip_ch
            out_ch = skip_ch
            blocks = nn.Sequential(*[
                ResBlock(
                    in_ch if j == 0 else out_ch,
                    out_ch,
                    res_scale=res_scale,
                    learnable_scale=learnable_scale,
                )
                for j in range(num_blocks)
            ])
            self.up_blocks.append(blocks)
            ch = out_ch
            self.upsamples.append(Upsample(ch) if i > 0 else nn.Identity())

        self.head_act = nn.SiLU(inplace=True)
        self.head_conv = nn.Conv2d(ch, out_channels, kernel_size=3, padding=1)
        nn.init.zeros_(self.head_conv.weight)
        if self.head_conv.bias is not None:
            nn.init.zeros_(self.head_conv.bias)

    @staticmethod
    def _upsample_to(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, x_t: torch.Tensor, lr: torch.Tensor, t=None) -> torch.Tensor:
        """
        `t` is kept for backward-compatibility only. I wanted to experiment
        with time-conditioned multi-head attention, but the whole model became a bit too complex.
        The current network ignores `t`; supplying it does not affect outputs.
        its not referenced anywhere so theres no problem
        """
        lr_up = self._upsample_to(lr, x_t)
        x = torch.cat([x_t, lr_up], dim=1)
        x = self.stem(x)

        skips = []
        for blocks, down in zip(self.down_blocks, self.downsamples):
            x = blocks(x)
            skips.append(x)
            x = down(x)

        x = self.bot_block1(x)
        x = self.bot_block2(x)

        for blocks, up in zip(self.up_blocks, self.upsamples):
            skip = skips.pop()
            x = torch.cat([x, skip], dim=1)
            x = blocks(x)
            x = up(x)

        x = self.head_conv(self.head_act(x))
        return x
