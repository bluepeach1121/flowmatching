"""
Data utilities for x4 Super-Resolution with Flow Matching.

- Reads HR images from a directory (train or valid set).
- For training: random HR crops of size `hr_crop` (default 192), on-the-fly bicubic x4 downsampling to make LR patches.
- Augmentations: horizontal flip, 90-degree rotations (optional).
- Returns (lr_tensor, hr_tensor) in [0, 1], shapes: (3, 48, 48) and (3, 192, 192) for scale=4.
"""

from pathlib import Path
from typing import List, Tuple, Optional, Dict

import random
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode


VALID_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def list_image_paths(root_dir: str) -> List[Path]:
    root = Path(root_dir)
    assert root.exists(), f"HR directory does not exist: {root}"
    paths = [p for p in sorted(root.iterdir()) if p.suffix.lower() in VALID_EXTENSIONS]
    assert len(paths) > 0, f"No images found in {root}"
    return paths


class SuperResolutionDataset(Dataset):
    """
    DIV2K-style HR-only dataset that creates LR on-the-fly by bicubic downsampling.
    """

    def __init__(
        self,
        hr_dir: str,
        scale: int = 4,
        hr_crop: int = 192,
        augment: Optional[Dict[str, bool]] = None,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.hr_paths = list_image_paths(hr_dir)
        self.scale = int(scale)
        self.hr_crop = int(hr_crop)
        self.lr_crop = self.hr_crop // self.scale
        self.augment = augment or {"hflip": True, "rot90": True}

        
        self.rng = random.Random(seed) if seed is not None else random.Random()

        assert self.hr_crop % self.scale == 0, (
            f"hr_crop ({self.hr_crop}) must be divisible by scale ({self.scale})."
        )

    def __len__(self) -> int:
        return len(self.hr_paths)

    def _open_rgb(self, path: Path) -> Image.Image:
        # convert to RGB explicitly.
        with Image.open(path) as img:
            return img.convert("RGB")

    def _random_hr_crop(self, img: Image.Image) -> Image.Image:
        """Random 192x192 crop (default) from the HR image."""
        width, height = img.size
        if width < self.hr_crop or height < self.hr_crop:
            new_w = max(width, self.hr_crop)
            new_h = max(height, self.hr_crop)
            img = img.resize((new_w, new_h), Image.Resampling.BICUBIC)

        x_max = width - self.hr_crop
        y_max = height - self.hr_crop
        left = self.rng.randint(0, x_max)
        top = self.rng.randint(0, y_max)
        return img.crop((left, top, left + self.hr_crop, top + self.hr_crop))

    def _apply_augmentations(self, img: Image.Image) -> Image.Image:
        """Optional H-flip and rotation by a random multiple of 90 degrees."""
        if self.augment.get("hflip", False) and self.rng.random() < 0.5:
            img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

        if self.augment.get("rot90", False):
            # Choose k in {0,1,2,3}; 0 means no rotation.
            k = self.rng.randint(0, 3)
            if k:
                # Rotate by 90 * k without resampling artifacts.
                for _ in range(k):
                    img = img.transpose(Image.Transpose.ROTATE_90)

        return img

    def _make_lr_from_hr(self, hr_img: Image.Image) -> torch.Tensor:
        hr_t = TF.pil_to_tensor(hr_img).float() / 255.0
        lr_t = TF.resize(
            hr_t,
            [self.lr_crop, self.lr_crop],
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        )
        return lr_t

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (lr_tensor, hr_tensor) in [0, 1].
        Shapes for scale=4 and hr_crop=192:
        lr_tensor: (3, 48, 48)
        hr_tensor: (3, 192, 192)
        """
        # 1) Load HR image as RGB (PIL)
        hr_img = self._open_rgb(self.hr_paths[index])

        # 2) Random HR crop, then augmentations (keeps size consistent)
        hr_img = self._random_hr_crop(hr_img)
        hr_img = self._apply_augmentations(hr_img)

        # 3) Create LR tensor from the augmented HR crop (bicubic + antialias)
        lr_tensor = self._make_lr_from_hr(hr_img)  # -> torch.Tensor, (3, 48, 48)

        # 4) Convert HR crop to tensor in [0, 1], channel-first
        hr_tensor = TF.pil_to_tensor(hr_img).float() / 255.0  # (3, 192, 192)

        return lr_tensor, hr_tensor



def make_dataloader(
    hr_dir: str,
    scale: int = 4,
    hr_crop: int = 192,
    batch_size: int = 8,
    num_workers: int = 2,
    augment: Optional[Dict[str, bool]] = None,
    seed: Optional[int] = None,
    shuffle: bool = True,
    persistent_workers: bool = False,
) -> DataLoader:
    """
    Build a PyTorch DataLoader for the SR dataset.

    Notes:
    - Keep `num_workers` small on systems with low free RAM.
    - `persistent_workers=False` is a bit safer on Windows when starting and stopping loaders frequently.
    """
    dataset = SuperResolutionDataset(
        hr_dir=hr_dir,
        scale=scale,
        hr_crop=hr_crop,
        augment=augment,
        seed=seed,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,           
        persistent_workers=persistent_workers,
    )
    return loader
