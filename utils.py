"""
Utility helpers shared across the project.

- load_config: read YAML config into a plain dict (and echo important bits).
- seed_everything: set seeds for reproducibility.
- get_device: choose "cuda" if available and print GPU name.
- prepare_experiment_dir: create {exp_dir, checkpoints, samples, logs} and
  save a copy of the config there for traceability.

"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Dict, Any

import yaml
import numpy as np
import torch


def load_config(config_path: str | os.PathLike) -> Dict[str, Any]:
    """
    Load a YAML config file into a dictionary.
    """
    config_path = Path(config_path)
    assert config_path.exists(), f"Config not found: {config_path}"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    assert "paths" in config and "data" in config, "Config must have 'paths' and 'data' sections."
    scale = int(config["data"].get("scale", 4))
    hr_crop = int(config["data"].get("hr_crop", 192))
    assert hr_crop % scale == 0, f"hr_crop ({hr_crop}) must be divisible by scale ({scale})."

    # Derived: lr_crop for convenience
    config["data"]["lr_crop"] = hr_crop // scale
    return config


def seed_everything(seed: int = 1337, deterministic: bool = False) -> None:
    """
    Set seeds for Python, NumPy and PyTorch.

    deterministic=True can slow things down; leave False for everyday training.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False
    else:
        # This lets cuDNN pick fast kernels for your input sizes.
        torch.backends.cudnn.benchmark = True


def get_device() -> torch.device:
    """
    Return a torch.device, preferring CUDA if available.
    Prints a short description to help debugging.
    """
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"[device] cuda:0 -> {gpu_name}")
        return torch.device("cuda", 0)
    print("[device] cpu")
    return torch.device("cpu")


def prepare_experiment_dir(config: Dict[str, Any]) -> Path:
    """
    Create the experiment directory and subfolders, and save a copy of the config.
    Returns the path to the experiment directory.
    """
    exp_dir = Path(config["paths"]["exp_dir"]).resolve()
    (exp_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (exp_dir / "samples").mkdir(parents=True, exist_ok=True)
    (exp_dir / "logs").mkdir(parents=True, exist_ok=True)

    # Save an immutable copy of the config used for this run
    with open(exp_dir / "config.used.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    print(f"[exp] using {exp_dir}")
    return exp_dir


def human_readable_num(num: int) -> str:
    """Format integers like 1_500_000 -> '1.5M' for prettier logs."""
    if num >= 1_000_000:
        return f"{num/1_000_000:.1f}M"
    if num >= 1_000:
        return f"{num/1_000:.1f}k"
    return str(num)
