"""
config.py — Central configuration for the SAR-TopoGuard project.

All hyperparameters, paths, and toggles live here so that every module
imports a single source of truth.  Nothing in this file triggers
computation; it is pure data.
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple

__all__ = [
    "PROJECT_ROOT", "DATA_ROOT", "MANIFEST_PATH", "CHECKPOINT_DIR", "LOG_DIR",
    "get_device",
    "UNSEEN_ROIS", "STANDARD_TEST_ROIS", "VAL_FRACTION",
    "PATCH_SIZE", "SAR_CHANNELS", "OPTICAL_CHANNELS",
    "TrainConfig", "TRAIN_CFG",
    "EDGE_DENSITY_MANIFEST", "CANNY_LOW", "CANNY_HIGH",
    "VERIFIER_EPSILON", "VERIFIER_FLATNESS_GAIN", "VERIFIER_GM_CLAMP_MAX",
]

# ──────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT / "data" / "SEN12MS"
MANIFEST_PATH = PROJECT_ROOT / "manifests"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
LOG_DIR = PROJECT_ROOT / "logs"

# Create dirs on import (no side-effects beyond mkdir)
for _d in (DATA_ROOT, MANIFEST_PATH, CHECKPOINT_DIR, LOG_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# ──────────────────────────────────────────────────────────────────────
# Device Selection
# ──────────────────────────────────────────────────────────────────────
import torch

def get_device() -> torch.device:
    """Return the optimal device available (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ──────────────────────────────────────────────────────────────────────
# Zero-Shot Region Split
# ──────────────────────────────────────────────────────────────────────
# ROIs reserved EXCLUSIVELY for the unseen-region test set.
# These must NEVER appear in the training or validation loaders.
UNSEEN_ROIS: List[str] = [
    "ROI_1005",   # Africa — arid/semi-arid savanna
    "ROI_1049",   # South America — tropical forest
    "ROI_1012",   # Southeast Asia — dense urban + coast
]

# Standard held-out test ROIs (seen-region distribution, but not trained on)
STANDARD_TEST_ROIS: List[str] = [
    "ROI_1019",
    "ROI_1031",
]

# Everything else → training + validation (80/20 random scene split)
VAL_FRACTION: float = 0.15


# ──────────────────────────────────────────────────────────────────────
# Image / Tensor Geometry
# ──────────────────────────────────────────────────────────────────────
PATCH_SIZE: int = 256          # spatial H = W
SAR_CHANNELS: int = 2          # VV + VH polarisation bands
OPTICAL_CHANNELS: int = 3      # RGB (or first 3 Sentinel-2 bands)


# ──────────────────────────────────────────────────────────────────────
# RTX 3050 Memory Budget
# ──────────────────────────────────────────────────────────────────────
@dataclass
class TrainConfig:
    """Hyperparameters sized for ≤ 4 GB peak VRAM on RTX 3050."""

    # --- Optimiser --------------------------------------------------
    lr_g: float = 2e-4
    lr_d: float = 2e-4
    betas: Tuple[float, float] = (0.5, 0.999)

    # --- Batch & workers --------------------------------------------
    batch_size: int = 4          # safe for 4 GB; bump to 8 on 8 GB
    num_workers: int = 2         # Windows-safe; raise to 4 on Linux
    pin_memory: bool = True

    # --- Schedule ---------------------------------------------------
    epochs: int = 100
    save_every: int = 5          # checkpoint interval (epochs)
    log_every: int = 50          # log interval (iterations)

    # --- Loss weights -----------------------------------------------
    lambda_l1: float = 100.0     # pix2pix default
    lambda_edge: float = 10.0    # train-time edge loss weight
    use_edge_loss: bool = False  # False → Baseline 1; True → Baseline 2

    # --- Mixed precision --------------------------------------------
    use_amp: bool = True         # automatic mixed precision (AMP)

    # --- Misc -------------------------------------------------------
    seed: int = 42
    resume_from: str = ""        # path to checkpoint to resume from


# Singleton default config
TRAIN_CFG = TrainConfig()


# ──────────────────────────────────────────────────────────────────────
# Stratification
# ──────────────────────────────────────────────────────────────────────
EDGE_DENSITY_MANIFEST = MANIFEST_PATH / "stratified_manifest.json"
CANNY_LOW: int = 50
CANNY_HIGH: int = 150


# ──────────────────────────────────────────────────────────────────────
# Verifier
# ──────────────────────────────────────────────────────────────────────
VERIFIER_EPSILON: float = 1e-8
VERIFIER_FLATNESS_GAIN: float = 10.0
VERIFIER_GM_CLAMP_MAX: float = 0.3
