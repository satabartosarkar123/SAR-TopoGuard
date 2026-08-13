#!/usr/bin/env python3
"""
Module 1 — SEN1-2 Data Ingestion & Zero-Shot Splitting
=======================================================

Responsibilities
----------------
1.  Parse the standard SEN1-2 / SEN12MS directory layout::

        <DATA_ROOT>/
            ROI_XXXX/
                s1_XX/            ← SAR (Sentinel-1)  GeoTIFF patches
                s2_XX/            ← Optical (Sentinel-2) GeoTIFF patches

2.  Load 256×256 patches, normalise SAR (dB → [0,1]) and Optical ([0,1]).
3.  Enforce the **Zero-Shot Unseen-Region** split:
    • Training set  — all ROIs *except* those in UNSEEN_ROIS & STANDARD_TEST_ROIS
    • Standard test — ROIs in STANDARD_TEST_ROIS
    • Unseen test   — ROIs in UNSEEN_ROIS
4.  Provide a download helper for the (very large) SEN1-2 archive.

Author : SAR-TopoGuard team
Date   : Day 1 — Morning
"""

from __future__ import annotations

import logging
import json
import random
import subprocess
import sys
import tarfile
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# ── Local imports ──────────────────────────────────────────────────────
from config import (
    DATA_ROOT,
    MANIFEST_PATH,
    PATCH_SIZE,
    SAR_CHANNELS,
    OPTICAL_CHANNELS,
    UNSEEN_ROIS,
    STANDARD_TEST_ROIS,
    VAL_FRACTION,
    TRAIN_CFG,
)

logger = logging.getLogger(__name__)

# We try importing rasterio/tifffile for GeoTIFF; fall back to PIL for
# standard image patches.
try:
    import rasterio                         # preferred for GeoTIFFs
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False
    logger.info(
        "rasterio not installed — falling back to tifffile / PIL. "
        "Install via `pip install rasterio` for full GeoTIFF support."
    )

try:
    import tifffile                         # lightweight TIFF reader
    HAS_TIFFFILE = True
except ImportError:
    HAS_TIFFFILE = False

from PIL import Image


# ======================================================================
# 1.  Download helper
# ======================================================================

# Official SEN12MS hosting (MediaTUM, TU Munich)
SEN12MS_URLS: List[str] = [
    # The dataset is ~500 GB split into multiple tar archives.
    # Add actual URLs here once the user has access credentials.
    # Example placeholder entries:
    "https://dataserv.ub.tum.de/s/m1474000/download?path=/&files=ROI_1005.tar.gz",
    "https://dataserv.ub.tum.de/s/m1474000/download?path=/&files=ROI_1012.tar.gz",
    "https://dataserv.ub.tum.de/s/m1474000/download?path=/&files=ROI_1019.tar.gz",
    "https://dataserv.ub.tum.de/s/m1474000/download?path=/&files=ROI_1031.tar.gz",
    "https://dataserv.ub.tum.de/s/m1474000/download?path=/&files=ROI_1049.tar.gz",
]


def _safe_extract_tar(tar_path: Path, dest_dir: Path) -> None:
    """
    Extract a tar archive safely, filtering out path-traversal attacks.

    Python 3.12+ supports ``filter='data'`` natively.  For older
    versions we fall back to manual member filtering that rejects
    absolute paths and ``..`` components.
    """
    with tarfile.open(tar_path, "r:*") as tf:
        if sys.version_info >= (3, 12):
            # Python 3.12+ safe extraction filter
            tf.extractall(path=dest_dir, filter="data")
        else:
            # Manual safe extraction for Python < 3.12
            safe_members = []
            for member in tf.getmembers():
                # Reject absolute paths and parent-directory traversal
                if member.name.startswith("/") or ".." in member.name.split("/"):
                    logger.warning(
                        "Skipping potentially unsafe tar member: %s", member.name
                    )
                    continue
                safe_members.append(member)
            tf.extractall(path=dest_dir, members=safe_members)


def download_sen12ms(
    dest_dir: Path = DATA_ROOT,
    urls: Optional[List[str]] = None,
) -> None:
    """
    Download SEN12MS archives with wget (resume-capable via ``-c``).

    Parameters
    ----------
    dest_dir : Path
        Where to save and extract archives.
    urls : list[str], optional
        Override the default URL list (useful for partial downloads).
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    urls = urls or SEN12MS_URLS

    for url in urls:
        fname = url.split("files=")[-1] if "files=" in url else url.split("/")[-1]
        target = dest_dir / fname
        logger.info("Downloading %s → %s", fname, target)

        # wget with resume (-c) and retry (--tries)
        cmd = [
            "wget", "-c",
            "--tries=5",
            "--timeout=120",
            "-O", str(target),
            url,
        ]
        try:
            subprocess.run(cmd, check=True)
        except FileNotFoundError:
            logger.error(
                "wget not found. On Windows, install via "
                "`winget install GnuWin32.Wget` or use WSL."
            )
            raise
        except subprocess.CalledProcessError as exc:
            logger.error("Download failed for %s: %s", fname, exc)
            continue

        # Extract if tarball (using safe extraction)
        if target.suffix in (".gz", ".tgz", ".tar"):
            logger.info("Extracting %s …", target.name)
            _safe_extract_tar(target, dest_dir)
            logger.info("Done extracting %s", target.name)


# ======================================================================
# 2.  Directory parsing
# ======================================================================

def _discover_patch_pairs(
    root: Path,
) -> List[Dict[str, Any]]:
    """
    Walk the SEN12MS directory tree and pair SAR ↔ Optical patches.

    Expected layout (per the official SEN12MS structure)::

        <root>/
            ROI_XXXX/
                s1_XX/          ← Sentinel-1 (SAR)  .tif patches
                    ROI_XXXX_s1_XX_pNN.tif
                s2_XX/          ← Sentinel-2 (Optical) .tif patches
                    ROI_XXXX_s2_XX_pNN.tif

    Returns a list of dicts:
        { "roi": str, "season": str, "patch_id": str,
          "sar_path": Path, "optical_path": Path }
    """
    pairs: List[Dict[str, Any]] = []
    if not root.exists():
        logger.warning("Data root %s does not exist.", root)
        return pairs

    for roi_dir in sorted(root.iterdir()):
        if not roi_dir.is_dir() or not roi_dir.name.startswith("ROI"):
            continue
        roi_name = roi_dir.name  # e.g. "ROI_1005"

        # Collect season sub-directories for s1 and s2
        s1_dirs = sorted(roi_dir.glob("s1_*"))
        s2_dirs = sorted(roi_dir.glob("s2_*"))

        # Map season tag → directory
        s1_map = {d.name.replace("s1_", ""): d for d in s1_dirs}
        s2_map = {d.name.replace("s2_", ""): d for d in s2_dirs}

        common_seasons = sorted(set(s1_map.keys()) & set(s2_map.keys()))
        if not common_seasons:
            # Alternative flat layout: tifs directly under ROI dir
            sar_files = sorted([f for f in roi_dir.glob("*_s1_*.*") if f.suffix.lower() in [".tif", ".png"]])
            opt_files = sorted([f for f in roi_dir.glob("*_s2_*.*") if f.suffix.lower() in [".tif", ".png"]])
            sar_by_pid = {f.stem.split("_p")[-1]: f for f in sar_files}
            opt_by_pid = {f.stem.split("_p")[-1]: f for f in opt_files}
            for pid in sorted(set(sar_by_pid) & set(opt_by_pid)):
                pairs.append({
                    "roi": roi_name,
                    "season": "unknown",
                    "patch_id": pid,
                    "sar_path": sar_by_pid[pid],
                    "optical_path": opt_by_pid[pid],
                })
            continue

        for season in common_seasons:
            sar_files = sorted([f for f in s1_map[season].glob("*.*") if f.suffix.lower() in [".tif", ".png"]])
            opt_files = sorted([f for f in s2_map[season].glob("*.*") if f.suffix.lower() in [".tif", ".png"]])

            # Build a mapping from patch number to file
            sar_by_pid: Dict[str, Path] = {}
            for f in sar_files:
                # Extract patch id: everything after the last "_p"
                parts = f.stem.rsplit("_p", maxsplit=1)
                pid = parts[-1] if len(parts) == 2 else f.stem
                sar_by_pid[pid] = f

            opt_by_pid: Dict[str, Path] = {}
            for f in opt_files:
                parts = f.stem.rsplit("_p", maxsplit=1)
                pid = parts[-1] if len(parts) == 2 else f.stem
                opt_by_pid[pid] = f

            matched_pids = sorted(set(sar_by_pid.keys()) & set(opt_by_pid.keys()))
            for pid in matched_pids:
                pairs.append({
                    "roi": roi_name,
                    "season": season,
                    "patch_id": pid,
                    "sar_path": sar_by_pid[pid],
                    "optical_path": opt_by_pid[pid],
                })

    logger.info("Discovered %d SAR↔Optical patch pairs under %s", len(pairs), root)
    return pairs


# ======================================================================
# 3.  Splitting logic
# ======================================================================

SplitName = Literal["train", "val", "test_standard", "test_unseen"]


def assign_splits(
    pairs: List[Dict[str, Any]],
    val_fraction: float = VAL_FRACTION,
    seed: int = TRAIN_CFG.seed,
) -> Dict[SplitName, List[Dict[str, Any]]]:
    """
    Partition patch pairs into train / val / test_standard / test_unseen.

    Zero-Shot Rule
    ~~~~~~~~~~~~~~
    • Any pair whose ROI ∈ UNSEEN_ROIS  → ``test_unseen``  (NEVER trained on)
    • Any pair whose ROI ∈ STANDARD_TEST_ROIS → ``test_standard``
    • Remaining pairs → shuffled, then split into ``train`` / ``val``
      by a scene-level (ROI+season) stratified random split.
    """
    splits: Dict[SplitName, List[Dict[str, Any]]] = {
        "train": [],
        "val": [],
        "test_standard": [],
        "test_unseen": [],
    }

    trainval_pairs: List[Dict[str, Any]] = []

    for p in pairs:
        roi = p["roi"]
        if roi in UNSEEN_ROIS:
            splits["test_unseen"].append(p)
        elif roi in STANDARD_TEST_ROIS:
            splits["test_standard"].append(p)
        else:
            trainval_pairs.append(p)

    # Scene-level split: group by (ROI, season) to prevent data leakage
    scene_keys: Dict[str, List[Dict[str, Any]]] = {}
    for p in trainval_pairs:
        key = f"{p['roi']}_{p['season']}"
        scene_keys.setdefault(key, []).append(p)

    all_scenes = sorted(scene_keys.keys())
    rng = random.Random(seed)
    rng.shuffle(all_scenes)

    n_val = max(1, int(len(all_scenes) * val_fraction))
    val_scenes = set(all_scenes[:n_val])

    for scene, scene_pairs in scene_keys.items():
        target = "val" if scene in val_scenes else "train"
        splits[target].extend(scene_pairs)

    for name, items in splits.items():
        logger.info("Split %-15s : %6d pairs", name, len(items))

    return splits


def save_split_manifest(
    splits: Dict[SplitName, List[Dict[str, Any]]],
    path: Path = MANIFEST_PATH / "split_manifest.json",
) -> None:
    """Persist the split assignments to a JSON manifest."""
    serialisable = {}
    for split_name, items in splits.items():
        serialisable[split_name] = [
            {
                "roi": it["roi"],
                "season": it["season"],
                "patch_id": it["patch_id"],
                "sar_path": str(it["sar_path"]),
                "optical_path": str(it["optical_path"]),
            }
            for it in items
        ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(serialisable, f, indent=2)
    logger.info("Saved split manifest → %s", path)


# ======================================================================
# 4.  Image I/O helpers
# ======================================================================

def _load_tiff(path: Path, channels: int) -> np.ndarray:
    """
    Load a GeoTIFF (or plain TIFF) as a float32 numpy array of shape
    ``(channels, H, W)``.

    Handles:
    • rasterio   — full GeoTIFF support (CRS, transforms, etc.)
    • tifffile   — lightweight fallback
    • PIL/Pillow — last resort (RGB only, no 16-bit SAR)
    """
    if str(path).lower().endswith('.png') or str(path).lower().endswith('.jpg'):
        pil_img = Image.open(path)
        img = np.array(pil_img, dtype=np.float32)
        if img.ndim == 2:
            img = img[np.newaxis, ...]
        elif img.ndim == 3:
            img = np.transpose(img, (2, 0, 1))
    elif HAS_RASTERIO:
        with rasterio.open(path) as src:
            img = src.read()  # (bands, H, W), dtype varies
    elif HAS_TIFFFILE:
        img = tifffile.imread(str(path))  # (H, W) or (H, W, C) or (C, H, W)
        if img.ndim == 2:
            img = img[np.newaxis, ...]     # (1, H, W)
        elif img.ndim == 3 and img.shape[-1] <= 13:
            # Likely (H, W, C) layout → transpose to (C, H, W)
            img = np.transpose(img, (2, 0, 1))
    else:
        pil_img = Image.open(path)
        img = np.array(pil_img, dtype=np.float32)
        if img.ndim == 2:
            img = img[np.newaxis, ...]
        elif img.ndim == 3:
            img = np.transpose(img, (2, 0, 1))

    img = img.astype(np.float32)

    # Select the requested number of channels (first N)
    if img.shape[0] > channels:
        img = img[:channels]
    elif img.shape[0] < channels:
        # Repeat last channel to fill (edge case with single-band files)
        pad = np.repeat(img[-1:], channels - img.shape[0], axis=0)
        img = np.concatenate([img, pad], axis=0)

    return img  # (channels, H, W)


def _center_crop(arr: np.ndarray, size: int) -> np.ndarray:
    """Centre-crop spatial dims to ``(C, size, size)``."""
    _, h, w = arr.shape
    if h < size or w < size:
        # Pad with zeros if smaller than target
        pad_h = max(0, size - h)
        pad_w = max(0, size - w)
        arr = np.pad(
            arr,
            ((0, 0), (pad_h // 2, pad_h - pad_h // 2),
             (pad_w // 2, pad_w - pad_w // 2)),
            mode="reflect",
        )
        _, h, w = arr.shape
    top = (h - size) // 2
    left = (w - size) // 2
    return arr[:, top : top + size, left : left + size]


# ======================================================================
# 5.  Normalisation  (Percentile-based, outputs [-1, 1] for Tanh)
# ======================================================================

def normalise_sar(img: np.ndarray) -> np.ndarray:
    """
    Normalise SAR intensities to [-1, 1].

    SEN1-2 SAR patches may arrive as:
      - **dB scale** (typical range  -25 ... +5 dB)
      - **Linear intensity** (0 ... large)
      - **uint8 PNG** (0 ... 255) — mini_sen12_data pre-processed patches

    For uint8/PNG data, we use percentile stretching.
    For dB data (median < 0), we clip to [-25, 0] and linearly scale.
    For linear power data (large values), we convert to dB first.

    Final output: [-1, 1] float32, shape (C, H, W).
    """
    img = img.astype(np.float32)

    if img.max() <= 1.0 + 1e-6 and img.min() >= -1.0 - 1e-6:
        # Already normalised (unlikely but safe guard)
        return img.astype(np.float32)

    if np.median(img) < 0:
        # dB scale
        img = np.clip(img, -25.0, 0.0)
        img = (img + 25.0) / 25.0  # -> [0, 1]
    elif img.max() > 300.0:
        # Linear power scale -> convert to dB first
        img = 10.0 * np.log10(img + 1e-6)
        img = np.clip(img, -25.0, 0.0)
        img = (img + 25.0) / 25.0  # -> [0, 1]
    else:
        # uint8 or already [0, 255] range: percentile stretch
        p2 = np.percentile(img, 2)
        p98 = np.percentile(img, 98)
        if (p98 - p2) > 1e-6:
            img = (img - p2) / (p98 - p2)
        else:
            img = np.zeros_like(img)
        img = np.clip(img, 0.0, 1.0)

    # Map [0, 1] -> [-1, 1]
    img = (img * 2.0) - 1.0
    return img.astype(np.float32)


def normalise_optical(img: np.ndarray) -> np.ndarray:
    """
    Normalise Sentinel-2 optical reflectance to [-1, 1].

    Handles all common formats:
      - **uint16 surface reflectance x 10000** (SEN12MS GeoTIFF)
      - **uint8 [0, 255]** (pre-processed PNGs in mini_sen12_data)
      - **float [0, 1]** (already normalised)

    Uses robust 2nd/98th percentile stretching to handle outliers
    (clouds, shadows, sensor saturation).

    Final output: [-1, 1] float32, shape (C, H, W).
    """
    img = img.astype(np.float32)

    # Percentile-based contrast stretch (per-image, all bands jointly)
    p2 = np.percentile(img, 2)
    p98 = np.percentile(img, 98)

    if (p98 - p2) > 1e-6:
        img = (img - p2) / (p98 - p2)
    else:
        # Flat image (rare) — map to mid-gray
        img = np.zeros_like(img)

    # Clip to [0, 1]
    img = np.clip(img, 0.0, 1.0)

    # Map [0, 1] -> [-1, 1] for Tanh generator output
    img = (img * 2.0) - 1.0
    return img.astype(np.float32)


def denormalize_for_display(img: np.ndarray) -> np.ndarray:
    """
    Inverse of the [-1, 1] normalisation. Used ONLY for saving images
    or computing metrics — NOT during training.

    Input:  [-1, 1]
    Output: [0, 1]
    """
    img = (img + 1.0) / 2.0
    return np.clip(img, 0.0, 1.0)


# ======================================================================
# 6.  PyTorch Dataset
# ======================================================================

def _worker_init_fn(worker_id: int) -> None:
    """
    Seed each DataLoader worker's NumPy RNG differently.

    Without this, all workers share the same ``np.random`` state after
    forking, producing identical augmentation sequences.  We mix the
    worker_id with the base seed to guarantee unique-per-worker streams.
    """
    base_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(base_seed + worker_id)


class SEN12MS_Dataset(Dataset):
    """
    PyTorch Dataset for paired SAR ↔ Optical patches.

    Parameters
    ----------
    records : list[dict]
        Each dict must contain ``sar_path`` and ``optical_path`` (str or Path).
    patch_size : int
        Spatial crop size (default 256).
    augment : bool
        If True, apply random horizontal/vertical flips (training only).
    """

    def __init__(
        self,
        records: List[Dict[str, Any]],
        patch_size: int = PATCH_SIZE,
        augment: bool = False,
    ) -> None:
        super().__init__()
        self.records = records
        self.patch_size = patch_size
        self.augment = augment

    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        return len(self.records)

    # ------------------------------------------------------------------ #
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"n_samples={len(self)}, "
            f"patch_size={self.patch_size}, "
            f"augment={self.augment})"
        )

    # ------------------------------------------------------------------ #
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rec = self.records[idx]

        # --- Load raw arrays ----------------------------------------- #
        sar = _load_tiff(Path(rec["sar_path"]), channels=SAR_CHANNELS)
        opt = _load_tiff(Path(rec["optical_path"]), channels=OPTICAL_CHANNELS)

        # --- Crop to patch_size --------------------------------------- #
        sar = _center_crop(sar, self.patch_size)
        opt = _center_crop(opt, self.patch_size)

        # --- Normalise ------------------------------------------------ #
        sar = normalise_sar(sar)
        opt = normalise_optical(opt)

        # --- Data augmentation (train only) --------------------------- #
        # Uses np.random which is seeded per-worker via _worker_init_fn,
        # ensuring different augmentation sequences across workers.
        if self.augment:
            if np.random.random() > 0.5:
                sar = sar[:, :, ::-1].copy()   # horizontal flip
                opt = opt[:, :, ::-1].copy()
            if np.random.random() > 0.5:
                sar = sar[:, ::-1, :].copy()   # vertical flip
                opt = opt[:, ::-1, :].copy()

        return {
            "sar": torch.from_numpy(sar),       # (SAR_CHANNELS, H, W)
            "optical": torch.from_numpy(opt),   # (OPTICAL_CHANNELS, H, W)
            "roi": rec["roi"],
            "patch_id": rec["patch_id"],
        }


# ======================================================================
# 7.  Convenience: build DataLoaders from a manifest
# ======================================================================

def build_dataloaders(
    manifest_path: Optional[Path] = None,
    batch_size: int = TRAIN_CFG.batch_size,
    num_workers: int = TRAIN_CFG.num_workers,
    pin_memory: bool = TRAIN_CFG.pin_memory,
) -> Dict[str, DataLoader]:
    """
    Build train / val / test_standard / test_unseen DataLoaders.

    If no manifest exists yet, discover pairs from DATA_ROOT, split them,
    save the manifest, and proceed.
    """
    manifest_path = manifest_path or MANIFEST_PATH / "split_manifest.json"

    if manifest_path.exists():
        logger.info("Loading existing manifest from %s", manifest_path)
        with open(manifest_path) as f:
            raw = json.load(f)
        splits = {k: v for k, v in raw.items()}
    else:
        logger.info("No manifest found — discovering patches …")
        pairs = _discover_patch_pairs(DATA_ROOT)
        if not pairs:
            raise FileNotFoundError(
                f"No SAR↔Optical pairs found under {DATA_ROOT}. "
                "Run `download_sen12ms()` first or check your data layout."
            )
        split_dict = assign_splits(pairs)
        save_split_manifest(split_dict, manifest_path)
        # Flatten back to serialisable dicts
        splits = {}
        for k, v in split_dict.items():
            splits[k] = [
                {
                    "roi": it["roi"],
                    "season": it["season"],
                    "patch_id": it["patch_id"],
                    "sar_path": str(it["sar_path"]),
                    "optical_path": str(it["optical_path"]),
                }
                for it in v
            ]

    loaders: Dict[str, DataLoader] = {}
    for split_name, records in splits.items():
        is_train = split_name == "train"
        ds = SEN12MS_Dataset(
            records=records,
            patch_size=PATCH_SIZE,
            augment=is_train,
        )
        loaders[split_name] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=is_train,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=is_train,          # avoid partial batches during train
            persistent_workers=num_workers > 0,
            worker_init_fn=_worker_init_fn,  # per-worker RNG seeding
        )
        logger.info(
            "DataLoader %-15s : %5d samples, batch=%d",
            split_name, len(ds), batch_size,
        )

    return loaders


# ======================================================================
# 8.  CLI entry-point
# ======================================================================

if __name__ == "__main__":
    import argparse

    # Only configure root logger when run as a script
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="SEN12MS data ingestion & zero-shot split builder."
    )
    parser.add_argument(
        "--download", action="store_true",
        help="Download SEN12MS archives (requires wget).",
    )
    parser.add_argument(
        "--data-root", type=Path, default=DATA_ROOT,
        help="Root directory containing SEN12MS ROI folders.",
    )
    parser.add_argument(
        "--build-manifest", action="store_true",
        help="Discover patches and write the split manifest JSON.",
    )
    args = parser.parse_args()

    if args.download:
        download_sen12ms(dest_dir=args.data_root)

    if args.build_manifest:
        pairs = _discover_patch_pairs(args.data_root)
        if pairs:
            split_dict = assign_splits(pairs)
            save_split_manifest(split_dict)
        else:
            logger.error(
                "No patches found under %s — cannot build manifest.", args.data_root
            )
