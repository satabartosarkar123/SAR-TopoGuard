#!/usr/bin/env python3
"""
Module 2 — Automated Structural Stratification
===============================================

Responsibilities
----------------
1.  Compute **Edge Density** (fraction of Canny-edge pixels) for every
    *optical* image in the training set.
2.  Derive the **median** training-set edge density as the decision
    boundary.
3.  Tag each test image (Standard *and* Unseen-Region) as either
    ``"Urban"`` (density > median) or ``"Natural"`` (density ≤ median).
4.  Persist a JSON manifest with
    ``[file_path, split_type, stratification_tag]``.

Edge Density Definition
~~~~~~~~~~~~~~~~~~~~~~~
Given an optical image I of H×W pixels::

    ED(I) = # { (i,j) : Canny(I)[i,j] > 0 } / (H × W)

A high ED implies dense man-made structures (roads, buildings), while a
low ED implies natural terrain (forest, water, bare soil).

Author : SAR-TopoGuard team
Date   : Day 1 — Afternoon
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from tqdm import tqdm

# ── Local imports ──────────────────────────────────────────────────────
from config import (
    MANIFEST_PATH,
    EDGE_DENSITY_MANIFEST,
    CANNY_LOW,
    CANNY_HIGH,
    PATCH_SIZE,
    OPTICAL_CHANNELS,
)

logger = logging.getLogger(__name__)


# ======================================================================
# 1.  Edge Density computation
# ======================================================================

def compute_edge_density(
    image: np.ndarray,
    canny_low: int = CANNY_LOW,
    canny_high: int = CANNY_HIGH,
) -> float:
    """
    Compute the edge-density of a single optical image.

    Parameters
    ----------
    image : np.ndarray
        Shape ``(C, H, W)`` or ``(H, W, C)`` or ``(H, W)``, float32 in [0, 1]
        or uint8 in [0, 255].
    canny_low, canny_high : int
        Canny hysteresis thresholds.

    Returns
    -------
    float
        Fraction of edge pixels ∈ [0, 1].

    Mathematical Note
    -----------------
    ``cv2.Canny`` applies Gaussian smoothing, non-maximum suppression,
    and double-threshold hysteresis.  The ratio of "surviving" edge
    pixels to total pixels is our structural-density proxy.
    """
    # ── Channel handling ────────────────────────────────────────────
    if image.ndim == 3:
        # Detect (C, H, W) layout: first dim is small AND both spatial
        # dims are large.  This avoids misdetection on square images.
        if (image.shape[0] <= 13
                and image.shape[1] > 13
                and image.shape[2] > 13):
            # (C, H, W) → (H, W, C)
            image = np.transpose(image, (1, 2, 0))

        # Convert to grayscale
        if image.shape[-1] >= 3:
            img_u8 = (
                (image[..., :3] * 255).astype(np.uint8)
                if image.max() <= 1.0
                else image[..., :3].astype(np.uint8)
            )
            gray = cv2.cvtColor(img_u8, cv2.COLOR_RGB2GRAY)
        else:
            ch0 = image[..., 0]
            gray = (
                (ch0 * 255).astype(np.uint8)
                if image.max() <= 1.0
                else ch0.astype(np.uint8)
            )
    elif image.ndim == 2:
        gray = (
            (image * 255).astype(np.uint8)
            if image.max() <= 1.0
            else image.astype(np.uint8)
        )
    else:
        raise ValueError(f"Unexpected image shape: {image.shape}")

    edges = cv2.Canny(gray, canny_low, canny_high)

    # Edge density = fraction of non-zero pixels
    total_pixels = edges.shape[0] * edges.shape[1]
    edge_pixels = int(np.count_nonzero(edges))
    return edge_pixels / total_pixels


# ======================================================================
# 2.  Load a lightweight image (avoids rasterio dependency here)
# ======================================================================

def _load_optical_for_stratification(path: Path) -> np.ndarray:
    """
    Load an optical patch as ``(H, W, C)`` float32 in [0, 1].

    Uses tifffile → PIL fallback chain.  We do NOT need the full
    rasterio stack for edge-density computation.
    """
    try:
        import tifffile
        img = tifffile.imread(str(path)).astype(np.float32)
    except Exception:
        from PIL import Image
        img = np.array(Image.open(path), dtype=np.float32)

    # Normalise to [0, 1]
    if img.max() > 1.0:
        img = img / max(img.max(), 1.0)

    # Ensure (H, W, C) — use the improved heuristic
    if img.ndim == 2:
        img = img[:, :, np.newaxis]
    elif img.ndim == 3:
        # Detect (C, H, W): first dim small, both spatial dims large
        if (img.shape[0] <= 13
                and img.shape[1] > 13
                and img.shape[2] > 13):
            img = np.transpose(img, (1, 2, 0))

    return img


# ======================================================================
# 3.  Stratification pipeline
# ======================================================================

def stratify_dataset(
    manifest_path: Path = MANIFEST_PATH / "split_manifest.json",
    output_path: Path = EDGE_DENSITY_MANIFEST,
) -> List[Dict[str, Any]]:
    """
    Full pipeline:

    1. Load the split manifest (produced by Module 1).
    2. Compute edge density for every training optical image.
    3. Derive the **median** training-set edge density.
    4. Tag each test image as "Urban" or "Natural".
    5. Save the stratified manifest JSON.

    Returns
    -------
    list[dict]
        Each entry: ``{ file_path, split_type, stratification_tag,
                        edge_density }``.
    """
    # ── Load manifest ───────────────────────────────────────────────
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Split manifest not found at {manifest_path}. "
            "Run Module 1 (dataset.py --build-manifest) first."
        )

    with open(manifest_path) as f:
        splits: Dict[str, List[Dict[str, Any]]] = json.load(f)

    # ── Step 1: Edge densities for the *training* set ──────────────
    logger.info("Computing edge densities for training set …")
    train_records = splits.get("train", [])
    train_densities: List[float] = []

    for rec in tqdm(train_records, desc="Train ED", unit="img"):
        opt_path = Path(rec["optical_path"])
        if not opt_path.exists():
            logger.warning("Missing optical file: %s — skipping.", opt_path)
            continue
        try:
            img = _load_optical_for_stratification(opt_path)
            ed = compute_edge_density(img)
            train_densities.append(ed)
        except Exception as exc:
            logger.warning("Failed to process %s: %s", opt_path, exc)

    if not train_densities:
        raise RuntimeError(
            "No valid training images found — cannot compute median edge density."
        )

    median_ed = float(np.median(train_densities))
    logger.info(
        "Training-set edge density — min=%.4f  median=%.4f  max=%.4f",
        min(train_densities), median_ed, max(train_densities),
    )

    # ── Step 2: Tag every image across all splits ──────────────────
    results: List[Dict[str, Any]] = []

    for split_name, records in splits.items():
        desc = f"Stratify {split_name}"
        for rec in tqdm(records, desc=desc, unit="img"):
            opt_path = Path(rec["optical_path"])
            if not opt_path.exists():
                tag = "MISSING"
                ed = -1.0
            else:
                try:
                    img = _load_optical_for_stratification(opt_path)
                    ed = compute_edge_density(img)
                except Exception:
                    ed = -1.0
                    tag = "ERROR"
                else:
                    # ── Decision rule ────────────────────────────────
                    # ED > training median → "Urban"
                    # ED ≤ training median → "Natural"
                    tag = "Urban" if ed > median_ed else "Natural"

            results.append({
                "sar_path": rec.get("sar_path", ""),
                "optical_path": rec.get("optical_path", ""),
                "roi": rec.get("roi", ""),
                "patch_id": rec.get("patch_id", ""),
                "split_type": split_name,
                "edge_density": round(ed, 6),
                "stratification_tag": tag,
            })

    # ── Step 3: Persist ────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(
            {
                "metadata": {
                    "median_train_edge_density": round(median_ed, 6),
                    "canny_low": CANNY_LOW,
                    "canny_high": CANNY_HIGH,
                    "num_records": len(results),
                },
                "records": results,
            },
            f,
            indent=2,
        )
    logger.info("Saved stratified manifest → %s  (%d records)", output_path, len(results))

    # ── Summary statistics ─────────────────────────────────────────
    for split_name in splits:
        subset = [r for r in results if r["split_type"] == split_name]
        n_urban = sum(1 for r in subset if r["stratification_tag"] == "Urban")
        n_natural = sum(1 for r in subset if r["stratification_tag"] == "Natural")
        logger.info(
            "  %-15s : Urban=%d  Natural=%d  (total=%d)",
            split_name, n_urban, n_natural, len(subset),
        )

    return results


# ======================================================================
# CLI entry-point
# ======================================================================

if __name__ == "__main__":
    import argparse

    # Only configure root logger when run as a script
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Automated structural stratification of SEN12MS patches."
    )
    parser.add_argument(
        "--manifest", type=Path,
        default=MANIFEST_PATH / "split_manifest.json",
        help="Path to the split manifest JSON (from Module 1).",
    )
    parser.add_argument(
        "--output", type=Path,
        default=EDGE_DENSITY_MANIFEST,
        help="Where to save the stratified manifest.",
    )
    args = parser.parse_args()

    stratify_dataset(
        manifest_path=args.manifest,
        output_path=args.output,
    )
