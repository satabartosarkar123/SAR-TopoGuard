#!/usr/bin/env python3
"""
PHASE 1: DATA HEALTH VERIFICATION GATE
=======================================
Mathematically proves the optical targets are now healthy BEFORE training.
"""
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader

# Ensure the project root is importable
sys.path.insert(0, str(Path(__file__).parent))

from dataset import SEN12MS_Dataset, denormalize_for_display


def build_loaders(split: str, max_images: int = 50):
    """Build a dataloader for the given split directory."""
    data_root = Path("mini_sen12_data")
    ds = SEN12MS_Dataset(records=[], patch_size=256)
    
    s1_dir = data_root / split / "s1"
    s2_dir = data_root / split / "s2"
    
    s1_files = sorted([p for p in s1_dir.glob("*.*") if p.suffix.lower() in [".tif", ".png"]])
    records = []
    for p in s1_files:
        opt_path = s2_dir / p.name
        if opt_path.exists():
            records.append({
                "sar_path": str(p),
                "optical_path": str(opt_path),
                "roi": "mini",
                "patch_id": p.stem,
            })
    
    ds.records = records[:max_images]
    loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=0)
    return loader, len(ds.records)


def main():
    print("=" * 70)
    print("PHASE 1: DATA HEALTH VERIFICATION GATE")
    print("=" * 70)
    
    # ── STEP 1.1: Ground Truth Health Audit ─────────────────────────────
    print("\nSTEP 1.1: Ground Truth (Optical) Health Audit")
    print("-" * 50)
    
    # Try val first, fall back to train
    try:
        loader, n_records = build_loaders("val", max_images=50)
        split_used = "val"
    except Exception:
        loader, n_records = build_loaders("train", max_images=50)
        split_used = "train"
    
    print(f"Using split '{split_used}' with {n_records} images")
    
    opt_means = []
    opt_vars = []
    opt_mins = []
    opt_maxs = []
    opt_darks = []
    
    sar_means = []
    sar_vars = []
    sar_has_nan = False
    sar_has_inf = False
    
    # For visualization
    vis_sar = []
    vis_opt = []
    
    count = 0
    for batch in loader:
        sar_batch = batch["sar"]    # (B, 2, H, W) in [-1, 1]
        opt_batch = batch["optical"]  # (B, 3, H, W) in [-1, 1]
        
        for i in range(opt_batch.shape[0]):
            if count >= 50:
                break
            
            # Denormalize optical from [-1,1] to [0,1] for audit
            opt_display = denormalize_for_display(opt_batch[i].numpy())  # (3, H, W)
            
            var = np.var(opt_display, dtype=np.float64)
            mean = np.mean(opt_display, dtype=np.float64)
            min_val = float(np.min(opt_display))
            max_val = float(np.max(opt_display))
            dark_pct = float(np.mean(opt_display < 0.1)) * 100.0
            
            opt_means.append(mean)
            opt_vars.append(var)
            opt_mins.append(min_val)
            opt_maxs.append(max_val)
            opt_darks.append(dark_pct)
            
            # SAR audit
            sar_np = sar_batch[i].numpy()  # (2, H, W) in [-1, 1]
            sar_display = denormalize_for_display(sar_np)  # (2, H, W) in [0, 1]
            sar_means.append(np.mean(sar_np, dtype=np.float64))
            sar_vars.append(np.var(sar_np, dtype=np.float64))
            if np.any(np.isnan(sar_np)):
                sar_has_nan = True
            if np.any(np.isinf(sar_np)):
                sar_has_inf = True
            
            if count < 4:
                vis_sar.append(sar_display[0])  # First channel for display
                vis_opt.append(np.transpose(opt_display, (1, 2, 0)))  # (H, W, 3)
            
            if count < 10:
                print(f"  GT Img {count+1:02d}: Var={var:.8f} | Mean={mean:.4f} | "
                      f"Min={min_val:.4f} | Max={max_val:.4f} | Dark={dark_pct:.1f}%")
            
            count += 1
        
        if count >= 50:
            break
    
    # Aggregates
    avg_mean = np.mean(opt_means)
    avg_var = np.mean(opt_vars)
    avg_max = np.mean(opt_maxs)
    avg_dark = np.mean(opt_darks)
    
    print(f"\n  ... ({count} images total, showing first 10)")
    print(f"\n  AVERAGE ACROSS {count} GT IMAGES:")
    print(f"    Mean Brightness:  {avg_mean:.4f}")
    print(f"    Variance:         {avg_var:.8f}")
    print(f"    Max Pixel Value:  {avg_max:.4f}")
    print(f"    Dark Pixels:      {avg_dark:.1f}%")
    
    # ── STEP 1.2: Health Assertions ─────────────────────────────────────
    print("\nSTEP 1.2: Health Assertions")
    print("-" * 50)
    
    all_passed = True
    
    # A1: Mean brightness between 0.25 and 0.75
    if 0.25 <= avg_mean <= 0.75:
        print(f"  [PASS] Mean brightness = {avg_mean:.4f} (in [0.25, 0.75])")
    else:
        print(f"  [FAIL] Mean brightness = {avg_mean:.4f} (OUTSIDE [0.25, 0.75])")
        all_passed = False
    
    # A2: Variance > 0.02
    if avg_var > 0.02:
        print(f"  [PASS] Variance = {avg_var:.6f} (> 0.02)")
    else:
        print(f"  [FAIL] Variance = {avg_var:.8f} (NOT > 0.02)")
        all_passed = False
    
    # A3: Max pixel >= 0.9
    if avg_max >= 0.9:
        print(f"  [PASS] Max pixel value = {avg_max:.4f} (>= 0.9)")
    else:
        print(f"  [FAIL] Max pixel value = {avg_max:.4f} (NOT >= 0.9)")
        all_passed = False
    
    # A4: Dark pixels < 40%
    if avg_dark < 40.0:
        print(f"  [PASS] Dark pixels = {avg_dark:.1f}% (< 40%)")
    else:
        print(f"  [FAIL] Dark pixels = {avg_dark:.1f}% (NOT < 40%)")
        all_passed = False
    
    # ── STEP 1.3: SAR Health Audit ──────────────────────────────────────
    print("\nSTEP 1.3: SAR Health Audit")
    print("-" * 50)
    
    avg_sar_mean = np.mean(sar_means)
    avg_sar_var = np.mean(sar_vars)
    
    print(f"  SAR Mean (in [-1,1]):  {avg_sar_mean:.4f}")
    print(f"  SAR Variance:          {avg_sar_var:.6f}")
    print(f"  SAR has NaN:           {sar_has_nan}")
    print(f"  SAR has Inf:           {sar_has_inf}")
    
    # SAR assertions
    if -0.5 <= avg_sar_mean <= 0.5:
        print(f"  [PASS] SAR mean = {avg_sar_mean:.4f} (in [-0.5, 0.5])")
    else:
        print(f"  [FAIL] SAR mean = {avg_sar_mean:.4f} (OUTSIDE [-0.5, 0.5])")
        all_passed = False
    
    if avg_sar_var > 0.01:
        print(f"  [PASS] SAR variance = {avg_sar_var:.6f} (> 0.01)")
    else:
        print(f"  [FAIL] SAR variance = {avg_sar_var:.6f} (NOT > 0.01)")
        all_passed = False
    
    if not sar_has_nan:
        print(f"  [PASS] No NaN values in SAR")
    else:
        print(f"  [FAIL] NaN values detected in SAR")
        all_passed = False
    
    if not sar_has_inf:
        print(f"  [PASS] No Inf values in SAR")
    else:
        print(f"  [FAIL] Inf values detected in SAR")
        all_passed = False
    
    # ── STEP 1.4: Visual Proof ──────────────────────────────────────────
    print("\nSTEP 1.4: Visual Proof")
    print("-" * 50)
    
    Path("results").mkdir(exist_ok=True)
    
    n_vis = min(4, len(vis_sar))
    fig, axes = plt.subplots(2, n_vis, figsize=(4 * n_vis, 8))
    if n_vis == 1:
        axes = axes.reshape(2, 1)
    
    for i in range(n_vis):
        axes[0, i].imshow(vis_sar[i], cmap="gray", vmin=0, vmax=1)
        axes[0, i].set_title(f"SAR Input {i+1}", fontsize=11)
        axes[0, i].axis("off")
        
        axes[1, i].imshow(np.clip(vis_opt[i], 0, 1))
        axes[1, i].set_title(f"Optical GT {i+1}", fontsize=11)
        axes[1, i].axis("off")
    
    fig.suptitle("Phase 1: Data Health Verification\n"
                 f"Mean={avg_mean:.3f}  Var={avg_var:.5f}  Dark={avg_dark:.0f}%",
                 fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig("results/phase1_data_health.png", dpi=150, bbox_inches="tight")
    print(f"  Saved visualization -> results/phase1_data_health.png")
    
    # ── FINAL GATE ──────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    if all_passed:
        print("PHASE 1 PASSED: Data is healthy. Cleared for training.")
    else:
        print("PHASE 1 FAILED: One or more health assertions failed. STOPPING.")
        print("Do NOT proceed to training until data pipeline is fixed.")
        sys.exit(1)
    print("=" * 70)


if __name__ == "__main__":
    main()
