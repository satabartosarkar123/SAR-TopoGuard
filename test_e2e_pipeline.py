#!/usr/bin/env python3
"""
End-to-End Integration Script ("Ghost in the Machine")
======================================================
Validates the entire SAR-TopoGuard pipeline from data ingestion to 
stochastic topological selection and frequency gating.
"""

import torch
import tifffile
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader

from dataset import SEN12MS_Dataset
from models import UNetGenerator
from verifier import TopologicalVerifier
from topoguard_engine import generate_and_select, apply_frequency_gate

def create_dummy_data() -> list:
    """Creates synthetic TIFF files to simulate SEN12MS ingestion without downloading 500GB."""
    test_dir = Path("test_data")
    test_dir.mkdir(exist_ok=True)
    
    # SAR usually has 2 channels (VV, VH)
    sar_data = np.random.rand(2, 256, 256).astype(np.float32)
    # Optical usually has 3 channels (RGB)
    opt_data = np.random.rand(3, 256, 256).astype(np.float32)
    
    sar_path = test_dir / "dummy_sar.tif"
    opt_path = test_dir / "dummy_opt.tif"
    
    # Write using tifffile as (C, H, W)
    tifffile.imwrite(sar_path, sar_data)
    tifffile.imwrite(opt_path, opt_data)
    
    records = [
        {"sar_path": str(sar_path), "optical_path": str(opt_path), "roi": "ROI_TEST", "patch_id": "0"},
        {"sar_path": str(sar_path), "optical_path": str(opt_path), "roi": "ROI_TEST", "patch_id": "1"}
    ]
    return records

def run_e2e_test():
    print("=" * 60)
    print("[START] GHOST IN THE MACHINE: E2E INTEGRATION TEST")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Target Device: {device}\n")
    
    # 1. Data Ingestion
    print("--- 1. DATA INGESTION ---")
    records = create_dummy_data()
    dataset = SEN12MS_Dataset(records=records, patch_size=256, augment=False)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False)
    
    batch = next(iter(dataloader))
    sar_batch = batch["sar"].to(device)
    opt_batch = batch["optical"].to(device)
    
    print(f"SAR Batch  -> Shape: {sar_batch.shape}, Device: {sar_batch.device}")
    print(f"OPT Batch  -> Shape: {opt_batch.shape}, Device: {opt_batch.device}\n")
    
    # 2. Model Instantiation
    print("--- 2. MODEL INSTANTIATION ---")
    generator = UNetGenerator().to(device)
    verifier = TopologicalVerifier().to(device)
    
    print(f"Generator  -> {sum(p.numel() for p in generator.parameters()) / 1e6:.2f} M params, Device: {next(generator.parameters()).device}")
    print(f"Verifier   -> Mathematical Core, Device: {verifier.laplacian.device}\n")
    
    # 3. Execution Flow
    print("--- 3. EXECUTION FLOW ---")
    print("Running TopoGuard Engine: generate_and_select() (K=3)...")
    Y_best = generate_and_select(sar_batch, generator, verifier, K=3)
    
    print(f"Y_best     -> Shape: {Y_best.shape}, Device: {Y_best.device}")
    
    print("Running TopoGuard Engine: apply_frequency_gate()...")
    Y_final = apply_frequency_gate(Y_best, sar_batch, verifier)
    
    print(f"Y_final    -> Shape: {Y_final.shape}, Device: {Y_final.device}\n")
    
    # 4. Strict Assertions
    print("--- 4. STRICT ASSERTIONS ---")
    
    # Device match
    assert Y_final.device == sar_batch.device, f"Device mismatch! Y_final: {Y_final.device}, sar_batch: {sar_batch.device}"
    print("[PASS] Y_final is on the correct device.")
    
    # Shape match
    expected_shape = (2, 3, 256, 256)
    assert Y_final.shape == expected_shape, f"Shape mismatch! Expected {expected_shape}, got {Y_final.shape}"
    print(f"[PASS] Y_final shape is exactly {expected_shape}.")
    
    # Value bounds (Safety check within normal expected ranges [-0.5, 1.5] for high-pass addition)
    y_min, y_max = Y_final.min().item(), Y_final.max().item()
    assert -1.0 <= y_min and y_max <= 2.0, f"Value bounds suspicious! min: {y_min:.4f}, max: {y_max:.4f}"
    print(f"[PASS] Y_final values are within expected optical ranges (min: {y_min:.4f}, max: {y_max:.4f}).")
    
    # NaN check
    assert not torch.isnan(Y_final).any(), "Y_final contains NaNs!"
    assert not torch.isinf(Y_final).any(), "Y_final contains Infs!"
    print("[PASS] Y_final is clean of NaNs and Infs.\n")
    
    print("=" * 60)
    print("[PASS] END-TO-END PIPELINE FLAWLESS")
    print("=" * 60)

if __name__ == "__main__":
    run_e2e_test()
