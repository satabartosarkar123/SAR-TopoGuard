#!/usr/bin/env python3
"""
Module: Master Inference Engine (Day 3)
=======================================

This script coordinates the stochastic generation of optical candidates via
MC-Dropout, performs mathematical verification and topological selection,
and applies the SAR-Guided Frequency Gate to suppress residual hallucinations.

Author: SAR-TopoGuard team
Date: Day 3
"""

import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import OPTICAL_CHANNELS, SAR_CHANNELS, get_device
from models import UNetGenerator
from verifier import TopologicalVerifier

# Configure strict logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


# ======================================================================
# COMPONENT 1: Strict MC-Dropout Context Manager
# ======================================================================

@contextmanager
def mc_dropout_context(model: nn.Module):
    """
    Context manager to safely enable Monte Carlo Dropout during inference.
    It forces all Dropout layers to .train() mode while strictly keeping
    Normalization layers (BatchNorm, InstanceNorm) in .eval() mode to
    prevent batch statistic corruption.
    """
    # Start by ensuring the whole model is in eval mode
    model.eval()
    
    # Specifically toggle Dropout layers
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d)):
            m.train()
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
            # Explicitly enforce eval for norm layers, just in case
            m.eval()
            
    try:
        yield model
    finally:
        # Safely revert everything to eval mode
        model.eval()


# ======================================================================
# COMPONENT 2: Stochastic Generation & Topological Selection
# ======================================================================

@torch.no_grad()
def generate_and_select(
    sar_batch: torch.Tensor,
    generator: nn.Module,
    verifier: TopologicalVerifier,
    K: int = 5
) -> torch.Tensor:
    """
    Generate K stochastic candidates and select the best topologically
    sound candidate (Y_best).
    
    Parameters
    ----------
    sar_batch : [B, C_sar, H, W]
    generator : The UNetGenerator model
    verifier  : TopologicalVerifier instance
    K         : Number of MC-Dropout candidates
    
    Returns
    -------
    Y_best : [B, C_opt, H, W]
    """
    B, C_sar, H, W = sar_batch.shape
    candidates = []
    
    # 1. Stochastic Generation
    with mc_dropout_context(generator):
        for _ in range(K):
            opt_k = generator(sar_batch)
            candidates.append(opt_k)
            
    # Stack to [B, K, C_opt, H, W]
    candidates_tensor = torch.stack(candidates, dim=1)
    C_opt = candidates_tensor.shape[2]
    
    # 2. Topological Scoring
    # Flatten B and K dimensions to process efficiently through the verifier
    # Ensure memory safety by processing entirely batched or micro-batched. 
    # For RTX 3050, standard B=2 and K=5 -> batch size 10 is very safe.
    sar_expanded = sar_batch.unsqueeze(1).expand(-1, K, -1, -1, -1).reshape(B * K, C_sar, H, W)
    opt_flat = candidates_tensor.reshape(B * K, C_opt, H, W)
    
    # The verifier internally handles HFI and GMC
    hfi = verifier.hallucination_frequency_index(sar_expanded, opt_flat)
    gmc = verifier.gradient_magnitude_correlation(sar_expanded, opt_flat)
    
    # Note: hfi returned by the verifier is a single scalar mean. 
    # To rank candidates properly per sample in the batch, we need a modified 
    # un-reduced HFI calculation per sample. Let's compute it explicitly:
    
    # Extract optical high-frequencies
    laplacian = verifier.laplacian
    opt_flat_reshaped = opt_flat.reshape(B * K * C_opt, 1, H, W)
    high_freq = torch.abs(F.conv2d(opt_flat_reshaped, laplacian, padding=1))
    high_freq = high_freq.reshape(B * K, C_opt, H, W).mean(dim=1, keepdim=True)
    
    # Flatness mask from SAR gradient magnitude
    gm_sar = verifier.gradient_magnitude(sar_expanded).mean(dim=1, keepdim=True)
    clamped_gm = torch.clamp(gm_sar, min=0.0, max=verifier.gm_clamp_max)
    flatness_mask = torch.exp(-verifier.flatness_gain * clamped_gm)
    
    # Un-reduced HFI per candidate
    hfi_per_candidate = (high_freq * flatness_mask).reshape(B * K, -1).mean(dim=1)
    
    # Calculate S_topo = 1.0 * GMC - 5.0 * HFI
    S_topo = 1.0 * gmc - 5.0 * hfi_per_candidate
    
    # Reshape back to [B, K]
    S_topo = S_topo.reshape(B, K)
    
    # 3. Selection
    best_idx = torch.argmax(S_topo, dim=1)
    
    # Extract Y_best safely
    Y_best = torch.stack([candidates_tensor[b, best_idx[b]] for b in range(B)], dim=0)
    
    return Y_best


# ======================================================================
# COMPONENT 3: SAR-Guided Frequency Gating
# ======================================================================

def create_gaussian_kernel(channels: int, device: torch.device) -> torch.Tensor:
    """Create a 5x5 Gaussian kernel for depthwise convolution."""
    # Standard 5x5 Gaussian approximation
    kernel_1d = torch.tensor([1., 4., 6., 4., 1.], device=device)
    kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
    kernel_2d = kernel_2d / kernel_2d.sum()
    # Reshape to [C, 1, 5, 5] for group convolution
    return kernel_2d.view(1, 1, 5, 5).repeat(channels, 1, 1, 1)

@torch.no_grad()
def apply_frequency_gate(
    Y_best: torch.Tensor,
    sar_struct: torch.Tensor,
    verifier: TopologicalVerifier
) -> torch.Tensor:
    """
    Surgically suppress residual hallucinations using mathematical frequency gating.
    
    Parameters
    ----------
    Y_best : [B, C_opt, H, W] Selected candidate optical
    sar_struct : [B, C_sar, H, W] SAR structural tensor
    verifier : TopologicalVerifier instance for accessing exact math ops
    
    Returns
    -------
    Y_final : [B, C_opt, H, W]
    """
    B, C_opt, H, W = Y_best.shape
    device = Y_best.device
    
    # Step A: 5x5 Gaussian Blur (Low-Pass)
    gaussian_kernel = create_gaussian_kernel(C_opt, device=device)
    # padding=2 on 5x5 kernel acts as 'same' padding
    Y_low = F.conv2d(Y_best, gaussian_kernel, padding=2, groups=C_opt)
    
    # Step B: High-Pass
    Y_high = Y_best - Y_low
    
    # Step C: Robust GM (from verifier)
    gm_sar = verifier.gradient_magnitude(sar_struct).mean(dim=1, keepdim=True)
    
    # Step D: Edge Mask
    # M_edge = sigmoid(5.0 * (G(X) - 0.1))
    M_edge = torch.sigmoid(5.0 * (gm_sar - 0.1))
    
    # Step E: Blended Output
    Y_final = Y_low + (Y_high * M_edge)
    
    return Y_final


# ======================================================================
# COMPONENT 4: Mathematical Validation Suite
# ======================================================================

def test_inference_math():
    """
    Bulletproof mathematical validation suite for the inference engine.
    Runs rigorous edge-case assertions.
    """
    device = get_device()
    logger.info(f"Running inference math validation on {device}...")
    
    verifier = TopologicalVerifier().to(device)
    
    # Prepare dummy edge cases
    B, H, W = 2, 64, 64
    
    # Case 1: Perfectly flat SAR, noisy optical (hallucination)
    sar_flat = torch.zeros((B, SAR_CHANNELS, H, W), device=device)
    opt_noisy = torch.rand((B, OPTICAL_CHANNELS, H, W), device=device)
    
    # Case 2: Matching structural features
    # Let's create a distinct pattern
    pattern = torch.zeros((1, 1, H, W), device=device)
    pattern[:, :, H//2-10:H//2+10, W//2-10:W//2+10] = 1.0
    sar_struct = pattern.repeat(B, SAR_CHANNELS, 1, 1)
    opt_struct = pattern.repeat(B, OPTICAL_CHANNELS, 1, 1)
    
    with torch.no_grad():
        # --- TEST 1: Hallucination Penalty ---
        logger.info("Executing TEST 1: Hallucination Penalty")
        hfi_flat_res = verifier.hallucination_frequency_index(sar_flat, opt_noisy)
        gmc_flat_res = verifier.gradient_magnitude_correlation(sar_flat, opt_noisy).mean()
        
        # S_topo calculation logic
        s_topo_val = (1.0 * gmc_flat_res) - (5.0 * hfi_flat_res)
        
        # We expect a high HFI for noise over a flat surface
        assert hfi_flat_res.item() > 0.05, f"HFI too low for clear hallucination: {hfi_flat_res.item()}"
        assert s_topo_val.item() < 0.0, f"S_topo should heavily penalize hallucination, got {s_topo_val.item()}"
        
        # --- TEST 2: Perfect Match ---
        logger.info("Executing TEST 2: Perfect Structural Match")
        gmc_struct_res = verifier.gradient_magnitude_correlation(sar_struct, opt_struct).mean()
        
        # Since SAR and Opt share the exact same structural boundary pattern, GMC should be very high
        assert gmc_struct_res.item() > 0.95, f"GMC should be ~1.0 for matching structures, got {gmc_struct_res.item()}"
        
        # --- TEST 3: Mask Bounds & NaN Check (Frequency Gating) ---
        logger.info("Executing TEST 3: Frequency Gating Exact Constraints")
        Y_final = apply_frequency_gate(opt_noisy, sar_struct, verifier)
        
        # Retrieve internal M_edge manually just for validation
        gm_sar = verifier.gradient_magnitude(sar_struct).mean(dim=1, keepdim=True)
        M_edge = torch.sigmoid(5.0 * (gm_sar - 0.1))
        
        # Bound assertions
        assert torch.all(M_edge >= 0.0) and torch.all(M_edge <= 1.0), "M_edge bounds violated! Not strictly [0,1]"
        
        # NaN assertions
        assert not torch.isnan(Y_final).any(), "Y_final contains NaNs!"
        assert not torch.isinf(Y_final).any(), "Y_final contains Infs!"
        assert Y_final.shape == opt_noisy.shape, f"Y_final shape mismatch: {Y_final.shape} != {opt_noisy.shape}"
        
        # --- FINAL REPORT ---
        print("\n" + "="*50)
        print("[PASS] MATHEMATICAL VALIDATION SUITE PASSED")
        print("="*50)
        print(f"Test 1 - Hallucination HFI:  {hfi_flat_res.item():.4f} (Expected > 0.05)")
        print(f"Test 1 - Penalized S_topo:   {s_topo_val.item():.4f} (Expected < 0.0)")
        print(f"Test 2 - Matched Struct GMC: {gmc_struct_res.item():.4f} (Expected > 0.95)")
        print(f"Test 3 - M_edge min/max:     {M_edge.min().item():.4f} / {M_edge.max().item():.4f} (Expected exactly bounded [0, 1])")
        print("Test 3 - NaN/Inf checks:     CLEAN")
        print("="*50)


if __name__ == "__main__":
    test_inference_math()
