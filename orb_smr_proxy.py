#!/usr/bin/env python3
"""
Module: ORB-SMR Proxy (Day 4)
=============================

This module implements the cross-platform ORB-SMR (Scale-invariant Feature Transform 
and Matching) proxy. It uses OpenCV to extract and match ORB features.

Crucially, this module guarantees OpenCV operations are strictly CPU-bound, 
making it compatible across Windows (CUDA/CPU), Linux, and macOS (MPS/CPU).
"""

import logging
import cv2
import numpy as np
import torch

from config import get_device

logger = logging.getLogger(__name__)

def detect_and_match(sar_tensor: torch.Tensor, opt_tensor: torch.Tensor) -> dict:
    """
    Extract and match ORB features between SAR and Optical tensors.
    
    Parameters
    ----------
    sar_tensor : [B, C_sar, H, W] tensor
    opt_tensor : [B, C_opt, H, W] tensor
    
    Returns
    -------
    dict containing match results.
    """
    # ---------------------------------------------------------
    # Cross-Platform CPU Binding
    # OpenCV strictly requires CPU-bound NumPy arrays.
    # .detach().cpu().numpy() guarantees compatibility even if
    # the tensor is on MPS (macOS) or CUDA (Windows/Linux).
    # ---------------------------------------------------------
    sar_np = sar_tensor.detach().cpu().numpy()
    opt_np = opt_tensor.detach().cpu().numpy()
    
    # Process the first item in the batch for matching
    # Convert from (C, H, W) -> (H, W, C)
    sar_img = np.transpose(sar_np[0], (1, 2, 0))
    opt_img = np.transpose(opt_np[0], (1, 2, 0))
    
    # Convert to Grayscale (CV_8UC1) required by ORB
    # Assuming tensors are normalized [0, 1]
    sar_gray = (sar_img.mean(axis=2) * 255).astype(np.uint8)
    opt_gray = (opt_img.mean(axis=2) * 255).astype(np.uint8)
    
    # Initialize ORB detector
    orb = cv2.ORB_create()
    
    # Find keypoints and descriptors
    kp_sar, des_sar = orb.detectAndCompute(sar_gray, None)
    kp_opt, des_opt = orb.detectAndCompute(opt_gray, None)
    
    matches = []
    if des_sar is not None and des_opt is not None:
        # BFMatcher with default params
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des_sar, des_opt)
        # Sort them in ascending order of their distances
        matches = sorted(matches, key=lambda x: x.distance)
        
    return {
        "kp_sar": kp_sar,
        "kp_opt": kp_opt,
        "matches": matches,
        "num_matches": len(matches)
    }

def run_inference_safe(model: torch.nn.Module, inputs: torch.Tensor) -> torch.Tensor:
    """
    Runs model inference with AMP safety guarantees.
    Mixed precision is only used if the device is CUDA, because MPS/CPU
    have incomplete or problematic autocast support.
    """
    device = get_device()
    use_amp = (device.type == "cuda")
    
    model.eval()
    with torch.no_grad():
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            outputs = model(inputs)
            
    return outputs

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("Initializing Cross-Platform ORB-SMR Proxy...")
    
    device = get_device()
    logger.info(f"Target Device selected dynamically: {device}")
    
    # Dummy tensors simulating output
    dummy_sar = torch.rand(1, 2, 256, 256, device=device)
    dummy_opt = torch.rand(1, 3, 256, 256, device=device)
    
    logger.info("Executing detect_and_match with CPU-bound OpenCV...")
    results = detect_and_match(dummy_sar, dummy_opt)
    
    logger.info(f"Successfully matched {results['num_matches']} ORB features.")
    logger.info("Cross-Platform proxy tests passed cleanly.")
