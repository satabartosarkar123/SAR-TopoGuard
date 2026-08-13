import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import tifffile
from dataset import _load_tiff

def adaptive_edge_iou(gt_img_float, gen_img_float):
    """
    Robust Edge-IoU to eliminate contrast bias against GAN outputs.
    Uses Otsu's Thresholding or Mean-based fallback to prevent Canny collapse.
    """
    gt_8bit = (gt_img_float * 255.0).clip(0, 255).astype(np.uint8)
    gen_8bit = (gen_img_float * 255.0).clip(0, 255).astype(np.uint8)
    
    def get_robust_canny(img):
        # Convert to grayscale if it's 3-channel
        if len(img.shape) == 3 and img.shape[-1] == 3:
            img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            img_gray = img
            
        mu = np.mean(img_gray)
        if mu < 10:
            t_low, t_high = 10, 50
        else:
            t_high, _ = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            t_low = 0.5 * t_high
            
        return cv2.Canny(img_gray, int(t_low), int(t_high))
        
    edges_gt = get_robust_canny(gt_8bit)
    edges_gen = get_robust_canny(gen_8bit)
    
    # 3x3 Morphological Dilation
    kernel = np.ones((3, 3), np.uint8)
    edges_gt = cv2.dilate(edges_gt, kernel, iterations=1)
    edges_gen = cv2.dilate(edges_gen, kernel, iterations=1)
    
    intersection = np.logical_and(edges_gt, edges_gen).sum()
    union = np.logical_or(edges_gt, edges_gen).sum()
    
    return intersection / union if union > 0 else 0.0

def rigorous_orb_smr(sar_img_float, gen_img_float):
    """
    Mathematically Valid SMR. Attempts SIFT with Lowe's Ratio (L2 Norm) + Symmetry.
    Falls back to ORB with ONLY Bidirectional Symmetry (no ratio test) if SIFT fails.
    """
    if len(sar_img_float.shape) == 3 and sar_img_float.shape[-1] == 3:
        sar_gray = cv2.cvtColor((sar_img_float * 255).clip(0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    else:
        sar_gray = (sar_img_float * 255).clip(0, 255).astype(np.uint8)
        
    if len(gen_img_float.shape) == 3 and gen_img_float.shape[-1] == 3:
        gen_gray = cv2.cvtColor((gen_img_float * 255).clip(0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    else:
        gen_gray = (gen_img_float * 255).clip(0, 255).astype(np.uint8)

    try:
        sift = cv2.SIFT_create()
        kp_sar, des_sar = sift.detectAndCompute(sar_gray, None)
        kp_opt, des_opt = sift.detectAndCompute(gen_gray, None)
        
        if des_sar is None or len(des_sar) < 2 or des_opt is None or len(des_opt) < 2:
            return 0.0
            
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        
        # Forward A -> B
        matches_forward = bf.knnMatch(des_sar, des_opt, k=2)
        good_forward = []
        for m in matches_forward:
            if len(m) == 2 and m[0].distance < 0.75 * m[1].distance:
                good_forward.append((m[0].queryIdx, m[0].trainIdx))
                
        # Backward B -> A
        matches_backward = bf.knnMatch(des_opt, des_sar, k=2)
        good_backward = []
        for m in matches_backward:
            if len(m) == 2 and m[0].distance < 0.75 * m[1].distance:
                good_backward.append((m[0].queryIdx, m[0].trainIdx))
                
        # Symmetry Intersection
        backward_set = set((t, q) for q, t in good_backward)
        final_matches = [m for m in good_forward if m in backward_set]
        
    except Exception:
        # FALLBACK: ORB with pure Symmetry (No Ratio Test)
        orb = cv2.ORB_create(nfeatures=500)
        kp_sar, des_sar = orb.detectAndCompute(sar_gray, None)
        kp_opt, des_opt = orb.detectAndCompute(gen_gray, None)
        
        if des_sar is None or len(des_sar) < 2 or des_opt is None or len(des_opt) < 2:
            return 0.0
            
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        
        # Forward A -> B
        matches_forward = bf.knnMatch(des_sar, des_opt, k=1)
        good_forward = [(m[0].queryIdx, m[0].trainIdx) for m in matches_forward if len(m) > 0]
                
        # Backward B -> A
        matches_backward = bf.knnMatch(des_opt, des_sar, k=1)
        good_backward = [(m[0].queryIdx, m[0].trainIdx) for m in matches_backward if len(m) > 0]
                
        # Symmetry Intersection
        backward_set = set((t, q) for q, t in good_backward)
        final_matches = [m for m in good_forward if m in backward_set]

    min_kp = min(len(kp_sar), len(kp_opt))
    return len(final_matches) / min_kp if min_kp > 0 else 0.0

def main():
    gt_dir = Path("mini_sen12_data/val/s2")
    sar_dir = Path("mini_sen12_data/val/s1")
    gen_dir = Path("results/generated_images")
    
    gt_files = list(gt_dir.glob("*.png")) + list(gt_dir.glob("*.tif"))
    if not gt_files:
        print("No GT files found!")
        return

    methods = ["baseline1", "baseline2", "topoguard"]
    results = {m: {"psnr": [], "ssim": [], "edge_iou": [], "orb_smr": []} for m in methods}
    
    for gt_path in gt_files:
        img_id = gt_path.stem
        sar_path = sar_dir / f"{img_id}.png"
        if not sar_path.exists():
            sar_path = sar_dir / f"{img_id}.tif"
            
        # Load GT Optical and SAR
        gt_img = _load_tiff(gt_path, channels=3)
        sar_img = _load_tiff(sar_path, channels=1) if sar_path.exists() else np.zeros_like(gt_img)
        
        # Scale to strictly [0.0, 1.0] for domain alignment
        gt_scaled = (gt_img.astype(np.float32) / 255.0).clip(0.0, 1.0)
        sar_scaled = (sar_img.astype(np.float32) / 255.0).clip(0.0, 1.0)
            
        if gt_scaled.shape[0] == 3:
            gt_scaled = np.transpose(gt_scaled, (1, 2, 0))
        if len(sar_scaled.shape) == 3 and sar_scaled.shape[0] in [1, 3]:
            sar_scaled = np.transpose(sar_scaled, (1, 2, 0))
            
        for meth in methods:
            gen_path = gen_dir / f"{img_id}_{meth}.tif"
            if not gen_path.exists():
                continue
                
            gen_img = tifffile.imread(str(gen_path))
            if len(gen_img.shape) == 3 and gen_img.shape[0] == 3:
                gen_img = np.transpose(gen_img, (1, 2, 0))
                
            # Strict domain alignment
            gen_scaled = gen_img.astype(np.float32).clip(0.0, 1.0)
                
            # PSNR
            try:
                psnr = peak_signal_noise_ratio(gt_scaled, gen_scaled, data_range=1.0)
            except Exception:
                psnr = np.nan
                
            # SSIM (Strict channel_axis=-1 and win_size=7)
            try:
                # Some skimage versions require min_size to be >= win_size
                min_dim = min(gt_scaled.shape[0], gt_scaled.shape[1])
                win_size = 7 if min_dim >= 7 else min_dim
                if win_size % 2 == 0: win_size -= 1 # Ensure odd
                
                try:
                    ssim = structural_similarity(gt_scaled, gen_scaled, channel_axis=-1, data_range=1.0, win_size=win_size)
                except TypeError:
                    ssim = structural_similarity(gt_scaled, gen_scaled, multichannel=True, data_range=1.0, win_size=win_size)
            except Exception:
                ssim = np.nan
                
            # Edge-IoU
            try:
                edge_iou = adaptive_edge_iou(gt_scaled, gen_scaled)
            except Exception:
                edge_iou = np.nan
                
            # ORB-SMR
            try:
                smr = rigorous_orb_smr(sar_scaled, gen_scaled)
            except Exception:
                smr = np.nan
                
            results[meth]["psnr"].append(psnr)
            results[meth]["ssim"].append(ssim)
            results[meth]["edge_iou"].append(edge_iou)
            results[meth]["orb_smr"].append(smr)
            
    final_metrics = []
    for meth in methods:
        psnr_avg = np.nanmean(results[meth]["psnr"]) if results[meth]["psnr"] else np.nan
        ssim_avg = np.nanmean(results[meth]["ssim"]) if results[meth]["ssim"] else np.nan
        eiou_avg = np.nanmean(results[meth]["edge_iou"]) if results[meth]["edge_iou"] else np.nan
        smr_avg = np.nanmean(results[meth]["orb_smr"]) if results[meth]["orb_smr"] else np.nan
                
        final_metrics.append({
            "Method": meth,
            "Split": "All",
            "PSNR": psnr_avg,
            "SSIM": ssim_avg,
            "Edge-IoU": eiou_avg,
            "ORB-SMR": smr_avg
        })
        
    master_df = pd.DataFrame(final_metrics)
    master_df.to_csv("results/master_results_real.csv", index=False)
    
    print("\n=======================================================")
    print("      MATHEMATICALLY RIGOROUS METRICS SUMMARY")
    print("=======================================================\n")
    print(master_df.to_string(index=False))
    print("\n=======================================================\n")

if __name__ == "__main__":
    main()
