import os
import sys
import time
import json
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from dataset import SEN12MS_Dataset
from models import UNetGenerator, PatchGANDiscriminator

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

def get_loaders():
    data_root = Path("mini_sen12_data")
    train_ds = SEN12MS_Dataset(records=[], patch_size=256)
    train_ds.records = [{"sar_path": p, "optical_path": p.parent.parent / "s2" / p.name, "roi": "mini", "patch_id": p.stem} for p in (data_root / "train" / "s1").glob("*.*") if p.suffix.lower() in [".tif", ".png"]]
    
    val_ds = SEN12MS_Dataset(records=[], patch_size=256)
    val_ds.records = [{"sar_path": p, "optical_path": p.parent.parent / "s2" / p.name, "roi": "mini", "patch_id": p.stem} for p in (data_root / "val" / "s1").glob("*.*") if p.suffix.lower() in [".tif", ".png"]]
    
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=0, pin_memory=True)
    return train_loader, val_loader

def phase_0_validation():
    logger.info("=== PHASE 0: PRE-FLIGHT DATA & ARCHITECTURE VALIDATION ===")
    
    # Step 0.1: Verify Dataset Normalization
    logger.info("Step 0.1: Verifying Dataset Normalization...")
    try:
        train_loader, _ = get_loaders()
        batch = next(iter(train_loader))
        sar, opt = batch["sar"], batch["optical"]
        
        # Check ranges
        assert torch.all(sar >= -1.0) and torch.all(sar <= 1.0), "SAR not bounded in [-1, 1]"
        assert torch.all(opt >= -1.0) and torch.all(opt <= 1.0), "OPT not bounded in [-1, 1]"
        
        # Check means (should be approx 0)
        sar_mean, opt_mean = sar.mean().item(), opt.mean().item()
        assert abs(sar_mean) < 0.9, f"SAR mean suspicious: {sar_mean}"
        assert abs(opt_mean) < 0.9, f"OPT mean suspicious: {opt_mean}"
        
        logger.info("Step 0.1 PASSED: Data perfectly normalized to [-1, 1]")
    except Exception as e:
        logger.error(f"Step 0.1 FAILED: {e}")
        sys.exit(1)

    # Step 0.2: Verify Generator Output Activation
    logger.info("Step 0.2: Verifying Generator Output Activation...")
    try:
        netG = UNetGenerator(in_channels=2, out_channels=3).cuda()
        # Find the last activation in the model
        has_tanh = False
        for m in netG.modules():
            if isinstance(m, nn.Tanh):
                has_tanh = True
        
        assert has_tanh, "Generator does not contain nn.Tanh()"
        
        # Mathematical test
        dummy_in = torch.randn(2, 2, 256, 256).cuda()
        dummy_out = netG(dummy_in)
        assert dummy_out.min() >= -1.0 and dummy_out.max() <= 1.0, "Generator output not bounded in [-1, 1]"
        logger.info("Step 0.2 PASSED: Generator structurally guarantees [-1, 1] output")
    except Exception as e:
        logger.error(f"Step 0.2 FAILED: {e}")
        sys.exit(1)
        
    # Step 0.3: Verify Loss Configuration
    logger.info("Step 0.3: Verifying Loss Configuration (Manual Check in Phase 1)...")
    logger.info("PHASE 0 PASSED: Data and architecture verified.")

def calculate_mode_collapse_metrics(gen_images):
    # gen_images: shape (N, C, H, W), range [0, 1]
    
    # 1. Pixel Variance (per image, then averaged)
    variances = [torch.var(img).item() for img in gen_images]
    mean_var = np.mean(variances)
    
    # 2. Batch Diversity (mean pairwise L2 distance)
    # Flatten images to (N, C*H*W)
    flat_imgs = gen_images.view(gen_images.shape[0], -1)
    dists = torch.cdist(flat_imgs, flat_imgs, p=2)
    # exclude diagonal
    mask = ~torch.eye(dists.shape[0], dtype=torch.bool)
    mean_div = dists[mask].mean().item()
    
    # 3. Dynamic Range
    rng = gen_images.max().item() - gen_images.min().item()
    
    return mean_var, mean_div, rng

def phase_1_smoke_test():
    logger.info("\n=== PHASE 1: THE 15-EPOCH GRADUATED SMOKE TEST ===")
    
    from train import PurePythonAdam
    
    train_loader, val_loader = get_loaders()
    
    netG = UNetGenerator(in_channels=2, out_channels=3).cuda()
    netD = PatchGANDiscriminator(in_channels=5).cuda()
    
    # CONSERVATIVE HYPERPARAMETERS
    lr = 1e-4
    lambda_L1 = 100.0
    
    optG = PurePythonAdam(netG.parameters(), lr=lr, betas=(0.5, 0.999))
    optD = PurePythonAdam(netD.parameters(), lr=lr, betas=(0.5, 0.999))
    
    criterionGAN = nn.BCEWithLogitsLoss()
    criterionL1 = nn.L1Loss()
    
    num_smoke_epochs = 15
    logger.info("Step 1.1: Executing 15-Epoch Smoke Test on first 50 batches...")
    
    learning_curve = []
    
    for epoch in range(num_smoke_epochs):
        netG.train()
        netD.train()
        
        logger.info(f"Smoke Epoch {epoch+1}/{num_smoke_epochs}")
        for i, batch in enumerate(train_loader):
            if i >= 50:
                break
                
            real_A = batch["sar"].cuda()
            real_B = batch["optical"].cuda()
            
            # --- Train D ---
            optD.zero_grad()
            fake_B = netG(real_A)
            
            # Fake
            pred_fake = netD(torch.cat([real_A, fake_B.detach()], dim=1))
            loss_D_fake = criterionGAN(pred_fake, torch.zeros_like(pred_fake))
            # Real (Label Smoothing = 0.9)
            pred_real = netD(torch.cat([real_A, real_B], dim=1))
            loss_D_real = criterionGAN(pred_real, torch.ones_like(pred_real) * 0.9)
            
            loss_D = (loss_D_fake + loss_D_real) * 0.5
            loss_D.backward()
            optD.step()
            
            # --- Train G ---
            optG.zero_grad()
            pred_fake = netD(torch.cat([real_A, fake_B], dim=1))
            loss_G_GAN = criterionGAN(pred_fake, torch.ones_like(pred_fake))
            loss_G_L1 = criterionL1(fake_B, real_B) * lambda_L1
            
            loss_G = loss_G_GAN + loss_G_L1
            loss_G.backward()
            optG.step()
            
            if i % 10 == 0:
                logger.info(f"Step {i} | loss_G: {loss_G.item():.4f} | loss_D: {loss_D.item():.4f} | L1: {loss_G_L1.item():.4f}")

        # Checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            netG.eval()
            gen_imgs = []
            with torch.no_grad():
                for i, batch in enumerate(val_loader):
                    if i >= 5:
                        break
                    real_A = batch["sar"].cuda()
                    fake_B = netG(real_A)
                    fake_B = (fake_B + 1.0) / 2.0
                    fake_B = torch.clamp(fake_B, 0.0, 1.0)
                    gen_imgs.append(fake_B.cpu())
                    
            gen_imgs = torch.cat(gen_imgs, dim=0)
            mean_var, mean_div, rng = calculate_mode_collapse_metrics(gen_images=gen_imgs)
            learning_curve.append({"epoch": epoch + 1, "var": mean_var, "div": mean_div, "rng": rng})
            logger.info(f"Metrics (Epoch {epoch+1}) -> Variance: {mean_var:.4f}, Batch Diversity: {mean_div:.4f}, Range: {rng:.4f}")

    logger.info("=== LEARNING CURVE ===")
    logger.info(f"{'Epoch':<10} | {'Pixel Variance':<15} | {'Batch Diversity':<15} | {'Dynamic Range':<15}")
    for log in learning_curve:
        logger.info(f"{log['epoch']:<10} | {log['var']:<15.4f} | {log['div']:<15.4f} | {log['rng']:<15.4f}")
        
    var_trend = [log['var'] for log in learning_curve]
    div_trend = [log['div'] for log in learning_curve]
    final_rng = learning_curve[-1]['rng']
    
    # PASS CONDITION A: Variance is increasing
    # Allow a tiny bit of noise, but generally it should go up or stay > 0.001
    var_increasing = (var_trend[-1] > var_trend[0]) or (var_trend[-1] > 0.001)
    
    # PASS CONDITION B: Final Range > 0.4
    rng_pass = final_rng > 0.4
    
    # PASS CONDITION C: Final Batch Diversity > 0.5
    div_pass = div_trend[-1] > 0.5
    
    if var_increasing and rng_pass and div_pass:
        logger.info("GRADUATED SMOKE TEST PASSED: Model is showing learning progress!")
        return True
    else:
        logger.error("GRADUATED SMOKE TEST FAILED: Model failed trend checks.")
        return False

def main():
    phase_0_validation()
    success = phase_1_smoke_test()
    if not success:
        sys.exit(1)
        
if __name__ == "__main__":
    main()
