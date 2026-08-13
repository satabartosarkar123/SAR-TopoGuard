import torch
import numpy as np
import matplotlib.pyplot as plt
import os

from run_recovery_pipeline import get_loaders
from models import UNetGenerator

def main():
    print("=====================================================================")
    print("PART A: GROUND TRUTH ANALYSIS")
    print("=====================================================================")
    
    _, val_loader = get_loaders()
    
    gt_variances = []
    gt_means = []
    gt_mins = []
    gt_maxs = []
    gt_darks = []
    
    count = 0
    gt_images = []
    sar_images = []
    
    for batch in val_loader:
        sar_batch = batch["sar"]
        opt_batch = batch["optical"] # (B, 3, H, W) normalized to [0, 1] usually
        
        for i in range(opt_batch.shape[0]):
            if count >= 20:
                break
            
            # Use raw optical batch
            opt = opt_batch[i]
            img_np = opt.numpy()
            
            var = np.var(img_np, dtype=np.float64)
            mean = np.mean(img_np, dtype=np.float64)
            min_val = np.min(img_np)
            max_val = np.max(img_np)
            dark_pct = np.mean(img_np < 0.1) * 100.0
            
            gt_variances.append(var)
            gt_means.append(mean)
            gt_mins.append(min_val)
            gt_maxs.append(max_val)
            gt_darks.append(dark_pct)
            
            if count < 3:
                # Store images for visualization (channel last)
                gt_images.append(np.transpose(img_np, (1, 2, 0)))
                # SAR has 2 channels. We can just take the first channel for vis.
                sar_images.append(sar_batch[i, 0].numpy())
                
            print(f"GT Img {count+1:02d}: Var={var:.8f} | Mean={mean:.4f} | Min={min_val:.4f} | Max={max_val:.4f} | Dark Pixels: {dark_pct:.1f}%")
            count += 1
            
        if count >= 20:
            break
            
    avg_var = np.mean(gt_variances)
    avg_mean = np.mean(gt_means)
    print("---------------------------------------------------------------------")
    print("AVERAGE ACROSS 20 GT IMAGES:")
    print(f"Variance: {avg_var:.8f}")
    print(f"Mean Brightness: {avg_mean:.4f}")
    print(f"Avg Dark Pixels: {np.mean(gt_darks):.1f}%")
    
    print("\n=====================================================================")
    print("PART B: GENERATED IMAGE AUTOPSY")
    print("=====================================================================")
    
    netG = UNetGenerator(in_channels=2, out_channels=3).cuda()
    checkpoint_path = "checkpoints/pix2pix_baseline_epoch049.pt"
    
    try:
        ckpt = torch.load(checkpoint_path, map_location="cuda")
        if "G_state_dict" in ckpt:
            netG.load_state_dict(ckpt["G_state_dict"])
        else:
            netG.load_state_dict(ckpt)
        print(f"Loaded checkpoint: {checkpoint_path}")
    except Exception as e:
        print(f"Could not load checkpoint: {e}")
    
    netG.eval()
    
    count = 0
    gen_images = []
    
    with torch.no_grad():
        for batch in val_loader:
            sar_batch = batch["sar"].cuda()
            
            # Models.py actually does the scaling natively: return (out + 1.0) * 0.5
            # To get raw tanh, we must capture it before that line, but we can't easily without editing models.py.
            # We'll just print the output of the model (which is already [0, 1]).
            
            fake_B = netG(sar_batch)
            
            for i in range(fake_B.shape[0]):
                if count >= 5:
                    break
                
                final_out = fake_B[i].cpu().numpy()
                # To reconstruct the raw tanh output:
                raw_out = (final_out * 2.0) - 1.0
                
                print(f"Gen Img {count+1}:")
                print(f"  Raw Tanh  -> Min={np.min(raw_out):.4f} | Max={np.max(raw_out):.4f} | Mean={np.mean(raw_out):.4f} | Var={np.var(raw_out, dtype=np.float64):.8f}")
                print(f"  Denormed  -> Min={np.min(final_out):.4f} | Max={np.max(final_out):.4f} | Mean={np.mean(final_out):.4f} | Var={np.var(final_out, dtype=np.float64):.8f}")
                
                # Saturated?
                sat = np.mean(np.abs(raw_out) > 0.99) * 100
                print(f"  Saturated -> {sat:.1f}% of pixels are stuck at ±1.0")
                
                if count < 3:
                    gen_images.append(np.transpose(final_out, (1, 2, 0)))
                count += 1
                
            if count >= 5:
                break
                
    # Create matplotlib visualization
    os.makedirs("results", exist_ok=True)
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    for i in range(3):
        # Row 1: SAR
        axes[0, i].imshow(sar_images[i], cmap='gray')
        axes[0, i].set_title(f"SAR Input {i+1}")
        axes[0, i].axis('off')
        
        # Row 2: GT
        axes[1, i].imshow(np.clip(gt_images[i], 0, 1))
        axes[1, i].set_title(f"Ground Truth {i+1}")
        axes[1, i].axis('off')
        
        # Row 3: Generated
        axes[2, i].imshow(np.clip(gen_images[i], 0, 1))
        axes[2, i].set_title(f"Generated {i+1}")
        axes[2, i].axis('off')
        
    plt.tight_layout()
    plt.savefig("results/diagnostic_autopsy.png", dpi=150)
    print("\nSaved visualization to results/diagnostic_autopsy.png")
    
    print("\n=====================================================================")
    print("PART D: THE VERDICT")
    print("=====================================================================")
    if avg_var < 0.01:
        print("DATASET ISSUE: Ground truth is dark/low-variance. Model is behaving correctly. Recommend: normalize/brighten targets OR use different subset.")
    elif np.var(gen_images[0]) > 0.0001:
        print("METRIC BUG: Variance was rounding to zero; model is actually learning. Recommend: proceed with training.")
    else:
        print("TRUE COLLAPSE: GT has high variance but generated is flat. Recommend: [specific architecture fix].")

if __name__ == "__main__":
    main()
