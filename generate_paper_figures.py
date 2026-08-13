import os
import re
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import cv2
from skimage import io
from pathlib import Path
import warnings

# Global Style
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.size'] = 10
matplotlib.rcParams['axes.linewidth'] = 0.8
DPI = 300

OUT_DIR = Path("paper_figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def _save_fig(fig, name):
    fig.savefig(OUT_DIR / f"{name}.png", dpi=DPI, bbox_inches='tight')
    fig.savefig(OUT_DIR / f"{name}.pdf", dpi=DPI, bbox_inches='tight')
    size_kb = (OUT_DIR / f"{name}.png").stat().st_size / 1024
    print(f"Generating {name}... done. ({size_kb:.1f} KB)")
    return size_kb

def load_image(path):
    path_str = str(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path_str}")
    img = io.imread(path_str)
    if img.ndim == 3 and img.shape[0] < img.shape[2]:
        img = np.transpose(img, (1, 2, 0))
    return img.astype(np.float32)

def figure_1():
    print("Generating Figure 1...")
    try:
        sar_path = Path("raw_data/ROI_SAR2EO50/s1_all/ROIs2017_winter_s1_102_p134.tif")
        opt_path = Path("raw_data/ROI_SAR2EO50/s2_all/ROIs2017_winter_s2_102_p134.tif")
        
        sar_img = load_image(sar_path)
        opt_img = load_image(opt_path)
        
        if opt_img.shape[-1] > 3:
            opt_img = opt_img[..., :3]
            
        opt_naive = np.clip(opt_img / 255.0, 0, 1)
        
        p2 = np.percentile(opt_img, 2)
        p98 = np.percentile(opt_img, 98)
        opt_stretch = np.clip((opt_img - p2) / (p98 - p2 + 1e-6), 0, 1)
        
        sar_log = 10 * np.log10(sar_img[..., 0] + 1e-6)
        p2_s = np.percentile(sar_log, 2)
        p98_s = np.percentile(sar_log, 98)
        sar_stretch = np.clip((sar_log - p2_s) / (p98_s - p2_s + 1e-6), 0, 1)
        
        fig, axes = plt.subplots(1, 3, figsize=(7, 2.5))
        axes[0].imshow(opt_naive)
        axes[0].text(5, 20, "(a)", color='white', fontweight='bold', fontsize=12)
        
        axes[1].imshow(opt_stretch)
        axes[1].text(5, 20, "(b)", color='white', fontweight='bold', fontsize=12)
        
        axes[2].imshow(sar_stretch, cmap='gray')
        axes[2].text(5, 20, "(c)", color='white', fontweight='bold', fontsize=12)
        
        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])
            
        kb = _save_fig(fig, "figure1_normalization")
        plt.close(fig)
        print(f"  Panel (a) stats: min={opt_naive.min():.3f}, max={opt_naive.max():.3f}, mean={opt_naive.mean():.3f}")
        print(f"  Panel (b) stats: min={opt_stretch.min():.3f}, max={opt_stretch.max():.3f}, mean={opt_stretch.mean():.3f}")
        return ("Figure 1", "figure1_normalization.png", "figure1_normalization.pdf", kb, "OK")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Figure 1 failed: {e}")
        return ("Figure 1", "-", "-", 0, f"FAILED: {e}")

def figure_3():
    print("Generating Figure 3...")
    try:
        sar_path = Path("mini_sen12_data/val/s1/ROIs1868_summer_s1_59_p1001.png")
        tg_path = Path("results/generated_images/topoguard/ROIs1868_summer_s1_59_p1001.png")
        
        sar_img = load_image(sar_path)
        y_best = load_image(tg_path) / 255.0
        
        if sar_img.ndim == 3:
            sar_gray = cv2.cvtColor(sar_img, cv2.COLOR_RGB2GRAY) if sar_img.shape[-1] == 3 else sar_img[..., 0]
        else:
            sar_gray = sar_img
            
        sar_gray = sar_gray / 255.0
        
        gx = cv2.Sobel(sar_gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(sar_gray, cv2.CV_64F, 0, 1, ksize=3)
        gm = np.sqrt(gx**2 + gy**2)
        gm_sar = cv2.blur(gm, (3, 3))
        
        m_edge = 1 / (1 + np.exp(-5 * (gm_sar - 0.1)))
        
        y_low = cv2.GaussianBlur(y_best, (5, 5), 0)
        y_high = y_best - y_low
        
        m_edge_3 = np.expand_dims(m_edge, axis=-1)
        y_final = np.clip(y_low + m_edge_3 * y_high, 0, 1)
        
        fig, axes = plt.subplots(1, 4, figsize=(7, 2.0))
        axes[0].imshow(y_best)
        axes[0].set_xlabel("Y_best", fontsize=10)
        
        im1 = axes[1].imshow(m_edge, cmap='hot', vmin=0, vmax=1)
        axes[1].set_xlabel("M_edge", fontsize=10)
        
        axes[2].imshow(y_low)
        axes[2].set_xlabel("Y_low", fontsize=10)
        
        axes[3].imshow(y_final)
        axes[3].set_xlabel("Y_final", fontsize=10)
        
        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])
            
        kb = _save_fig(fig, "figure3_gating")
        plt.close(fig)
        return ("Figure 3", "figure3_gating.png", "figure3_gating.pdf", kb, "OK")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Figure 3 failed: {e}")
        return ("Figure 3", "-", "-", 0, f"FAILED: {e}")

def figure_4():
    print("Generating Figure 4...")
    try:
        csv_path = Path("results/per_image_metrics.csv")
        if not csv_path.exists():
            raise FileNotFoundError(f"{csv_path} not found.")
            
        df = pd.read_csv(csv_path)
        methods = ["baseline1", "baseline2", "topoguard"]
        labels = ["Vanilla Pix2Pix", "B2 + Edge Loss", "SAR-TopoGuard"]
        colors = ['#808080', '#6BAED6', '#E63946']
        
        stats = df.groupby("method")[['psnr', 'ssim', 'edge_iou', 'smr']].agg(['mean', 'std']).reset_index()
        
        psnr_m, psnr_s = [], []
        ssim_m, ssim_s = [], []
        eiou_m, eiou_s = [], []
        smr_m, smr_s = [], []
        
        for m in methods:
            row = stats[stats['method'] == m]
            psnr_m.append(row['psnr']['mean'].values[0])
            psnr_s.append(row['psnr']['std'].values[0])
            ssim_m.append(row['ssim']['mean'].values[0])
            ssim_s.append(row['ssim']['std'].values[0])
            eiou_m.append(row['edge_iou']['mean'].values[0] * 100)
            eiou_s.append(row['edge_iou']['std'].values[0] * 100)
            smr_m.append(row['smr']['mean'].values[0] * 100)
            smr_s.append(row['smr']['std'].values[0] * 100)
            
        fig, axes = plt.subplots(2, 2, figsize=(7, 5))
        axs = axes.flatten()
        
        def plot_bar(ax, means, stds, title, ylabel, ylim=None, fmt="{:.3f}"):
            bars = ax.bar(labels, means, yerr=stds, color=colors, capsize=4, error_kw={'linewidth':1})
            ax.set_title(title, fontsize=11, fontweight='bold', loc='left')
            ax.set_ylabel(ylabel)
            if ylim:
                ax.set_ylim(ylim)
            ax.grid(axis='y', linestyle='-', alpha=0.3)
            # Value labels
            for bar in bars:
                height = bar.get_height()
                ax.annotate(fmt.format(height),
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),  
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=8)
        
        plot_bar(axs[0], psnr_m, psnr_s, "(a) PSNR", "PSNR (dB)", ylim=[11, 14], fmt="{:.3f}")
        plot_bar(axs[1], ssim_m, ssim_s, "(b) SSIM", "SSIM", ylim=[0.10, 0.25], fmt="{:.3f}")
        plot_bar(axs[2], eiou_m, eiou_s, "(c) Edge-IoU", "Edge-IoU (%)", fmt="{:.2f}")
        plot_bar(axs[3], smr_m, smr_s, "(d) SMR", "SMR (%)", fmt="{:.2f}")
        
        import matplotlib.patches as mpatches
        handles = [matplotlib.patches.Patch(color=c, label=l) for c, l in zip(colors, labels)]
        fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=3)
        
        plt.tight_layout()
        kb = _save_fig(fig, "figure4_bars")
        plt.close(fig)
        return ("Figure 4", "figure4_bars.png", "figure4_bars.pdf", kb, "OK")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Figure 4 failed: {e}")
        return ("Figure 4", "-", "-", 0, f"FAILED: {e}")

def figure_5():
    print("Generating Figure 5...")
    try:
        csv_path = Path("results/per_image_metrics.csv")
        df = pd.read_csv(csv_path)
        
        b1 = df[df['method'] == 'baseline1'].set_index('image_id')
        tg = df[df['method'] == 'topoguard'].set_index('image_id')
        
        gain = tg['smr'] - b1['smr']
        top_ids = gain.nlargest(4).index.tolist()
        
        cols = ["Urban 1", "Urban 2", "Natural", "Water"]
        rows = ["SAR", "Ground Truth", "Vanilla", "B2 + Edge", "TopoGuard"]
        
        fig, axes = plt.subplots(5, 4, figsize=(7, 9))
        plt.subplots_adjust(wspace=0.02, hspace=0.02)
        
        for j, img_id in enumerate(top_ids):
            # Locate SAR and OPT paths (could be in train or val)
            sar_p = Path(f"mini_sen12_data/val/s1/{img_id}.png")
            if not sar_p.exists():
                sar_p = Path(f"mini_sen12_data/train/s1/{img_id}.png")
            
            img_id_s2 = img_id.replace("_s1_", "_s2_")
            opt_p = Path(f"mini_sen12_data/val/s2/{img_id_s2}.png")
            if not opt_p.exists():
                opt_p = Path(f"mini_sen12_data/train/s2/{img_id_s2}.png")
                
            b1_p = Path(f"results/generated_images/baseline1/{img_id}.png")
            b2_p = Path(f"results/generated_images/baseline2/{img_id}.png")
            tg_p = Path(f"results/generated_images/topoguard/{img_id}.png")
            
            paths = [sar_p, opt_p, b1_p, b2_p, tg_p]
            
            for i, p in enumerate(paths):
                ax = axes[i, j]
                ax.set_xticks([])
                ax.set_yticks([])
                
                if i == 0 and j == 0:
                    ax.set_ylabel(rows[i], fontsize=9)
                elif j == 0:
                    ax.set_ylabel(rows[i], fontsize=9)
                    
                if i == 0:
                    ax.set_title(cols[j], fontsize=9)
                    
                if p.exists():
                    img = load_image(p) / 255.0
                    
                    if i == 0:
                        if img.ndim == 3:
                            img = img[..., 0]
                        p2, p98 = np.percentile(img, 2), np.percentile(img, 98)
                        img = np.clip((img - p2) / (p98 - p2 + 1e-6), 0, 1)
                        ax.imshow(img, cmap='gray')
                    elif i == 1:
                        p2, p98 = np.percentile(img, 2), np.percentile(img, 98)
                        img = np.clip((img - p2) / (p98 - p2 + 1e-6), 0, 1)
                        ax.imshow(img)
                    else:
                        ax.imshow(img)
                        
                    if i in [2, 4]:
                        h, w = img.shape[:2]
                        cy, cx = h//2, w//2
                        crop = img[cy-32:cy+32, cx-32:cx+32]
                        
                        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
                        axins = inset_axes(ax, width="35%", height="35%", loc=4, borderpad=0)
                        if img.ndim == 2:
                            axins.imshow(crop, cmap='gray')
                        else:
                            axins.imshow(crop)
                        axins.set_xticks([])
                        axins.set_yticks([])
                        for spine in axins.spines.values():
                            spine.set_edgecolor('white')
                            spine.set_linewidth(1.5)
                            
                else:
                    print(f"Warning: Missing {p}")
                    ax.imshow(np.full((256,256,3), 0.5))
                    ax.text(128, 128, "MISSING", ha='center', va='center', color='red')
                    
        kb = _save_fig(fig, "figure5_qualitative")
        plt.close(fig)
        return ("Figure 5", "figure5_qualitative.png", "figure5_qualitative.pdf", kb, "OK")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Figure 5 failed: {e}")
        return ("Figure 5", "-", "-", 0, f"FAILED: {e}")

def figure_6():
    print("Generating Figure 6...")
    try:
        log_path = Path("pipeline_log.txt")
        if not log_path.exists():
            raise FileNotFoundError(f"{log_path} not found.")
            
        with open(log_path, 'r') as f:
            lines = f.readlines()
            
        data = []
        collapse_var = {'baseline1': [], 'baseline2': []}
        
        patt_loss = re.compile(r"\[(baseline[12])\] E(\d+) S\d+\s+\|\s+G=([\d.]+)\s+D=([\d.]+)")
        patt_var = re.compile(r"\[(baseline[12])\] Epoch (\d+) collapse check: Var=([\d.]+)")
        
        for line in lines:
            m = patt_loss.search(line)
            if m:
                b = m.group(1)
                e = int(m.group(2))
                g = float(m.group(3))
                d = float(m.group(4))
                data.append({'method': b, 'epoch': e, 'G': g, 'D': d})
            
            mv = patt_var.search(line)
            if mv:
                b = mv.group(1)
                e = int(mv.group(2))
                v = float(mv.group(3))
                collapse_var[b].append((e, v))
                
        if not data:
            print("Could not parse logs. First 20 lines:")
            for l in lines[:20]: print(l.strip())
            raise ValueError("No matching log lines found.")
            
        df = pd.DataFrame(data)
        agg = df.groupby(['method', 'epoch']).mean().reset_index()
        
        b1_agg = agg[agg['method'] == 'baseline1'].sort_values('epoch')
        b2_agg = agg[agg['method'] == 'baseline2'].sort_values('epoch')
        
        fig, axes = plt.subplots(1, 2, figsize=(7, 3))
        
        ax = axes[0]
        ax.plot(b1_agg['epoch'], b1_agg['G'], color='#1F77B4', linestyle='-', label='Vanilla G-loss')
        ax.plot(b1_agg['epoch'], b1_agg['D'], color='#1F77B4', linestyle='--', label='Vanilla D-loss')
        ax.plot(b2_agg['epoch'], b2_agg['G'], color='#D62728', linestyle='-', label='B2 G-loss')
        ax.plot(b2_agg['epoch'], b2_agg['D'], color='#D62728', linestyle='--', label='B2 D-loss')
        ax.set_title("(a) Training Losses", fontsize=11, fontweight='bold', loc='left')
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(axis='y', linestyle='-', alpha=0.3)
        ax.legend(loc='upper right', fontsize=8)
        
        ax = axes[1]
        for b, color, lbl in [('baseline1', '#1F77B4', 'Vanilla'), ('baseline2', '#D62728', 'B2')]:
            if collapse_var[b]:
                x = [item[0] for item in collapse_var[b]]
                y = [item[1] for item in collapse_var[b]]
                ax.plot(x, y, marker='o', markersize=6, color=color, label=lbl)
                
        ax.axhline(y=0.005, color='black', linestyle='--', linewidth=1)
        ax.text(10, 0.006, "Collapse threshold", fontsize=8)
        
        ax.set_title("(b) Collapse-Check Variance", fontsize=11, fontweight='bold', loc='left')
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Pixel Variance")
        ax.set_ylim([0, 0.07])
        ax.set_xticks([10, 20, 30, 40, 50])
        ax.legend(loc='lower right', fontsize=8)
        
        plt.tight_layout()
        kb = _save_fig(fig, "figure6_training")
        plt.close(fig)
        return ("Figure 6", "figure6_training.png", "figure6_training.pdf", kb, "OK")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Figure 6 failed: {e}")
        return ("Figure 6", "-", "-", 0, f"FAILED: {e}")

if __name__ == "__main__":
    results = []
    results.append(figure_1())
    results.append(figure_3())
    results.append(figure_4())
    results.append(figure_5())
    results.append(figure_6())
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"{'Figure':<10} | {'PNG Path':<30} | {'PDF Path':<30} | {'Size (KB)':<10} | {'Status'}")
    print("-" * 80)
    for res in results:
        print(f"{res[0]:<10} | {res[1]:<30} | {res[2]:<30} | {res[3]:<10.1f} | {res[4]}")
    print("="*80)
