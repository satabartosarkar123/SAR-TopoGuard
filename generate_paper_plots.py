import os
import matplotlib.pyplot as plt
import numpy as np

def setup_style():
    # Basic styling for publication quality
    plt.style.use('seaborn-v0_8-paper')
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight'
    })

def create_structural_fidelity_plot(methods, edge_iou, orb_smr, out_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Structural Fidelity & Downstream Matching Utility", y=1.05)
    
    # Colors
    colors = ['#8c8c8c', '#5e81ac', '#d08770'] # Gray, Blue, Orange (TopoGuard)
    
    # Subplot 1: Edge-IoU
    bars1 = ax1.bar(methods, edge_iou, color=colors, edgecolor='black')
    ax1.set_ylabel("Edge-IoU (Higher is better)")
    ax1.set_title("Edge Preservation")
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    # Add values on top
    for bar in bars1:
        height = bar.get_height()
        ax1.annotate(f'{height:.4f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

    # Subplot 2: ORB-SMR
    bars2 = ax2.bar(methods, orb_smr, color=colors, edgecolor='black')
    ax2.set_ylabel("ORB-SMR (Higher is better)")
    ax2.set_title("ORB Feature Matching")
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    # Add values on top
    for bar in bars2:
        height = bar.get_height()
        ax2.annotate(f'{height:.6f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')

    # Rotate x labels
    for ax in [ax1, ax2]:
        plt.sca(ax)
        plt.xticks(rotation=15, ha='right')

    plt.tight_layout()
    
    # Save
    fig.savefig(os.path.join(out_dir, "fig1_structural_fidelity.pdf"))
    fig.savefig(os.path.join(out_dir, "fig1_structural_fidelity.png"))
    plt.close()

def create_tradeoff_plot(methods, psnr, edge_iou, out_dir):
    fig, ax = plt.subplots(figsize=(8, 6))
    
    colors = ['#8c8c8c', '#5e81ac', '#d08770']
    markers = ['o', 's', '^']
    
    for i, method in enumerate(methods):
        ax.scatter(psnr[i], edge_iou[i], 
                   color=colors[i], marker=markers[i], s=200, 
                   label=method, edgecolor='black', zorder=5)
        
        # Add labels slightly offset
        offset_y = 0.005 if i != 1 else -0.008
        ax.annotate(method, (psnr[i], edge_iou[i]), 
                   xytext=(10, 10 if i!=1 else -15), textcoords='offset points',
                   fontsize=11, fontweight='bold' if 'TopoGuard' in method else 'normal')
        
    ax.set_title("Structure-Perception Trade-off")
    ax.set_xlabel("PSNR (Perceptual Quality)")
    ax.set_ylabel("Edge-IoU (Structural Fidelity)")
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Add a dashed trend arrow pointing top right
    ax.annotate("", xy=(12.65, 0.22), xytext=(11.9, 0.15),
                arrowprops=dict(arrowstyle="->", color="black", ls="dashed", lw=1.5, alpha=0.5))
    ax.text(12.2, 0.18, "Ideal Region", rotation=35, alpha=0.7, fontweight='bold')

    # Expand limits for padding
    ax.set_xlim(min(psnr)-0.2, max(psnr)+0.2)
    ax.set_ylim(min(edge_iou)-0.02, max(edge_iou)+0.03)

    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig2_tradeoff.pdf"))
    fig.savefig(os.path.join(out_dir, "fig2_tradeoff.png"))
    plt.close()

def main():
    # Data
    methods = ['Vanilla Pix2Pix', 'Pix2Pix + Edge Loss', 'SAR-TopoGuard (Ours)']
    edge_iou = [0.1543, 0.2072, 0.2061]
    orb_smr = [0.000311, 0.003697, 0.004116]
    psnr = [11.97, 12.60, 12.58]
    
    out_dir = "results/figures"
    os.makedirs(out_dir, exist_ok=True)
    
    setup_style()
    create_structural_fidelity_plot(methods, edge_iou, orb_smr, out_dir)
    create_tradeoff_plot(methods, psnr, edge_iou, out_dir)
    
    print("Figures generated successfully in results/figures/!")

if __name__ == "__main__":
    main()
