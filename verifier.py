#!/usr/bin/env python3
"""
Module 5 — Speckle-Aware Topological Verifier (Math Core)
==========================================================

Pure PyTorch tensor operations implementing three inference-time
quality metrics.  **No OpenCV** — everything is differentiable and
traceable.

Metric Definitions
------------------

1.  **Robust Gradient Magnitude (GM)**

    Suppress SAR speckle with a 3×3 average pool, then compute Sobel
    gradient magnitude::

        P(I) = AvgPool₃ₓ₃(I)
        G(I) = sqrt( (Kx * P(I))² + (Ky * P(I))² + ε )

    where ε = 1e-8 for numerical stability.

2.  **Clamped Hallucination Frequency Index (HFI)**

    Detect phantom high-frequency detail that the generator
    "hallucinates" in structurally flat SAR regions::

        H(Y) = |Lap * Y|                                    (high-freq of optical)
        M_flat(X) = exp( −10.0 · clamp(G(X), 0, 0.3) )     (flatness mask from SAR)
        HFI = mean( H(Y) ⊙ M_flat(X) )

    A *high* HFI → the model is injecting edges where the SAR input
    has no structural evidence → hallucination.

3.  **Pooled Gradient Magnitude Correlation (GMC)**

    Measures structural alignment between SAR and Optical gradient
    fields::

        GMC = cosine_similarity( pool(G(X))_flat , pool(G(Y))_flat )

    A GMC close to 1.0 → the optical output faithfully reproduces
    the SAR structure.

Author : SAR-TopoGuard team
Date   : Day 2 — Evening
"""

from __future__ import annotations

import logging
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Local imports ──────────────────────────────────────────────────────
from config import (
    VERIFIER_EPSILON,
    VERIFIER_FLATNESS_GAIN,
    VERIFIER_GM_CLAMP_MAX,
)

logger = logging.getLogger(__name__)


class TopologicalVerifier(nn.Module):
    """
    Inference-time topological verifier for SAR → Optical translation.

    All kernels (Sobel, Laplacian) are registered as **non-learnable
    buffers** and follow the exact LaTeX formulations from the paper.

    Parameters
    ----------
    epsilon : float
        Small constant for numerical stability in sqrt.
    flatness_gain : float
        Exponential decay rate for the flatness mask (default 10.0).
    gm_clamp_max : float
        Upper clamp for gradient magnitude inside the flatness mask
        (default 0.3).
    """

    def __init__(
        self,
        epsilon: float = VERIFIER_EPSILON,
        flatness_gain: float = VERIFIER_FLATNESS_GAIN,
        gm_clamp_max: float = VERIFIER_GM_CLAMP_MAX,
    ) -> None:
        super().__init__()
        self.epsilon = epsilon
        self.flatness_gain = flatness_gain
        self.gm_clamp_max = gm_clamp_max

        # ──────────────────────────────────────────────────────────────
        #  Sobel kernels  (1, 1, 3, 3)
        # ──────────────────────────────────────────────────────────────
        # Kx detects horizontal gradients (vertical edges)
        sobel_kx = torch.tensor(
            [[-1.0, 0.0, 1.0],
             [-2.0, 0.0, 2.0],
             [-1.0, 0.0, 1.0]],
            dtype=torch.float32,
        ).unsqueeze(0).unsqueeze(0)

        # Ky detects vertical gradients (horizontal edges)
        sobel_ky = torch.tensor(
            [[-1.0, -2.0, -1.0],
             [ 0.0,  0.0,  0.0],
             [ 1.0,  2.0,  1.0]],
            dtype=torch.float32,
        ).unsqueeze(0).unsqueeze(0)

        self.register_buffer("sobel_kx", sobel_kx)
        self.register_buffer("sobel_ky", sobel_ky)

        # ──────────────────────────────────────────────────────────────
        #  3×3 Laplacian kernel  (1, 1, 3, 3)
        #
        #  Standard Laplacian with 8-connectivity:
        #      [[ 1,  1,  1],
        #       [ 1, -8,  1],
        #       [ 1,  1,  1]]
        #
        #  |Lap * Y| captures high-frequency oscillations (textures,
        #  edges, noise) — exactly the signal we want to flag as
        #  potential hallucinations.
        # ──────────────────────────────────────────────────────────────
        laplacian = torch.tensor(
            [[ 1.0,  1.0,  1.0],
             [ 1.0, -8.0,  1.0],
             [ 1.0,  1.0,  1.0]],
            dtype=torch.float32,
        ).unsqueeze(0).unsqueeze(0)

        self.register_buffer("laplacian", laplacian)

        # ──────────────────────────────────────────────────────────────
        #  3×3 Average pooling kernel (for speckle suppression)
        #  Implemented as F.avg_pool2d for efficiency.
        # ──────────────────────────────────────────────────────────────

    # ================================================================== #
    #  1.  Robust Gradient Magnitude  G(I)
    # ================================================================== #

    def gradient_magnitude(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the Robust Gradient Magnitude.

        .. math::

            \\mathcal{P}(I) = \\text{AvgPool}_{3 \\times 3}(I)

            G(I) = \\sqrt{(K_x * \\mathcal{P}(I))^2 +
                          (K_y * \\mathcal{P}(I))^2 + \\varepsilon}

        Parameters
        ----------
        x : (B, C, H, W)
            Input structural map (SAR or Optical).  Multi-channel inputs
            are processed channel-by-channel.

        Returns
        -------
        (B, C, H, W) — gradient magnitude map.
        """
        B, C, H, W = x.shape

        # ── Step 1: Speckle suppression via 3×3 average pooling ──────
        # F.avg_pool2d with padding=1, count_include_pad=False to
        # handle borders correctly.
        pooled = F.avg_pool2d(
            x, kernel_size=3, stride=1, padding=1,
            count_include_pad=False,
        )

        # ── Step 2: Sobel convolution (per channel) ──────────────────
        # Reshape to (B*C, 1, H, W) for single-channel conv
        pooled_flat = pooled.reshape(B * C, 1, H, W)

        gx = F.conv2d(pooled_flat, self.sobel_kx, padding=1)  # (B*C,1,H,W)
        gy = F.conv2d(pooled_flat, self.sobel_ky, padding=1)   # (B*C,1,H,W)

        # ── Step 3: Gradient magnitude with ε ────────────────────────
        gm = torch.sqrt(gx ** 2 + gy ** 2 + self.epsilon)

        return gm.reshape(B, C, H, W)

    # ================================================================== #
    #  2.  Clamped Hallucination Frequency Index  (HFI)
    # ================================================================== #

    def hallucination_frequency_index(
        self,
        sar: torch.Tensor,
        optical: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the Clamped HFI.

        .. math::

            H(Y) = |\\text{Lap} * Y|

            M_{\\text{flat}}(X) = \\exp\\bigl(
                -\\gamma \\cdot \\text{clamp}(G(X), 0, c_{\\max})
            \\bigr)

            \\text{HFI} = \\text{mean}\\bigl( H(Y) \\odot M_{\\text{flat}}(X) \\bigr)

        where γ = 10.0, c_max = 0.3.

        Interpretation
        ~~~~~~~~~~~~~~
        •  H(Y) — high-frequency content in the *optical* output.
        •  M_flat(X) — mask that is ≈ 1.0 in structurally flat SAR
           regions and ≈ 0.0 near edges.
        •  HFI — expected high-frequency energy in flat SAR regions.
           High value → hallucination.

        Parameters
        ----------
        sar     : (B, C_sar, H, W)   — SAR input.
        optical : (B, C_opt, H, W)   — generated (or real) optical.

        Returns
        -------
        Scalar tensor (batch-averaged HFI).
        """
        # ── Optical high-frequency extraction ────────────────────────
        # Process each channel independently
        B_opt, C_opt, H, W = optical.shape
        opt_flat = optical.reshape(B_opt * C_opt, 1, H, W)
        high_freq = torch.abs(
            F.conv2d(opt_flat, self.laplacian, padding=1)
        ).reshape(B_opt, C_opt, H, W)
        # Average across optical channels → (B, 1, H, W)
        high_freq = high_freq.mean(dim=1, keepdim=True)

        # ── SAR gradient magnitude ───────────────────────────────────
        gm_sar = self.gradient_magnitude(sar)
        # Average across SAR channels → (B, 1, H, W)
        gm_sar = gm_sar.mean(dim=1, keepdim=True)

        # ── Flatness mask ────────────────────────────────────────────
        # M_flat(X) = exp( −γ · clamp(G(X), 0, c_max) )
        #
        # • Near edges:  G(X) ≈ c_max  → M ≈ exp(−γ·c_max) ≈ 0.05  (suppressed)
        # • Flat regions: G(X) ≈ 0     → M ≈ exp(0) = 1.0           (active)
        clamped_gm = torch.clamp(gm_sar, min=0.0, max=self.gm_clamp_max)
        flatness_mask = torch.exp(-self.flatness_gain * clamped_gm)

        # ── HFI = mean( H(Y) ⊙ M_flat(X) ) ─────────────────────────
        hfi = (high_freq * flatness_mask).mean()

        return hfi

    # ================================================================== #
    #  3.  Pooled Gradient Magnitude Correlation  (GMC)
    # ================================================================== #

    def gradient_magnitude_correlation(
        self,
        sar: torch.Tensor,
        optical: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the Pooled GMC.

        .. math::

            \\text{GMC} = \\text{cosine\\_similarity}\\bigl(
                \\text{pool}(G(X))_{\\text{flat}},\\;
                \\text{pool}(G(Y))_{\\text{flat}}
            \\bigr)

        Steps
        ~~~~~
        1. Compute G(X) and G(Y).
        2. Average-pool both to a coarser resolution (reduces noise).
        3. Flatten spatial dims.
        4. Cosine similarity across the spatial (feature) dimension.

        Parameters
        ----------
        sar     : (B, C_sar, H, W)
        optical : (B, C_opt, H, W)

        Returns
        -------
        (B,) — per-sample cosine similarity.
        """
        # ── Gradient magnitudes ──────────────────────────────────────
        gm_sar = self.gradient_magnitude(sar)      # (B, C_sar, H, W)
        gm_opt = self.gradient_magnitude(optical)   # (B, C_opt, H, W)

        # ── Channel averaging → (B, 1, H, W) ────────────────────────
        gm_sar = gm_sar.mean(dim=1, keepdim=True)
        gm_opt = gm_opt.mean(dim=1, keepdim=True)

        # ── 3×3 Average pooling (further denoising) ──────────────────
        gm_sar_pooled = F.avg_pool2d(
            gm_sar, kernel_size=3, stride=1, padding=1,
            count_include_pad=False,
        )
        gm_opt_pooled = F.avg_pool2d(
            gm_opt, kernel_size=3, stride=1, padding=1,
            count_include_pad=False,
        )

        # ── Flatten spatial dimensions ───────────────────────────────
        B = gm_sar_pooled.shape[0]
        flat_sar = gm_sar_pooled.reshape(B, -1)   # (B, H*W)
        flat_opt = gm_opt_pooled.reshape(B, -1)   # (B, H*W)

        # ── Cosine similarity per sample ─────────────────────────────
        gmc = F.cosine_similarity(flat_sar, flat_opt, dim=1)  # (B,)

        return gmc

    # ================================================================== #
    #  Unified forward pass
    # ================================================================== #

    @torch.no_grad()
    def forward(
        self,
        sar: torch.Tensor,
        optical: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute both HFI and GMC for a batch.

        This method is decorated with ``@torch.no_grad()`` because the
        verifier is an inference-only module — no parameters are learned,
        and we do not need to build a computation graph.

        Parameters
        ----------
        sar     : (B, C_sar, H, W)  — SAR input tensors.
        optical : (B, C_opt, H, W)  — optical candidate tensors.

        Returns
        -------
        dict with keys:
            ``"hfi"``  — scalar (batch-averaged hallucination index)
            ``"gmc"``  — (B,) per-sample gradient correlation
        """
        hfi = self.hallucination_frequency_index(sar, optical)
        gmc = self.gradient_magnitude_correlation(sar, optical)
        return {"hfi": hfi, "gmc": gmc}


# ======================================================================
#  Smoke test — shape & NaN validation
# ======================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from config import SAR_CHANNELS, OPTICAL_CHANNELS

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    verifier = TopologicalVerifier().to(device)

    # ── Generate random noise tensors ────────────────────────────────
    B, H, W = 4, 256, 256
    sar_dummy = torch.rand(B, SAR_CHANNELS, H, W, device=device)
    opt_dummy = torch.rand(B, OPTICAL_CHANNELS, H, W, device=device)

    print(f"\nInput shapes:")
    print(f"  SAR     : {sar_dummy.shape}")
    print(f"  Optical : {opt_dummy.shape}")

    # ── Forward pass ─────────────────────────────────────────────────
    results = verifier(sar_dummy, opt_dummy)

    hfi = results["hfi"]
    gmc = results["gmc"]

    print(f"\nOutput shapes & values:")
    print(f"  HFI : shape={hfi.shape}  value={hfi.item():.6f}")
    print(f"  GMC : shape={gmc.shape}  values={gmc.tolist()}")

    # ── NaN check ────────────────────────────────────────────────────
    assert not torch.isnan(hfi), "HFI contains NaN!"
    assert not torch.any(torch.isnan(gmc)), "GMC contains NaN!"

    # ── Range check ──────────────────────────────────────────────────
    assert hfi >= 0.0, f"HFI should be non-negative, got {hfi.item()}"
    assert torch.all(gmc >= -1.0) and torch.all(gmc <= 1.0), \
        f"GMC should be in [-1, 1], got {gmc.tolist()}"

    # ── Gradient magnitude standalone test ───────────────────────────
    gm_sar = verifier.gradient_magnitude(sar_dummy)
    gm_opt = verifier.gradient_magnitude(opt_dummy)
    print(f"\n  GM(SAR) : shape={gm_sar.shape}  "
          f"range=[{gm_sar.min().item():.4f}, {gm_sar.max().item():.4f}]")
    print(f"  GM(Opt) : shape={gm_opt.shape}  "
          f"range=[{gm_opt.min().item():.4f}, {gm_opt.max().item():.4f}]")
    assert not torch.any(torch.isnan(gm_sar)), "GM(SAR) contains NaN!"
    assert not torch.any(torch.isnan(gm_opt)), "GM(Opt) contains NaN!"

    # ── Memory cleanup ───────────────────────────────────────────────
    del sar_dummy, opt_dummy, gm_sar, gm_opt
    torch.cuda.empty_cache()

    print("\n[PASS] All assertions passed - no shape mismatches, no NaNs.")
