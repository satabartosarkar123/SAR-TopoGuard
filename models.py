#!/usr/bin/env python3
"""
Module 3 — RTX 3050-Optimised Generator & Discriminator
========================================================

Architecture Summary
--------------------

**Generator (U-Net with skip connections)**

    Input : (B, SAR_CHANNELS, 256, 256)       — SAR VV/VH
    Output: (B, OPTICAL_CHANNELS, 256, 256)    — predicted RGB optical

    Encoder: 8 down-sampling blocks
    Decoder: 8 up-sampling blocks with skip connections from encoder
    Norm  : InstanceNorm2d (NOT BatchNorm — prevents batch-size artifacts
            and uses less memory because no running statistics are stored)
    Acts  : LeakyReLU(0.2) in encoder, ReLU in decoder

**Discriminator (70×70 PatchGAN)**

    Input : (B, SAR_CHANNELS + OPTICAL_CHANNELS, 256, 256)  — concatenated pair
    Output: (B, 1, 30, 30)   — 30×30 grid of real/fake predictions
                               (each cell's receptive field ≈ 70×70)

    4 convolutional layers, InstanceNorm on layers 2-3, sigmoid-free
    (we use BCEWithLogitsLoss → raw logits).

VRAM Notes
~~~~~~~~~~
•  InstanceNorm2d stores no running mean/var → saves ~2× norm memory.
•  No attention layers → O(HW) memory, not O(H²W²).
•  With batch_size=4 @ 256² + AMP, peak VRAM ≈ 2.5 GB.

Author : SAR-TopoGuard team
Date   : Day 1 — Evening
"""

from __future__ import annotations

import logging
from typing import Union

import torch
import torch.nn as nn

# ── Local imports ──────────────────────────────────────────────────────
from config import SAR_CHANNELS, OPTICAL_CHANNELS

logger = logging.getLogger(__name__)


# ======================================================================
# 1.  Weight Initialisation
# ======================================================================

def weights_init(m: nn.Module) -> None:
    """
    Initialise Conv2d and InstanceNorm2d parameters from N(0, 0.02).

    This is the standard pix2pix / DCGAN initialisation scheme.
    InstanceNorm2d has *affine* weight and bias when ``affine=True``
    (our default).
    """
    classname = m.__class__.__name__
    if "Conv" in classname and hasattr(m, "weight"):
        nn.init.normal_(m.weight.data, mean=0.0, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)
    elif "InstanceNorm" in classname and m.affine:
        nn.init.normal_(m.weight.data, mean=1.0, std=0.02)
        nn.init.constant_(m.bias.data, 0.0)


# ======================================================================
# 2.  Building blocks
# ======================================================================

class UNetDownBlock(nn.Module):
    """
    Encoder block: Conv → InstanceNorm → LeakyReLU.

    Down-samples spatial dimensions by 2 (stride=2).
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        use_norm: bool = True,
    ) -> None:
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False),
        ]
        if use_norm:
            layers.append(nn.InstanceNorm2d(out_ch, affine=True))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNetUpBlock(nn.Module):
    """
    Decoder block: ConvTranspose → InstanceNorm → ReLU (+ optional Dropout).

    Up-samples spatial dimensions by 2 (stride=2).
    Skip connections are handled externally via ``torch.cat``.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        use_dropout: bool = False,
    ) -> None:
        super().__init__()
        layers = [
            nn.ConvTranspose2d(
                in_ch, out_ch,
                kernel_size=4, stride=2, padding=1, bias=False,
            ),
            nn.InstanceNorm2d(out_ch, affine=True),
            nn.ReLU(inplace=True),
        ]
        if use_dropout:
            layers.append(nn.Dropout(0.5))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ======================================================================
# 3.  Generator (U-Net)
# ======================================================================

class UNetGenerator(nn.Module):
    """
    U-Net generator optimised for 256×256 inputs.

    Architecture (filter counts mirror the original pix2pix paper)::

        Encoder                    Decoder (+ skip from encoder)
        ─────────────────          ──────────────────────────────
        e1: SAR_CH  → 64          d8: 512       → 512  (dropout)
        e2: 64      → 128         d7: 512+512   → 512  (dropout)
        e3: 128     → 256         d6: 512+512   → 512  (dropout)
        e4: 256     → 512         d5: 512+512   → 512
        e5: 512     → 512         d4: 512+512   → 256
        e6: 512     → 512         d3: 256+256   → 128
        e7: 512     → 512         d2: 128+128   → 64
        e8: 512     → 512 (btlnk) final: 64+64  → OPTICAL_CH, Tanh

    The Tanh output is rescaled to [0, 1] via ``(tanh + 1) / 2``
    so that the output directly matches our [0, 1]-normalised
    optical targets.
    """

    def __init__(
        self,
        in_channels: int = SAR_CHANNELS,
        out_channels: int = OPTICAL_CHANNELS,
    ) -> None:
        super().__init__()

        # ── Encoder ──────────────────────────────────────────────────
        # e1 has no InstanceNorm (standard pix2pix convention)
        self.e1 = UNetDownBlock(in_channels, 64, use_norm=False)   # 256→128
        self.e2 = UNetDownBlock(64,  128)                           # 128→64
        self.e3 = UNetDownBlock(128, 256)                           # 64→32
        self.e4 = UNetDownBlock(256, 512)                           # 32→16
        self.e5 = UNetDownBlock(512, 512)                           # 16→8
        self.e6 = UNetDownBlock(512, 512)                           # 8→4
        self.e7 = UNetDownBlock(512, 512)                           # 4→2

        # Bottleneck (no norm, no skip connection below)
        self.e8 = UNetDownBlock(512, 512, use_norm=False)           # 2→1

        # ── Decoder ──────────────────────────────────────────────────
        # in_ch is doubled because of skip-connection concatenation
        self.d8 = UNetUpBlock(512,       512, use_dropout=True)     # 1→2
        self.d7 = UNetUpBlock(512 + 512, 512, use_dropout=True)     # 2→4
        self.d6 = UNetUpBlock(512 + 512, 512, use_dropout=True)     # 4→8
        self.d5 = UNetUpBlock(512 + 512, 512)                       # 8→16
        self.d4 = UNetUpBlock(512 + 512, 256)                       # 16→32
        self.d3 = UNetUpBlock(256 + 256, 128)                       # 32→64
        self.d2 = UNetUpBlock(128 + 128, 64)                        # 64→128

        # Final up-sample: 128→256
        self.final = nn.Sequential(
            nn.ConvTranspose2d(
                64 + 64, out_channels,
                kernel_size=4, stride=2, padding=1,
            ),
            nn.Tanh(),
        )

    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor of shape ``(B, SAR_CHANNELS, 256, 256)``

        Returns
        -------
        Tensor of shape ``(B, OPTICAL_CHANNELS, 256, 256)`` in [0, 1].
        """
        # ── Encoder forward ─────────────────────────────────────────
        enc1 = self.e1(x)        # (B,  64, 128, 128)
        enc2 = self.e2(enc1)     # (B, 128,  64,  64)
        enc3 = self.e3(enc2)     # (B, 256,  32,  32)
        enc4 = self.e4(enc3)     # (B, 512,  16,  16)
        enc5 = self.e5(enc4)     # (B, 512,   8,   8)
        enc6 = self.e6(enc5)     # (B, 512,   4,   4)
        enc7 = self.e7(enc6)     # (B, 512,   2,   2)
        enc8 = self.e8(enc7)     # (B, 512,   1,   1) — bottleneck

        # ── Decoder forward (skip connections via cat) ───────────────
        dec8 = self.d8(enc8)                                # (B, 512,  2,  2)
        dec7 = self.d7(torch.cat([dec8, enc7], dim=1))      # (B, 512,  4,  4)
        dec6 = self.d6(torch.cat([dec7, enc6], dim=1))      # (B, 512,  8,  8)
        dec5 = self.d5(torch.cat([dec6, enc5], dim=1))      # (B, 512, 16, 16)
        dec4 = self.d4(torch.cat([dec5, enc4], dim=1))      # (B, 256, 32, 32)
        dec3 = self.d3(torch.cat([dec4, enc3], dim=1))      # (B, 128, 64, 64)
        dec2 = self.d2(torch.cat([dec3, enc2], dim=1))      # (B,  64, 128,128)

        out = self.final(torch.cat([dec2, enc1], dim=1))    # (B,  OC, 256,256)

        # Tanh → [-1, 1]; rescale to [0, 1]
        return (out + 1.0) * 0.5


# ======================================================================
# 4.  Discriminator (70×70 PatchGAN)
# ======================================================================

class PatchGANDiscriminator(nn.Module):
    """
    70×70 PatchGAN discriminator (Isola et al., 2017).

    Takes a concatenated (SAR, Optical) pair and outputs a grid of
    real/fake logits.  Each spatial cell's receptive field covers a
    ~70×70 patch of the input.

    Architecture::

        Layer   In          Out     Kernel  Stride  Norm
        ─────   ──          ───     ──────  ──────  ────
        c1      SAR+OPT     64      4×4     2       —
        c2      64          128     4×4     2       InstanceNorm
        c3      128         256     4×4     2       InstanceNorm
        c4      256         512     4×4     1       InstanceNorm
        out     512         1       4×4     1       —

    For 256×256 input: output shape ≈ (B, 1, 30, 30).

    No sigmoid at the end — we use BCEWithLogitsLoss.
    """

    def __init__(
        self,
        in_channels: int = SAR_CHANNELS + OPTICAL_CHANNELS,
    ) -> None:
        super().__init__()

        def _disc_block(
            in_ch: int, out_ch: int, stride: int, use_norm: bool,
        ) -> nn.Sequential:
            layers = [
                nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=stride, padding=1, bias=False),
            ]
            if use_norm:
                layers.append(nn.InstanceNorm2d(out_ch, affine=True))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)

        self.model = nn.Sequential(
            _disc_block(in_channels, 64,  stride=2, use_norm=False),  # 256→128
            _disc_block(64,         128, stride=2, use_norm=True),    # 128→64
            _disc_block(128,        256, stride=2, use_norm=True),    # 64→32
            _disc_block(256,        512, stride=1, use_norm=True),    # 32→31
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1),   # 31→30
            # No sigmoid — raw logits for BCEWithLogitsLoss
        )

    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor of shape ``(B, SAR_CH + OPT_CH, 256, 256)``
            Concatenation of (SAR input, Optical candidate) along C.

        Returns
        -------
        Tensor of shape ``(B, 1, 30, 30)`` — raw logits.
        """
        return self.model(x)


# ======================================================================
# 5.  Factory helpers
# ======================================================================

def build_generator(device: Union[torch.device, str] = "cpu") -> UNetGenerator:
    """Instantiate, initialise, and move the generator to *device*."""
    gen = UNetGenerator().to(device)
    gen.apply(weights_init)
    logger.info(
        "Generator created — %.2f M params",
        sum(p.numel() for p in gen.parameters()) / 1e6,
    )
    return gen


def build_discriminator(device: Union[torch.device, str] = "cpu") -> PatchGANDiscriminator:
    """Instantiate, initialise, and move the discriminator to *device*."""
    disc = PatchGANDiscriminator().to(device)
    disc.apply(weights_init)
    logger.info(
        "Discriminator created — %.2f M params",
        sum(p.numel() for p in disc.parameters()) / 1e6,
    )
    return disc


# ======================================================================
# 6.  Smoke test
# ======================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    gen = build_generator(device)
    disc = build_discriminator(device)

    # Synthetic forward pass — proves shapes are correct
    batch = 2
    sar_dummy = torch.randn(batch, SAR_CHANNELS, 256, 256, device=device)
    opt_fake = gen(sar_dummy)
    print(f"Generator  input : {sar_dummy.shape}")
    print(f"Generator  output: {opt_fake.shape}  "
          f"range=[{opt_fake.min().item():.3f}, {opt_fake.max().item():.3f}]")

    disc_input = torch.cat([sar_dummy, opt_fake], dim=1)
    disc_out = disc(disc_input)
    print(f"Discriminator input : {disc_input.shape}")
    print(f"Discriminator output: {disc_out.shape}")

    # Clean up VRAM
    del sar_dummy, opt_fake, disc_input, disc_out
    torch.cuda.empty_cache()
    print("[PASS] Smoke test passed - no shape mismatches.")
