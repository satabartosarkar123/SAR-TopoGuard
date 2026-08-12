#!/usr/bin/env python3
"""
Module 4 — Mixed-Precision Training Loop & Edge Loss
=====================================================

This module implements the full Pix2Pix training engine with:

1.  **Baseline 1** — Standard Pix2Pix (Adversarial + L1)
2.  **Baseline 2** — Pix2Pix + Train-Time Edge Loss (Adversarial + L1 + Edge)

Hardware Safety
~~~~~~~~~~~~~~~
•  ``torch.cuda.amp.autocast`` + ``GradScaler`` for FP16 mixed precision.
•  Explicit ``torch.cuda.empty_cache()`` at checkpoint boundaries.
•  ``pin_memory=True`` dataloaders (set up in Module 1).

Loss Equations
~~~~~~~~~~~~~~

**Adversarial (GAN) loss** — standard BCE on logits::

    L_adv = BCE_with_logits(D(x, G(x)), target)

**L1 reconstruction loss**::

    L_L1 = ||G(x) − y||₁

**Train-Time Edge Loss** — Sobel-gradient L1::

    E(I) = sqrt( (Kx * I)² + (Ky * I)² + ε )
    L_edge = || E(G(x)) − E(y) ||₁

where Kx, Ky are 3×3 Sobel kernels and ε = 1e-8.

Combined generator loss::

    L_G = L_adv + λ_L1 · L_L1                       (Baseline 1)
    L_G = L_adv + λ_L1 · L_L1 + λ_edge · L_edge     (Baseline 2)

Author : SAR-TopoGuard team
Date   : Day 2 — Morning / Afternoon
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# ── Local imports ──────────────────────────────────────────────────────
from config import (
    CHECKPOINT_DIR,
    LOG_DIR,
    SAR_CHANNELS,
    OPTICAL_CHANNELS,
    TRAIN_CFG,
)
from models import build_generator, build_discriminator

logger = logging.getLogger(__name__)


# ======================================================================
# 1.  Custom Loss: Train-Time Edge Loss
# ======================================================================

class TrainTimeEdgeLoss(nn.Module):
    """
    Compute the L1 distance between the gradient magnitudes of two
    images, where gradients are extracted via 3×3 Sobel filters applied
    with ``F.conv2d``.

    Mathematical Formulation
    ~~~~~~~~~~~~~~~~~~~~~~~~
    Let Kx, Ky be Sobel kernels::

        Kx = [[-1, 0, 1],      Ky = [[-1, -2, -1],
              [-2, 0, 2],            [ 0,  0,  0],
              [-1, 0, 1]]            [ 1,  2,  1]]

    Gradient magnitude of image I::

        E(I) = sqrt( (Kx * I)² + (Ky * I)² + ε )

    Edge loss::

        L_edge = mean( |E(pred) − E(target)| )

    The Sobel kernels are registered as **non-learnable buffers** so they
    live on the same device as the module without consuming optimiser
    state.
    """

    def __init__(self, epsilon: float = 1e-8) -> None:
        super().__init__()
        self.epsilon = epsilon

        # ── Sobel Kx (horizontal edges) ─────────────────────────────
        kx = torch.tensor(
            [[-1.0, 0.0, 1.0],
             [-2.0, 0.0, 2.0],
             [-1.0, 0.0, 1.0]],
            dtype=torch.float32,
        ).unsqueeze(0).unsqueeze(0)  # (1, 1, 3, 3)

        # ── Sobel Ky (vertical edges) ───────────────────────────────
        ky = torch.tensor(
            [[-1.0, -2.0, -1.0],
             [ 0.0,  0.0,  0.0],
             [ 1.0,  2.0,  1.0]],
            dtype=torch.float32,
        ).unsqueeze(0).unsqueeze(0)  # (1, 1, 3, 3)

        # Register as buffers — they will move with .to(device)
        self.register_buffer("kx", kx)
        self.register_buffer("ky", ky)

    # ------------------------------------------------------------------ #
    def _gradient_magnitude(self, img: torch.Tensor) -> torch.Tensor:
        """
        Compute per-channel gradient magnitude.

        Parameters
        ----------
        img : (B, C, H, W)

        Returns
        -------
        (B, C, H, W) — gradient magnitude for each channel.
        """
        B, C, H, W = img.shape

        # Reshape to (B*C, 1, H, W) so we can apply a single-channel
        # Sobel filter to every channel independently.
        img_flat = img.reshape(B * C, 1, H, W)

        gx = F.conv2d(img_flat, self.kx, padding=1)   # (B*C, 1, H, W)
        gy = F.conv2d(img_flat, self.ky, padding=1)    # (B*C, 1, H, W)

        # Gradient magnitude with ε for numerical stability
        gm = torch.sqrt(gx ** 2 + gy ** 2 + self.epsilon)

        return gm.reshape(B, C, H, W)

    # ------------------------------------------------------------------ #
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        pred, target : (B, C, H, W) — predicted & ground-truth optical.

        Returns
        -------
        Scalar L1 distance between gradient magnitudes.
        """
        return F.l1_loss(
            self._gradient_magnitude(pred),
            self._gradient_magnitude(target),
        )


# ======================================================================
# 2.  Training engine
# ======================================================================

class Pix2PixTrainer:
    """
    Encapsulates the full Pix2Pix GAN training loop with AMP.

    Parameters
    ----------
    cfg : TrainConfig
        Hyperparameter container (from ``config.py``).
    device : torch.device
        Target accelerator.
    """

    def __init__(
        self,
        cfg: Any = TRAIN_CFG,
        device: Optional[torch.device] = None,
    ) -> None:
        self.cfg = cfg
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # ── Models ───────────────────────────────────────────────────
        self.gen = build_generator(self.device)
        self.disc = build_discriminator(self.device)

        # ── Optimisers ───────────────────────────────────────────────
        self.opt_g = torch.optim.Adam(
            self.gen.parameters(), lr=cfg.lr_g, betas=cfg.betas,
        )
        self.opt_d = torch.optim.Adam(
            self.disc.parameters(), lr=cfg.lr_d, betas=cfg.betas,
        )

        # ── Losses ───────────────────────────────────────────────────
        self.criterion_gan = nn.BCEWithLogitsLoss()
        self.criterion_l1 = nn.L1Loss()
        self.criterion_edge = TrainTimeEdgeLoss().to(self.device)

        # ── AMP scalers (one per optimiser for safety) ───────────────
        self.scaler_g = torch.amp.GradScaler("cuda", enabled=cfg.use_amp)
        self.scaler_d = torch.amp.GradScaler("cuda", enabled=cfg.use_amp)

        # ── Bookkeeping ──────────────────────────────────────────────
        self.global_step = 0
        self.start_epoch = 0

        # Resume from checkpoint
        if cfg.resume_from:
            self._load_checkpoint(Path(cfg.resume_from))

    # ================================================================== #
    #  Core: single training step
    # ================================================================== #

    def train_step(
        self,
        sar: torch.Tensor,
        real_optical: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Execute one generator + discriminator update.

        Parameters
        ----------
        sar          : (B, SAR_CH, 256, 256)  on ``self.device``
        real_optical : (B, OPT_CH, 256, 256)  on ``self.device``

        Returns
        -------
        dict with scalar loss components for logging.

        Implementation Notes
        --------------------
        The generator forward pass is computed **twice**: once inside the
        discriminator update (with ``.detach()`` to block gradients into G),
        and once freshly inside the generator update.  This avoids the
        subtle AMP precision mismatch that occurs when reusing a tensor
        created under a different ``autocast`` context.
        """
        cfg = self.cfg

        # ────────────────────────────────────────────────────────────
        #  1)  UPDATE DISCRIMINATOR
        # ────────────────────────────────────────────────────────────
        self.opt_d.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=cfg.use_amp):
            # Generate fake optical (detached — no G gradients)
            with torch.no_grad():
                fake_optical_d = self.gen(sar)

            # ── Real pair ──
            real_pair = torch.cat([sar, real_optical], dim=1)
            pred_real = self.disc(real_pair)
            label_real = torch.ones_like(pred_real)
            loss_d_real = self.criterion_gan(pred_real, label_real)

            # ── Fake pair ──
            fake_pair = torch.cat([sar, fake_optical_d], dim=1)
            pred_fake = self.disc(fake_pair)
            label_fake = torch.zeros_like(pred_fake)
            loss_d_fake = self.criterion_gan(pred_fake, label_fake)

            loss_d = (loss_d_real + loss_d_fake) * 0.5

        self.scaler_d.scale(loss_d).backward()
        self.scaler_d.step(self.opt_d)
        self.scaler_d.update()

        # ────────────────────────────────────────────────────────────
        #  2)  UPDATE GENERATOR
        #
        #  CRITICAL: We re-generate fake_optical here with a FRESH
        #  forward pass so that:
        #    a) The computation graph is rooted in this autocast ctx.
        #    b) AMP precision is consistent end-to-end.
        #    c) Gradients flow through G (no detach).
        # ────────────────────────────────────────────────────────────
        self.opt_g.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=cfg.use_amp):
            fake_optical = self.gen(sar)

            fake_pair_g = torch.cat([sar, fake_optical], dim=1)
            pred_fake_g = self.disc(fake_pair_g)
            label_real_g = torch.ones_like(pred_fake_g)

            # Adversarial loss — fool the discriminator
            loss_g_gan = self.criterion_gan(pred_fake_g, label_real_g)

            # L1 reconstruction
            loss_g_l1 = self.criterion_l1(fake_optical, real_optical)

            # Combined generator loss
            loss_g = loss_g_gan + cfg.lambda_l1 * loss_g_l1

            # ── Optional: Train-Time Edge Loss (Baseline 2) ─────────
            loss_edge = torch.tensor(0.0, device=self.device)
            if cfg.use_edge_loss:
                loss_edge = self.criterion_edge(fake_optical, real_optical)
                loss_g = loss_g + cfg.lambda_edge * loss_edge

        self.scaler_g.scale(loss_g).backward()
        self.scaler_g.step(self.opt_g)
        self.scaler_g.update()

        self.global_step += 1

        return {
            "loss_d": loss_d.item(),
            "loss_g": loss_g.item(),
            "loss_g_gan": loss_g_gan.item(),
            "loss_g_l1": loss_g_l1.item(),
            "loss_edge": loss_edge.item(),
        }

    # ================================================================== #
    #  Full epoch loop
    # ================================================================== #

    def train_epoch(
        self,
        loader: DataLoader,
        epoch: int,
    ) -> Dict[str, float]:
        """
        Train for a single epoch.

        Parameters
        ----------
        loader : DataLoader yielding dicts with ``"sar"`` and ``"optical"``.
        epoch  : current epoch index (for logging).

        Returns
        -------
        dict — epoch-averaged losses.
        """
        self.gen.train()
        self.disc.train()

        running: Dict[str, float] = {}
        n_batches = 0

        pbar = tqdm(loader, desc=f"Epoch {epoch:03d}", unit="batch")
        for batch in pbar:
            sar = batch["sar"].to(self.device, non_blocking=True)
            opt = batch["optical"].to(self.device, non_blocking=True)

            losses = self.train_step(sar, opt)

            # Accumulate
            for k, v in losses.items():
                running[k] = running.get(k, 0.0) + v
            n_batches += 1

            # Progress bar
            if self.global_step % self.cfg.log_every == 0:
                pbar.set_postfix({
                    "D": f"{losses['loss_d']:.4f}",
                    "G": f"{losses['loss_g']:.4f}",
                    "L1": f"{losses['loss_g_l1']:.4f}",
                    "Edge": f"{losses['loss_edge']:.4f}",
                })

        # Average
        avg = {k: v / max(n_batches, 1) for k, v in running.items()}
        logger.info(
            "Epoch %03d | D=%.4f  G=%.4f  L1=%.4f  Edge=%.4f",
            epoch, avg["loss_d"], avg["loss_g"],
            avg["loss_g_l1"], avg["loss_edge"],
        )
        return avg

    # ================================================================== #
    #  Validation loop
    # ================================================================== #

    @torch.no_grad()
    def validate_epoch(
        self,
        loader: DataLoader,
        epoch: int,
    ) -> Dict[str, float]:
        """
        Run validation (generator-only, no D update) and return avg losses.

        Parameters
        ----------
        loader : DataLoader for validation split.
        epoch  : current epoch index (for logging).

        Returns
        -------
        dict — epoch-averaged val losses (L1 and optionally edge).
        """
        self.gen.eval()

        running: Dict[str, float] = {}
        n_batches = 0

        pbar = tqdm(loader, desc=f"Val   {epoch:03d}", unit="batch")
        for batch in pbar:
            sar = batch["sar"].to(self.device, non_blocking=True)
            real_opt = batch["optical"].to(self.device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=self.cfg.use_amp):
                fake_opt = self.gen(sar)
                val_l1 = self.criterion_l1(fake_opt, real_opt)

                val_edge = torch.tensor(0.0, device=self.device)
                if self.cfg.use_edge_loss:
                    val_edge = self.criterion_edge(fake_opt, real_opt)

            losses = {
                "val_l1": val_l1.item(),
                "val_edge": val_edge.item(),
            }
            for k, v in losses.items():
                running[k] = running.get(k, 0.0) + v
            n_batches += 1

        avg = {k: v / max(n_batches, 1) for k, v in running.items()}
        logger.info(
            "Val   %03d | L1=%.4f  Edge=%.4f",
            epoch, avg["val_l1"], avg["val_edge"],
        )
        return avg

    # ================================================================== #
    #  Fit (multi-epoch driver — does NOT auto-run on import)
    # ================================================================== #

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
    ) -> None:
        """
        Main training driver.  Call this explicitly to begin training.

        Parameters
        ----------
        train_loader : DataLoader for training split.
        val_loader   : Optional DataLoader for validation split.
                       If provided, runs validation after every epoch.

        Saves checkpoints every ``cfg.save_every`` epochs and clears
        CUDA cache after each checkpoint save to prevent OOM.
        """
        logger.info(
            "Starting training | epochs=%d  batch=%d  AMP=%s  edge_loss=%s",
            self.cfg.epochs, self.cfg.batch_size,
            self.cfg.use_amp, self.cfg.use_edge_loss,
        )

        for epoch in range(self.start_epoch, self.cfg.epochs):
            t0 = time.time()
            self.train_epoch(train_loader, epoch)

            # ── Validation ───────────────────────────────────────────
            if val_loader is not None:
                self.validate_epoch(val_loader, epoch)

            elapsed = time.time() - t0
            logger.info("Epoch %03d completed in %.1f s", epoch, elapsed)

            # ── Checkpoint ───────────────────────────────────────────
            if (epoch + 1) % self.cfg.save_every == 0 or epoch == self.cfg.epochs - 1:
                self._save_checkpoint(epoch)
                torch.cuda.empty_cache()  # reclaim VRAM after save

    # ================================================================== #
    #  Checkpointing
    # ================================================================== #

    def _save_checkpoint(self, epoch: int) -> None:
        tag = "edge" if self.cfg.use_edge_loss else "baseline"
        path = CHECKPOINT_DIR / f"pix2pix_{tag}_epoch{epoch:03d}.pt"
        torch.save(
            {
                "epoch": epoch + 1,
                "global_step": self.global_step,
                "gen_state": self.gen.state_dict(),
                "disc_state": self.disc.state_dict(),
                "opt_g_state": self.opt_g.state_dict(),
                "opt_d_state": self.opt_d.state_dict(),
                "scaler_g_state": self.scaler_g.state_dict(),
                "scaler_d_state": self.scaler_d.state_dict(),
                "cfg": vars(self.cfg),
            },
            path,
        )
        logger.info("Checkpoint saved → %s", path)

    def _load_checkpoint(self, path: Path) -> None:
        if not path.exists():
            logger.warning("Checkpoint %s not found — starting fresh.", path)
            return
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.gen.load_state_dict(ckpt["gen_state"])
        self.disc.load_state_dict(ckpt["disc_state"])
        self.opt_g.load_state_dict(ckpt["opt_g_state"])
        self.opt_d.load_state_dict(ckpt["opt_d_state"])
        self.scaler_g.load_state_dict(ckpt["scaler_g_state"])
        self.scaler_d.load_state_dict(ckpt["scaler_d_state"])
        self.start_epoch = ckpt["epoch"]
        self.global_step = ckpt["global_step"]
        logger.info(
            "Resumed from %s  (epoch %d, step %d)",
            path, self.start_epoch, self.global_step,
        )


# ======================================================================
# CLI entry-point (does NOT auto-train on import)
# ======================================================================

if __name__ == "__main__":
    import argparse

    # Only configure root logger when run as a script
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )

    parser = argparse.ArgumentParser(description="SAR-TopoGuard Pix2Pix Trainer")
    parser.add_argument(
        "--epochs", type=int, default=TRAIN_CFG.epochs,
        help="Number of epochs to train.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=TRAIN_CFG.batch_size,
        help="Batch size.",
    )
    parser.add_argument(
        "--use-edge-loss", action="store_true",
        help="Enable train-time edge loss (Baseline 2).",
    )
    parser.add_argument(
        "--resume", type=str, default="",
        help="Path to checkpoint to resume from.",
    )
    parser.add_argument(
        "--no-amp", action="store_true",
        help="Disable automatic mixed precision.",
    )
    args = parser.parse_args()

    # Override config
    TRAIN_CFG.epochs = args.epochs
    TRAIN_CFG.batch_size = args.batch_size
    TRAIN_CFG.use_edge_loss = args.use_edge_loss
    TRAIN_CFG.resume_from = args.resume
    TRAIN_CFG.use_amp = not args.no_amp

    # Build data loaders (Module 1)
    from dataset import build_dataloaders
    loaders = build_dataloaders(batch_size=TRAIN_CFG.batch_size)

    # Build trainer and run (with validation if available)
    trainer = Pix2PixTrainer(cfg=TRAIN_CFG)
    trainer.fit(
        train_loader=loaders["train"],
        val_loader=loaders.get("val"),
    )
