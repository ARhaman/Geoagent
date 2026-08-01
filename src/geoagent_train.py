"""
geoagent_train.py
==================
Training script for the Pix2Pix model on paired Porosity-Facies images.

Usage:
    python geoagent_train.py --data_root ./24Paired_Porosity_Facies_split \
                             --direction AB \
                             --epochs 200 \
                             --batch_size 4 \
                             --save_dir ./checkpoints

Author: Abdulrahman Al-Fakih
"""

import os
import argparse
import time
import json
import torch
import numpy as np
from torchvision.utils import save_image

from geoagent_dataset import build_dataloaders
try:
    from geoagent_dataset import build_3d_dataloaders
except ImportError:
    build_3d_dataloaders = None
from geoagent_model import Pix2PixModel, count_params


# ── Argument Parser ───────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train Pix2Pix for Reservoir Property Translation")
    p.add_argument("--data_root",   default="./24Paired_Porosity_Facies_split",
                   help="Root folder with train/val/test subfolders")
    p.add_argument("--direction",   default="AB", choices=["AB", "BA"],
                   help="AB = Porosity->Facies  |  BA = Facies->Porosity")
    p.add_argument("--epochs",      type=int,   default=200)
    p.add_argument("--batch_size",  type=int,   default=4)
    p.add_argument("--image_size",  type=int,   default=256)
    p.add_argument("--ngf",         type=int,   default=64,   help="Generator filters")
    p.add_argument("--ndf",         type=int,   default=64,   help="Discriminator filters")
    p.add_argument("--lambda_l1",   type=float, default=100.0)
    p.add_argument("--lr",          type=float, default=2e-4)
    p.add_argument("--beta1",       type=float, default=0.5)
    p.add_argument("--use_3d",       action="store_true", default=False)
    p.add_argument("--norm",        default="batch", choices=["batch", "instance"])
    p.add_argument("--save_dir",    default="./checkpoints")
    p.add_argument("--save_freq",   type=int,   default=10,  help="Save checkpoint every N epochs")
    p.add_argument("--vis_freq",    type=int,   default=5,   help="Save sample images every N epochs")
    p.add_argument("--num_workers", type=int,   default=4)
    p.add_argument("--resume",      default=None, help="Path to checkpoint to resume from")
    return p.parse_args()


# ── Utilities ─────────────────────────────────────────────────────────────────

def denorm(t: torch.Tensor) -> torch.Tensor:
    """Convert [-1,1] tensor to [0,1] for visualization."""
    return (t.clamp(-1, 1) + 1) / 2


def save_samples(model, val_loader, epoch: int, out_dir: str, n: int = 4):
    """Save a grid of [input | generated | target] from validation set."""
    os.makedirs(out_dir, exist_ok=True)
    batch = next(iter(val_loader))
    inp, tgt = batch["input"][:n], batch["target"][:n]
    gen = model.predict(inp)  # already on device; returns CPU tensor after .cpu()
    gen = gen.cpu()

    grid = torch.cat([denorm(inp), denorm(gen), denorm(tgt)], dim=-1)  # side-by-side
    path = os.path.join(out_dir, f"epoch_{epoch:04d}.png")
    save_image(grid, path, nrow=1)
    print(f"  [vis] Saved samples -> {path}")


def log_losses(history: list, epoch: int, losses: dict, elapsed: float):
    entry = {"epoch": epoch, "elapsed_s": round(elapsed, 1), **losses}
    history.append(entry)
    print(
        f"Epoch [{epoch:4d}]  "
        f"loss_G={losses['loss_G']:.4f}  "
        f"loss_G_gan={losses['loss_G_gan']:.4f}  "
        f"loss_G_l1={losses['loss_G_l1']:.4f}  "
        f"loss_D={losses['loss_D']:.4f}  "
        f"({elapsed:.1f}s)"
    )


# ── Training Loop ─────────────────────────────────────────────────────────────

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  GeoAgent — Pix2Pix Training")
    print(f"  Direction : {args.direction}  ({'Porosity->Facies' if args.direction=='AB' else 'Facies->Porosity'})")
    print(f"  Device    : {device}")
    print(f"  Epochs    : {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"{'='*60}\n")

    # ── Data ──────────────────────────────────────────────────────────────────
    loaders = (build_3d_dataloaders if args.use_3d else build_dataloaders)(
        data_root=args.data_root,
        direction=args.direction,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = Pix2PixModel(
        device=device,
        ngf=args.ngf,
        ndf=args.ndf,
        lambda_l1=args.lambda_l1,
        norm=args.norm,
        lr=args.lr,
        beta1=args.beta1,
    )
    print(f"  {count_params(model)}\n")

    start_epoch = 1
    if args.resume:
        model.load(args.resume)
        start_epoch = int(os.path.basename(args.resume).split("_ep")[1].split(".")[0]) + 1
        print(f"  Resuming from epoch {start_epoch}")

    os.makedirs(args.save_dir, exist_ok=True)
    vis_dir  = os.path.join(args.save_dir, "samples")
    log_path = os.path.join(args.save_dir, f"training_log_{args.direction}.json")
    history  = []

    # ── Epoch loop ─────────────────────────────────────────────────────────
    for epoch in range(start_epoch, args.epochs + 1):
        model.G.train()
        model.D.train()

        epoch_losses = {"loss_G": 0, "loss_G_gan": 0, "loss_G_l1": 0, "loss_D": 0}
        t0 = time.time()

        for batch in loaders["train"]:
            inp = batch["input"]
            tgt = batch["target"]
            losses = model.train_step(inp, tgt)
            for k in epoch_losses:
                epoch_losses[k] += losses[k]

        n_batches = len(loaders["train"])
        avg_losses = {k: v / n_batches for k, v in epoch_losses.items()}
        elapsed = time.time() - t0

        log_losses(history, epoch, avg_losses, elapsed)
        model.step_schedulers()

        # Validation visualization
        if epoch % args.vis_freq == 0:
            save_samples(model, loaders["val"], epoch, vis_dir)

        # Checkpoint
        if epoch % args.save_freq == 0 or epoch == args.epochs:
            ckpt_name = f"pix2pix_{args.direction}_ep{epoch:04d}.pth"
            model.save(os.path.join(args.save_dir, ckpt_name))

        # Save log
        with open(log_path, "w") as f:
            json.dump(history, f, indent=2)

    print(f"\nTraining complete. Checkpoints -> {args.save_dir}")
    return history


if __name__ == "__main__":
    args = parse_args()
    train(args)
