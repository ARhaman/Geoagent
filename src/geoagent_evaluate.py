"""
geoagent_evaluate.py
=====================
Full evaluation of GeoAgent on the test split.

Computes paper-standard metrics:
  - SSIM  (Structural Similarity Index)
  - PSNR  (Peak Signal-to-Noise Ratio)
  - MAE   (Mean Absolute Error in [0,1] space)
  - Pixel Accuracy  (fraction within +/-10% tolerance)
  - Acceptance Rate (fraction of samples passing SSIM threshold)
  - Agent Efficiency (avg iterations and time per sample)

Outputs:
  - CSV with per-image metrics
  - JSON summary (mean +/- std for all metrics)
  - Visual comparison grid (input | predicted | target)

Usage:
    python geoagent_evaluate.py \
        --data_root ./24Paired_Porosity_Facies_split \
        --model_AB  ./checkpoints/pix2pix_AB_ep0200.pth \
        --model_BA  ./checkpoints/pix2pix_BA_ep0200.pth \
        --direction AB \
        --output_dir ./results

Author: Abdulrahman Al-Fakih
"""

import os
import csv
import json
import argparse
import torch
import numpy as np
from torchvision.utils import save_image

from geoagent_dataset import build_dataloaders
from geoagent_agent import GeoAgent, compute_ssim, compute_pixel_accuracy


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    pred_01   = (pred.cpu().clamp(-1, 1) + 1) / 2
    target_01 = (target.cpu().clamp(-1, 1) + 1) / 2
    mse = ((pred_01 - target_01) ** 2).mean().item()
    if mse < 1e-10:
        return 100.0
    return float(10 * np.log10(1.0 / mse))


def compute_mae(pred: torch.Tensor, target: torch.Tensor) -> float:
    pred_01   = (pred.cpu().clamp(-1, 1) + 1) / 2
    target_01 = (target.cpu().clamp(-1, 1) + 1) / 2
    return float((pred_01 - target_01).abs().mean().item())


# ── Visualization ─────────────────────────────────────────────────────────────

def save_comparison_grid(inp, pred, tgt, path, n=8):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    inp, pred, tgt = inp[:n], pred[:n], tgt[:n]
    denorm = lambda t: (t.clamp(-1, 1) + 1) / 2
    grid = torch.cat([denorm(inp), denorm(pred), denorm(tgt)], dim=-1)
    save_image(grid, path, nrow=1, padding=4, pad_value=1.0)


# ── Evaluation Loop ───────────────────────────────────────────────────────────

def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  GeoAgent Evaluation | direction={args.direction} | split={args.split}")
    print(f"{'='*60}\n")

    os.makedirs(args.output_dir, exist_ok=True)

    loaders = build_dataloaders(
        data_root=args.data_root,
        direction=args.direction,
        image_size=args.image_size,
        batch_size=1,
        num_workers=0,
    )
    loader = loaders[args.split]

    agent = GeoAgent(
        model_AB_path=args.model_AB,
        model_BA_path=args.model_BA,
        device=device,
        ssim_threshold=args.ssim_threshold,
        max_iterations=args.max_iterations,
        image_size=args.image_size,
        verbose=False,
    )

    rows = []
    all_preds, all_inps, all_tgts = [], [], []

    for idx, batch in enumerate(loader):
        inp, tgt = batch["input"], batch["target"]
        path = batch["path"][0]
        coords = tuple(int(c[0]) for c in batch["coords"])

        result = agent.run(inp, tgt, direction=args.direction)
        pred = result.predicted_image.cpu()

        rows.append({
            "idx":        idx,
            "filename":   os.path.basename(path),
            "i": coords[0], "j": coords[1], "k": coords[2],
            "ssim":       round(compute_ssim(pred, tgt),           4),
            "psnr_db":    round(compute_psnr(pred, tgt),           2),
            "mae":        round(compute_mae(pred, tgt),            4),
            "pixel_acc":  round(compute_pixel_accuracy(pred, tgt), 4),
            "n_iter":     result.n_iterations,
            "accepted":   result.accepted,
            "confidence": round(result.confidence, 3),
            "elapsed_ms": round(result.elapsed_ms, 1),
        })
        all_preds.append(pred)
        all_inps.append(inp)
        all_tgts.append(tgt)

        if (idx + 1) % 25 == 0 or (idx + 1) == len(loader):
            mean_ssim = np.mean([r["ssim"] for r in rows])
            print(f"  [{idx+1:3d}/{len(loader)}]  mean SSIM={mean_ssim:.4f}")

    # Summary
    metrics_keys = ["ssim", "psnr_db", "mae", "pixel_acc", "n_iter", "elapsed_ms"]
    summary = {
        m: {
            "mean": round(float(np.mean([r[m] for r in rows])), 4),
            "std":  round(float(np.std([r[m]  for r in rows])), 4),
        }
        for m in metrics_keys
    }
    summary["acceptance_rate"] = round(sum(r["accepted"] for r in rows) / len(rows), 4)
    summary["n_samples"]  = len(rows)
    summary["direction"]  = args.direction

    print(f"\n  SSIM={summary['ssim']['mean']:.4f}+/-{summary['ssim']['std']:.4f}  "
          f"PSNR={summary['psnr_db']['mean']:.2f}dB  "
          f"MAE={summary['mae']['mean']:.4f}  "
          f"AcceptRate={summary['acceptance_rate']*100:.1f}%\n")

    # Save CSV
    csv_path = os.path.join(args.output_dir, f"metrics_{args.direction}_{args.split}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    # Save JSON
    json_path = os.path.join(args.output_dir, f"summary_{args.direction}_{args.split}.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Visual grid
    grid_path = os.path.join(args.output_dir, f"grid_{args.direction}_{args.split}.png")
    save_comparison_grid(
        torch.cat(all_inps[:8]),
        torch.cat(all_preds[:8]),
        torch.cat(all_tgts[:8]),
        grid_path
    )

    print(f"  Outputs saved to: {args.output_dir}/")
    return summary


def evaluate_bidirectional(args):
    """Evaluate both AB and BA, print combined table."""
    results = {}
    for direction in ["AB", "BA"]:
        args.direction = direction
        results[direction] = evaluate(args)

    print("\n" + "="*60)
    print(f"  {'Metric':<18} {'AB (Por->Fac)':>20} {'BA (Fac->Por)':>20}")
    print("  " + "-"*56)
    for m in ["ssim", "psnr_db", "mae", "pixel_acc"]:
        ab = results["AB"][m]
        ba = results["BA"][m]
        print(f"  {m:<18} {ab['mean']:>8.4f}+/-{ab['std']:.4f}   "
              f"{ba['mean']:>8.4f}+/-{ba['std']:.4f}")

    combined_path = os.path.join(args.output_dir, "summary_bidirectional.json")
    with open(combined_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Combined JSON -> {combined_path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root",      default="./24Paired_Porosity_Facies_split")
    p.add_argument("--model_AB",       required=True)
    p.add_argument("--model_BA",       required=True)
    p.add_argument("--direction",      default="AB", choices=["AB", "BA", "both"])
    p.add_argument("--split",          default="test", choices=["train", "val", "test"])
    p.add_argument("--image_size",     type=int,   default=256)
    p.add_argument("--ssim_threshold", type=float, default=0.70)
    p.add_argument("--max_iterations", type=int,   default=3)
    p.add_argument("--output_dir",     default="./results")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.direction == "both":
        evaluate_bidirectional(args)
    else:
        evaluate(args)
