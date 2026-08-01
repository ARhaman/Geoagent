"""
run_geoagent.py
================
Quick-start script — demonstrates the full GeoAgent pipeline on your
24Paired_Porosity_Facies_split dataset without needing pre-trained checkpoints.

Steps performed:
  1. Load a sample batch from the test split
  2. Initialize GeoAgent with UNTRAINED models (shows the loop works)
  3. Run the agentic inference loop
  4. Print the full reasoning trace and metrics

To run for real: first train both directions with geoagent_train.py,
then pass the checkpoint paths to GeoAgent.

Usage:
    python run_geoagent.py --data_root ./24Paired_Porosity_Facies_split

Author: Abdulrahman Al-Fakih
"""

import argparse
import torch
from geoagent_dataset import build_dataloaders
from geoagent_model import Pix2PixModel
from geoagent_agent import GeoAgent

DUMMY_AB = "/tmp/geoagent_dummy_AB.pth"
DUMMY_BA = "/tmp/geoagent_dummy_BA.pth"


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[GeoAgent Quick-Start]  device={device}\n")

    # 1. Load data
    loaders = build_dataloaders(
        data_root=args.data_root,
        direction="AB",
        image_size=256,
        batch_size=1,
        num_workers=0,
    )
    batch = next(iter(loaders["test"]))
    inp = batch["input"]   # (1, 3, 256, 256)
    tgt = batch["target"]  # (1, 3, 256, 256)
    print(f"Sample: {batch['path'][0]}")
    print(f"Input tensor:  {inp.shape}  range=[{inp.min():.2f}, {inp.max():.2f}]")
    print(f"Target tensor: {tgt.shape}  range=[{tgt.min():.2f}, {tgt.max():.2f}]\n")

    # 2. Save dummy checkpoints (replace with real trained paths)
    if args.model_AB is None or args.model_BA is None:
        print("[INFO] No checkpoints provided — using untrained dummy models.")
        print("       Train first with: python geoagent_train.py --direction AB")
        print("       Then:             python geoagent_train.py --direction BA\n")
        dummy = Pix2PixModel(device=device)
        dummy.save(DUMMY_AB)
        dummy.save(DUMMY_BA)
        model_AB = DUMMY_AB
        model_BA = DUMMY_BA
    else:
        model_AB = args.model_AB
        model_BA = args.model_BA

    # 3. Initialize GeoAgent
    agent = GeoAgent(
        model_AB_path=model_AB,
        model_BA_path=model_BA,
        device=device,
        ssim_threshold=args.ssim_threshold,
        max_iterations=args.max_iterations,
        verbose=True,
    )

    # 4. Run agentic loop
    print("\n" + "="*55)
    print("  Running GeoAgent (Porosity -> Facies, direction=AB)")
    print("="*55)
    result = agent.run(inp, tgt, direction="AB")

    # 5. Print results
    print("\n" + result.summary())

    print(f"Reasoning trace ({len(result.reasoning_trace)} steps):")
    for step in result.reasoning_trace:
        mark = "✓" if step.accepted else ("✗" if step.accepted is False else " ")
        print(f"  Step {step.step:2d} [{step.action:<16}] {mark}  {step.observation}")

    # 6. Save output image
    from geoagent_agent import tensor_to_pil
    out_img = tensor_to_pil(result.predicted_image)
    out_path = "/tmp/geoagent_output.png"
    out_img.save(out_path)
    print(f"\nPrediction saved -> {out_path}")
    print("Done.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_root",      default="./24Paired_Porosity_Facies_split")
    p.add_argument("--model_AB",       default=None, help="Trained AB checkpoint")
    p.add_argument("--model_BA",       default=None, help="Trained BA checkpoint")
    p.add_argument("--ssim_threshold", type=float, default=0.70)
    p.add_argument("--max_iterations", type=int,   default=3)
    main(p.parse_args())
