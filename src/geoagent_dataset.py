"""
geoagent_dataset.py
====================
Dataset loader for paired Porosity–Facies images from the Pix2Geomodel format.

Each image is 512×556 RGB — left half (256×556) is one property,
right half (256×556) is the paired property (Porosity | Facies).

Directory structure expected:
    root/
        train/  cube_i*_j*_k*.png
        val/    cube_i*_j*_k*.png
        test/   cube_i*_j*_k*.png

Author: Abdulrahman Al-Fakih
"""

import os
import re
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np


# ── Transforms ──────────────────────────────────────────────────────────────

def get_transform(image_size: int = 256, augment: bool = True):
    """Returns a torchvision transform pipeline for one half-image."""
    ops = []
    if augment:
        ops += [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
        ]
    ops += [
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
    return transforms.Compose(ops)


# ── Dataset ──────────────────────────────────────────────────────────────────

class PairedGeoDataset(Dataset):
    """
    Loads side-by-side paired reservoir property images.

    Args:
        root_dir   : Path to split folder (e.g. .../train/)
        direction  : 'AB' = left→right (Porosity→Facies)
                     'BA' = right→left (Facies→Porosity)
                     'both' = returns both directions (for bidirectional training)
        image_size : Spatial size to resize each half to (square)
        augment    : Apply random flips during training
    """

    def __init__(
        self,
        root_dir: str,
        direction: str = "AB",
        image_size: int = 256,
        augment: bool = True,
    ):
        super().__init__()
        self.root_dir = root_dir
        self.direction = direction
        self.image_size = image_size

        self.files = sorted([
            os.path.join(root_dir, f)
            for f in os.listdir(root_dir)
            if f.lower().endswith(".png")
        ])
        assert len(self.files) > 0, f"No PNG files found in {root_dir}"

        # Shared transform — same seed applied to both halves so augmentations match
        self.transform = get_transform(image_size, augment)

        print(f"[PairedGeoDataset] Loaded {len(self.files)} images from '{root_dir}'  "
              f"| direction={direction} | size={image_size}×{image_size}")

    def __len__(self):
        return len(self.files)

    def _split_pair(self, img: Image.Image):
        """Split a 512-wide image into left and right halves."""
        w, h = img.size
        mid = w // 2
        left  = img.crop((0,   0, mid, h))   # Property A (Porosity)
        right = img.crop((mid, 0, w,   h))   # Property B (Facies)
        return left, right

    def _apply_same_transform(self, imgA: Image.Image, imgB: Image.Image):
        """Apply identical random augmentation to both halves."""
        # Stack → transform → unstack to guarantee same random ops
        seed = torch.randint(0, 2**32, (1,)).item()
        torch.manual_seed(seed)
        tA = self.transform(imgA)
        torch.manual_seed(seed)
        tB = self.transform(imgB)
        return tA, tB

    def __getitem__(self, idx):
        path = self.files[idx]
        img  = Image.open(path).convert("RGB")
        A, B = self._split_pair(img)
        tA, tB = self._apply_same_transform(A, B)

        # Parse 3D coordinates from filename
        fname = os.path.basename(path)
        m = re.search(r"cube_i(\d+)_j(\d+)_k(\d+)", fname)
        coords = (int(m.group(1)), int(m.group(2)), int(m.group(3))) if m else (-1,-1,-1)

        if self.direction == "AB":
            return {"input": tA, "target": tB, "path": path, "coords": coords}
        elif self.direction == "BA":
            return {"input": tB, "target": tA, "path": path, "coords": coords}
        elif self.direction == "both":
            return {"A": tA, "B": tB, "path": path, "coords": coords}
        else:
            raise ValueError(f"Unknown direction: {self.direction}")


# ── Factory ───────────────────────────────────────────────────────────────────

def build_dataloaders(
    data_root: str,
    direction: str = "AB",
    image_size: int = 256,
    batch_size: int = 4,
    num_workers: int = 4,
):
    """
    Builds train / val / test DataLoaders from the pre-split directory.

    Args:
        data_root  : Root folder containing train/, val/, test/ sub-dirs
        direction  : 'AB', 'BA', or 'both'
        image_size : Each half resized to (image_size × image_size)
        batch_size : Mini-batch size
        num_workers: Parallel data workers

    Returns:
        dict with keys 'train', 'val', 'test'
    """
    loaders = {}
    for split in ("train", "val", "test"):
        path = os.path.join(data_root, split)
        augment = (split == "train")
        ds = PairedGeoDataset(path, direction=direction,
                              image_size=image_size, augment=augment)
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=True,
            drop_last=(split == "train"),
        )
    return loaders


# ── Quick test ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else "./24Paired_Porosity_Facies_split"
    loaders = build_dataloaders(root, direction="both", batch_size=2, num_workers=0)

    batch = next(iter(loaders["train"]))
    print(f"\nA (Porosity): {batch['A'].shape}  range [{batch['A'].min():.2f}, {batch['A'].max():.2f}]")
    print(f"B (Facies):  {batch['B'].shape}  range [{batch['B'].min():.2f}, {batch['B'].max():.2f}]")
    print(f"Coords:  {batch['coords']}")
    print(f"Path:    {batch['path'][0]}")
    print("\nDataset loader OK.")
