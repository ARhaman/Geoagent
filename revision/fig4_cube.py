"""
fig4_cube.py -- Supplementary Figure S18: block view of a STACK OF SLICE-WISE 2D predictions, same sub-volume in all four directions (Reviewer 1, comment 8, 3D version).
Sub-volume = the same tile (rows r0.., cols c0.., 512x512 px) in consecutive full-field layers l0..l1; every layer is predicted independently by the 2D models (no 3D model).
Each layer is sub-sampled to 32x32 cells (centre pixel of each 8x8 block) so the blocks stay crisp. Top face = first layer; side faces = cross-sections through the stack.
Columns: Input | GeoAgent (TTAx8) | Ground truth.  Colours are normalised per layer (no absolute colour bars).
NOTE: the split membership of these layers is not recorded in the dataset; the figure is illustrative and is not evidence of held-out accuracy.
Run (geoldm_gpu):  python fig4_cube.py            (defaults: tile 2304/384 as in Figure 4, layers 140-160)
Output: figures\\FigS18_cubes.png
"""
import os, sys, argparse
import numpy as np, torch
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
p = argparse.ArgumentParser()
p.add_argument("--code_dir", default=r"D:\First year\Papers\GeoAgent"); p.add_argument("--ckpt_base", default=r"D:\First year\Papers\GeoAgent\checkpoints"); p.add_argument("--rev_dir", default=r"D:\First year\Papers\GeoAgent\revision_ACAGS")
p.add_argument("--real", default=r"E:\2D Dutch's Dataset\Real layers"); p.add_argument("--l0", type=int, default=140); p.add_argument("--l1", type=int, default=160); p.add_argument("--r0", type=int, default=2304); p.add_argument("--c0", type=int, default=384)
p.add_argument("--cells", type=int, default=32)
a = p.parse_args(); sys.path.insert(0, a.code_dir); sys.path.insert(0, a.rev_dir)
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from nb_model import Pix2PixModel
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
layers = list(range(a.l0, a.l1 + 1)); nz = len(layers); step = 512 // a.cells
def load(d, pre, n): return np.array(Image.open(os.path.join(a.real, d, f"{pre}_layer_{n}_ultra_high_res.png")).convert("RGB"))[a.r0:a.r0 + 512, a.c0:a.c0 + 512]
SRC = {"fa": ("Facies_Orginal_Dataset", "Facies"), "po": ("Poros_Orginal_Dataset", "Poro"), "pe": ("Perm_Orginal_Dataset", "Perm")}
vol = {k: [load(d, pre, n) for n in layers] for k, (d, pre) in SRC.items()}
tens = lambda im: (torch.from_numpy(np.asarray(Image.fromarray(im).resize((256, 256), Image.BILINEAR))).float().permute(2, 0, 1) / 255 - 0.5) / 0.5
def d4(k_, f):
    fw = lambda x: torch.rot90(torch.flip(x, [-1]) if f else x, k_, [-2, -1]); iv = lambda y: (torch.flip(torch.rot90(y, -k_, [-2, -1]), [-1]) if f else torch.rot90(y, -k_, [-2, -1])); return fw, iv
T8 = [d4(k_, f) for f in (False, True) for k_ in range(4)]
to_img = lambda t: (((t.clamp(-1, 1) + 1) / 2).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
def predict(M, im):
    x = tens(im).unsqueeze(0).to(dev)
    with torch.no_grad(): t8 = torch.stack([iv(M.G(fw(x))) for fw, iv in T8]).mean(0)[0]
    return np.asarray(Image.fromarray(to_img(t8)).resize((512, 512), Image.NEAREST))
def cells(im): return im[step // 2::step, step // 2::step] / 255.0            # (cells, cells, 3)
rows = [("Porosity → Facies", "PorFac", "AB", "po", "fa"), ("Facies → Porosity", "PorFac", "BA", "fa", "po"), ("Permeability → Facies", "PermFac", "AB", "pe", "fa"), ("Facies → Permeability", "PermFac", "BA", "fa", "pe")]
fig = plt.figure(figsize=(10.5, 13)); cols = ["Input", "GeoAgent (TTA×8)", "Ground truth"]
for ri, (name, task, d, si, ti) in enumerate(rows):
    M = Pix2PixModel(device=dev); M.load(os.path.join(a.ckpt_base, f"{task}_{d}", f"pix2pix_{d}_ep0100.pth")); M.G.eval()
    pred = [predict(M, vol[si][z]) for z in range(nz)]
    stacks = [[vol[si][z] for z in range(nz)], pred, [vol[ti][z] for z in range(nz)]]
    for ci, st in enumerate(stacks):
        arr = np.stack([cells(im) for im in st], axis=-2)                       # (rows, cols, nz, 3): axis0 = image row, axis1 = image col, axis2 = layer
        arr = np.transpose(arr, (1, 0, 2, 3))[:, ::-1][:, :, ::-1]               # x = col, y = flipped row, z = layer with first layer on top
        fc = np.concatenate([arr, np.ones(arr.shape[:3] + (1,))], axis=-1); filled = np.ones(arr.shape[:3], bool)
        ax = fig.add_subplot(4, 3, ri * 3 + ci + 1, projection="3d"); ax.voxels(filled, facecolors=fc, edgecolor=(0, 0, 0, 0.12), linewidth=0.1)
        ax.set_box_aspect((1, 1, 0.7)); ax.view_init(elev=24, azim=-58); ax.set_axis_off()
        if ri == 0: ax.set_title(cols[ci], fontsize=11)
        if ci == 0: ax.text2D(-0.02, 0.5, name, transform=ax.transAxes, rotation=90, va="center", ha="right", fontsize=11)
    print(name, "done", flush=True)
fig.suptitle(f"Stack of slice-wise 2D predictions: same sub-volume in all rows (layers {a.l0}-{a.l1}, tile rows {a.r0}-{a.r0+511}, cols {a.c0}-{a.c0+511})", fontsize=10); plt.tight_layout(rect=[0, 0, 1, 0.97])
os.makedirs(os.path.join(a.rev_dir, "figures"), exist_ok=True); out = os.path.join(a.rev_dir, "figures", "FigS18_cubes.png"); plt.savefig(out, dpi=170); print("Saved", out)
