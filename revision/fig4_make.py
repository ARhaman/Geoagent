"""
fig4_make.py -- corrected Figure 4: ONE co-registered sample (same layer, same tile) for all four directions (Reviewer 1, comment 8).
Facies, porosity and permeability tiles are cropped from the SAME full-field layer renders (Real layers\*_Orginal_Dataset) at the same pixel window (512x512), so every row shows the same location.
Rows: Porosity->Facies, Facies->Porosity, Permeability->Facies, Facies->Permeability.  Columns: input | single pass | GeoAgent (TTAx8) | ground truth.
Run (geoldm_gpu):  python fig4_make.py --layer 150          (optional: --r0 768 --c0 1536 to choose the tile)
Output: figures\Fig4_new.png  (+ fig4_info.txt).  NOTE: the layer's split membership is not recorded in the dataset; the figure is illustrative, quantitative evidence is in Tables 2-5.
"""
import os, sys, argparse
import numpy as np, torch
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
p = argparse.ArgumentParser()
p.add_argument("--code_dir", default=r"D:\First year\Papers\GeoAgent"); p.add_argument("--ckpt_base", default=r"D:\First year\Papers\GeoAgent\checkpoints"); p.add_argument("--rev_dir", default=r"D:\First year\Papers\GeoAgent\revision_ACAGS")
p.add_argument("--real", default=r"E:\2D Dutch's Dataset\Real layers"); p.add_argument("--layer", type=int, default=150); p.add_argument("--r0", type=int, default=-1); p.add_argument("--c0", type=int, default=-1)
a = p.parse_args(); sys.path.insert(0, a.code_dir); sys.path.insert(0, a.rev_dir)
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from nb_model import Pix2PixModel
import legend_decode as L
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n = a.layer; ld = lambda d, pre: np.array(Image.open(os.path.join(a.real, d, f"{pre}_layer_{n}_ultra_high_res.png")).convert("RGB"))
FA, PO, PE = ld("Facies_Orginal_Dataset", "Facies"), ld("Poros_Orginal_Dataset", "Poro"), ld("Perm_Orginal_Dataset", "Perm")
H, W, _ = FA.shape
def inside(tile): s = tile.sum(-1); return ((s > 30) & (s < 740)).mean()
best = None
if a.r0 >= 0 and a.c0 >= 0: best = (a.r0, a.c0)
else:
    for r0 in range(0, H - 511, 384):
        for c0 in range(0, W - 511, 384):
            t = FA[r0:r0 + 512, c0:c0 + 512]
            if inside(t) < 0.999: continue
            cls = L.decode_facies(t[4::8, 4::8]); frac = [(cls == k).mean() for k in range(3)]; score = min(frac)         # prefer tiles where all 3 facies are well represented
            if best is None or score > best[2]: best = (r0, c0, score)
r0, c0 = best[0], best[1]; print("layer", n, "tile", r0, c0)
crop = lambda im: im[r0:r0 + 512, c0:c0 + 512]
fa, po, pe = crop(FA), crop(PO), crop(PE)
tens = lambda im: (torch.from_numpy(np.asarray(Image.fromarray(im).resize((256, 256), Image.BILINEAR))).float().permute(2, 0, 1) / 255 - 0.5) / 0.5
def d4(k_, f):
    fw = lambda x: torch.rot90(torch.flip(x, [-1]) if f else x, k_, [-2, -1]); iv = lambda y: (torch.flip(torch.rot90(y, -k_, [-2, -1]), [-1]) if f else torch.rot90(y, -k_, [-2, -1])); return fw, iv
T8 = [d4(k_, f) for f in (False, True) for k_ in range(4)]
img = lambda t: ((t.clamp(-1, 1) + 1) / 2).permute(1, 2, 0).cpu().numpy()
rows = [("Porosity → Facies", "PorFac", "AB", po, fa), ("Facies → Porosity", "PorFac", "BA", fa, po), ("Permeability → Facies", "PermFac", "AB", pe, fa), ("Facies → Permeability", "PermFac", "BA", fa, pe)]
fig, ax = plt.subplots(4, 4, figsize=(10, 10.4)); cols = ["Input", "Single pass", "GeoAgent (TTA×8)", "Ground truth"]
for ri, (name, task, d, inp, tgt) in enumerate(rows):
    M = Pix2PixModel(device=dev); M.load(os.path.join(a.ckpt_base, f"{task}_{d}", f"pix2pix_{d}_ep0100.pth")); M.G.eval()
    x = tens(inp).unsqueeze(0).to(dev); y = tens(tgt)
    with torch.no_grad(): s = M.G(x)[0]; t8 = torch.stack([iv(M.G(fw(x))) for fw, iv in T8]).mean(0)[0]
    for ci, im in enumerate([img(x[0]), img(s), img(t8), img(y)]):
        ax[ri, ci].imshow(im); ax[ri, ci].set_xticks([]); ax[ri, ci].set_yticks([])
        if ri == 0: ax[ri, ci].set_title(cols[ci], fontsize=11)
    ax[ri, 0].set_ylabel(name, fontsize=11)
fig.suptitle(f"Same sample in all rows (full-field layer {n}, tile rows {r0}-{r0+511}, cols {c0}-{c0+511})", fontsize=10); plt.tight_layout(rect=[0, 0, 1, 0.97])
os.makedirs(os.path.join(a.rev_dir, "figures"), exist_ok=True); out = os.path.join(a.rev_dir, "figures", "Fig4_new.png"); plt.savefig(out, dpi=200); print("Saved", out)
open(os.path.join(a.rev_dir, "figures", "fig4_info.txt"), "w").write(f"layer {n}; tile rows {r0}-{r0+511}; cols {c0}-{c0+511}\n")
