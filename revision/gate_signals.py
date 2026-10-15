"""
gate_signals.py  --  ACAGS-D-26-00312, Reviewer 2 / Comment 1 (and Reviewer 1 / Comment 1)
Computes, for every patch of a split, the ground-truth SSIM (as in the paper) AND
several REFERENCE-FREE quality signals that could replace it at deployment.
Nothing is trained; checkpoints are only used for inference.

Signals (all computed WITHOUT looking at the target y):
  tta_cons   mean SSIM between each TTA member and the TTA mean   (self-consistency)
  tta_std    mean pixel-wise std across the TTA members
  mc_std     mean pixel-wise std across MC-dropout passes          (epistemic uncertainty)
  mc_cons    SSIM between MC-dropout mean and the TTA mean
  cycle      SSIM( R(pred), x )  with R = reverse-direction checkpoint (cycle-consistency)
Reference quantities (need y; used ONLY to evaluate the signals):
  ssim_true  SSIM(TTAx5 mean, y)    -> reproduces the paper's number
  ssim_single SSIM(single pass, y)

Run example (Anaconda prompt, GPU PC):
  python gate_signals.py --direction AB --split val  --task PorFac
  python gate_signals.py --direction AB --split test --task PorFac
"""
import os, sys, argparse, csv, time
import numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader

p = argparse.ArgumentParser()
p.add_argument("--code_dir", default=r"D:\First year\Papers\GeoAgent")
p.add_argument("--data_base", default=r"E:\2D Dutch's Dataset\patches_512")
p.add_argument("--ckpt_base", default=r"D:\First year\Papers\GeoAgent\checkpoints")
p.add_argument("--task", required=True, choices=["PorFac", "PermFac"])
p.add_argument("--direction", required=True, choices=["AB", "BA"])
p.add_argument("--split", required=True, choices=["val", "test"])
p.add_argument("--epoch", default="0100", help="checkpoint epoch tag, e.g. 0100")
p.add_argument("--n", type=int, default=0, help="limit #patches (0 = all). Use 32 for a smoke test")
p.add_argument("--mc", type=int, default=8, help="MC-dropout passes")
p.add_argument("--bs", type=int, default=32)
p.add_argument("--out_dir", default=r"D:\First year\Papers\GeoAgent\revision_ACAGS\signals")
a = p.parse_args()

sys.path.insert(0, a.code_dir); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nb_model import Pix2PixModel   # notebook architecture = checkpoint layer names
from geoagent_dataset import PairedGeoDataset

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
rev = "BA" if a.direction == "AB" else "AB"

def ck(d):
    return os.path.join(a.ckpt_base, f"{a.task}_{d}", f"pix2pix_{d}_ep{a.epoch}.pth")
F = Pix2PixModel(device=dev); F.load(ck(a.direction)); F.G.eval()
R = Pix2PixModel(device=dev); R.load(ck(rev));         R.G.eval()

ds = PairedGeoDataset(os.path.join(a.data_base, a.task, a.split),
                      direction=a.direction, image_size=256, augment=False)
if a.n > 0:
    ds.files = ds.files[:a.n]
dl = DataLoader(ds, batch_size=a.bs, shuffle=False, num_workers=0)

# --- SSIM exactly as in geoagent_agent.compute_ssim (global, per image, 3 channels) ---
def ssim_g(p_, t_):
    p_ = (p_ + 1) / 2; t_ = (t_ + 1) / 2
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    d = (1, 2, 3)
    mp, mt = p_.mean(d), t_.mean(d)
    sp, st = p_.std(d, unbiased=False), t_.std(d, unbiased=False)
    spt = ((p_ - mp[:, None, None, None]) * (t_ - mt[:, None, None, None])).mean(d)
    return ((2 * mp * mt + C1) * (2 * spt + C2)) / ((mp ** 2 + mt ** 2 + C1) * (sp ** 2 + st ** 2 + C2))

# --- same 5 transforms as the repository's tta_predict ---
TT = [(lambda x: x, lambda x: x),
      (lambda x: torch.flip(x, [-1]), lambda x: torch.flip(x, [-1])),
      (lambda x: torch.flip(x, [-2]), lambda x: torch.flip(x, [-2])),
      (lambda x: torch.flip(x, [-1, -2]), lambda x: torch.flip(x, [-1, -2])),
      (lambda x: torch.rot90(x, 1, [-2, -1]), lambda x: torch.rot90(x, -1, [-2, -1]))]

def mc_dropout_on(G):
    G.eval()
    for m in G.modules():
        if isinstance(m, nn.Dropout):
            m.train()

rows = []; t0 = time.time()
with torch.no_grad():
    for bi, b in enumerate(dl):
        x = b["input"].to(dev); y = b["target"].to(dev)
        members = torch.stack([inv(F.G(fw(x))) for fw, inv in TT])      # (5,B,3,H,W)
        p5 = members.mean(0); p1 = members[0]
        tta_cons = torch.stack([ssim_g(m, p5) for m in members]).mean(0)
        tta_std = members.std(0, unbiased=False).mean((1, 2, 3))
        mc_dropout_on(F.G)
        mcs = torch.stack([F.G(x) for _ in range(a.mc)])
        F.G.eval()
        mc_std = mcs.std(0, unbiased=False).mean((1, 2, 3))
        mc_cons = ssim_g(mcs.mean(0), p5)
        cyc = ssim_g(R.G(p5), x)                                          # cycle-consistency
        st, ss = ssim_g(p5, y), ssim_g(p1, y)
        for i in range(x.shape[0]):
            rows.append(dict(file=os.path.basename(b["path"][i]),
                             ssim_true=float(st[i]), ssim_single=float(ss[i]),
                             tta_cons=float(tta_cons[i]), tta_std=float(tta_std[i]),
                             mc_std=float(mc_std[i]), mc_cons=float(mc_cons[i]),
                             cycle=float(cyc[i])))
        print(f"batch {bi+1}/{len(dl)}  mean ssim_true so far {np.mean([r['ssim_true'] for r in rows]):.4f}"
              f"  ({time.time()-t0:.0f}s)", flush=True)

os.makedirs(a.out_dir, exist_ok=True)
out = os.path.join(a.out_dir, f"{a.task}_{a.direction}_{a.split}.csv")
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print("\nSaved", out, "| n =", len(rows))
print(f"SANITY (compare with paper, TTAx5 on full test split): mean ssim_true = "
      f"{np.mean([r['ssim_true'] for r in rows]):.4f}   mean single-pass = {np.mean([r['ssim_single'] for r in rows]):.4f}")
