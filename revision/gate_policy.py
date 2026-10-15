"""
gate_policy.py -- ACAGS-D-26-00312, Reviewer 2 comments 2-3, Reviewer 1 comment 4.
For EVERY patch of a split, computes the output of four well-defined actions and, for each,
the true SSIM/PSNR (evaluation only) and a reference-free consistency score (what the gate sees):
  single : one forward pass
  tta5   : mean of 5 transforms (identity, h-flip, v-flip, both flips, rot90)  [paper's default]
  tta8   : mean over the full 8-element symmetry group (4 rotations x optional h-flip)
  mc10   : mean of 10 MC-dropout passes (dropout active, BatchNorm in eval)   ["average realisations"]
No policy is decided here: policy_analysis.py simulates fixed / escalating / oracle policies from this table.
Run (geoldm_gpu env):  python gate_policy.py --task PorFac --direction AB --split val
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
p.add_argument("--epoch", default="0100")
p.add_argument("--n", type=int, default=0)
p.add_argument("--mc", type=int, default=10)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--bs", type=int, default=32)
p.add_argument("--out_dir", default=r"D:\First year\Papers\GeoAgent\revision_ACAGS\policy")
a = p.parse_args()

sys.path.insert(0, a.code_dir); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nb_model import Pix2PixModel
from geoagent_dataset import PairedGeoDataset
torch.manual_seed(a.seed); np.random.seed(a.seed)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
F = Pix2PixModel(device=dev)
F.load(os.path.join(a.ckpt_base, f"{a.task}_{a.direction}", f"pix2pix_{a.direction}_ep{a.epoch}.pth")); F.G.eval()
ds = PairedGeoDataset(os.path.join(a.data_base, a.task, a.split), direction=a.direction, image_size=256, augment=False)
if a.n > 0: ds.files = ds.files[:a.n]
dl = DataLoader(ds, batch_size=a.bs, shuffle=False, num_workers=0)

def ssim_g(p_, t_):                      # identical to geoagent_agent.compute_ssim (global SSIM)
    p_ = (p_ + 1) / 2; t_ = (t_ + 1) / 2
    C1, C2 = 0.01 ** 2, 0.03 ** 2; d = (1, 2, 3)
    mp, mt = p_.mean(d), t_.mean(d); sp, st = p_.std(d, unbiased=False), t_.std(d, unbiased=False)
    spt = ((p_ - mp[:, None, None, None]) * (t_ - mt[:, None, None, None])).mean(d)
    return ((2 * mp * mt + C1) * (2 * spt + C2)) / ((mp ** 2 + mt ** 2 + C1) * (sp ** 2 + st ** 2 + C2))
def psnr(p_, t_):
    mse = (((p_ + 1) / 2 - (t_ + 1) / 2) ** 2).mean((1, 2, 3)).clamp_min(1e-10)
    return -10 * torch.log10(mse)

T5 = [(lambda x: x, lambda x: x), (lambda x: torch.flip(x, [-1]), lambda x: torch.flip(x, [-1])),
      (lambda x: torch.flip(x, [-2]), lambda x: torch.flip(x, [-2])),
      (lambda x: torch.flip(x, [-1, -2]), lambda x: torch.flip(x, [-1, -2])),
      (lambda x: torch.rot90(x, 1, [-2, -1]), lambda x: torch.rot90(x, -1, [-2, -1]))]
def d4(k, f):                             # rot90^k after optional h-flip ; exact inverse
    fw = lambda x: torch.rot90(torch.flip(x, [-1]) if f else x, k, [-2, -1])
    iv = lambda y: (torch.flip(torch.rot90(y, -k, [-2, -1]), [-1]) if f else torch.rot90(y, -k, [-2, -1]))
    return fw, iv
T8 = [d4(k, f) for f in (False, True) for k in range(4)]

def mc_on(G):
    G.eval()
    for m in G.modules():
        if isinstance(m, nn.Dropout): m.train()

def cons(members, mean):                  # reference-free score = mean SSIM(member, ensemble mean)
    return torch.stack([ssim_g(m, mean) for m in members]).mean(0)

rows = []; t0 = time.time()
with torch.no_grad():
    for bi, b in enumerate(dl):
        x = b["input"].to(dev); y = b["target"].to(dev)
        F.G.eval()
        m5 = torch.stack([iv(F.G(fw(x))) for fw, iv in T5]); p5 = m5.mean(0)
        m8 = torch.stack([iv(F.G(fw(x))) for fw, iv in T8]); p8 = m8.mean(0)
        p1 = m5[0]
        mc_on(F.G); mm = torch.stack([F.G(x) for _ in range(a.mc)]); F.G.eval(); pm = mm.mean(0)
        out = {"single": p1, "tta5": p5, "tta8": p8, "mc10": pm}
        sc = {"tta5": cons(m5, p5), "tta8": cons(m8, p8), "mc10": cons(mm, pm)}
        S = {k: ssim_g(v, y) for k, v in out.items()}; P = {k: psnr(v, y) for k, v in out.items()}
        for i in range(x.shape[0]):
            r = {"file": os.path.basename(b["path"][i])}
            for k in out: r[f"ssim_{k}"] = float(S[k][i]); r[f"psnr_{k}"] = float(P[k][i])
            for k in sc: r[f"cons_{k}"] = float(sc[k][i])
            rows.append(r)
        print(f"batch {bi+1}/{len(dl)}  mean true SSIM: " +
              "  ".join(f"{k}={np.mean([r['ssim_'+k] for r in rows]):.4f}" for k in out) + f"  ({time.time()-t0:.0f}s)", flush=True)
os.makedirs(a.out_dir, exist_ok=True)
fn = os.path.join(a.out_dir, f"{a.task}_{a.direction}_{a.split}.csv")
with open(fn, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print("\nSaved", fn, "n =", len(rows))
