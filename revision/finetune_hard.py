"""
finetune_hard.py -- E3: agent-driven refinement of the backbone (Reviewer 1, comment 5).
Protocol (active-learning view of the deployment loop):
  1. The reference-free gate (cons_tta8, no ground truth) flags the least-reliable --q fraction of the VAL pool.
  2. Those flagged patches get their labels (as if an expert/well data supplied them) and the backbone is fine-tuned on
     them, mixed 1:1 with replay patches from TRAIN (limits forgetting).
  3. CONTROL (--mode random): identical budget, same recipe, but randomly chosen VAL patches.
  4. Both are evaluated on TEST with TTA x8; the analysis reports gains on the gate-flagged ("hard") test patches
     and on the remaining patches. The TEST hard set is defined by the BASE model's gate score, identical for all runs.
Run (geoldm_gpu), e.g.:  python finetune_hard.py --task PermFac --direction AB --mode hard   --seed 0
                         python finetune_hard.py --task PermFac --direction AB --mode random --seed 0
Needs policy\{task}_{direction}_val.csv (from gate_policy.py).
"""
import os, sys, argparse, csv, time, random
import numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader, Dataset, ConcatDataset, Subset
p = argparse.ArgumentParser()
p.add_argument("--code_dir", default=r"D:\First year\Papers\GeoAgent")
p.add_argument("--data_base", default=r"E:\2D Dutch's Dataset\patches_512")
p.add_argument("--ckpt_base", default=r"D:\First year\Papers\GeoAgent\checkpoints")
p.add_argument("--rev_dir", default=r"D:\First year\Papers\GeoAgent\revision_ACAGS")
p.add_argument("--task", required=True, choices=["PorFac", "PermFac"])
p.add_argument("--direction", required=True, choices=["AB", "BA"])
p.add_argument("--mode", required=True, choices=["hard", "random", "replay"],
               help="replay = control with NO new val data: same number of extra TRAIN patches, same recipe")
p.add_argument("--budget", type=float, default=0.10, help="fraction of the VAL pool flagged / sampled")
p.add_argument("--epochs", type=int, default=5); p.add_argument("--lr", type=float, default=5e-5)
p.add_argument("--seed", type=int, default=0); p.add_argument("--bs", type=int, default=16)
p.add_argument("--epoch", default="0100")
a = p.parse_args()
sys.path.insert(0, a.code_dir); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nb_model import Pix2PixModel
from geoagent_dataset import PairedGeoDataset
random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
root = os.path.join(a.data_base, a.task)
M = Pix2PixModel(device=dev, lr=a.lr)
M.load(os.path.join(a.ckpt_base, f"{a.task}_{a.direction}", f"pix2pix_{a.direction}_ep{a.epoch}.pth"))
for g in M.oG.param_groups: g["lr"] = a.lr
for g in M.oD.param_groups: g["lr"] = a.lr

val_ds = PairedGeoDataset(os.path.join(root, "val"), direction=a.direction, image_size=256, augment=True)
tr_ds = PairedGeoDataset(os.path.join(root, "train"), direction=a.direction, image_size=256, augment=True)
vcsv = pd.read_csv(os.path.join(a.rev_dir, "policy", f"{a.task}_{a.direction}_val.csv"))
name2idx = {os.path.basename(f): i for i, f in enumerate(val_ds.files)}
k = int(round(a.budget * len(vcsv)))
if a.mode == "hard":
    chosen = vcsv.sort_values("cons_tta8").file.iloc[:k].tolist()          # lowest reference-free consistency
elif a.mode == "random":
    chosen = vcsv.sample(n=k, random_state=a.seed).file.tolist()
else:
    chosen = []
idx = [name2idx[f] for f in chosen]
replay = random.sample(range(len(tr_ds)), k if a.mode != "replay" else 2 * k)
ds = ConcatDataset([Subset(val_ds, idx), Subset(tr_ds, replay)])
dl = DataLoader(ds, batch_size=a.bs, shuffle=True, num_workers=0, drop_last=True)
print(f"mode={a.mode} seed={a.seed} budget={a.budget}: {len(idx)} {a.mode} val patches + {len(replay)} train replay, {a.epochs} epochs, lr={a.lr}")
t0 = time.time()
for ep in range(a.epochs):
    M.G.train(); M.D.train(); L = []
    for b in dl:
        L.append(M.train_step(b["input"], b["target"])["loss_G_l1"])
    print(f"  epoch {ep+1}/{a.epochs}  L1 term {np.mean(L):.3f}  ({time.time()-t0:.0f}s)", flush=True)
M.G.eval()

def d4(k_, f):
    fw = lambda x: torch.rot90(torch.flip(x, [-1]) if f else x, k_, [-2, -1])
    iv = lambda y: (torch.flip(torch.rot90(y, -k_, [-2, -1]), [-1]) if f else torch.rot90(y, -k_, [-2, -1]))
    return fw, iv
T8 = [d4(k_, f) for f in (False, True) for k_ in range(4)]
def ssim_g(p_, t_):
    p_ = (p_ + 1) / 2; t_ = (t_ + 1) / 2; C1, C2 = 0.01 ** 2, 0.03 ** 2; d = (1, 2, 3)
    mp, mt = p_.mean(d), t_.mean(d); sp, st = p_.std(d, unbiased=False), t_.std(d, unbiased=False)
    spt = ((p_ - mp[:, None, None, None]) * (t_ - mt[:, None, None, None])).mean(d)
    return ((2 * mp * mt + C1) * (2 * spt + C2)) / ((mp ** 2 + mt ** 2 + C1) * (sp ** 2 + st ** 2 + C2))
te = PairedGeoDataset(os.path.join(root, "test"), direction=a.direction, image_size=256, augment=False)
rows = []
with torch.no_grad():
    for b in DataLoader(te, batch_size=32, shuffle=False, num_workers=0):
        x = b["input"].to(dev); y = b["target"].to(dev)
        mem = torch.stack([iv(M.G(fw(x))) for fw, iv in T8]); m = mem.mean(0)
        s = ssim_g(m, y); c = torch.stack([ssim_g(q_, m) for q_ in mem]).mean(0)
        for i in range(x.shape[0]):
            rows.append(dict(file=os.path.basename(b["path"][i]), ssim_ft=float(s[i]), cons_ft=float(c[i])))
od = os.path.join(a.rev_dir, "finetune"); os.makedirs(od, exist_ok=True)
sfx = "" if abs(a.budget - 0.10) < 1e-9 else f"_b{a.budget}"
fn = os.path.join(od, f"{a.task}_{a.direction}_{a.mode}_s{a.seed}{sfx}.csv")
with open(fn, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["file", "ssim_ft", "cons_ft"]); w.writeheader(); w.writerows(rows)
print("Saved", fn, "| test mean SSIM (TTAx8) after fine-tune =", round(float(np.mean([r['ssim_ft'] for r in rows])), 4))
