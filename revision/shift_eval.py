"""
shift_eval.py -- the agent under orientation shift.  For each task/direction/orientation set rendered by shift_render.py: TTA x8 prediction, geologic risk (same
definitions as ablation_risk.py), reference-free consistency, percentile vs the VAL calibration set, and the gate decision (accept >=25, review 10-25, abstain <10).
Run (geoldm_gpu):  python shift_eval.py --task PorFac --direction AB       (all: run_shift.bat)   Output: geo6\{task}_{dir}_{orient}.csv, shift_results.txt via shift_summary.py
"""
import os, sys, argparse, csv
import numpy as np, pandas as pd, torch, torch.nn.functional as Fn
from torch.utils.data import DataLoader
p = argparse.ArgumentParser()
p.add_argument("--code_dir", default=r"D:\First year\Papers\GeoAgent"); p.add_argument("--ckpt_base", default=r"D:\First year\Papers\GeoAgent\checkpoints"); p.add_argument("--rev_dir", default=r"D:\First year\Papers\GeoAgent\revision_ACAGS")
p.add_argument("--task", required=True, choices=["PorFac", "PermFac"]); p.add_argument("--direction", required=True, choices=["AB", "BA"]); p.add_argument("--epoch", default="0100"); p.add_argument("--set", default="")
a = p.parse_args(); sys.path.insert(0, a.code_dir); sys.path.insert(0, a.rev_dir)
from nb_model import Pix2PixModel
from geoagent_dataset import PairedGeoDataset
import legend_decode as L
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
M = Pix2PixModel(device=dev); M.load(os.path.join(a.ckpt_base, f"{a.task}_{a.direction}", f"pix2pix_{a.direction}_ep{a.epoch}.pth")); M.G.eval()
FAC = torch.tensor(L.FACIES_RGB, device=dev)
if a.direction == "BA": LUT = torch.tensor(L.PORO_RGB if a.task == "PorFac" else L.PERM_RGB, device=dev); VAL = torch.tensor(L.PORO_VAL if a.task == "PorFac" else L.PERM_LOG10, dtype=torch.float32, device=dev); NB = VAL.shape[0]; edge = int(0.02 * NB)
rgb = lambda x: ((x.clamp(-1, 1) + 1) / 2 * 255).permute(0, 2, 3, 1)
def nearest(px, lut, chunk=65536):
    shp = px.shape[:-1]; px = px.reshape(-1, 3); idx = torch.empty(px.shape[0], dtype=torch.long, device=px.device); dist = torch.empty(px.shape[0], device=px.device)
    for i in range(0, px.shape[0], chunk): m, ix = torch.cdist(px[i:i + chunk], lut).min(1); idx[i:i + chunk] = ix; dist[i:i + chunk] = m
    return idx.view(shp), dist.view(shp)
def boundary(c, k=5): c = c.float().unsqueeze(1); return (Fn.max_pool2d(c, k, 1, k // 2) != -Fn.max_pool2d(-c, k, 1, k // 2)).squeeze(1)
def d4(k_, f):
    fw = lambda x: torch.rot90(torch.flip(x, [-1]) if f else x, k_, [-2, -1]); iv = lambda y: (torch.flip(torch.rot90(y, -k_, [-2, -1]), [-1]) if f else torch.rot90(y, -k_, [-2, -1])); return fw, iv
T8 = [d4(k_, f) for f in (False, True) for k_ in range(4)]
def ssim_g(p_, t_):
    p_ = (p_ + 1) / 2; t_ = (t_ + 1) / 2; C1, C2 = 0.01 ** 2, 0.03 ** 2; d = (1, 2, 3); mp, mt = p_.mean(d), t_.mean(d); sp, st = p_.std(d, unbiased=False), t_.std(d, unbiased=False)
    spt = ((p_ - mp[:, None, None, None]) * (t_ - mt[:, None, None, None])).mean(d); return ((2 * mp * mt + C1) * (2 * spt + C2)) / ((mp ** 2 + mt ** 2 + C1) * (sp ** 2 + st ** 2 + C2))
def risks(pr, y):
    P, T = rgb(pr), rgb(y); s = T.sum(-1); inside = (s > 30) & (s < 740); out = []
    if a.direction == "AB":
        cp, _ = nearest(P, FAC); ct, dt = nearest(T, FAC); m = inside & (dt < 60); mt = m & ~boundary(ct)
        for i in range(P.shape[0]): out.append(float(1 - (cp[i][mt[i]] == ct[i][mt[i]]).float().mean()) if (int(m[i].sum()) >= 500 and int(mt[i].sum()) > 200) else float("nan"))
    else:
        ip, _ = nearest(P, LUT); it, dt = nearest(T, LUT); ok = inside & (dt < 45) & (it >= edge) & (it < NB - edge)
        for i in range(P.shape[0]): out.append(float((VAL[ip[i]][ok[i]] - VAL[it[i]][ok[i]]).abs().mean()) if int(ok[i].sum()) >= 500 else float("nan"))
    return out
od = os.path.join(a.rev_dir, "geo6"); os.makedirs(od, exist_ok=True)
for orient in ("K", "I", "J"):
    ds = PairedGeoDataset(os.path.join(a.rev_dir, "shift", a.set or a.task, orient), direction=a.direction, image_size=256, augment=False); rows = []
    with torch.no_grad():
        for b in DataLoader(ds, batch_size=16, shuffle=False, num_workers=0):
            x = b["input"].to(dev); y = b["target"].to(dev); mem = torch.stack([iv(M.G(fw(x))) for fw, iv in T8]); m = mem.mean(0); r = risks(m, y); c = torch.stack([ssim_g(q_, m) for q_ in mem]).mean(0)
            for i, pa in enumerate(b["path"]): rows.append(dict(file=os.path.basename(pa), orient=orient, risk=r[i], cons=float(c[i])))
    fn = os.path.join(od, f"{a.set or a.task}_{a.direction}_{orient}.csv")
    with open(fn, "w", newline="") as f: w = csv.DictWriter(f, fieldnames=["file", "orient", "risk", "cons"]); w.writeheader(); w.writerows(rows)
    print("Saved", fn, len(rows), "| mean risk", round(float(np.nanmean([q["risk"] for q in rows])), 4), flush=True)
