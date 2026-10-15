"""
ablation_risk.py -- per-patch geologic risk of SINGLE-PASS and TTA x8 predictions on VAL and TEST (same risk definition as gated_learning.py),
 plus a SENSITIVITY variant for BA directions that INCLUDES saturated pixels (risk_sat).  Output: geo5\{task}_{dir}_{split}.csv  (file, risk_single, risk_tta8[, sat versions])
Run (geoldm_gpu):  python ablation_risk.py --task PorFac --direction AB      (all four: run_ablation.bat)
"""
import os, sys, argparse, csv, time
import numpy as np, torch, torch.nn.functional as Fn
from torch.utils.data import DataLoader
p = argparse.ArgumentParser()
p.add_argument("--code_dir", default=r"D:\First year\Papers\GeoAgent"); p.add_argument("--data_base", default=r"E:\2D Dutch's Dataset\patches_512")
p.add_argument("--ckpt_base", default=r"D:\First year\Papers\GeoAgent\checkpoints"); p.add_argument("--rev_dir", default=r"D:\First year\Papers\GeoAgent\revision_ACAGS")
p.add_argument("--task", required=True, choices=["PorFac", "PermFac"]); p.add_argument("--direction", required=True, choices=["AB", "BA"]); p.add_argument("--epoch", default="0100"); p.add_argument("--n", type=int, default=0)
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
def risks(pr, y):                                   # returns list of (risk, risk_sat) per patch
    P, T = rgb(pr), rgb(y); s = T.sum(-1); inside = (s > 30) & (s < 740); out = []
    if a.direction == "AB":
        cp, _ = nearest(P, FAC); ct, dt = nearest(T, FAC); m = inside & (dt < 60); mt = m & ~boundary(ct)
        for i in range(P.shape[0]): out.append((float(1 - (cp[i][mt[i]] == ct[i][mt[i]]).float().mean()) if (int(m[i].sum()) >= 500 and int(mt[i].sum()) > 200) else float("nan"), float("nan")))
    else:
        ip, _ = nearest(P, LUT); it, dt = nearest(T, LUT); good = inside & (dt < 45); ok = good & (it >= edge) & (it < NB - edge)
        for i in range(P.shape[0]):
            r = float((VAL[ip[i]][ok[i]] - VAL[it[i]][ok[i]]).abs().mean()) if int(ok[i].sum()) >= 500 else float("nan")
            rs = float((VAL[ip[i]][good[i]] - VAL[it[i]][good[i]]).abs().mean()) if int(good[i].sum()) >= 500 else float("nan"); out.append((r, rs))
    return out
od = os.path.join(a.rev_dir, "geo5"); os.makedirs(od, exist_ok=True)
for split in ("val", "test"):
    ds = PairedGeoDataset(os.path.join(a.data_base, a.task, split), direction=a.direction, image_size=256, augment=False)
    if a.n: ds.files = ds.files[:a.n]
    rows = []; t0 = time.time()
    with torch.no_grad():
        for bi, b in enumerate(DataLoader(ds, batch_size=16, shuffle=False, num_workers=0)):
            x = b["input"].to(dev); y = b["target"].to(dev); rs_ = risks(M.G(x), y); rt = risks(torch.stack([iv(M.G(fw(x))) for fw, iv in T8]).mean(0), y)
            for i, pa in enumerate(b["path"]): rows.append(dict(file=os.path.basename(pa), risk_single=rs_[i][0], risk_tta8=rt[i][0], risk_sat_single=rs_[i][1], risk_sat_tta8=rt[i][1]))
            if bi % 50 == 0: print(f"{split} {bi+1}/{len(ds)//16+1} ({time.time()-t0:.0f}s)", flush=True)
    fn = os.path.join(od, f"{a.task}_{a.direction}_{split}.csv")
    with open(fn, "w", newline="") as f: w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print("Saved", fn, "| mean single", round(float(np.nanmean([r['risk_single'] for r in rows])), 4), "| mean TTAx8", round(float(np.nanmean([r['risk_tta8'] for r in rows])), 4))
