"""
lambda_sweep.py -- sensitivity of the reconstruction weight lambda (Reviewer 1, comment 3).  The original value lambda=100 was adopted from Pix2Pix (Isola et al., 2017) and was not tuned.
Trains the same backbone with a different lambda (same recipe: 100 epochs, batch 32, Adam 2e-4, beta1 0.5, flips augmentation) and scores it on the VALIDATION split with the same geologic risk
(single pass and TTA x8).  lambda=100 reference = existing checkpoint, read from geo5\{task}_{dir}_val.csv.   Selection uses VAL only.
Run (geoldm_gpu):  python lambda_sweep.py --task PermFac --direction AB --lam 10      (also --lam 300; ~1-3 h per model; saves checkpoint + lambda\*.json; restartable: skips if json exists)
"""
import os, sys, argparse, json, time
import numpy as np, pandas as pd, torch, torch.nn.functional as Fn
from torch.utils.data import DataLoader
p = argparse.ArgumentParser()
p.add_argument("--code_dir", default=r"D:\First year\Papers\GeoAgent"); p.add_argument("--data_base", default=r"E:\2D Dutch's Dataset\patches_512"); p.add_argument("--rev_dir", default=r"D:\First year\Papers\GeoAgent\revision_ACAGS")
p.add_argument("--task", required=True, choices=["PorFac", "PermFac"]); p.add_argument("--direction", required=True, choices=["AB", "BA"]); p.add_argument("--lam", type=float, required=True)
p.add_argument("--epochs", type=int, default=100); p.add_argument("--bs", type=int, default=32); p.add_argument("--seed", type=int, default=0)
a = p.parse_args(); sys.path.insert(0, a.code_dir); sys.path.insert(0, a.rev_dir)
from nb_model import Pix2PixModel
from geoagent_dataset import PairedGeoDataset
import legend_decode as L
od = os.path.join(a.rev_dir, "lambda"); os.makedirs(od, exist_ok=True); tag = f"{a.task}_{a.direction}_lam{a.lam:g}"; fj = os.path.join(od, tag + ".json")
if os.path.exists(fj): sys.exit("already done: " + fj)
if __name__ == "__main__":
    torch.manual_seed(a.seed); np.random.seed(a.seed); dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = os.path.join(a.data_base, a.task); tr = PairedGeoDataset(os.path.join(root, "train"), direction=a.direction, image_size=256, augment=True)
    dl = DataLoader(tr, batch_size=a.bs, shuffle=True, num_workers=6, persistent_workers=True, prefetch_factor=4, drop_last=True); M = Pix2PixModel(device=dev, lambda_l1=a.lam); t0 = time.time()
    for ep in range(a.epochs):
        M.G.train(); M.D.train(); Ls = [M.train_step(b["input"], b["target"])["loss_G_l1"] / a.lam for b in dl]; print(f"epoch {ep+1}/{a.epochs} L1 {np.mean(Ls):.4f} ({time.time()-t0:.0f}s)", flush=True)
    os.makedirs(os.path.join(a.rev_dir, "lambda_ckpt"), exist_ok=True); M.save(os.path.join(a.rev_dir, "lambda_ckpt", tag + ".pth")); M.G.eval()
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
    def risks(pr, y):
        P, T = rgb(pr), rgb(y); s = T.sum(-1); inside = (s > 30) & (s < 740); out = []
        if a.direction == "AB":
            cp, _ = nearest(P, FAC); ct, dt = nearest(T, FAC); m = inside & (dt < 60); mt = m & ~boundary(ct)
            for i in range(P.shape[0]): out.append(float(1 - (cp[i][mt[i]] == ct[i][mt[i]]).float().mean()) if (int(m[i].sum()) >= 500 and int(mt[i].sum()) > 200) else float("nan"))
        else:
            ip, _ = nearest(P, LUT); it, dt = nearest(T, LUT); ok = inside & (dt < 45) & (it >= edge) & (it < NB - edge)
            for i in range(P.shape[0]): out.append(float((VAL[ip[i]][ok[i]] - VAL[it[i]][ok[i]]).abs().mean()) if int(ok[i].sum()) >= 500 else float("nan"))
        return out
    va = PairedGeoDataset(os.path.join(root, "val"), direction=a.direction, image_size=256, augment=False); rs, rt = [], []
    with torch.no_grad():
        for b in DataLoader(va, batch_size=16, shuffle=False, num_workers=0):
            x = b["input"].to(dev); y = b["target"].to(dev); rs += risks(M.G(x), y); rt += risks(torch.stack([iv(M.G(fw(x))) for fw, iv in T8]).mean(0), y)
    ref = pd.read_csv(os.path.join(a.rev_dir, "geo5", f"{a.task}_{a.direction}_val.csv"))
    res = dict(task=a.task, direction=a.direction, lam=a.lam, val_risk_single=float(np.nanmean(rs)), val_risk_tta8=float(np.nanmean(rt)), ref_lam100_single=float(ref.risk_single.mean()), ref_lam100_tta8=float(ref.risk_tta8.mean()))
    json.dump(res, open(fj, "w"), indent=1); print(res)
