"""
geo_metrics.py -- geological quality of every TEST patch, decoded through the ORIGINAL legend (legend_decode.py).
Prediction = default tool (TTA x8). Evaluated only on pixels that are inside the mapped field in the TARGET.
 AB directions (property -> facies): pixel accuracy, mean IoU over the 3 classes, absolute error of facies proportions.
 BA directions (facies -> property): MAE in v/v (porosity) or log10 mD (permeability), over target pixels inside the
   legend range (saturated pixels, i.e. the top/bottom 2% of the colour bar, are excluded: they cannot be decoded).
Run (geoldm_gpu):  python geo_metrics.py --task PorFac --direction AB
"""
import os, sys, argparse, csv, time
import numpy as np, torch
from torch.utils.data import DataLoader
p = argparse.ArgumentParser()
p.add_argument("--code_dir", default=r"D:\First year\Papers\GeoAgent")
p.add_argument("--data_base", default=r"E:\2D Dutch's Dataset\patches_512")
p.add_argument("--ckpt_base", default=r"D:\First year\Papers\GeoAgent\checkpoints")
p.add_argument("--rev_dir", default=r"D:\First year\Papers\GeoAgent\revision_ACAGS")
p.add_argument("--task", required=True, choices=["PorFac", "PermFac"]); p.add_argument("--direction", required=True, choices=["AB", "BA"])
p.add_argument("--split", default="test"); p.add_argument("--epoch", default="0100"); p.add_argument("--n", type=int, default=0)
p.add_argument("--bs", type=int, default=16)
a = p.parse_args()
sys.path.insert(0, a.code_dir); sys.path.insert(0, a.rev_dir)
from nb_model import Pix2PixModel
from geoagent_dataset import PairedGeoDataset
import legend_decode as L
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
M = Pix2PixModel(device=dev); M.load(os.path.join(a.ckpt_base, f"{a.task}_{a.direction}", f"pix2pix_{a.direction}_ep{a.epoch}.pth")); M.G.eval()
ds = PairedGeoDataset(os.path.join(a.data_base, a.task, a.split), direction=a.direction, image_size=256, augment=False)
if a.n > 0: ds.files = ds.files[:a.n]
def d4(k, f):
    fw = lambda x: torch.rot90(torch.flip(x, [-1]) if f else x, k, [-2, -1])
    iv = lambda y: (torch.flip(torch.rot90(y, -k, [-2, -1]), [-1]) if f else torch.rot90(y, -k, [-2, -1]))
    return fw, iv
T8 = [d4(k, f) for f in (False, True) for k in range(4)]
def to_rgb(x): return ((x.clamp(-1, 1) + 1) / 2 * 255).permute(0, 2, 3, 1)               # (B,H,W,3) float 0..255
def nearest(rgb, lut, chunk=65536):
    px = rgb.reshape(-1, 3); out = torch.empty(px.shape[0], dtype=torch.long, device=px.device); dist = torch.empty(px.shape[0], device=px.device)
    for i in range(0, px.shape[0], chunk):
        d = torch.cdist(px[i:i + chunk], lut); m, ix = d.min(1); out[i:i + chunk] = ix; dist[i:i + chunk] = m
    return out.view(rgb.shape[:-1]), dist.view(rgb.shape[:-1])
LUT = {"fac": torch.tensor(L.FACIES_RGB, device=dev)}
if a.direction == "BA":
    if a.task == "PorFac": LUT["lut"] = torch.tensor(L.PORO_RGB, device=dev); VAL = torch.tensor(L.PORO_VAL, dtype=torch.float32, device=dev)
    else:                  LUT["lut"] = torch.tensor(L.PERM_RGB, device=dev); VAL = torch.tensor(L.PERM_LOG10, dtype=torch.float32, device=dev)
    n = VAL.shape[0]; edge = int(0.02 * n)
import torch.nn.functional as Fn
def boundary(c, k=5):                       # True where the class map is not constant in a k x k neighbourhood
    c = c.float().unsqueeze(1); mx = Fn.max_pool2d(c, k, 1, k // 2); mn = -Fn.max_pool2d(-c, k, 1, k // 2); return (mx != mn).squeeze(1)
def isolated(c):                            # pixel whose class differs from all 4 neighbours
    cc = Fn.pad(c.float().unsqueeze(1), (1, 1, 1, 1), mode="replicate")
    nb = [cc[..., :-2, 1:-1], cc[..., 2:, 1:-1], cc[..., 1:-1, :-2], cc[..., 1:-1, 2:]]; me = cc[..., 1:-1, 1:-1]
    return torch.stack([(n != me) for n in nb]).all(0).squeeze(1)
rows = []; t0 = time.time()
with torch.no_grad():
    for bi, b in enumerate(DataLoader(ds, batch_size=a.bs, shuffle=False, num_workers=0)):
        x = b["input"].to(dev); y = b["target"].to(dev)
        pr = torch.stack([iv(M.G(fw(x))) for fw, iv in T8]).mean(0)
        P, T, X = to_rgb(pr), to_rgb(y), to_rgb(x); s = T.sum(-1); inside = (s > 30) & (s < 740)            # target field mask
        si = X.sum(-1); inmask = (si > 30) & (si < 740)                                          # field mask known at deployment (from the INPUT)
        if a.direction == "AB":
            cp, _ = nearest(P, LUT["fac"]); ct, dt = nearest(T, LUT["fac"]); inside = inside & (dt < 60); bt = boundary(ct); bp = boundary(cp); ip_ = isolated(cp)
            for i in range(x.shape[0]):
                m = inside[i]; npx = int(m.sum())
                if npx < 500: rows.append(dict(file=os.path.basename(b["path"][i]), n_valid=npx)); continue
                pp, tt = cp[i][m], ct[i][m]; acc = float((pp == tt).float().mean()); ious = []
                for c in range(3):
                    inter = ((pp == c) & (tt == c)).sum().item(); uni = ((pp == c) | (tt == c)).sum().item()
                    if uni > 0: ious.append(inter / uni)
                prop = [abs(((pp == c).float().mean() - (tt == c).float().mean()).item()) for c in range(3)]
                mt = m & ~bt[i]; acc_tol = float((cp[i][mt] == ct[i][mt]).float().mean()) if int(mt.sum()) > 200 else float("nan")
                fm = inmask[i]; nf = max(int(fm.sum()), 1)
                rows.append(dict(file=os.path.basename(b["path"][i]), n_valid=npx, pix_acc=acc, pix_acc_tol=acc_tol, miou=float(np.mean(ious)), prop_err=float(np.mean(prop)),
                                 f_fr0=float(((cp[i] == 0) & fm).sum() / nf), f_fr1=float(((cp[i] == 1) & fm).sum() / nf), f_fr2=float(((cp[i] == 2) & fm).sum() / nf),
                                 f_bdry=float((bp[i] & fm).sum() / nf), f_speckle=float((ip_[i] & fm).sum() / nf)))
        else:
            ip, _ = nearest(P, LUT["lut"]); it, dt = nearest(T, LUT["lut"]); fp, _ = nearest(T, LUT["fac"])
            ok = inside & (dt < 45) & (it >= edge) & (it < n - edge)
            vp, vt = VAL[ip], VAL[it]; fi, _ = nearest(X, LUT["fac"]); bi_ = boundary(fi)
            for i in range(x.shape[0]):
                m = ok[i]; npx = int(m.sum())
                if npx < 500: rows.append(dict(file=os.path.basename(b["path"][i]), n_valid=npx)); continue
                e = (vp[i][m] - vt[i][m]).abs(); fm = inmask[i]; nf = max(int(fm.sum()), 1); pv = vp[i]
                gx = (pv[:, 1:] - pv[:, :-1]).abs().mean().item(); gy = (pv[1:, :] - pv[:-1, :]).abs().mean().item()
                rows.append(dict(file=os.path.basename(b["path"][i]), n_valid=npx, mae=float(e.mean()), rmse=float(e.pow(2).mean().sqrt()), bias=float((vp[i][m] - vt[i][m]).mean()),
                                 f_pmean=float(pv[fm].mean()), f_pstd=float(pv[fm].std()), f_rough=float(gx + gy), f_sat=float(((ip[i] < edge) | (ip[i] >= n - edge))[fm].float().mean()),
                                 f_in0=float(((fi[i] == 0) & fm).sum() / nf), f_in1=float(((fi[i] == 1) & fm).sum() / nf), f_in2=float(((fi[i] == 2) & fm).sum() / nf),
                                 f_inbdry=float((bi_[i] & fm).sum() / nf)))
        if bi % 20 == 0: print(f"batch {bi+1}/{len(ds)//a.bs+1} ({time.time()-t0:.0f}s)", flush=True)
od = os.path.join(a.rev_dir, "geo2"); os.makedirs(od, exist_ok=True)
fn = os.path.join(od, f"{a.task}_{a.direction}_{a.split}.csv"); keys = sorted({k for r in rows for k in r}, key=lambda k: (k != "file", k))
with open(fn, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
print("Saved", fn, len(rows), "| mean:", {k: round(float(np.nanmean([r[k] for r in rows if k in r])), 4) for k in keys if k not in ("file",)})
