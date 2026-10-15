"""
baselines.py -- conventional baselines scored with the SAME geological metrics, masks and decoder as the GAN (geo_metrics.py).
 Facies -> property (BA): CLASS-MEAN LOOKUP  : every pixel gets the mean decoded property of its facies class (statistics from a TRAIN subsample).
 Property -> facies (AB): BAYES (MAP) CLASSIFIER on the decoded property value (40 bins; statistics from a TRAIN subsample).
No training of any network. Outputs geo3/{task}_{direction}_test.csv with per-patch metrics.
Run (geoldm_gpu):  python baselines.py --task PorFac --direction AB
"""
import os, sys, argparse, csv, time, random
import numpy as np, torch, torch.nn.functional as Fn
p = argparse.ArgumentParser()
p.add_argument("--code_dir", default=r"D:\First year\Papers\GeoAgent"); p.add_argument("--data_base", default=r"E:\2D Dutch's Dataset\patches_512"); p.add_argument("--rev_dir", default=r"D:\First year\Papers\GeoAgent\revision_ACAGS")
p.add_argument("--task", required=True, choices=["PorFac", "PermFac"]); p.add_argument("--direction", required=True, choices=["AB", "BA"])
p.add_argument("--ntrain", type=int, default=3000); p.add_argument("--n", type=int, default=0); p.add_argument("--seed", type=int, default=0); p.add_argument("--bins", type=int, default=40)
a = p.parse_args()
sys.path.insert(0, a.code_dir); sys.path.insert(0, a.rev_dir)
from geoagent_dataset import PairedGeoDataset
import legend_decode as L
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
rgb = lambda x: ((x.clamp(-1, 1) + 1) / 2 * 255).permute(0, 2, 3, 1)
def nearest(px, lut, chunk=65536):
    shp = px.shape[:-1]; px = px.reshape(-1, 3); idx = torch.empty(px.shape[0], dtype=torch.long, device=px.device); dist = torch.empty(px.shape[0], device=px.device)
    for i in range(0, px.shape[0], chunk): m, ix = torch.cdist(px[i:i + chunk], lut).min(1); idx[i:i + chunk] = ix; dist[i:i + chunk] = m
    return idx.view(shp), dist.view(shp)
def boundary(c, k=5): c = c.float().unsqueeze(1); return (Fn.max_pool2d(c, k, 1, k // 2) != -Fn.max_pool2d(-c, k, 1, k // 2)).squeeze(1)
FAC = torch.tensor(L.FACIES_RGB, device=dev)
LUT = torch.tensor(L.PORO_RGB if a.task == "PorFac" else L.PERM_RGB, device=dev); VAL = torch.tensor(L.PORO_VAL if a.task == "PorFac" else L.PERM_LOG10, dtype=torch.float32, device=dev)
NB = VAL.shape[0]; edge = int(0.02 * NB)
root = os.path.join(a.data_base, a.task)
def loader(split, n=0, shuffle_seed=None):
    ds = PairedGeoDataset(os.path.join(root, split), direction=a.direction, image_size=256, augment=False)
    idx = list(range(len(ds)))
    if shuffle_seed is not None: random.Random(shuffle_seed).shuffle(idx)
    if n: idx = idx[:n]
    return ds, idx
def batches(ds, idx, bs=16):
    for i in range(0, len(idx), bs):
        items = [ds[j] for j in idx[i:i + bs]]
        yield torch.stack([it["input"] for it in items]).to(dev), torch.stack([it["target"] for it in items]).to(dev), [os.path.basename(it["path"]) for it in items]
# ------------------------------------------------ statistics from a TRAIN subsample
t0 = time.time(); ds, idx = loader("train", a.ntrain, a.seed)
if a.direction == "BA":                      # input = facies, target = property : class means
    s = torch.zeros(3, device=dev); c = torch.zeros(3, device=dev)
    for X, Y, _ in batches(ds, idx):
        fi, df = nearest(rgb(X), FAC); it, dt = nearest(rgb(Y), LUT); T = rgb(Y).sum(-1); ok = (T > 30) & (T < 740) & (dt < 45) & (it >= edge) & (it < NB - edge) & (df < 60)
        for k in range(3): m = ok & (fi == k); s[k] += VAL[it][m].sum(); c[k] += m.sum()
    stat = (s / c.clamp_min(1)); print("class-mean lookup from train:", [round(float(v), 4) for v in stat], "(sand, shaly sand, shale)", f"({time.time()-t0:.0f}s)")
else:                                        # input = property, target = facies : Bayes table
    cnt = torch.zeros(a.bins, 3, device=dev)
    for X, Y, _ in batches(ds, idx):
        iv, dv = nearest(rgb(X), LUT); ct, dt = nearest(rgb(Y), FAC); T = rgb(Y).sum(-1); S = rgb(X).sum(-1); ok = (T > 30) & (T < 740) & (S > 30) & (S < 740) & (dt < 60) & (dv < 45)
        b = (iv * a.bins // NB).clamp(max=a.bins - 1)
        for k in range(3): cnt[:, k] += torch.bincount(b[ok & (ct == k)], minlength=a.bins).float()
    table = cnt.argmax(1); prior = cnt.sum(0).argmax(); table[cnt.sum(1) == 0] = prior
    print("Bayes table (bin -> class) from train, class fractions:", [round(float(v), 3) for v in cnt.sum(0) / cnt.sum()], f"({time.time()-t0:.0f}s)")
# ------------------------------------------------ evaluate on TEST
ds, idx = loader("test", a.n); rows = []
for X, Y, names in batches(ds, idx):
    Pr, Tg = rgb(X), rgb(Y); T = Tg.sum(-1); inside = (T > 30) & (T < 740)
    if a.direction == "AB":
        iv, dv = nearest(Pr, LUT); b = (iv * a.bins // NB).clamp(max=a.bins - 1); cp = table[b]; ct, dt = nearest(Tg, FAC); inside = inside & (dt < 60); bt = boundary(ct)
        for i, nm in enumerate(names):
            m = inside[i]; npx = int(m.sum())
            if npx < 500: continue
            mt = m & ~bt[i]; accs = lambda mm: float((cp[i][mm] == ct[i][mm]).float().mean())
            prop = np.mean([abs(float(((cp[i][m] == k).float().mean() - (ct[i][m] == k).float().mean()))) for k in range(3)])
            rows.append(dict(file=nm, pix_acc=accs(m), pix_acc_tol=accs(mt) if int(mt.sum()) > 200 else float("nan"), prop_err=float(prop)))
    else:
        fi, df = nearest(Pr, FAC); it, dt = nearest(Tg, LUT); ok = inside & (dt < 45) & (it >= edge) & (it < NB - edge); pv = stat[fi.clamp(0, 2)]
        for i, nm in enumerate(names):
            m = ok[i]
            if int(m.sum()) < 500: continue
            rows.append(dict(file=nm, mae=float((pv[i][m] - VAL[it][i][m]).abs().mean())))
od = os.path.join(a.rev_dir, "geo3"); os.makedirs(od, exist_ok=True); fn = os.path.join(od, f"{a.task}_{a.direction}_test.csv"); keys = list(rows[0].keys())
with open(fn, "w", newline="") as f: w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
print("Saved", fn, len(rows), "| mean:", {k: round(float(np.nanmean([r[k] for r in rows])), 4) for k in keys if k != "file"})
