"""
baselines2.py -- conventional ML baseline WITH multi-scale spatial context (fair comparison to the GAN), same decoder/masks/metrics.
 per-pixel features from the INPUT map only: AB: decoded property value, validity mask, local means at 5/15/31 px, local std at 5/15 px
                                            BA: one-hot facies, validity mask, local class fractions at 5/15/31 px
 model: small MLP (F-128-128-out) trained on ~2000 random valid pixels per patch from a TRAIN subsample (CE for facies, L1 for property).
Run (geoldm_gpu):  python baselines2.py --task PorFac --direction AB          (smoke test: --ntrain 100 --n 32 --epochs 1)
Outputs geo3b/{task}_{direction}_test.csv   ->   python baselines_analysis.py --sub geo3b
"""
import os, sys, argparse, csv, time, random
import numpy as np, torch, torch.nn as nn, torch.nn.functional as Fn
p = argparse.ArgumentParser()
p.add_argument("--code_dir", default=r"D:\First year\Papers\GeoAgent"); p.add_argument("--data_base", default=r"E:\2D Dutch's Dataset\patches_512"); p.add_argument("--rev_dir", default=r"D:\First year\Papers\GeoAgent\revision_ACAGS")
p.add_argument("--task", required=True, choices=["PorFac", "PermFac"]); p.add_argument("--direction", required=True, choices=["AB", "BA"])
p.add_argument("--ntrain", type=int, default=3000); p.add_argument("--n", type=int, default=0); p.add_argument("--seed", type=int, default=0); p.add_argument("--epochs", type=int, default=8)
p.add_argument("--px_per_patch", type=int, default=2000)
a = p.parse_args(); torch.manual_seed(a.seed)
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
NB = VAL.shape[0]; edge = int(0.02 * NB); root = os.path.join(a.data_base, a.task)
def pool(x, k): return Fn.avg_pool2d(x, k, 1, k // 2, count_include_pad=False)
def features(X):                              # X: (B,3,H,W) in [-1,1] input map  ->  (B,F,H,W)
    R = rgb(X); S = R.sum(-1)
    if a.direction == "AB":
        iv, dv = nearest(R, LUT); mask = ((S > 30) & (S < 740) & (dv < 45)).float().unsqueeze(1); v = (iv.float() / NB).unsqueeze(1) * mask; ch = [v, mask]
        den = lambda k: pool(mask, k).clamp_min(1e-6)
        for k in (5, 15, 31): ch.append(pool(v, k) / den(k))
        for k in (5, 15):
            m1 = pool(v, k) / den(k); m2 = pool(v * v, k) / den(k); ch.append((m2 - m1 ** 2).clamp_min(0).sqrt())
    else:
        fi, df = nearest(R, FAC); mask = ((S > 30) & (S < 740) & (df < 60)).float().unsqueeze(1); oh = Fn.one_hot(fi.clamp(0, 2), 3).permute(0, 3, 1, 2).float() * mask; ch = [oh, mask]
        for k in (5, 15, 31): ch.append(pool(oh, k) / pool(mask, k).clamp_min(1e-6))
    return torch.cat(ch, 1)
def loader(split, n=0, seed=None):
    ds = PairedGeoDataset(os.path.join(root, split), direction=a.direction, image_size=256, augment=False); idx = list(range(len(ds)))
    if seed is not None: random.Random(seed).shuffle(idx)
    return ds, (idx[:n] if n else idx)
def batches(ds, idx, bs=16):
    for i in range(0, len(idx), bs):
        it = [ds[j] for j in idx[i:i + bs]]; yield torch.stack([t["input"] for t in it]).to(dev), torch.stack([t["target"] for t in it]).to(dev), [os.path.basename(t["path"]) for t in it]
def target_and_mask(Y):                       # returns target (B,H,W) and valid mask (B,H,W) following the GAN evaluation
    R = rgb(Y); T = R.sum(-1); inside = (T > 30) & (T < 740)
    if a.direction == "AB": ct, dt = nearest(R, FAC); return ct, inside & (dt < 60)
    it, dt = nearest(R, LUT); return VAL[it], inside & (dt < 45) & (it >= edge) & (it < NB - edge)
# ---------------------------------------------- build training pixels
t0 = time.time(); ds, idx = loader("train", a.ntrain, a.seed); Xs, Ys = [], []
g = torch.Generator(device="cpu").manual_seed(a.seed)
for X, Y, _ in batches(ds, idx):
    Fe = features(X); tg, ok = target_and_mask(Y)
    for i in range(X.shape[0]):
        pos = ok[i].flatten().nonzero().squeeze(1)
        if len(pos) == 0: continue
        sel = pos[torch.randperm(len(pos), generator=g)[:a.px_per_patch].to(dev)]
        Xs.append(Fe[i].flatten(1)[:, sel].T); Ys.append(tg[i].flatten()[sel])
Xs = torch.cat(Xs); Ys = torch.cat(Ys); print(f"training pixels: {len(Xs):,}  features: {Xs.shape[1]}  ({time.time()-t0:.0f}s)")
nout = 3 if a.direction == "AB" else 1; net = nn.Sequential(nn.Linear(Xs.shape[1], 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, nout)).to(dev)
opt = torch.optim.Adam(net.parameters(), 2e-3); mu, sd = (Ys.float().mean(), Ys.float().std()) if a.direction == "BA" else (0, 1)
for ep in range(a.epochs):
    perm = torch.randperm(len(Xs), device=dev); tot = 0; nb = 0
    for i in range(0, len(Xs), 65536):
        b = perm[i:i + 65536]; out = net(Xs[b])
        loss = Fn.cross_entropy(out, Ys[b]) if a.direction == "AB" else Fn.l1_loss(out.squeeze(1), (Ys[b] - mu) / sd)
        opt.zero_grad(); loss.backward(); opt.step(); tot += float(loss); nb += 1
    print(f"  epoch {ep+1}/{a.epochs} loss {tot/nb:.4f}", flush=True)
net.eval(); nf_ = Xs.shape[1]; del Xs, Ys
if a.direction == "BA":
    os.makedirs(os.path.join(a.rev_dir, "geo3b"), exist_ok=True)
    torch.save(dict(state=net.state_dict(), mu=float(mu), sd=float(sd), nf=nf_), os.path.join(a.rev_dir, "geo3b", f"{a.task}_BA_mlp.pt"))
# ---------------------------------------------- evaluate on TEST exactly like the GAN
ds, idx = loader("test", a.n); rows = []
with torch.no_grad():
    for X, Y, names in batches(ds, idx):
        out = net(features(X).permute(0, 2, 3, 1)); tg, ok = target_and_mask(Y)
        if a.direction == "AB":
            cp = out.argmax(-1); bt = boundary(tg)
            for i, nm in enumerate(names):
                m = ok[i]
                if int(m.sum()) < 500: continue
                mt = m & ~bt[i]; acc = lambda mm: float((cp[i][mm] == tg[i][mm]).float().mean())
                prop = np.mean([abs(float(((cp[i][m] == k).float().mean() - (tg[i][m] == k).float().mean()))) for k in range(3)])
                rows.append(dict(file=nm, pix_acc=acc(m), pix_acc_tol=acc(mt) if int(mt.sum()) > 200 else float("nan"), prop_err=float(prop)))
        else:
            pv = out.squeeze(-1) * sd + mu
            for i, nm in enumerate(names):
                m = ok[i]
                if int(m.sum()) < 500: continue
                rows.append(dict(file=nm, mae=float((pv[i][m] - tg[i][m]).abs().mean())))
od = os.path.join(a.rev_dir, "geo3b"); os.makedirs(od, exist_ok=True); fn = os.path.join(od, f"{a.task}_{a.direction}_test.csv"); keys = list(rows[0].keys())
with open(fn, "w", newline="") as f: w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
print("Saved", fn, len(rows), "| mean:", {k: round(float(np.nanmean([r[k] for r in rows])), 4) for k in keys if k != "file"})
