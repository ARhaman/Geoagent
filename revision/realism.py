"""
realism.py -- does the GAN reproduce WITHIN-FACIES HETEROGENEITY (Fac->Por, Fac->Perm) better than a context-aware MLP?
MAE rewards smooth conditional means; a realistic model should instead match the truth's within-facies spread, value distribution and spatial structure.
Compares TRUTH vs GAN (TTA x8) vs MLP (saved by baselines2.py) on the same valid pixels:
  (1) within-facies std per facies class   (2) Wasserstein-1 distance of the within-facies value distribution to truth
  (3) within-facies semivariogram at lags 1,2,4,8,16 px (pairs with both pixels valid and in the same input facies); report gamma ratio to truth + normalised RMSE
Run (geoldm_gpu):  python realism.py --task PorFac      (needs geo3b\\{task}_BA_mlp.pt: rerun baselines2.py --direction BA first)
"""
import os, sys, argparse, json, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as Fn
from torch.utils.data import DataLoader
p = argparse.ArgumentParser()
p.add_argument("--code_dir", default=r"D:\First year\Papers\GeoAgent"); p.add_argument("--data_base", default=r"E:\2D Dutch's Dataset\patches_512")
p.add_argument("--ckpt_base", default=r"D:\First year\Papers\GeoAgent\checkpoints"); p.add_argument("--rev_dir", default=r"D:\First year\Papers\GeoAgent\revision_ACAGS")
p.add_argument("--task", required=True, choices=["PorFac", "PermFac"]); p.add_argument("--n", type=int, default=2000); p.add_argument("--bs", type=int, default=16); p.add_argument("--epoch", default="0100")
a = p.parse_args(); sys.path.insert(0, a.code_dir); sys.path.insert(0, a.rev_dir)
from nb_model import Pix2PixModel
from geoagent_dataset import PairedGeoDataset
import legend_decode as L
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
M = Pix2PixModel(device=dev); M.load(os.path.join(a.ckpt_base, f"{a.task}_BA", f"pix2pix_BA_ep{a.epoch}.pth")); M.G.eval()
ds = PairedGeoDataset(os.path.join(a.data_base, a.task, "test"), direction="BA", image_size=256, augment=False)
step = max(1, len(ds.files) // a.n); ds.files = ds.files[::step][:a.n]
FAC = torch.tensor(L.FACIES_RGB, device=dev)
LUT = torch.tensor(L.PORO_RGB if a.task == "PorFac" else L.PERM_RGB, device=dev); VAL = torch.tensor(L.PORO_VAL if a.task == "PorFac" else L.PERM_LOG10, dtype=torch.float32, device=dev)
NB = VAL.shape[0]; edge = int(0.02 * NB)
rgb = lambda x: ((x.clamp(-1, 1) + 1) / 2 * 255).permute(0, 2, 3, 1)
def nearest(px, lut, chunk=65536):
    shp = px.shape[:-1]; px = px.reshape(-1, 3); idx = torch.empty(px.shape[0], dtype=torch.long, device=px.device); dist = torch.empty(px.shape[0], device=px.device)
    for i in range(0, px.shape[0], chunk): m, ix = torch.cdist(px[i:i + chunk], lut).min(1); idx[i:i + chunk] = ix; dist[i:i + chunk] = m
    return idx.view(shp), dist.view(shp)
def pool(x, k): return Fn.avg_pool2d(x, k, 1, k // 2, count_include_pad=False)
def features(X):                              # identical to baselines2.py (BA)
    R = rgb(X); S = R.sum(-1); fi, df = nearest(R, FAC); mask = ((S > 30) & (S < 740) & (df < 60)).float().unsqueeze(1)
    oh = Fn.one_hot(fi.clamp(0, 2), 3).permute(0, 3, 1, 2).float() * mask; ch = [oh, mask]
    for k in (5, 15, 31): ch.append(pool(oh, k) / pool(mask, k).clamp_min(1e-6))
    return torch.cat(ch, 1), fi, mask.squeeze(1) > 0
ck = torch.load(os.path.join(a.rev_dir, "geo3b", f"{a.task}_BA_mlp.pt"), map_location=dev)
net = nn.Sequential(nn.Linear(ck["nf"], 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 1)).to(dev); net.load_state_dict(ck["state"]); net.eval(); mu, sd = ck["mu"], ck["sd"]
def d4(k, f):
    fw = lambda x: torch.rot90(torch.flip(x, [-1]) if f else x, k, [-2, -1]); iv = lambda y: (torch.flip(torch.rot90(y, -k, [-2, -1]), [-1]) if f else torch.rot90(y, -k, [-2, -1])); return fw, iv
T8 = [d4(k, f) for f in (False, True) for k in range(4)]
LAGS = [1, 2, 4, 8, 16]; SRC = ["truth", "gan", "mlp"]; NBIN = 300
_v = VAL[edge:NB - edge]; lo, hi = float(_v.min()), float(_v.max()); bins = torch.linspace(lo, hi, NBIN + 1, device=dev)
S1 = {s: torch.zeros(3, device=dev, dtype=torch.float64) for s in SRC}; S2 = {s: torch.zeros(3, device=dev, dtype=torch.float64) for s in SRC}; N = torch.zeros(3, device=dev, dtype=torch.float64)
H = {s: torch.zeros(3, NBIN, device=dev, dtype=torch.float64) for s in SRC}
G = {s: torch.zeros(3, len(LAGS), device=dev, dtype=torch.float64) for s in SRC}; GN = torch.zeros(3, len(LAGS), device=dev, dtype=torch.float64)
t0 = time.time()
with torch.no_grad():
    for bi, b in enumerate(DataLoader(ds, batch_size=a.bs, shuffle=False, num_workers=0)):
        x = b["input"].to(dev); y = b["target"].to(dev)
        pr = torch.stack([iv(M.G(fw(x))) for fw, iv in T8]).mean(0); Fe, fi, fmask = features(x)
        R = rgb(y); T = R.sum(-1); it, dt = nearest(R, LUT); ok = (T > 30) & (T < 740) & (dt < 45) & (it >= edge) & (it < NB - edge) & fmask
        ip, _ = nearest(rgb(pr), LUT); vm = (net(Fe.permute(0, 2, 3, 1)).squeeze(-1) * sd + mu)
        vals = {"truth": VAL[it], "gan": VAL[ip], "mlp": vm.clamp(lo, hi)}
        for c in range(3):
            m = ok & (fi == c); N[c] += m.sum()
            for s in SRC:
                v = vals[s][m].double(); S1[s][c] += v.sum(); S2[s][c] += (v * v).sum()
                H[s][c] += torch.histc(vals[s][m].float(), NBIN, lo, hi).double()
            for li, l in enumerate(LAGS):                               # horizontal + vertical pairs, same facies, both valid
                for (sa, sb) in ((np.s_[:, :, :-l], np.s_[:, :, l:]), (np.s_[:, :-l, :], np.s_[:, l:, :])):
                    pm = ok[sa] & ok[sb] & (fi[sa] == c) & (fi[sb] == c); GN[c, li] += pm.sum()
                    for s in SRC: G[s][c, li] += ((vals[s][sa][pm] - vals[s][sb][pm]).double() ** 2).sum() / 2
        if bi % 20 == 0: print(f"batch {bi+1}/{len(ds)//a.bs+1} ({time.time()-t0:.0f}s)", flush=True)
names = ["sand(yellow)", "shaly sand(orange)", "shale(gray)"]; unit = "v/v" if a.task == "PorFac" else "log10 mD"; out = {"task": a.task, "unit": unit, "n_patches": len(ds.files)}
sd_ = {s: ((S2[s] / N - (S1[s] / N) ** 2).clamp_min(0)).sqrt().cpu().numpy() for s in SRC}
cdf = {s: (H[s].cumsum(1) / H[s].sum(1, keepdim=True).clamp_min(1)) for s in SRC}; w = (hi - lo) / NBIN
w1 = {s: ((cdf[s] - cdf["truth"]).abs().sum(1) * w).cpu().numpy() for s in SRC}
gam = {s: (G[s] / GN.clamp_min(1)).cpu().numpy() for s in SRC}
print(f"\n{a.task} Fac->{'Por' if a.task=='PorFac' else 'Perm'} realism on {len(ds.files)} test patches ({unit})")
for c in range(3):
    print(f"\n{names[c]}  (n pixels = {int(N[c]):,})")
    print(f"  within-facies std      truth {sd_['truth'][c]:.4f} | GAN {sd_['gan'][c]:.4f} ({sd_['gan'][c]/sd_['truth'][c]:.2f}x) | MLP {sd_['mlp'][c]:.4f} ({sd_['mlp'][c]/sd_['truth'][c]:.2f}x)")
    print(f"  Wasserstein-1 to truth GAN {w1['gan'][c]:.4f} | MLP {w1['mlp'][c]:.4f}")
    print("  semivariogram ratio to truth (1.0 = perfect), lags " + str(LAGS))
    for s in ("gan", "mlp"): print(f"    {s.upper()}: " + "  ".join(f"{gam[s][c,i]/gam['truth'][c,i]:.2f}" for i in range(len(LAGS))))
    out[names[c]] = dict(n=int(N[c]), std={s: float(sd_[s][c]) for s in SRC}, w1={s: float(w1[s][c]) for s in ("gan", "mlp")}, gamma={s: gam[s][c].tolist() for s in SRC})
od = os.path.join(a.rev_dir, "geo4"); os.makedirs(od, exist_ok=True); fn = os.path.join(od, f"{a.task}_realism.json"); json.dump(out, open(fn, "w"), indent=1); print("\nSaved", fn)
