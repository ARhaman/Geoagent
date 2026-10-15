"""
ood_signal.py -- second reliability signal that looks at the INPUT: Mahalanobis distance of the generator's bottleneck features (global-avg-pooled e5, 512-d) to the TRAIN feature distribution
(fit on 3000 train patches, shrinkage covariance, single forward pass, no ground truth).  Calibrated on VAL (percentile).
Reports, per direction: (1) in-distribution TEST: Spearman(signal, risk) and delivered risk at 90/80/70% coverage for consistency alone, OOD alone, and the equal-weight combination
 (rank average, no fitting);  (2) shift sets (geo6): share flagged by each signal (percentile < 10 for consistency, > 90 for OOD), risk of accepted vs flagged under the combined gate.
Run (geoldm_gpu):  python ood_signal.py --task PorFac --direction AB      (all: run_ood.bat -> ood_results.txt)
"""
import os, sys, argparse, json
import numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader
p = argparse.ArgumentParser()
p.add_argument("--code_dir", default=r"D:\First year\Papers\GeoAgent"); p.add_argument("--data_base", default=r"E:\2D Dutch's Dataset\patches_512")
p.add_argument("--ckpt_base", default=r"D:\First year\Papers\GeoAgent\checkpoints"); p.add_argument("--rev_dir", default=r"D:\First year\Papers\GeoAgent\revision_ACAGS")
p.add_argument("--task", required=True, choices=["PorFac", "PermFac"]); p.add_argument("--direction", required=True, choices=["AB", "BA"]); p.add_argument("--epoch", default="0100"); p.add_argument("--ntrain", type=int, default=3000)
a = p.parse_args(); sys.path.insert(0, a.code_dir); sys.path.insert(0, a.rev_dir)
from nb_model import Pix2PixModel
from geoagent_dataset import PairedGeoDataset
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
M = Pix2PixModel(device=dev); M.load(os.path.join(a.ckpt_base, f"{a.task}_{a.direction}", f"pix2pix_{a.direction}_ep{a.epoch}.pth")); M.G.eval(); G = M.G
def feat(x):
    e1 = G.e1(x); e2 = G.e2(e1); e3 = G.e3(e2); e4 = G.e4(e3); e5 = G.e5(e4); return e5.mean((2, 3))
def feats(folder, n=0, step=1):
    ds = PairedGeoDataset(folder, direction=a.direction, image_size=256, augment=False)
    if step > 1: ds.files = ds.files[::step]
    if n: ds.files = ds.files[:n]
    out = []; names = []
    with torch.no_grad():
        for b in DataLoader(ds, batch_size=32, shuffle=False, num_workers=0): out.append(feat(b["input"].to(dev)).double().cpu()); names += [os.path.basename(q) for q in b["path"]]
    return torch.cat(out), names
root = os.path.join(a.data_base, a.task); tr, _ = feats(os.path.join(root, "train"), a.ntrain, step=max(1, 27000 // a.ntrain)); mu = tr.mean(0); X = tr - mu
S = X.T @ X / (len(X) - 1); S = 0.9 * S + 0.1 * torch.diag(torch.diag(S)) + 1e-6 * torch.eye(S.shape[0], dtype=S.dtype); P = torch.linalg.inv(S)
def dist(F): d = F - mu; return torch.sqrt(((d @ P) * d).sum(1).clamp_min(0)).numpy()
fv, nv = feats(os.path.join(root, "val")); ft, nt = feats(os.path.join(root, "test")); dv = dist(fv); dt = dist(ft); dvs = np.sort(dv)
pct = lambda d: 100 * np.searchsorted(dvs, d) / len(dvs)
rev = a.rev_dir; Vc = pd.read_csv(os.path.join(rev, "policy", f"{a.task}_{a.direction}_val.csv")); vcs = np.sort(Vc.cons_tta8.values); cpct = lambda c: 100 * np.searchsorted(vcs, c) / len(vcs)
T = pd.read_csv(os.path.join(rev, "geo5", f"{a.task}_{a.direction}_test.csv")).merge(pd.read_csv(os.path.join(rev, "policy", f"{a.task}_{a.direction}_test.csv"))[["file", "cons_tta8"]], on="file")
T = T.merge(pd.DataFrame(dict(file=nt, ood=dt)), on="file").dropna(subset=["risk_tta8"]); T["c_pct"] = cpct(T.cons_tta8.values); T["o_pct"] = pct(T.ood.values); T["comb"] = 0.5 * T.c_pct + 0.5 * (100 - T.o_pct)
print(f"\n{a.task} {a.direction}: in-distribution TEST (n={len(T)})  Spearman with risk: consistency {T.cons_tta8.corr(T.risk_tta8, method='spearman'):+.3f} | OOD distance {T.ood.corr(T.risk_tta8, method='spearman'):+.3f} | combined {T.comb.corr(T.risk_tta8, method='spearman'):+.3f}")
base = T.risk_tta8.mean(); line = f"  delivered risk (all patches {base:.4f}) at coverage 90/80/70%:  "
for name, s in (("consistency", T.c_pct.values), ("OOD", 100 - T.o_pct.values), ("combined", T.comb.values)):
    res = []
    for cov in (0.9, 0.8, 0.7): thr = np.quantile(s, 1 - cov); res.append(T.risk_tta8.values[s >= thr].mean())
    line += f"{name} " + "/".join(f"{r:.4f}" for r in res) + " | "
print(line)
rows = []
for sett in ("PorFac", "PorFacN", "PermFacN"):
    if not sett.startswith(a.task[:4]): continue
    for o in ("K", "I", "J"):
        d_ = os.path.join(rev, "shift", sett, o); f6 = os.path.join(rev, "geo6", f"{sett}_{a.direction}_{o}.csv")
        if not (os.path.isdir(d_) and os.path.exists(f6)): continue
        fs, ns = feats(d_); S6 = pd.read_csv(f6).merge(pd.DataFrame(dict(file=ns, ood=dist(fs))), on="file").dropna(subset=["risk"]); S6["c_pct"] = cpct(S6.cons.values); S6["o_pct"] = pct(S6.ood.values)
        fl_c = S6.c_pct < 10; fl_o = S6.o_pct > 90; fl_b = fl_c | fl_o; r = S6.risk.values
        print(f"  shift set {sett}-{o}: n={len(S6)} risk {r.mean():.4f} | flagged: consistency {fl_c.mean():.0%} | OOD {fl_o.mean():.0%} | either {fl_b.mean():.0%} | risk kept {r[~fl_b].mean() if (~fl_b).any() else float('nan'):.4f} vs flagged {r[fl_b].mean() if fl_b.any() else float('nan'):.4f} | Spearman(OOD,risk) {S6.ood.corr(S6.risk, method='spearman'):+.2f}")
