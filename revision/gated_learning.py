"""
gated_learning.py -- validation-gated learning loop (the 'learn' step of the agent), with rollback.
 1. SPLIT   the VAL pool (labelled, held out from training) into an adaptation pool A and a CHECK split B (seeded, 50/50).
 2. FLAG    the reference-free gate (cons_tta8, no ground truth) flags the least reliable patches in A (k = budget x |VAL|); they get labels.
 3. LEARN   candidate = backbone fine-tuned on flagged patches + equal TRAIN replay (5 epochs, lr 5e-5), as in finetune_hard.py.
 4. CHECK   candidate vs base on B with GEOLOGIC RISK (AB: 1 - boundary-tolerant facies accuracy; BA: MAE in v/v or log10 mD), TTA x8.
 5. DECIDE  ACCEPT iff (a) the hard half of B (lowest 10% by cons_tta8) improves (95% bootstrap upper bound of delta-risk < 0) AND
            (b) overall B is non-inferior (upper bound of delta-risk < 2% of base risk). Otherwise ROLLBACK (deploy base).
 6. AUDIT   candidate vs base on TEST (never used in 1-5): hard-10% and overall delta-risk with CIs; deployed delta = candidate if accepted, else 0.
Run (geoldm_gpu):  python gated_learning.py --task PermFac --direction AB --seed 0        (smoke: --n_test 64 --n_check 64 --epochs 1)
Needs policy\{task}_{direction}_{val,test}.csv. Output: gated\{task}_{direction}_s{seed}[_b].json
"""
import os, sys, argparse, json, time, random
import numpy as np, pandas as pd, torch, torch.nn.functional as Fn
from torch.utils.data import DataLoader, ConcatDataset, Subset
p = argparse.ArgumentParser()
p.add_argument("--code_dir", default=r"D:\First year\Papers\GeoAgent"); p.add_argument("--data_base", default=r"E:\2D Dutch's Dataset\patches_512")
p.add_argument("--ckpt_base", default=r"D:\First year\Papers\GeoAgent\checkpoints"); p.add_argument("--rev_dir", default=r"D:\First year\Papers\GeoAgent\revision_ACAGS")
p.add_argument("--task", required=True, choices=["PorFac", "PermFac"]); p.add_argument("--direction", required=True, choices=["AB", "BA"])
p.add_argument("--budget", type=float, default=0.05); p.add_argument("--epochs", type=int, default=5); p.add_argument("--lr", type=float, default=5e-5)
p.add_argument("--seed", type=int, default=0); p.add_argument("--bs", type=int, default=16); p.add_argument("--epoch", default="0100")
p.add_argument("--n_check", type=int, default=0); p.add_argument("--n_test", type=int, default=0); p.add_argument("--margin", type=float, default=0.02)
a = p.parse_args(); sys.path.insert(0, a.code_dir); sys.path.insert(0, a.rev_dir)
from nb_model import Pix2PixModel
from geoagent_dataset import PairedGeoDataset
import legend_decode as L
random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu"); root = os.path.join(a.data_base, a.task)
FAC = torch.tensor(L.FACIES_RGB, device=dev)
if a.direction == "BA":
    LUT = torch.tensor(L.PORO_RGB if a.task == "PorFac" else L.PERM_RGB, device=dev); VAL = torch.tensor(L.PORO_VAL if a.task == "PorFac" else L.PERM_LOG10, dtype=torch.float32, device=dev); NB = VAL.shape[0]; edge = int(0.02 * NB)
rgb = lambda x: ((x.clamp(-1, 1) + 1) / 2 * 255).permute(0, 2, 3, 1)
def nearest(px, lut, chunk=65536):
    shp = px.shape[:-1]; px = px.reshape(-1, 3); idx = torch.empty(px.shape[0], dtype=torch.long, device=px.device); dist = torch.empty(px.shape[0], device=px.device)
    for i in range(0, px.shape[0], chunk): m, ix = torch.cdist(px[i:i + chunk], lut).min(1); idx[i:i + chunk] = ix; dist[i:i + chunk] = m
    return idx.view(shp), dist.view(shp)
def boundary(c, k=5): c = c.float().unsqueeze(1); return (Fn.max_pool2d(c, k, 1, k // 2) != -Fn.max_pool2d(-c, k, 1, k // 2)).squeeze(1)
def d4(k_, f):
    fw = lambda x: torch.rot90(torch.flip(x, [-1]) if f else x, k_, [-2, -1]); iv = lambda y: (torch.flip(torch.rot90(y, -k_, [-2, -1]), [-1]) if f else torch.rot90(y, -k_, [-2, -1])); return fw, iv
T8 = [d4(k_, f) for f in (False, True) for k_ in range(4)]
def risk_eval(M, ds, idx):                         # per-patch geologic risk (NaN if too few valid pixels) -> dict file->risk
    M.G.eval(); out = {}
    with torch.no_grad():
        for b in DataLoader(Subset(ds, idx), batch_size=16, shuffle=False, num_workers=0):
            x = b["input"].to(dev); y = b["target"].to(dev); pr = torch.stack([iv(M.G(fw(x))) for fw, iv in T8]).mean(0); P, T = rgb(pr), rgb(y); s = T.sum(-1); inside = (s > 30) & (s < 740)
            if a.direction == "AB":
                cp, _ = nearest(P, FAC); ct, dt = nearest(T, FAC); m = inside & (dt < 60); mt = m & ~boundary(ct)
                for i, pa in enumerate(b["path"]):
                    out[os.path.basename(pa)] = float(1 - (cp[i][mt[i]] == ct[i][mt[i]]).float().mean()) if (int(m[i].sum()) >= 500 and int(mt[i].sum()) > 200) else float("nan")
            else:
                ip, _ = nearest(P, LUT); it, dt = nearest(T, LUT); ok = inside & (dt < 45) & (it >= edge) & (it < NB - edge)
                for i, pa in enumerate(b["path"]):
                    out[os.path.basename(pa)] = float((VAL[ip[i]][ok[i]] - VAL[it[i]][ok[i]]).abs().mean()) if int(ok[i].sum()) >= 500 else float("nan")
    return out
def boot(d, n=2000, seed=0):                        # mean and 95% bootstrap CI of paired deltas
    d = np.asarray(d, float); d = d[~np.isnan(d)]; r = np.random.RandomState(seed); m = np.array([d[r.randint(0, len(d), len(d))].mean() for _ in range(n)])
    return dict(mean=float(d.mean()), lo=float(np.percentile(m, 2.5)), hi=float(np.percentile(m, 97.5)), n=int(len(d)))
def deltas(rb, rf, files): return [rf[f] - rb[f] for f in files if f in rb and f in rf]
def build(): M = Pix2PixModel(device=dev, lr=a.lr); M.load(os.path.join(a.ckpt_base, f"{a.task}_{a.direction}", f"pix2pix_{a.direction}_ep{a.epoch}.pth")); return M
# ---- data
val_aug = PairedGeoDataset(os.path.join(root, "val"), direction=a.direction, image_size=256, augment=True); val_ev = PairedGeoDataset(os.path.join(root, "val"), direction=a.direction, image_size=256, augment=False)
tr_ds = PairedGeoDataset(os.path.join(root, "train"), direction=a.direction, image_size=256, augment=True); te = PairedGeoDataset(os.path.join(root, "test"), direction=a.direction, image_size=256, augment=False)
vcsv = pd.read_csv(os.path.join(a.rev_dir, "policy", f"{a.task}_{a.direction}_val.csv")).set_index("file"); tcsv = pd.read_csv(os.path.join(a.rev_dir, "policy", f"{a.task}_{a.direction}_test.csv")).set_index("file")
vfiles = [os.path.basename(f) for f in val_ev.files]; v2i = {f: i for i, f in enumerate(vfiles)}; tfiles = [os.path.basename(f) for f in te.files]; t2i = {f: i for i, f in enumerate(tfiles)}
rs = np.random.RandomState(1000 + a.seed); perm = rs.permutation(len(vfiles)); half = len(vfiles) // 2; A = [vfiles[i] for i in perm[:half]]; B = [vfiles[i] for i in perm[half:]]
if a.n_check: B = B[:a.n_check]
k = int(round(a.budget * len(vfiles))); flagged = sorted(A, key=lambda f: vcsv.loc[f, "cons_tta8"])[:k]
cB = sorted(B, key=lambda f: vcsv.loc[f, "cons_tta8"]); hardB = cB[:max(1, len(cB) // 10)]
ttest = tfiles[:a.n_test] if a.n_test else tfiles; ctest = sorted(ttest, key=lambda f: tcsv.loc[f, "cons_tta8"]); hardT = ctest[:max(1, len(ctest) // 10)]
print(f"{a.task} {a.direction} seed {a.seed}: |A|={len(A)} |B|={len(B)} flagged={len(flagged)} (hard check={len(hardB)}) test={len(ttest)} (hard={len(hardT)})", flush=True)
# ---- base risks (check split + test)
t0 = time.time(); Mb = build(); rbB = risk_eval(Mb, val_ev, [v2i[f] for f in B]); rbT = risk_eval(Mb, te, [t2i[f] for f in ttest]); print(f"base evaluated ({time.time()-t0:.0f}s)", flush=True)
# ---- candidate
M = build()
for g in M.oG.param_groups: g["lr"] = a.lr
for g in M.oD.param_groups: g["lr"] = a.lr
ds = ConcatDataset([Subset(val_aug, [v2i[f] for f in flagged]), Subset(tr_ds, random.sample(range(len(tr_ds)), len(flagged)))])
dl = DataLoader(ds, batch_size=a.bs, shuffle=True, num_workers=0, drop_last=True)
for ep in range(a.epochs):
    M.G.train(); M.D.train(); Ls = [M.train_step(b["input"], b["target"])["loss_G_l1"] for b in dl]; print(f"  epoch {ep+1}/{a.epochs} L1 {np.mean(Ls):.3f} ({time.time()-t0:.0f}s)", flush=True)
rfB = risk_eval(M, val_ev, [v2i[f] for f in B])
chk_h = boot(deltas(rbB, rfB, hardB)); chk_all = boot(deltas(rbB, rfB, B)); base_all = float(np.nanmean(list(rbB.values())))
acc_a = chk_h["hi"] < 0; acc_b = chk_all["hi"] < a.margin * base_all; accept = bool(acc_a and acc_b)
print(f"CHECK hard dRisk {chk_h['mean']:+.4f} [{chk_h['lo']:+.4f},{chk_h['hi']:+.4f}] ({'ok' if acc_a else 'FAIL'}) | overall {chk_all['mean']:+.4f} [{chk_all['lo']:+.4f},{chk_all['hi']:+.4f}] vs margin {a.margin*base_all:.4f} ({'ok' if acc_b else 'FAIL'}) -> {'ACCEPT' if accept else 'ROLLBACK'}", flush=True)
rfT = risk_eval(M, te, [t2i[f] for f in ttest]); th = boot(deltas(rbT, rfT, hardT)); ta = boot(deltas(rbT, rfT, ttest))
print(f"TEST (audit) candidate: hard dRisk {th['mean']:+.4f} [{th['lo']:+.4f},{th['hi']:+.4f}] | overall {ta['mean']:+.4f} [{ta['lo']:+.4f},{ta['hi']:+.4f}] | deployed = {'candidate' if accept else 'base (rolled back)'}")
od = os.path.join(a.rev_dir, "gated"); os.makedirs(od, exist_ok=True); sfx = "" if abs(a.budget - 0.05) < 1e-9 else f"_b{a.budget}"
json.dump(dict(task=a.task, direction=a.direction, seed=a.seed, budget=a.budget, n_flagged=len(flagged), base_check_risk=base_all, check_hard=chk_h, check_all=chk_all, accept=accept,
               test_hard=th, test_all=ta, base_test_risk=float(np.nanmean(list(rbT.values())))), open(os.path.join(od, f"{a.task}_{a.direction}_s{a.seed}{sfx}.json"), "w"), indent=1)
