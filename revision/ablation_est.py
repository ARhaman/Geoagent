"""
ablation_est.py -- agent-vs-static ablation with the LEARNED reference-free estimator as the gate (ridge on consistency scores + prediction/input map features, fitted on VAL, applied to TEST),
plus SSIM-with-ground-truth for comparison and a val-calibrated risk-controlled operating point.  Writes ablation_est.json and prints a table.
Delivered risk = mean geologic risk (TTAx8 predictions) of patches the gate delivers; abstained patches go to review (not scored).
"""
import os, json, numpy as np, pandas as pd
rev = os.path.dirname(os.path.abspath(__file__)); rng = np.random.default_rng(0)
def load(t, d, s):
    g = pd.read_csv(os.path.join(rev, "geo2", f"{t}_{d}_{s}.csv")).set_index("file"); b = pd.read_csv(os.path.join(rev, "policy", f"{t}_{d}_{s}.csv")).set_index("file")
    a5 = pd.read_csv(os.path.join(rev, "geo5", f"{t}_{d}_{s}.csv")).set_index("file")[["risk_single", "risk_tta8"]]
    df = g.join(b[["ssim_tta8", "cons_tta5", "cons_tta8", "cons_mc10"]], how="inner").join(a5, how="inner"); df["risk"] = (1 - df["pix_acc_tol"]) if d == "AB" else df["mae"]; return df.dropna(subset=["risk", "risk_single"])
def ridge_fit(X, y, alpha=10.0):
    mu, sd = X.mean(0), X.std(0) + 1e-9; Z1 = np.c_[np.ones(len(X)), (X - mu) / sd]; w = np.linalg.solve(Z1.T @ Z1 + alpha * np.diag([0] + [1] * X.shape[1]), Z1.T @ y); return lambda Xn: np.c_[np.ones(len(Xn)), (Xn - mu) / sd] @ w
sq = lambda df, cols: np.c_[df[cols].values, df[cols].values ** 2]
def bs(x, n=1000): b = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)]; return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))
out = {}
for t in ("PorFac", "PermFac"):
    for d in ("AB", "BA"):
        v, x = load(t, d, "val"), load(t, d, "test"); gate = ["cons_tta5", "cons_tta8", "cons_mc10"]; feats = [c for c in v.columns if c.startswith("f_")]
        m = ridge_fit(sq(v, gate + feats), v.risk.values); sv, sx = -m(sq(v, gate + feats)), -m(sq(x, gate + feats)); r = x.risk.values
        res = dict(n=int(len(x)), static=float(x.risk_single.mean()), tta8=float(r.mean()), static_ci=bs(x.risk_single.values), tta8_ci=bs(r))
        for name, s_v, s_x in (("est", sv, sx), ("ssim_gt", v.ssim_tta8.values, x.ssim_tta8.values), ("cons", v.cons_tta8.values, x.cons_tta8.values)):
            for cov in (0.9, 0.8, 0.7):
                thr = np.quantile(s_v, 1 - cov); sel = s_x >= thr          # threshold calibrated on VAL
                res[f"{name}_{int(cov*100)}"] = dict(cov=float(sel.mean()), risk=float(r[sel].mean()), ci=bs(r[sel]), abst_risk=float(r[~sel].mean()))
        target = 0.85 * v.risk.values.mean(); chosen = None
        for cov in np.arange(1.0, 0.04, -0.01):
            thr = np.quantile(sv, 1 - cov)
            if v.risk.values[sv >= thr].mean() <= target: chosen = thr; break
        if chosen is None: res["ctrl"] = dict(target=float(target), reached=False)
        else: sel = sx >= chosen; res["ctrl"] = dict(target=float(target), reached=True, cov=float(sel.mean()), risk=float(r[sel].mean()), met=bool(r[sel].mean() <= target))
        out[f"{t}_{d}"] = res
        e = {k: res[f"est_{k}"] for k in (90, 80, 70)}
        print(f"{t} {d}: static {res['static']:.4f} | TTAx8 {res['tta8']:.4f} | est-gate delivered risk @90/80/70: " + " / ".join(f"{e[k]['risk']:.4f} (cov {e[k]['cov']:.0%}, abst {e[k]['abst_risk']:.4f})" for k in (90, 80, 70)) +
              f" | SSIM(GT) @80: {res['ssim_gt_80']['risk']:.4f} | consistency @80: {res['cons_80']['risk']:.4f} | ctrl: {res['ctrl']}")
json.dump(out, open(os.path.join(rev, "ablation_est.json"), "w"), indent=1)
