"""
quality_estimator.py -- reference-free quality estimator trained against GEOLOGICAL error (VAL), evaluated on TEST.
 target (risk): AB directions -> 1 - boundary-tolerant facies pixel accuracy ; BA directions -> MAE (v/v or log10 mD)
 features (all reference-free): gate scores cons_tta5/cons_tta8/cons_mc10 + properties of the predicted map / input (f_*)
 models: (a) gate alone = -cons_tta8 ; (b) ridge on gate scores only ; (c) ridge on all features (+ squares)
 report: Spearman(score, risk), AURC closure vs random/oracle, and mean risk of the kept set at 90% / 80% coverage.
"""
import argparse, os, numpy as np, pandas as pd
p = argparse.ArgumentParser(); p.add_argument("--rev_dir", default=r"D:\First year\Papers\GeoAgent\revision_ACAGS"); p.add_argument("--alpha", type=float, default=10.0)
a = p.parse_args()
def load(t, d, s):
    g = pd.read_csv(os.path.join(a.rev_dir, "geo2", f"{t}_{d}_{s}.csv")).set_index("file")
    b = pd.read_csv(os.path.join(a.rev_dir, "policy", f"{t}_{d}_{s}.csv")).set_index("file")[["ssim_tta8", "cons_tta5", "cons_tta8", "cons_mc10"]]
    return g.join(b, how="inner")
def spearman(x, y): return pd.Series(x).rank().corr(pd.Series(y).rank())
def aurc(score_good, risk):                  # accept highest score first
    o = np.argsort(-score_good); return (np.cumsum(risk[o]) / np.arange(1, len(risk) + 1)).mean()
def ridge_fit(X, y, alpha):
    mu, sd = X.mean(0), X.std(0) + 1e-9; Z = (X - mu) / sd; Z1 = np.c_[np.ones(len(Z)), Z]
    w = np.linalg.solve(Z1.T @ Z1 + alpha * np.diag([0] + [1] * Z.shape[1]), Z1.T @ y); return lambda Xn: np.c_[np.ones(len(Xn)), (Xn - mu) / sd] @ w
for t in ["PorFac", "PermFac"]:
    for d in ["AB", "BA"]:
        try: v, x = load(t, d, "val"), load(t, d, "test")
        except Exception as e: print(t, d, "missing:", e); continue
        risk_col = "risk" ; 
        for df in (v, x):
            df["risk"] = (1 - df["pix_acc_tol"]) if d == "AB" else df["mae"]
        v, x = v.dropna(subset=["risk"]), x.dropna(subset=["risk"])
        gate = ["cons_tta5", "cons_tta8", "cons_mc10"]; feats = [c for c in v.columns if c.startswith("f_")]
        def sq(df, cols): return np.c_[df[cols].values, df[cols].values ** 2]
        m_gate = ridge_fit(v[gate].values, v.risk.values, a.alpha); m_all = ridge_fit(sq(v, gate + feats), v.risk.values, a.alpha)
        bd = [c for c in feats if c in ("f_bdry", "f_inbdry")]
        m_f = ridge_fit(sq(v, feats), v.risk.values, a.alpha); m_b = ridge_fit(sq(v, bd), v.risk.values, a.alpha)
        scores = {"random": np.random.default_rng(0).random(len(x)), "gate alone (cons_tta8)": x.cons_tta8.values,
                  "ridge on gate scores": -m_gate(x[gate].values), "ridge on all features": -m_all(sq(x, gate + feats)),
                  "ridge, map features only": -m_f(sq(x, feats)), "ridge, boundary density only": -m_b(sq(x, bd)),
                  "SSIM (needs ground truth)": x.ssim_tta8.values}
        r = x.risk.values; full = r.mean(); ao = aurc(-r, r); ar = aurc(scores["random"], r)
        print(f"\n{t} {d}: val n={len(v)}, test n={len(x)} | risk = {'1 - boundary-tolerant facies accuracy' if d=='AB' else 'MAE'}; mean test risk {full:.4f}; AURC random {ar:.4f}, oracle {ao:.4f}")
        for n, s in scores.items():
            o = np.argsort(-s); r90 = r[o[:int(.9 * len(o))]].mean(); r80 = r[o[:int(.8 * len(o))]].mean(); ag = aurc(s, r)
            print(f"  {n:27s} rho(score,-risk) {spearman(s, -r):+.3f} | closes {(ar-ag)/(ar-ao):6.1%} of gap | risk kept 90%: {r90:.4f} ({(r90-full)/full:+.1%}), 80%: {r80:.4f} ({(r80-full)/full:+.1%})")
