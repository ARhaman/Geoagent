"""
llm_analysis.py -- does the BLIND LLM score add information beyond the learned quality estimator (and beyond SSIM/gate)?
On the LLM-judged test subsample: Spearman(score, -geological risk) for LLM, estimator, gate; combined score by 5-fold cross-validated ridge on
[estimator score, LLM score]; risk of the kept set at 80% coverage with bootstrap 95% CI; failure-mode breakdown.
"""
import argparse, os, json, numpy as np, pandas as pd
p = argparse.ArgumentParser(); p.add_argument("--rev_dir", default=r"D:\First year\Papers\GeoAgent\revision_ACAGS"); p.add_argument("--alpha", type=float, default=10.0)
a = p.parse_args(); rng = np.random.default_rng(0)
def load(t, d, s):
    g = pd.read_csv(os.path.join(a.rev_dir, "geo2", f"{t}_{d}_{s}.csv")).set_index("file")
    b = pd.read_csv(os.path.join(a.rev_dir, "policy", f"{t}_{d}_{s}.csv")).set_index("file")[["ssim_tta8", "cons_tta5", "cons_tta8", "cons_mc10"]]
    df = g.join(b, how="inner"); df["risk"] = (1 - df["pix_acc_tol"]) if d == "AB" else df["mae"]; return df.dropna(subset=["risk"])
def rank_corr(x, y): return pd.Series(x).rank().corr(pd.Series(y).rank())
def ridge_fit(X, y, al):
    mu, sd = X.mean(0), X.std(0) + 1e-9; Z1 = np.c_[np.ones(len(X)), (X - mu) / sd]
    w = np.linalg.solve(Z1.T @ Z1 + al * np.diag([0] + [1] * X.shape[1]), Z1.T @ y); return lambda Xn: np.c_[np.ones(len(Xn)), (Xn - mu) / sd] @ w
def kept_risk(score, risk, c=0.8): o = np.argsort(-score); return risk[o[:int(c * len(o))]].mean()
for t in ["PorFac", "PermFac"]:
    for d in ["AB", "BA"]:
        f = os.path.join(a.rev_dir, "llm", f"{t}_{d}.jsonl")
        if not os.path.exists(f): continue
        recs = [json.loads(l) for l in open(f, encoding="utf-8")]; ok = [r for r in recs if "quality_score" in r]
        if len(ok) < 30: print(f"\n{t} {d}: only {len(ok)} successful LLM judgements so far - skipped"); continue
        v, x = load(t, d, "val"), load(t, d, "test"); gate = ["cons_tta5", "cons_tta8", "cons_mc10"]; feats = [c for c in v.columns if c.startswith("f_")]
        sq = lambda df, cols: np.c_[df[cols].values, df[cols].values ** 2]
        m = ridge_fit(sq(v, gate + feats), v.risk.values, a.alpha); x["est"] = -m(sq(x, gate + feats))
        L = pd.DataFrame(ok).drop_duplicates("file", keep="last").set_index("file")[["quality_score", "failure_mode"]]; z = x.join(L, how="inner").dropna(subset=["quality_score"])
        r = z.risk.values; n = len(z)
        print(f"\n{t} {d}: LLM-judged patches {n} (calls failed: {len(recs)-len(ok)}); mean risk {r.mean():.4f}")
        print(f"  Spearman with -risk:  LLM {rank_corr(z.quality_score.values, -r):+.3f} | estimator {rank_corr(z.est.values, -r):+.3f} | gate {rank_corr(z.cons_tta8.values, -r):+.3f} | SSIM {rank_corr(z.ssim_tta8.values, -r):+.3f}"
              f" | corr(LLM, estimator) {rank_corr(z.quality_score.values, z.est.values):+.3f}")
        folds = np.arange(n) % 5; rng.shuffle(folds); comb = np.zeros(n)
        for k in range(5):
            tr, te = folds != k, folds == k; mm = ridge_fit(np.c_[z.est.values, z.quality_score.values][tr], r[tr], 1.0); comb[te] = -mm(np.c_[z.est.values, z.quality_score.values][te])
        res = {}
        for nme, s in {"estimator": z.est.values, "LLM": z.quality_score.values, "estimator+LLM (CV)": comb}.items():
            bs = [kept_risk(s[ii], r[ii]) for ii in (rng.integers(0, n, n) for _ in range(500))]; res[nme] = (kept_risk(s, r), np.percentile(bs, [2.5, 97.5]))
            print(f"  risk of kept 80%: {nme:20s} {res[nme][0]:.4f} [{res[nme][1][0]:.4f}, {res[nme][1][1]:.4f}]   (all patches: {r.mean():.4f})")
        print("  failure modes (mean geological risk | count): " + "; ".join(f"{k}: {g.risk.mean():.3f} | {len(g)}" for k, g in z.groupby("failure_mode")))
