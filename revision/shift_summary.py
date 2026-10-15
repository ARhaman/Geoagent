import os, numpy as np, pandas as pd
rev = r"D:\First year\Papers\GeoAgent\revision_ACAGS"; rng = np.random.default_rng(0)
def bm(x, n=1000): x = np.asarray(x, float); x = x[~np.isnan(x)]; b = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)]; return x.mean(), np.percentile(b, 2.5), np.percentile(b, 97.5)
print("[risk: AB = 1 - boundary-tolerant facies accuracy; BA = MAE (v/v | log10 mD); gate: percentile of cons vs VAL calibration set; accept>=25, review 10-25, abstain<10]")
for task, sett in (("PorFac", "PorFacN"), ("PermFac", "PermFacN")):
    for d in ("AB", "BA"):
        v = np.sort(pd.read_csv(os.path.join(rev, "policy", f"{task}_{d}_val.csv"))["cons_tta8"].values); ref = pd.read_csv(os.path.join(rev, "geo5", f"{task}_{d}_test.csv")).risk_tta8.mean()
        print(f"\n{task} {d}  (reference: in-distribution test risk {ref:.4f})")
        for o in ("K", "I", "J"):
            fn = os.path.join(rev, "geo6", f"{sett}_{d}_{o}.csv")
            if not os.path.exists(fn): continue
            T = pd.read_csv(fn).dropna(subset=["risk"]); pct = 100 * np.searchsorted(v, T.cons.values) / len(v); m, lo, hi = bm(T.risk.values)
            acc = pct >= 25; ab = pct < 10; rho = pd.Series(T.cons.values).corr(pd.Series(T.risk.values), method="spearman")
            print(f"  {o}{' (depth, seen)' if o=='K' else ' (cross-section, unseen)'}: n={len(T)} | risk {m:.4f} [{lo:.4f},{hi:.4f}] | median cons pct {np.median(pct):.0f} | accept {acc.mean():.0%} review {((pct>=10)&(pct<25)).mean():.0%} abstain {ab.mean():.0%} | "
                  f"risk accepted {T.risk.values[acc].mean() if acc.any() else float('nan'):.4f} vs abstained {T.risk.values[ab].mean() if ab.any() else float('nan'):.4f} | Spearman(cons,risk) {rho:+.2f}")
