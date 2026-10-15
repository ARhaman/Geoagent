"""
geo_analysis.py -- does the reference-free gate (cons_tta8, no ground truth) rank patches by GEOLOGICAL quality?
 (1) Is SSIM itself a good proxy for geological quality?  Spearman(ssim_tta8, geo metric)
 (2) Spearman(gate score, geo metric)
 (3) Selective prediction on geological risk, no threshold needed: risk = 1-pixel accuracy (facies) or MAE (property).
     AURC for random / gate / oracle ranking, and the risk of the kept set at 90% / 80% coverage.
"""
import argparse, os, numpy as np, pandas as pd
p = argparse.ArgumentParser(); p.add_argument("--rev_dir", default=r"D:\First year\Papers\GeoAgent\revision_ACAGS")
a = p.parse_args()
def rank_corr(x, y): return pd.Series(x).rank().corr(pd.Series(y).rank())
def aurc(score, risk):                       # accept highest score first; mean risk of accepted set, averaged over coverage
    o = np.argsort(-score); return (np.cumsum(risk[o]) / np.arange(1, len(risk) + 1)).mean()
for t in ["PorFac", "PermFac"]:
    for d in ["AB", "BA"]:
        f = os.path.join(a.rev_dir, "geo", f"{t}_{d}_test.csv")
        if not os.path.exists(f): continue
        g = pd.read_csv(f).set_index("file"); b = pd.read_csv(os.path.join(a.rev_dir, "policy", f"{t}_{d}_test.csv")).set_index("file")
        x = g.join(b[["ssim_tta8", "cons_tta8"]], how="inner").dropna(subset=["ssim_tta8", "cons_tta8"])
        metrics = {"pix_acc": 1, "miou": 1, "prop_err": -1} if d == "AB" else {"mae": -1}
        print(f"\n{t} {d}: n={len(x)} patches with a decodable target")
        for m, sign in metrics.items():
            if m not in x or x[m].isna().all(): continue
            xx = x.dropna(subset=[m]); q = sign * xx[m].values            # q: higher = better geology
            risk = -q if sign == 1 else xx[m].values                      # risk: higher = worse
            if sign == 1: risk = 1 - xx[m].values
            full = risk.mean(); ar, ag, ao = full, aurc(xx.cons_tta8.values, risk), aurc(-risk, risk)
            o = np.argsort(-xx.cons_tta8.values); r90 = risk[o[:int(.9 * len(o))]].mean(); r80 = risk[o[:int(.8 * len(o))]].mean()
            print(f"  {m:8s} mean {xx[m].mean():.4f} | rho(SSIM,{m}) {rank_corr(xx.ssim_tta8.values, q):+.3f} | rho(gate,{m}) {rank_corr(xx.cons_tta8.values, q):+.3f}"
                  f" | AURC random {ar:.4f} gate {ag:.4f} oracle {ao:.4f} (gate closes {(ar-ag)/(ar-ao):.0%}) | risk kept 90%: {r90:.4f}, 80%: {r80:.4f} vs all {full:.4f}")
