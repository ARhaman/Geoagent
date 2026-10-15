"""
selective_prediction.py -- reference-free gate as SELECTIVE PREDICTION (abstention) with a statistical guarantee.
Action = TTA x8 (best fixed default). Gate score = cons_tta8 (reference-free).  'bad' = true SSIM < --good.
 (a) Risk-coverage: accept the top-c fraction by score; report mean true SSIM of the accepted set and %bad among accepted,
     vs random acceptance and vs an oracle that ranks by true SSIM.  AURC = area under (1 - mean SSIM) vs coverage.
 (b) Risk control: on VAL, pick the LOWEST threshold tau whose Wilson 95% upper bound on P(bad | accepted) is <= alpha;
     apply unchanged to TEST and report realised risk and coverage.
"""
import argparse, os, numpy as np, pandas as pd
p = argparse.ArgumentParser()
p.add_argument("--dir", default=r"D:\First year\Papers\GeoAgent\revision_ACAGS\policy")
p.add_argument("--good", type=float, default=0.70)
p.add_argument("--alphas", default="0.01,0.05")
a = p.parse_args()
def wilson_ub(k, n, z=1.645):
    if n == 0: return 1.0
    ph = k / n; d = 1 + z * z / n
    return (ph + z * z / (2 * n) + z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n))) / d
def rc(score, ssim):
    o = np.argsort(-score); s = ssim[o]; cum = np.cumsum(s) / np.arange(1, len(s) + 1)
    return 1 - cum.mean()                                  # AURC (risk = 1 - mean SSIM of accepted)
alphas = [float(x) for x in a.alphas.split(",")]
print(f"action = TTAx8, gate = cons_tta8 (reference-free); bad = true SSIM < {a.good}")
for t in ["PorFac", "PermFac"]:
    for d in ["AB", "BA"]:
        v = pd.read_csv(os.path.join(a.dir, f"{t}_{d}_val.csv")); x = pd.read_csv(os.path.join(a.dir, f"{t}_{d}_test.csv"))
        s, y = x.cons_tta8.values, x.ssim_tta8.values
        full = y.mean(); aurc_g = rc(s, y); aurc_o = rc(y, y); aurc_r = 1 - full
        print(f"\n{t} {d}: test n={len(x)}, mean SSIM with no abstention = {full:.4f}, %bad = {(y<a.good).mean():.2%}")
        print(f"  AURC (lower better): random {aurc_r:.4f} | reference-free gate {aurc_g:.4f} | oracle {aurc_o:.4f}"
              f"   -> gate closes {(aurc_r-aurc_g)/(aurc_r-aurc_o):.0%} of the random-to-oracle gap")
        o = np.argsort(-s)
        for c in (0.9, 0.8):
            k = int(c * len(x)); sel = o[:k]
            print(f"  keep top {c:.0%}: mean SSIM {y[sel].mean():.4f}, %bad among kept {(y[sel]<a.good).mean():.2%}")
        sv, yv = v.cons_tta8.values, v.ssim_tta8.values; ov = np.argsort(-sv)
        for al in alphas:
            tau = None
            for k in range(len(sv), 0, -1):                 # largest coverage first
                sel = ov[:k]
                if wilson_ub((yv[sel] < a.good).sum(), k) <= al: tau = sv[ov[k - 1]]; break
            if tau is None: print(f"  risk control alpha={al}: no threshold satisfies the bound on VAL"); continue
            acc = s >= tau; nb = (y[acc] < a.good).sum()
            print(f"  risk control alpha={al}: tau={tau:.4f} (from val) -> TEST coverage {acc.mean():.1%}, "
                  f"realised %bad among accepted {nb/max(acc.sum(),1):.2%} ({nb}/{acc.sum()}), mean SSIM accepted {y[acc].mean():.4f}")
