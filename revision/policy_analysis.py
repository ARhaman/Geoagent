"""
policy_analysis.py -- simulates action policies from the gate_policy.py tables.
 Fixed policies : single, tta5, tta8, mc10   (always run this one action)
 Loop           : tta5 -> tta8 -> mc10, stopping at the first action whose reference-free consistency
                  score passes its threshold (thresholds calibrated on VAL only, keep=95% of good val patches);
                  if none passes, returns the LAST attempt flagged 'unaccepted'.
 Oracle         : per patch, the best action by TRUE SSIM (upper bound, not deployable).
Reports on TEST: mean true SSIM/PSNR, paired difference vs fixed tta5 with bootstrap 95% CI,
 loop acceptance / false-accept / mean actions per patch, and what the retry actually buys on the
 patches that the first gate rejected (SSIM of tta5 vs the loop's final answer on those patches).
"""
import argparse, os, numpy as np, pandas as pd
p = argparse.ArgumentParser()
p.add_argument("--dir", default=r"D:\First year\Papers\GeoAgent\revision_ACAGS\policy")
p.add_argument("--task", required=True); p.add_argument("--direction", required=True)
p.add_argument("--good", type=float, default=0.70); p.add_argument("--keep", type=float, default=0.95)
a = p.parse_args()
v = pd.read_csv(os.path.join(a.dir, f"{a.task}_{a.direction}_val.csv"))
t = pd.read_csv(os.path.join(a.dir, f"{a.task}_{a.direction}_test.csv"))
rng = np.random.default_rng(0)
def boot(d, n=2000):
    idx = rng.integers(0, len(d), (n, len(d))); m = d[idx].mean(1); return np.percentile(m, [2.5, 97.5])
acts = ["tta5", "tta8", "mc10"]
tau = {k: np.quantile(v[f"cons_{k}"][v[f"ssim_{k}"] >= a.good], 1 - a.keep) for k in acts}
print(f"\n{a.task} {a.direction}   test n={len(t)}   gate thresholds (val-calibrated): " + "  ".join(f"{k}={tau[k]:.4f}" for k in acts))
print(f"\n{'policy':14s} {'SSIM':>7s} {'PSNR':>7s} {'dSSIM vs tta5 [95% CI]':>28s}")
base = t.ssim_tta5.values
for k in ["single", "tta5", "tta8", "mc10"]:
    d = t[f"ssim_{k}"].values - base; ci = boot(d)
    print(f"{'fixed '+k:14s} {t['ssim_'+k].mean():7.4f} {t['psnr_'+k].mean():7.2f} {d.mean():+9.4f} [{ci[0]:+.4f},{ci[1]:+.4f}]")
# loop simulation
n = len(t); final = np.zeros(n); acc = np.zeros(n, bool); calls = np.zeros(n, int); first_rej = np.zeros(n, bool)
for i in range(n):
    for j, k in enumerate(acts):
        calls[i] += 1; final[i] = t[f"ssim_{k}"].iat[i]
        if t[f"cons_{k}"].iat[i] >= tau[k]:
            acc[i] = True; break
        if j == 0: first_rej[i] = True
d = final - base; ci = boot(d)
print(f"{'loop 5>8>mc10':14s} {final.mean():7.4f} {'':7s} {d.mean():+9.4f} [{ci[0]:+.4f},{ci[1]:+.4f}]")
best = t[[f"ssim_{k}" for k in ["tta5", "tta8", "mc10"]]].max(1).values
d = best - base; ci = boot(d)
print(f"{'oracle best':14s} {best.mean():7.4f} {'':7s} {d.mean():+9.4f} [{ci[0]:+.4f},{ci[1]:+.4f}]   (upper bound)")
good = final >= a.good
print(f"\nLoop: accepted {acc.mean():.3f} | false-accept (accepted but true SSIM<{a.good}) {((acc)&(~good)).sum()/max(acc.sum(),1):.3f}"
      f" | mean actions/patch {calls.mean():.2f} | first-gate rejections {first_rej.sum()} ({first_rej.mean():.3f})")
if first_rej.sum():
    r5 = base[first_rej]; rf = final[first_rej]
    print(f"On the {first_rej.sum()} patches rejected by the first gate: true SSIM tta5 = {r5.mean():.4f} -> loop final = {rf.mean():.4f}  "
          f"(change {np.mean(rf-r5):+.4f}, improved in {np.mean(rf>r5):.1%}, still below {a.good}: {np.mean(rf<a.good):.1%})")
