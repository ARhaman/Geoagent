"""
gate_analysis.py -- reads the val/test CSVs from gate_signals.py and answers:
 (1) Which reference-free signal tracks the true SSIM?  (Spearman rho, val and test)
 (2) Can it work as an acceptance gate?  Threshold is calibrated on VAL ONLY:
     tau = 5th percentile of the signal among val samples whose TRUE SSIM >= 0.70
     (i.e. the gate keeps ~95% of truly-good samples). Then applied unchanged to TEST.
 Reported on TEST: accept rate, false-accept rate (accepted but true SSIM < 0.70),
     failure-catch rate (truly bad samples that the gate rejects), AUROC.
Usage: python gate_analysis.py --task PorFac --direction AB
"""
import argparse, os, numpy as np, pandas as pd
def spearmanr(x, y):
    return (pd.Series(x).rank().corr(pd.Series(np.asarray(y)).rank()),)
def roc_auc_score(y, s):
    r = pd.Series(s).rank().values; y = np.asarray(y, bool); n1, n0 = y.sum(), (~y).sum()
    return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
p = argparse.ArgumentParser()
p.add_argument("--dir", default=r"D:\First year\Papers\GeoAgent\revision_ACAGS\signals")
p.add_argument("--task", required=True); p.add_argument("--direction", required=True)
p.add_argument("--good", type=float, default=0.70, help="true-SSIM level that defines a good sample")
p.add_argument("--keep", type=float, default=0.95, help="share of good val samples the gate must keep")
a = p.parse_args()
val = pd.read_csv(os.path.join(a.dir, f"{a.task}_{a.direction}_val.csv"))
tst = pd.read_csv(os.path.join(a.dir, f"{a.task}_{a.direction}_test.csv"))
# higher = better for these; for *_std signals lower = better, so we negate them
signals = {"tta_cons": 1, "mc_cons": 1, "cycle": 1, "tta_std": -1, "mc_std": -1}
print(f"\n{a.task} {a.direction}: val n={len(val)}, test n={len(tst)}; 'good' = true SSIM >= {a.good}")
print(f"test: good={int((tst.ssim_true>=a.good).sum())}  bad={int((tst.ssim_true<a.good).sum())}")
print(f"{'signal':9s} {'rho_val':>8s} {'rho_test':>9s} {'AUROC':>7s} {'tau(val)':>9s} {'accept':>7s} {'false-acc':>10s} {'bad caught':>11s}")
for s, sign in signals.items():
    v, t = sign * val[s].values, sign * tst[s].values
    rv = spearmanr(v, val.ssim_true)[0]; rt = spearmanr(t, tst.ssim_true)[0]
    good_t = (tst.ssim_true.values >= a.good)
    auc = roc_auc_score(good_t, t) if 0 < good_t.sum() < len(good_t) else float("nan")
    tau = np.quantile(v[val.ssim_true.values >= a.good], 1 - a.keep)      # calibrated on VAL only
    acc = t >= tau
    fa = (acc & ~good_t).sum() / max(acc.sum(), 1)
    caught = ((~acc) & ~good_t).sum() / max((~good_t).sum(), 1)
    print(f"{s:9s} {rv:8.3f} {rt:9.3f} {auc:7.3f} {sign*tau:9.4f} {acc.mean():7.3f} {fa:10.3f} {caught:11.3f}")
print("\nReference line: oracle gate (true SSIM >= 0.70) accepts", f"{(tst.ssim_true>=a.good).mean():.3f}", "of test patches.")
