"""
ablation_table.py -- agent-vs-no-agent ablation.  Needs geo5\*.csv (ablation_risk.py), policy\*.csv (reference-free cons_tta8), gated\*.json.
Rows: A static single-pass (deliver all) | B + TTA x8 action (deliver all) | C + reliability gate: abstain lowest (1-c) fraction by cons_tta8 (coverage 90/80/70%)
      D risk-controlled: threshold calibrated on VAL so delivered risk <= target = 0.85 x VAL TTAx8 mean risk (test result reported as-is) | E gated learning (mean deployed dRisk from gated\) .
Paired bootstrap 95% CIs on TEST.  Delivered risk = mean risk over patches the agent delivers; abstained patches are routed to review (not scored).
"""
import os, json, glob, numpy as np, pandas as pd
rev = r"D:\First year\Papers\GeoAgent\revision_ACAGS"; rng = np.random.default_rng(0)
def ci(x):
    x = np.asarray(x, float); x = x[~np.isnan(x)]; b = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(1500)]; return x.mean(), np.percentile(b, 2.5), np.percentile(b, 97.5)
print("[risk: AB = 1 - boundary-tolerant facies accuracy ; BA = MAE (v/v or log10 mD); lower is better]")
for task in ("PorFac", "PermFac"):
    for d in ("AB", "BA"):
        try:
            V = pd.read_csv(os.path.join(rev, "geo5", f"{task}_{d}_val.csv")).merge(pd.read_csv(os.path.join(rev, "policy", f"{task}_{d}_val.csv"))[["file", "cons_tta8"]], on="file").dropna(subset=["risk_tta8", "risk_single"])
            T = pd.read_csv(os.path.join(rev, "geo5", f"{task}_{d}_test.csv")).merge(pd.read_csv(os.path.join(rev, "policy", f"{task}_{d}_test.csv"))[["file", "cons_tta8"]], on="file").dropna(subset=["risk_tta8", "risk_single"])
        except Exception as e: print(task, d, "missing:", e); continue
        a_, b_ = T.risk_single.values, T.risk_tta8.values; mA = a_.mean(); m, lo, hi = ci(b_ - a_)
        print(f"\n{task} {d} (n={len(T)} test)\n  A static single-pass, deliver all : risk {mA:.4f}\n  B + TTAx8 action, deliver all      : risk {b_.mean():.4f}  (B-A {m:+.4f} [{lo:+.4f},{hi:+.4f}])")
        for cov in (0.9, 0.8, 0.7):
            thr = np.quantile(V.cons_tta8, 1 - cov); sel = T.cons_tta8.values >= thr; dr = b_[sel].mean(); full_sel = b_[~sel].mean() if (~sel).any() else float("nan")
            bs = [(b_[sel][i].mean()) for i in (rng.integers(0, sel.sum(), sel.sum()) for _ in range(1000))]
            print(f"  C gate, target coverage {cov:.0%} (threshold from VAL): delivered {sel.mean():.1%} | delivered risk {dr:.4f} [{np.percentile(bs,2.5):.4f},{np.percentile(bs,97.5):.4f}] vs A {mA:.4f} | abstained-patch risk {full_sel:.4f}")
        target = 0.85 * V.risk_tta8.mean(); best = None
        for cov in np.arange(1.0, 0.04, -0.01):
            thr = np.quantile(V.cons_tta8, 1 - cov); sel = V.cons_tta8.values >= thr
            if V.risk_tta8.values[sel].mean() <= target: best = thr; break
        if best is None: print(f"  D risk-controlled (target {target:.4f}): no threshold on VAL reaches the target")
        else:
            sel = T.cons_tta8.values >= best; print(f"  D risk-controlled (target {target:.4f} from VAL): TEST delivered {sel.mean():.1%} | TEST delivered risk {b_[sel].mean():.4f} ({'target met' if b_[sel].mean() <= target else 'target NOT met on test'})")
        if "risk_sat_tta8" in T.columns and d == "BA": print(f"  sensitivity: MAE INCLUDING saturated pixels: single {T.risk_sat_single.mean():.4f} | TTAx8 {T.risk_sat_tta8.mean():.4f}")
g = [json.load(open(f)) for f in glob.glob(os.path.join(rev, "gated", "*.json"))]
if g:
    print("\nE gated learning (mean over seeds of deployed dRisk on test; from gated\\):")
    for task in ("PorFac", "PermFac"):
        for d in ("AB", "BA"):
            r = [x for x in g if x["task"] == task and x["direction"] == d]
            if r: print(f"  {task} {d}: accepted {sum(x['accept'] for x in r)}/{len(r)} | deployed dRisk (all patches) {np.mean([x['test_all']['mean'] if x['accept'] else 0 for x in r]):+.4f}")
