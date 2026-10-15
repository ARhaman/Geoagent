"""finetune_analysis.py -- hard-example fine-tuning vs random control vs replay-only control vs untouched backbone, on TEST.
 Hard test set = bottom --q of the BASE model's reference-free score (same for every run). Groups runs by (mode, budget)."""
import argparse, os, re, glob, numpy as np, pandas as pd
p = argparse.ArgumentParser(); p.add_argument("--rev_dir", default=r"D:\First year\Papers\GeoAgent\revision_ACAGS")
p.add_argument("--q", type=float, default=0.10); p.add_argument("--good", type=float, default=0.70)
a = p.parse_args(); rng = np.random.default_rng(0)
def ci(d, n=2000):
    m = d[rng.integers(0, len(d), (n, len(d)))].mean(1); return np.percentile(m, [2.5, 97.5])
for t in ["PorFac", "PermFac"]:
    for d in ["AB", "BA"]:
        base = pd.read_csv(os.path.join(a.rev_dir, "policy", f"{t}_{d}_test.csv")).set_index("file")
        hard = base.cons_tta8 <= base.cons_tta8.quantile(a.q)
        groups = {}
        for f in glob.glob(os.path.join(a.rev_dir, "finetune", f"{t}_{d}_*.csv")):
            m = re.search(r"_(hard|random|replay)_s(\d+)(?:_b([0-9.]+))?\.csv$", f)
            if m: groups.setdefault((float(m.group(3) or 0.10), m.group(1)), []).append(f)
        if not groups: continue
        print(f"\n{t} {d}: test n={len(base)}, hard test patches (base gate bottom {a.q:.0%}) = {int(hard.sum())}; base TTAx8 SSIM hard {base.ssim_tta8[hard].mean():.4f} / rest {base.ssim_tta8[~hard].mean():.4f}")
        for (bud, mode), fs in sorted(groups.items()):
            runs = [pd.read_csv(f).set_index("file").reindex(base.index) for f in sorted(fs)]
            dh = np.mean([r.ssim_ft[hard].values - base.ssim_tta8[hard].values for r in runs], 0)
            dr = np.mean([r.ssim_ft[~hard].values - base.ssim_tta8[~hard].values for r in runs], 0)
            bad0 = (base.ssim_tta8[hard] < a.good).mean(); bad1 = np.mean([(r.ssim_ft[hard] < a.good).mean() for r in runs])
            c1, c2 = ci(dh), ci(dr)
            print(f"  budget {bud:<5} {mode:6s} (seeds={len(runs)}): hard dSSIM {dh.mean():+.4f} [{c1[0]:+.4f},{c1[1]:+.4f}] | rest {dr.mean():+.4f} [{c2[0]:+.4f},{c2[1]:+.4f}] | bad on hard {bad0:.1%} -> {bad1:.1%}")
