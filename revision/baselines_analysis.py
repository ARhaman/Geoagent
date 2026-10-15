"""baselines_analysis.py -- GAN (TTAx8) vs conventional baselines on the SAME test patches (paired bootstrap on the per-patch difference)."""
import argparse, os, numpy as np, pandas as pd
p = argparse.ArgumentParser(); p.add_argument("--rev_dir", default=r"D:\First year\Papers\GeoAgent\revision_ACAGS"); p.add_argument("--sub", default="geo3"); a = p.parse_args(); rng = np.random.default_rng(0)
for t in ["PorFac", "PermFac"]:
    for d in ["AB", "BA"]:
        f = os.path.join(a.rev_dir, a.sub, f"{t}_{d}_test.csv")
        if not os.path.exists(f): continue
        g = pd.read_csv(os.path.join(a.rev_dir, "geo2", f"{t}_{d}_test.csv")).set_index("file"); b = pd.read_csv(f).set_index("file")
        mets = ["pix_acc_tol", "pix_acc", "prop_err"] if d == "AB" else ["mae"]; name = ("Bayes classifier" if d == "AB" else "class-mean lookup") if a.sub == "geo3" else "context-aware MLP"
        print(f"\n{t} {d}: baseline = {name}")
        for m in mets:
            z = g[[m]].join(b[[m]], lsuffix="_gan", rsuffix="_base", how="inner").dropna(); dlt = (z[m + "_gan"] - z[m + "_base"]).values
            bs = np.percentile([dlt[i].mean() for i in (rng.integers(0, len(dlt), len(dlt)) for _ in range(1000))], [2.5, 97.5]); better = "higher" if m.startswith("pix_acc") else "lower"
            print(f"  {m:12s} GAN {z[m+'_gan'].mean():.4f} | baseline {z[m+'_base'].mean():.4f} | GAN minus baseline {dlt.mean():+.4f} [95% CI {bs[0]:+.4f},{bs[1]:+.4f}] ({better} is better) | n={len(z)}")
