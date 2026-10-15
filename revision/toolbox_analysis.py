"""toolbox_analysis.py -- E1 result: per-tool mean SSIM and the ORACLE ceiling of choosing a tool per patch,
 and how often each tool is the per-patch best. A large oracle gap = there is something for a controller (rule/LLM) to learn."""
import argparse, os, numpy as np, pandas as pd
p = argparse.ArgumentParser(); p.add_argument("--dir", default=r"D:\First year\Papers\GeoAgent\revision_ACAGS\toolbox")
a = p.parse_args(); tools = ["tta8", "snap", "med3", "snap_med3"]
for t in ["PorFac", "PermFac"]:
    for d in ["AB", "BA"]:
        f = os.path.join(a.dir, f"{t}_{d}_test.csv")
        if not os.path.exists(f): continue
        x = pd.read_csv(f); S = x[[f"ssim_{k}" for k in tools]].values; best = S.argmax(1); base = x.ssim_tta8.values
        print(f"\n{t} {d} (n={len(x)})")
        for j, k in enumerate(tools):
            dd = S[:, j] - base; print(f"  {k:10s} mean SSIM {S[:,j].mean():.4f}  vs tta8 {dd.mean():+.4f}   best-for-patch in {np.mean(best==j):.1%}")
        print(f"  ORACLE per-patch choice: {S.max(1).mean():.4f}  (+{(S.max(1)-base).mean():.4f} over tta8)")
