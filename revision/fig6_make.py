"""fig6_make.py -- Figure 6 (within-facies spatial structure): semivariogram ratio to truth for single-pass GAN, GAN TTAx8, context-aware MLP. Needs geo4\\{task}_realism_single.json (realism_single.py)."""
import os, json, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
rev = os.path.dirname(os.path.abspath(__file__)); out = os.path.join(rev, "figures"); os.makedirs(out, exist_ok=True)
fig, axs = plt.subplots(2, 3, figsize=(10.8, 6.0), sharey=True); lags = [1, 2, 4, 8, 16]; cls = ["sand(yellow)", "shaly sand(orange)", "shale(gray)"]
for ri, t in enumerate(("PorFac", "PermFac")):
    J = json.load(open(os.path.join(rev, "geo4", f"{t}_realism_single.json")))
    for ci, c in enumerate(cls):
        ax = axs[ri, ci]; g = J[c]["gamma"]; tr = np.array(g["truth"])
        ax.plot(lags, np.array(g["single"]) / tr, "^-", color="#d62728", label="GAN single pass")
        ax.plot(lags, np.array(g["gan"]) / tr, "o-", color="#2ca02c", label="GAN TTA×8")
        ax.plot(lags, np.array(g["mlp"]) / tr, "s-", color="#1f77b4", label="context-aware MLP"); ax.axhline(1, color="black", ls="--", lw=1)
        ax.set_xscale("log", base=2); ax.set_xticks(lags); ax.set_xticklabels(lags); s_ = J[c]["std"]
        ax.set_title(f"{'Fac→Por' if t=='PorFac' else 'Fac→Perm'}: {c.split('(')[0]}\nstd: truth {s_['truth']:.3g}, single {s_['single']:.3g}\nTTA×8 {s_['gan']:.3g}, MLP {s_['mlp']:.3g}", fontsize=8); ax.grid(alpha=.25)
        if ci == 0: ax.set_ylabel("semivariogram ratio to truth")
        if ri == 1: ax.set_xlabel("lag (px)")
axs[0, 0].legend(fontsize=7.5); plt.tight_layout(); plt.savefig(os.path.join(out, "Fig7_new.png"), dpi=220); print("ok")
