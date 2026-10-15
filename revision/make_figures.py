import os, json, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
rev = os.path.dirname(os.path.abspath(__file__)); out = os.path.join(rev, "figures"); os.makedirs(out, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
C = dict(blue="#2b6cb0", green="#2f855a", orange="#dd6b20", red="#c53030", grey="#718096", purple="#6b46c1", teal="#2c7a7b", light="#f7fafc")
def box(ax, x, y, w, h, text, fc, ec=None, fs=8.5, tc="black", bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12", fc=fc, ec=ec or fc, lw=1.2)); ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=tc, fontweight="bold" if bold else "normal", wrap=True)
def arrow(ax, p, q, c="#4a5568", style="-|>", lw=1.4, rad=0.0): ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=12, lw=lw, color=c, connectionstyle=f"arc3,rad={rad}"))
# ---------------- Fig 1: overview
fig, ax = plt.subplots(figsize=(10.5, 6.0)); ax.set_xlim(0, 21); ax.set_ylim(0, 12); ax.axis("off")
box(ax, 0.3, 8.0, 4.6, 3.4, "Groningen static model\n(facies, porosity,\npermeability grids)\nsliced, rendered, tiled into\npaired 256×256 patches", "#ebf8ff", C["blue"], fs=8)
box(ax, 5.5, 8.0, 4.4, 3.4, "Pix2Pix backbone\none model per direction\nPor→Fac · Fac→Por\nPerm→Fac · Fac→Perm", "#ebf8ff", C["blue"], fs=8)
arrow(ax, (4.9, 9.7), (5.5, 9.7), C["blue"])
box(ax, 10.5, 8.0, 4.2, 3.4, "PERCEIVE\nTTA×8 prediction +\nreference-free\nreliability estimate", "#f0fff4", C["green"], fs=8)
arrow(ax, (9.9, 9.7), (10.5, 9.7), C["green"])
box(ax, 15.4, 8.0, 5.3, 3.4, "DECIDE\nthresholds calibrated on\nvalidation data:\naccept · review · abstain", "#fffaf0", C["orange"], fs=8)
arrow(ax, (14.7, 9.7), (15.4, 9.7), C["orange"])
box(ax, 15.4, 4.2, 5.3, 2.9, "DELIVER\naccepted maps released;\nreview / abstain → geologist,\nlabels requested", "#fffaf0", C["orange"], fs=8); arrow(ax, (18.05, 8.0), (18.05, 7.1), C["orange"])
box(ax, 8.6, 4.2, 5.6, 2.9, "LEARN (validation-gated)\nflagged patches labelled → fine-tune;\ndeploy only if held-out check\nimproves, otherwise ROLL BACK", "#faf5ff", C["purple"], fs=8); arrow(ax, (15.4, 5.65), (14.2, 5.65), C["purple"]); arrow(ax, (12.6, 7.1), (12.6, 8.0), C["purple"])
box(ax, 0.3, 0.4, 20.4, 2.7, "AUDIT & ORCHESTRATION (LLM)\nApplies the written policy through tool calls (get_reliability → decide), writes a machine-readable trace and a batch audit note;\ndecisions are verified against the deterministic policy. The LLM does not score map quality.", "#fff5f5", C["red"], fs=8)
for x in (11.4, 18.05): arrow(ax, (x, 4.2), (x, 3.1), C["red"])
ax.text(2.6, 6.6, "Baselines for comparison:\nclass-mean lookup · Bayes classifier ·\ncontext-aware neural network", ha="center", fontsize=7.8, color=C["grey"]); arrow(ax, (2.6, 8.0), (2.6, 7.4), C["grey"])
plt.tight_layout(); plt.savefig(os.path.join(out, "Fig1_new.png"), dpi=220); plt.close()
# ---------------- Fig 3: controller
fig, ax = plt.subplots(figsize=(10.5, 4.4)); ax.set_xlim(0, 21); ax.set_ylim(0, 8.4); ax.axis("off")
st = [("1  PERCEIVE", "8 dihedral TTA passes\n(4 rotations × flip),\naveraged\n\nreliability features:\n· TTA consistency\n· map complexity\n· input OOD distance", C["green"], "#f0fff4"),
      ("2  DECIDE", "learned estimator\n(ridge, fitted on VAL)\n→ percentile vs VAL\n\n≥ accept threshold: accept\nmiddle band: review\nlow: abstain", C["orange"], "#fffaf0"),
      ("3  ACT", "accepted: release\nwith trace\nreview / abstain:\nroute to geologist,\nrequest labels", C["blue"], "#ebf8ff"),
      ("4  LEARN", "fine-tune on labelled\nhard patches + replay\n\nheld-out CHECK split:\nhard patches improve AND\noverall non-inferior?\nyes: deploy · no: rollback", C["purple"], "#faf5ff"),
      ("5  AUDIT", "LLM orchestrator\ncalls tools under a\nwritten policy\nJSON trace per patch\nbatch audit note\n(numbers traceable)", C["red"], "#fff5f5")]
for i, (t, body, ec, fc) in enumerate(st):
    x = 0.2 + i * 4.18; box(ax, x, 1.0, 3.8, 7.0, "", fc, ec); ax.text(x + 1.9, 7.3, t, ha="center", fontsize=10, fontweight="bold", color=ec); ax.text(x + 1.9, 4.0, body, ha="center", va="center", fontsize=8)
    if i < 4: arrow(ax, (x + 3.8, 4.5), (x + 4.18, 4.5), "#4a5568")
ax.text(10.5, 0.35, "Thresholds and the estimator are calibrated on validation data only; test data are used for audit.", ha="center", fontsize=8.5, color=C["grey"])
plt.tight_layout(); plt.savefig(os.path.join(out, "Fig3_new.png"), dpi=220); plt.close()
# ---------------- Fig 5: risk-coverage
def load(t, d, s):
    g = pd.read_csv(os.path.join(rev, "geo2", f"{t}_{d}_{s}.csv")).set_index("file"); b = pd.read_csv(os.path.join(rev, "policy", f"{t}_{d}_{s}.csv")).set_index("file")
    a5 = pd.read_csv(os.path.join(rev, "geo5", f"{t}_{d}_{s}.csv")).set_index("file")[["risk_single"]]; df = g.join(b[["ssim_tta8", "cons_tta5", "cons_tta8", "cons_mc10"]], how="inner").join(a5, how="inner")
    df["risk"] = (1 - df["pix_acc_tol"]) if d == "AB" else df["mae"]; return df.dropna(subset=["risk", "risk_single"])
def ridge_fit(X, y, alpha=10.0):
    mu, sd = X.mean(0), X.std(0) + 1e-9; Z1 = np.c_[np.ones(len(X)), (X - mu) / sd]; w = np.linalg.solve(Z1.T @ Z1 + alpha * np.diag([0] + [1] * X.shape[1]), Z1.T @ y); return lambda Xn: np.c_[np.ones(len(Xn)), (Xn - mu) / sd] @ w
sq = lambda df, cols: np.c_[df[cols].values, df[cols].values ** 2]
fig, axs = plt.subplots(1, 4, figsize=(12.5, 3.3)); names = {"PorFac_AB": "Porosity → Facies", "PorFac_BA": "Facies → Porosity", "PermFac_AB": "Permeability → Facies", "PermFac_BA": "Facies → Permeability"}
for ax, (k, nm) in zip(axs, names.items()):
    t, d = k.split("_"); v, x = load(t, d, "val"), load(t, d, "test"); gate = ["cons_tta5", "cons_tta8", "cons_mc10"]; feats = [c for c in v.columns if c.startswith("f_")]
    m = ridge_fit(sq(v, gate + feats), v.risk.values); r = x.risk.values; cov = np.linspace(0.3, 1.0, 36)
    def curve(s): o = np.argsort(-s); cr = np.cumsum(r[o]) / np.arange(1, len(r) + 1); return [cr[int(c * len(r)) - 1] for c in cov]
    ax.plot(cov, curve(-m(sq(x, gate + feats))), color=C["green"], lw=2, label="learned estimator (no reference)"); ax.plot(cov, curve(x.cons_tta8.values), color=C["blue"], lw=1.6, label="TTA consistency (no reference)"); ax.plot(cov, curve(x.ssim_tta8.values), color=C["grey"], lw=1.4, ls="--", label="SSIM (needs ground truth)")
    ax.axhline(x.risk_single.mean(), color=C["red"], ls=":", lw=1.5, label="static single pass (deliver all)"); ax.axhline(r.mean(), color="black", ls=":", lw=1.0, label="TTA×8, deliver all")
    ax.set_title(nm, fontsize=10); ax.set_xlabel("coverage (fraction delivered)"); ax.set_ylabel("facies error (1 − accuracy)" if d == "AB" else "MAE (colour-scale units)"); ax.grid(alpha=.25); ax.invert_xaxis()
axs[0].legend(fontsize=6.8, loc="lower left"); plt.tight_layout(); plt.savefig(os.path.join(out, "Fig5_new.png"), dpi=220); plt.close()
# ---------------- Fig 7: realism (semivariogram ratio)
fig, axs = plt.subplots(2, 3, figsize=(10.5, 5.6), sharey=True); lags = [1, 2, 4, 8, 16]; cls = ["sand(yellow)", "shaly sand(orange)", "shale(gray)"]
for ri, t in enumerate(("PorFac", "PermFac")):
    J = json.load(open(os.path.join(rev, "geo4", f"{t}_realism.json")))
    for ci, c in enumerate(cls):
        ax = axs[ri, ci]; g = J[c]["gamma"]; tr = np.array(g["truth"])
        ax.plot(lags, np.array(g["gan"]) / tr, "o-", color=C["green"], label="GAN (TTA×8)"); ax.plot(lags, np.array(g["mlp"]) / tr, "s-", color=C["blue"], label="context-aware MLP"); ax.axhline(1, color="black", ls="--", lw=1)
        ax.set_xscale("log", base=2); ax.set_xticks(lags); ax.set_xticklabels(lags); s_ = J[c]["std"]; ax.set_title(f"{'Fac→Por' if t=='PorFac' else 'Fac→Perm'}: {c.split('(')[0]}\nwithin-facies std: truth {s_['truth']:.3g} · GAN {s_['gan']:.3g} · MLP {s_['mlp']:.3g}", fontsize=8.5); ax.grid(alpha=.25)
        if ci == 0: ax.set_ylabel("semivariogram ratio to truth")
        if ri == 1: ax.set_xlabel("lag (px)")
axs[0, 0].legend(fontsize=7.5); plt.tight_layout(); plt.savefig(os.path.join(out, "Fig7_new.png"), dpi=220); plt.close()
print("ok")
