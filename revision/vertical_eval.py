"""
vertical_eval.py -- vertical coherence of predicted facies (reviewers: 3D claims).  Facies predicted layer by layer (2D model, TTA x8) from the property rendered on 33 CONSECUTIVE depth layers
(k=100..132) at 10 fixed tiles; compared with the TRUE facies in the same tiles.  Metrics (pixel-level, 256x256):
 accuracy (boundary-tolerant), vertical agreement at lags 1,2,4,8 (same class at layer k and k+lag) for PRED vs TRUTH, and an 'adjacent-layer copy' baseline (predict layer k by TRUE layer k-1).
CAVEAT printed with results: whether layers 100-132 were in the model's training split is not recorded in the dataset.
Rendering: Por = absolute legend scale (validated control); Perm = per-slice log10 min-max (partially validated control).
Run (geoldm_gpu):  python vertical_eval.py --task PorFac      /      python vertical_eval.py --task PermFac
"""
import os, sys, argparse
import numpy as np, torch, torch.nn.functional as Fn
from PIL import Image
p = argparse.ArgumentParser()
p.add_argument("--code_dir", default=r"D:\First year\Papers\GeoAgent"); p.add_argument("--ckpt_base", default=r"D:\First year\Papers\GeoAgent\checkpoints"); p.add_argument("--rev_dir", default=r"D:\First year\Papers\GeoAgent\revision_ACAGS")
p.add_argument("--task", required=True, choices=["PorFac", "PermFac"]); p.add_argument("--k0", type=int, default=100); p.add_argument("--nl", type=int, default=33); p.add_argument("--ntile", type=int, default=10); p.add_argument("--seed", type=int, default=0)
a = p.parse_args(); sys.path.insert(0, a.code_dir); sys.path.insert(0, a.rev_dir)
from nb_model import Pix2PixModel
import legend_decode as L
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu"); GR = os.path.join(a.code_dir, "grids")
F = np.load(os.path.join(GR, "facies.npy"), mmap_mode="r"); V = np.load(os.path.join(GR, "porosity.npy" if a.task == "PorFac" else "perm.npy"), mmap_mode="r"); H, W, K = F.shape
def lut_fn(rgb, val): o = np.argsort(val); v = val[o]; c = rgb[o]; return lambda x: np.stack([np.interp(np.clip(x, v[0], v[-1]), v, c[:, k]) for k in range(3)], -1)
f_poro = lut_fn(L.PORO_RGB, L.PORO_VAL); f_perm = lut_fn(L.PERM_RGB, L.PERM_LOG10)
def prop_rgb(slice_full, blk):
    if a.task == "PorFac": return f_poro(blk)
    lg = np.log10(np.maximum(slice_full, 1e-6)); lo, hi = np.nanmin(lg), np.nanmax(lg); return f_perm(np.clip((np.log10(np.maximum(blk, 1e-6)) - lo) / (hi - lo), 0, 1) * 3.0)
ks = list(range(a.k0, a.k0 + a.nl)); valid = np.ones((H, W), bool)
for k in ks: valid &= ~np.isnan(np.asarray(F[:, :, k]))
cands = [(r, c) for r in range(0, H - 63, 48) for c in range(0, W - 63, 48) if valid[r:r + 64, c:c + 64].all()]; rng = np.random.default_rng(a.seed); pos = [cands[i] for i in rng.choice(len(cands), min(a.ntile, len(cands)), replace=False)]
print(f"{a.task}: {len(cands)} tiles valid in all {a.nl} layers; using {len(pos)}")
up = lambda x: np.kron(x, np.ones((8, 8, 1)))
def tens(img): im = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).resize((256, 256), Image.BILINEAR); return (torch.from_numpy(np.asarray(im)).float().permute(2, 0, 1) / 255 - 0.5) / 0.5
M = Pix2PixModel(device=dev); M.load(os.path.join(a.ckpt_base, f"{a.task}_AB", "pix2pix_AB_ep0100.pth")); M.G.eval()
def d4(k_, f):
    fw = lambda x: torch.rot90(torch.flip(x, [-1]) if f else x, k_, [-2, -1]); iv = lambda y: (torch.flip(torch.rot90(y, -k_, [-2, -1]), [-1]) if f else torch.rot90(y, -k_, [-2, -1])); return fw, iv
T8 = [d4(k_, f) for f in (False, True) for k_ in range(4)]
FAC = torch.tensor(L.FACIES_RGB, device=dev); rgb = lambda x: ((x.clamp(-1, 1) + 1) / 2 * 255).permute(0, 2, 3, 1)
def cls(x):
    R = rgb(x); s = R.sum(-1); d = torch.cdist(R.reshape(-1, 3), FAC); m, ix = d.min(1); return ix.view(R.shape[:-1]), (m.view(R.shape[:-1]) < 60) & (s > 30) & (s < 740)
PC = np.zeros((len(pos), a.nl, 256, 256), np.int8); TC = np.zeros_like(PC); OK = np.zeros((len(pos), a.nl, 256, 256), bool)
with torch.no_grad():
    for ti, (r, c) in enumerate(pos):
        for li, k in enumerate(ks):
            full = np.asarray(V[:, :, k]); blk = full[r:r + 64, c:c + 64]; sf = np.asarray(F[r:r + 64, c:c + 64, k]); x = tens(up(prop_rgb(full, blk))).unsqueeze(0).to(dev); y = tens(up(L.FACIES_RGB[sf.astype(int)])).unsqueeze(0).to(dev)
            pr = torch.stack([iv(M.G(fw(x))) for fw, iv in T8]).mean(0); pc, _ = cls(pr); tc, ok = cls(y); PC[ti, li] = pc.cpu().numpy(); TC[ti, li] = tc.cpu().numpy(); OK[ti, li] = ok.cpu().numpy()
def bd(c, k=5): t = torch.from_numpy(c.astype(np.float32))[:, None]; return (Fn.max_pool2d(t, k, 1, k // 2) != -Fn.max_pool2d(-t, k, 1, k // 2)).squeeze(1).numpy()
acc = []
for ti in range(len(pos)):
    B = bd(TC[ti]); m = OK[ti] & ~B; acc.append((PC[ti][m] == TC[ti][m]).mean())
print(f"facies accuracy on these tiles (boundary-tolerant, TTAx8): {np.mean(acc):.4f}")
print("vertical agreement (same class at layer k and k+lag), pixels valid in both:")
print("  lag | TRUE | PRED | copy-adjacent-TRUE-layer baseline accuracy (lag 1 only)")
for lag in (1, 2, 4, 8):
    mt = []; mp = []
    for ti in range(len(pos)):
        m = OK[ti][:-lag] & OK[ti][lag:]; mt.append(((TC[ti][:-lag] == TC[ti][lag:])[m]).mean()); mp.append(((PC[ti][:-lag] == PC[ti][lag:])[m]).mean())
    extra = ""
    if lag == 1:
        cp = []
        for ti in range(len(pos)): m = OK[ti][:-1] & OK[ti][1:]; cp.append((TC[ti][:-1][m] == TC[ti][1:][m]).mean())
        extra = f" | {np.mean(cp):.4f}"
    print(f"  {lag:3d} | {np.mean(mt):.4f} | {np.mean(mp):.4f}{extra}")
print("CAVEAT: membership of layers 100-132 in the training split is not recorded in the dataset; values are therefore descriptive of vertical coherence, not held-out accuracy.")
