"""
shift_render.py -- render orientation-shift test sets from the NAM grids (grids\*.npy) in the TRAINING legend (legend_decode LUTs), no GPU.
 K = depth slices (orientation seen in training; rendering-fidelity CONTROL), I and J = vertical cross-sections (orientation NEVER seen in training).
 Tiles = 64x64 cells (stride 48) rendered x8 -> 512x512 (same scale as the training tiles), only tiles fully inside the field (training tiles contain no background).
 Cells are rendered square (vertical sections are therefore stretched/compressed relative to true thickness: stated limitation).
 Pair images (1024x512: property | facies) are written to shift\{task}\{K|I|J}\ and read by the normal PairedGeoDataset.
Run:  python shift_render.py            (about 2-5 min, CPU)
"""
import os, sys, numpy as np
from PIL import Image
GR = r"D:\First year\Papers\GeoAgent\grids"; REV = r"D:\First year\Papers\GeoAgent\revision_ACAGS"; sys.path.insert(0, REV)
import legend_decode as L
rng = np.random.default_rng(0); N_PER = int(sys.argv[1]) if len(sys.argv) > 1 else 500
F = np.load(os.path.join(GR, "facies.npy"), mmap_mode="r"); P = np.load(os.path.join(GR, "porosity.npy"), mmap_mode="r"); K = np.load(os.path.join(GR, "perm.npy"), mmap_mode="r")
def lut_fn(rgb, val, lo=None, hi=None):
    o = np.argsort(val); v = val[o]; c = rgb[o]
    return lambda x: np.stack([np.interp(np.clip(x, v[0], v[-1]), v, c[:, k]) for k in range(3)], -1)
f_poro = lut_fn(L.PORO_RGB, L.PORO_VAL); f_perm = lut_fn(L.PERM_RGB, L.PERM_LOG10)
def facies_rgb(a): return L.FACIES_RGB[np.clip(np.nan_to_num(a, nan=0).astype(int), 0, 2)]
def slice_(vol, axis, idx):
    s = np.asarray(np.take(vol, idx, axis=axis)); return s if axis == 2 else s.T   # K: (I,J) ; I: (J,K)->(K,J) rows=depth ; J: (I,K)->(K,I)
up = lambda a: np.kron(a, np.ones((8, 8, 1))) if a.ndim == 3 else np.kron(a, np.ones((8, 8)))
def norm_poro(sv_full):        # per-slice percentile normalisation (p2/p98), mapped onto the legend bar 0.04..0.147
    lo, hi = np.nanquantile(sv_full, [0.02, 0.98]); return lambda x: f_poro(0.04 + np.clip((x - lo) / (hi - lo), 0, 1) * (0.1471 - 0.04))
def norm_perm(sv_full):        # per-slice log10 min-max normalisation, mapped onto the legend bar 0..3 (log10 mD)
    lg = np.log10(np.maximum(sv_full, 1e-6)); lo, hi = np.nanmin(lg), np.nanmax(lg); return lambda x: f_perm(np.clip((np.log10(np.maximum(x, 1e-6)) - lo) / (hi - lo), 0, 1) * 3.0)
for task, V, nf in (("PorFacN", P, norm_poro), ("PermFacN", K, norm_perm)):
    for name, axis in (("K", 2), ("I", 0), ("J", 1)):
        out = os.path.join(REV, "shift", task, name); os.makedirs(out, exist_ok=True)
        n_sl = V.shape[axis]; cands = []
        for idx in range(n_sl):
            sf = slice_(F, axis, idx); H, W = sf.shape
            for r in range(0, H - 63, 48):
                for c in range(0, W - 63, 48):
                    if not np.isnan(sf[r:r + 64, c:c + 64]).any(): cands.append((idx, r, c))
        sel = [cands[i] for i in rng.choice(len(cands), min(N_PER, len(cands)), replace=False)]; sel.sort()
        for n, (idx, r, c) in enumerate(sel):
            sf = slice_(F, axis, idx)[r:r + 64, c:c + 64]; sfull = slice_(V, axis, idx); sv = sfull[r:r + 64, c:c + 64]; fn = nf(sfull)
            if np.isnan(sv).any(): continue
            left = np.clip(fn(sv), 0, 255).astype(np.uint8); right = facies_rgb(sf).astype(np.uint8)
            im = np.concatenate([up(left), up(right)], 1).astype(np.uint8); Image.fromarray(im).save(os.path.join(out, f"{n:06d}_{name}{idx}_r{r}_c{c}.png"))
        print(task, name, "candidates", len(cands), "written", len(sel), flush=True)
