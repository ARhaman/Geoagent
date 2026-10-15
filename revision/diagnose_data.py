"""
diagnose_data.py -- is permeability harder because of the DATA (saturation, quantisation, discontinuity)? Uses TARGET property images of the TEST split (direction BA). No training.
Per task: undecodable fraction, saturated fraction (top/bottom 2% of colour bar), colour-bar levels occupied, share of pixel mass in the 3 most-used levels,
 roughness = within-facies semivariogram at lag 1 / within-facies variance (0 = smooth, 1 = white noise), and fraction of large jumps between neighbours (> 15% of the bar range).
Run (geoldm_gpu):  python diagnose_data.py
"""
import os, sys, numpy as np, torch
sys.path.insert(0, r"D:\First year\Papers\GeoAgent"); sys.path.insert(0, r"D:\First year\Papers\GeoAgent\revision_ACAGS")
from geoagent_dataset import PairedGeoDataset
import legend_decode as L
from torch.utils.data import DataLoader
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu"); rgb = lambda x: ((x.clamp(-1, 1) + 1) / 2 * 255).permute(0, 2, 3, 1)
def nearest(px, lut, chunk=65536):
    shp = px.shape[:-1]; px = px.reshape(-1, 3); idx = torch.empty(px.shape[0], dtype=torch.long, device=px.device); dist = torch.empty(px.shape[0], device=px.device)
    for i in range(0, px.shape[0], chunk): m, ix = torch.cdist(px[i:i + chunk], lut).min(1); idx[i:i + chunk] = ix; dist[i:i + chunk] = m
    return idx.view(shp), dist.view(shp)
FAC = torch.tensor(L.FACIES_RGB, device=dev)
for task in ("PorFac", "PermFac"):
    LUT = torch.tensor(L.PORO_RGB if task == "PorFac" else L.PERM_RGB, device=dev); VAL = torch.tensor(L.PORO_VAL if task == "PorFac" else L.PERM_LOG10, dtype=torch.float32, device=dev); NB = VAL.shape[0]; edge = int(0.02 * NB)
    rng_ = float(VAL.max() - VAL.min()); ds = PairedGeoDataset(os.path.join(r"E:\2D Dutch's Dataset\patches_512", task, "test"), direction="BA", image_size=256, augment=False); ds.files = ds.files[::3][:1500]
    tot = inside_n = undec = sat = 0; occ = torch.zeros(NB, device=dev, dtype=torch.float64); g1 = v1 = 0.0; g4 = 0.0; npair = npair4 = 0; jumps = jp = 0
    with torch.no_grad():
        for b in DataLoader(ds, batch_size=32, shuffle=False, num_workers=0):
            Y = rgb(b["target"].to(dev)); X = rgb(b["input"].to(dev)); s = Y.sum(-1); inside = (s > 30) & (s < 740); it, dt = nearest(Y, LUT); fi, _ = nearest(X, FAC)
            good = inside & (dt < 45); inside_n += int(inside.sum()); undec += int((inside & ~good).sum()); sat += int((good & ((it < edge) | (it >= NB - edge))).sum()); tot += int(good.sum())
            occ += torch.bincount(it[good], minlength=NB).double(); v = VAL[it]
            for l, tag in ((1, 1), (4, 4)):
                for sa, sb in ((np.s_[:, :, :-l], np.s_[:, :, l:]), (np.s_[:, :-l, :], np.s_[:, l:, :])):
                    pm = good[sa] & good[sb] & (fi[sa] == fi[sb]); d = (v[sa][pm] - v[sb][pm]).double()
                    if l == 1: g1 += float((d ** 2).sum() / 2); npair += int(pm.sum()); jumps += int((d.abs() > 0.15 * rng_).sum()); jp += int(pm.sum())
                    else: g4 += float((d ** 2).sum() / 2); npair4 += int(pm.sum())
            for c in range(3):
                m = good & (fi == c)
                if m.any(): vv = v[m].double(); v1 += float(((vv - vv.mean()) ** 2).sum()); 
    p_ = (occ / occ.sum()).cpu().numpy(); var_within = v1 / max(tot, 1)
    print(f"\n{task} ({'porosity v/v' if task=='PorFac' else 'log10 permeability'}), {len(ds.files)} test patches")
    print(f"  undecodable (off colour bar) {undec/max(inside_n,1):.2%} | saturated at bar limits {sat/max(tot,1):.2%} | levels occupied (>0.1% mass each) {int((p_>0.001).sum())}/{NB} | top-3 levels hold {np.sort(p_)[-3:].sum():.1%}")
    print(f"  within-facies roughness: gamma(1)/var = {g1/max(npair,1)/var_within:.3f} | gamma(4)/var = {g4/max(npair4,1)/var_within:.3f} | big jumps between neighbours (>15% of bar) {jumps/max(jp,1):.3%}")
