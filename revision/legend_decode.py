"""
legend_decode.py -- converts rendered RGB maps back to class / physical values using the ORIGINAL legend colour bars
(read from legend.png).  Valid only inside the legend range; pixels outside the range saturate in the render
(porosity >~0.147 -> red, <0.04 -> blue; permeability >1000 -> red, <1 -> magenta) and cannot be decoded beyond that.
 Facies : 0 = sand (yellow), 1 = shaly sand (orange), 2 = shale (gray); -1 = background / unmatched
 Porosity (v/v) and permeability (log10 mD): float, NaN where the pixel is background or not on the colour bar.
"""
import os, numpy as np
from PIL import Image
HERE = os.path.dirname(os.path.abspath(__file__))
def _build():
    im = np.array(Image.open(os.path.join(HERE, "legend.png")).convert("RGB")).astype(np.float32)
    pe = im[568:893, 45:90].mean(1)                         # permeability bar, top (1000 mD) -> bottom (1 mD)
    pe_val = 3.0 - 3.0 * (np.arange(pe.shape[0]) / (pe.shape[0] - 1))            # log10 mD
    po = im[1011:1328, 52:95].mean(1)                       # porosity bar, top (~0.1471) -> bottom (0.04)
    po_val = 0.04 + (1328 - np.arange(1011, 1328)) * 0.01 / 29.6
    return pe, pe_val, po, po_val
PERM_RGB, PERM_LOG10, PORO_RGB, PORO_VAL = _build()
FACIES_RGB = np.array([[255, 255, 0], [255, 170, 0], [122, 122, 122]], np.float32)   # sand, shaly sand, shale (measured from the rendered layers)
def _nearest(px, lut, chunk=200000):
    idx = np.empty(len(px), np.int32); d = np.empty(len(px), np.float32)
    for i in range(0, len(px), chunk):
        dd = ((px[i:i + chunk, None, :] - lut[None]) ** 2).sum(2); idx[i:i + chunk] = dd.argmin(1); d[i:i + chunk] = np.sqrt(dd.min(1))
    return idx, d
def _decode(rgb, lut, vals, tol):
    px = rgb.reshape(-1, 3).astype(np.float32); out = np.full(len(px), np.nan, np.float32)
    s = px.sum(1); ok = (s > 30) & (s < 740)
    if ok.any():
        i, d = _nearest(px[ok], lut); v = vals[i].astype(np.float32); v[d > tol] = np.nan; out[ok] = v
    return out.reshape(rgb.shape[:2])
def decode_poro(rgb, tol=45): return _decode(rgb, PORO_RGB, PORO_VAL, tol)
def decode_perm_log10(rgb, tol=45): return _decode(rgb, PERM_RGB, PERM_LOG10, tol)
def decode_facies(rgb, tol=60):
    px = rgb.reshape(-1, 3).astype(np.float32); out = np.full(len(px), -1, np.int8); s = px.sum(1); ok = (s > 30) & (s < 740)
    if ok.any():
        i, d = _nearest(px[ok], FACIES_RGB); c = i.astype(np.int8); c[d > tol] = -1; out[ok] = c
    return out.reshape(rgb.shape[:2])
