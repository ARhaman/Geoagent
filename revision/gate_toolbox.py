"""
gate_toolbox.py -- E1: does an agent with a richer toolbox have anything to choose from?
For each patch computes, with TRUE SSIM (evaluation only) and agreement-with-default (reference-free):
  tta8      default tool (8-symmetry TTA, final checkpoint)
  snap      snapshot ensemble: TTA8 of checkpoint A and checkpoint B averaged   (--epoch_a / --epoch_b)
  med3      3x3 median filter of tta8      (remedy for speckle)
  snap_med3 3x3 median filter of snap
Run (geoldm_gpu):  python gate_toolbox.py --task PorFac --direction AB --split test
Then:              python toolbox_analysis.py --task PorFac --direction AB
"""
import os, sys, argparse, csv, time
import numpy as np, torch, torch.nn.functional as Fn
from torch.utils.data import DataLoader
p = argparse.ArgumentParser()
p.add_argument("--code_dir", default=r"D:\First year\Papers\GeoAgent")
p.add_argument("--data_base", default=r"E:\2D Dutch's Dataset\patches_512")
p.add_argument("--ckpt_base", default=r"D:\First year\Papers\GeoAgent\checkpoints")
p.add_argument("--task", required=True, choices=["PorFac", "PermFac"])
p.add_argument("--direction", required=True, choices=["AB", "BA"])
p.add_argument("--split", required=True, choices=["val", "test"])
p.add_argument("--epoch_a", default="0100"); p.add_argument("--epoch_b", default="0080")
p.add_argument("--n", type=int, default=0); p.add_argument("--bs", type=int, default=32)
p.add_argument("--out_dir", default=r"D:\First year\Papers\GeoAgent\revision_ACAGS\toolbox")
a = p.parse_args()
sys.path.insert(0, a.code_dir); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nb_model import Pix2PixModel
from geoagent_dataset import PairedGeoDataset
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def load(ep):
    m = Pix2PixModel(device=dev)
    m.load(os.path.join(a.ckpt_base, f"{a.task}_{a.direction}", f"pix2pix_{a.direction}_ep{ep}.pth")); m.G.eval(); return m
MA, MB = load(a.epoch_a), load(a.epoch_b)
ds = PairedGeoDataset(os.path.join(a.data_base, a.task, a.split), direction=a.direction, image_size=256, augment=False)
if a.n > 0: ds.files = ds.files[:a.n]
dl = DataLoader(ds, batch_size=a.bs, shuffle=False, num_workers=0)
def ssim_g(p_, t_):
    p_ = (p_ + 1) / 2; t_ = (t_ + 1) / 2; C1, C2 = 0.01 ** 2, 0.03 ** 2; d = (1, 2, 3)
    mp, mt = p_.mean(d), t_.mean(d); sp, st = p_.std(d, unbiased=False), t_.std(d, unbiased=False)
    spt = ((p_ - mp[:, None, None, None]) * (t_ - mt[:, None, None, None])).mean(d)
    return ((2 * mp * mt + C1) * (2 * spt + C2)) / ((mp ** 2 + mt ** 2 + C1) * (sp ** 2 + st ** 2 + C2))
def d4(k, f):
    fw = lambda x: torch.rot90(torch.flip(x, [-1]) if f else x, k, [-2, -1])
    iv = lambda y: (torch.flip(torch.rot90(y, -k, [-2, -1]), [-1]) if f else torch.rot90(y, -k, [-2, -1]))
    return fw, iv
T8 = [d4(k, f) for f in (False, True) for k in range(4)]
def tta8(M, x): return torch.stack([iv(M.G(fw(x))) for fw, iv in T8]).mean(0)
def med3(x):
    B, C, H, W = x.shape; u = Fn.unfold(Fn.pad(x, (1, 1, 1, 1), mode="reflect"), 3)       # (B, C*9, H*W)
    return u.view(B, C, 9, H * W).median(2).values.view(B, C, H, W)
rows = []; t0 = time.time()
with torch.no_grad():
    for bi, b in enumerate(dl):
        x = b["input"].to(dev); y = b["target"].to(dev)
        pa = tta8(MA, x); pb = tta8(MB, x); sn = (pa + pb) / 2
        out = {"tta8": pa, "snap": sn, "med3": med3(pa), "snap_med3": med3(sn)}
        for i in range(x.shape[0]):
            r = {"file": os.path.basename(b["path"][i])}
            for k, v in out.items():
                r[f"ssim_{k}"] = float(ssim_g(v[i:i+1], y[i:i+1])); r[f"agree_{k}"] = float(ssim_g(v[i:i+1], pa[i:i+1]))
            rows.append(r)
        print(f"batch {bi+1}/{len(dl)} " + " ".join(f"{k}={np.mean([r['ssim_'+k] for r in rows]):.4f}" for k in out) + f" ({time.time()-t0:.0f}s)", flush=True)
os.makedirs(a.out_dir, exist_ok=True)
fn = os.path.join(a.out_dir, f"{a.task}_{a.direction}_{a.split}.csv")
with open(fn, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print("Saved", fn, len(rows))
