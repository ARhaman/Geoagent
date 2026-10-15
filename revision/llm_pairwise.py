"""
llm_pairwise.py -- LLM as ACTION SELECTOR (pre-registered test).
For contested TEST patches (where single-pass and TTAx8 disagree most, a reference-free criterion) the blind LLM sees INPUT | A | B in random order
and picks the better prediction. Outcome = geological risk of the chosen candidate (facies: 1 - boundary-tolerant pixel accuracy; property: MAE).
Compared with: always TTAx8, always single, random pick, oracle.  Decision rule fixed in advance: the LLM adds value only if its picks beat
'always TTAx8' with a paired-bootstrap 95% CI excluding 0 in >= 2 of the 4 directions.
Run:  python llm_pairwise.py --task PorFac --direction AB --n 150        (set ANTHROPIC_API_KEY first; safe to restart)
"""
import os, sys, io, json, time, base64, argparse, random, re
import numpy as np, torch, torch.nn.functional as Fn
from PIL import Image
p = argparse.ArgumentParser()
p.add_argument("--code_dir", default=r"D:\First year\Papers\GeoAgent"); p.add_argument("--data_base", default=r"E:\2D Dutch's Dataset\patches_512")
p.add_argument("--ckpt_base", default=r"D:\First year\Papers\GeoAgent\checkpoints"); p.add_argument("--rev_dir", default=r"D:\First year\Papers\GeoAgent\revision_ACAGS")
p.add_argument("--task", required=True, choices=["PorFac", "PermFac"]); p.add_argument("--direction", required=True, choices=["AB", "BA"])
p.add_argument("--n", type=int, default=150); p.add_argument("--pool", type=int, default=500); p.add_argument("--seed", type=int, default=0); p.add_argument("--epoch", default="0100")
p.add_argument("--model", default="claude-sonnet-5"); p.add_argument("--max_tokens", type=int, default=300); p.add_argument("--max_consecutive_errors", type=int, default=5)
p.add_argument("--analyse_only", action="store_true")
a = p.parse_args()
od = os.path.join(a.rev_dir, "llm_pair"); os.makedirs(od, exist_ok=True); fn = os.path.join(od, f"{a.task}_{a.direction}.jsonl")
def analyse():
    recs = {}
    for l in open(fn, encoding="utf-8"):
        r = json.loads(l)
        if "pick" in r: recs[r["file"]] = r
    R = list(recs.values()); n = len(R)
    if n < 20: print(f"{a.task} {a.direction}: only {n} judged"); return
    rt = np.array([r["risk_tta8"] for r in R]); rs = np.array([r["risk_single"] for r in R]); rl = np.array([r["risk_tta8"] if r["pick"] == "tta8" else r["risk_single"] for r in R])
    rng = np.random.default_rng(0); rnd = 0.5 * (rt + rs); orc = np.minimum(rt, rs)
    def ci(x): b = [x[i].mean() for i in (rng.integers(0, n, n) for _ in range(2000))]; return np.percentile(b, [2.5, 97.5])
    d = rl - rt; c = ci(d); agree = np.mean([(r["pick"] == "tta8") == (r["risk_tta8"] <= r["risk_single"]) for r in R])
    print(f"\n{a.task} {a.direction}: n={n} contested patches | mean risk: always-TTAx8 {rt.mean():.4f} | always-single {rs.mean():.4f} | random {rnd.mean():.4f} | LLM picks {rl.mean():.4f} | oracle {orc.mean():.4f}")
    print(f"  LLM minus always-TTAx8: {d.mean():+.4f}  95% CI [{c[0]:+.4f}, {c[1]:+.4f}]  -> {'LLM BETTER' if c[1] < 0 else ('LLM WORSE' if c[0] > 0 else 'no significant difference')}"
          f" | LLM picks the truly better candidate in {agree:.1%} (TTAx8 truly better in {np.mean(rt <= rs):.1%}) | LLM picked TTAx8 in {np.mean([r['pick']=='tta8' for r in R]):.1%}")
if a.analyse_only: analyse(); sys.exit()
key = os.environ.get("ANTHROPIC_API_KEY", "")
if not key: sys.exit("Set ANTHROPIC_API_KEY first.")
import anthropic; client = anthropic.Anthropic(api_key=key)
sys.path.insert(0, a.code_dir); sys.path.insert(0, a.rev_dir)
from nb_model import Pix2PixModel
from geoagent_dataset import PairedGeoDataset
import legend_decode as L
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
M = Pix2PixModel(device=dev); M.load(os.path.join(a.ckpt_base, f"{a.task}_{a.direction}", f"pix2pix_{a.direction}_ep{a.epoch}.pth")); M.G.eval()
ds = PairedGeoDataset(os.path.join(a.data_base, a.task, "test"), direction=a.direction, image_size=256, augment=False)
def d4(k, f):
    fw = lambda x: torch.rot90(torch.flip(x, [-1]) if f else x, k, [-2, -1])
    iv = lambda y: (torch.flip(torch.rot90(y, -k, [-2, -1]), [-1]) if f else torch.rot90(y, -k, [-2, -1]))
    return fw, iv
T8 = [d4(k, f) for f in (False, True) for k in range(4)]
rgb = lambda x: ((x.clamp(-1, 1) + 1) / 2 * 255).permute(1, 2, 0)
def nearest(px, lut, chunk=65536):
    px = px.reshape(-1, 3); idx = torch.empty(px.shape[0], dtype=torch.long, device=px.device); dist = torch.empty(px.shape[0], device=px.device)
    for i in range(0, px.shape[0], chunk): m, ix = torch.cdist(px[i:i + chunk], lut).min(1); idx[i:i + chunk] = ix; dist[i:i + chunk] = m
    return idx, dist
FAC = torch.tensor(L.FACIES_RGB, device=dev)
if a.direction == "BA":
    LUT = torch.tensor(L.PORO_RGB if a.task == "PorFac" else L.PERM_RGB, device=dev); VAL = torch.tensor(L.PORO_VAL if a.task == "PorFac" else L.PERM_LOG10, dtype=torch.float32, device=dev); edge = int(0.02 * VAL.shape[0])
def boundary(c, k=5): c = c.float()[None, None]; return (Fn.max_pool2d(c, k, 1, k // 2) != -Fn.max_pool2d(-c, k, 1, k // 2))[0, 0]
def risk(pred, tgt):                         # pred/tgt: (H,W,3) float 0..255 ; returns geological risk or None
    s = tgt.sum(-1); inside = (s > 30) & (s < 740)
    if a.direction == "AB":
        cp, _ = nearest(pred, FAC); ct, dt = nearest(tgt, FAC); H = tgt.shape[0]; cp, ct, dt = cp.view(H, H), ct.view(H, H), dt.view(H, H)
        m = inside & (dt < 60) & ~boundary(ct)
        return None if int(m.sum()) < 200 else float(1 - (cp[m] == ct[m]).float().mean())
    ip, _ = nearest(pred, LUT); it, dt = nearest(tgt, LUT); H = tgt.shape[0]; ip, it, dt = ip.view(H, H), it.view(H, H), dt.view(H, H)
    m = inside & (dt < 45) & (it >= edge) & (it < VAL.shape[0] - edge)
    return None if int(m.sum()) < 500 else float((VAL[ip][m] - VAL[it][m]).abs().mean())
# --- choose contested patches (reference-free): lowest SSIM between single-pass and TTAx8, among a random pool with a decodable target
def ssim_g(p_, t_):
    p_ = (p_ + 1) / 2; t_ = (t_ + 1) / 2; C1, C2 = 0.01 ** 2, 0.03 ** 2; mp, mt = p_.mean(), t_.mean(); sp, st = p_.std(unbiased=False), t_.std(unbiased=False); spt = ((p_ - mp) * (t_ - mt)).mean()
    return float(((2 * mp * mt + C1) * (2 * spt + C2)) / ((mp ** 2 + mt ** 2 + C1) * (sp ** 2 + st ** 2 + C2)))
random.seed(a.seed); pool = random.sample(range(len(ds)), a.pool); cand = []
with torch.no_grad():
    for i in pool:
        it_ = ds[i]; x = it_["input"].unsqueeze(0).to(dev); y = it_["target"].to(dev)
        mem = torch.stack([iv(M.G(fw(x))) for fw, iv in T8]); p8 = mem.mean(0)[0]; p1 = M.G(x)[0] if False else mem[0][0]
        rt, rs = risk(rgb(p8), rgb(y)), risk(rgb(p1), rgb(y))
        if rt is None or rs is None: continue
        cand.append(dict(i=i, dis=ssim_g(p1, p8), rt=rt, rs=rs))
cand = sorted(cand, key=lambda c: c["dis"])[:a.n]
print(f"{len(cand)} contested patches selected from a pool of {a.pool} (disagreement SSIM <= {cand[-1]['dis']:.3f})")
SYSTEM = ("You are a senior reservoir geologist reviewing AI-generated subsurface maps. You are given an INPUT map and two candidate AI predictions (A and B) of a different property "
          "for the same layer. You do NOT have the true answer. Choose the candidate that is more geologically plausible and more consistent with the input. Black or white areas are background.")
PROP = {"PorFac": "porosity (rainbow: blue 0.04 ... red 0.14 v/v)", "PermFac": "permeability (log rainbow: purple/blue 1 mD ... green 100 mD ... red 1000 mD)"}; FACD = "facies (yellow sand, orange shaly sand, gray shale)"
ind, outd = (PROP[a.task], FACD) if a.direction == "AB" else (FACD, PROP[a.task])
PROMPT = (f"Panels left to right: INPUT ({ind}); candidate A and candidate B (both {outd}).\nWhich candidate is the better prediction? "
          'Return ONLY JSON: {"better": "A" or "B", "confidence": <0-1>, "reason": "<max 20 words>"}')
def b64(panels):
    arr = [((t.clamp(-1, 1) + 1) / 2 * 255).permute(1, 2, 0).cpu().numpy().astype("uint8") for t in panels]; h, w = arr[0].shape[:2]; cv = np.full((h, len(arr) * w + 8 * (len(arr) - 1), 3), 255, "uint8")
    for k, im_ in enumerate(arr): cv[:, k * (w + 8):k * (w + 8) + w] = im_
    im = Image.fromarray(cv).resize((cv.shape[1] * 2, cv.shape[0] * 2), Image.NEAREST); buf = io.BytesIO(); im.save(buf, format="PNG"); return base64.b64encode(buf.getvalue()).decode()
done = set()
if os.path.exists(fn):
    for l in open(fn, encoding="utf-8"):
        try: r = json.loads(l); done.add(r["file"]) if "pick" in r else None
        except Exception: pass
rng = random.Random(a.seed); t0 = time.time(); nd = consec = tin = tout = 0
with open(fn, "a", encoding="utf-8") as fo:
    for c in cand:
        it_ = ds[c["i"]]; name = os.path.basename(it_["path"])
        if name in done: continue
        with torch.no_grad():
            x = it_["input"].unsqueeze(0).to(dev); mem = torch.stack([iv(M.G(fw(x))) for fw, iv in T8]); p8 = mem.mean(0)[0]; p1 = mem[0][0]
        a_is_tta = rng.random() < 0.5; A, B = (p8, p1) if a_is_tta else (p1, p8); img = b64([it_["input"], A, B])
        rec = {"file": name, "A_is": "tta8" if a_is_tta else "single", "risk_tta8": c["rt"], "risk_single": c["rs"], "dis": c["dis"]}
        for attempt in range(3):
            try:
                msg = client.messages.create(model=a.model, max_tokens=a.max_tokens, system=SYSTEM, messages=[{"role": "user", "content": [{"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": img}}, {"type": "text", "text": PROMPT}]}])
                text = msg.content[0].text.strip(); tin += msg.usage.input_tokens; tout += msg.usage.output_tokens
                m = re.search(r'"better"\s*:\s*"([AB])"', text)
                if not m: raise ValueError("no choice in response: " + text[:80])
                mc = re.search(r'"confidence"\s*:\s*([0-9.]+)', text); better = m.group(1)
                rec.update(better=better, pick=("tta8" if (better == "A") == a_is_tta else "single"), confidence=float(mc.group(1)) if mc else None, raw=text[:400]); rec.pop("error", None); break
            except Exception as e:
                rec["error"] = str(e)[:200]
                if "invalid_request_error" in str(e) or "authentication" in str(e).lower(): break
                time.sleep(2 ** attempt)
        fo.write(json.dumps(rec) + "\n"); fo.flush(); nd += 1; consec = consec + 1 if "pick" not in rec else 0
        if nd % 10 == 0 or nd == 1: print(f"{nd} done ({time.time()-t0:.0f}s) tokens in {tin}, out {tout}" + (f" | last error: {rec['error']}" if "error" in rec else ""), flush=True)
        if consec >= a.max_consecutive_errors: sys.exit(f"Stopping: {consec} consecutive errors. Last: {rec.get('error')}")
print(f"Finished {nd} new calls. Tokens: input {tin}, output {tout}."); analyse()
