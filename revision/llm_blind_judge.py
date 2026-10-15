"""
llm_blind_judge.py -- BLIND vision-LLM quality judge (no ground truth, no SSIM shown) on a random TEST subsample.
The model sees only: INPUT map | GeoAgent PREDICTION (TTA x8), plus a legend. It returns a 0-10 quality score, a failure mode and a
short rationale. Every raw response is stored in llm\{task}_{direction}.jsonl (auditable trace). Failed calls are recorded as errors,
NEVER replaced by a rule-based guess. Safe to restart: finished patches are skipped.
Setup (Anaconda Prompt, geoldm_gpu):   set ANTHROPIC_API_KEY=sk-ant-...      (the key is never printed or written to disk)
Smoke test (20 patches, prints token usage):  python llm_blind_judge.py --task PorFac --direction AB --n 20
Full run:                                      python llm_blind_judge.py --task PorFac --direction AB --n 300
"""
import os, sys, io, json, time, base64, argparse, random
import numpy as np, torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Subset
p = argparse.ArgumentParser()
p.add_argument("--code_dir", default=r"D:\First year\Papers\GeoAgent"); p.add_argument("--data_base", default=r"E:\2D Dutch's Dataset\patches_512")
p.add_argument("--ckpt_base", default=r"D:\First year\Papers\GeoAgent\checkpoints"); p.add_argument("--rev_dir", default=r"D:\First year\Papers\GeoAgent\revision_ACAGS")
p.add_argument("--task", required=True, choices=["PorFac", "PermFac"]); p.add_argument("--direction", required=True, choices=["AB", "BA"])
p.add_argument("--n", type=int, default=300); p.add_argument("--seed", type=int, default=0); p.add_argument("--epoch", default="0100")
p.add_argument("--model", default="claude-sonnet-5"); p.add_argument("--max_tokens", type=int, default=700); p.add_argument("--max_consecutive_errors", type=int, default=5)
a = p.parse_args()
key = os.environ.get("ANTHROPIC_API_KEY", "")
if not key: sys.exit("Set ANTHROPIC_API_KEY first (set ANTHROPIC_API_KEY=... in this prompt).")
import anthropic
client = anthropic.Anthropic(api_key=key)
sys.path.insert(0, a.code_dir); sys.path.insert(0, a.rev_dir)
from nb_model import Pix2PixModel
from geoagent_dataset import PairedGeoDataset
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
M = Pix2PixModel(device=dev); M.load(os.path.join(a.ckpt_base, f"{a.task}_{a.direction}", f"pix2pix_{a.direction}_ep{a.epoch}.pth")); M.G.eval()
ds = PairedGeoDataset(os.path.join(a.data_base, a.task, "test"), direction=a.direction, image_size=256, augment=False)
random.seed(a.seed); idx = sorted(random.sample(range(len(ds)), a.n))
def d4(k, f):
    fw = lambda x: torch.rot90(torch.flip(x, [-1]) if f else x, k, [-2, -1])
    iv = lambda y: (torch.flip(torch.rot90(y, -k, [-2, -1]), [-1]) if f else torch.rot90(y, -k, [-2, -1]))
    return fw, iv
T8 = [d4(k, f) for f in (False, True) for k in range(4)]
PROP = {"PorFac": "porosity (rainbow scale: blue = 0.04 v/v ... red = 0.14 v/v)", "PermFac": "permeability (log scale rainbow: purple/blue = 1 mD ... green = 100 mD ... red = 1000 mD)"}
FAC = "facies (yellow = sand, orange = shaly sand, gray = shale)"
inp_desc, out_desc = (PROP[a.task], FAC) if a.direction == "AB" else (FAC, PROP[a.task])
SYSTEM = ("You are a senior reservoir geologist reviewing AI-generated subsurface maps. You are given a map of one reservoir property (left) and "
          "an AI prediction of a different property for the same layer (right). You do NOT have the true answer; judge only whether the prediction "
          "is geologically plausible and consistent with the input. Black or white areas outside the field are background.")
PROMPT = (f"Left panel: INPUT, {inp_desc}.\nRight panel: PREDICTED, {out_desc}.\n"
          "Judge the prediction's reliability: is it spatially coherent, are bodies and boundaries plausible and consistent with the input pattern, "
          "is there speckle, blur, missing or spurious bodies, edge artifacts or colour errors?\n"
          'Return ONLY JSON: {"quality_score": <number 0-10, 10 = fully reliable>, "failure_mode": "<none|speckle|boundary_blur|missing_bodies|spurious_bodies|color_error|edge_artifact|mixed>", "rationale": "<max 25 words>"}')
def png_b64(left, right):
    to = lambda t: ((t.clamp(-1, 1) + 1) / 2 * 255).permute(1, 2, 0).cpu().numpy().astype("uint8")
    l, r = to(left), to(right); h, w = l.shape[:2]; canvas = np.full((h, 2 * w + 8, 3), 255, "uint8"); canvas[:, :w] = l; canvas[:, w + 8:] = r
    im = Image.fromarray(canvas).resize((canvas.shape[1] * 2, canvas.shape[0] * 2), Image.NEAREST); buf = io.BytesIO(); im.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()
od = os.path.join(a.rev_dir, "llm"); os.makedirs(od, exist_ok=True); fn = os.path.join(od, f"{a.task}_{a.direction}.jsonl")
done = set()
if os.path.exists(fn):
    for line in open(fn, encoding="utf-8"):
        try: r = json.loads(line); done.add(r["file"]) if "error" not in r else None
        except Exception: pass
tin = tout = 0; t0 = time.time(); nd = 0; consec = 0
with open(fn, "a", encoding="utf-8") as fo:
    for i in idx:
        item = ds[i]; name = os.path.basename(item["path"])
        if name in done: continue
        with torch.no_grad():
            x = item["input"].unsqueeze(0).to(dev); pred = torch.stack([iv(M.G(fw(x))) for fw, iv in T8]).mean(0)[0]
        b64 = png_b64(item["input"], pred); rec = {"file": name, "model": a.model}
        for attempt in range(4):
            try:
                msg = client.messages.create(model=a.model, max_tokens=a.max_tokens, system=SYSTEM,
                      messages=[{"role": "user", "content": [{"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64}}, {"type": "text", "text": PROMPT}]}])
                text = msg.content[0].text.strip(); tin += msg.usage.input_tokens; tout += msg.usage.output_tokens
                rec.update(raw=text[:1500], in_tokens=msg.usage.input_tokens, out_tokens=msg.usage.output_tokens, stop=msg.stop_reason)
                try: js = json.loads(text[text.find("{"):text.rfind("}") + 1])
                except Exception:                                       # truncated / slightly malformed JSON: recover the fields by pattern
                    import re
                    mq = re.search(r'"quality_score"\s*:\s*([0-9.]+)', text); mf = re.search(r'"failure_mode"\s*:\s*"([a-z_]+)"', text); mr = re.search(r'"rationale"\s*:\s*"(.*)', text, re.S)
                    if not mq: raise ValueError("no quality_score in response")
                    js = {"quality_score": mq.group(1), "failure_mode": mf.group(1) if mf else "", "rationale": (mr.group(1) if mr else "")[:300], "recovered": True}
                rec.update(quality_score=float(js["quality_score"]), failure_mode=js.get("failure_mode", ""), rationale=js.get("rationale", ""), recovered=bool(js.get("recovered", False))); break
            except Exception as e:
                rec = {"file": name, "model": a.model, "error": str(e)[:200]}
                if "invalid_request_error" in str(e) or "authentication" in str(e).lower(): break      # retrying cannot help
                time.sleep(2 ** attempt)
        fo.write(json.dumps(rec) + "\n"); fo.flush(); nd += 1
        consec = consec + 1 if "error" in rec else 0
        if consec >= a.max_consecutive_errors: sys.exit(f"Stopping: {consec} consecutive errors. Last: {rec.get('error')}  (no further calls made)")
        if nd % 10 == 0 or nd == 1: print(f"{nd} done ({time.time()-t0:.0f}s) | tokens in {tin}, out {tout}" + (f" | last error: {rec['error']}" if 'error' in rec else ""), flush=True)
print(f"\nFinished {nd} new calls. Tokens this run: input {tin}, output {tout}. Output: {fn}")
