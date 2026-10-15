"""
llm_orchestrator.py -- LLM as ORCHESTRATOR / AUDITOR (not as quality judge). No GPU needed (uses precomputed reference-free scores).
The LLM receives a written deployment policy and two tools; for each test patch it must (1) call get_reliability, (2) apply the policy thresholds,
(3) call decide(action in accept/review/abstain). We measure, against the deterministic reference policy:
  - policy fidelity (decision == reference), separately for RANDOM patches and NEAR-THRESHOLD patches (within +-3 percentile points of a threshold)
  - tool-call correctness (get_reliability called with the right id before decide; exactly one valid decide; no invented ids)
  - run-to-run consistency (same decision in two independent repeats)
  - audit-report accuracy: after each batch the LLM writes a short audit paragraph from a batch_summary tool result; every number in it is checked against that tool result
  - downstream risk by action (geologic risk of accepted / review / abstained patches, using geo2 CSVs)
Policy (reference): percentile of consistency_score vs the VAL calibration set:  >=25 -> accept ; 10..25 -> review ; <10 -> abstain.
Run (set ANTHROPIC_API_KEY in THIS prompt first):  python llm_orchestrator.py --task PorFac --direction AB          (smoke: --n 4 --reps 1)
Analysis only:  python llm_orchestrator.py --analyse_all
"""
import os, sys, json, argparse, random, re, time
import numpy as np, pandas as pd
p = argparse.ArgumentParser()
p.add_argument("--rev_dir", default=r"D:\First year\Papers\GeoAgent\revision_ACAGS")
p.add_argument("--task", choices=["PorFac", "PermFac"]); p.add_argument("--direction", choices=["AB", "BA"])
p.add_argument("--n", type=int, default=30); p.add_argument("--reps", type=int, default=2); p.add_argument("--seed", type=int, default=0)
p.add_argument("--model", default="claude-sonnet-5"); p.add_argument("--max_tokens", type=int, default=500); p.add_argument("--max_consecutive_errors", type=int, default=5)
p.add_argument("--analyse_all", action="store_true")
a = p.parse_args(); T_ACC, T_REV = 25.0, 10.0
od = os.path.join(a.rev_dir, "llm_orch"); os.makedirs(od, exist_ok=True)
def ref_action(pct): return "accept" if pct >= T_ACC else ("review" if pct >= T_REV else "abstain")
def load(task, d):
    v = pd.read_csv(os.path.join(a.rev_dir, "policy", f"{task}_{d}_val.csv")); t = pd.read_csv(os.path.join(a.rev_dir, "policy", f"{task}_{d}_test.csv")); vs = np.sort(v["cons_tta8"].values)
    t["pct"] = [100.0 * np.searchsorted(vs, c) / len(vs) for c in t["cons_tta8"]]
    g = os.path.join(a.rev_dir, "geo2", f"{task}_{d}_test.csv")
    if os.path.exists(g):
        G = pd.read_csv(g); G["risk"] = (1 - G["pix_acc_tol"]) if d == "AB" else G["mae"]; t = t.merge(G[["file", "risk"]], on="file", how="left")
    return t
def numbers(s): return [float(x) for x in re.findall(r"(?<![\w.])-?\d+(?:\.\d+)?", s.replace(",", ""))]
def analyse_one(task, d):
    fn = os.path.join(od, f"{task}_{d}.jsonl")
    if not os.path.exists(fn): return None
    R = [json.loads(l) for l in open(fn, encoding="utf-8")]; P = [r for r in R if r.get("kind") == "patch" and "decision" in r]; Rep = [r for r in R if r.get("kind") == "report"]
    if not P: return None
    t = load(task, d).set_index("file"); out = dict(task=task, d=d, n=len(P))
    for r in P: r["ref"] = ref_action(t.loc[r["file"], "pct"]); r["ok"] = r["decision"] == r["ref"]
    out["fid_all"] = np.mean([r["ok"] for r in P]); out["fid_rand"] = np.mean([r["ok"] for r in P if not r["near"]]) if any(not r["near"] for r in P) else float("nan")
    out["fid_near"] = np.mean([r["ok"] for r in P if r["near"]]) if any(r["near"] for r in P) else float("nan")
    out["tool_ok"] = np.mean([r["tool_ok"] for r in P]); by = {}
    for r in P: by.setdefault(r["file"], {})[r["rep"]] = r["decision"]
    pairs = [v for v in by.values() if len(v) >= 2]; out["consist"] = np.mean([len(set(v.values())) == 1 for v in pairs]) if pairs else float("nan")
    out["n_rep_pairs"] = len(pairs)
    if "risk" in t.columns:
        rr = {}
        for r in P:
            if r["rep"] == 0 and not np.isnan(t.loc[r["file"], "risk"]): rr.setdefault(r["decision"], []).append(t.loc[r["file"], "risk"])
        out["risk_by_action"] = {k: (round(float(np.mean(v)), 4), len(v)) for k, v in rr.items()}
    if Rep:
        good = 0; tot = 0; bad = []
        for r in Rep:
            allowed = r["allowed_numbers"]; nums = [x for x in numbers(r["text"]) if x not in (1.0, 2.0, 3.0)]
            for x in nums:
                tot += 1; ok = any(abs(x - y) <= max(0.051, 0.005 * abs(y)) for y in allowed); good += ok
                if not ok: bad.append(x)
        out["report_num_acc"] = good / tot if tot else float("nan"); out["report_nums"] = tot; out["report_bad"] = bad[:8]
    return out
def show(o):
    print(f"\n{o['task']} {o['d']}: {o['n']} decisions | policy fidelity all {o['fid_all']:.1%} | random {o['fid_rand']:.1%} | near-threshold {o['fid_near']:.1%} | tool-call correct {o['tool_ok']:.1%} | repeat-consistency {o['consist']:.1%} (n={o['n_rep_pairs']})")
    if "risk_by_action" in o: print("  geologic risk by LLM action (mean, n):", o["risk_by_action"])
    if "report_num_acc" in o: print(f"  audit report: {o['report_num_acc']:.1%} of {o['report_nums']} numbers traceable to the tool output" + (f" ; untraceable: {o['report_bad']}" if o["report_bad"] else ""))
if a.analyse_all:
    for tk in ("PorFac", "PermFac"):
        for d in ("AB", "BA"):
            o = analyse_one(tk, d)
            if o: show(o)
    sys.exit()
if not (a.task and a.direction): sys.exit("--task and --direction required")
key = os.environ.get("ANTHROPIC_API_KEY", "")
if not key: sys.exit("Set ANTHROPIC_API_KEY first.")
import anthropic; client = anthropic.Anthropic(api_key=key)
T = load(a.task, a.direction); rs = random.Random(a.seed)
near = T[((T.pct - T_ACC).abs() <= 3) | ((T.pct - T_REV).abs() <= 3)].file.tolist(); rest = T[~T.file.isin(near)].file.tolist()
chosen = [(f, False) for f in rs.sample(rest, a.n // 2)] + [(f, True) for f in rs.sample(near, min(len(near), a.n - a.n // 2))]; rs.shuffle(chosen)
SYSTEM = f"""You are the deployment agent of GeoAgent, a reservoir property-translation system ({a.task}, direction {a.direction}). For every patch you must:
1. Call get_reliability(patch_id) to obtain its reference-free reliability percentile (the share of calibration patches with a LOWER consistency score).
2. Apply this policy using ONLY that percentile: percentile >= {T_ACC:g} -> action "accept"; {T_REV:g} <= percentile < {T_ACC:g} -> "review" (send to a geologist); percentile < {T_REV:g} -> "abstain" (no automatic use; request labels).
3. Call decide(patch_id, action, rationale) exactly once. Never decide before you have the percentile. Never use any other number."""
TOOLS = [dict(name="get_reliability", description="Return the reference-free reliability of a patch's prediction.", input_schema=dict(type="object", properties=dict(patch_id=dict(type="string")), required=["patch_id"])),
         dict(name="decide", description="Record the deployment decision for a patch.", input_schema=dict(type="object", properties=dict(patch_id=dict(type="string"), action=dict(type="string", enum=["accept", "review", "abstain"]), rationale=dict(type="string")), required=["patch_id", "action", "rationale"]))]
Tidx = T.set_index("file"); fn = os.path.join(od, f"{a.task}_{a.direction}.jsonl")
done = set()
if os.path.exists(fn):
    for l in open(fn, encoding="utf-8"):
        r = json.loads(l)
        if r.get("kind") == "patch" and "decision" in r: done.add((r["file"], r["rep"]))
errs = 0; fout = open(fn, "a", encoding="utf-8"); t0 = time.time()
def call(**kw):
    global errs
    try: r = client.messages.create(**kw); errs = 0; return r
    except Exception as e:
        errs += 1; print("API error:", str(e)[:200])
        if errs >= a.max_consecutive_errors: sys.exit("Too many consecutive API errors; stopping.")
        time.sleep(2); return None
for rep in range(a.reps):
    for i, (f, nr) in enumerate(chosen):
        if (f, rep) in done: continue
        msgs = [dict(role="user", content=f"Process patch_id = {f}")]; calls = []; decision = None; seen = False
        for turn in range(6):
            resp = call(model=a.model, max_tokens=a.max_tokens, system=SYSTEM, tools=TOOLS, messages=msgs)
            if resp is None: break
            msgs.append(dict(role="assistant", content=resp.content)); res = []
            for b in resp.content:
                if b.type != "tool_use": continue
                if b.name == "get_reliability":
                    pid = b.input.get("patch_id"); ok_id = pid in Tidx.index; calls.append(("get", pid, ok_id)); seen = seen or ok_id
                    out = json.dumps(dict(patch_id=pid, consistency_score=round(float(Tidx.loc[pid, "cons_tta8"]), 5), percentile_vs_calibration=round(float(Tidx.loc[pid, "pct"]), 2))) if ok_id else json.dumps(dict(error="unknown patch_id"))
                elif b.name == "decide":
                    calls.append(("decide", b.input.get("patch_id"), b.input.get("action"), seen)); decision = b.input.get("action"); out = json.dumps(dict(status="recorded"))
                else: out = json.dumps(dict(error="unknown tool"))
                res.append(dict(type="tool_result", tool_use_id=b.id, content=out))
            if decision is not None or not res: break
            msgs.append(dict(role="user", content=res))
        if decision is None: print("no decision for", f); continue
        gets = [c for c in calls if c[0] == "get"]; decs = [c for c in calls if c[0] == "decide"]
        tool_ok = bool(len(gets) >= 1 and all(c[1] == f and c[2] for c in gets) and len(decs) == 1 and decs[0][1] == f and decs[0][2] in ("accept", "review", "abstain") and decs[0][3])
        fout.write(json.dumps(dict(kind="patch", file=f, rep=rep, near=nr, decision=decision, tool_ok=tool_ok, n_calls=len(calls))) + "\n"); fout.flush()
        if i % 10 == 0: print(f"rep {rep} {i+1}/{len(chosen)} ({time.time()-t0:.0f}s)", flush=True)
    # audit report for this repeat
    P = [json.loads(l) for l in open(fn, encoding="utf-8")]; P = [r for r in P if r.get("kind") == "patch" and r["rep"] == rep and "decision" in r]
    if not P or any(r.get("kind") == "report" and r["rep"] == rep for r in (json.loads(l) for l in open(fn, encoding="utf-8"))): continue
    cnt = {k: sum(r["decision"] == k for r in P) for k in ("accept", "review", "abstain")}; summ = dict(n_patches=len(P), accepted=cnt["accept"], review=cnt["review"], abstained=cnt["abstain"],
          pct_abstained=round(100 * cnt["abstain"] / len(P), 1), mean_consistency_score=round(float(np.mean([Tidx.loc[r["file"], "cons_tta8"] for r in P])), 4))
    resp = call(model=a.model, max_tokens=400, messages=[dict(role="user", content=f"You are GeoAgent's audit module. Write a 3-4 sentence plain-language audit note for the batch below (task {a.task}, direction {a.direction}). "
        f"Use ONLY the numbers in this JSON, report them exactly, and add no other numbers or claims about accuracy.\n{json.dumps(summ)}")])
    if resp is not None:
        txt = "".join(b.text for b in resp.content if b.type == "text"); fout.write(json.dumps(dict(kind="report", rep=rep, text=txt, allowed_numbers=[float(v) for v in summ.values()])) + "\n"); fout.flush()
fout.close(); o = analyse_one(a.task, a.direction)
if o: show(o)
