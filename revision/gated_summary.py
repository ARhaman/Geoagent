import json, glob, os, sys
rev = r"D:\First year\Papers\GeoAgent\revision_ACAGS" if os.name == "nt" else os.path.dirname(os.path.abspath(__file__))
rows = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(rev, "gated", "*.json")))]
print("task dir seed | decision | CHECK hard / all dRisk | TEST (audit) candidate hard / all dRisk  | deployed test dRisk (hard / all)  [risk: lower is better]")
tot = {}
for r in rows:
    ch, ca, th, ta = r["check_hard"], r["check_all"], r["test_hard"], r["test_all"]; acc = r["accept"]
    print(f"{r['task']:7s} {r['direction']} s{r['seed']} | {'ACCEPT  ' if acc else 'ROLLBACK'} | {ch['mean']:+.4f} / {ca['mean']:+.4f} | {th['mean']:+.4f} [{th['lo']:+.4f},{th['hi']:+.4f}] / {ta['mean']:+.4f} [{ta['lo']:+.4f},{ta['hi']:+.4f}] | {(th['mean'] if acc else 0):+.4f} / {(ta['mean'] if acc else 0):+.4f}")
    t = tot.setdefault((r['task'], r['direction']), dict(n=0, acc=0, ungH=0, ungA=0, depH=0, depA=0)); t['n'] += 1; t['acc'] += acc; t['ungH'] += th['mean']; t['ungA'] += ta['mean']; t['depH'] += th['mean'] if acc else 0; t['depA'] += ta['mean'] if acc else 0
print("\nPer direction (mean over seeds): accepted | ungated candidate dRisk hard/all | GATED deployed dRisk hard/all")
for (t_, d_), v in tot.items(): n = v['n']; print(f"{t_:7s} {d_}: {v['acc']}/{n} | {v['ungH']/n:+.4f} / {v['ungA']/n:+.4f} | {v['depH']/n:+.4f} / {v['depA']/n:+.4f}")
