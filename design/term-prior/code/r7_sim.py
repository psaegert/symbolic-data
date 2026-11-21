import sys, json, collections, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pool7 as p
ar = p.ar
def ops(t): return sum(1 for x in t if x in ar)
def pure(t):
    b = [x for x in t if x in ("+", "-", "*", "/")]
    if len(b) < 2: return None
    return "sum" if all(x in "+-" for x in b) else "prod" if all(x in "*/" for x in b) else "mixed"
def opfreq(rows, key):
    c = collections.Counter(); tot = 0
    for r in rows:
        for x in r[key]:
            if x in ar:
                g = x if x in ("+", "-", "*", "/", "pow", "rootn", "neg", "inv", "abs") else "function"
                c[g] += 1; tot += 1
    return {k: c[k] / tot for k in ("+", "-", "*", "/", "pow", "rootn", "function", "neg", "inv", "abs")}
def ks(a, b):
    a, b = np.sort(a), np.sort(b); xs = np.union1d(a, b)
    return float(np.max(np.abs(np.searchsorted(a, xs, side="right") / len(a) - np.searchsorted(b, xs, side="right") / len(b))))
def reuse(rows):
    nT = cp = alluse1 = with_T = 0
    for r in rows:
        if r["H"].get("const_law"): continue
        u = [q["uses"] for q in r["pool"] if q["kind"] in ("term", "level")]
        if u: with_T += 1; alluse1 += all(x == 1 for x in u)
        nT += len(u); cp += sum(u)
    return cp / max(nT, 1), alluse1 / max(with_T, 1)
mode = sys.argv[1]; N = int(sys.argv[2])
rng = np.random.default_rng(7)
if mode == "t8":
    rows = [p.draw_t8(rng) for _ in range(N)]
    json.dump(rows, open("t8_raw.json", "w")); print("t8 done"); sys.exit()
variant, scale = mode.split(":"); scale = float(scale)
P = dict(p.P7); P["B_scale"] = scale
if variant == "o": P["persist"] = False; P["cross_level"] = False
rows = [p.draw_r7(rng, P) for _ in range(N)]
json.dump(rows, open(f"r7_{variant}_{scale}.json", "w"))
print("done", mode)
