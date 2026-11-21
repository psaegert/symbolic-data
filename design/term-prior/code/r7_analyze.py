import sys, json, collections, warnings
warnings.filterwarnings("ignore")
import numpy as np
sys.argv = ["x", "none", "0"]
exec(open("r7_sim.py").read().split("mode = sys.argv[1]")[0])
from classify import classify
vset = p.vset
def cls(t):
    c = classify(t, ar, vset); return c
T8d = [json.loads(l)["sk"].split() for l in open("../prior/draws.jsonl")][:4000]       # T8 as delivered (simplified)
T8r = json.load(open("t8_raw.json"))                                                    # traced T8 draws (raw + simplified)
bank = json.load(open("samples.json")); R6 = [r for r in bank if r["prior"] == "pool4"]
V = {"T8 prior": {"simp": T8d, "raw": [r["raw"] for r in T8r], "rows": None},
     "round 6, n1<=4": {"simp": [r["simp"] for r in R6], "raw": [r["raw"] for r in R6], "rows": None}}
for tag, label in [("n_1.0", "r7 B x1.0"), ("n_1.3", "r7 B x1.3"), ("n_1.6", "r7 B x1.6"), ("n_2.0", "r7 B x2.0"),
                   ("o_1.6", "r7 B x1.6, old nesting"), ("nc_1.6", "r7 B x1.6, corpus fn weights")]:
    rows = json.load(open(f"r7_{tag}.json"))
    V[label] = {"simp": [r["simp"] for r in rows], "raw": [r["raw"] for r in rows], "rows": rows}
t8ops = [ops(t) for t in T8d]
def row(label, f): print(f"| {label} | " + " | ".join(f(v) for v in V.values()) + " |")
print("| | " + " | ".join(V) + " |"); print("|---" * (len(V) + 1) + "|")
row("median / p90 operators", lambda v: f"{np.median([ops(t) for t in v['simp']]):.0f} / {np.percentile([ops(t) for t in v['simp']], 90):.0f}")
row("median / p90 tokens", lambda v: f"{np.median([len(t) for t in v['simp']]):.0f} / {np.percentile([len(t) for t in v['simp']], 90):.0f}")
row("KS distance of operator count to T8", lambda v: f"{ks([ops(t) for t in v['simp']], t8ops):.2f}")
row("> 35 / > 40 tokens (%)", lambda v: f"{100*np.mean([len(t)>35 for t in v['simp']]):.1f} / {100*np.mean([len(t)>40 for t in v['simp']]):.1f}")
row("trivial <= 5 tokens (%)", lambda v: f"{100*np.mean([len(t)<=5 for t in v['simp']]):.1f}")
C = {k: [cls(t) for t in v["simp"]] for k, v in V.items()}
def crow(label, f): print(f"| {label} | " + " | ".join(f(C[k]) for k in V) + " |")
crow("median variables / univariate (%)", lambda c: f"{np.median([x['nv'] for x in c]):.0f} / {100*np.mean([x['nv']==1 for x in c]):.1f}")
crow(">= 2 / >= 3 additive terms (%)", lambda c: f"{100*np.mean([x['k_terms']>=2 for x in c]):.0f} / {100*np.mean([x['k_terms']>=3 for x in c]):.0f}")
crow("polynomial >= 3 terms (%)", lambda c: f"{100*np.mean([x['poly_ge3'] for x in c]):.2f}")
crow("univariate polynomial >= 3 terms (%)", lambda c: f"{100*np.mean([x['poly_1v_ge3'] for x in c]):.2f}")
crow("sum of >= 2 rational terms (%)", lambda c: f"{100*np.mean([x['sum_rat_ge2'] for x in c]):.2f}")
crow(">= 3 terms of one form (%)", lambda c: f"{100*np.mean([x['superpos_ge3'] for x in c]):.2f}")
crow("contains a transcendental (%)", lambda c: f"{100*np.mean([x['transcendental'] for x in c]):.0f}")
def pshare(v, kind):
    ps = [pure(t) for t in v["simp"]]; ps = [x for x in ps if x is not None]
    return 100 * np.mean([x == kind for x in ps])
row("pure sum / pure product, of draws with >= 2 binary ops (%)", lambda v: f"{pshare(v, 'sum'):.0f} / {pshare(v, 'prod'):.0f}")
row("levels used: 2 / 3 (%)", lambda v: "-" if v["rows"] is None else f"{100*np.mean([p.levels_used(r) >= 1 for r in v['rows'] if 'pool' in r]):.0f} / {100*np.mean([p.levels_used(r) >= 2 for r in v['rows'] if 'pool' in r]):.0f}")
row("uses per composite / all used once (%)", lambda v: "-" if v["rows"] is None else (lambda u: f"{u[0]:.2f} / {100*u[1]:.0f}")(reuse(v["rows"])))
print("\nOperator frequencies BEFORE simplification (share of operator tokens, %):")
keys = ["+", "-", "*", "/", "pow", "rootn", "function", "neg", "inv", "abs"]
print("| | " + " | ".join(keys) + " |"); print("|---" * (len(keys) + 1) + "|")
for k, v in V.items():
    f = opfreq([{"raw": t} for t in v["raw"]], "raw")
    print(f"| {k} | " + " | ".join(f"{100*f[x]:.1f}" for x in keys) + " |")

FSET = {"sin", "cos", "tan", "asin", "acos", "atan", "sinh", "cosh", "tanh", "asinh", "acosh", "atanh", "exp", "log", "abs"}
INV = {"asin", "acos", "atan", "asinh", "acosh", "atanh"}
def max_fn_depth(t):
    pos = 0; best = 0
    def walk(c):
        nonlocal pos, best
        x = t[pos]; pos += 1
        cc = c + (1 if x in FSET else 0); best = max(best, cc)
        for _ in range(ar.get(x, 0)): walk(cc)
    walk(0); return best
print("\nNesting and function choice (simplified expressions):")
print("| | " + " | ".join(V) + " |"); print("|---" * (len(V) + 1) + "|")
row("function inside a function, anywhere (%)", lambda v: f"{100*np.mean([max_fn_depth(t) >= 2 for t in v['simp']]):.1f}")
row("3 nested functions (%)", lambda v: f"{100*np.mean([max_fn_depth(t) >= 3 for t in v['simp']]):.1f}")
row("function nodes per expression", lambda v: f"{np.mean([sum(1 for x in t if x in FSET) for t in v['simp']]):.2f}")
row("inverse trig/hyperbolic, share of function nodes (%)", lambda v: (lambda a, b: f"{100*a/max(b,1):.0f}")(sum(1 for t in v['simp'] for x in t if x in INV), sum(1 for t in v['simp'] for x in t if x in FSET)))
