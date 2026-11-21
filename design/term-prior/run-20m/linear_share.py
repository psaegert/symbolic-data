import gzip, json, sys, numpy as np
sys.path.insert(0, "/home/psaegert/Projects/flash-ansr-rl/runs/prior_structure")
from classify import classify, parse, terms, has_var
from simplipy import SimpliPyEngine
eng = SimpliPyEngine.load("acj-5-4-llm"); ar = dict(eng.operator_arity); VS = {f"x{i}" for i in range(1, 18)}
def lin(t):
    if isinstance(t, str): return t in VS
    if t[0] in ("*", "/"): return sum(has_var(c, VS) for c in t[1]) == 1 and all(isinstance(c, str) or not has_var(c, VS) or lin(c) for c in t[1]) and not (t[0] == "/" and has_var(t[1][1], VS))
    if t[0] == "neg": return lin(t[1][0])
    return False
def linear(s):
    try: tr = parse(s, ar)
    except Exception: return False
    T = [t for t in terms(tr) if has_var(t, VS)]
    return bool(T) and all(lin(t) for t in T)
D = {}
for f in ("rows_base.json.gz", "rows_ladder.json.gz", "rows_r10c.json.gz"):
    D.update(json.load(gzip.open(f)))
corpus = json.load(gzip.open("corpus_prefix.json.gz", "rt"))
D["corpus"] = []
for c in corpus:
    try: D["corpus"].append({"simp": c["simp"], "cls": {k: v for k, v in zip(("poly3", "sup3"), (lambda q: (q["poly_ge3"], q["superpos_ge3"]))(classify(c["simp"], ar, VS)))}})
    except Exception: pass
print("| | draws | poly >= 3 terms | of which linear sums | poly >= 3, non-linear | >= 3 terms of one form | of which linear | one form, non-linear |")
for k in ("t8", "r8", "r10c", "corpus"):
    R = D[k]; n = len(R)
    p3 = [r for r in R if r["cls"]["poly3"]]; s3 = [r for r in R if r["cls"]["sup3"]]
    lp = sum(linear(r["simp"]) for r in p3); ls = sum(linear(r["simp"]) for r in s3)
    print(f"| {k} | {n} | {100*len(p3)/n:.2f}% | {lp} of {len(p3)} | {100*(len(p3)-lp)/n:.2f}% | {100*len(s3)/n:.2f}% | {ls} of {len(s3)} | {100*(len(s3)-ls)/n:.2f}% |")
