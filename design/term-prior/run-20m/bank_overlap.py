"""Overlap of 'polynomial with >= 3 terms' and '>= 3 terms of one form' in the 3,000-draw banks and the corpus, and how
many of the overlap are weighted sums of single variables. Run from flash-ansr-rl/runs/prior_structure/explorer."""
import gzip, json, sys
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
    T = [t for t in terms(parse(s, ar)) if has_var(t, VS)]
    return bool(T) and all(lin(t) for t in T)
D = {}
for f in ("rows_base.json.gz", "rows_ladder.json.gz", "rows_r10c.json.gz"):
    D.update({k: [(r["simp"], r["cls"]["poly3"], r["cls"]["sup3"]) for r in v] for k, v in json.load(gzip.open(f)).items()})
C = []
for c in json.load(gzip.open("corpus_prefix.json.gz", "rt")):
    try: q = classify(c["simp"], ar, VS); C.append((c["simp"], q["poly_ge3"], q["superpos_ge3"]))
    except Exception: pass
D["corpus"] = C
print("| | polynomial >= 3 | one form >= 3 | both | both and linear |")
for k in ("t8", "r8", "r10c", "corpus"):
    R = D[k]; both = [s for s, p, o in R if p and o]
    print(f"| {k} | {sum(p for _, p, _ in R)} | {sum(o for _, _, o in R)} | {len(both)} | {sum(linear(s) for s in both)} |")
