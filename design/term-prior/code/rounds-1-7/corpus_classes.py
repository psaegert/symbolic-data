import json, sys, collections, signal
import sympy as sp
C = "/home/psaegert/Projects/research/archive/symbolic-data/2026-11-03-family-coherence-corpus/t83/corpus/"
rows = [json.loads(l) for f in ("corpus_ops_wikidata.jsonl", "corpus_ops_wikipedia.jsonl") for l in open(C + f)]
TR = (sp.sin, sp.cos, sp.tan, sp.exp, sp.log, sp.asin, sp.acos, sp.atan, sp.sinh, sp.cosh, sp.tanh)
def shape(e):
    if e.is_Symbol: return "S"
    if e.is_Number: return "N"
    return "(" + type(e).__name__ + " " + " ".join(sorted(shape(a) for a in e.args)) + ")"
def motif(t):
    c, rest = t.as_coeff_Mul()
    return shape(rest)
class TO(Exception): pass
def h(*a): raise TO()
signal.signal(signal.SIGALRM, h)
cnt = collections.Counter(); n = 0
for r in rows:
    try:
        signal.alarm(2)
        import re
        src = re.sub(r"[{}\\]", "_", r["sympy"]).replace("'", "p")
        e = sp.sympify(src, locals={n: sp.Symbol(n) for n in set(re.findall(r"[A-Za-z_][A-Za-z_0-9]*", src)) if n not in ("sqrt","exp","log","sin","cos","tan","asin","acos","atan","sinh","cosh","tanh","Abs","pi","E","I")})
        syms = sorted(e.free_symbols, key=str)
        if not syms: signal.alarm(0); continue
        T = [t for t in sp.Add.make_args(e) if t.free_symbols]
        k = len(T)
        poly = e.is_polynomial(*syms)
        rat = e.is_rational_function(*syms) and not poly
        mc = collections.Counter(motif(t) for t in T).most_common(1)[0][1]
        prat = sum(1 for t in T if t.is_rational_function(*syms) and not t.is_polynomial(*syms))
        d = dict(poly=poly, rational=rat, poly_ge3=poly and k >= 3, poly_1v_ge3=poly and len(syms) == 1 and k >= 3,
                 sum_rat_ge2=bool(e.is_rational_function(*syms)) and prat >= 2,
                 repeat_motif_ge2=k >= 2 and mc >= 2, superpos_ge3=k >= 3 and mc == k,
                 k_ge2=k >= 2, k_ge3=k >= 3, transcendental=e.has(*TR))
        signal.alarm(0)
        n += 1
        for key, v in d.items(): cnt[key] += bool(v)
    except Exception:
        signal.alarm(0)
print(f"corpus formulas classified: {n} of {len(rows)}")
for key in ["poly","rational","poly_ge3","poly_1v_ge3","sum_rat_ge2","repeat_motif_ge2","superpos_ge3","k_ge2","k_ge3","transcendental"]:
    print(f"  {key:18s} {100*cnt[key]/n:6.2f}%")
