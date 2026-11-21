"""The 11,668 real formulas (family-coherence corpus) as simplified prefix skeletons, the way the priors' draws are:
symbols renamed x1..xk in sorted order (k <= 17), pi and e as constants, simplified with the catalog's permissive canon."""
import json, re, signal, gzip, warnings, sys
warnings.filterwarnings("ignore")
import sympy as sp
import pool7 as p
from symbolic_data.token_ops import desugar_sqrt
C = "/home/psaegert/Projects/research/archive/symbolic-data/2026-11-03-family-coherence-corpus/t83/corpus/"
rows = [json.loads(l) for f in ("corpus_ops_wikidata.jsonl", "corpus_ops_wikipedia.jsonl") for l in open(C + f)]
FUN = ("sqrt", "exp", "log", "sin", "cos", "tan", "asin", "acos", "atan", "sinh", "cosh", "tanh", "asinh", "acosh", "atanh", "Abs", "pi", "E", "I")
class TO(Exception): pass
def alarm(*a): raise TO()
signal.signal(signal.SIGALRM, alarm)
out, why = [], {}
for r in rows:
    try:
        signal.alarm(3)
        src = re.sub(r"[{}\\]", "_", r["sympy"]).replace("'", "p")
        e = sp.sympify(src, locals={n: sp.Symbol(n) for n in set(re.findall(r"[A-Za-z_][A-Za-z_0-9]*", src)) if n not in FUN})
        if e.has(sp.I) or not e.free_symbols: raise ValueError("complex or constant")
        syms = sorted(e.free_symbols, key=str)
        if len(syms) > 17: raise ValueError("more than 17 symbols")
        e = e.subs({s: sp.Symbol(f"x{i + 1}") for i, s in enumerate(syms)}).subs({sp.pi: sp.Float(3.14159265), sp.E: sp.Float(2.71828183)})
        infix = str(e).replace("Abs(", "abs(")
        toks = ["pow" if t in ("**", "^") else t for t in p.eng.infix_to_prefix(infix)]
        toks = desugar_sqrt(toks, p.eng.operator_arity_compat)
        s, w = p.simplify_checked(toks)
        signal.alarm(0)
        if s is None: raise ValueError(w)
        out.append({"label": r.get("label"), "simp": s})
    except Exception as ex:
        signal.alarm(0); k = type(ex).__name__ if not isinstance(ex, ValueError) else str(ex); why[k] = why.get(k, 0) + 1
json.dump(out, gzip.open("corpus_prefix.json.gz", "wt"))
print("converted", len(out), "of", len(rows), "| skipped:", why)
