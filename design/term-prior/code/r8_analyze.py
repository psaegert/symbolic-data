"""Round-8 ladder: comparison table, T8-equivalence check of r8a, ECDF plot, 10 unfiltered draws per step."""
import json, gzip, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
D = json.load(gzip.open("rows_ladder.json.gz", "rt")); D.update(json.load(gzip.open("rows_base.json.gz", "rt")))
COLS = [("t8", "T8 prior"), ("r7", "round 7"), ("r8a", "8a: T8 case"), ("r8b", "8b: +recursion"), ("r8c", "8c: +reuse"),
        ("r8d", "8d: +inheritance"), ("r8e", "8e: +copy changes"), ("r8f", "8f: +budget x1.3"), ("r8", "round 8 base")]
AR = None
def ops(t):
    return sum(1 for x in t if x in ARITY)
import pool7 as p
ARITY = p.ar

def ks(a, b):
    a, b = np.sort(a), np.sort(b); xs = np.union1d(a, b)
    return float(np.max(np.abs(np.searchsorted(a, xs, side="right") / len(a) - np.searchsorted(b, xs, side="right") / len(b))))

def pct(xs): return 100 * np.mean(xs)
rows = []
def row(label, f):
    rows.append(f"| {label} | " + " | ".join(f(D[k]) for k, _ in COLS) + " |")
t8ops = [ops(r["simp"]) for r in D["t8"]]
row("draws", lambda R: f"{len(R):,}")
row("median / p90 operators", lambda R: f"{np.median([ops(r['simp']) for r in R]):.0f} / {np.percentile([ops(r['simp']) for r in R], 90):.0f}")
row("median / p90 tokens", lambda R: f"{np.median([len(r['simp']) for r in R]):.0f} / {np.percentile([len(r['simp']) for r in R], 90):.0f}")
row("KS distance of operator count to T8", lambda R: f"{ks([ops(r['simp']) for r in R], t8ops):.3f}")
row("> 35 tokens (%)", lambda R: f"{pct([len(r['simp']) > 35 for r in R]):.1f}")
row("trivial, <= 5 tokens (%)", lambda R: f"{pct([len(r['simp']) <= 5 for r in R]):.1f}")
row("raw operators - budget: mean / exact (%)", lambda R: f"{np.mean([r['raw_ops'] - r['B'] for r in R]):+.2f} / {pct([r['raw_ops'] == r['B'] for r in R]):.0f}")
row("median variables / one variable (%)", lambda R: f"{np.median([r['cls']['nv'] for r in R]):.0f} / {pct([r['cls']['nv'] == 1 for r in R]):.1f}")
row(">= 2 / >= 3 additive terms (%)", lambda R: f"{pct([r['cls']['k'] >= 2 for r in R]):.0f} / {pct([r['cls']['k'] >= 3 for r in R]):.0f}")
row("two terms share a form (%)", lambda R: f"{pct([r['cls']['rep'] for r in R]):.1f}")
row(">= 3 terms of one form (%)", lambda R: f"{pct([r['cls']['sup3'] for r in R]):.2f}")
row("sum of >= 2 rational terms (%)", lambda R: f"{pct([r['cls']['rat2'] for r in R]):.2f}")
row("polynomial, >= 3 terms (%)", lambda R: f"{pct([r['cls']['poly3'] for r in R]):.2f}")
row("univariate polynomial, >= 3 terms (%)", lambda R: f"{pct([r['cls']['poly1v3'] for r in R]):.2f}")
row("contains a transcendental (%)", lambda R: f"{pct([r['cls']['trans'] for r in R]):.0f}")
row("function inside a function (%)", lambda R: f"{pct([r['fn_depth'] >= 2 for r in R]):.1f}")
row("nested terms used (%)", lambda R: f"{pct([r['levels'] >= 2 for r in R]):.0f}")
def reuse(R):
    u = [x for r in R for x in r["uses"]]; w = [r for r in R if r["uses"]]
    return f"{np.mean(u):.2f} / {pct([all(x == 1 for x in r['uses']) for r in w]):.0f}" if u else "–"
row("uses per top-level composite / all used once (%)", reuse)
row("held out, precise key (%)", lambda R: f"{pct([r['held'] for r in R]):.1f}")
def coh(R, k):
    if R[0].get("coh") is None: return "–"
    a = sum(r["coh"][k][0] for r in R); n = sum(r["coh"][k][1] for r in R)
    return f"{100 * a / max(n, 1):.0f} ({n:,})"
row("COHERENCE: nested variable atoms reusing an ancestor's variable (%, n)", lambda R: coh(R, "var"))
row("COHERENCE: nested trees with the parent's majority class (%, n)", lambda R: coh(R, "cls"))
row(">= 2 function families, of draws with >= 2 functions (%)", lambda R: f"{pct([len(r['cls']['fam']) >= 2 for r in R if r['nfn'] >= 2]):.0f}")
HYP = {"sinh", "cosh", "tanh", "asinh", "acosh", "atanh"}
def fshare(R, S):
    fn = [x for r in R for x in r["fn_raw"]]; return 100 * sum(x in S for x in fn) / max(len(fn), 1)
row("hyperbolic, of function nodes before simplification (%)", lambda R: f"{fshare(R, HYP):.1f}")
row("functions per draw before simplification", lambda R: f"{np.mean([len(r['fn_raw']) for r in R]):.2f}")
row("exp / log / sin+cos, of function nodes (%)", lambda R: f"{fshare(R, {'exp'}):.1f} / {fshare(R, {'log'}):.1f} / {fshare(R, {'sin', 'cos'}):.1f}")
print("| | " + " | ".join(n for _, n in COLS) + " |"); print("|---" * (len(COLS) + 1) + "|"); print("\n".join(rows))

# r8a against T8: two-sample KS (5% critical value) and two-proportion z-scores
a, b = D["r8a"], D["t8"]; n1, n2 = len(a), len(b); crit = 1.36 * np.sqrt((n1 + n2) / (n1 * n2))
print(f"\nr8a vs T8 ({n1} vs {n2} draws), KS 5% critical value {crit:.3f}:")
for lab, f in [("operators", lambda r: ops(r["simp"])), ("tokens", lambda r: len(r["simp"])), ("variables", lambda r: r["cls"]["nv"]),
               ("additive terms", lambda r: r["cls"]["k"]), ("raw operators", lambda r: r["raw_ops"])]:
    print(f"  KS {lab}: {ks([f(r) for r in a], [f(r) for r in b]):.3f}")
for lab, f in [("transcendental", lambda r: r["cls"]["trans"]), ("held out", lambda r: r["held"]), ("one variable", lambda r: r["cls"]["nv"] == 1),
               ("two terms share a form", lambda r: r["cls"]["rep"]), ("fn in fn", lambda r: r["fn_depth"] >= 2), ("<= 5 tokens", lambda r: len(r["simp"]) <= 5)]:
    x, y = np.mean([f(r) for r in a]), np.mean([f(r) for r in b]); pp = (x * n1 + y * n2) / (n1 + n2)
    z = (x - y) / max(np.sqrt(pp * (1 - pp) * (1 / n1 + 1 / n2)), 1e-12)
    print(f"  {lab}: {100*x:.1f}% vs {100*y:.1f}%  z = {z:+.2f}")

# 10 unfiltered draws per step (the first 10 of each traced bank)
bank = json.load(open("bank_ladder.json")) + json.load(open("bank_base.json"))
with open("r8_draws.txt", "w") as fh:
    for k, lab in COLS[2:]:
        fh.write(f"### {lab}\n")
        for i, r in enumerate([x for x in bank if x["prior"] == k][:10]):
            try: inf = __import__("pretty").pretty(r["simp"])
            except Exception: inf = " ".join(r["simp"])
            fh.write(f"{i + 1:2d}. [{len(r['simp'])} tok, B={r['B']}, {r['levels']} lvl] {inf}\n")
        fh.write("\n")
print("\nwrote r8_ecdf.png, r8_draws.txt")
