"""Round 9 against T8, the round-8 base and the real-formula corpus: table, structure-score plot data, 100 draws."""
import json, gzip, collections, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pool7 as p
import struct_scores as S

D = {"t8": json.load(gzip.open("rows_ladder.json.gz", "rt"))["t8"], "r8": json.load(gzip.open("rows_base.json.gz", "rt"))["r8"],
     "r9": json.load(gzip.open("rows_r9.json.gz", "rt"))["r9"], "r9b": json.load(gzip.open("rows_r9b.json.gz", "rt"))["r9b"]}
corpus = json.load(gzip.open("corpus_prefix.json.gz", "rt"))
D["corpus"] = [{"simp": c["simp"], "cls": p.record_classes(c["simp"]), "held": None, "fn_depth": None, "nfn": sum(1 for x in c["simp"] if x in S.FN)} for c in corpus]
COLS = [("t8", "T8 prior"), ("r8", "round 8 (base)"), ("r9", "round 9"), ("r9b", "9b: one s per expression"), ("corpus", "real formulas")]
for k in D:
    for r in D[k]: r.update(S.scores(r["simp"]))
ops = lambda t: sum(1 for x in t if x in p.ar)
pct = lambda xs: 100 * np.mean(xs) if len(xs) else float("nan")
lines = []
def row(label, f, only=None):
    lines.append(f"| {label} | " + " | ".join(f(D[k]) if (only is None or k in only) else "–" for k, _ in COLS) + " |")
row("draws / formulas", lambda R: f"{len(R):,}")
row("median / p90 operators", lambda R: f"{np.median([ops(r['simp']) for r in R]):.0f} / {np.percentile([ops(r['simp']) for r in R], 90):.0f}")
row("median / p90 tokens", lambda R: f"{np.median([len(r['simp']) for r in R]):.0f} / {np.percentile([len(r['simp']) for r in R], 90):.0f}")
row("> 35 tokens (%)", lambda R: f"{pct([len(r['simp']) > 35 for r in R]):.1f}")
row("trivial, <= 5 tokens (%)", lambda R: f"{pct([len(r['simp']) <= 5 for r in R]):.1f}")
row("median variables / one variable (%)", lambda R: f"{np.median([r['cls']['nv'] for r in R]):.0f} / {pct([r['cls']['nv'] == 1 for r in R]):.1f}")
row(">= 2 / >= 3 additive terms (%)", lambda R: f"{pct([r['cls']['k'] >= 2 for r in R]):.0f} / {pct([r['cls']['k'] >= 3 for r in R]):.0f}")
row("two terms share a form (%)", lambda R: f"{pct([r['cls']['rep'] for r in R]):.1f}")
row(">= 3 terms of one form (%) [target 3-5]", lambda R: f"{pct([r['cls']['sup3'] for r in R]):.2f}")
row("sum of >= 2 rational terms (%) [target 2-4]", lambda R: f"{pct([r['cls']['rat2'] for r in R]):.2f}")
row("polynomial, >= 3 terms (%) [target 3-5]", lambda R: f"{pct([r['cls']['poly3'] for r in R]):.2f}")
row("univariate polynomial, >= 3 terms (%)", lambda R: f"{pct([r['cls']['poly1v3'] for r in R]):.2f}")
row("contains a transcendental (%)", lambda R: f"{pct([r['cls']['trans'] for r in R]):.0f}")
row("function inside a function (%)", lambda R: f"{pct([r['fn_depth'] >= 2 for r in R]):.1f}", only=("t8", "r8", "r9", "r9b"))
row("held out, precise key (%)", lambda R: f"{pct([r['held'] for r in R]):.1f}", only=("t8", "r8", "r9", "r9b"))
row("STRUCTURE repetition: mean / > 0 (%) / >= 0.5 (%)", lambda R: f"{np.mean([r['rep'] for r in R]):.2f} / {pct([r['rep'] > 0 for r in R]):.0f} / {pct([r['rep'] >= 0.5 for r in R]):.1f}")
row("STRUCTURE alternation: <= 1 (%) / >= 3 (%)", lambda R: f"{pct([r['alt'] <= 1 for r in R]):.0f} / {pct([r['alt'] >= 3 for r in R]):.0f}")
row("STRUCTURE root: sum / product / function / power / atom (%)", lambda R: " / ".join(f"{pct([r['top'] == t for r in R]):.0f}" for t in ("sum", "product", "function", "power", "atom")))
row("root term is a structural form (%)", lambda R: f"{pct([r['form'] not in (None, 'LC tree') for r in R]):.0f}", only=("r9", "r9b"))
print("| | " + " | ".join(n for _, n in COLS) + " |"); print("|---" * (len(COLS) + 1) + "|"); print("\n".join(lines))
for k in ("r9", "r9b"): print(f"\n{k} root forms:", dict(collections.Counter((r["form"] or "?").split(" ")[0] for r in D[k]).most_common()))
# plot data for r9_plot.py (srbf venv)
json.dump({k: [[r["rep"], r["alt"], r["top"], len(r["simp"]), r["cls"]["k"]] for r in D[k]] for k in D}, gzip.open("r9_plotdata.json.gz", "wt"))

bank = json.load(open("bank_r9b.json"))
out = ["# 100 draws of round 9b", "", "Generated 2026-11-21. The first 100 draws, unfiltered, after simplification, as the model is trained on them. Round 9b is the round-8 base plus one structure level s ~ Beta(0.5, 1.5) per expression: every term picks a structural form (sum, product, ratio, power, function, affine) with probability s, otherwise an LC tree. `s` and the root term's form are given per draw.", ""]
for i, r in enumerate(bank[:100], 1):
    try: inf = __import__("pretty").pretty(r["simp"])
    except Exception: inf = " ".join(r["simp"])
    out.append(f"{i}. `{inf}` [s = {r['H']['theta']['s_struct']:.2f}, root: {r['H']['form']}, {len(r['simp'])} tok]")
open("draws_r9b.md", "w").write("\n".join(out) + "\n")
print("wrote draws_r9b.md, r9_plotdata.json.gz")
