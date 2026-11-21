"""Round 10 against T8, round 8, round 9b and real formulas: rates, structure scores, coverage, 100 draws."""
import json, gzip, collections, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pool7 as p
import struct_scores as S
import coverage as CV

L = lambda f, k: json.load(gzip.open(f, "rt"))[k]
D = {"t8": L("rows_ladder.json.gz", "t8"), "r8": L("rows_base.json.gz", "r8"), "r9b": L("rows_r9b.json.gz", "r9b"), "r10": L("rows_r10.json.gz", "r10"), "r10b": L("rows_r10b.json.gz", "r10b"), "r10c": L("rows_r10c.json.gz", "r10c")}
corpus = json.load(gzip.open("corpus_prefix.json.gz", "rt"))
D["corpus"] = [{"simp": c["simp"], "cls": p.record_classes(c["simp"]), "held": None, "fn_depth": None, "form": None} for c in corpus]
COLS = [("t8", "T8 prior"), ("r8", "round 8 (base)"), ("r10b", "10b: s mean 0.25"), ("r10c", "10c: s mean 0.5"), ("corpus", "real formulas")]
REL3 = {"repeat product, 2 copies", "repeat product, 3+ copies", "repeat sum, 2 copies", "repeat sum, 3+ copies"}

def mixed3(tokens):
    """Operators in chains of >= 3 different templates, and all operators (coverage with 'a repeat plus one part')."""
    bad = [0, 0]
    def visit(node):
        t, kids = node
        if not kids or CV.is_const(node): return
        if t == "neg": return visit(kids[0])
        if t in CV.ADD or t in CV.MUL:
            kind = "sum" if t in CV.ADD else "product"
            ops = CV.flatten(node, CV.ADD if kind == "sum" else CV.MUL, {"neg"} if kind == "sum" else {"inv"})
            ts = [CV.summand(o) if kind == "sum" else "pow(" + CV.tmpl(CV.base(o)) + ")" for o in ops if not CV.is_const(o)]
            bad[1] += len(ops) - 1
            if len(set(ts)) >= 3: bad[0] += len(ops) - 1
            for o in ops: visit(CV.base(o))
            return
        bad[1] += 1; visit(kids[0])
    visit(S.parse(tokens)); return bad

for k in D:
    for r in D[k]:
        r.update(S.scores(r["simp"]))
        lab, _, root = CV.decompose(r["simp"]); r["cov"] = lab; r["covroot"] = root; r["m3"] = mixed3(r["simp"])
ops = lambda t: sum(1 for x in t if x in p.ar)
pct = lambda xs: 100 * np.mean(xs) if len(xs) else float("nan")
lines = []
def row(label, f, only=None):
    lines.append(f"| {label} | " + " | ".join(f(D[k]) if (only is None or k in only) else "–" for k, _ in COLS) + " |")
def share(R, labs):
    tot = sum(sum(r["cov"].values()) for r in R); return 100 * sum(sum(r["cov"][l] for l in labs) for r in R) / max(tot, 1)
P = ("t8", "r8", "r9b", "r10", "r10b", "r10c")
row("draws / formulas", lambda R: f"{len(R):,}")
row("median / p90 operators", lambda R: f"{np.median([ops(r['simp']) for r in R]):.0f} / {np.percentile([ops(r['simp']) for r in R], 90):.0f}")
row("median / p90 tokens", lambda R: f"{np.median([len(r['simp']) for r in R]):.0f} / {np.percentile([len(r['simp']) for r in R], 90):.0f}")
row("> 35 tokens (%)", lambda R: f"{pct([len(r['simp']) > 35 for r in R]):.1f}")
row("trivial, <= 5 tokens (%)", lambda R: f"{pct([len(r['simp']) <= 5 for r in R]):.1f}")
row("median variables / one variable (%)", lambda R: f"{np.median([r['cls']['nv'] for r in R]):.0f} / {pct([r['cls']['nv'] == 1 for r in R]):.1f}")
row(">= 3 terms of one form (%) [target 3-5]", lambda R: f"{pct([r['cls']['sup3'] for r in R]):.2f}")
row("sum of >= 2 rational terms (%) [target 2-4]", lambda R: f"{pct([r['cls']['rat2'] for r in R]):.2f}")
row("polynomial, >= 3 terms (%) [target 3-5]", lambda R: f"{pct([r['cls']['poly3'] for r in R]):.2f}")
row("two terms share a form (%)", lambda R: f"{pct([r['cls']['rep'] for r in R]):.1f}")
row("contains a transcendental (%)", lambda R: f"{pct([r['cls']['trans'] for r in R]):.0f}")
row("function inside a function (%)", lambda R: f"{pct([r['fn_depth'] >= 2 for r in R]):.1f}", only=P)
row("held out, precise key (%)", lambda R: f"{pct([r['held'] for r in R]):.1f}", only=P)
row("COVERAGE fully explained, repeat may pair with one part (%)", lambda R: f"{pct([r['m3'][0] == 0 for r in R]):.1f}")
row("COVERAGE operators: repeat product / repeat sum / mixed (%)", lambda R: f"{share(R, {'repeat product, 2 copies', 'repeat product, 3+ copies'}):.0f} / {share(R, {'repeat sum, 2 copies', 'repeat sum, 3+ copies'}):.0f} / {share(R, {'mixed sum', 'mixed product'}):.0f}")
row("COVERAGE operators: functions (%)", lambda R: f"{share(R, {'composition (function)'}):.1f}")
row("COVERAGE root is a repeat product / repeat sum (%)", lambda R: f"{pct([r['covroot'].startswith('repeat product') for r in R]):.1f} / {pct([r['covroot'].startswith('repeat sum') for r in R]):.1f}")
row("STRUCTURE repetition >= 0.5 (%)", lambda R: f"{pct([r['rep'] >= 0.5 for r in R]):.1f}")
row("STRUCTURE alternation <= 1 / >= 3 (%)", lambda R: f"{pct([r['alt'] <= 1 for r in R]):.0f} / {pct([r['alt'] >= 3 for r in R]):.0f}")
row("root term is a relation (%)", lambda R: f"{pct([r['form'] not in (None, 'LC tree') for r in R]):.0f}", only=("r9b", "r10", "r10b", "r10c"))
print("| | " + " | ".join(n for _, n in COLS) + " |"); print("|---" * (len(COLS) + 1) + "|"); print("\n".join(lines))
print("\nround-10c root relation:", dict(collections.Counter((r["form"] or "?").split(" of")[0] for r in D["r10c"]).most_common(10)))
json.dump({k: [[r["rep"], r["alt"], r["top"], len(r["simp"]), r["cls"]["k"]] for r in D[k]] for k in ("t8", "r8", "r10", "r10b", "r10c", "corpus")},
          gzip.open("r10_plotdata.json.gz", "wt"))
bank = json.load(open("bank_r10c.json"))
out = ["# 100 draws of round 10c", "",
       "Generated 2026-11-21. The first 100 draws, unfiltered, after simplification, as the model is trained on them. Round 10c is round 10b with the structure level s drawn uniformly, s ~ Beta(1, 1) (mean 0.5) instead of Beta(0.5, 1.5) (mean 0.25). Round 10b is round 10 with its relation parameters (copies per repeat, exponents, divisions, coefficients, exponent changes in copies) set halfway to the values measured on 10,801 real formulas. Round 10 is the round-8 base plus one structure level s ~ Beta(0.5, 1.5) per expression: every term with operators is, with probability s, one of three relations (repeat and combine, composition, pairing), and otherwise an LC tree as in round 8. Relation shares are halfway between equal shares and those of 10,801 real formulas. `s` and the root term are given per draw.", ""]
for i, r in enumerate(bank[:100], 1):
    try: inf = __import__("pretty").pretty(r["simp"])
    except Exception: inf = " ".join(r["simp"])
    out.append(f"{i}. `{inf}` [s = {r['H']['theta']['s_struct']:.2f}, root: {r['H']['form']}, {len(r['simp'])} tok]")
open("draws_r10c.md", "w").write("\n".join(out) + "\n")
print("wrote draws_r10c.md, r10_plotdata.json.gz")
