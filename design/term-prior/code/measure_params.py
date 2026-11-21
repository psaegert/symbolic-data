"""Measure the relation parameters on simplified expressions (real formulas, or a prior's draws) with the coverage parse.

Repeat chains (every non-constant operand one template): number of copies k; how each copy differs from the first
(variables, exponents, constants inside the template); distinct variables in the repeat; for products the signed
exponent of every factor (1, -1, 2, 1/2, ...); for sums the share of summands with a constant coefficient and with a
minus sign. Pairing chains (two templates): which operator joins them.
"""
import json, gzip, collections, sys
from fractions import Fraction
import numpy as np
import struct_scores as S
import coverage as CV

def signed_flat(node, kind, sign=1):
    """Operands of a + - chain with their sign, or of a * / chain with the sign of their exponent."""
    t, kids = node
    if kind == "sum":
        if t == "+": return signed_flat(kids[0], kind, sign) + signed_flat(kids[1], kind, sign)
        if t == "-": return signed_flat(kids[0], kind, sign) + signed_flat(kids[1], kind, -sign)
        if t == "neg": return signed_flat(kids[0], kind, -sign)
    else:
        if t == "*": return signed_flat(kids[0], kind, sign) + signed_flat(kids[1], kind, sign)
        if t == "/": return signed_flat(kids[0], kind, sign) + signed_flat(kids[1], kind, -sign)
        if t == "inv": return signed_flat(kids[0], kind, -sign)
    return [(node, sign)]

def power_of(node):
    """(base, exponent) of a factor: pow, rootn and inv folded into the exponent."""
    e = Fraction(1); t, kids = node
    while t in ("pow", "rootn", "inv"):
        if t == "inv": e = -e
        else:
            try: v = Fraction(kids[1][0]).limit_denominator(64)
            except Exception: v = Fraction(1)
            e = e * (v if t == "pow" else (1 / v if v != 0 else 1))
        node = kids[0]; t, kids = node
    return node, e

def strip_coef(node):
    """A summand without its constant factors; returns (core, had_coefficient)."""
    t, kids = node
    while t == "neg": node = kids[0]; t, kids = node
    if t in ("*", "/", "inv"):
        ops = signed_flat(node, "product")
        nonc = [(o, s) for o, s in ops if not CV.is_const(o)]
        if len(nonc) < len(ops) and len(nonc) == 1 and nonc[0][1] == 1: return nonc[0][0], True
        if len(nonc) < len(ops): return node, True
    return node, False

def leaves(node, out):
    """Leaf tokens by role: ('v', name) variables, ('c', value) constants, ('e', literal) exponents/indices."""
    t, kids = node
    if not kids: out.append(("v" if S.is_var(t) else "c", t)); return out
    if t in ("pow", "rootn"): leaves(kids[0], out); out.append(("e", kids[1][0])); return out
    for k in kids: leaves(k, out)
    return out

def variables(node): return {x for r, x in leaves(node, []) if r == "v"}

def measure(exprs):
    M = {"k": {"product": collections.Counter(), "sum": collections.Counter()},
         "diff": {"product": collections.Counter(), "sum": collections.Counter()}, "copies": {"product": 0, "sum": 0},
         "vars_per_repeat": {"product": [], "sum": []}, "exp": collections.Counter(), "coef": [0, 0], "minus": [0, 0],
         "pair": collections.Counter(), "repeats": {"product": 0, "sum": 0}}
    def visit(node):
        t, kids = node
        if not kids or CV.is_const(node): return
        if t == "neg": return visit(kids[0])
        if t in CV.ADD or t in CV.MUL:
            kind = "sum" if t in CV.ADD else "product"
            ops = signed_flat(node, kind)
            nonc = [(o, s) for o, s in ops if not CV.is_const(o)]
            ts = [CV.summand(o) if kind == "sum" else "pow(" + CV.tmpl(CV.base(o)) + ")" for o, _ in nonc]
            if len(nonc) >= 2 and len(set(ts)) == 1:
                M["repeats"][kind] += 1; M["k"][kind][min(len(nonc), 6)] += 1
                if kind == "product":
                    parts = [power_of(o) for o, _ in nonc]
                    cores = [b for b, _ in parts]; exps = [e * s for (b, e), (_, s) in zip(parts, nonc)]
                    for e in exps: M["exp"][str(e)] += 1
                else:
                    sc = [strip_coef(o) for o, _ in nonc]; cores = [c for c, _ in sc]
                    M["coef"][0] += sum(h for _, h in sc); M["coef"][1] += len(sc)
                    M["minus"][0] += sum(s < 0 for _, s in nonc); M["minus"][1] += len(nonc); exps = [None] * len(nonc)
                L0 = leaves(cores[0], [])
                for i in range(1, len(cores)):
                    Li = leaves(cores[i], []); d = set()
                    if len(Li) == len(L0):
                        for (r0, x0), (r1, x1) in zip(L0, Li):
                            if x0 != x1: d.add({"v": "variable", "c": "constant", "e": "exponent"}[r0])
                    else: d.add("shape")
                    if kind == "product" and exps[i] != exps[0]: d.add("exponent")
                    M["copies"][kind] += 1
                    for x in (d or {"none"}): M["diff"][kind][x] += 1
                M["vars_per_repeat"][kind].append(len(set().union(*[variables(c) for c in cores])))
            elif len(nonc) == 2:
                M["pair"][(kind, "-" if kind == "sum" and nonc[1][1] < 0 else "/" if kind == "product" and nonc[1][1] < 0 else "+" if kind == "sum" else "*")] += 1
            for o, _ in ops: visit(CV.base(o))
            return
        visit(kids[0])
    for e in exprs: visit(S.parse(e))
    return M

def report(M, name):
    out = {"name": name}
    for kind in ("product", "sum"):
        n = sum(M["k"][kind].values()) or 1
        out[f"{kind}: k = 2/3/4/5/6+ (%)"] = " / ".join(f"{100 * M['k'][kind][k] / n:.0f}" for k in (2, 3, 4, 5, 6))
        c = M["copies"][kind] or 1
        out[f"{kind}: copies differing in variable / exponent / constant / shape (%)"] = " / ".join(f"{100 * M['diff'][kind][x] / c:.0f}" for x in ("variable", "exponent", "constant", "shape"))
        v = M["vars_per_repeat"][kind]; ks = M["k"][kind]
        out[f"{kind}: distinct variables per repeat, median / mean"] = f"{np.median(v) if v else 0:.0f} / {np.mean(v) if v else 0:.2f}"
    tot = sum(M["exp"].values()) or 1
    top = M["exp"].most_common(8)
    out["product: factor exponents (%)"] = ", ".join(f"{e}: {100 * c / tot:.0f}" for e, c in top)
    out["product: factors with a negative exponent (%)"] = f"{100 * sum(c for e, c in M['exp'].items() if Fraction(e) < 0) / tot:.0f}"
    out["product: factors with exponent other than +-1 (%)"] = f"{100 * sum(c for e, c in M['exp'].items() if abs(Fraction(e)) != 1) / tot:.0f}"
    out["sum: summands with a coefficient / with minus (%)"] = f"{100 * M['coef'][0] / max(M['coef'][1], 1):.0f} / {100 * M['minus'][0] / max(M['minus'][1], 1):.0f}"
    pt = sum(M["pair"].values()) or 1
    out["pairing: + / - / * / ÷ (%)"] = " / ".join(f"{100 * M['pair'][(k, o)] / pt:.0f}" for k, o in (("sum", "+"), ("sum", "-"), ("product", "*"), ("product", "/")))
    return out

if __name__ == "__main__":
    srcs = {"real formulas": [c["simp"] for c in json.load(gzip.open("corpus_prefix.json.gz", "rt"))],
            "round 10": [r["simp"] for r in json.load(gzip.open("rows_r10.json.gz", "rt"))["r10"]],
            "round 10b": [r["simp"] for r in json.load(gzip.open("rows_r10b.json.gz", "rt"))["r10b"]]}
    R = {k: report(measure(v), k) for k, v in srcs.items()}
    keys = [k for k in R["real formulas"] if k != "name"]
    print("| | " + " | ".join(srcs) + " |"); print("|---" * (len(srcs) + 1) + "|")
    for k in keys: print(f"| {k} | " + " | ".join(R[s][k] for s in srcs) + " |")
    json.dump(R, open("measured_params.json", "w"), indent=1)
