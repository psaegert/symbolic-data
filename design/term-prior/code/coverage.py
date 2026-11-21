"""Coverage test: decompose simplified prefix expressions top-down with three relations and count what is left.

Every maximal chain of + - (through neg) or * / (through inv) is flattened into its operands. Constant subtrees (no
variable) are offsets or coefficients, not operands. A chain with non-constant operands o_1..o_k is
  repeat and combine (R1)  k >= 2 and all o_i have one template,
  affine (R2a)             k = 1 (c*A + c, A/c, ...),
  pairing (R3)             k = 2 with two templates,
  mixed (M)                k >= 3 and not one template: what the relations do not explain.
A function, a power or root (exponent kept as a parameter), or inv outside a product is composition (R2).
Templates: variables -> x, constants -> c, + and - one operator, exponents masked; a summand's constant factors are
stripped (c_i * T is a copy of T); in a product every factor is "a power of its base" (x, 1/x, x^2, sqrt(x) are copies).
Labels are weighted by the operators they account for (a chain of k operands holds k - 1 binary operators).
"""
import json, gzip, collections, sys
import numpy as np
import struct_scores as S

ADD, MUL = {"+", "-"}, {"*", "/"}
def is_const(node):
    t, kids = node
    return (not kids and not S.is_var(t)) or (bool(kids) and all(is_const(k) for k in kids))

def flatten(node, ops, through):
    t, kids = node
    if t in ops: return flatten(kids[0], ops, through) + flatten(kids[1], ops, through)
    if t in through: return flatten(kids[0], ops, through)
    return [node]

def base(node):
    """A factor as a power of its base: strip inv, pow, rootn (the exponent is a parameter)."""
    t, kids = node
    while t in ("inv", "pow", "rootn"): node = kids[0]; t, kids = node
    return node

def tmpl(node):
    if is_const(node): return "c"
    t, kids = node
    if not kids: return "x"
    if t == "neg": return tmpl(kids[0])
    if t in ADD:
        ops = [o for o in flatten(node, ADD, {"neg"}) if not is_const(o)]
        return "sum(" + ",".join(sorted(summand(o) for o in ops)) + ")"
    if t in MUL or t == "inv":
        ops = [o for o in flatten(node, MUL, {"inv"}) if not is_const(o)]
        return "prod(" + ",".join(sorted("pow(" + tmpl(base(o)) + ")" for o in ops)) + ")"
    if t in ("pow", "rootn"): return "pow(" + tmpl(kids[0]) + ")"
    return f"{t}(" + tmpl(kids[0]) + ")"

def summand(node):
    """A summand's template with its constant coefficients stripped; a single power counts like its base's power."""
    t, kids = node
    while t == "neg": node = kids[0]; t, kids = node
    if t in MUL or t == "inv":
        ops = [o for o in flatten(node, MUL, {"inv"}) if not is_const(o)]
        if len(ops) == 1: return "pow(" + tmpl(base(ops[0])) + ")"
        return "prod(" + ",".join(sorted("pow(" + tmpl(base(o)) + ")" for o in ops)) + ")"
    if t in ("pow", "rootn") or not kids: return "pow(" + tmpl(base(node)) + ")"
    return tmpl(node)

def decompose(tokens):
    labels = collections.Counter(); leftovers = []; root = []
    def visit(node, top=False):
        t, kids = node
        if not kids or is_const(node): return
        if t == "neg": return visit(kids[0], top)
        if t in ADD or t in MUL:
            kind = "sum" if t in ADD else "product"
            ops_all = flatten(node, ADD if kind == "sum" else MUL, {"neg"} if kind == "sum" else {"inv"})
            nonc = [o for o in ops_all if not is_const(o)]
            ts = [summand(o) if kind == "sum" else "pow(" + tmpl(base(o)) + ")" for o in nonc]
            w = len(ops_all) - 1
            if len(nonc) == 1: lab = "affine"
            elif len(set(ts)) == 1: lab = f"repeat {kind}, {'2' if len(nonc) == 2 else '3+'} copies"
            elif len(nonc) == 2: lab = f"pairing ({kind})"
            else:
                lab = f"mixed {kind}"
                leftovers.append(tuple(sorted(collections.Counter(ts).values(), reverse=True)))
            labels[lab] += w
            if top: root.append(lab)
            for o in ops_all:
                # a factor's or summand's own power / root / inverse is a parameter of the copy, not a composition
                visit(base(o))
            return
        lab = "composition (function)" if t in S.FN else "composition (power, root, 1/x)"
        labels[lab] += 1
        if top: root.append(lab)
        visit(kids[0])
    visit(S.parse(tokens), top=True)
    return labels, leftovers, (root[0] if root else "atom")

def run(name, exprs):
    L = collections.Counter(); per = []; roots = collections.Counter(); left = collections.Counter()
    for e in exprs:
        lab, lo, r = decompose(e); L.update(lab); per.append(lab); roots[r] += 1; left.update(lo)
    return {"L": L, "per": per, "roots": roots, "left": left, "n": len(exprs)}

if __name__ == "__main__":
    D = {"T8": [r["simp"] for r in json.load(gzip.open("rows_ladder.json.gz", "rt"))["t8"]],
         "round 8": [r["simp"] for r in json.load(gzip.open("rows_base.json.gz", "rt"))["r8"]],
         "round 9b": [r["simp"] for r in json.load(gzip.open("rows_r9b.json.gz", "rt"))["r9b"]],
         "real formulas": [c["simp"] for c in json.load(gzip.open("corpus_prefix.json.gz", "rt"))]}
    R = {k: run(k, v) for k, v in D.items()}
    labs = ["repeat product, 2 copies", "repeat product, 3+ copies", "repeat sum, 2 copies", "repeat sum, 3+ copies", "affine",
            "pairing (sum)", "pairing (product)", "mixed sum", "mixed product", "composition (function)", "composition (power, root, 1/x)"]
    print("Share of operators explained by each relation (%):")
    print("| | " + " | ".join(D) + " |"); print("|---" * (len(D) + 1) + "|")
    for l in labs:
        print(f"| {l} | " + " | ".join(f"{100 * R[k]['L'][l] / max(sum(R[k]['L'].values()), 1):.1f}" for k in D) + " |")
    print("\nPer expression (%):")
    print("| | " + " | ".join(D) + " |"); print("|---" * (len(D) + 1) + "|")
    rows = [("no mixed chain (fully explained by the three relations)", lambda p: not (p["mixed sum"] or p["mixed product"])),
            ("uses repeat and combine", lambda p: any(p[l] for l in labs[:4])),
            ("uses repeat and combine with 3+ copies", lambda p: p["repeat product, 3+ copies"] or p["repeat sum, 3+ copies"]),
            ("uses a repeat sum (superposition of one template)", lambda p: p["repeat sum, 2 copies"] or p["repeat sum, 3+ copies"]),
            ("operators in mixed chains >= half", lambda p: (p["mixed sum"] + p["mixed product"]) * 2 >= max(sum(p.values()), 1))]
    for lab, f in rows:
        print(f"| {lab} | " + " | ".join(f"{100 * np.mean([bool(f(p)) for p in R[k]['per']]):.1f}" for k in D) + " |")
    print("\nRoot relation (%):")
    allroots = sorted({r for k in D for r in R[k]["roots"]})
    for r_ in allroots:
        print(f"| {r_} | " + " | ".join(f"{100 * R[k]['roots'][r_] / R[k]['n']:.1f}" for k in D) + " |")
    for k in ("real formulas", "T8"):
        tot = sum(R[k]["left"].values())
        print(f"\n{k}: mixed chains by template multiplicity (sorted counts of operands per template), top 8 of {tot}:")
        for pat, c in R[k]["left"].most_common(8): print(f"   {pat}: {c} ({100 * c / max(tot, 1):.0f}%)")
    json.dump({k: {"L": dict(R[k]["L"]), "roots": dict(R[k]["roots"])} for k in D}, open("coverage.json", "w"), indent=1)
