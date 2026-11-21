"""Round 8: the unified term prior.

One recursive object, Term(b, theta): a tree with n of the b operators, its leaf slots grouped, the remaining budget
split over the groups. A group with budget 0 is an atom (a variable or a constant); a group with budget >= 1 is a smaller
Term, and every slot of the group holds a copy of it. Three primitives do all the work:

  1. the budgeted recursion (raw length = B up to rounding; recursion ends because every term spends >= 1 operator),
  2. one grouping rule for slots (k = 1 + Binomial(L - 1, 1 - rho) groups, each takes one slot, the rest follow
     Dirichlet(alpha) weights); with rho ~ U(0,1) and alpha = infinity this IS T8's leaf rule,
  3. one copy-or-redraw kernel: a child term copies each hyperparameter (and draws its variables from the parent's
     alphabet) with probability lambda^depth, else redraws from the root prior; a copy of a composite term redraws
     parts of the template.

Ladder (each step adds one thing to the previous): r8a = T8 as a special case; r8b + recursion (and
round 7's nesting rule, no length cap); r8c + reuse (grouping with fewer groups, additive-share and family urns);
r8d + inheritance; r8e + copy changes; r8f = r8e with the budget scaled by 1.3.
"""
import collections, math
import numpy as np
import pool7 as p
from pool7 import ar, vs, vset, smp, URN, NS, T8_N, T8_P, num, sample_shape, slot_depths, q_bin, simplify_checked, modify_traced

KINDS = list(vs) + ["<c>"]
MAX_KINDS = len(vs)                     # T8: at most 17 distinct leaf kinds

BASE = dict(flat=1.0, rho="uniform", alpha=None, s_add=False, urn=False, lam=False, mutate=False, persist=False,
            max_len=35, B_scale=1.0)
LADDER = {"r8a": dict(BASE)}
LADDER["r8b"] = dict(LADDER["r8a"], flat=0.2, persist=True, max_len=None)                          # recursion
LADDER["r8c"] = dict(LADDER["r8b"], rho="beta21", alpha="logu", s_add=True, urn=True)             # reuse (urns)
LADDER["r8d"] = dict(LADDER["r8c"], lam=True)                                                      # inheritance
LADDER["r8e"] = dict(LADDER["r8d"], mutate=True)                                                   # copy changes
LADDER["r8f"] = dict(LADDER["r8e"], B_scale=1.3)                                                   # budget scale
LADDER["r8"] = dict(LADDER["r8f"])                 # round 8 base (= 8f; run with POOL7_FN_WEIGHTS=hyper=0.5)
LADDER["r9"] = dict(LADDER["r8"], struct=True)    # round 9: + structure level s and structural forms
LADDER["r9b"] = dict(LADDER["r9"], s_fixed=True)  # 9b: s is one value per expression, used by every term
LADDER["r10"] = dict(LADDER["r9b"], relations=True)  # round 10: structured terms are relations, not operator-level forms
# 10b: relation parameters halfway between round 10's values and those measured on 10,801 real formulas (measure_params.py)
LADDER["r10b"] = dict(LADDER["r10"], rp=dict(k_prod=([2, 3, 4, 5, 6], [0.51, 0.265, 0.13, 0.07, 0.025]),
                                             k_sum=([2, 3, 4, 5], [0.66, 0.205, 0.08, 0.055]),
                                             pow_wrap=0.33, div=0.26, coef=0.42, minus=0.24, p_exp=0.44))
LADDER["r10c"] = dict(LADDER["r10b"], s_beta=(1.0, 1.0))   # 10c: structure level uniform (mean 0.5) instead of Beta(0.5, 1.5)
NAMES = {"r8a": "T8 as a special case", "r8b": "+ recursion", "r8c": "+ reuse", "r8d": "+ inheritance",
         "r8e": "+ copy changes", "r8f": "+ budget x1.3"}

def root_prior(C):
    """The root prior pi of every per-term hyperparameter, for ladder config C."""
    pr = {
        "flat": lambda rng: bool(rng.random() < C["flat"]),
        "f": lambda rng: float(rng.beta(2, 3)),
        "rho": (lambda rng: float(rng.uniform())) if C["rho"] == "uniform" else (lambda rng: float(rng.beta(2, 1))),
        "alpha": (lambda rng: math.inf) if C["alpha"] is None else
                 (lambda rng: float(math.exp(rng.uniform(math.log(0.1), math.log(10))))),
    }
    if C["s_add"]: pr["s_add"] = lambda rng: float(rng.beta(0.5, 0.5))
    if C["mutate"]: pr["gate"] = lambda rng: bool(rng.random() < 0.5)
    if C.get("struct"): pr["s_struct"] = lambda rng: float(rng.beta(*C.get("s_beta", S_STRUCT)))
    return pr

# ---- round 9: structural forms. A term picks a form with probability s (its structure level), else an LC tree.
S_STRUCT = (0.5, 1.5)                                   # s ~ Beta(0.5, 1.5): mean 0.25, 40% below 0.1, 12% above 0.6
FORMS = ["sum", "product", "ratio", "power", "function", "affine"]
PI_FORM = np.array([0.25, 0.20, 0.15, 0.15, 0.15, 0.10]) # base shares of the per-expression form urn
ALPHA_FORM = 1.0                                         # form urn concentration (low: an expression commits to few forms)
K_SUM, P_SUM = [2, 3, 4, 5], [0.4, 0.3, 0.2, 0.1]
K_PROD, P_PROD = [2, 3, 4], [0.5, 0.3, 0.2]
P_MINUS, P_DIV = 0.25, 0.3

def family_label(rng, counts):
    """A function label from the family urn (no pow / rootn): the Function form."""
    if counts is None:
        F = [f for f in NS["F"] if f not in ("pow", "rootn")]
        w = np.array([NS["pF"][list(NS["F"]).index(f)] for f in F]); return str(rng.choice(F, p=w / w.sum()))
    p = URN["alpha"] * URN["pi"] + counts
    f = int(rng.choice(len(p), p=p / p.sum())); counts[f] += 1
    m = URN["members"][f]
    return m[0] if len(m) == 1 else str(rng.choice(m, p=URN["within"][f]))

def structural_form(rng, b, start, counts, fc):
    """One form whose operators fit the budget b >= 1. Returns its label and its tokens with slots @0, @1, ... in prefix
    order. fc: the expression's form-urn counts. A Function form is damped by the nesting decay at depth start."""
    w = ALPHA_FORM * PI_FORM + fc
    w[FORMS.index("function")] *= NS["gamma"] ** start
    if b < 2: w[FORMS.index("affine")] = 0.0
    j = int(rng.choice(len(FORMS), p=w / w.sum())); fc[j] += 1; f = FORMS[j]
    def chain(ops):          # left-deep chain over len(ops) + 1 slots: ((@0 o1 @1) o2 @2) ...
        return list(reversed(ops)) + [f"@{i}" for i in range(len(ops) + 1)]
    if f == "sum":
        k = min(int(rng.choice(K_SUM, p=P_SUM)), b + 1)
        return f"sum of {k}", chain(["-" if rng.random() < P_MINUS else "+" for _ in range(k - 1)])
    if f == "product":
        k = min(int(rng.choice(K_PROD, p=P_PROD)), b + 1)
        return f"product of {k}", chain(["/" if rng.random() < P_DIV else "*" for _ in range(k - 1)])
    if f == "ratio": return "ratio", ["/", "@0", "@1"]
    if f == "power":
        op = str(rng.choice(["pow", "rootn"]))
        return ("power" if op == "pow" else "root"), [op, "@0", num(smp._slot_literal(op, rng))]
    if f == "function":
        lab = family_label(rng, counts); return f"function {lab}", [lab, "@0"]
    return "affine", ["+", "*", num(smp._constant_literal(rng)), "@0", num(smp._constant_literal(rng))]

# ---- round 10: three relations (repeat and combine, composition, pairing). Shares halfway between equal shares and
# the root shares of 10,801 real formulas (coverage.py): repeat product 46.2, pairing 26.1, affine 11.1, power 8.0,
# repeat sum 7.4, function 1.2 (%).
REL = ["repeat product", "repeat sum", "pairing", "affine", "power", "function"]
PI_REL = np.array([31.5, 12.0, 21.4, 13.9, 12.3, 8.9]) / 100
ALPHA_REL = 1.0
K_REP, P_REP = [2, 3, 4, 5], [0.45, 0.30, 0.15, 0.10]
P_POW_WRAP, P_DIV, P_COEF, P_MINUS_REP = 0.4, 0.3, 0.5, 0.25
P_PAIR_PROD, P_PAIR_DIV, P_PAIR_MINUS = 0.6, 0.4, 0.25

def _operands(tokens, kind):
    """Non-constant operands of the root chain of tokens (sum or product), for the no-merge check."""
    import coverage as cv, struct_scores as ss
    node = ss.parse(tokens)
    while node[0] == "neg": node = node[1][0]
    ops = cv.flatten(node, cv.ADD if kind == "sum" else cv.MUL, {"neg"} if kind == "sum" else {"inv"})
    return [o for o in ops if not cv.is_const(o)]

def assemble(shape_t, leaves):
    raw, rawo = [], []
    for x in shape_t:
        if x.startswith("@"):
            lf = leaves[int(x[1:])]; raw += lf["tokens"]; rawo += [lf["pool"]] * len(lf["tokens"])
        else: raw.append(x); rawo.append(-1)
    return raw, rawo

RP10 = dict(k_prod=(K_REP, P_REP), k_sum=(K_REP, P_REP), pow_wrap=P_POW_WRAP, div=P_DIV, coef=P_COEF, minus=P_MINUS_REP, p_exp=0.5)

def relation_term(rng, b, depth, th, src, lam_d, alphabet, counts, C, lam, c0, fc, no_comp):
    """A structured term with budget b >= 1, as one of the three relations."""
    rp = C.get("rp", RP10)
    start = c0 if C["persist"] else 0
    w = ALPHA_REL * PI_REL + fc
    if no_comp: w[[REL.index("affine"), REL.index("power"), REL.index("function")]] = 0.0
    w[REL.index("function")] *= NS["gamma"] ** start
    if b < 2: w[REL.index("affine")] = 0.0
    j = int(rng.choice(len(REL), p=w / w.sum())); fc[j] += 1; rel = REL[j]
    child = lambda bb, cc, nc=False: term(rng, bb, depth + 1, th, alphabet, counts, C, lam, cc, True, fc, nc)
    pool, leaves, rounding = [], [], 0
    if rel.startswith("repeat"):
        kind = "product" if rel == "repeat product" else "sum"
        ks, ps = rp["k_prod"] if kind == "product" else rp["k_sum"]
        k = min(int(rng.choice(ks, p=np.array(ps) / sum(ps))), b + 1)
        R = b - (k - 1)
        wraps = []
        for i in range(k):
            p_wrap = rp["pow_wrap"] if kind == "product" else rp["coef"]
            if R >= 1 and rng.random() < p_wrap: wraps.append(True); R -= 1
            else: wraps.append(False)
        bT = R // k; rounding = R - k * bT
        tpl = child(bT, start)
        if kind == "product": ops = ["/" if rng.random() < rp["div"] else "*" for _ in range(k - 1)]
        else: ops = ["-" if rng.random() < rp["minus"] else "+" for _ in range(k - 1)]
        slot = []
        for i in range(k):
            if not wraps[i]: slot.append([f"@{i}"])
            elif kind == "product":
                op = str(rng.choice(["pow", "rootn"], p=[0.75, 0.25])); slot.append([op, f"@{i}", num(smp._slot_literal(op, rng))])
            else: slot.append(["*", num(smp._constant_literal(rng)), f"@{i}"])
        shape_t = list(reversed(ops)) + [t for sl in slot for t in sl]
        P = {"p_struct": 0.0, "p_exp": rp["p_exp"], "p_var_swap": 0.7, "p_shared_const": 0.0}
        leaves = [{"pool": 0, "tokens": list(tpl["raw"]), "changes": [None] * len(tpl["raw"]), "how": "copy"}]
        for i in range(1, k):
            for attempt in range(5):
                toks, ch = modify_traced(tpl["raw"], rng, True, dict(P, p_var_swap=0.7 if attempt == 0 else 1.0), counts)
                trial = leaves + [{"pool": 0, "tokens": toks, "changes": ch, "how": "copy"}]
                part, _ = assemble(list(reversed(ops[:i])) + [t for sl in slot[:i + 1] for t in sl], trial)
                s_part, _ = simplify_checked(part)
                if s_part is None or len(_operands(s_part, kind)) >= i + 1: break
            leaves = trial
        pool = [{"kind": "level", "tokens": tpl["raw"], "sub": tpl, "target": bT, "uses": k, "c0": start}]
        assign, label, n, budgets = [0] * k, f"{rel} of {k}", (k - 1) + sum(wraps), [bT]
    elif rel == "pairing":
        if rng.random() < P_PAIR_PROD: op = "/" if rng.random() < P_PAIR_DIV else "*"
        else: op = "-" if rng.random() < P_PAIR_MINUS else "+"
        R = b - 1; bA = int(round(rng.uniform() * R)); bB = R - bA
        A, Bt = child(bA, start), child(bB, start)
        shape_t = [op, "@0", "@1"]
        pool = [{"kind": "level", "tokens": A["raw"], "sub": A, "target": bA, "uses": 1, "c0": start},
                {"kind": "level", "tokens": Bt["raw"], "sub": Bt, "target": bB, "uses": 1, "c0": start}]
        leaves = [{"pool": t, "tokens": list(x["raw"]), "changes": [None] * len(x["raw"]), "how": "copy"} for t, x in enumerate((A, Bt))]
        assign, label, n, budgets = [0, 1], f"pairing {op}", 1, [bA, bB]
    else:
        if rel == "function":
            lab = family_label(rng, counts); shape_t = [lab, "@0"]; own, inner_c = 1, start + 1; label = f"function {lab}"
        elif rel == "power":
            op = str(rng.choice(["pow", "rootn"])); shape_t = [op, "@0", num(smp._slot_literal(op, rng))]; own, inner_c = 1, start; label = "power" if op == "pow" else "root"
        else:
            shape_t = ["+", "*", num(smp._constant_literal(rng)), "@0", num(smp._constant_literal(rng))]; own, inner_c = 2, start; label = "affine"
        A = child(b - own, inner_c, True)
        pool = [{"kind": "level", "tokens": A["raw"], "sub": A, "target": b - own, "uses": 1, "c0": inner_c}]
        leaves = [{"pool": 0, "tokens": list(A["raw"]), "changes": [None] * len(A["raw"]), "how": "copy"}]
        assign, n, budgets = [0], own, [b - own]
    raw, rawo = assemble(shape_t, leaves)
    sd = [start + d for d in slot_depths(shape_t, C["persist"])]
    H = {"depth": depth, "b": b, "n": n, "form": label, "relation": rel, "L": len(assign), "M": len(pool), "R": b - n,
         "rounding": rounding, "c0": c0, "slot_depths": sd, "lam_d": round(lam_d, 3), "alphabet_in": list(alphabet),
         "theta": {key: (round(v, 3) if isinstance(v, float) and not math.isinf(v) else ("∞" if isinstance(v, float) else v)) for key, v in th.items()},
         "src": src, "budgets": budgets}
    return {"H": H, "shape": shape_t, "pool": pool, "assign": assign, "leaves": leaves, "raw": raw, "rawo": rawo}

def split_budget(rng, R, uses):
    """Split R operators over the groups: group t gets the share v_t of R, v ~ Dirichlet(1), so each of its |t| copies
    gets x_t = R v_t / |t| operators, rounded up with probability equal to the fractional part (unbiased: the expected
    total is R exactly; an exact split does not always exist when every group has several slots).
    Returns per-copy budgets and the rounding error R - sum |t| b_t (negative: overshoot)."""
    k = len(uses)
    if R <= 0: return [0] * k, 0
    v = rng.dirichlet(np.ones(k)) if k > 1 else np.ones(1)
    bt = []
    for t in range(k):
        x = R * v[t] / uses[t]
        bt.append(int(math.floor(x)) + int(rng.random() < x - math.floor(x)))
    return bt, R - sum(bt[t] * uses[t] for t in range(k))

def term(rng, b, depth, parent, alphabet, counts, C, lam, c0, need_var, fc=None, no_comp=False):
    """One term with budget b. parent: the parent's hyperparameters (None at the root). alphabet: variables the parent
    and its ancestors use (inherited atoms draw from it)."""
    lam_d = lam ** depth if parent is not None else 0.0
    th, src = {}, {}
    for key, draw in root_prior(C).items():
        if parent is not None and key == "s_struct" and C.get("s_fixed"): th[key], src[key] = parent[key], "expression"
        elif parent is not None and key in parent and rng.random() < lam_d: th[key], src[key] = parent[key], "inherited"
        else: th[key], src[key] = draw(rng), ("root" if parent is None else "redrawn")
    start = c0 if C["persist"] else 0
    form = None
    if "s_struct" in th and b >= 1 and rng.random() < th["s_struct"]:
        if C.get("relations"): return relation_term(rng, b, depth, th, src, lam_d, alphabet, counts, C, lam, c0, fc, no_comp)
        form, shape_t = structural_form(rng, b, start, counts, fc)
        n = sum(1 for x in shape_t if x in ar); L = sum(1 for x in shape_t if x.startswith("@"))
    else:
        n = b if th["flat"] else (max(1, min(b, int(round(th["f"] * b)))) if b >= 1 else 0)
        q = q_bin(th["s_add"]) if "s_add" in th else None
        shape = sample_shape(n, rng, counts, q, c0=start, persist=C["persist"])
        shape_t, L = [], 0
        for x in shape:
            if x is None: shape_t.append(f"@{L}"); L += 1
            else: shape_t.append(num(x) if (x not in ar and x not in vset) else x)
    sd = [start + d for d in slot_depths(shape_t, C["persist"])] if n > 0 else [start]
    # 2. grouping: k groups, each takes one slot, the rest follow Dirichlet(alpha) weights (alpha = inf: uniform)
    k = 1 + int(rng.binomial(L - 1, 1 - th["rho"])) if L > 1 else 1
    k = min(k, L, MAX_KINDS)
    if math.isinf(th["alpha"]): extra = [int(x) for x in rng.integers(0, k, L - k)]
    else: extra = [int(x) for x in rng.choice(k, L - k, p=rng.dirichlet([th["alpha"]] * k))]
    assign = list(range(k)) + extra; rng.shuffle(assign)
    uses = [assign.count(t) for t in range(k)]
    # 1. budget split: groups with budget 0 are atoms
    R = b - n
    bt, unspent = split_budget(rng, R, uses)
    pool = [None] * k
    used, own_vars = set(), []
    def atom_kind():
        inherited = [x for x in alphabet if x not in used]
        if inherited and rng.random() < lam_d: x, how = str(rng.choice(inherited)), "parent's alphabet"
        else:
            fresh = [x for x in KINDS if x not in used] or KINDS
            x, how = str(rng.choice(fresh)), "fresh"
        used.add(x); return x, how
    atoms = [t for t in range(k) if bt[t] == 0]; comps = [t for t in range(k) if bt[t] > 0]
    for t in atoms:
        x, how = atom_kind()
        pool[t] = {"kind": "const" if x == "<c>" else "var", "val": x, "uses": uses[t], "src": how}
        if x != "<c>": own_vars.append(x)
    if need_var and not comps and atoms and not own_vars:
        t = atoms[0]; used.discard("<c>")
        cand = [x for x in alphabet if x not in used]
        x = str(rng.choice(cand)) if cand and rng.random() < lam_d else str(rng.choice([v for v in vs if v not in used]))
        pool[t] = {"kind": "var", "val": x, "uses": uses[t], "src": "forced: a term needs a variable"}; own_vars.append(x)
    child_alpha = list(dict.fromkeys(list(alphabet) + own_vars))
    for t in comps:
        cj = max(sd[i] for i in range(L) if assign[i] == t) if C["persist"] else 0
        sub = term(rng, bt[t], depth + 1, th, child_alpha, counts, C, lam, cj, True, fc)
        pool[t] = {"kind": "level", "tokens": sub["raw"], "sub": sub, "target": bt[t], "uses": uses[t], "c0": cj}
    # 3. copies: the first slot of a group holds the template; further slots hold copies (constants always redrawn)
    P = {"p_struct": 0.1 if C["mutate"] else 0.0, "p_exp": 0.5 if C["mutate"] else 0.0, "p_var_swap": 0.5, "p_shared_const": 0.0}
    gate = bool(th.get("gate", False))
    seen, leaves = set(), []
    for i in range(L):
        t = assign[i]; pt = pool[t]
        if pt["kind"] == "var": leaves.append({"pool": t, "tokens": [pt["val"]], "how": "var"})
        elif pt["kind"] == "const":
            c = num(smp._constant_literal(rng))
            if t not in seen: pt["val_first"] = c
            leaves.append({"pool": t, "tokens": [c], "how": "first" if t not in seen else "fresh"})
        elif t not in seen:
            leaves.append({"pool": t, "tokens": list(pt["tokens"]), "changes": [None] * len(pt["tokens"]), "how": "copy"})
        else:
            toks, ch = modify_traced(pt["tokens"], rng, gate, P, counts)
            leaves.append({"pool": t, "tokens": toks, "changes": ch, "how": "copy"})
        seen.add(t)
    for pt in pool:                       # a constant group shows its first value
        if pt["kind"] == "const": pt["val"] = pt.pop("val_first")
    raw, rawo = [], []
    for x in shape_t:
        if x.startswith("@"):
            lf = leaves[int(x[1:])]; raw += lf["tokens"]; rawo += [lf["pool"]] * len(lf["tokens"])
        else: raw.append(x); rawo.append(-1)
    H = {"depth": depth, "b": b, "n": n, "form": form or "LC tree", "L": L, "M": k, "R": R, "rounding": unspent, "c0": c0, "slot_depths": sd,
         "lam_d": round(lam_d, 3), "alphabet_in": list(alphabet),
         "theta": {key: (round(v, 3) if isinstance(v, float) and not math.isinf(v) else ("∞" if isinstance(v, float) else v)) for key, v in th.items()},
         "src": src, "budgets": bt}
    return {"H": H, "shape": shape_t, "pool": pool, "assign": assign, "leaves": leaves, "raw": raw, "rawo": rawo}

def depth_of(rec):
    return 1 + max([depth_of(q["sub"]) for q in rec["pool"] if q["kind"] == "level"] or [0])

def draw_r8(rng, C):
    rejected = collections.Counter()
    while True:
        B0 = int(rng.choice(T8_N, p=T8_P)); B = int(round(B0 * C["B_scale"]))
        lam = float(rng.beta(3, 2)) if C["lam"] else 0.0
        counts = np.zeros(len(URN["members"])) if C["urn"] else None
        fc = np.zeros(len(REL) if C.get("relations") else len(FORMS))
        rec = term(rng, B, 0, None, [], counts, C, lam, 0, False, fc)
        s, why = simplify_checked(rec["raw"])
        if s is None: rejected[why] += 1; continue
        if C["max_len"] and len(s) > C["max_len"]: rejected[f"longer than {C['max_len']} tokens"] += 1; continue
        return {"B": B, "lam": round(lam, 3), **rec, "simp": s, "levels": depth_of(rec), "rejected": dict(rejected)}
