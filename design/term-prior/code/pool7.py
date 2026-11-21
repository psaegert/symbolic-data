"""Sample bank with full generative traces for the prior explorer page.
Priors: the T8 prior (LC, as delivered: 35-token cap, cancellation rejected) and the round-6 term-pool prior (n1 caps 4, 8)."""
import sys, json, math, collections, warnings, os
warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/psaegert/Projects/symbolic-data/.claude/worktrees/holdout-precise/src")
import numpy as np, yaml
from symbolic_data import LampleChartonCatalog
from simplipy.utils import explicit_constant_placeholders
from classify import classify, parse, terms, motif, has_var

CFG = "/home/psaegert/Projects/flash-ansr/configs/v25.0-T8.1-120M/catalog_train.yaml"
FAMILIES = {"abs": ["abs"], "exp": ["exp"], "hyperbolic": ["sinh", "cosh", "tanh"], "inverse_hyperbolic": ["asinh", "acosh", "atanh"],
            "inverse_trig": ["asin", "acos", "atan"], "log": ["log"], "trig": ["sin", "cos", "tan"]}
FAMILY_OF = {op: f for f, ops in FAMILIES.items() for op in ops}
cfg = yaml.safe_load(open(CFG)); cfg["holdout_pools"] = []
import os
if os.environ.get("POOL7_FN_WEIGHTS") == "corpus":
    # Function weights from the 11,668 real formulas (family-affinity corpus), add-one smoothed and tempered with a
    # square root; the total function weight stays T8's (each of the 14 functions had weight 1), so tree shapes and the
    # unary/binary balance are unchanged and only WHICH function fills a node moves.
    _c = collections.Counter()
    for _f in ("corpus_ops_wikidata.jsonl", "corpus_ops_wikipedia.jsonl"):
        for _l in open("/home/psaegert/Projects/research/archive/symbolic-data/2026-11-03-family-coherence-corpus/t83/corpus/" + _f):
            for _k, _v in json.loads(_l).get("ops", {}).items(): _c[_k] += _v
    _fns = ["sin", "cos", "tan", "asin", "acos", "atan", "sinh", "cosh", "tanh", "asinh", "acosh", "atanh", "exp", "log", "abs"]
    _fns = [f for f in _fns if cfg["operator_weights"].get(f, 0) > 0]
    _tot = sum(cfg["operator_weights"][f] for f in _fns)
    _raw = {f: (_c[f] + 1) ** 0.5 for f in _fns}
    for f in _fns: cfg["operator_weights"][f] = _tot * _raw[f] / sum(_raw.values())
_mode = os.environ.get("POOL7_FN_WEIGHTS", "")
if _mode.startswith("hyper="):
    # Owner 2026-11-20: hyperbolic functions less likely in general. The six hyperbolic functions get the given weight
    # (T8: 1); every other weight stays, so a function node's label shifts toward all other functions proportionally.
    for _f in ("sinh", "cosh", "tanh", "asinh", "acosh", "atanh"): cfg["operator_weights"][_f] = float(_mode.split("=")[1])
FN_WEIGHTS = {f: cfg["operator_weights"][f] for f in ("sin", "cos", "tan", "asin", "acos", "atan", "sinh", "cosh", "tanh", "asinh", "acosh", "atanh", "exp", "log", "abs") if f in cfg["operator_weights"]}
cfg["family_affinity"] = {"concentration": 0.36, "families": FAMILIES, "independent": ["pow", "rootn"]}
cat = LampleChartonCatalog.from_config(cfg)
eng = cat.simplipy_engine; ar = dict(eng.operator_arity); vs = list(cat.variables); vset = set(vs)
smp = cat.skeleton_sampler; slot_ops = set(smp.typed_slots); NS = smp._nesting; URN = NS["affinity"]
UNARY = [o for o in smp.unary_operators if ar[o] == 1]
BINARY = [o for o in smp.binary_operators if ar[o] == 2 and o not in slot_ops]
SS = cat.sample_strategy
T8_N = np.arange(SS["min_operators"], SS["max_operators"] + 1)
T8_P = np.exp(T8_N ** SS["power"] / SS["lambda"]); T8_P /= T8_P.sum()
ADD, MUL = {"+", "-"}, {"*", "/"}

def num(v):
    """Compact literal for the page (full precision is not needed to read the trace)."""
    try:
        f = float(v)
        return str(int(f)) if f.is_integer() else f"{f:.6g}"
    except Exception:
        return v

def function_label(rng, counts):
    if counts is None:
        return str(rng.choice(NS["F"], p=NS["pF"]))
    if URN["independent"] and rng.random() < URN["p_independent"]:
        return str(rng.choice(URN["independent"], p=URN["within_independent"]))
    p = URN["alpha"] * URN["pi"] + counts
    f = int(rng.choice(len(p), p=p / p.sum())); counts[f] += 1
    m = URN["members"][f]
    return m[0] if len(m) == 1 else str(rng.choice(m, p=URN["within"][f]))

def q_bin(s_add):
    """Binary label probabilities for an additive share s_add: + and - share s_add, * and / share 1 - s_add."""
    return np.array([s_add / 2 if op in ("+", "-") else (1 - s_add) / 2 for op in smp.binary_operators])

_PERSIST = {}
def persist_table(nmax):
    """Count table for the depth prior when a binary node does NOT reset the function depth (every function above a
    node counts, not only an unbroken chain): C[c][m] = w1 (mF g^c C[c+1][m-1] + mS C[c][m-1]) + w2 sum C[c][i] C[c][m-1-i]."""
    if _PERSIST.get("nmax", -1) >= nmax: return _PERSIST["C"]
    w1, w2, g, mF, mS = smp._n_unary_operators, smp._n_binary_operators, NS["gamma"], NS["mF"], NS["mS"]
    D = nmax + 2
    C = [[1.0] + [0.0] * nmax for _ in range(D + 1)]
    for m in range(1, nmax + 1):
        for c in range(D, -1, -1):
            binary = w2 * sum(C[c][i] * C[c][m - 1 - i] for i in range(m))
            C[c][m] = w1 * (mF * g ** c * C[min(c + 1, D)][m - 1] + mS * C[c][m - 1]) + binary
    _PERSIST.update(nmax=nmax, C=C)
    return C

def sample_shape(n, rng, counts, q=None, c0=0, persist=False):
    """The T8 exact-size depth/width-prior draw; growing leaves are None, typed exponent slots are written.
    q: per-level binary label probabilities (None = the catalog's). c0: the function depth the tree starts at (a pool
    term placed under functions). persist: a binary node keeps the function depth instead of resetting it."""
    C = persist_table(max(n, 24)) if persist else smp._nesting_table(max(n, NS["nmax"])); D = len(C) - 1
    w1, w2, g, delta = smp._n_unary_operators, smp._n_binary_operators, NS["gamma"], NS["delta"]
    sticky = delta < 1.0; out = []
    def emit(op, children):
        spec = smp.typed_slots.get(op); out.append(op); it = iter(children)
        for a in range(ar[op]):
            if spec is not None and a == spec["argument"]: out.append(smp._slot_literal(op, rng))
            else: grow(*next(it))
    def grow(c, m, k):
        if m == 0: out.append(None); return
        pF = w1 * NS["mF"] * g ** c * C[min(c + 1, D)][m - 1] if NS["F"] else 0.0
        pS = w1 * NS["mS"] * C[c][m - 1] if NS["S"] else 0.0
        cb = c if persist else 0
        splits = np.array([w2 * C[cb][i] * C[cb][m - 1 - i] for i in range(m)]); pB = float(splits.sum())
        r = rng.random() * (pF + pS + pB)
        if r < pF: emit(function_label(rng, counts), [(min(c + 1, D), m - 1, None)])
        elif r < pF + pS: emit(str(rng.choice(NS["S"], p=NS["pS"])), [(c, m - 1, k)])
        else:
            i = int(rng.choice(m, p=splits / pB))
            base = smp.binary_operator_probs if q is None else q
            if sticky and k is not None:
                pw = np.array([pb * (1.0 if kk == k else delta) for pb, kk in zip(base, NS["bin_class"])])
                pw = pw if pw.sum() > 0 else np.ones_like(pw)
                op = str(rng.choice(smp.binary_operators, p=pw / pw.sum()))
            else:
                op = str(rng.choice(smp.binary_operators, p=base / base.sum()))
            k2 = NS["bin_class"][smp.binary_operators.index(op)] if sticky else None
            emit(op, [(cb, i, k2), (cb, m - 1 - i, k2)])
    grow(min(int(c0), D), int(n), None)
    return out

def slot_depths(shape_t, persist):
    """Function depth at every leaf slot @i of a shape: the counted functions above it (all of them with persist,
    only the unbroken chain directly above without). neg / inv are transparent, as in the depth prior."""
    F = set(NS["F"]); out = {}; pos = 0
    def walk(c):
        nonlocal pos
        t = shape_t[pos]; pos += 1
        if t.startswith("@"): out[int(t[1:])] = c; return
        a = ar.get(t, 0)
        for k in range(a):
            if t in slot_ops and k == smp.typed_slots[t]["argument"]: pos += 1; continue
            if t in F: walk(c + 1)
            elif a == 1: walk(c)
            else: walk(c if persist else 0)
    walk(0)
    return [out[i] for i in range(len(out))]

def slot_mask(tokens):
    """True at the token positions that are typed exponent / root-index slots."""
    mask = [False] * len(tokens); pos = 0
    def walk(is_slot):
        nonlocal pos
        i = pos; t = tokens[pos]; pos += 1; mask[i] = is_slot
        a = ar.get(t, 0) if t is not None else 0
        for k in range(a):
            walk(t in slot_ops and k == smp.typed_slots[t]["argument"])
    walk(False)
    return mask

def fill_leaves(shape, alphabet, rng, need_var):
    """LC's leaf rule over an alphabet plus the constant slot. Returns tokens and the kinds drawn."""
    t = sum(1 for x in shape if x is None)
    kinds = list(alphabet) + ["<c>"]
    k = int(rng.integers(1, min(t, len(kinds)) + 1))
    uniq = [str(x) for x in rng.choice(kinds, k, replace=False)]
    if need_var and all(x == "<c>" for x in uniq):
        uniq = [str(rng.choice(list(alphabet)))]
    leaves = uniq + [str(x) for x in rng.choice(uniq, t - len(uniq), replace=True)]; rng.shuffle(leaves)
    out = list(shape); it = iter(leaves)
    for i in range(len(out)):
        if out[i] is None:
            x = next(it)
            out[i] = num(smp._constant_literal(rng)) if x == "<c>" else x
    return [num(x) if (x not in ar and x not in vset) else x for x in out], uniq

def lc_alphabet(n_leaves, rng):
    k = int(rng.integers(1, min(max(n_leaves, 1), len(vs)) + 1))
    return [str(v) for v in rng.choice(vs, k, replace=False)]

def modify_traced(tokens, rng, gate, P, counts):
    """One use of a pool term. Same changes as the simulations; returns tokens and a per-token change list."""
    out = list(tokens); ch = [None] * len(out); mask = slot_mask(out)
    if rng.random() < P["p_struct"]:
        ops = [i for i, t in enumerate(out) if t in ar and t not in slot_ops]
        if ops:
            i = ops[int(rng.integers(len(ops)))]; old = out[i]
            if ar[old] == 1:
                new = function_label(rng, counts) if old in FAMILY_OF else str(rng.choice(UNARY))
                if ar.get(new, 1) != 1: new = old
            else:
                new = str(rng.choice(BINARY))
            if new != old: out[i] = new; ch[i] = ["op", old]
    for i, t in enumerate(out):
        if t in ar: continue
        if mask[i]:
            if rng.random() < P["p_exp"]:
                new = num(smp._slot_literal(_slot_owner(out, i), rng))
                if new != t: out[i] = new; ch[i] = ["exp", t]
        elif t in vset:
            if gate and rng.random() < P["p_var_swap"]:
                new = str(rng.choice(vs))
                if new != t: out[i] = new; ch[i] = ["var", t]
        else:
            if rng.random() < P["p_shared_const"]:
                ch[i] = ["shared", t]
            else:
                out[i] = num(smp._constant_literal(rng)); ch[i] = ["fresh", t]
    return out, ch

def _slot_owner(tokens, i):
    """The pow/rootn whose slot sits at position i (the operator two positions before a single-token base, etc.)."""
    pos = 0; owner = {}
    def walk():
        nonlocal pos
        j = pos; t = tokens[pos]; pos += 1
        for k in range(ar.get(t, 0)):
            if t in slot_ops and k == smp.typed_slots[t]["argument"]: owner[pos] = t
            walk()
    walk()
    return owner[i]

def tree_of(tokens):
    it = iter(tokens)
    def rec():
        t = next(it); a = ar.get(t, 0)
        return [t] + [rec() for _ in range(a)] if a else t
    return rec()
def flat(n): return [n] if isinstance(n, str) else [n[0]] + [t for c in n[1:] for t in flat(c)]
def get(n, p):
    for i in p: n = n[i]
    return n

def leaf_paths(n, path=()):
    if isinstance(n, str): return [path] if n.startswith("@") else []
    out = []
    for i, c in enumerate(n[1:]): out += leaf_paths(c, path + (i + 1,))
    return out

def bag(t, p):
    if not p: return None
    parent = get(t, p[:-1])[0]
    cls = "A" if parent in ADD | {"neg"} else "M" if parent in MUL | {"inv"} else None
    if cls is None: return None
    S, flip = (ADD | {"neg"}, ("-", "neg")) if cls == "A" else (MUL | {"inv"}, ("/", "inv"))
    i, parity = len(p), 0
    while i > 0 and get(t, p[:i - 1])[0] in S:
        node = get(t, p[:i - 1])
        if node[0] == flip[1] or (node[0] == flip[0] and p[i - 1] == 2): parity ^= 1
        i -= 1
    return (p[:i], cls, parity)

def slen(tokens):
    try: return len(eng.simplify(tokens, mode=cat.simplify_mode))
    except Exception: return None

def record_classes(s):
    c = classify(s, ar, vset)
    T = [t for t in terms(parse(s, ar)) if has_var(t, vset)]
    ms = collections.Counter(motif(t, vset) for t in T if motif(t, vset) != "V")
    return {"nv": c["nv"], "k": c["k_terms"], "poly": c["poly"], "poly3": c["poly_ge3"], "poly1v3": c["poly_1v_ge3"],
            "rat2": c["sum_rat_ge2"], "aff": c["affine_ge2v"], "sup3": c["superpos_ge3"],
            "rep": bool(ms) and ms.most_common(1)[0][1] >= 2, "trans": c["transcendental"],
            "fam": sorted({FAMILY_OF[x] for x in s if x in FAMILY_OF})}

def simplify_checked(raw):
    had = any(x in vset for x in raw)
    try: s = [num(x) if (x not in ar and x not in vset) else x for x in eng.simplify(raw, mode=cat.simplify_mode)]
    except Exception: return None, "simplification failed"
    if had and not any(x in vset for x in s): return None, "every variable cancelled"
    if any("inf" in x or "nan" in x for x in s): return None, "non-finite literal"
    return s, None


# ---------------- T8 prior (traced) ----------------

def draw_t8(rng):
    rejected = collections.Counter()
    while True:
        n = int(rng.choice(T8_N, p=T8_P))
        shape = sample_shape(n, rng, None)
        t = sum(1 for x in shape if x is None)
        cap = min(t, len(vs)); k = int(rng.integers(1, cap + 1))
        kinds = [str(x) for x in rng.choice(vs + ["<c>"], k, replace=False)]
        leaves = kinds + [str(x) for x in rng.choice(kinds, t - k, replace=True)]; rng.shuffle(leaves)
        vals = [num(smp._constant_literal(rng)) if x == "<c>" else x for x in leaves]
        shape_t, li = [], 0
        for x in shape:
            if x is None: shape_t.append(f"@{li}"); li += 1
            else: shape_t.append(num(x) if (x not in ar and x not in vset) else x)
        raw = [vals[int(x[1:])] if isinstance(x, str) and x.startswith("@") else x for x in shape_t]
        s, why = simplify_checked(raw)
        if s is None: rejected[why] += 1; continue
        if len(s) > SS["max_length"]: rejected["longer than 35 tokens"] += 1; continue
        return {"prior": "t8", "H": {"n_ops": n, "leaves": t, "n_kinds": k, "kinds": kinds},
                "shape": shape_t, "leafvals": vals, "raw": raw, "simp": s, "rejected": dict(rejected)}


# ---------------- round 7: recursive hierarchical term pool ----------------
P_CONST_LAW = 0.003   # T8's constant-law rate (measured on 20k raw T8.1 draws: 0.3%)
BRANCHES = ["parent-full", "parent-subset", "shared", "own"]
ROOT = {
    "s_add": lambda rng: float(rng.beta(0.5, 0.5)),            # additive share of binary labels (U-shaped: often pure)
    "f": lambda rng: float(rng.beta(2, 3)),                    # share of the level budget for the level's own tree
    "rho": lambda rng: float(rng.beta(2, 1)),                  # reuse level (round 6: U(0,1))
    "alpha": lambda rng: float(math.exp(rng.uniform(math.log(0.1), math.log(10)))),
    "r": lambda rng: float(rng.uniform()),
    "kappa": lambda rng: float(rng.uniform(0, 0.5)),
    "gate": lambda rng: bool(rng.random() < 0.5),
    "branch": lambda rng: str(rng.choice(BRANCHES)),
}
P7 = dict(lam={1: 0.6, 2: 0.35, 3: 0.2}, max_levels=3, persist=True, n_cap=8, leaf_cap=4, mess_cap=8, p_mess=0.2, B_scale=1.0,
          p_var_swap=0.5, p_exp=0.5, p_shared_const=0.1, p_struct=0.1)

def level_params(rng, parent, depth, P):
    """Root: every hyperparameter from its root prior. Depth d: copy the parent's value with probability lam[d],
    otherwise redraw from the root prior -- so every level's marginal is the root prior."""
    th, src = {}, {}
    for k, f in ROOT.items():
        if parent is not None and rng.random() < P["lam"][depth]:
            th[k], src[k] = parent[k], "inherited"
        else:
            th[k], src[k] = f(rng), ("root" if parent is None else "redrawn")
    return th, src

def lc_term(rng, n1, alphabet, counts, s_add, c0=0, persist=False):
    shape = sample_shape(n1, rng, counts, q_bin(s_add), c0=c0, persist=persist)
    toks, drawn = fill_leaves(shape, alphabet, rng, need_var=True)
    return toks, drawn

def sample_level(rng, b, depth, parent_theta, A_in, counts, P, c0=0):
    th, src = level_params(rng, parent_theta, depth, P)
    n = int(np.clip(round(th["f"] * b), 1, P["n_cap"])) if b >= 1 else 0
    cross = P.get("cross_level", True)
    shape = sample_shape(n, rng, counts, q_bin(th["s_add"]), c0=c0 if cross else 0, persist=P["persist"]) if n > 0 else [None]
    shape_t, L = [], 0
    for x in shape:
        if x is None: shape_t.append(f"@{L}"); L += 1
        else: shape_t.append(num(x) if (x not in ar and x not in vset) else x)
    A = list(A_in) if A_in else lc_alphabet(L, rng)
    sd = [(c0 if cross else 0) + d for d in slot_depths(shape_t, P["persist"])] if n > 0 else [c0 if cross else 0]
    M = 1 + int(rng.binomial(L - 1, 1 - th["rho"])) if L > 1 else 1
    w = rng.dirichlet([th["alpha"]] * M)
    assign = list(range(M)) + [int(x) for x in rng.choice(M, L - M, p=w)]; rng.shuffle(assign)
    uses = collections.Counter(assign)
    R = max(b - n, 0)
    r_eff = th["r"] + (1 - th["r"]) * (R / b if b else 0.0)
    kinds = ["T" if rng.random() < r_eff else "a" for _ in range(M)]
    nT = kinds.count("T"); var_bearing = nT > 0
    A_sh = [str(v) for v in rng.choice(vs, int(rng.integers(1, min(len(vs), max(1, round(R)) + 1) + 1)), replace=False)] if th["branch"] == "shared" else None
    pool, a_cycle = [None] * M, [str(v) for v in rng.permutation(A)]
    for j in [int(x) for x in rng.permutation(M)]:
        if kinds[j] == "T":
            bj = R / (nT * uses[j]) if nT else 1.0
            if th["branch"] == "parent-full": alph, asrc = A, "this level's alphabet"
            elif th["branch"] == "parent-subset":
                alph = [str(v) for v in rng.choice(A, int(rng.integers(1, len(A) + 1)), replace=False)]; asrc = "subset of this level's alphabet"
            elif th["branch"] == "shared": alph, asrc = A_sh, "shared alphabet of this level's terms"
            else: alph, asrc = None, "own draw"
            child_th, child_src = level_params(rng, th, depth + 1, P)
            cj = max(sd[i] for i in range(L) if assign[i] == j) if cross else 0
            if bj <= P["leaf_cap"] or depth + 1 >= P["max_levels"]:
                n1 = int(np.clip(round(bj) + rng.integers(-1, 2), 1, P["leaf_cap"]))
                toks, drawn = lc_term(rng, n1, alph or list(vs), counts, child_th["s_add"], cj, P["persist"])
                pool[j] = {"kind": "term", "tokens": toks, "n1": n1, "target": round(bj, 2), "alphabet": alph, "alph_src": asrc, "c0": cj,
                           "uses": uses[j], "s_add": round(child_th["s_add"], 3), "s_src": child_src["s_add"]}
            elif rng.random() < P["p_mess"]:
                n1 = int(np.clip(round(bj), 1, P["mess_cap"]))
                toks, drawn = lc_term(rng, n1, alph or list(vs), counts, child_th["s_add"], cj, P["persist"])
                pool[j] = {"kind": "term", "mess": True, "c0": cj, "tokens": toks, "n1": n1, "target": round(bj, 2), "alphabet": alph,
                           "alph_src": asrc, "uses": uses[j], "s_add": round(child_th["s_add"], 3), "s_src": child_src["s_add"]}
            else:
                sub = sample_level(rng, bj, depth + 1, th, alph, counts, P, c0=cj)
                pool[j] = {"kind": "level", "c0": cj, "tokens": sub["raw"], "sub": sub, "target": round(bj, 2), "alph_src": asrc, "uses": uses[j]}
        elif var_bearing and rng.random() < th["kappa"]:
            pool[j] = {"kind": "const", "val": num(smp._constant_literal(rng)), "uses": uses[j]}
        else:
            if not a_cycle: a_cycle = [str(v) for v in rng.permutation(A)]
            pool[j] = {"kind": "var", "val": a_cycle.pop(), "uses": uses[j]}; var_bearing = True
    seen, leaves = set(), []
    for i in range(L):
        j = assign[i]; pt = pool[j]
        if pt["kind"] == "var": leaves.append({"pool": j, "tokens": [pt["val"]], "how": "var"})
        elif pt["kind"] == "const":
            if j not in seen: leaves.append({"pool": j, "tokens": [pt["val"]], "how": "first"})
            elif rng.random() < P["p_shared_const"]: leaves.append({"pool": j, "tokens": [pt["val"]], "how": "shared"})
            else: leaves.append({"pool": j, "tokens": [num(smp._constant_literal(rng))], "how": "fresh"})
        else:
            toks, ch = modify_traced(pt["tokens"], rng, th["gate"], P, counts)
            leaves.append({"pool": j, "tokens": toks, "changes": ch, "how": "copy"})
        seen.add(j)
    raw, rawo = [], []
    for x in shape_t:
        if x.startswith("@"):
            lf = leaves[int(x[1:])]; raw += lf["tokens"]; rawo += [lf["pool"]] * len(lf["tokens"])
        else:
            raw.append(x); rawo.append(-1)
    H = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in th.items()}
    H.update(src=src, depth=depth, c0=c0, slot_depths=sd, b=round(b, 2), n=n, L=L, M=M, R=round(R, 2), r_eff=round(r_eff, 3), A=A, A_sh=A_sh)
    return {"H": H, "shape": shape_t, "pool": pool, "assign": assign, "leaves": leaves, "raw": raw, "rawo": rawo}

def draw_r7(rng, P=P7):
    rejected = collections.Counter()
    while True:
        B0 = int(rng.choice(T8_N, p=T8_P)); B = int(round(B0 * P["B_scale"]))
        if rng.random() < P_CONST_LAW:
            c = num(smp._constant_literal(rng))
            lvl = {"H": {"const_law": True, "B": B}, "shape": ["@0"], "pool": [{"kind": "const", "val": c, "uses": 1}],
                   "assign": [0], "leaves": [{"pool": 0, "tokens": [c], "how": "first"}], "raw": [c], "rawo": [0]}
            return {"prior": "r7", "B": B, **lvl, "simp": [c], "rejected": dict(rejected)}
        counts = np.zeros(len(URN["members"]))
        lvl = sample_level(rng, B, 0, None, None, counts, P)
        s, why = simplify_checked(lvl["raw"])
        if s is None: rejected[why] += 1; continue
        return {"prior": "r7", "B": B, **lvl, "simp": s, "rejected": dict(rejected)}

def levels_used(lvl, depth=0):
    deepest = depth
    for p in lvl["pool"]:
        if p["kind"] == "level": deepest = max(deepest, levels_used(p["sub"], depth + 1))
    return deepest
