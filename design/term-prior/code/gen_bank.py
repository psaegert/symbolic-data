"""Sample bank with full generative traces for the prior explorer page.
Priors: the T8 prior (LC, as delivered: 35-token cap, cancellation rejected) and the round-6 term-pool prior (n1 caps 4, 8)."""
import sys, json, math, collections, warnings
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

def sample_shape(n, rng, counts):
    """The T8 exact-size depth/width-prior draw; growing leaves are None, typed exponent slots are written."""
    C = smp._nesting_table(max(n, NS["nmax"])); D = len(C) - 1
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
        splits = np.array([w2 * C[0][i] * C[0][m - 1 - i] for i in range(m)]); pB = float(splits.sum())
        r = rng.random() * (pF + pS + pB)
        if r < pF: emit(function_label(rng, counts), [(min(c + 1, D), m - 1, None)])
        elif r < pF + pS: emit(str(rng.choice(NS["S"], p=NS["pS"])), [(c, m - 1, k)])
        else:
            i = int(rng.choice(m, p=splits / pB))
            if sticky and k is not None:
                pw = np.array([pb * (1.0 if kk == k else delta) for pb, kk in zip(smp.binary_operator_probs, NS["bin_class"])])
                op = str(rng.choice(smp.binary_operators, p=pw / pw.sum()))
            else:
                op = str(rng.choice(smp.binary_operators, p=smp.binary_operator_probs))
            k2 = NS["bin_class"][smp.binary_operators.index(op)] if sticky else None
            emit(op, [(0, i, k2), (0, m - 1 - i, k2)])
    grow(0, int(n), None)
    return out

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

# ---------------- T8 prior ----------------
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

# ---------------- term-pool prior, round 6 ----------------
P6 = dict(n0_max=8, rho_beta=(1, 1), alpha=(0.1, 10), r=(0, 1), kappa=(0, 0.5), p_new_vars_gate=0.5,
          p_var_swap=0.5, p_exp=0.5, p_shared_const=0.1, p_struct=0.1)
T8_ROWS = [json.loads(l) for l in open("../prior/draws.jsonl")]
P_CONST_LAW = float(np.mean([r["nv"] == 0 for r in T8_ROWS]))

def draw_pool(rng, cap):
    P = P6
    while True:
        H = {"cap": cap}
        H["B"] = int(rng.choice(T8_N, p=T8_P))
        H["n0"] = int(rng.integers(1, min(P["n0_max"], H["B"]) + 1)) if H["B"] >= 1 else 0
        H["rho"] = float(rng.beta(*P["rho_beta"]))
        H["alpha"] = float(math.exp(rng.uniform(math.log(P["alpha"][0]), math.log(P["alpha"][1]))))
        H["r"] = float(rng.uniform(*P["r"])); H["kappa"] = float(rng.uniform(*P["kappa"]))
        H["gate"] = bool(rng.random() < P["p_new_vars_gate"])
        if rng.random() < P_CONST_LAW:
            c = num(smp._constant_literal(rng))
            H.update(const_law=True)
            return {"prior": f"pool{cap}", "H": H, "shape": ["@0"], "pool": [{"kind": "const", "val": c, "uses": 1}],
                    "assign": [0], "leaves": [{"pool": 0, "tokens": [c], "how": "first"}], "raw": [c], "rawo": [0],
                    "simp": [c], "redundant": [], "rejected": {}}
        counts = np.zeros(len(URN["members"]))
        shape = sample_shape(H["n0"], rng, counts) if H["n0"] > 0 else [None]
        shape_t, L = [], 0
        for x in shape:
            if x is None: shape_t.append(f"@{L}"); L += 1
            else: shape_t.append(num(x) if (x not in ar and x not in vset) else x)
        A0 = lc_alphabet(L, rng)
        u = rng.random()
        H["case"] = ("tau0-full" if rng.random() < 0.5 else "tau0-subset") if u < 0.5 else ("shared-L1" if u < 0.75 else "own")
        M = 1 + int(rng.binomial(L - 1, 1 - H["rho"])) if L > 1 else 1
        w = rng.dirichlet([H["alpha"]] * M)
        assign = list(range(M)) + [int(x) for x in rng.choice(M, L - M, p=w)]; rng.shuffle(assign)
        uses = collections.Counter(assign)
        R = max(H["B"] - H["n0"], 0)
        r_eff = H["r"] + (1 - H["r"]) * (R / H["B"] if H["B"] else 0.0)
        H.update(L=L, M=M, R=R, r_eff=r_eff, A0=A0)
        kinds = ["T" if rng.random() < r_eff else "a" for _ in range(M)]
        nT = kinds.count("T"); var_bearing = nT > 0
        shapes, n1s = {}, {}
        for j in range(M):
            if kinds[j] == "T":
                target = R / (nT * uses[j]) if nT else 1
                n1 = int(np.clip(round(target) + rng.integers(-1, 2), 1, cap))
                shapes[j] = sample_shape(n1, rng, counts); n1s[j] = (n1, target)
        total_leaves = sum(sum(1 for x in sh if x is None) for sh in shapes.values())
        A1 = lc_alphabet(total_leaves, rng) if H["case"] == "shared-L1" else None
        H["A1"] = A1
        pool, a0_cycle = [None] * M, [str(v) for v in rng.permutation(A0)]
        for j in [int(x) for x in rng.permutation(M)]:
            if kinds[j] == "T":
                sh = shapes[j]
                if H["case"] == "own": alph, src = list(vs), "all 17 variables (LC's own draw)"
                elif H["case"] == "shared-L1": alph, src = A1, "shared level-1 alphabet"
                elif H["case"] == "tau0-full": alph, src = A0, "tau0's alphabet"
                else:
                    alph = [str(v) for v in rng.choice(A0, int(rng.integers(1, len(A0) + 1)), replace=False)]; src = "subset of tau0's alphabet"
                toks, drawn = fill_leaves(sh, alph, rng, need_var=True)
                pool[j] = {"kind": "term", "tokens": toks, "n1": n1s[j][0], "target": round(n1s[j][1], 2),
                           "alphabet": alph if H["case"] != "own" else None, "alph_src": src, "drawn": drawn, "uses": uses[j]}
            elif var_bearing and rng.random() < H["kappa"]:
                pool[j] = {"kind": "const", "val": num(smp._constant_literal(rng)), "uses": uses[j]}
            else:
                if not a0_cycle: a0_cycle = [str(v) for v in rng.permutation(A0)]
                pool[j] = {"kind": "var", "val": a0_cycle.pop(), "uses": uses[j]}; var_bearing = True
        seen, leaves = set(), []
        for i in range(L):
            j = assign[i]; pt = pool[j]
            if pt["kind"] == "var":
                leaves.append({"pool": j, "tokens": [pt["val"]], "how": "var"})
            elif pt["kind"] == "const":
                if j not in seen: leaves.append({"pool": j, "tokens": [pt["val"]], "how": "first"})
                elif rng.random() < P["p_shared_const"]: leaves.append({"pool": j, "tokens": [pt["val"]], "how": "shared"})
                else: leaves.append({"pool": j, "tokens": [num(smp._constant_literal(rng))], "how": "fresh"})
            else:
                toks, ch = modify_traced(pt["tokens"], rng, H["gate"], P, counts)
                leaves.append({"pool": j, "tokens": toks, "changes": ch, "how": "copy"})
            seen.add(j)
        raw, rawo = [], []
        for x in shape_t:
            if x.startswith("@"):
                lf = leaves[int(x[1:])]; raw += lf["tokens"]; rawo += [lf["pool"]] * len(lf["tokens"])
            else:
                raw.append(x); rawo.append(-1)
        s, why = simplify_checked(raw)
        if s is None: continue
        # copy merges: a copy that simplifies together with an earlier copy of the same term in the same sum / product
        t0 = tree_of(shape_t); paths = leaf_paths(t0)
        groups = collections.defaultdict(list)
        for i, p in enumerate(paths):
            if leaves[i]["how"] != "copy": continue
            b = bag(t0, p)
            groups[(leaves[i]["pool"], b[0] if b else ("solo", p), b[1] if b else None)].append((i, b[2] if b else 0))
        redundant = []
        for key, mem in groups.items():
            if key[2] is None or len(mem) < 2: continue
            kept = []
            for i, par in mem:
                ci = leaves[i]["tokens"]; li_ = slen(ci); merged = False
                for i2, par2 in kept:
                    c2 = leaves[i2]["tokens"]; l2 = slen(c2)
                    op = ("+" if par == par2 else "-") if key[2] == "A" else ("*" if par == par2 else "/")
                    lj = slen([op] + c2 + ci)
                    if None not in (li_, l2, lj) and lj < li_ + l2 + 1: merged = True; redundant.append([i, i2]); break
                if not merged: kept.append((i, par))
        H["rho"] = round(H["rho"], 3); H["alpha"] = round(H["alpha"], 3); H["r"] = round(H["r"], 3)
        H["kappa"] = round(H["kappa"], 3); H["r_eff"] = round(H["r_eff"], 3)
        return {"prior": f"pool{cap}", "H": H, "shape": shape_t, "pool": pool, "assign": assign, "leaves": leaves,
                "raw": raw, "rawo": rawo, "simp": s, "redundant": redundant, "rejected": {}}

N = int(sys.argv[1]); rng = np.random.default_rng(20261120)
bank = []
for i in range(N): bank.append(draw_t8(rng))
for cap in (4, 8):
    for i in range(N): bank.append(draw_pool(rng, cap))
print("drawn", len(bank), flush=True)
cfg2 = yaml.safe_load(open(CFG)); cfg2["holdout_family"] = "precise"
hcat = LampleChartonCatalog.from_config(cfg2); heng = hcat.simplipy_engine
for k, rec in enumerate(bank):
    s = rec["simp"]
    try:
        _, c = explicit_constant_placeholders(heng.operators_to_realizations(s), inplace=False, convert_numbers_to_constant=False)
        rec["held"] = bool(hcat.is_held_out(s, c, assume_canonical=hcat._skeleton_is_holdout_canonical))
    except Exception:
        rec["held"] = True
    rec["cls"] = record_classes(s)
    rec["id"] = k
with open("samples.json", "w") as f:
    f.write("[\n" + ",\n".join(json.dumps(r, separators=(",", ":")) for r in bank) + "\n]\n")
print("DONE", flush=True)
