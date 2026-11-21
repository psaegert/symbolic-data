"""Hierarchical term-pool prior, round 4: size budget, shared family urn, copy-merge accounting."""
import sys, json, warnings, collections, math
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
ADD, MUL = {"+", "-"}, {"*", "/"}

def function_label(rng, counts):
    """The depth prior's counted-unary label; with counts it is the family urn shared by the whole expression."""
    if counts is None:
        return str(rng.choice(NS["F"], p=NS["pF"]))
    if URN["independent"] and rng.random() < URN["p_independent"]:
        return str(rng.choice(URN["independent"], p=URN["within_independent"]))
    p = URN["alpha"] * URN["pi"] + counts
    f = int(rng.choice(len(p), p=p / p.sum())); counts[f] += 1
    m = URN["members"][f]
    return m[0] if len(m) == 1 else str(rng.choice(m, p=URN["within"][f]))

def sample_shared(n, rng, counts):
    """SkeletonSampler._sample_nesting + leaf fill, with the family urn counts passed in (None = independent)."""
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
    leaves = smp._get_leaves(t_leaves=sum(1 for t in out if t is None), rng=rng)
    for i in range(len(out) - 1, -1, -1):
        if out[i] is None: out[i] = leaves.pop()
    return out

def tree(tokens):
    it = iter(tokens)
    def rec():
        t = next(it); a = ar.get(t, 0)
        return [t] + [rec() for _ in range(a)] if a else t
    return rec()
def flat(n): return [n] if isinstance(n, str) else [n[0]] + [t for c in n[1:] for t in flat(c)]
def is_slot(op, i): return op in slot_ops and i == smp.typed_slots[op]["argument"]
def growing_leaf_paths(n, path=()):
    if isinstance(n, str): return [path]
    out = []
    for i, c in enumerate(n[1:]):
        if not is_slot(n[0], i): out += growing_leaf_paths(c, path + (i + 1,))
    return out
def op_paths(n, path=()):
    if isinstance(n, str): return []
    out = [path]
    for i, c in enumerate(n[1:]): out += op_paths(c, path + (i + 1,))
    return out
def get(n, p):
    for i in p: n = n[i]
    return n
def put(n, p, v):
    if not p: return v
    get(n, p[:-1])[p[-1]] = v; return n

def modify(t, rng, H, P, counts):
    c = json.loads(json.dumps(t)); flags = {"var": False, "exp": False, "struct": False}
    if rng.random() < P["p_struct"]:
        ops = [p for p in op_paths(c) if get(c, p)[0] not in slot_ops]
        if ops:
            node = get(c, ops[rng.integers(len(ops))]); old = node[0]
            if ar[old] == 1:
                node[0] = function_label(rng, counts) if (counts is not None and old in FAMILY_OF) else str(rng.choice(UNARY))
                if ar.get(node[0], 1) != 1: node[0] = old       # the urn may return pow/rootn (independent): keep the old label
            else:
                node[0] = str(rng.choice(BINARY))
            flags["struct"] = node[0] != old
    def walk(n, slot_op=None):
        if isinstance(n, str):
            if slot_op is not None:
                if rng.random() < P["p_exp"]:
                    new = smp._slot_literal(slot_op, rng); flags["exp"] |= new != n; return new
                return n
            if n in vset:
                if H["new_vars"] and rng.random() < P["p_var_swap"]:
                    new = str(rng.choice(vs)); flags["var"] |= new != n; return new
                return n
            return n if rng.random() < P["p_shared_const"] else smp._constant_literal(rng)
        return [n[0]] + [walk(ch, n[0] if is_slot(n[0], i) else None) for i, ch in enumerate(n[1:])]
    return walk(c), flags

def draw(rng, P):
    H = {"n0": int(rng.integers(P["n0"][0], P["n0"][1] + 1))}
    H["B"] = int(rng.integers(P["B"][0], P["B"][1] + 1)) if P.get("B") else None
    H["rho"] = float(rng.beta(*P["rho_beta"]))
    H["alpha"] = float(math.exp(rng.uniform(math.log(P["alpha"][0]), math.log(P["alpha"][1]))))
    H["r"] = float(rng.uniform(*P["r"])); H["kappa"] = float(rng.uniform(*P["kappa"]))
    H["new_vars"] = bool(rng.random() < P["p_new_vars_gate"])
    counts = np.zeros(len(URN["members"])) if P["shared_urn"] else None
    t0 = tree(sample_shared(H["n0"], rng, counts))
    paths = growing_leaf_paths(t0); L = len(paths)
    M = 1 + int(rng.binomial(L - 1, 1 - H["rho"])) if L > 1 else 1
    pool, used, kinds, n1s = [], set(), [], []
    for _ in range(M):
        if rng.random() < H["r"]:
            n1 = int(np.clip(H["B"] - H["n0"] + rng.integers(-1, 2), 1, 10)) if H["B"] else int(rng.integers(1, 5))
            pool.append(("term", tree(sample_shared(n1, rng, counts)))); kinds.append("T"); n1s.append(n1)
        elif rng.random() < H["kappa"]:
            pool.append(("const", smp._constant_literal(rng))); kinds.append("c")
        else:
            v = str(rng.choice([x for x in vs if x not in used])); used.add(v); pool.append(("var", v)); kinds.append("x")
    w = rng.dirichlet([H["alpha"]] * M)
    assign = list(range(M)) + list(rng.choice(M, L - M, p=w)); rng.shuffle(assign)
    seen, copies = set(), []
    for p, j in zip(paths, assign):
        kind, val = pool[j]
        if kind == "var": leaf = val
        elif kind == "const": leaf = val if (j not in seen or rng.random() < P["p_shared_const"]) else smp._constant_literal(rng)
        else:
            leaf, flags = modify(val, rng, H, P, counts); copies.append((p, j, leaf, flags))
        seen.add(j); t0 = put(t0, p, leaf)
    uses = collections.Counter(assign)
    H.update(L=L, M=M, pool="".join(kinds), uses=[uses[j] for j in range(M)], n1=n1s)
    return t0, copies, H

def bag(t, p):
    """(bag root path, class, parity) of the leaf position p: climb through +,-,neg (additive) or *,/,inv (multiplicative)."""
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

def merge_account(t, copies):
    """Copies of the same pool term that sit in one additive or multiplicative bag and simplify together."""
    res = []
    by = collections.defaultdict(list)
    for p, j, c, flags in copies:
        b = bag(t, p)
        by[(j, b[0] if b else ("solo", p), b[1] if b else None)].append((c, flags, b[2] if b else 0))
        res.append({"flags": flags, "redundant": False})
    k = 0
    idx = {}
    for p, j, c, flags in copies:
        idx.setdefault(id(c), k); k += 1
    flat_list = [(p, j, c, f) for p, j, c, f in copies]
    for key, members in by.items():
        if key[2] is None or len(members) < 2: continue
        kept = []
        for c, flags, par in members:
            fc = flat(c); lc = slen(fc)
            merged = False
            for c2, par2 in kept:
                f2 = flat(c2); l2 = slen(f2)
                op = ("+" if par == par2 else "-") if key[2] == "A" else ("*" if par == par2 else "/")
                lj = slen([op] + f2 + fc)
                if None not in (lc, l2, lj) and lj < lc + l2 + 1:
                    merged = True; break
            if merged:
                for rec_, (pp, jj, cc, ff) in zip(res, flat_list):
                    if cc is c: rec_["redundant"] = True
            else:
                kept.append((c, par))
    return res

def families_in(sk):
    return {FAMILY_OF[t] for t in sk if t in FAMILY_OF}

def record(sk):
    c = classify(sk, ar, vset)
    T = [t for t in terms(parse(sk, ar)) if has_var(t, vset)]
    ms = collections.Counter(motif(t, vset) for t in T if motif(t, vset) != "V")
    c["rep_nontrivial"] = bool(ms) and ms.most_common(1)[0][1] >= 2
    c["len"] = len(sk); c["sk"] = " ".join(sk); c["n_fam"] = len(families_in(sk)); return c

base = dict(n0=(1, 8), rho_beta=(1, 1), alpha=(0.1, 10), r=(0, 1), kappa=(0, 0.5), p_new_vars_gate=0.5,
            p_var_swap=0.5, p_exp=0.5, p_shared_const=0.1, p_struct=0.1, shared_urn=False, B=None)
variants = {"HP (round 3)": base,
            "HP + budget": {**base, "B": (5, 10)},
            "HP + budget + shared urn": {**base, "B": (5, 10), "shared_urn": True}}
N = int(sys.argv[1]); rng = np.random.default_rng(0)
out = {}
for name, P in variants.items():
    rows = []; copy_recs = []
    while len(rows) < N:
        t, copies, H = draw(rng, P)
        sk = flat(t); had = any(x in vset for x in sk)
        try: s = list(eng.simplify(sk, mode=cat.simplify_mode))
        except Exception: continue
        if had and not any(x in vset for x in s): continue
        if any("inf" in x or "nan" in x for x in s): continue
        r = record(s); r["H"] = H; r["raw_len"] = len(sk)
        acc = merge_account(t, copies)
        r["n_copies"] = len(acc); r["n_redundant"] = sum(a["redundant"] for a in acc)
        copy_recs += acc; rows.append(r)
    out[name] = {"rows": rows, "copies": [{"redundant": a["redundant"], **a["flags"]} for a in copy_recs]}
    print(f"generated {name}", flush=True)
json.dump(out, open("compose_v4_rows.json", "w"))
cfg2 = yaml.safe_load(open(CFG)); cfg2["holdout_family"] = "precise"
hcat = LampleChartonCatalog.from_config(cfg2); heng = hcat.simplipy_engine
for name, d in out.items():
    for r in d["rows"][:3000]:
        sk = r["sk"].split()
        try:
            _, c = explicit_constant_placeholders(heng.operators_to_realizations(sk), inplace=False, convert_numbers_to_constant=False)
            r["held"] = bool(hcat.is_held_out(sk, c, assume_canonical=hcat._skeleton_is_holdout_canonical))
        except Exception:
            r["held"] = True
json.dump(out, open("compose_v4_rows.json", "w"))
print("DONE", flush=True)
