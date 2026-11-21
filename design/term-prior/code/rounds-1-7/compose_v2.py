"""Term-reuse prior, revision 2 (owner's suggestions 2026-11-20) and the fully general term pool."""
import sys, json, warnings, collections
warnings.filterwarnings("ignore")
import numpy as np, yaml
from symbolic_data import LampleChartonCatalog
from classify import classify, parse, terms, motif, has_var
cfg = yaml.safe_load(open("/home/psaegert/Projects/flash-ansr/configs/v25.0-T8.1-120M/catalog_train.yaml"))
cfg["holdout_pools"] = []
cat = LampleChartonCatalog.from_config(cfg)
eng = cat.simplipy_engine; ar = dict(eng.operator_arity); vs = list(cat.variables); vset = set(vs)
smp = cat.skeleton_sampler; ss = cat.sample_strategy
slot_ops = set(smp.typed_slots)
UNARY = [o for o in smp.unary_operators if ar[o] == 1]          # true arity 1 (pow/rootn excluded: bug fix)
BINARY = [o for o in smp.binary_operators if ar[o] == 2 and o not in slot_ops]
t8_n = np.arange(ss["min_operators"], ss["max_operators"] + 1)
t8_p = np.exp(t8_n ** ss["power"] / ss["lambda"]); t8_p /= t8_p.sum()

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
def copy(n): return json.loads(json.dumps(n))

def modify(t, rng, P):
    """One use of a composite term: 10% one same-arity operator relabel, variables swapped to ANY of x1..x17,
    exponents redrawn, constants 90% fresh / 10% shared."""
    c = copy(t)
    if isinstance(c, str):
        return c
    if rng.random() < P["p_struct"]:
        ops = [p for p in op_paths(c) if get(c, p)[0] not in slot_ops]
        if ops:
            node = get(c, ops[rng.integers(len(ops))])
            node[0] = str(rng.choice(UNARY if ar[node[0]] == 1 else BINARY))
    def walk(n, slot_op=None):
        if isinstance(n, str):
            if slot_op is not None:
                return smp._slot_literal(slot_op, rng) if rng.random() < P["p_exp"] else n
            if n in vset:
                return str(rng.choice(vs)) if rng.random() < P["p_var"] else n
            return n if rng.random() < P["p_shared_const"] else smp._constant_literal(rng)
        return [n[0]] + [walk(ch, n[0] if is_slot(n[0], i) else None) for i, ch in enumerate(n[1:])]
    return walk(c)

def draw_n0(rng, P):
    return int(rng.integers(1, 9)) if P["n0"] == "u8" else int(rng.choice(t8_n, p=t8_p))

def compose_B(rng, P):
    t0 = tree(smp.sample(draw_n0(rng, P), rng))
    t1 = tree(smp.sample(int(rng.integers(1, 5)), rng))
    q = rng.random()                                     # per-expression reuse rate
    for p in growing_leaf_paths(t0):
        if rng.random() < q:
            t0 = put(t0, p, modify(t1, rng, P))
    return flat(t0)

def compose_C(rng, P):
    t0 = tree(smp.sample(draw_n0(rng, P), rng))          # shape only; its leaves are re-filled from the pool
    paths = growing_leaf_paths(t0); L = len(paths)
    r = rng.random()                                     # per-expression share of composite pool terms
    if P["pool"] == "lc":
        M = int(rng.integers(1, min(L, len(vs)) + 1))    # LC's own rule for the number of distinct leaf kinds
    else:
        M = min(L, 1 + int(rng.geometric(P["geo"])) - 1) # small pools: 1 + Geometric
        M = max(M, 1)
    pool, atoms_used = [], set()
    for _ in range(M):
        if rng.random() < r:
            pool.append(tree(smp.sample(int(rng.integers(1, 5)), rng)))
        else:                                            # atom: a variable or the constant slot, unique as in LC
            kinds = [k for k in vs + ["<constant>"] if k not in atoms_used]
            k = str(rng.choice(kinds)); atoms_used.add(k); pool.append(k)
    assign = list(range(M)) + list(rng.integers(0, M, L - M)) if L >= M else list(range(L))
    rng.shuffle(assign)
    for p, j in zip(paths, assign):
        term = pool[j]
        if term == "<constant>":
            leaf = smp._constant_literal(rng)
        elif isinstance(term, str):
            leaf = term
        else:
            leaf = modify(term, rng, P)
        t0 = put(t0, p, leaf)
    return flat(t0)

def record(sk):
    c = classify(sk, ar, vset)
    T = [t for t in terms(parse(sk, ar)) if has_var(t, vset)]
    ms = collections.Counter(motif(t, vset) for t in T if motif(t, vset) != "V")
    c["rep_nontrivial"] = bool(ms) and ms.most_common(1)[0][1] >= 2
    c["len"] = len(sk); c["sk"] = " ".join(sk); return c

def stats(rows, label):
    L = np.array([r["len"] for r in rows])
    ks = ["poly", "poly_ge3", "poly_1v_ge3", "sum_rat_ge2", "affine_ge2v", "superpos_ge3", "rep_nontrivial", "transcendental"]
    print(f"{label:30s} n={len(rows)} | len med {np.median(L):.0f} p90 {np.percentile(L,90):.0f} >35 {100*np.mean(L>35):.0f}% >40 {100*np.mean(L>40):.0f}% >60 {100*np.mean(L>60):.0f}% | vars med {np.median([r['nv'] for r in rows]):.0f} | "
          + " ".join(f"{k} {100*np.mean([r[k] for r in rows]):.2f}" for k in ks)
          + f" | terms>=2 {100*np.mean([r['k_terms']>=2 for r in rows]):.0f}% >=3 {100*np.mean([r['k_terms']>=3 for r in rows]):.0f}%", flush=True)

base = dict(p_struct=0.1, p_var=0.5, p_exp=0.5, p_shared_const=0.1)
variants = {"B  (n0 1-8)": (compose_B, {**base, "n0": "u8"}),
            "B  (n0 T8 prior)": (compose_B, {**base, "n0": "t8"}),
            "C  pool, LC pool size": (compose_C, {**base, "n0": "t8", "pool": "lc"}),
            "C-small pool (1+Geom 0.5)": (compose_C, {**base, "n0": "t8", "pool": "small", "geo": 0.5})}
N = int(sys.argv[1]); rng = np.random.default_rng(0)
R = [json.loads(l) for l in open("draws.jsonl")][:N]
stats([record(r["sk"].split()) for r in R], "plain LC (T8.1)")
out = {}
for name, (fn, P) in variants.items():
    rows = []
    while len(rows) < N:
        sk = fn(rng, P)
        had = any(t in vset for t in sk)
        try: s = list(eng.simplify(sk, mode=cat.simplify_mode))
        except Exception: continue
        if had and not any(t in vset for t in s): continue
        if any("inf" in t or "nan" in t for t in s): continue
        rows.append(record(s))
    stats(rows, name); out[name] = rows
json.dump(out, open("compose_v2_rows.json", "w"))
for name in ("B  (n0 T8 prior)", "C-small pool (1+Geom 0.5)"):
    print(f"\n[{name}] first 6 draws:")
    for r in out[name][:6]: print("   ", eng.prefix_to_infix(r["sk"].split())[:220])
