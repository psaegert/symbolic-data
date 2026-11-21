"""Prototype of the term-reuse prior: tau0 (short LC) whose leaves may take modified copies of tau1 (short LC)."""
import sys, json, warnings, collections
warnings.filterwarnings("ignore")
import numpy as np, yaml
from symbolic_data import LampleChartonCatalog
from classify import classify, parse, terms, motif, has_var
cfg = yaml.safe_load(open("/home/psaegert/Projects/flash-ansr/configs/v25.0-T8.1-120M/catalog_train.yaml"))
cfg["holdout_pools"] = []
cat = LampleChartonCatalog.from_config(cfg)
eng = cat.simplipy_engine; ar = dict(eng.operator_arity); vs = list(cat.variables); vset = set(vs)
smp = cat.skeleton_sampler
slot_ops = set(smp.typed_slots)
unary = [o for o in smp.unary_operators]; binary = [o for o in smp.binary_operators]

def tree(tokens):
    it = iter(tokens)
    def rec():
        t = next(it); a = ar.get(t, 0)
        return [t] + [rec() for _ in range(a)] if a else t
    return rec()
def flat(n):
    return [n] if isinstance(n, str) else [n[0]] + [t for c in n[1:] for t in flat(c)]
def leaf_paths(n, path=()):
    """Growing leaves only (typed exponent slots excluded)."""
    if isinstance(n, str): return [path]
    out = []
    for i, c in enumerate(n[1:]):
        if n[0] in slot_ops and i == smp.typed_slots[n[0]]["argument"]: continue
        out += leaf_paths(c, path + (i + 1,))
    return out
def get(n, p):
    for i in p: n = n[i]
    return n
def put(n, p, v):
    if not p: return v
    get(n, p[:-1])[p[-1]] = v; return n
def is_lit(t): return isinstance(t, str) and t not in ar and t not in vset

def modify(t1, rng, P, pool):
    c = json.loads(json.dumps(t1))
    if rng.random() < P["p_struct"]:      # operator relabel within its arity class (arity-preserving variant)
        ops = [p for p in leaf_paths_all(c) if not isinstance(get(c, p), str)]
        if ops:
            node = get(c, ops[rng.integers(len(ops))]); a = ar[node[0]]
            if node[0] not in slot_ops:
                node[0] = str(rng.choice(unary if a == 1 else binary))
    def walk(n, slot=False):
        if isinstance(n, str):
            if slot:
                return smp._slot_literal(slotop[0], rng) if rng.random() < P["p_exp"] else n
            if n in vset:
                return str(rng.choice(pool)) if rng.random() < P["p_var"] else n
            return n if rng.random() < P["p_shared_const"] else smp._constant_literal(rng)
        out = [n[0]]
        for i, ch in enumerate(n[1:]):
            s = n[0] in slot_ops and i == smp.typed_slots[n[0]]["argument"]
            if s: slotop[0] = n[0]
            out.append(walk(ch, s))
        return out
    slotop = [None]
    return walk(c)
def leaf_paths_all(n, path=()):
    if isinstance(n, str): return [path]
    out = [path]
    for i, c in enumerate(n[1:]): out += leaf_paths_all(c, path + (i + 1,))
    return out

def compose(rng, P):
    if rng.random() < P.get("p_sum_mode", 0.0):
        K = int(rng.integers(2, 5))
        t0 = "L0"
        for i in range(1, K):
            t0 = ["+", t0, f"L{i}"]
        t0 = ["+", t0, smp._constant_literal(rng)] if rng.random() < 0.5 else t0
        sum_mode = True
    else:
        t0 = tree(smp.sample(int(rng.integers(P["n0"][0], P["n0"][1] + 1)), rng))
        sum_mode = False
    t1 = tree(smp.sample(int(rng.integers(P["n1"][0], P["n1"][1] + 1)), rng))
    if isinstance(t1, str): return None
    if P.get("alphabet_k"):
        k = int(rng.choice([1, 2, 3, 4], p=P["alphabet_k"]))
        alpha = [str(v) for v in rng.choice(vs, k, replace=False)]
        def rn_all(n, m):
            if isinstance(n, str):
                if n in vset:
                    m.setdefault(n, str(rng.choice(alpha)))
                    return m[n]
                return n
            return [n[0]] + [rn_all(c, m) for c in n[1:]]
        t0 = rn_all(t0, {}); t1 = rn_all(t1, {})
    if P.get("shared_alphabet"):
        v0 = sorted({t for t in flat(t0) if t in vset}) or [str(rng.choice(vs))]
        ren = {}
        def rn(n):
            if isinstance(n, str):
                if n in vset:
                    ren.setdefault(n, str(rng.choice(v0)))
                    return ren[n]
                return n
            return [n[0]] + [rn(c) for c in n[1:]]
        t1 = rn(t1)
    pool = sorted({t for t in flat(t0) + flat(t1) if t in vset}) or vs[:1]
    paths = leaf_paths(t0)
    if sum_mode:
        paths = [p for p in paths if isinstance(get(t0, p), str) and get(t0, p).startswith("L")]
        for p in paths:
            c = modify(t1, rng, P, pool)
            if rng.random() < P.get("p_coef", 0.0):
                c = ["*", smp._constant_literal(rng), c]
            t0 = put(t0, p, c)
        return flat(t0)
    if P.get("additive_slots"):
        def additive(p):
            n = t0
            for i in p:
                if n[0] not in ("+", "-", "neg"): return False
                n = n[i]
            return True
        pick = [p for p in paths if rng.random() < (P["q_add"] if additive(p) else P["q_other"])]
    else:
        pick = [p for p in paths if rng.random() < P["q"]]
    if not pick: pick = [paths[rng.integers(len(paths))]]
    for p in pick:
        c = modify(t1, rng, P, pool)
        if rng.random() < P.get("p_coef", 0.0):
            c = ["*", smp._constant_literal(rng), c]
        t0 = put(t0, p, c)
    return flat(t0)

def stats(rows, label):
    n = len(rows); L = np.array([r["len"] for r in rows])
    ks = ["poly", "poly_ge3", "poly_1v_ge3", "rational", "sum_rat_ge2", "affine_ge2v", "superpos_ge3", "rep_nontrivial", "transcendental"]
    s = {k: 100 * np.mean([r[k] for r in rows]) for k in ks}
    print(f"{label:34s} n={n:5d} len med {np.median(L):4.0f} p90 {np.percentile(L,90):3.0f} >35 {100*np.mean(L>35):4.1f}% >40 {100*np.mean(L>40):4.1f}% vars med {np.median([r['nv'] for r in rows]):.0f} | " +
          " ".join(f"{k} {v:.2f}" for k, v in s.items()), flush=True)

def record(sk):
    c = classify(sk, ar, vset)
    T = [t for t in terms(parse(sk, ar)) if has_var(t, vset)]
    ms = collections.Counter(motif(t, vset) for t in T if motif(t, vset) != "V")
    c["rep_nontrivial"] = bool(ms) and ms.most_common(1)[0][1] >= 2
    c["len"] = len(sk); return c

rng = np.random.default_rng(0)
base = dict(n0=(1, 4), n1=(1, 4), q=0.5, p_var=0.5, p_exp=0.5, p_shared_const=0.1, p_struct=0.1)
variants = {"user's numbers": base,
            "+ shared alphabet": {**base, "shared_alphabet": True},
            "+ shared alphabet + additive slots": {**base, "shared_alphabet": True, "additive_slots": True, "q_add": 0.8, "q_other": 0.15},
            "+ both, tau0 2-6 ops": {**base, "shared_alphabet": True, "additive_slots": True, "q_add": 0.8, "q_other": 0.15, "n0": (2, 6)}}
full = {**variants["+ both, tau0 2-6 ops"], "p_coef": 0.8, "shared_alphabet": False, "alphabet_k": [0.4, 0.3, 0.2, 0.1]}
variants = {"+ sum mode 30%": {**full, "p_sum_mode": 0.3},
            "sum mode only (100%)": {**full, "p_sum_mode": 1.0}}
EXAMPLES = {}
N = int(sys.argv[1])
# plain LC baseline from the earlier 20k raw draws
R = [json.loads(l) for l in open("draws.jsonl")]
base_rows = [record(r["sk"].split()) for r in R[:N]]
stats(base_rows, "plain LC (T8.1 prior)")
for name, P in variants.items():
    rows = []
    while len(rows) < N:
        sk = compose(rng, P)
        if sk is None: continue
        had = any(t in vset for t in sk)
        try: s = list(eng.simplify(sk, mode=cat.simplify_mode))
        except Exception: continue
        if had and not any(t in vset for t in s): continue
        if any("inf" in t or "nan" in t for t in s): continue
        r = record(s); r["sk"] = " ".join(s); rows.append(r)
    stats(rows, "composed: " + name)
    EXAMPLES[name] = rows
    if name == "user's numbers":
        ex = [r for r in rows if r["poly_ge3"] or r["sum_rat_ge2"]][:0]

for name in ("sum mode only (100%)",):
    rows = EXAMPLES[name]
    print(f"\n[{name}] first 8 draws:")
    for r in rows[:8]: print("   ", eng.prefix_to_infix(r["sk"].split()))
    print(f"[{name}] polynomials with >= 3 terms:")
    for r in [r for r in rows if r["poly_ge3"]][:4]: print("   ", eng.prefix_to_infix(r["sk"].split()))
    print(f"[{name}] sums of >= 2 rational terms:")
    for r in [r for r in rows if r["sum_rat_ge2"]][:4]: print("   ", eng.prefix_to_infix(r["sk"].split()))
