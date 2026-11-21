"""Hierarchical term-pool prior (owner design 2026-11-20): per-expression hyperparameters, then the expression."""
import sys, json, warnings, collections, math
warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/psaegert/Projects/symbolic-data/.claude/worktrees/holdout-precise/src")
import numpy as np, yaml
from symbolic_data import LampleChartonCatalog
from simplipy.utils import explicit_constant_placeholders
from classify import classify, parse, terms, motif, has_var
CFG = "/home/psaegert/Projects/flash-ansr/configs/v25.0-T8.1-120M/catalog_train.yaml"
cfg = yaml.safe_load(open(CFG)); cfg["holdout_pools"] = []
cat = LampleChartonCatalog.from_config(cfg)
eng = cat.simplipy_engine; ar = dict(eng.operator_arity); vs = list(cat.variables); vset = set(vs)
smp = cat.skeleton_sampler; slot_ops = set(smp.typed_slots)
UNARY = [o for o in smp.unary_operators if ar[o] == 1]
BINARY = [o for o in smp.binary_operators if ar[o] == 2 and o not in slot_ops]

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

def modify(t, rng, H, P):
    c = json.loads(json.dumps(t))
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
                return str(rng.choice(vs)) if (H["new_vars"] and rng.random() < P["p_var_swap"]) else n
            return n if rng.random() < P["p_shared_const"] else smp._constant_literal(rng)
        return [n[0]] + [walk(ch, n[0] if is_slot(n[0], i) else None) for i, ch in enumerate(n[1:])]
    return walk(c)

def draw(rng, P):
    # ---- layer 1: per-expression hyperparameters
    H = {"n0": int(rng.integers(P["n0"][0], P["n0"][1] + 1))}
    H["rho"] = float(rng.beta(*P["rho_beta"]))
    H["alpha"] = float(math.exp(rng.uniform(math.log(P["alpha"][0]), math.log(P["alpha"][1]))))
    H["r"] = float(rng.uniform(*P["r"]))
    H["kappa"] = float(rng.uniform(*P["kappa"]))
    H["new_vars"] = bool(rng.random() < P["p_new_vars_gate"])
    # ---- layer 2: the expression
    t0 = tree(smp.sample(H["n0"], rng))
    paths = growing_leaf_paths(t0); L = len(paths)
    M = 1 + int(rng.binomial(L - 1, 1 - H["rho"])) if L > 1 else 1
    pool, used_vars, kinds = [], set(), []
    for _ in range(M):
        if rng.random() < H["r"]:
            pool.append(("term", tree(smp.sample(int(rng.integers(1, 5)), rng)))); kinds.append("T")
        elif rng.random() < H["kappa"]:
            pool.append(("const", smp._constant_literal(rng))); kinds.append("c")
        else:
            v = str(rng.choice([x for x in vs if x not in used_vars])); used_vars.add(v)
            pool.append(("var", v)); kinds.append("x")
    w = rng.dirichlet([H["alpha"]] * M)
    assign = list(range(M)) + list(rng.choice(M, L - M, p=w))
    rng.shuffle(assign)
    seen = set()
    for p, j in zip(paths, assign):
        kind, val = pool[j]
        if kind == "var":
            leaf = val
        elif kind == "const":
            leaf = val if (j not in seen or rng.random() < P["p_shared_const"]) else smp._constant_literal(rng)
        else:
            leaf = modify(val, rng, H, P)
        seen.add(j)
        t0 = put(t0, p, leaf)
    uses = collections.Counter(assign)
    H.update(L=L, M=M, pool="".join(kinds), uses=[uses[j] for j in range(M)])
    return flat(t0), H

def record(sk):
    c = classify(sk, ar, vset)
    T = [t for t in terms(parse(sk, ar)) if has_var(t, vset)]
    ms = collections.Counter(motif(t, vset) for t in T if motif(t, vset) != "V")
    c["rep_nontrivial"] = bool(ms) and ms.most_common(1)[0][1] >= 2
    c["len"] = len(sk); c["sk"] = " ".join(sk); return c

def stats(rows, label, held=None):
    L = np.array([r["len"] for r in rows])
    ks = ["poly", "poly_ge3", "poly_1v_ge3", "sum_rat_ge2", "affine_ge2v", "superpos_ge3", "rep_nontrivial", "transcendental"]
    s = (f"{label:24s} n={len(rows)} | len med {np.median(L):.0f} p90 {np.percentile(L,90):.0f} >35 {100*np.mean(L>35):.1f}% >40 {100*np.mean(L>40):.1f}% | vars med {np.median([r['nv'] for r in rows]):.0f} 1-var {100*np.mean([r['nv']==1 for r in rows]):.0f}% | "
         + " ".join(f"{k} {100*np.mean([r[k] for r in rows]):.2f}" for k in ks)
         + f" | terms>=2 {100*np.mean([r['k_terms']>=2 for r in rows]):.0f}% >=3 {100*np.mean([r['k_terms']>=3 for r in rows]):.0f}%")
    if held is not None: s += f" | held(precise) {100*np.mean(held):.1f}%"
    print(s, flush=True)

base = dict(n0=(1, 8), rho_beta=(1, 1), alpha=(0.1, 10), r=(0, 1), kappa=(0, 0.5), p_new_vars_gate=0.5,
            p_var_swap=0.5, p_exp=0.5, p_shared_const=0.1, p_struct=0.1)
variants = {"HP default": base, "HP reuse-heavy": {**base, "rho_beta": (3, 1), "alpha": (0.1, 1)}}
N = int(sys.argv[1]); rng = np.random.default_rng(0)
out = {}
for name, P in variants.items():
    rows = []
    while len(rows) < N:
        sk, H = draw(rng, P)
        had = any(t in vset for t in sk)
        try: s = list(eng.simplify(sk, mode=cat.simplify_mode))
        except Exception: continue
        if had and not any(t in vset for t in s): continue
        if any("inf" in t or "nan" in t for t in s): continue
        r = record(s); r["H"] = H; rows.append(r)
    out[name] = rows
json.dump(out, open("compose_v3_rows.json", "w"))
# holdout scoring under the precise key (29 T8 pools)
cfg2 = yaml.safe_load(open(CFG)); cfg2["holdout_family"] = "precise"
hcat = LampleChartonCatalog.from_config(cfg2); heng = hcat.simplipy_engine
R = [json.loads(l) for l in open("draws.jsonl")][:N]
stats([record(r["sk"].split()) for r in R], "plain LC (T8.1)")
for name, rows in out.items():
    held = []
    for r in rows[:3000]:
        sk = r["sk"].split()
        try:
            _, c = explicit_constant_placeholders(heng.operators_to_realizations(sk), inplace=False, convert_numbers_to_constant=False)
            held.append(bool(hcat.is_held_out(sk, c, assume_canonical=hcat._skeleton_is_holdout_canonical)))
        except Exception:
            held.append(True)
    for r, h in zip(rows, held): r["held"] = h
    stats(rows, name, held)
json.dump(out, open("compose_v3_rows.json", "w"))
print("DONE", flush=True)
