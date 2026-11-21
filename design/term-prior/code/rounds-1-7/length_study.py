import sys, json, re, time, warnings, collections
warnings.filterwarnings("ignore")
import numpy as np, yaml
from symbolic_data import LampleChartonCatalog
from classify import classify
cfg = yaml.safe_load(open("/home/psaegert/Projects/flash-ansr/configs/v25.0-T8.1-120M/catalog_train.yaml"))
cfg["holdout_pools"] = []
cat = LampleChartonCatalog.from_config(cfg)
eng = cat.simplipy_engine; ar = dict(eng.operator_arity); vs = set(cat.variables)
ss = cat.sample_strategy
def nprior(maxops):
    n = np.arange(ss["min_operators"], maxops + 1); p = np.exp(n ** ss["power"] / ss["lambda"]); return n, p / p.sum()
rng = np.random.default_rng(0)
out = {}
for maxops in (17, 20):
    n, p = nprior(maxops); rows = []
    for _ in range(int(sys.argv[1])):
        k = int(rng.choice(n, p=p))
        sk = cat.skeleton_sampler.sample(k, rng)
        had = any(t in vs for t in sk)
        try:
            s = list(eng.simplify(sk, mode=cat.simplify_mode))
        except Exception:
            continue
        if had and not any(t in vs for t in s):
            continue
        if any(t in s for t in ['float("inf")', 'float("-inf")', 'float("nan")']):
            continue
        c = classify(s, ar, vs)
        lits = sum(1 for t in s if t not in ar and t not in vs)
        rows.append({"n_ops": k, "len": len(s), "lits": lits, **{key: c[key] for key in ("poly", "poly_ge3", "sum_rat_ge2", "affine_ge2v", "superpos_ge3", "rational")}, "nv": c["nv"]})
    out[maxops] = rows
    L = np.array([r["len"] for r in rows])
    print(f"max_ops {maxops}: {len(rows)} delivered-before-cap draws; len median {np.median(L):.0f}, "
          f"<=35 {100*np.mean(L<=35):.1f}%, 36-40 {100*np.mean((L>35)&(L<=40)):.1f}%, >40 {100*np.mean(L>40):.1f}%", flush=True)
    for key in ("poly", "poly_ge3", "rational", "sum_rat_ge2", "affine_ge2v", "superpos_ge3"):
        a = [r for r in rows if r[key]]
        if a:
            La = np.array([r["len"] for r in a])
            print(f"   {key:14s} n={len(a):5d}  >35: {100*np.mean(La>35):5.1f}%  >40: {100*np.mean(La>40):5.1f}%")
    by = collections.defaultdict(list)
    for r in rows: by[r["n_ops"]].append(r["len"])
    print("   share >35 by n_ops:", {k: f"{100*np.mean(np.array(v)>35):.0f}%" for k, v in sorted(by.items()) if k >= 10})
    lit = np.array([r["lits"] for r in rows])
    print(f"   corr(len>35, #literals): mean literals among >35 {lit[L>35].mean():.1f} vs <=35 {lit[L<=35].mean():.1f}")
json.dump({str(k): v for k, v in out.items()}, open("length_draws.json", "w"))

# benchmark laws in the catalog's own unit (explicit-binary canonical prefix, literal values spelled)
laws = json.load(open("holdout_laws.json"))
per = collections.defaultdict(list); seen = set()
for r in laws:
    key = (r["pool"], r["law"])
    if key in seen: continue
    seen.add(key)
    try:
        tok = ["pow" if t in ("**", "^") else (("x" + t[1:]) if re.fullmatch(r"v\d+", t) else t) for t in eng.infix_to_prefix(r["law"])]
        from symbolic_data.token_ops import desugar_sqrt
        tok = desugar_sqrt(tok, eng.operator_arity_compat)
        s = list(eng.simplify(tok, mode=cat.simplify_mode))
    except Exception:
        continue
    per[r["pool"]].append(len(s))
allL = []
print("\nbenchmark laws, canonical length (catalog unit):")
for pool in sorted(per, key=lambda p: -len(per[p])):
    L = np.array(per[pool]); allL += list(L)
    print(f"  {pool:20s} n={len(L):5d} median {np.median(L):4.0f}  >35 {100*np.mean(L>35):5.1f}%  >40 {100*np.mean(L>40):5.1f}%  max {L.max()}")
A = np.array(allL)
print(f"  {'ALL':20s} n={len(A):5d} median {np.median(A):4.0f}  >35 {100*np.mean(A>35):5.1f}%  36-40 {100*np.mean((A>35)&(A<=40)):5.1f}%  >40 {100*np.mean(A>40):5.1f}%")
