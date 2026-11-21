import sys, json, re, time, warnings, collections
warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/psaegert/Projects/symbolic-data/.claude/worktrees/holdout-precise/src")
import numpy as np, yaml
import symbolic_data
from symbolic_data import LampleChartonCatalog
from symbolic_data.generative import HOLDOUT_EFFORT
from simplipy import masking
from simplipy.utils import explicit_constant_placeholders
assert "holdout-precise" in symbolic_data.__file__
mode = sys.argv[1]
tag = sys.argv[2] if len(sys.argv) > 2 else mode
if len(sys.argv) > 3:
    import symbolic_data.generative as G; G.HOLDOUT_PROTOTYPE_PASSES = int(sys.argv[3])
cfg = yaml.safe_load(open("/home/psaegert/Projects/flash-ansr/configs/v25.0-T8.1-120M/catalog_train.yaml"))
pools = cfg["holdout_pools"]; cfg["holdout_pools"] = []; cfg["holdout_family"] = mode
cat = LampleChartonCatalog.from_config(cfg)
eng = cat.simplipy_engine
laws = []; cur = {"pool": None}
orig = cat._holdout_law_prototypes
def wrapped(tokens):
    laws.append((cur["pool"], [str(t) for t in tokens])); return orig(tokens)
cat._holdout_law_prototypes = wrapped
t0 = time.time()
for p in pools:
    cur["pool"] = p; cat.register_holdout_pool(p)
print(f"[{tag}] registered {len(laws)} law renderings, {len(cat.holdout_manager.skeleton_hashes)} keys, {time.time()-t0:.0f}s", flush=True)
canon_ok = cat._skeleton_is_holdout_canonical

def deliver(tokens):
    """What the sampler would deliver for this expression: the target canonicalization."""
    return list(eng.simplify(tokens, mode=cat.simplify_mode))

def held(sk):
    _, consts = explicit_constant_placeholders(eng.operators_to_realizations(sk), inplace=False, convert_numbers_to_constant=False)
    return cat.is_held_out(sk, consts, assume_canonical=canon_ok)

rng = np.random.default_rng(1)
def redraw(sk):
    out = list(sk)
    for i, v, role in masking.literal_sites(out, eng):
        if masking.mask_fittable(v, role) is not None:
            out[i] = repr(round(float(rng.choice([-1, 1]) * rng.uniform(0.2, 5.0)), 3))
    return out

miss = collections.Counter(); tot = collections.Counter(); examples = collections.defaultdict(list); missed_all = []
for pool, tokens in laws:
    tok = [("x" + t[1:]) if re.fullmatch(r"v\d+", t) else ("pow" if t in ("**", "^") else t) for t in tokens]
    if any(re.fullmatch(r"x(\d+)", t) and int(t[1:]) > 17 for t in tok):
        continue
    try:
        base = deliver(tok)
    except Exception:
        continue
    members = {"canonical": base}
    for k in range(3):
        try: members[f"redraw{k}"] = deliver(redraw(base))
        except Exception: pass
    a, b = round(rng.uniform(0.5, 3), 3), round(rng.uniform(-3, 3), 3)
    try: members["affine"] = deliver(["+", "*", repr(a), *base, repr(b)])
    except Exception: pass
    for kind, sk in members.items():
        if not any(re.fullmatch(r"x\d+", t) for t in sk):
            continue    # a member that lost every variable is not a draw the sampler delivers
        tot[kind] += 1
        try: h = held(sk)
        except Exception: h = True
        if not h:
            miss[kind] += 1
            missed_all.append([pool, " ".join(tokens), kind, " ".join(sk)])
            if len(examples[kind]) < 8: examples[kind].append(f"{pool}: {' '.join(tokens)}  ->  {' '.join(sk)}")
json.dump(missed_all, open(f"missed_{tag}.json", "w"))
print(f"[{tag}] family-member leak audit:", {k: f"{miss[k]}/{tot[k]}" for k in tot}, flush=True)
for k, ex in examples.items():
    print(f"  missed {k}:"); [print("    ", e) for e in ex]

# 20k recorded prior draws
R = [json.loads(l) for l in open("draws.jsonl")]
flags = []
for r in R:
    sk = r["sk"].split()
    try: flags.append(bool(held(sk)))
    except Exception: flags.append(True)
json.dump(flags, open(f"flags_{tag}.json", "w"))
print(f"[{tag}] 20k draws held out: {100*np.mean(flags):.2f}%  ({time.time()-t0:.0f}s)", flush=True)
