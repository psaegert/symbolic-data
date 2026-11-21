import sys, time, json, warnings
warnings.filterwarnings("ignore")
import numpy as np
from symbolic_data import LampleChartonCatalog, NoValidSampleFoundError
from simplipy.utils import explicit_constant_placeholders
from classify import classify
CFG = "/home/psaegert/Projects/flash-ansr/configs/v25.0-T8.1-120M/catalog_train.yaml"
N = int(sys.argv[1]); out = open(sys.argv[2], "w")
t0 = time.time()
cat = LampleChartonCatalog.from_config(CFG)
eng = cat.simplipy_engine
arity = dict(eng.operator_arity)
vs = set(cat.variables)
hm = cat.holdout_manager
print(f"built {time.time()-t0:.0f}s", flush=True)

def reason(sk, consts):
    proto = cat.holdout_family_prototype(sk)
    if proto is None: return "proto_none"
    if tuple(hm._normalize_tokens(proto)) in hm.skeleton_hashes: return "hash"
    return "image" if cat.is_held_out(sk, consts) else "-"

# Part A: probes
probes = ["x1", "2.3*x1 + 0.7", "1.3*x1**2 + 0.4", "1.3*x1**2 + 0.4*x1 + 2.1",
 "1.3*x1**3 + 0.4*x1**2 - 0.8*x1 + 2.1", "0.5*x1**5 - 1.3*x1**3 + 0.4*x1 + 2.1",
 "1.3*x1*x2 + 0.5*x1 + 0.2", "1.3*x1**2 + 0.6*x2**2 + 0.2", "0.3*x1 + 0.7*x2 + 1.1*x3 + 0.2",
 "1.5/(x1 + 0.3) + 2.2/(x1 - 1.7)", "1.5/(x1 + 0.3) + 2.2/(x2 - 1.7)", "1.5/(x1**2 + 0.3) + 2.2",
 "(x1 + 0.4)/(x1**2 + 1.3)", "2.1*exp(-1.3*x1) + 0.7*exp(-0.2*x1)", "1.2*sin(0.7*x1) + 0.4*sin(2.3*x1)",
 "1.2*exp(-0.3*x1)", "x1*exp(-0.5*x2)", "1.3*sqrt(x1**2 + x2**2)", "1.7*x1*x2/x3**2",
 "1.3*x1**2*x2 + 0.4*x3", "x1**2 + 0.7*sin(x2)", "0.7*x1*cos(1.3*x2) + 0.2*x3**3"]
for s in probes:
    try:
        sk = ["pow" if t in ("**", "^") else t for t in eng.infix_to_prefix(s)]
        sk = list(eng.simplify(sk, mode=cat.simplify_mode))
        _, consts = explicit_constant_placeholders(eng.operators_to_realizations(sk), inplace=False, convert_numbers_to_constant=False)
        proto = cat.holdout_family_prototype(sk)
        print(f"PROBE {reason(sk, consts):10s} {s:45s} sk={' '.join(sk)}  proto={' '.join(proto)}", flush=True)
    except Exception as e:
        print(f"PROBE ERR {s}: {type(e).__name__}: {e}", flush=True)

# Part B: raw prior draws, holdout applied afterwards
rng = np.random.default_rng(0)
ss = cat.sample_strategy
ops_range = np.arange(ss['min_operators'], ss['max_operators'] + 1)
p = np.exp(ops_range ** ss['power'] / ss['lambda']); p /= p.sum()
done = fail = 0; t1 = time.time()
while done < N:
    try:
        sk, code, consts = cat.sample_skeleton(new=True, decontaminate=False, rng=rng)
    except NoValidSampleFoundError:
        fail += 1; continue
    sk = list(sk)
    ho = cat.is_held_out(sk, consts, assume_canonical=cat._skeleton_is_holdout_canonical)
    rec = classify(sk, arity, vs)
    rec.update(sk=" ".join(sk), held_out=bool(ho), n_tok=len(sk))
    out.write(json.dumps(rec) + "\n"); done += 1
    if done % 1000 == 0:
        out.flush(); print(f"{done} draws, {fail} fails, {time.time()-t1:.0f}s", flush=True)
print("DONE", flush=True)
