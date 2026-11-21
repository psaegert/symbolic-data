import sys, json, warnings, collections
warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/psaegert/Projects/symbolic-data/.claude/worktrees/holdout-precise/src")
import yaml, numpy as np
from symbolic_data import LampleChartonCatalog
from simplipy.utils import explicit_constant_placeholders
cfg = yaml.safe_load(open("/home/psaegert/Projects/flash-ansr/configs/v25.0-T8.1-120M/catalog_train.yaml"))
cfg["holdout_family"] = "precise"
cat = LampleChartonCatalog.from_config(cfg); eng = cat.simplipy_engine
for tag in ("user", "sum"):
    rows = json.load(open(f"composed_{tag}.json"))[:3000]
    held = []
    for r in rows:
        sk = r["sk"].split()
        try:
            _, c = explicit_constant_placeholders(eng.operators_to_realizations(sk), inplace=False, convert_numbers_to_constant=False)
            held.append(bool(cat.is_held_out(sk, c, assume_canonical=cat._skeleton_is_holdout_canonical)))
        except Exception:
            held.append(True)
    h = np.array(held)
    print(f"[{tag}] held out {100*h.mean():.1f}% of {len(rows)}")
    for k in ["poly", "poly_ge3", "poly_1v_ge3", "sum_rat_ge2", "affine_ge2v", "superpos_ge3", "rep_nontrivial"]:
        idx = [i for i, r in enumerate(rows) if r[k]]
        if idx: print(f"   {k:16s} n={len(idx):4d} held {100*h[idx].mean():5.1f}%")
    m1 = [i for i, r in enumerate(rows) if r["nv"] == 1 and r["k_terms"] >= 2]
    if m1: print(f"   univariate, >=2 terms n={len(m1)} held {100*h[m1].mean():.1f}%")
