"""Export problems from the round-10c training catalog (v26.0-TP10c-20M), like front_study/export_training_problems.py.

Same deviations from a training instance as the T8 export (2026-11-16): 512 support + 512 validation points, noise 0,
unused variable columns zeroed (all 17 kept), placeholders / non-finite / constant targets skipped and counted. The
catalog is the exact training catalog (holdout pools, precise key, 35-token cap). Two modes:
  random      the first n problems in sampling order, no selection;
  structured  problems selected by EXPRESSION class only (never by model performance): up to n/3 each of polynomial
              with >= 3 terms, >= 3 terms of one form, sum of >= 2 proper rational terms, in sampling order.
  nonlinear   the same three classes, excluding weighted sums of single variables (added 2026-11-21 after 41 of 48
              polynomials in the first two sets turned out to be such linear sums).
Each problem carries its class flags (classify.py of the prior_structure run) under the key `cls`.

Usage: PYTHONPATH=<term-prior worktree>/src python export_tp_problems.py <out_dir> <n> <seed> <random|structured> <name>
"""
import pickle
import sys
import warnings
from collections import Counter

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
import symbolic_data  # noqa: E402
from flash_ansr.utils.config_io import load_config  # noqa: E402
from flash_ansr.utils.tensor_ops import mask_unused_variable_columns  # noqa: E402
from symbolic_data import ProblemSource, build_catalog  # noqa: E402

sys.path.insert(0, "/home/psaegert/Projects/flash-ansr-rl/runs/prior_structure")
from classify import classify, has_var, motif, parse, terms  # noqa: E402


def _lin(t, vs):
    if isinstance(t, str):
        return t in vs
    if t[0] in ("*", "/"):
        return (sum(has_var(c, vs) for c in t[1]) == 1 and not (t[0] == "/" and has_var(t[1][1], vs))
                and all(isinstance(c, str) or not has_var(c, vs) or _lin(c, vs) for c in t[1]))
    if t[0] == "neg":
        return _lin(t[1][0], vs)
    return False

assert "term-prior" in symbolic_data.__file__, symbolic_data.__file__
CATALOG = "/home/psaegert/Projects/flash-ansr-rl/runs/term_prior_20m/configs/v26.0-TP10c-20M/catalog_train.yaml"
out, n, seed, mode, name = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], sys.argv[5]
catalog = build_catalog(load_config(CATALOG))
ar = dict(catalog.simplipy_engine.operator_arity)
vset = set(catalog.variables)


def classes(toks):
    c = classify(toks, ar, vset)
    T = [t for t in terms(parse(toks, ar)) if has_var(t, vset)]
    ms = Counter(motif(t, vset) for t in T if motif(t, vset) != "V")
    return {"nv": c["nv"], "k": c["k_terms"], "poly3": c["poly_ge3"], "rat2": c["sum_rat_ge2"], "sup3": c["superpos_ge3"],
            "rep": bool(ms) and ms.most_common(1)[0][1] >= 2, "trans": c["transcendental"],
            "linear": bool(T) and all(_lin(t, vset) for t in T)}


src = ProblemSource({"catalog": catalog, "sampling": {"n_support": 512, "n_validation": 512, "noise": 0.0,
                                                      "problems_per_expression": 1}}, rng=np.random.default_rng(seed))
keys = ["benchmark_eq_id", "eval_row_index", "x", "y", "y_noisy", "x_val", "y_val", "y_val_noisy", "variable_names",
        "skeleton", "ground_truth_infix", "ground_truth_prefix", "n_used_variables", "noise", "cls"]
F = {k: [] for k in keys}
skipped = Counter()
quota = {"poly3": n // 3, "sup3": n // 3, "rat2": n - 2 * (n // 3)}
taken = Counter()
seen = 0
it = iter(src)
while len(F["x"]) < n:
    p = next(it)
    seen += 1
    if p.is_placeholder or p.skeleton is None:
        skipped["placeholder"] += 1
        continue
    toks = [str(t) for t in p.skeleton]
    x, xv = np.array(p.x_support, float), np.array(p.x_validation, float)
    y, yv = np.asarray(p.y_support, float).reshape(-1), np.asarray(p.y_validation, float).reshape(-1)
    if not all(np.isfinite(a).all() for a in (x, y, xv, yv)):
        skipped["non-finite"] += 1
        continue
    if np.var(y) == 0:
        skipped["constant y"] += 1
        continue
    cls = classes(toks)
    if mode == "nonlinear" and cls["linear"]:
        continue
    if mode in ("structured", "nonlinear"):
        hit = next((c for c in ("poly3", "sup3", "rat2") if cls[c] and taken[c] < quota[c]), None)
        if hit is None:
            continue
        taken[hit] += 1
        cls["selected_as"] = hit
    names = [f"x{j + 1}" for j in range(x.shape[1])]
    mask_unused_variable_columns(arrays=(x, xv), variables=names, skeleton_tokens=toks, padding="zero")
    i = len(F["x"])
    rec = dict(benchmark_eq_id=f"{name}_{i:03d}", eval_row_index=i, x=x, y=y, y_noisy=y.copy(), x_val=xv, y_val=yv,
               y_val_noisy=yv.copy(), variable_names=names, skeleton=toks, ground_truth_infix=" ".join(toks),
               ground_truth_prefix=toks, n_used_variables=len({t for t in toks if t in set(names)}), noise=0.0, cls=cls)
    for k in keys:
        F[k].append(rec[k])
print("exported", len(F["x"]), "of", seen, "streamed; skipped", dict(skipped), "| taken", dict(taken))
print("classes", {c: sum(r[c] for r in F["cls"]) for c in ("poly3", "sup3", "rat2", "rep", "trans")},
      "| used variables", sorted(Counter(F["n_used_variables"]).items()))
pickle.dump(F, open(f"{out}/{name}_problems.pkl", "wb"))
pd.DataFrame(dict(law=F["benchmark_eq_id"], row=F["eval_row_index"], group=name)).to_csv(f"{out}/{name}_laws.csv", index=False)
