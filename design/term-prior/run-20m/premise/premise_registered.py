"""PLAN.md step 0 as registered (added 2026-11-21 after the fact-check): T8-catalog against random round-10c problems
"at matched size and variable count" (strata used variables 1-4 / 5-6 / 7+ x structural size <=15 / 16-22 / 23+,
weighted by T8's strata, 2,000 bootstrap resamples), and the registered read-out "by the coverage parse: fully
explained or not" (coverage.py of the prior_structure run on the ground truth; strict = no mixed chain; with the rule
= every mixed chain is k copies of one template plus one other part)."""
import pickle, sys
import numpy as np, pandas as pd
sys.path.insert(0, "/home/psaegert/Projects/flash-ansr-rl/runs/prior_structure/explorer")
import coverage as COV  # noqa: E402
D = pd.read_csv("premise_problems.csv")
P = {"tp10c": pickle.load(open("data/tp10c_problems.pkl", "rb")),
     "t8": pickle.load(open("/home/psaegert/Projects/flash-ansr-rl/runs/front_study/data/training_problems.pkl", "rb"))}
gt = {f"{e}__{r}": [str(t) for t in p] for S in P.values() for e, r, p in zip(S["benchmark_eq_id"], S["eval_row_index"], S["ground_truth_prefix"])}
R = D[D.set.isin(["t8", "tp10c"])].copy()
R["vb"] = pd.cut(R.nvar, [0, 4, 6, 17], labels=["1-4", "5-6", "7+"])
R["sb"] = pd.cut(R["size"], [0, 15, 22, 40], labels=["<=15", "16-22", "23+"])
print("== registered comparison: strata used variables x size, T8 weights")
print(R.pivot_table(index=["vb", "sb"], columns="set", values="pool_rec", aggfunc=["mean", "size"], observed=False).round(2).to_string())
def strat(R):
    w = R[R.set == "t8"].groupby(["vb", "sb"], observed=True).size()
    m = R.groupby(["set", "vb", "sb"], observed=True).pool_rec.mean()
    keys = [k for k in w.index if ("t8",) + k in m.index and ("tp10c",) + k in m.index]
    ww = np.array([w[k] for k in keys], float)
    return np.sum(ww * np.array([m[("tp10c",) + k] - m[("t8",) + k] for k in keys])) / ww.sum(), int(ww.sum())
obs, ncov = strat(R)
rng = np.random.default_rng(0); B = []
A, T = R[R.set == "t8"], R[R.set == "tp10c"]
for _ in range(2000):
    B.append(strat(pd.concat([A.sample(len(A), replace=True, random_state=int(rng.integers(1e9))),
                              T.sample(len(T), replace=True, random_state=int(rng.integers(1e9)))]))[0])
print(f"stratified difference tp10c - t8: {obs:+.1%} [{np.percentile(B, 2.5):+.1%}, {np.percentile(B, 97.5):+.1%}] "
      f"({ncov} of {len(A)} T8 problems in strata that both sets fill)")
print("\n== coverage parse of the ground truth: fully explained or not")
def full(tokens):
    lab, lo, _ = COV.decompose(tokens)
    strict = not any(k.startswith("mixed") for k in lab)
    rule = all(len(x) == 2 and x[1] == 1 for x in lo)   # every mixed chain: k copies of one template + one part
    return strict, strict or rule
R[["full_strict", "full_rule"]] = [full(gt[p]) for p in R.problem]
for col in ("full_strict", "full_rule"):
    print(f"-- {col}")
    print(R.groupby(["set", col]).pool_rec.agg(["mean", "size"]).round(3).to_string())
