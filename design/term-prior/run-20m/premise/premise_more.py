"""Follow-ups to premise_analyze.py: what the structured classes contain, and set differences within strata."""
import pickle, sys
import numpy as np, pandas as pd
from simplipy import SimpliPyEngine
sys.path.insert(0, "/home/psaegert/Projects/flash-ansr-rl/runs/prior_structure")
from classify import classify, parse, terms, has_var, is_poly
eng = SimpliPyEngine.load("acj-5-4-llm"); ar = dict(eng.operator_arity); VS = {f"x{i}" for i in range(1, 18)}
D = pd.read_csv("premise_problems.csv")
P = {n: pickle.load(open(f"data/{f}_problems.pkl", "rb")) for n, f in (("tp10c", "tp10c"), ("struct", "tp10cstruct"))}
P["t8"] = pickle.load(open("/home/psaegert/Projects/flash-ansr-rl/runs/front_study/data/training_problems.pkl", "rb"))
gt = {f"{e}__{r}": [str(t) for t in p] for S in P.values() for e, r, p in zip(S["benchmark_eq_id"], S["eval_row_index"], S["ground_truth_prefix"])}
def degree1(tree):
    """every variable term is (constant times) a single variable"""
    T = [t for t in terms(tree) if has_var(t, VS)]
    def lin(t):
        if isinstance(t, str): return t in VS
        if t[0] in ("*", "/"): return sum(has_var(c, VS) for c in t[1]) == 1 and all(isinstance(c, str) or not has_var(c, VS) or lin(c) for c in t[1]) and not (t[0] == "/" and has_var(t[1][1], VS))
        if t[0] == "neg": return lin(t[1][0])
        return False
    return bool(T) and all(lin(t) for t in T)
D["linear"] = [degree1(parse(gt[p], ar)) for p in D.problem]
print("== linear (weighted sum of single variables) among classes")
for c in ("poly3", "sup3", "rat2"):
    g = D[D[c]]
    print(f"{c}: n={len(g)} linear {g.linear.mean():.0%} | recovery linear {g[g.linear].pool_rec.mean():.0%} (n={g.linear.sum()}) non-linear {g[~g.linear].pool_rec.mean():.0%} (n={(~g.linear).sum()})")
print("overlap poly3 & sup3:", int((D.poly3 & D.sup3).sum()), "of", int(D.poly3.sum()), "/", int(D.sup3.sum()))
print("\n== examples: struct set, non-linear, by selection")
for sel in ("poly3", "sup3", "rat2"):
    g = D[(D.set == "struct") & (D.selected_as == sel)]
    print(f"-- selected as {sel}: n={len(g)}, linear {g.linear.mean():.0%}, recovery {g.pool_rec.mean():.0%}")
    for _, r in g[~g.linear].head(4).iterrows():
        try: inf = eng.prefix_to_infix(gt[r.problem])
        except Exception: inf = " ".join(gt[r.problem])
        print(f"   rec={int(r.pool_rec)} vars={r.nvar} {inf[:140]}")
print("\n== t8 vs tp10c within strata (transcendental x used-variable bin)")
D["vbin"] = pd.cut(D.nvar, [0, 4, 6, 17], labels=["1-4", "5-6", "7+"])
R = D[D.set.isin(["t8", "tp10c"])]
print(R.pivot_table(index=["trans", "vbin"], columns="set", values="pool_rec", aggfunc=["mean", "size"], observed=False).round(2).to_string())
# stratified difference (tp10c - t8), weights = t8 stratum sizes, bootstrap over problems within sets
rng = np.random.default_rng(0)
def strat_diff(R):
    w = R[R.set == "t8"].groupby(["trans", "vbin"], observed=True).size()
    m = R.groupby(["set", "trans", "vbin"], observed=True).pool_rec.mean()
    keys = [k for k in w.index if ("t8",) + k in m.index and ("tp10c",) + k in m.index]
    ww = np.array([w[k] for k in keys], float)
    return np.sum(ww * np.array([m[("tp10c",) + k] - m[("t8",) + k] for k in keys])) / ww.sum()
obs = strat_diff(R); B = []
A, T = R[R.set == "t8"], R[R.set == "tp10c"]
for _ in range(2000):
    B.append(strat_diff(pd.concat([A.sample(len(A), replace=True, random_state=rng.integers(1e9)), T.sample(len(T), replace=True, random_state=rng.integers(1e9))])))
print(f"\nstratified recovery difference tp10c - t8 (t8 stratum weights): {obs:+.1%} [{np.percentile(B, 2.5):+.1%}, {np.percentile(B, 97.5):+.1%}]")
print(f"raw difference: {T.pool_rec.mean() - A.pool_rec.mean():+.1%}")
