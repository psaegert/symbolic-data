"""Premise test (PLAN.md step 0): does T8-20M recover round-10c problems worse than problems from its own prior?

Pools (T8-20M, 512 draws, T = 1, run_probe_temp.py, identical settings):
  t8      front_study/probe/t8-20m-training: 100 problems from T8's catalog (seed 20261116)
  tp10c   premise/probe/t8-20m-tp10c: 200 problems from the round-10c catalog (seed 20261121), no selection
  struct  premise/probe/t8-20m-tp10cstruct: 60 round-10c problems selected by expression class (seed 20261122)
  nonlin  premise/probe/t8-20m-tp10cnonlin: 60 selected the same way, weighted sums of single variables excluded
          (seed 20261123; added after the first two sets showed 41 of 48 polynomials to be such linear sums)
Per problem: pool recovery (any draw validation-numerically recovered, srbf's vNRR), share of bad draws (unfitted or
FVU >= 0.5), used variables, structural size of the ground truth (operators + variables), class flags (classify.py).
Comparisons: recovery rate per set with a Wilson interval; a logistic fit of recovery on set + used variables + size
(the sets differ in size and variable count); recovery by class pooled over sets.
"""
import pickle
import sys
from collections import Counter

import numpy as np
import pandas as pd
from simplipy import SimpliPyEngine

sys.path.insert(0, "/home/psaegert/Projects/flash-ansr-rl/scripts")
sys.path.insert(0, "/home/psaegert/Projects/flash-ansr-rl/runs/front_study")
sys.path.insert(0, "/home/psaegert/Projects/flash-ansr-rl/runs/prior_structure")
from training_check import per_draw, struct_size  # noqa: E402
from classify import classify, has_var, motif, parse, terms  # noqa: E402

FS = "/home/psaegert/Projects/flash-ansr-rl/runs/front_study"
PR = "/home/psaegert/Projects/flash-ansr-rl/runs/term_prior_20m/premise"
SETS = {"t8": (f"{FS}/probe/t8-20m-training", f"{FS}/data/training_problems.pkl"),
        "tp10c": (f"{PR}/probe/t8-20m-tp10c", f"{PR}/data/tp10c_problems.pkl"),
        "struct": (f"{PR}/probe/t8-20m-tp10cstruct", f"{PR}/data/tp10cstruct_problems.pkl"),
        "nonlin": (f"{PR}/probe/t8-20m-tp10cnonlin", f"{PR}/data/tp10cnonlin_problems.pkl")}
eng = SimpliPyEngine.load("acj-5-4-llm")
ar = dict(eng.operator_arity)
VS = {f"x{i}" for i in range(1, 18)}


def _lin(t):
    if isinstance(t, str):
        return t in VS
    if t[0] in ("*", "/"):
        return (sum(has_var(c, VS) for c in t[1]) == 1 and not (t[0] == "/" and has_var(t[1][1], VS))
                and all(isinstance(c, str) or not has_var(c, VS) or _lin(c) for c in t[1]))
    if t[0] == "neg":
        return _lin(t[1][0])
    return False


def classes(toks):
    c = classify(toks, ar, VS)
    T = [t for t in terms(parse(toks, ar)) if has_var(t, VS)]
    ms = Counter(motif(t, VS) for t in T if motif(t, VS) != "V")
    return {"poly3": c["poly_ge3"], "rat2": c["sum_rat_ge2"], "sup3": c["superpos_ge3"],
            "rep": bool(ms) and ms.most_common(1)[0][1] >= 2, "trans": c["transcendental"],
            "linear": bool(T) and all(_lin(t) for t in T)}


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


rows = []
for name, (pdir, ppath) in SETS.items():
    P = pickle.load(open(ppath, "rb"))
    for i, (e, r) in enumerate(zip(P["benchmark_eq_id"], P["eval_row_index"])):
        stem = f"{e}__{r}"
        toks = [str(t) for t in P["ground_truth_prefix"][i]]
        try:
            L, F, R, cls, meta = per_draw(pdir, stem)
        except FileNotFoundError:
            continue
        if len(F) < 5:
            continue
        bad = R.y.isna().values | (R.y.values >= np.log10(0.5))
        rows.append(dict(set=name, problem=stem, nvar=int(P["n_used_variables"][i]), size=struct_size(toks),
                         tokens=len(toks), pool_rec=bool(F.vnrr.any()), bad=float(bad.mean()),
                         selected_as=P["cls"][i].get("selected_as") if "cls" in P else None,
                         **classes(toks)))
D = pd.DataFrame(rows)
D.to_csv(f"{PR}/premise_problems.csv", index=False)
print("problems with a pool:", D.groupby("set").size().to_dict())

print("\n== recovery and bad draws by set")
for name, g in D.groupby("set", sort=False):
    k, n = int(g.pool_rec.sum()), len(g)
    lo, hi = wilson(k, n)
    print(f"{name:7s} n={n:3d} pool recovery {k / n:6.1%} [{lo:.1%}, {hi:.1%}] | median bad {g.bad.median():.0%} | "
          f"median used vars {g.nvar.median():.0f} | median size {g['size'].median():.0f} | "
          f"poly3 {g.poly3.mean():.0%} sup3 {g.sup3.mean():.0%} rat2 {g.rat2.mean():.0%} rep {g.rep.mean():.0%} trans {g.trans.mean():.0%}")

print("\n== recovery by used variables (bins) and set")
D["vbin"] = pd.cut(D.nvar, [0, 2, 4, 6, 17], labels=["1-2", "3-4", "5-6", "7+"])
print(D.pivot_table(index="vbin", columns="set", values="pool_rec", aggfunc=["mean", "size"], observed=False).round(2).to_string())
print("\n== recovery by structural size (bins) and set")
D["sbin"] = pd.cut(D["size"], [0, 5, 10, 15, 40], labels=["<=5", "6-10", "11-15", "16+"])
print(D.pivot_table(index="sbin", columns="set", values="pool_rec", aggfunc=["mean", "size"], observed=False).round(2).to_string())

print("\n== logistic fit: pool_rec ~ set + nvar + size (t8 = reference)")
try:
    import statsmodels.formula.api as smf
    m = smf.logit("pool_rec ~ C(set, Treatment('t8')) + nvar + size", data=D.assign(pool_rec=D.pool_rec.astype(int))).fit(disp=0)
    print(m.summary2().tables[1].round(3).to_string())
except Exception as exc:  # noqa: BLE001
    print("statsmodels fit failed:", exc)

print("\n== recovery by class, pooled over all three sets (each problem once)")
for c in ("poly3", "sup3", "rat2", "rep", "trans"):
    for v in (True, False):
        g = D[D[c] == v]
        lo, hi = wilson(int(g.pool_rec.sum()), len(g))
        print(f"{c:6s}={str(v):5s} n={len(g):3d} recovery {g.pool_rec.mean():6.1%} [{lo:.1%}, {hi:.1%}] | median size {g['size'].median():.0f} | median vars {g.nvar.median():.0f}")

print("\n== structured problems vs unstructured round-10c controls, stratified (used variables x size x transcendental)")
D["struct_nl"] = (D.poly3 | D.sup3 | D.rat2) & ~D.linear
D["struct_lin"] = (D.poly3 | D.sup3) & D.linear
C = D[(D.set == "tp10c") & ~(D.poly3 | D.sup3 | D.rat2)]
D["vb"] = pd.cut(D.nvar, [0, 4, 6, 17], labels=["1-4", "5-6", "7+"])
D["sb"] = pd.cut(D["size"], [0, 15, 22, 40], labels=["<=15", "16-22", "23+"])
C = D.loc[C.index]
rng = np.random.default_rng(0)


def strat(G, C):
    w = G.groupby(["vb", "sb", "trans"], observed=True).size()
    mc = C.groupby(["vb", "sb", "trans"], observed=True).pool_rec.mean()
    keys = [k for k in w.index if k in mc.index]
    if not keys:
        return np.nan, np.nan, 0
    g = G.set_index(["vb", "sb", "trans"]).loc[keys]
    exp = np.sum([w[k] * mc[k] for k in keys]) / np.sum([w[k] for k in keys])
    return g.pool_rec.mean(), exp, len(g)


for label, mask in (("non-linear structured", D.struct_nl), ("linear sums (poly/one-form)", D.struct_lin),
                    ("rational sums", D.rat2)):
    G = D[mask & D.set.isin(["struct", "nonlin", "tp10c"])]
    obs, exp, n = strat(G, C)
    B = []
    for _ in range(2000):
        Gb = G.sample(len(G), replace=True, random_state=int(rng.integers(1e9)))
        Cb = C.sample(len(C), replace=True, random_state=int(rng.integers(1e9)))
        o, e, _ = strat(Gb, Cb)
        B.append(o - e)
    print(f"{label:28s} n={len(G):3d} (matched {n}) recovery {G.pool_rec.mean():5.1%} | matched-stratum controls {exp:5.1%} | "
          f"difference {obs - exp:+.1%} [{np.nanpercentile(B, 2.5):+.1%}, {np.nanpercentile(B, 97.5):+.1%}] | controls n={len(C)}")
D.to_csv(f"{PR}/premise_problems.csv", index=False)
