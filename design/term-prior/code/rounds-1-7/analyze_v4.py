import json, re, warnings, collections
warnings.filterwarnings("ignore")
import numpy as np
from simplipy import SimpliPyEngine
e = SimpliPyEngine.load("acj-5-4-llm", install=True, modes=("f64",))
out = json.load(open("compose_v4_rows.json"))
FAM = {"abs": ["abs"], "exp": ["exp"], "hyperbolic": ["sinh", "cosh", "tanh"], "inverse_hyperbolic": ["asinh", "acosh", "atanh"],
       "inverse_trig": ["asin", "acos", "atan"], "log": ["log"], "trig": ["sin", "cos", "tan"]}
FOF = {op: f for f, ops in FAM.items() for op in ops}
plain = [json.loads(l) for l in open("draws.jsonl")][:4000]
pf = [len({FOF[t] for t in r["sk"].split() if t in FOF}) for r in plain]
print(f"plain LC: families>=2 among draws with a function {100*np.mean([f >= 2 for f in pf if f >= 1]):.0f}%  len<=5 {100*np.mean([r['n_tok'] <= 5 for r in plain]):.1f}%")
ks = ["poly", "poly_ge3", "poly_1v_ge3", "sum_rat_ge2", "affine_ge2v", "superpos_ge3", "rep_nontrivial", "transcendental"]
for name, d in out.items():
    R = d["rows"]; C = d["copies"]; L = np.array([r["len"] for r in R])
    held = [r["held"] for r in R if "held" in r]
    fam = [r["n_fam"] for r in R]
    print(f"\n### {name}")
    print(f"  len med {np.median(L):.0f} p90 {np.percentile(L,90):.0f} >35 {100*np.mean(L>35):.1f}% >40 {100*np.mean(L>40):.1f}% | len<=5 {100*np.mean(L<=5):.1f}% | vars med {np.median([r['nv'] for r in R]):.0f} univariate {100*np.mean([r['nv']==1 for r in R]):.0f}% | held {100*np.mean(held):.1f}%")
    print("  " + " ".join(f"{k} {100*np.mean([r[k] for r in R]):.2f}" for k in ks) + f" | terms>=2 {100*np.mean([r['k_terms']>=2 for r in R]):.0f}%")
    print(f"  families>=2 among draws with a function: {100*np.mean([f >= 2 for f in fam if f >= 1]):.0f}%  (draws with any function {100*np.mean([f >= 1 for f in fam]):.0f}%)")
    nc = np.array([r["n_copies"] for r in R]); nr = np.array([r["n_redundant"] for r in R])
    print(f"  copies per draw mean {nc.mean():.2f}; draws with >=2 copies {100*np.mean(nc>=2):.0f}%; redundant copies {100*nr.sum()/max(nc.sum(),1):.1f}% of all copies, "
          f"{100*nr.sum()/max(sum(max(c-1,0) for c in nc),1):.1f}% of re-uses (copies beyond the first); draws with >=1 redundant {100*np.mean(nr>=1):.1f}%")
    cat_ = collections.defaultdict(lambda: [0, 0])
    for c in C:
        k = "constants only" if not (c["var"] or c["exp"] or c["struct"]) else "+".join(x for x in ("var", "exp", "struct") if c[x])
        cat_[k][0] += 1; cat_[k][1] += c["redundant"]
    print("  redundancy by what the copy changed: " + "; ".join(f"{k}: {v[1]}/{v[0]} ({100*v[1]/v[0]:.0f}%)" for k, v in sorted(cat_.items(), key=lambda kv: -kv[1][0])))
    sh = collections.defaultdict(list)
    for r in R: sh[min(r["n_copies"], 4)].append(r["len"] / r["raw_len"])
    print("  simplified/raw length by #copies: " + "  ".join(f"{k}{'+' if k==4 else ''}: {np.median(v):.2f}" for k, v in sorted(sh.items())))
def short(s, n=140):
    s = re.sub(r"(\d+\.\d{3})\d+", r"\1", s)
    return s if len(s) <= n else s[:n-3] + "..."
for name in list(out)[1:]:
    print(f"\n=== {name}: 10 draws ===")
    for r in out[name]["rows"][:10]:
        H = r["H"]
        print(f"  [{r['len']:3d} tok, {r['nv']} var{', HELD' if r.get('held') else ''}] {short(e.prefix_to_infix(r['sk'].split()))}")
        print(f"        n0={H['n0']} B={H['B']} n1={H['n1']} L={H['L']} M={H['M']} pool={H['pool']} uses={H['uses']} copies={r['n_copies']} redundant={r['n_redundant']} r={H['r']:.2f} new={'y' if H['new_vars'] else 'n'}")
