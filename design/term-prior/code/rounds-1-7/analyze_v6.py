import json, re, sys, warnings, collections
warnings.filterwarnings("ignore")
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from simplipy import SimpliPyEngine
sys.path.insert(0, ".")
from classify import classify, parse, terms, motif, has_var
e = SimpliPyEngine.load("acj-5-4-llm", install=True, modes=("f64",))
ar = dict(e.operator_arity); vset = {f"x{i}" for i in range(1, 18)}
FAM = {"abs": ["abs"], "exp": ["exp"], "hyperbolic": ["sinh", "cosh", "tanh"], "inverse_hyperbolic": ["asinh", "acosh", "atanh"],
       "inverse_trig": ["asin", "acos", "atan"], "log": ["log"], "trig": ["sin", "cos", "tan"]}
FOF = {op: f for f, ops in FAM.items() for op in ops}
out5 = json.load(open("compose_v5_rows.json")); out6 = json.load(open("compose_v6_rows.json"))
out = {"round 5, n1 <= 4": out5["round 5, n1 <= 4"], "round 6, n1 <= 4": out6["round 6, n1 <= 4"],
       "round 5, n1 <= 8": out5["round 5, n1 <= 8"], "round 6, n1 <= 8": out6["round 6, n1 <= 8"]}
# T8 prior reference: the raw T8.1 draws (catalog cap: 35 tokens), held-out flags from the precise-key audit
t8 = [json.loads(l) for l in open("draws.jsonl")][:4000]
t8_held = json.load(open("flags_precise_closure.json"))[:4000]
T8 = []
for r, h in zip(t8, t8_held):
    sk = r["sk"].split(); c = classify(sk, ar, vset)
    T = [t for t in terms(parse(sk, ar)) if has_var(t, vset)]
    ms = collections.Counter(motif(t, vset) for t in T if motif(t, vset) != "V")
    c.update(rep_nontrivial=bool(ms) and ms.most_common(1)[0][1] >= 2, len=len(sk), ops=sum(1 for x in sk if x in ar),
             n_fam=len({FOF[x] for x in sk if x in FOF}), held=h, sk=r["sk"])
    T8.append(c)
V = {"T8 prior": T8}
for name, d in out.items(): V[name] = d["rows"]
def pct(x): return f"{100*np.mean(x):.1f}"
rows = [
 ("median / p90 tokens", lambda R: f"{np.median([r['len'] for r in R]):.0f} / {np.percentile([r['len'] for r in R], 90):.0f}"),
 ("median / p90 operators", lambda R: f"{np.median([r['ops'] for r in R]):.0f} / {np.percentile([r['ops'] for r in R], 90):.0f}"),
 ("> 35 / > 40 tokens (%)", lambda R: f"{pct([r['len']>35 for r in R])} / {pct([r['len']>40 for r in R])}"),
 ("trivial, <= 5 tokens (%)", lambda R: pct([r['len'] <= 5 for r in R])),
 ("constant laws (%)", lambda R: pct([r['nv'] == 0 for r in R])),
 ("median variables / univariate (%)", lambda R: f"{np.median([r['nv'] for r in R]):.0f} / {pct([r['nv']==1 for r in R])}"),
 ("additive terms >= 2 / >= 3 (%)", lambda R: f"{pct([r['k_terms']>=2 for r in R])} / {pct([r['k_terms']>=3 for r in R])}"),
 ("polynomial >= 3 terms (%)", lambda R: f"{100*np.mean([r['poly_ge3'] for r in R]):.2f}"),
 ("univariate polynomial >= 3 terms (%)", lambda R: f"{100*np.mean([r['poly_1v_ge3'] for r in R]):.2f}"),
 ("sum of >= 2 rational terms (%)", lambda R: f"{100*np.mean([r['sum_rat_ge2'] for r in R]):.2f}"),
 ("weighted sum of >= 2 variables (%)", lambda R: f"{100*np.mean([r['affine_ge2v'] for r in R]):.2f}"),
 (">= 3 terms of one form (%)", lambda R: f"{100*np.mean([r['superpos_ge3'] for r in R]):.2f}"),
 ("two terms share a non-trivial form (%)", lambda R: f"{100*np.mean([r['rep_nontrivial'] for r in R]):.2f}"),
 ("contains a transcendental (%)", lambda R: pct([r['transcendental'] for r in R])),
 ("mixes >= 2 families, of draws with a function (%)", lambda R: pct([r['n_fam'] >= 2 for r in R if r['n_fam'] >= 1])),
 ("held out, precise key (%)", lambda R: pct([r['held'] for r in R if 'held' in r])),
 ("raw operators / budget B, median", lambda R: f"{np.median([r['raw_ops']/r['H']['B'] for r in R if r.get('H', {}).get('B')]):.2f}" if R and 'H' in R[0] and R[0]['H'].get('B') else "-"),
]
names = list(V)
print("| | " + " | ".join(names) + " |"); print("|---" * (len(names) + 1) + "|")
for label, f in rows:
    print(f"| {label} | " + " | ".join(f(V[n]) for n in names) + " |")
for name in ("round 6, n1 <= 4", "round 6, n1 <= 8"):
    R = out[name]["rows"]
    print(f"\n{name}: by alphabet branch")
    by = collections.defaultdict(list)
    for r in R: by[r["H"].get("case", "?")].append(r)
    for case in ("tau0-full", "tau0-subset", "shared-L1", "own", "const"):
        G = by.get(case, [])
        if not G: continue
        print(f"   {case:12s} n={len(G):4d} | vars median {np.median([g['nv'] for g in G]):.0f}, univariate {pct([g['nv']==1 for g in G])}%, <=2 vars {pct([g['nv']<=2 for g in G])}% | "
              f"shared form {100*np.mean([g['rep_nontrivial'] for g in G]):.1f}%, poly>=3 {100*np.mean([g['poly_ge3'] for g in G]):.1f}%, sum rat {100*np.mean([g['sum_rat_ge2'] for g in G]):.1f}%, "
              f">=3 one form {100*np.mean([g['superpos_ge3'] for g in G]):.1f}% | tokens median {np.median([g['len'] for g in G]):.0f}")
for name in []:
    R = out[name]["rows"]; fill = [ (r["raw_ops"] - r["H"]["n0"]) / r["H"]["R"] for r in R if r["H"].get("R")]
    print(f"\n{name}: leaves' operators / remaining budget R: median {np.median(fill):.2f}, IQR {np.percentile(fill,25):.2f}-{np.percentile(fill,75):.2f}")
    sm = collections.defaultdict(list)
    for r in R:
        if r["H"].get("R") is not None and not r["H"].get("const_law"): sm[min(r["H"]["n0"], 8)].append(r["raw_ops"] / max(r["H"]["B"], 1))
    print("   raw operators / B by n0: " + "  ".join(f"n0={k}: {np.median(v):.2f}" for k, v in sorted(sm.items())))
# ---- ECDF plot
colors = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]; styles = ["-", "--", "-.", ":"]
fig, axes = plt.subplots(1, 4, figsize=(18, 4.2), constrained_layout=True)
panels = [("tokens", "len", "Expression length (tokens, simplified)"), ("operators", "ops", "Operators (simplified)"),
          ("terms", "k_terms", "Top-level additive terms with a variable"), ("vars", "nv", "Distinct variables")]
PLOT = {k: V[k] for k in ("T8 prior", "round 5, n1 <= 4", "round 6, n1 <= 4", "round 6, n1 <= 8")}
for ax, (_, key, title) in zip(axes, panels):
    for (name, R), c, ls in zip(PLOT.items(), colors, styles):
        x = np.sort([r[key] for r in R]); y = np.arange(1, len(x) + 1) / len(x)
        ax.step(x, y, where="post", color=c, ls=ls, lw=2, label=name)
    ax.set_title(title, fontsize=11, loc="left", color="#1f1f1d")
    ax.set_ylim(0, 1.0); ax.grid(axis="y", color="#e6e5e0", lw=0.8); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color("#b5b4ad")
    ax.tick_params(colors="#55554f", labelsize=9)
    ax.set_ylabel("share of draws <= x", fontsize=9, color="#55554f")
axes[0].axvline(35, color="#8a897f", lw=1, ls=(0, (2, 2))); axes[0].text(35.5, 0.04, "T8 cap 35", fontsize=8, color="#55554f")
axes[0].axvline(40, color="#8a897f", lw=1, ls=(0, (2, 2))); axes[0].text(40.5, 0.12, "40", fontsize=8, color="#55554f")
axes[0].set_xlim(0, 80); axes[1].set_xlim(0, 35); axes[2].set_xlim(0, 12); axes[3].set_xlim(0, 17)
axes[3].legend(frameon=False, fontsize=9, loc="lower right")
fig.suptitle("Size ECDFs: T8 prior vs term-pool prior (4,000 draws each; T8 is capped at 35 tokens by its catalog)", fontsize=11, x=0.01, ha="left", color="#1f1f1d")
fig.patch.set_facecolor("#fcfcfb")
for ax in axes: ax.set_facecolor("#fcfcfb")
fig.savefig("size_ecdf_round6.png", dpi=150)
def short(s, n=140):
    s = re.sub(r"(\d+\.\d{3})\d+", r"\1", s)
    return s if len(s) <= n else s[:n-3] + "..."
for name in ("round 6, n1 <= 4", "round 6, n1 <= 8"):
    print(f"\n=== {name}: 10 draws ===")
    for r in out[name]["rows"][:10]:
        H = r["H"]
        print(f"  [{r['len']:3d} tok, {r['ops']:2d} ops, {r['nv']} var{', HELD' if r.get('held') else ''}] {short(e.prefix_to_infix(r['sk'].split()))}")
        print(f"        case={H.get('case')} |A0|={H.get('A0')} |A1|={H.get('A1')} B={H['B']} n0={H['n0']} M={H['M']} pool={H['pool']} uses={H['uses']} n1={H['n1']}")
