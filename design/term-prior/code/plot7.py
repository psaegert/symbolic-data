import json, sys, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, "../prior"); from classify import classify
from simplipy import SimpliPyEngine
e = SimpliPyEngine.load("acj-5-4-llm", install=True, modes=("f64",)); ar = dict(e.operator_arity); vset = {f"x{i}" for i in range(1, 18)}
T8 = [json.loads(l)["sk"].split() for l in open("../prior/draws.jsonl")][:4000]
bank = json.load(open("samples.json")); R6 = [r["simp"] for r in bank if r["prior"] == "pool4"]
R7 = [r["simp"] for r in json.load(open("r7_n_1.3.json"))]; R7c = [r["simp"] for r in json.load(open("r7_nc_1.3.json"))]
V = {"T8 prior": T8, "round 6, n1 <= 4": R6, "round 7, B x1.3": R7, "round 7, B x1.3, corpus fn weights": R7c}
def feats(t):
    c = classify(t, ar, vset); return len(t), sum(1 for x in t if x in ar), c["k_terms"], c["nv"]
F = {k: np.array([feats(t) for t in v]) for k, v in V.items()}
colors = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]; styles = ["-", "--", "-.", ":"]
fig, axes = plt.subplots(1, 4, figsize=(18, 4.2), constrained_layout=True)
titles = ["Expression length (tokens, simplified)", "Operators (simplified)", "Top-level additive terms with a variable", "Distinct variables"]
for i, (ax, title) in enumerate(zip(axes, titles)):
    for (name, f), c, ls in zip(F.items(), colors, styles):
        x = np.sort(f[:, i]); y = np.arange(1, len(x) + 1) / len(x)
        ax.step(x, y, where="post", color=c, ls=ls, lw=2, label=name)
    ax.set_title(title, fontsize=11, loc="left", color="#1f1f1d"); ax.set_ylim(0, 1)
    ax.grid(axis="y", color="#e6e5e0", lw=0.8); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color("#b5b4ad")
    ax.tick_params(colors="#55554f", labelsize=9); ax.set_ylabel("share of draws <= x", fontsize=9, color="#55554f"); ax.set_facecolor("#fcfcfb")
axes[0].axvline(35, color="#8a897f", lw=1, ls=(0, (2, 2))); axes[0].text(35.5, 0.04, "T8 cap 35", fontsize=8, color="#55554f")
axes[0].set_xlim(0, 90); axes[1].set_xlim(0, 45); axes[2].set_xlim(0, 12); axes[3].set_xlim(0, 17)
axes[3].legend(frameon=False, fontsize=9, loc="lower right")
fig.suptitle("Size ECDFs: T8 prior vs round 6 and round 7 (recursive pool, nesting fix, budget x1.3); 1,500-4,000 draws each", fontsize=11, x=0.01, ha="left", color="#1f1f1d")
fig.patch.set_facecolor("#fcfcfb"); fig.savefig("size_ecdf_round7.png", dpi=150)
