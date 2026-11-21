"""ECDF plot of the round-8 ladder (needs matplotlib; run with the srbf venv)."""
import json, gzip
import numpy as np
D = json.load(gzip.open("rows_ladder.json.gz", "rt")); D.update(json.load(gzip.open("rows_base.json.gz", "rt")))
OPS = {"+", "-", "*", "/", "pow", "rootn", "neg", "inv", "abs", "sin", "cos", "tan", "asin", "acos", "atan", "sinh", "cosh", "tanh", "asinh", "acosh", "atanh", "exp", "log"}
def ops(t): return sum(1 for x in t if x in OPS)
# ECDFs against T8
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
SHOW = [("t8", "T8 prior", "#2a78d6", "-"), ("r7", "round 7", "#8a897f", "-"), ("r8a", "8a: T8 as a special case", "#2a78d6", (0, (1.5, 2))),
        ("r8c", "8c: + recursion + reuse", "#eb6834", "--"), ("r8f", "8f: all three, budget x1.3", "#1baf7a", "-"), ("r8", "round 8 base (hyperbolic x0.5)", "#eda100", "--")]
fig, axes = plt.subplots(1, 4, figsize=(18, 4.2), constrained_layout=True)
spec = [("Length after simplification (tokens)", lambda r: len(r["simp"]), (0, 70)), ("Operators after simplification", lambda r: ops(r["simp"]), (0, 35)),
        ("Top-level additive terms", lambda r: r["cls"]["k"], (0, 12)), ("Distinct variables", lambda r: r["cls"]["nv"], (0, 17))]
for ax, (title, f, xl) in zip(axes, spec):
    for k, lab, c, ls in SHOW:
        x = np.sort([f(r) for r in D[k]]); y = np.arange(1, len(x) + 1) / len(x)
        ax.step(x, y, where="post", color=c, ls=ls, lw=2, label=lab)
    ax.set_title(title, fontsize=11, loc="left", color="#1f1f1d"); ax.set_xlim(*xl); ax.set_ylim(0, 1)
    ax.grid(axis="y", color="#e6e5e0", lw=0.8); ax.set_axisbelow(True); ax.set_facecolor("#fcfcfb")
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color("#b5b4ad")
    ax.tick_params(colors="#55554f", labelsize=9); ax.set_ylabel("share of draws <= x", fontsize=9, color="#55554f")
axes[0].axvline(35, color="#8a897f", lw=1, ls=(0, (2, 2))); axes[0].text(35.6, 0.04, "T8 cap 35", fontsize=8, color="#55554f")
axes[3].legend(frameon=False, fontsize=9, loc="lower right")
fig.suptitle(f"Round 8 ladder against T8 and round 7: ECDFs after simplification, {len(D['t8']):,} draws each", fontsize=11, x=0.01, ha="left", color="#1f1f1d")
fig.patch.set_facecolor("#fcfcfb"); fig.savefig("r8_ecdf.png", dpi=150)

