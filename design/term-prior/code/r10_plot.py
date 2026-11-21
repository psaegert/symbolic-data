"""Structure scores and length: T8, round 8, round 9, real formulas (srbf venv)."""
import json, gzip
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
D = json.load(gzip.open("r10_plotdata.json.gz", "rt"))
SHOW = [("t8", "T8 prior", "#2a78d6", "-"), ("r8", "round 8 (base)", "#8a897f", "-"), ("r10b", "round 10b (s mean 0.25)", "#1baf7a", "--"), ("r10c", "round 10c (s mean 0.5)", "#1baf7a", "-"), ("corpus", "real formulas", "#eda100", "--")]
INK, MUTED, GRID, BG = "#1f1f1d", "#55554f", "#e6e5e0", "#fcfcfb"
fig, ax = plt.subplots(1, 4, figsize=(19, 4.3), constrained_layout=True)
def ecdf(a, i, title, xl):
    for k, lab, c, ls in SHOW:
        x = np.sort([r[i] for r in D[k]]); y = np.arange(1, len(x) + 1) / len(x)
        a.step(x, y, where="post", color=c, ls=ls, lw=2, label=lab)
    a.set_title(title, loc="left"); a.set_xlim(*xl); a.set_ylim(0, 1); a.set_ylabel("share <= x")
ecdf(ax[0], 0, "Repetition: share of nodes in a repeated subtree", (0, 1))
ecdf(ax[3], 3, "Length after simplification (tokens)", (0, 70))
ax[3].axvline(35, color="#8a897f", lw=1, ls=(0, (2, 2))); ax[3].text(35.6, 0.04, "T8 cap 35", fontsize=8, color=MUTED)
w = 0.16
for j, (k, lab, c, ls) in enumerate(SHOW):
    alt = np.array([r[1] for r in D[k]]); xs = np.arange(6)
    share = [np.mean(np.minimum(alt, 5) == v) for v in xs]
    ax[1].bar(xs + (j - 2) * w, share, w * 0.92, color=c, label=lab, hatch="//" if ls == "--" else None, edgecolor=BG if ls != "--" else c, fill=ls != "--")
    tops = ["sum", "product", "function", "power", "atom"]
    ax[2].bar(np.arange(5) + (j - 2) * w, [np.mean([r[2] == t for r in D[k]]) for t in tops], w * 0.92, color=c, label=lab,
              hatch="//" if ls == "--" else None, edgecolor=BG if ls != "--" else c, fill=ls != "--")
ax[1].set_xticks(range(6), ["0", "1", "2", "3", "4", "5+"]); ax[1].set_title("Alternation: most + − / × ÷ switches on a path", loc="left"); ax[1].set_ylabel("share of draws")
ax[2].set_xticks(range(5), ["sum", "product", "function", "power", "atom"]); ax[2].set_title("Kind of the root", loc="left"); ax[2].set_ylabel("share of draws")
for a in ax:
    a.title.set_fontsize(11); a.title.set_color(INK); a.grid(axis="y", color=GRID, lw=0.8); a.set_axisbelow(True); a.set_facecolor(BG)
    for s in ("top", "right"): a.spines[s].set_visible(False)
    for s in ("left", "bottom"): a.spines[s].set_color("#b5b4ad")
    a.tick_params(colors=MUTED, labelsize=9); a.yaxis.label.set_color(MUTED); a.yaxis.label.set_fontsize(9)
ax[0].legend(frameon=False, fontsize=9, loc="lower right")
fig.suptitle(f"Structure scores after simplification: 3,000 draws per prior, {len(D['corpus']):,} real formulas", fontsize=11, x=0.01, ha="left", color=INK)
fig.patch.set_facecolor(BG); fig.savefig("r10_structure.png", dpi=150)
print("wrote r10_structure.png")
