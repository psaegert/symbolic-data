import json, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
d = json.load(open("rows.json")); a = np.load("analysis.npz")
R = {k: np.array(d[k]["rows"]) for k in ["A", "B", "C", "T8"]}; n = {k: d[k]["n"] for k in R}
q0 = {k: a[f"{k}_q0"] for k in R}; T = {k: a[f"t_{k}"] for k in ["t8ext", "stretch", "decay"]}
def wts(k, t):
    N = R[k][:, 0].astype(int); w = np.where(N >= 1, T[t][N] / np.maximum(q0[k][N], 1e-300), 0); return w / w.sum()
INK, MUTED, GRID, BG = "#1f1f1d", "#55554f", "#e6e5e0", "#fcfcfb"
BLUE, ORANGE, GREEN, AMBER, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#8a897f"
fig, ax = plt.subplots(1, 4, figsize=(19, 4.4), constrained_layout=True)
Ns = np.arange(1, 101)
# (a) length pmf
x = ax[0]
x.plot(np.arange(0, 18), q0["T8"][:18], color=BLUE, lw=2, label="T8 prior")
x.plot(Ns, q0["A"][1:], color=GRAY, lw=1.5, label=f"current spec, unconditioned ({d['A']['aborted']/n['A']:.0%} above 100)")
x.plot(Ns, T["t8ext"][1:], color=ORANGE, lw=2, ls="-", label="target: T8 form exp(N^0.7) to 50, tail")
x.plot(Ns, T["stretch"][1:], color=ORANGE, lw=2, ls="--", label="target: T8 shape stretched 17 -> 50, tail")
x.plot(Ns, T["decay"][1:], color=GREEN, lw=2, ls="-.", label="target: decaying exp(-N/15)")
x.set_yscale("log"); x.set_ylim(1e-5, 0.4); x.set_xlim(0, 100)
x.set_title("Length distribution (operators)", loc="left"); x.set_ylabel("probability")
x.legend(frameon=False, fontsize=8, loc="lower right", bbox_to_anchor=(1.0, 0.0))
# (b) size ECDF
x = ax[1]
def ecdf_from_pmf(p, color, ls, label, scale=1.0):
    c = np.cumsum(p) * scale; x.step(np.arange(len(p)), c, where="post", color=color, ls=ls, lw=2, label=label)
ecdf_from_pmf(q0["T8"][:101], BLUE, "-", "T8 prior")
ecdf_from_pmf(q0["A"], GRAY, "-", "current spec, unconditioned")
ecdf_from_pmf(T["stretch"], ORANGE, "--", "conditioned on stretched-T8 target")
ecdf_from_pmf(T["decay"], GREEN, "-.", "conditioned on decaying target")
x.annotate(f"{1-q0['A'].sum():.0%} of current-spec draws\nexceed 100 operators", (100, q0["A"].sum()), (64, 0.28), fontsize=8, color=MUTED, arrowprops=dict(arrowstyle="-", color=GRAY, lw=0.8))
x.set_xlim(0, 100); x.set_ylim(0, 1); x.set_title("Operators, ECDF", loc="left"); x.set_ylabel("share of draws <= x")
x.legend(frameon=False, fontsize=8, loc="lower right")
# (c) composite share vs length
x = ax[2]
lab = {"A": ("spec: all patterns p_T8 (n <= 17)", GRAY, "-"), "B": ("deeper layers capped 8, 4, 2, 1", AMBER, "--"), "C": ("as B, root size uniform 0..17", GREEN, "-")}
for k, (l, c, ls) in lab.items():
    N = R[k][:, 0].astype(int); comp = R[k][:, 4] >= 2
    cnt = np.bincount(N, minlength=101)[1:101]; cc = np.bincount(N, weights=comp, minlength=101)[1:101]
    x.plot(Ns, cc / np.maximum(cnt, 1), color=c, ls=ls, lw=2, label=l)
x.axvline(17.5, color=GRAY, lw=1, ls=(0, (2, 2))); x.text(19, 0.45, "largest single\npattern: 17", fontsize=8, color=MUTED)
x.set_xlim(0, 100); x.set_ylim(0, 1.02); x.set_title("Share of draws with a replacement, given the length", loc="left")
x.set_ylabel("share composite"); x.legend(frameon=False, fontsize=8, loc="lower right")
# (d) top-level additive terms ECDF
x = ax[3]
def wecdf(vals, w, color, ls, label):
    xs = np.arange(1, 16); c = [w[vals <= v].sum() for v in xs]; x.step(xs, c, where="post", color=color, ls=ls, lw=2, label=label)
wecdf(R["T8"][:, 2], np.full(len(R["T8"]), 1 / len(R["T8"])), BLUE, "-", "T8 prior")
wecdf(R["A"][:, 2], wts("A", "stretch"), GRAY, "-", "spec | stretched-T8 target")
wecdf(R["C"][:, 2], wts("C", "stretch"), GREEN, "-", "variant C | stretched-T8 target")
wecdf(R["C"][:, 2], wts("C", "decay"), GREEN, "-.", "variant C | decaying target")
x.set_xlim(1, 12); x.set_ylim(0, 1); x.set_title("Top-level additive terms, ECDF", loc="left"); x.set_ylabel("share of draws <= x")
x.legend(frameon=False, fontsize=8, loc="lower right")
for x in ax:
    x.title.set_fontsize(11); x.title.set_color(INK); x.grid(axis="y", color=GRID, lw=0.8); x.set_axisbelow(True); x.set_facecolor(BG)
    for s in ("top", "right"): x.spines[s].set_visible(False)
    for s in ("left", "bottom"): x.spines[s].set_color("#b5b4ad")
    x.tick_params(colors=MUTED, labelsize=9); x.yaxis.label.set_color(MUTED); x.yaxis.label.set_fontsize(9)
fig.suptitle("Length conditioning of the term-pool prior: 1.5M prior draws per variant (aborted above 100 operators), 200k T8 draws", fontsize=11, x=0.01, ha="left", color=INK)
fig.patch.set_facecolor(BG); fig.savefig("length_conditioning.png", dpi=150)
