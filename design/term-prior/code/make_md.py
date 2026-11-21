"""First 100 draws (unfiltered) of the T8 prior and the round-8 base as a markdown file."""
import os, sys, json, gzip, warnings
warnings.filterwarnings("ignore")
os.environ["POOL7_FN_WEIGHTS"] = "hyper=0.5"          # round 8 base; T8 draws come from the stored run below
import numpy as np, pool7 as p, pool8 as q
t8 = [r["simp"] for r in json.load(gzip.open("r8_rows.json.gz", "rt"))["t8"][:100]]   # draw_t8, seed 20261122 + 7
rng = np.random.default_rng(20261122 + 8)                                              # same seed as gen8's base run
r8 = [q.draw_r8(rng, q.LADDER["r8"]) for _ in range(100)]
def infix(s):
    try: return __import__("pretty").pretty(s)
    except Exception: return " ".join(s)
def nv(s): return len({x for x in s if x in p.vset})
out = ["# 100 draws each: T8 prior and round 8 (base)", "",
       "Generated 2026-11-20. The first 100 draws of each prior, unfiltered, after simplification (permissive canon), as the model is trained on them; constants show their sampled values. `[t tok, v var]` gives the length in prefix tokens and the number of distinct variables.", "",
       "- **T8 prior**: T8.1 as trained (35-token cap, cancellations resampled).",
       "- **Round 8 (base)**: the unified term prior with recursion, reuse, inheritance and copy changes, budget ×1.3 (step 8f), and the six hyperbolic functions (sinh, cosh, tanh, asinh, acosh, atanh) at half weight. No length cap.", ""]
for title, rows in [("T8 prior", t8), ("Round 8 (base)", [r["simp"] for r in r8])]:
    L = [len(s) for s in rows]
    out += [f"## {title}", "", f"Median {int(np.median(L))} tokens, {sum(nv(s) == 1 for s in rows)} of 100 with one variable.", ""]
    out += [f"{i}. `{infix(s)}` [{len(s)} tok, {nv(s)} var]" for i, s in enumerate(rows, 1)] + [""]
open("draws_t8_r8.md", "w").write("\n".join(out))
print("wrote draws_t8_r8.md", len(t8), len(r8))
