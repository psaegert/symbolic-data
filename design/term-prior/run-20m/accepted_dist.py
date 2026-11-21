"""The accepted training distribution: class rates among draws of at most 35 tokens that the precise key keeps
(T8's draws are capped already). Run from flash-ansr-rl/runs/prior_structure/explorer (the 3,000-draw rows)."""
import gzip, json, numpy as np
D = {}
for f in ("rows_base.json.gz", "rows_ladder.json.gz", "rows_r10c.json.gz"):
    D.update(json.load(gzip.open(f)))
def rates(R):
    f = lambda c: 100 * np.mean([r["cls"][c] for r in R])
    return (f"n={len(R):5d} poly3 {f('poly3'):.2f}% sup3 {f('sup3'):.2f}% rat2 {f('rat2'):.2f}% rep {f('rep'):.1f}% "
            f"transcendental {f('trans'):.1f}% | median variables {np.median([r['cls']['nv'] for r in R]):.0f} tokens {np.median([len(r['simp']) for r in R]):.0f}")
for k in ("t8", "r8", "r10c"):
    R = D[k]; C = [r for r in R if len(r["simp"]) <= 35]; T = [r for r in C if not r.get("held")]
    print(f"{k:5s} all          {rates(R)}"); print(f"{k:5s} <=35 tokens  {rates(C)}"); print(f"{k:5s} <=35, kept   {rates(T)}")
