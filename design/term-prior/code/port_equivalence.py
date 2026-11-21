import os, sys, copy, warnings, time
warnings.filterwarnings("ignore")
os.environ["POOL7_FN_WEIGHTS"] = "hyper=0.5"
WT = "/home/psaegert/Projects/symbolic-data/.claude/worktrees/term-prior"
sys.path.insert(0, WT + "/src")
import symbolic_data                                   # this branch's package, before pool7 inserts its own path
from symbolic_data.term_prior import TermPriorCatalog
sys.path.insert(0, WT + "/design/term-prior/code"); os.chdir(WT + "/design/term-prior/code")
import numpy as np, pool7 as p, pool8 as q
assert symbolic_data.__file__.startswith(WT), symbolic_data.__file__
cat = TermPriorCatalog.from_config(dict(copy.deepcopy(p.cfg), term_prior={}))
tp = cat._sampler()
N = int(sys.argv[1]) if len(sys.argv) > 1 else 300   # 300 of 300 identical on 2026-11-21
r1, r2 = np.random.default_rng(424242), np.random.default_rng(424242)
t0 = time.time(); same = 0
for i in range(N):
    a = q.draw_r8(r1, q.LADDER["r10c"]); s, raw, B, rec = tp.draw(r2)
    ok = a["simp"] == s and a["raw"] == raw and a["B"] == B
    same += ok
    if not ok: print("DIFF at", i, a["B"], B, " ".join(a["raw"][:20]), "|", " ".join(raw[:20])); break
print(f"identical {same} of {N} draws ({time.time() - t0:.1f}s)")
r3 = np.random.default_rng(7); sk = [cat.sample_skeleton(new=True, decontaminate=False, rng=r3) for _ in range(50)]
print("sample_skeleton ok:", len(sk), "max len", max(len(x[0]) for x in sk), "| example:", " ".join(sk[0][0]))
