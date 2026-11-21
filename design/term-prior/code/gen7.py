"""Round-7 traces for the explorer (B x1.3, nesting fix). POOL7_FN_WEIGHTS=corpus selects corpus-tempered function weights."""
import sys, json, collections, warnings, os
warnings.filterwarnings("ignore")
import numpy as np
import pool7 as p
tag = sys.argv[1]; N = int(sys.argv[2])
P = dict(p.P7, B_scale=1.3)
rng = np.random.default_rng(20261121 + (1 if tag == "r7c" else 0))
def redundant_copies(r):
    if r["H"].get("const_law"): return []
    t0 = p.tree_of(r["shape"]); paths = p.leaf_paths(t0); leaves = r["leaves"]
    groups = collections.defaultdict(list)
    for i, pth in enumerate(paths):
        if leaves[i]["how"] != "copy": continue
        b = p.bag(t0, pth)
        groups[(leaves[i]["pool"], b[0] if b else ("solo", pth), b[1] if b else None)].append((i, b[2] if b else 0))
    out = []
    for key, mem in groups.items():
        if key[2] is None or len(mem) < 2: continue
        kept = []
        for i, par in mem:
            ci = leaves[i]["tokens"]; li = p.slen(ci); merged = False
            for i2, par2 in kept:
                c2 = leaves[i2]["tokens"]; l2 = p.slen(c2)
                op = ("+" if par == par2 else "-") if key[2] == "A" else ("*" if par == par2 else "/")
                lj = p.slen([op] + c2 + ci)
                if None not in (li, l2, lj) and lj < li + l2 + 1: merged = True; out.append([i, i2]); break
            if not merged: kept.append((i, par))
    return out
rows = []
for _ in range(N):
    r = p.draw_r7(rng, P); r["prior"] = tag; r["redundant"] = redundant_copies(r)
    r["levels"] = (p.levels_used(r) + 1) if not r["H"].get("const_law") else 1
    rows.append(r)
json.dump(rows, open(f"bank_{tag}.json", "w"))
print("done", tag, len(rows))
