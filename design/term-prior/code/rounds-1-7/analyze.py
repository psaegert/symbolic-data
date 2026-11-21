import json, collections
R = [json.loads(l) for l in open("draws.jsonl")]
n = len(R); ho = sum(r["held_out"] for r in R); nd = n - ho
print(f"raw draws {n}, held out {ho} ({100*ho/n:.1f}%)")
keys = ["poly", "rational", "poly_ge3", "poly_1v_ge3", "sum_rat_ge2", "affine_ge2v", "repeat_motif_ge2", "superpos_ge3", "transcendental"]
print(f"{'class':18s} {'raw %':>7s} {'n_raw':>6s} {'held-out %':>11s} {'delivered %':>12s} {'n_deliv':>8s}")
for k in keys + ["k>=2", "k>=3"]:
    f = (lambda r: r["k_terms"] >= int(k[-1])) if k.startswith("k>=") else (lambda r, k=k: r[k])
    c = [r for r in R if f(r)]; ch = sum(r["held_out"] for r in c); cd = len(c) - ch
    print(f"{k:18s} {100*len(c)/n:7.2f} {len(c):6d} {100*ch/max(len(c),1):11.1f} {100*cd/nd:12.3f} {cd:8d}")
print("\nexamples of DELIVERED draws per class:")
for k in ["poly_ge3", "sum_rat_ge2", "superpos_ge3", "affine_ge2v"]:
    ex = [r["sk"] for r in R if r[k] and not r["held_out"]][:4]
    print(k, ex)
print("\nexamples of HELD-OUT draws per class:")
for k in ["poly_ge3", "sum_rat_ge2", "superpos_ge3", "affine_ge2v"]:
    ex = [r["sk"] for r in R if r[k] and r["held_out"]][:3]
    print(k, ex)
