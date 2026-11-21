import json, numpy as np
d = json.load(open("rows.json"))
def target(kind):
    N = np.arange(101); w = np.full(101, -np.inf)
    if kind == "t8ext": w[1:51] = N[1:51]**0.7; w[51:] = 50**0.7 - (N[51:]-50)/10
    if kind == "stretch": f = lambda x: (17*x/50)**0.7; w[1:51] = f(N[1:51]); w[51:] = f(50) - (N[51:]-50)/10
    if kind == "decay": w[1:] = -N[1:]/15
    p = np.exp(w - w[1:].max()); p[0] = 0; return p/p.sum()
T = {k: target(k) for k in ["t8ext", "stretch", "decay"]}
for k, t in T.items():
    c = np.cumsum(t); print(f"target {k:8s} P(N<=10)={c[10]:.3f} P(N<=25)={c[25]:.3f} P(N<=50)={c[50]:.3f} median={np.searchsorted(c,0.5)} mean={np.dot(np.arange(101),t):.1f}")
res = {}
for key in ["A", "B", "C", "T8"]:
    rows = np.array(d[key]["rows"]); n = d[key]["n"]
    N = rows[:,0].astype(int)
    q0 = np.bincount(N, minlength=101)[:101] / n
    res[key] = dict(rows=rows, q0=q0, n=n)
    print(f"\n== {key}: P(N>100)={d[key]['aborted']/n:.3f}  P(N=0)={q0[0]:.4f}  min count over N=1..100: {np.bincount(N, minlength=101)[1:101].min()}")
    if key == "T8": continue
    for k, t in T.items():
        ratio = np.where(t > 0, t/np.maximum(q0, 1e-300), 0); Mx = ratio.max()
        print(f"  target {k:8s}: acceptance 1/M = {1/Mx:.4f} (worst N = {ratio.argmax()})")
    bins = [(1,5),(6,10),(11,17),(18,25),(26,35),(36,50),(51,75),(76,100)]
    print("  bin      share  composite reuse  superpos  copies deepest rootshare summands tokens/ops")
    for lo, hi in bins:
        m = (N >= lo) & (N <= hi); R = rows[m]
        print(f"  {lo:3d}-{hi:<3d}  {m.sum()/n:.4f}  {R[:,4].__ge__(2).mean():.3f}    {R[:,5].mean():.3f}  {R[:,3].mean():.3f}     {R[:,4].mean():5.2f}  {R[:,6].mean():.2f}    {R[:,7].mean():.2f}      {R[:,2].mean():.2f}    {(R[:,1]/R[:,0]).mean():.2f}")
    for k, t in T.items():
        w = np.where(N >= 1, t[np.minimum(N,100)]/np.maximum(q0[N],1e-300), 0); w /= w.sum()
        tok = rows[:,1]; o = np.argsort(tok); cw = np.cumsum(w[o])
        q = lambda p: tok[o][np.searchsorted(cw, p)]
        print(f"  under {k:8s}: composite {np.dot(w, rows[:,4]>=2):.3f} reuse {np.dot(w, rows[:,5]):.3f} superpos {np.dot(w, rows[:,3]):.3f} summands {np.dot(w, rows[:,2]):.2f}  tokens median {q(0.5):.0f} p90 {q(0.9):.0f} p99 {q(0.99):.0f} max {tok[w>0].max():.0f}")
np.savez("analysis.npz", **{f"{k}_q0": v["q0"] for k, v in res.items()}, **{f"t_{k}": v for k, v in T.items()})
