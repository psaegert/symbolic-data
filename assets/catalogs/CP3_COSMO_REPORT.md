# cp3-cosmo build report

Known-GT cosmology subset of cp3-bench (cp3-bench / Things-to-bench (Thing & Koksbang, arXiv:2406.15531; github.com/CP3-Origins/Things-to-bench, MIT, commit 69a45dd)). Constants upstream-stated (no refitting); every law verified against every CSV row. Deterministic rebuild (`tools/build_cp3_cosmo.py`).

| eq_id | n | d | max rel err | FVU | gt_kind | noise |
|---|---|---|---|---|---|---|
| C1a | 1000 | 1 | 4.91e-08 | 2.906e-14 | exact | - |
| C1b | 1000 | 1 | 5.54e-01 | 1.099e-01 | reference | 10% multiplicative Gaussian (frozen unseeded realization) |
| C1c | 10000 | 2 | 4.91e-08 | 2.341e-14 | exact | - |
| C1d | 125000 | 3 | 4.72e-16 | 5.310e-32 | exact | - |
| C2a | 10000 | 2 | 5.17e-13 | 2.130e-31 | exact | - |
| C2b | 10000 | 2 | 8.56e-14 | 1.799e-32 | exact | - |
| C3g | 2500 | 3 | 5.35e-09 | 3.331e-18 | exact | - |
| C3h | 2500 | 4 | 5.94e-08 | 1.001e-14 | exact | - |
| C5a | 1000 | 2 | 0.00e+00 | 0.000e+00 | exact | - |
| C5b | 1000 | 2 | 3.13e-02 | 1.742e-04 | reference | 1% multiplicative Gaussian (frozen unseeded realization) |
| C5c | 1000 | 2 | 4.50e-01 | 1.205e-02 | reference | 10% multiplicative Gaussian (frozen unseeded realization) |
| C5d | 10000 | 3 | 0.00e+00 | 0.000e+00 | exact | - |
| C5e | 10000 | 3 | 3.26e-16 | 4.478e-33 | exact | - |
| C5f | 20000 | 3 | 9.80e-16 | 3.522e-33 | exact | - |
| C6a | 1000 | 1 | 5.27e-13 | 6.344e-31 | exact | - |
| C6b | 1000 | 4 | 4.04e-15 | 4.960e-33 | exact | - |
| C6c | 10000 | 2 | 3.21e-11 | 2.691e-30 | exact | - |

17 problems -> `cp3-cosmo.npz`.

Upstream paper-vs-data discrepancies resolved empirically (data wins): C3h sign/prefactor, C6a phase, C6c equal-mass; C2b eos range. Excluded: C3a-f/C4a-e (no GT), F1-F8 (dedup to DSO suites, incl. F8's mutated 6.78 constant vs Korns' 6.87).
