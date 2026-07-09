# P1 GP-toy catalogs: validation state (2026-07-10)

Every entry realized one problem end-to-end via ProblemSource (n_support=32, ppe=1).

CLEAN (13 catalogs, 78 entries): constant(10) grammarvae(1) jin(6) keijzer(15) korns(12)
koza(2) meier(2) neat(8) pagie(1) poly(6) r-rationals(6) sine(1) vladislavleva(8).
Note: DSO renders Keijzer-6 as the closed form x1*(x1+1)/2; Neat-6 (harmonic) excluded.

UPDATE 2026-07-10: livermore (25/25) + nonic (1/1) CLEAN under simplipy 0.4.2 (the phantom-powN
fix: non-smooth exponents like x**7 now stay binary pow instead of emitting unrealizable pow7
tokens). 15/16 catalogs validated.

OPEN (1 catalog, NOT publishable yet):
- livermore2 (150): 95/150 max_trials_exhausted — whole-draw all-finite rejection meets
  partial domains (measured per-point finite fractions 0.12-0.91 over the upstream DSO
  U-ranges). Pending the sampling-policy decision (per-point rejection = distributionally
  identical conditional law; disclosure via per-entry finite_fraction; low-validity floor).

Nguyen rows (13, incl. 12a) logged only: published nguyen@1 stays untouched (forward-only).
NONE of these are published to HF yet; publishing = the release step after all 16 validate.
