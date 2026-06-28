# Holdout & leak-safety

`HoldoutManager` decides whether a candidate skeleton is *held out* — i.e. functionally equivalent to
a known test/validation prototype — so training data generation can exclude evaluation problems.

```python
from symbolic_data import HoldoutManager

manager = HoldoutManager(n_variables=3, allow_nan=False)
manager.register_skeleton(tokens, compiled_fn, num_constants=0)
manager.is_held_out(tokens, compiled_fn, num_constants=0, n_variables=3)  # -> bool
```

Membership is tested two ways and a hit on **either** counts:

1. **Symbolic** — normalized token form (variable-renaming / reordering aware).
2. **Numeric image** — the function's values on a fixed reference grid, so functionally-equivalent
   expressions with different token forms still match.

## v1 reproducibility scope: leak-safety, not byte-identity

The reference grid is **seeded and shipped** (a fixed realization), not regenerated per process — two
consumers therefore test membership against the *same* grid, so a held-out problem stays held out
across machines and runs. This is the v1 guarantee: **leak-safety**.

v1 does **not** promise cross-consumer *byte-identical* sampling of training data. Threading an
explicit `numpy.random.Generator` through the whole sampling stack (for bit-exact regeneration across
consumers) is a separate, later phase; it is not required for leak-safety, which the shipped grid +
robust matcher already provide.

The matcher's fingerprint constants (grid, slice convention, rounding) are versioned together with the
`simplipy` engine pin, so a benchmark's holdout decisions are reproducible and auditable.
