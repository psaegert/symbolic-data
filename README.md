# symbolic_data

The model-agnostic symbolic-regression **data layer**, carved out of
[flash-ansr](https://github.com/psaegert/flash-ansr): skeleton/expression sampling,
priors, `(X, y)` support sampling, holdout management, and dataset construction.

Both symbolic-regression methods (for training holdout) and the
[srbf](https://github.com/psaegert/srbf) eval framework depend on it, so training,
holdout, and evaluation draw from one source of truth. Depends only on
[`simplipy`](https://github.com/psaegert/simplipy) + numpy/sklearn.

## Install

```bash
pip install symbolic-data
```

## Quick start

```python
import symbolic_data

# 1. Load a curated catalog (level 1: expressions + their intrinsic per-variable sampling). The
#    three curated sets ship as package data (no download); `load_catalog("name@version")` resolves
#    a versioned catalog from Hugging Face when a manifest is available, and
#    `load_catalog("user/repo:name")` loads a third party's published catalog.
feynman = symbolic_data.load_catalog("feynman")            # 100 equations (Udrescu & Tegmark 2020)
entry = feynman["I.6.2a"]
entry.prepared, entry.variables                            # expression + intrinsic per-variable sampling

# 2. Draw (X, y) Problems from a ProblemSource (level 2). Mode is inferred from the config:
#    a catalog ref (set), a `generator` block (on-the-fly), or inline `problems` (fixed).
src = symbolic_data.ProblemSource({"catalog": "feynman",
                                   "sampling": {"n_support": 32, "n_validation": 32, "noise": 0.01}})
for problem in src:
    problem.x_support, problem.y_support, problem.y_support_noisy, problem.expression  # fit / tokenize

# 3. Freeze for exact reproduction (no seeds): materialize() -> a fixed source that re-iterates
#    byte-identical Problems, identical across models/runs.
frozen = src.materialize()
```

## Extensibility

Distributions are pluggable via a registry: in-process with
`@symbolic_data.DISTRIBUTIONS.register("name")`, or across packages via an `importlib.metadata`
entry point in the `symbolic_data.distributions` group. A registered name drops into the same
`{"name": ..., "kwargs": ...}` config slot as a builtin (e.g. the `fastsrb` distribution).

Catalogs are extensible through the resolver: publish your own to a Hugging Face dataset repo
with a `manifest.json`, then `symbolic_data.load_catalog("your-user/your-repo:name@version")`.

## Versioning / reproducibility

Reproducibility comes from **fixed data, not seeds**: sampling draws from a threaded
`numpy.random.Generator` (entropy by default), and exact reproduction across runs/models is
obtained from a fixed (materialized) catalog rather than by re-seeding. Versioned catalogs resolve
from Hugging Face with a pinned revision **and a sha256 integrity check**; the curated sets ship
vendored from their canonical upstreams as the offline fallback.

> Status: 0.5.0. The full public stack: `Problem`, the unified distribution framework (incl. the
> `fastsrb` distribution), `ProblemCatalog` + `load_catalog` + the versioned HF resolver, and
> `ProblemSource` (set / on-the-fly generate / fixed; holdouts/filters; `materialize()` +
> `to_catalog()` for frozen, byte-reproducible catalogs). Generate-mode is fully
> `Generator`-driven (no global `np.random`); the skeleton engine is private (`_generate`), so the
> public surface is just Problem / Catalog / Source. Curated catalogs (FastSRB, Feynman, Nguyen)
> ship vendored. CLI: `symbolic-data materialize`.
> Deferred: publishing the HF asset manifest + a frozen holdout grid; functional-equivalence
> `exclude` (currently exact normalized-expression match).
