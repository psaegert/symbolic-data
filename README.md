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

# 1. Sample (X, y) problems from a skeleton pool (the model-agnostic seam)
pool = symbolic_data.SkeletonPool.from_config("skeleton_pool.yaml")
pool.create(100)
for sample in symbolic_data.iter_samples(pool, n_support=32, noise_level=0.01):
    sample.x_support, sample.y_support, sample.expression  # ready to fit / tokenize

# 2. Load a curated catalog (the level-1 declarative collection: expressions + their intrinsic
#    per-variable sampling). The three curated sets ship as package data (no download);
#    `load_catalog("name@version")` resolves a versioned catalog from Hugging Face when a manifest
#    is available, and `load_catalog("user/repo:name")` loads a third party's published catalog.
feynman = symbolic_data.load_catalog("feynman")            # 100 equations (Udrescu & Tegmark 2020)
fastsrb = symbolic_data.load_catalog("fastsrb")            # 120 equations (Martinek, viktmar/FastSRB)
nguyen = symbolic_data.load_catalog("nguyen")              # 12 equations (Uy et al. 2011; DSO)
entry = feynman["I.6.2a"]
entry.prepared, entry.variables                            # expression + intrinsic per-variable sampling
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

> Status: 0.4.0 (in development). The `Problem` unit, the unified distribution framework (incl. the
> `fastsrb` distribution), `ProblemCatalog` + `load_catalog`, and the versioned HF resolver are in;
> the curated catalogs (FastSRB, Feynman, Nguyen) ship vendored from their canonical upstreams.
> Forthcoming: `ProblemSource` (the level-2 sampler that turns a catalog into `Problem`s, with
> holdouts/filters and materialization).
