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
for sample in symbolic_data.iter_samples(pool, n_support=32, noise_level=0.01, seed=0):
    sample.x_support, sample.y_support, sample.expression  # ready to fit / tokenize

# 2. Load a benchmark. All three curated sets ship as package data (no download), vendored
#    from their canonical upstreams.
fastsrb = symbolic_data.load_benchmark("fastsrb")          # 120 equations (Martinek, viktmar/FastSRB)
feynman = symbolic_data.load_benchmark("feynman")          # 100 equations (Udrescu & Tegmark 2020)
nguyen = symbolic_data.load_benchmark("nguyen")            # 12 equations (Uy et al. 2011; DSO)
dataset = feynman.sample("I.6.2a", n_points=100, random_state=0)
```

## Extensibility

Distributions and benchmarks are pluggable via registries: in-process with
`@symbolic_data.DISTRIBUTIONS.register("name")` / `@symbolic_data.BENCHMARKS.register("name")`, or
across packages via `importlib.metadata` entry points (groups `symbolic_data.distributions`,
`symbolic_data.benchmarks`). A registered name drops into the same config slot as a builtin.

## Versioning / reproducibility

v1 guarantees **leak-safety** (a seeded, shipped holdout grid + a robust symbolic/numeric
matcher), not cross-consumer byte-identical regeneration (the rng-Generator threading is a
separate, later phase). Curated benchmark specs ship as package data, vendored from their canonical
upstreams (`tools/build_benchmark_specs.py`), and stamp their source on `.provenance`.

> Status: v0.3.0. Registry, `iter_samples` seam, the data-prep CLI, and curated `load_benchmark`
> loaders (FastSRB, Feynman, Nguyen) are in, each vendored from its canonical upstream. The
> Feynman/Nguyen specs are numerically verified against their source formulas
> (`tools/build_benchmark_specs.py`).
> Deferred: the MIA matched-control audit, the canonical-v1 holdout grid mint, and cross-consumer
> byte-identical sampling.
