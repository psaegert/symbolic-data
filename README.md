# sr-data

The model-agnostic symbolic-regression **data layer**, carved out of
[flash-ansr](https://github.com/psaegert/flash-ansr): skeleton/expression sampling,
priors, `(X, y)` support sampling, holdout management, and dataset construction.

Both symbolic-regression methods (for training holdout) and the
[srbf](https://github.com/psaegert/srbf) eval framework depend on it, so training,
holdout, and evaluation draw from one source of truth. Depends only on
[`simplipy`](https://github.com/psaegert/simplipy) + numpy/sklearn.

## Install

```bash
pip install sr-data
```

## Quick start

```python
import sr_data

# 1. Sample (X, y) problems from a skeleton pool (the model-agnostic seam)
pool = sr_data.SkeletonPool.from_config("skeleton_pool.yaml")
pool.create(100)
for sample in sr_data.iter_samples(pool, n_support=32, noise_level=0.01, seed=0):
    sample.x_support, sample.y_support, sample.expression  # ready to fit / tokenize

# 2. Load a benchmark (spec fetched + cached from the psaegert/ansr-data HF dataset)
fastsrb = sr_data.load_benchmark("fastsrb")
dataset = fastsrb.sample("II.38.3", n_points=100)
```

## Extensibility

Distributions and benchmarks are pluggable via registries: in-process with
`@sr_data.DISTRIBUTIONS.register("name")` / `@sr_data.BENCHMARKS.register("name")`, or
across packages via `importlib.metadata` entry points (groups `sr_data.distributions`,
`sr_data.benchmarks`). A registered name drops into the same config slot as a builtin.

## Versioning / reproducibility

v1 guarantees **leak-safety** (a seeded, shipped holdout grid + a robust symbolic/numeric
matcher), not cross-consumer byte-identical regeneration (the rng-Generator threading is a
separate, later phase). Benchmark specs are HF-versioned and stamped on `.provenance`.

> Status: v0.1.0. Registry, `iter_samples` seam, and `load_benchmark` (FastSRB) are in.
> Deferred: the MIA matched-control audit, curated benchmark loaders (Feynman/Nguyen), the
> canonical-v1 holdout grid mint, and cross-consumer byte-identical sampling.
