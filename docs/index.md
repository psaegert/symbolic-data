# symbolic-data: the model-agnostic symbolic-regression data layer

`symbolic-data` (import name `symbolic_data`) owns the model-agnostic data substrate for
symbolic regression: skeleton/expression sampling, priors, `(X, y)` support sampling, holdout
management, and benchmark datasets. It was carved out of
[flash-ansr](https://github.com/psaegert/flash-ansr) so that symbolic-regression methods (for
training-time holdout) and the [srbf](https://github.com/psaegert/srbf) evaluation framework draw
from **one source of truth**.

It is a deep leaf: it depends only on
[`simplipy`](https://github.com/psaegert/simplipy) (the expression engine) plus numpy / scikit-learn.

```
simplipy  ◄──  symbolic-data  ◄──  { flash-ansr[train], srbf, other SR methods }
```

## Install

```bash
pip install symbolic-data
```

## Quickstart

```python
import symbolic_data

# 1. Sample (X, y) problems from a skeleton pool — the model-agnostic generation seam
pool = symbolic_data.SkeletonPool.from_config("skeleton_pool.yaml")
pool.create(100)
for sample in symbolic_data.iter_samples(pool, n_support=32, noise_level=0.01, seed=0):
    sample.x_support, sample.y_support, sample.expression  # ready to fit / tokenize

# 2. Load a benchmark — the spec is fetched + cached from the psaegert/ansr-data HF dataset
fastsrb = symbolic_data.load_benchmark("fastsrb")
dataset = fastsrb.sample("II.38.3", n_points=100)
```

## What's inside

- **[Sampling data](sampling.md)** — `SkeletonPool`, the `iter_samples` / `Sample` seam, noise + masking.
- **[Benchmarks](benchmarks.md)** — `load_benchmark`, FastSRB, HuggingFace-versioned specs, provenance.
- **[Registries & extensibility](registries.md)** — plug in custom distributions / benchmarks in-process
  or across packages via entry points.
- **[Holdout & leak-safety](holdout.md)** — `HoldoutManager` and the v1 reproducibility scope.

## Versioning & reproducibility

v1 guarantees **leak-safety** (a seeded, shipped holdout grid + a robust symbolic/numeric matcher),
not cross-consumer byte-identical regeneration (the RNG-Generator threading is a separate, later
phase). Benchmark specs are HuggingFace-versioned and stamped on `.provenance`. See
[Holdout & leak-safety](holdout.md).
