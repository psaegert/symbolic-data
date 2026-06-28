# Sampling data

The core of `symbolic-data` is turning a **skeleton pool** into concrete `(X, y)` regression
problems, in a way that is independent of any particular model.

## SkeletonPool

A `SkeletonPool` samples and holds expression *skeletons* (token sequences with constant
placeholders) and compiles them to callables for generating data.

```python
from symbolic_data import SkeletonPool

pool = SkeletonPool.from_config("skeleton_pool.yaml")   # also: from_dict / load
pool.create(100)                                        # sample 100 unique skeletons
pool.save("pools/my_pool")                              # persist (pickles the skeleton set)
```

The constructor is model-agnostic — it takes a `simplipy` engine, a sampling strategy, a literal
prior, the variable names, and optional support-sampler / operator-weights / holdout settings. No
tokenizer or model is involved.

`SkeletonPool.sample_data(code, n_constants, n_support=None) -> (X, y, literals)` is the lowest-level
generation seam: it realizes constants and samples support points for one compiled skeleton.

## iter_samples

`iter_samples` is the convenience loop most consumers use. It yields ready-to-use
[`Sample`](#sample) objects, reproducibly.

```python
from symbolic_data import iter_samples

for sample in iter_samples(
    pool,
    n_support=32,            # support points per problem (validation gets the same count)
    noise_level=0.01,        # Gaussian noise scaled by noise_level * std(y); 0.0 = clean
    mask_unused_variables=False,
    datasets_per_expression=1,
    seed=0,                  # one seeded Generator drives the whole stream
):
    ...
```

Semantics (matched exactly to the flash-ansr / srbf training + eval paths, so a consumer can
delegate without changing the data it produces):

- `sample_data` is asked for `2 * n_support` points; the first `n_support` are the **support** set,
  the rest **validation**.
- Noise is additive Gaussian, scaled by `noise_level * std(y)`, applied per set (a constant or empty
  array is returned unchanged).
- `mask_unused_variables=True` zeroes the columns of `X` for variables absent from the skeleton.

## sample_from_skeleton

For full control over iteration order, resume, or pinning, drive the per-skeleton core directly and
do your own sequencing:

```python
from symbolic_data import sample_from_skeleton
import numpy as np

rng = np.random.default_rng(0)
sample = sample_from_skeleton(pool, skeleton, n_support=32, noise_level=0.01, rng=rng)
# -> Sample | None  (None after max_trials failed draws)
```

## Sample

A `Sample` is one model-agnostic problem:

| field | meaning |
|---|---|
| `skeleton` | the ground-truth skeleton tokens (prefix) |
| `expression` | GT tokens with constants substituted + normalized (via `simplipy`) |
| `constants` | the realized constant literals |
| `variables` / `n_variables_used` | pool variable names (X column order) / count used in the skeleton |
| `x_support`, `y_support` | the support set (`float32`) |
| `x_validation`, `y_validation` | the held-out validation set |
| `y_support_noisy`, `y_validation_noisy` | noised targets (identical copies when `noise_level=0`) |
| `noise_level`, `complexity` | the noise level used; token-length of the substituted expression |
| `placeholder`, `placeholder_reason` | set when a skeleton could not be sampled (`skip_failed=False`) |

The model-coupling parts (tokenization, prompt serialization, evaluation bookkeeping) stay with the
consumer, which wraps each `Sample` in its own record.
