# Registries & extensibility

The pieces of the data layer that select an implementation *by name* from a config are backed by a
small `Registry`. A custom implementation drops into the **same config slot** as a builtin — there is
no separate idiom to learn — and a custom prior/distribution changes *what* is sampled, never *how*
holdout membership is tested.

Two registries ship today:

- `symbolic_data.DISTRIBUTIONS` — constant-value distributions (group `symbolic_data.distributions`).
- `symbolic_data.BENCHMARKS` — benchmark loaders (group `symbolic_data.benchmarks`).

## Registering in-process

```python
import numpy as np
from symbolic_data import DISTRIBUTIONS

@DISTRIBUTIONS.register("student_t")
def student_t_dist(df, loc=0.0, scale=1.0, min_value=None, max_value=None, size=1):
    s = loc + scale * np.random.standard_t(df, size=size)
    return np.clip(s, min_value, max_value) if min_value is not None else s
```

It is now usable from the same `{"name": ..., "kwargs": ...}` config slot as a builtin, including in a
mixture prior:

```yaml
literal_prior:
  - {name: normal,    kwargs: {loc: 0, scale: 5}, weight: 0.7}
  - {name: student_t, kwargs: {df: 3},            weight: 0.3}
```

## Registering across packages (entry points)

A third-party package can add implementations **without editing symbolic-data**, via
`importlib.metadata` entry points:

```toml
# in your package's pyproject.toml
[project.entry-points."symbolic_data.distributions"]
student_t = "my_pkg:student_t_dist"

[project.entry-points."symbolic_data.benchmarks"]
my_bench = "my_pkg:load_my_bench"
```

They are discovered lazily on first lookup.

## Collision policy & reproducibility

- A direct `register` of an existing name **warns and keeps the existing one** unless `overwrite=True`.
- An entry point that shadows an existing name is **ignored with a warning** (first-registered wins),
  so a third-party typo cannot silently replace a builtin.
- `DISTRIBUTIONS.get(name, strict_builtins=True)` resolves **only** the shipped builtins and never
  triggers entry-point loading — pin a reproducibility-critical run to the shipped implementations
  regardless of what is installed.

The builtins are the source of truth (`symbolic_data.BASE_DISTRIBUTIONS`); the registry is seeded from
them.
