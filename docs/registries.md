# Registries & extensibility

The pieces of the data layer that select an implementation *by name* from a config are pluggable. A
custom implementation drops into the **same config slot** as a builtin — there is no separate idiom
to learn — and a custom prior/distribution changes *what* is sampled, never *how* holdout membership
is tested.

Two extension seams ship today:

- **Distributions** — the `symbolic_data.DISTRIBUTIONS` registry (a `Registry`), backing the
  `{name: ..., kwargs: ...}` config slot used by per-variable sampling, literal priors, and support
  priors. Entry-point group: `symbolic_data.distributions`.
- **Generative catalogs** — `register_generative_catalog(name, cls)`, backing the `type: <name>`
  config slot a generative catalog spec selects (`lample_charton` is the builtin).

## Distributions

### Registering in-process

```python
import numpy as np
from symbolic_data import DISTRIBUTIONS

@DISTRIBUTIONS.register("student_t")
def student_t_dist(df, loc=0.0, scale=1.0, min_value=None, max_value=None, size=1, rng=None):
    g = rng if rng is not None else np.random.default_rng()
    s = loc + scale * g.standard_t(df, size=size)
    return np.clip(s, min_value, max_value) if min_value is not None else s
```

It is now usable from the same `{name: ..., kwargs: ...}` config slot as a builtin (`uniform`,
`normal`, `log_uniform`, `log_normal`, `gamma`, `cauchy`, `binomial`, `fastsrb`), including in a
mixture prior:

```yaml
literal_prior:
  - {name: normal,    kwargs: {loc: 0, scale: 5}, weight: 0.7}
  - {name: student_t, kwargs: {df: 3},            weight: 0.3}
```

### Registering across packages (entry points)

A third-party package can add distributions **without editing symbolic-data**, via
`importlib.metadata` entry points in the `symbolic_data.distributions` group:

```toml
# in your package's pyproject.toml
[project.entry-points."symbolic_data.distributions"]
student_t = "my_pkg:student_t_dist"
```

They are discovered lazily on first lookup.

### Collision policy & reproducibility

- A direct `register` of an existing name **warns and keeps the existing one** unless `overwrite=True`.
- An entry point that shadows an existing name is **ignored with a warning** (first-registered wins),
  so a third-party typo cannot silently replace a builtin.
- `DISTRIBUTIONS.get(name, strict_builtins=True)` resolves **only** the shipped builtins and never
  triggers entry-point loading — pin a reproducibility-critical run to the shipped implementations
  regardless of what is installed.

The builtins are the source of truth (`symbolic_data.BASE_DISTRIBUTIONS`); the registry is seeded from
them.

### From config to callable: `get_distribution` / `build_prior_callable`

The `{name, kwargs}` config slot is turned into an actual sampler by two public helpers. Both return
a callable with the signature `(size=1, rng=None) -> np.ndarray`, so you thread your own
`numpy.random.Generator` for reproducible draws.

`get_distribution(config)` builds one distribution callable from a single `{name, kwargs}` mapping.
Besides any builtin / registered `name`, it understands two special forms — `constant` (a fixed fill)
and `sampler` (a distribution whose parameters are themselves drawn from nested distributions):

```python
import numpy as np
from symbolic_data import get_distribution

rng = np.random.default_rng(0)

normal = get_distribution({"name": "normal", "kwargs": {"loc": 0.0, "scale": 5.0}})
normal(size=4, rng=rng)                       # -> ndarray of shape (4,)

const = get_distribution({"name": "constant", "kwargs": {"value": 3.0}})
const(size=3)                                 # -> array([3., 3., 3.])
```

`build_prior_callable(config)` builds a prior sampler. Given a single `{name, kwargs}` mapping it
delegates to `get_distribution`; given a **list** of `{name, kwargs, weight}` entries it builds a
**mixture** prior — per draw it picks a component with probability proportional to `weight` (weights
are normalized; they need only be positive), then samples from it. This is the same list form the
mixture-prior yaml above uses:

```python
from symbolic_data import build_prior_callable

mixture = build_prior_callable([
    {"name": "normal",  "kwargs": {"loc": 0.0, "scale": 5.0}, "weight": 0.7},
    {"name": "uniform", "kwargs": {"low": -1.0, "high": 1.0}, "weight": 0.3},
])
mixture(size=4, rng=np.random.default_rng(2))  # -> ndarray of shape (4,)
```

## Generative catalogs

A generative catalog spec selects its implementation by a `type:` key (the builtin is
`lample_charton`). Register a custom `GenerativeCatalog` subclass under a new `type:` name:

```python
from symbolic_data import GenerativeCatalog, register_generative_catalog

class MyCatalog(GenerativeCatalog):
    ...  # implement sample_skeleton + the Catalog interface (iter_entries / realize)

register_generative_catalog("my_catalog", MyCatalog)
```

It is now built from a `{type: my_catalog, ...}` mapping (or a resolved yaml with that `type:`) by
`build_catalog(...)` / `ProblemSource({"catalog": {...}})`, exactly like the builtin recipe.
