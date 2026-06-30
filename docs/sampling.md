# Sampling data

The core of `symbolic-data` is turning a **catalog** of expressions into concrete `(X, y)`
regression problems, in a way that is independent of any particular model. There are two levels:
a level-1 `Catalog` (what to sample) and a level-2 `ProblemSource` (how to sample it). Both produce
the one central unit, a [`Problem`](#problem).

> **Migration note (0.4.0+).** The old `SkeletonPool` class and the `iter_samples` / `Sample` seam
> were removed. On-the-fly skeleton generation is now `LampleChartonCatalog` (a `GenerativeCatalog`);
> drive it through a `ProblemSource`, which yields `Problem`s.

## Catalog (level 1)

A `Catalog` supplies *expressions* and their *intrinsic* per-variable sampling. It owns no usage
policy (draw counts, noise, holdouts — those are the `ProblemSource`'s job). There are two kinds.

### Declarative: `ProblemCatalog`

A versioned, reusable collection of expression templates, loaded by reference:

```python
from symbolic_data import load_catalog

nguyen = load_catalog("nguyen")        # resolve "nguyen" via the HF manifest (cached after first use)
len(nguyen)                            # 12
list(nguyen)[0]                        # a CatalogEntry: id, raw, prepared, n_variables, variables, meta
"Nguyen-5" in nguyen                   # True
entry = nguyen["Nguyen-5"]
```

A `load_catalog(ref)` reference is a **local path**, a **`name[@version]`** (resolved from the
official Hugging Face manifest), or **`repo_id:name[@version]`** (a third-party manifest). On disk a
declarative catalog is a single yaml with a `metadata` block and an `expressions` block:

```yaml
metadata:
  name: nguyen
  version: 1
  description: Nguyen symbolic-regression suite (Uy et al. 2011), 12 equations.
  sampling_defaults: {n_points: 20, method: random, noise: 0.0}
  conventions: {sampling: "vars use the `fastsrb` distribution (sample_range/sample_type)."}
expressions:
  Nguyen-5:
    raw: sin(x1 ** 2) * cos(x1) - 1
    prepared: sin(v1 ** 2) * cos(v1) - 1     # normalized form the simplipy engine consumes
    n_variables: 1
    vars:
      v1: {name: x1, sample_range: [-1.0, 1.0], sample_type: [uni, pos]}
      v0: {name: y}
```

Each variable's `sample_range` / `sample_type` drive the `fastsrb` distribution (see
[Registries](registries.md)): `sample_type` is `[base, sign]` where `base` is `uni` / `log` / `int`
and `sign` is `pos` / `neg` / `pos_neg`.

### Generative: `LampleChartonCatalog`

An *on-the-fly* catalog that grows random unary-binary operator trees (the Lample-Charton recipe)
rather than holding a fixed set. It is the public counterpart to a declarative catalog: a
`ProblemSource` samples from it identically.

```python
from symbolic_data import LampleChartonCatalog, build_catalog

# Build from a config (a `type: lample_charton` spec): a simplipy engine, a sample strategy, a
# literal prior, the variable names, and a support-sampler config.
catalog = LampleChartonCatalog.from_config("recipe.yaml")

# ...or resolve any catalog ref to a Catalog (declarative OR generative) in one call:
catalog = build_catalog("lample-charton-v23")     # the published open v23 training recipe
```

A `type:` generative spec carrying inline `skeletons:` is **frozen** (a fixed skeleton set — e.g. a
held-out validation set distributed as one self-contained yaml); without it, the catalog is **open**
and generates fresh skeletons indefinitely. The published `v23-val` catalog is the frozen
1000-skeleton validation set; `lample-charton-v23` is the open training recipe.

## ProblemSource (level 2)

A `ProblemSource` applies the *usage policy* a catalog omits, turning it into `Problem`s. The **mode
is inferred** from the config: a `catalog:` ref (a name / path / mapping / `Catalog` instance) ->
`set` or `generate`; inline `problems:` -> `fixed`. An open generative ref streams (`generate`); a
declarative or frozen ref iterates a fixed set (`set`).

```python
from symbolic_data import ProblemSource

source = ProblemSource({
    "catalog": "feynman",
    "sampling": {
        "n_support": 100,        # support points per problem
        "n_validation": 100,     # held-out validation points (default: equal to n_support)
        "noise": 0.0,            # additive Gaussian on y only, scaled by noise * std(y)
        "problems_per_expression": 1,
        "method": "iterate",     # iterate | random_without_replacement | random_with_replacement
                                 #   (generative default: "procedural")
        "layout": "random",      # X-point layout: "random" (i.i.d.) or "grid"
    },
})

for problem in source:
    if problem.is_placeholder:
        continue
    problem.x_support, problem.y_support, problem.expression   # ready to fit / tokenize
```

### Generating fresh expressions

For a generative catalog, `sampling.size` caps how many expressions to draw (omit it for an
unbounded stream), and `n_support: prior` draws the per-sample support size from the catalog's own
support prior (the training-time pattern — variable support sizes). `n_support: prior` requires
`n_validation: 0` (all realized rows are support; there is no validation split).

```python
train = ProblemSource({
    "catalog": "lample-charton-v23",
    "sampling": {"n_support": "prior", "n_validation": 0, "size": 1000},
})
```

### Reproducibility: materialize / freeze

A `ProblemSource` samples with process entropy by default. To pin exact data, sample once and
**freeze** — re-iterating the frozen artifact yields byte-identical `Problem`s:

```python
fixed = source.materialize(n=1000)            # -> a FIXED-mode ProblemSource (re-iterable, identical)
catalog = source.to_catalog(name="my_val")    # -> a FROZEN ProblemCatalog
catalog.save("data/my_val.npz")               # share it; load_catalog reads it back
```

`size_hint()` reports the finite problem count when known (`None` for an unbounded stream);
`max_n_support` reports the upper bound on a sampled support size (for buffer pre-allocation).

## Problem

A `Problem` is one model-agnostic symbolic-regression problem. It is produced by *every* source
(curated sets, on-the-fly generation, inline/materialized data); there is no dict-vs-dataclass split.

| field | meaning |
|---|---|
| `skeleton` | the ground-truth skeleton tokens (prefix), constants as placeholders |
| `expression` | GT tokens with constants substituted + normalized (via `simplipy`) |
| `constants` | the realized constant literals |
| `variables` / `n_variables_used` | pool variable names (X column order) / count used in the skeleton |
| `complexity` | token length of the substituted expression |
| `x_support`, `y_support` | the support set (`float32`) |
| `x_validation`, `y_validation` | the held-out validation set |
| `y_support_noisy`, `y_validation_noisy` | noised targets (identical copies when `noise` is 0) |
| `noise` | the realized noise applied (provenance) |
| `eq_id` | catalog id, when set-sourced (`None` for generated) |
| `meta` | source-specific provenance (units, moniker, source, ...) |
| `is_placeholder`, `placeholder_reason` | set when the source could not fill a slot |

Conventions baked in:

- Noise is on the **target `y` only** (measurement noise); `X` is never noised. Both the clean `y_*`
  (for ground-truth evaluation, e.g. FVU) and `y_*_noisy` (what a model fits) are kept; when the
  noise spec is null/zero, `y_*_noisy` is the same array.
- The **placeholder protocol** lets a source yield a marked, empty `Problem` (`is_placeholder=True`,
  with a `placeholder_reason`) instead of silently skipping when it cannot produce a valid problem
  for a slot, so row-accounting / resume / indexing stay aligned and the failure is recorded
  honestly downstream. Filter them out with `if problem.is_placeholder: continue`.
- `Problem.is_finite()` is the validity gate: `True` iff every non-empty clean `X`/`y` array is
  all-finite.

`mask_unused_variable_columns(...)` is available to zero the `X` columns for variables absent from a
skeleton, when a consumer needs a fixed-width input. The model-coupling parts (tokenization, prompt
serialization, evaluation bookkeeping) stay with the consumer, which wraps each `Problem` in its own
record.
