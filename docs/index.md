# symbolic-data: the model-agnostic symbolic-regression data layer

`symbolic-data` (import name `symbolic_data`) owns the model-agnostic data substrate for
symbolic regression: expression catalogs (declarative and on-the-fly generative), priors,
`(X, y)` support sampling, holdout/decontamination, and curated benchmark sets. It was carved out
of [flash-ansr](https://github.com/psaegert/flash-ansr) so that symbolic-regression methods (for
training-time generation + holdout) and the [srbf](https://github.com/psaegert/srbf) evaluation
framework draw from **one source of truth**.

It is a deep leaf: it depends only on
[`simplipy`](https://github.com/psaegert/simplipy) (the expression engine) plus numpy / scikit-learn.

```
simplipy  ◄──  symbolic-data  ◄──  { flash-ansr, srbf, other SR methods }
```

## The two-level model

Everything is realized into one central unit, a [`Problem`](#problem): a ground-truth skeleton plus
its realized constants and sampled support/validation `(X, y)` data. Two levels produce `Problem`s:

- **Level 1 — a `Catalog`** supplies *expressions* and their *intrinsic* per-variable sampling. It
  is either a declarative `ProblemCatalog` (a versioned yaml of expressions, resolved by reference)
  or an on-the-fly `GenerativeCatalog` (e.g. `LampleChartonCatalog`, which grows random operator
  trees — the Lample-Charton recipe). A catalog owns no *usage policy*.
- **Level 2 — a `ProblemSource`** samples a catalog into `Problem`s under a usage policy: draw
  method, support/validation counts, noise, problems-per-expression, and holdouts/filters.

## Install

```bash
pip install symbolic-data
```

## Quickstart

```python
from symbolic_data import ProblemSource, load_catalog

# 1. Load a curated catalog by name (fetched + cached from the HF assets repo on first use).
feynman = load_catalog("feynman")     # a declarative ProblemCatalog, 100 expressions
len(feynman)                          # 100

# 2. Sample it into Problems via a ProblemSource (mode is inferred: a name -> "set" mode).
source = ProblemSource({
    "catalog": "feynman",
    "sampling": {"n_support": 100, "n_validation": 100, "noise": 0.0},
})
for problem in source:
    if problem.is_placeholder:        # a slot the source could not fill (recorded, not skipped)
        continue
    problem.x_support, problem.y_support, problem.expression   # ready to fit / tokenize

# 3. Generate fresh training expressions on the fly (an open generative recipe -> "generate" mode).
train = ProblemSource({
    "catalog": "lample-charton-v23",
    "sampling": {"n_support": "prior", "n_validation": 0, "size": 1000},
    "holdouts": [{"exclude": "v23-val"}],   # decontaminate against the held-out validation set
})
```

## What's inside

- **[Sampling data](sampling.md)** — `Catalog` (declarative `ProblemCatalog` / generative
  `LampleChartonCatalog`), `ProblemSource`, the `Problem` unit, noise + masking.
- **[Benchmarks](benchmarks.md)** — curated catalogs (`fastsrb`, `feynman`, `nguyen`),
  `load_catalog`, HuggingFace-versioned references, provenance.
- **[Registries & extensibility](registries.md)** — plug in custom distributions / generative
  catalogs in-process or across packages via entry points.
- **[Holdout & leak-safety](holdout.md)** — `ProblemSource` `holdouts:` (skeleton-level
  decontamination + filters) and the reproducibility scope.

## Versioning & reproducibility

Reproducibility never comes from a fixed seed. A `ProblemSource` samples with process entropy by
default; **exact** reproduction comes from *materializing* a source once and freezing it
(`source.materialize()` / `source.to_catalog(...).save(...)`), so re-iterating the frozen artifact
yields byte-identical `Problem`s across machines and runs.

Catalogs are distributed as **Hugging Face artifacts** (not bundled in the wheel): a bare `name`
resolves via the manifest on the `psaegert/symbolic-data-assets` dataset repo, with the version
pinning a git revision and per-file sha256 (integrity-checked, then cached). See
[Benchmarks](benchmarks.md) and [Holdout & leak-safety](holdout.md).
