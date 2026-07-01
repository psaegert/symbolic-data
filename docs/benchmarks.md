# Benchmarks (curated catalogs)

`load_catalog(ref)` resolves a named catalog to a ready-to-sample `ProblemCatalog`. Three curated
benchmark catalogs are registered by name — `fastsrb`, `feynman`, and `nguyen` (the members of
`symbolic_data.CATALOGS`). They are **Hugging Face artifacts**, not bundled in the wheel: a bare name
resolves via the manifest on the `psaegert/symbolic-data-assets` dataset repo (network on first use,
then cached) with the version pinning a git revision and per-file sha256.

```python
from symbolic_data import load_catalog, ProblemSource

feynman = load_catalog("feynman")        # a declarative ProblemCatalog
len(feynman)                             # 100
list(feynman)[:1]                        # CatalogEntry objects (id, raw, prepared, n_variables, ...)
"I.6.2a" in feynman                      # True
entry = feynman["I.9.18"]                # Newton's gravitation, 9 variables
```

A `ProblemCatalog` supplies expressions and their intrinsic per-variable ranges; it does **not**
realize `(X, y)` itself. To turn a catalog into sampled `Problem`s, hand the ref to a
[`ProblemSource`](sampling.md#problemsource-level-2) (mode is inferred — a curated name gives `set`
mode, iterating every expression once):

```python
source = ProblemSource({
    "catalog": "feynman",
    "sampling": {"n_support": 100, "n_validation": 0, "noise": 0.0},
})
for problem in source:                   # one Problem per expression
    if problem.is_placeholder:           # an expression that could not be sampled (see below)
        continue
    problem.eq_id, problem.x_support, problem.y_support, problem.expression
```

To realize a *single* equation, filter the iteration by `problem.eq_id` (there is no per-id
one-liner). The `eq_id` is the catalog key (`"I.9.18"`, `"Nguyen-5"`, `"II.38.3"`).

## Resolving references

`load_catalog(ref)` resolves **declarative** catalogs and **`.npz` frozen** catalogs only. `ref`
is a string that points at a declarative-yaml or frozen-`.npz` artifact:

- a curated **name** — `fastsrb`, `feynman`, `nguyen`;
- **`name@version`** — pin a specific version;
- **`repo_id:name[@version]`** — a third-party HF manifest, so anyone can publish + load their own;
- a **local path** — a `.yaml` declarative spec or a `.npz` frozen catalog.

`load_catalog` has no `type:` dispatch and rejects a mapping (it goes through `ProblemCatalog.load`
→ the resolver, which requires a string ref). To use a **generative** catalog — the published
**`v23-val`** validation set (a generative catalog carrying a fixed 1000-skeleton set), the open
`lample-charton-v23` training recipe, a local generative `.yaml` (a `type:` spec), or an inline
`{type: lample_charton, ...}` mapping — build it through `build_catalog(ref)` or
`ProblemSource({"catalog": ref})`, which dispatch on the `type:` key:

```python
from symbolic_data import build_catalog, ProblemSource

# generative names, a local generative .yaml, or an inline mapping — via build_catalog / ProblemSource
val = build_catalog("v23-val")                     # the frozen 1000-skeleton validation set (generative)
gen = build_catalog("lample-charton-v23")          # the open training recipe (a GenerativeCatalog)
src = ProblemSource({"catalog": {"type": "lample_charton"}})   # + generative config keys (inline mapping)
```

`build_catalog` / `ProblemSource` also accept every form `load_catalog` accepts (a curated name, a
`name@version`, a `repo_id:name`, a local declarative-yaml / `.npz` path); for a non-generative ref
they fall through to `ProblemCatalog.load`. Pass an explicit local path for fully offline operation;
a bare name needs network on first use.

## FastSRB

`load_catalog("fastsrb")` returns the **FastSRB** catalog (Viktor Martinek,
[arXiv:2508.14481](https://arxiv.org/abs/2508.14481); ~120 Feynman-family equations with
physically-motivated SRSD ranges). It is sourced from the upstream
[`viktmar/FastSRB`](https://github.com/viktmar/FastSRB) expressions (MIT). Its per-variable ranges
use the `fastsrb` distribution (see [Registries](registries.md)).

```python
source = ProblemSource({"catalog": "fastsrb", "sampling": {"n_support": 100, "n_validation": 0}})
problems = [p for p in source if not p.is_placeholder]
```

A couple of upstream FastSRB equations (e.g. `II.24.17`, `B4`) are mostly-non-finite by construction
under their own ranges (a `sqrt` whose argument is usually negative). The `ProblemSource` yields a
**placeholder** `Problem` for these (`is_placeholder=True`, with a `placeholder_reason`) rather than
crashing, so row-accounting stays aligned and the failure is recorded; filter them out as above.

## Feynman

`load_catalog("feynman")` returns the **100-equation Feynman Symbolic Regression Database**
(Udrescu & Tegmark 2020, [AI Feynman](https://arxiv.org/abs/1905.11481)). Formulas and per-variable
sampling ranges come from the canonical `FeynmanEquations.csv` (uniform FSReD ranges).

```python
feynman = load_catalog("feynman")
[e.id for e in list(feynman)[:3]]        # ['I.6.2a', 'I.6.2', 'I.6.2b']
```

> FastSRB vs Feynman: both draw on the Feynman equation family but are distinct catalogs. `fastsrb`
> is Martinek's curated set (~120 equations) with physically-motivated ranges (SRSD, Matsubara et al.
> 2024); `feynman` is the 100-equation core Feynman Symbolic Regression Database
> (`FeynmanEquations.csv`) with the original uniform FSReD ranges.

For equations with a restricted domain, the sampler rejects out-of-domain draws and resamples, so the
realized input distribution is conditioned on the function's domain rather than uniform over the full
declared box.

## Nguyen

`load_catalog("nguyen")` returns the **12-equation Nguyen suite** (Uy et al. 2011). Formulas and
sampling ranges are derived from the canonical `benchmarks.csv` of the DSO/DSR project (Petersen et
al. 2021, [deep-symbolic-optimization](https://github.com/dso-org/deep-symbolic-optimization)).

```python
nguyen = load_catalog("nguyen")
nguyen["Nguyen-5"].raw                    # 'sin(x1 ** 2) * cos(x1) - 1' on U[-1, 1]
```

## Provenance

A loaded `ProblemCatalog` carries its `name`, `version`, and `source` (`"local"` or `"huggingface"`)
plus the on-disk `metadata` block on `.meta` (description, sources, conventions, sampling defaults).
A resolved HF artifact pins the dataset revision and per-file sha256, so a downstream artifact can
record exactly which spec (and dataset revision) produced it.

```python
feynman = load_catalog("feynman")
feynman.name, feynman.source     # ('feynman', 'huggingface')
feynman.version                  # the resolved manifest version (an int)
feynman.meta.get("description")
```

## Registering a catalog

Custom **generative** catalog types plug into the `type:` config slot via
`register_generative_catalog(name, cls)`; custom **distributions** plug into the per-variable / prior
config slots via the `DISTRIBUTIONS` registry. See [Registries & extensibility](registries.md). To
publish your own declarative or frozen catalog for resolution by name, host its yaml/npz in an HF
dataset repo with a manifest and load it via `repo_id:name`.
