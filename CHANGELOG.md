# Changelog

All notable changes to `symbolic-data` are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and the project adheres to semantic versioning.

## [0.9.0] - 2026-06-30

Completes the family's by-name catalog transition + a terminology cleanup.

### Added
- **`register_holdout_pool` accepts declarative catalogs + by-name/HF refs.** A training catalog can
  now hold out the canonical (declarative) `fastsrb` benchmark by name, not just a saved skeleton-pool
  directory: a string ref resolves via `build_catalog` (name[@version] → HF, config path, inline), a
  directory still loads as before, and a declarative `ProblemCatalog`'s structural prototypes are
  derived from its expressions in the training catalog's space (variables canonicalized).

### Changed
- **Saved-catalog default filename `skeleton_pool.yaml` → `catalog.yaml`** (`LampleChartonCatalog.save`/
  `.load`). Breaking for loaders of pre-0.9 saved directories by the old name; the family now resolves
  catalogs by name, so nothing depends on the legacy filename.
- Added a package `__version__`; purged the term "skeleton pool" from source + docs (the public API has
  been `LampleChartonCatalog`/`ProblemSource`/`load_catalog` since 0.6; this finishes the prose/CLI/docs
  and drops a legacy `skeleton_pool:` config-key unwrap).

## [0.8.0] - 2026-06-30

Catalogs become **pure Hugging Face artifacts** (not bundled in the wheel), the curated **v23**
catalogs are published, and any catalog — declarative or generative — resolves **by name**.

### Added
- **Resolve generative catalogs by name/path.** `build_catalog(ref)` / `ProblemSource({"catalog": ref})`
  now resolve a string ref (local path or HF `name[@version]`) and dispatch on content: a `type:` spec
  builds a `GenerativeCatalog` — **open** (on-the-fly), or **frozen** if it carries inline `skeletons:`
  — while anything else is a declarative `ProblemCatalog`. `from_config` makes `holdout_pools` optional
  and loads optional inline `skeletons:`.
- **Two v23 catalogs published to the HF assets repo:** `v23-val` (the 1000-skeleton frozen validation
  set, a single self-contained generative spec) and `lample-charton-v23` (the open v23 training recipe).
  Resolve with `ProblemSource({"catalog": "v23-val"})` / `"lample-charton-v23"`.
- **Skeleton-level, variable-canonical decontamination.** `ProblemSource` `holdouts: [{exclude: <ref>}]`
  now drops a problem whose *skeleton* (constants collapsed and variables canonicalized via
  `normalize_skeleton`, e.g. `v1.. -> x1..`) matches the excluded catalog — which may be declarative
  (FastSRB) or generative (v23-val), so cross-namespace decontamination is leak-safe. This replaces the
  internal "skeleton pool" holdout for training generation.

### Changed (breaking)
- **Catalogs are HF-only (pure-HF).** The curated catalogs no longer ship in the wheel; a bare `name`
  resolves only via the HF manifest (network on first use, then cached). The vendored package-data
  offline fallback (`resolver._vendored_path` / `vendored_fallback`) is removed — pass an explicit
  local path for offline use.

### Fixed
- A **frozen** generative catalog in `set` mode now iterates its fixed skeleton set **once** (bounded);
  previously it streamed unbounded, so `list(ProblemSource("v23-val"))` would never terminate despite a
  finite `size_hint`, and a fully-excluded source looped forever.

## [0.7.2] - 2026-06-30

Lets a downstream trainer consume a *saved fixed* generative catalog (a held-out validation pool
loaded from disk) through `ProblemSource`, not just an on-the-fly generator.

### Added
- **`ProblemSource` accepts a pre-built `Catalog` instance** as `config["catalog"]` (a
  `GenerativeCatalog` instance -> generate mode), so a consumer can hand it an already-loaded
  catalog (e.g. `LampleChartonCatalog.load(dir)`) instead of only a config dict / ref.

### Changed
- **`GenerativeCatalog.iter_entries(size=None)`** now streams via `sample_skeleton(new=False)`: an
  EMPTY catalog generates a fresh skeleton each draw (training-time streaming), while a PRE-LOADED
  catalog samples from its existing fixed skeletons (a saved validation pool) -- restoring the old
  worker's `sample_skeleton()` default. (It previously forced `new=True`, which would wrongly
  generate fresh skeletons for a loaded pool.)

## [0.7.1] - 2026-06-30

### Added
- **`ProblemSource.catalog`** -- public accessor for the `Catalog` the source samples from (built
  lazily, cached). Lets a consumer that also needs the catalog directly (e.g. a trainer harvesting
  raw skeletons for prompt features) share the source's single catalog instance -- one simplipy
  engine -- instead of constructing a second one.

## [0.7.0] - 2026-06-30

Adds the training-time generation knobs a downstream trainer needs so it can consume a
`ProblemSource` directly (yielding `Problem`s) instead of reaching past it into the catalog's
low-level samplers. Additive; no breaking change.

### Added
- **`sampling.n_support: prior`** (generative catalogs only) -- draw the per-sample support size
  from the catalog's own `n_support_prior` (variable support sizes, the training pattern) instead of
  a fixed count. Requires `n_validation: 0`: every realized row is support, no validation split. The
  distribution is the catalog's existing `sample_data(n_support=None)` path, unchanged; it errors on
  a declarative catalog (no support prior).
- **`ProblemSource.max_n_support`** -- upper bound on a sampled support size (a generative catalog's
  configured support maximum, else the fixed `n_support`); lets a consumer pre-size buffers.

## [0.6.0] - 2026-06-30

Generalizes the catalog abstraction: a `ProblemSource` now samples from a **`Catalog`**, which is
either a declarative `ProblemCatalog` or an on-the-fly **`GenerativeCatalog`**. The procedural
skeleton engine is no longer a private `SkeletonPool` hidden behind a special `generator:` mode;
it is a first-class, public generative catalog (`LampleChartonCatalog`) that produces fresh
expressions and that flash-ansr (training + prompt features) and srbf (sampling baselines) can
consume directly. (0.5.0 hid the engine entirely; two first-party consumers genuinely need a public
generation API, so 0.6.0 exposes it cleanly as a catalog rather than re-exposing the pool.)

### Added
- **`Catalog` (abstract base)** -- the level-1 thing a `ProblemSource` samples from: supplies
  expressions and realizes each into raw `(X, y)` via its intrinsic sampling (`iter_entries` +
  `realize`). `ProblemCatalog` (declarative) and `GenerativeCatalog` (on-the-fly) both implement it.
- **`GenerativeCatalog` + `LampleChartonCatalog`** -- a public generative catalog that grows random
  unary-binary operator trees (the Lample-Charton recipe). Streams fresh skeletons unbounded
  (`iter_entries(size=None)`) or yields a finite reproducible set (`size=N`); exposes raw
  `sample_skeleton(...)` for structure-only consumers (prompt-term harvesting, sampling baselines).
- **`build_catalog(spec)` + `register_generative_catalog(name, cls)`** -- a string/path resolves to a
  declarative `ProblemCatalog`; a mapping with a `type:` key resolves to the registered generative
  catalog. Third parties can register their own generators.
- **`RealizedExpression`** -- the catalog's intrinsic output (`n_points` of `(X, y)` + ground truth),
  which `ProblemSource` splits/noises into a `Problem`.
- **Unbounded streaming generation.** A generative source without `size` streams `Problem`s forever
  (the training-time mode); `size_hint()` is `None`.

### Changed
- **`ProblemSource` config: `catalog:` replaces `generator:`.** A string/path `catalog:` is a
  declarative set; a mapping `catalog: {type: lample_charton, ...}` is generative. The number of
  expressions to draw moves to `sampling: {size: N}` (usage policy); `generator:` is gone.
- **Shared exceptions** live in `symbolic_data.errors` (`NoValidSampleFoundError` still public; new
  `CatalogEntryError` distinguishes a permanently-unrealizable entry from a transient retry).

### Migration
- `{"generator": {<skeleton-pool cfg>, "size": N}, "sampling": {...}}`
  -> `{"catalog": {<skeleton-pool cfg>, "type": "lample_charton"}, "sampling": {"size": N, ...}}`.
- `from symbolic_data._generate.skeleton_pool import SkeletonPool`
  -> `from symbolic_data import LampleChartonCatalog` (same `from_config`/`load`/`sample_skeleton`/
  `sample_data`/`create`/`clear_holdouts` API; it is now a public `GenerativeCatalog`).

## [0.5.0] - 2026-06-30

Completes the data-layer redesign: `SkeletonPool` (and the whole skeleton machinery) is removed
from the public surface, generate-mode is fully `Generator`-driven, and materialization is
shippable. (0.4.0 was a GitHub milestone; 0.5.0 is the first PyPI release of the new data layer.)

### Added
- **`ProblemSource.materialize()` + `to_catalog()` + frozen catalogs.** `materialize()` returns a
  fixed source that re-iterates byte-identical Problems; `to_catalog()` returns a FROZEN
  `ProblemCatalog` (realized `(X, y)`), persisted as a self-contained `.npz` via `.save()` and
  reloaded with `load_catalog` -- the shareable, exactly-reproducible form. This is the no-seed
  reproducibility mechanism.
- **`materialize` CLI command** -- `symbolic-data materialize -c <source-config> -o <out.npz>`
  samples a ProblemSource once and freezes it to a catalog.

### Changed
- **Generate-mode is fully `numpy.random.Generator`-threaded** -- the skeleton/support/holdout
  sampling no longer touches global `np.random`; the source's Generator controls everything
  (verified by a completeness test: same injected Generator + different global seed -> byte-identical
  output). Generate-mode builds `Problem`s natively.
- **The skeleton engine is now private** (`symbolic_data._generate`): `SkeletonPool`,
  `SkeletonSampler`, `SupportSampler`, `HoldoutManager`, and `structure` are ProblemSource's
  internal generate engine, not public modules/classes.

### Removed (breaking)
- **`Sample` / `sample_from_skeleton` / `iter_samples`** (`samples.py`) -- generate-mode emits
  `Problem` directly.
- **`ParserFactory` / `TestSetParser` (`convert_data.py`)** -- the legacy skeleton-ingest of raw
  benchmark files. Superseded by vendored curated catalogs + decontamination via
  `ProblemSource(holdouts=[{exclude: <catalog>}])`.
- **The `generate-skeleton-pool` / `import-data` / `split-skeleton-pool` CLI commands** -- replaced
  by the single `materialize` command.
- The public `symbolic_data.skeleton_pool` / `.skeleton_sampling` / `.support_sampling` /
  `.holdout` / `.structure` import paths (engine is private under `_generate`).
  `NoValidSampleFoundError` and `token_ops.apply_variable_mapping` remain available.

### Deferred (tracked for a later release)
- Publishing the Hugging Face asset manifest + a frozen `holdout_grid` asset; upgrading holdout
  `exclude` from exact normalized-expression match to functional-equivalence.

## [0.4.0] - 2026-06-29

A ground-up redesign of the data layer around one central unit and a clean, versioned, three-level
stack. **Breaking:** the `load_benchmark` / `SpecBenchmark` / `BENCHMARKS` API and the public
skeleton-sampling classes are removed (see Migration).

### Added
- **`Problem`** -- the one central data unit produced by every source (expression, skeleton,
  constants, X, clean + noisy y for support and validation, complexity, provenance, placeholder
  protocol). Noise is on the target y only; `y_*_noisy is y_*` when noise is zero.
- **`ProblemCatalog` + `load_catalog`** -- the level-1 declarative artifact (`{metadata,
  expressions}`): expressions + their intrinsic per-variable sampling. Curated catalogs `fastsrb`
  (120), `feynman` (100), `nguyen` (12) ship vendored as package data.
- **Versioned, repo-agnostic resolver** (`symbolic_data.resolver`): `load_catalog("name@version")`
  resolves from a Hugging Face dataset manifest with a pinned git revision **and a sha256 integrity
  check**, cached locally; `load_catalog("user/repo:name@version")` loads third-party catalogs;
  vendored package data is the offline fallback. Integrity failures never silently fall back.
- **`ProblemSource`** -- one concrete level-2 class (no ABC/subclasses), mode inferred from config:
  a catalog ref (SET), a `generator` block (on-the-fly GENERATE), or inline `problems` (FIXED). Owns
  the usage policy: draw `method`, `n_support`/`n_validation`, `noise`, `problems_per_expression`,
  `layout`, holdouts/filters, and `materialize()`.
  - Holdouts: a list of `{filter: {finite, max_complexity, n_variables, ...}}` and
    `{exclude: <catalog>}` (decontamination by exact normalized-expression match).
  - `materialize()` -> a FIXED source that re-iterates byte-identical Problems: the no-seed
    reproducibility mechanism (sample once, freeze).
- **Unified distribution framework**: the `fastsrb` distribution interprets the FastSRB
  `sample_range`/`sample_type` recipe as one nestable distribution within the existing
  named/nested/mixture vocabulary. All distributions thread a `numpy.random.Generator`. (Finding:
  log-uniform is base-invariant, so FastSRB's base-10 `log` is value-equivalent to the native
  natural-log `log_uniform`.)

### Changed / Removed (breaking)
- Removed `load_benchmark`, `load_spec`, `BENCHMARKS`, `SpecBenchmark`, `FastSRBBenchmark`, and
  `datasets.py` (replaced by `load_catalog` / `ProblemCatalog` / the resolver).
- The skeleton-sampling machinery (`SkeletonPool`, `SkeletonSampler`, `SupportSampler`,
  `HoldoutManager`, `Sample`, `sample_from_skeleton`, `iter_samples`) is no longer public -- it is an
  internal detail of generate-mode `ProblemSource`. `NoValidSampleFoundError` remains exported.
- Reproducibility is no longer seed-based: sampling threads a `Generator` (entropy by default) and
  exact reproduction comes from `materialize()`.

### Migration
- `load_benchmark("feynman")` -> `load_catalog("feynman")` (returns a `ProblemCatalog`; inspect
  `cat["I.6.2a"].prepared` / `.variables`).
- To get `(X, y)` problems: `ProblemSource({"catalog": "feynman", "sampling": {...}})` then iterate.

### Deferred (tracked for 0.4.x)
- Generate-mode's internal skeleton sampler still uses global `np.random`; threading it onto the
  source's `Generator` and fully folding it into `ProblemSource` internals is a 0.4.1 refinement (it
  is distribution-correct today, behind the clean `ProblemSource` API).
- Publishing the Hugging Face asset manifest + a frozen `holdout_grid` asset; `to_catalog()`
  (persistent frozen catalogs).

## [0.3.0] - 2026-06-28

### Added
- **Curated benchmark loaders `feynman` and `nguyen`** for `load_benchmark`, alongside `fastsrb` --
  all three now vendored as package data from their canonical upstreams (no download) and stamping
  `benchmark.provenance`:
  - `fastsrb` -- the 120-equation FastSRB spec, vendored verbatim from upstream `viktmar/FastSRB`
    `src/expressions.yaml` (MIT).
  - `feynman` -- the 100-equation Feynman Symbolic Regression Database (Udrescu & Tegmark 2020),
    formulas + uniform FSReD ranges (via the `psaegert/ansr-data` `FeynmanEquations.csv` mirror).
  - `nguyen` -- the 12-equation Nguyen suite (Uy et al. 2011), formulas + ranges from the
    `deep-symbolic-optimization` `benchmarks.csv` (Petersen et al. 2021, BSD-3).
- `SpecBenchmark` -- the general spec-driven sampler extracted from `FastSRBBenchmark` (now a thin
  subclass), accepting either a YAML path or an already-parsed mapping. Exported from `symbolic_data`.
- `tools/build_benchmark_specs.py` -- reproducible, self-verifying generator that fetches each
  benchmark from its canonical upstream, converts it, and gates the converted specs on a numerical
  oracle (`simplipy(prepared)` vs `sympy(raw)`, `allclose` on shared inputs) before writing.

### Fixed
- `load_benchmark("fastsrb")` now works out of the box. Previously the default resolved an
  `expressions.yaml` from the `psaegert/ansr-data` HF dataset that was never uploaded there, so the
  default 404'd. The spec is now vendored as package data (HF remains available via `revision=...`).

### Verified
- All 100 Feynman + 12 Nguyen equations pass the numerical oracle at `rtol=1e-9` (the converted
  specs). The same oracle runs offline over the shipped specs in the test suite (sympy-gated), plus a
  finite-sampling integrity guard. `fastsrb` is vendored verbatim, so it is gated on parse +
  finite-sampling integrity instead (118/120 sample finite; `II.24.17` and `B4` are mostly-non-finite
  by construction upstream and are skipped gracefully by `iter_samples`).
- Six known `# variables` count typos in the upstream `FeynmanEquations.csv` are corrected from the
  populated columns (reported by the build script, not silently dropped).

### Licenses
- `THIRD_PARTY_LICENSES` now reproduces the MIT (FastSRB / viktmar) and BSD-3-Clause (DSO) license
  texts and attributes the FSReD source; the curated specs reproduce only mathematical facts
  (formulas, ranges, variable names).

## [0.2.0] - 2026-06-28

### Added
- Data-prep CLI (`symbolic-data generate | import | split-skeleton-pool`) and benchmark ingest
  (`ParserFactory`), with the `[ingest]` extra.

## [0.1.0] - 2026-06-28

### Added
- Initial release: the model-agnostic symbolic-regression data layer carved from flash-ansr --
  skeleton/expression sampling, priors, holdout, `iter_samples`, registries, and `load_benchmark`.
