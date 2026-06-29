# Changelog

All notable changes to `symbolic-data` are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and the project adheres to semantic versioning.

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
