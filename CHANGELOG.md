# Changelog

All notable changes to `symbolic-data` are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and the project adheres to semantic versioning.

## [0.3.0] - 2026-06-28

### Added
- **Curated benchmark loaders `feynman` and `nguyen`** for `load_benchmark`, alongside `fastsrb`:
  - `feynman` -- the 100-equation Feynman Symbolic Regression Database (Udrescu & Tegmark 2020),
    formulas + uniform FSReD sampling ranges.
  - `nguyen` -- the 12-equation Nguyen suite (Uy et al. 2011), formulas + ranges per the DSO/DSR
    standard (Petersen et al. 2021); Nguyen-1..10 cross-confirmed against `psaegert/ansr-data`.
  - Both specs ship as package data (no download) and stamp `benchmark.provenance`.
- `SpecBenchmark` -- the general spec-driven sampler extracted from `FastSRBBenchmark` (now a thin
  subclass), accepting either a YAML path or an already-parsed mapping. Exported from `symbolic_data`.
- `tools/build_benchmark_specs.py` -- reproducible, self-verifying generator for the curated specs.
  It converts the canonical sources and gates on a numerical oracle (`simplipy(prepared)` vs
  `sympy(raw)`, `allclose` on shared inputs) for every equation before writing.

### Verified
- All 100 Feynman + 12 Nguyen equations pass the numerical oracle at `rtol=1e-7`. The same oracle
  runs offline over the shipped specs in the test suite (sympy-gated), plus a finite-sampling
  integrity guard over every equation.
- Six known `# variables` count typos in the upstream `FeynmanEquations.csv` are corrected from the
  populated columns (reported by the build script, not silently dropped).

### Notes
- `load_benchmark("fastsrb")` still resolves its spec from the `psaegert/ansr-data` HF dataset; that
  spec file is not yet present on the remote (a pre-existing gap, tracked separately). The `feynman`
  and `nguyen` loaders are unaffected (package data).

## [0.2.0] - 2026-06-28

### Added
- Data-prep CLI (`symbolic-data generate | import | split-skeleton-pool`) and benchmark ingest
  (`ParserFactory`), with the `[ingest]` extra.

## [0.1.0] - 2026-06-28

### Added
- Initial release: the model-agnostic symbolic-regression data layer carved from flash-ansr --
  skeleton/expression sampling, priors, holdout, `iter_samples`, registries, and `load_benchmark`.
