# Changelog

All notable changes to `symbolic-data` are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and the project adheres to semantic versioning.

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
