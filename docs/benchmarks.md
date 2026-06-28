# Benchmarks

`load_benchmark(name)` resolves a named benchmark to a ready-to-sample object. Three loaders ship
built in — `fastsrb`, `feynman`, and `nguyen` — each **vendored as package data** from its canonical
upstream (regenerated + verified by `tools/build_benchmark_specs.py`), so no download is needed.

```python
import symbolic_data

bench = symbolic_data.load_benchmark("feynman")          # package data, no download
dataset = bench.sample("I.6.2a", n_points=100, random_state=0)
ids = bench.equation_ids()
```

Every benchmark object exposes `equation_ids()`, `sample(eq_id, n_points=..., random_state=...)`,
`sample_multiple(...)`, and `iter_samples(...)`, and records where its spec came from on
`bench.provenance` (see [reproducibility](#provenance)).

## FastSRB

`load_benchmark("fastsrb")` returns the **FastSRB** benchmark (Viktor Martinek,
[arXiv:2508.14481](https://arxiv.org/abs/2508.14481); 120 Feynman-family equations with
physically-motivated SRSD ranges). The spec is vendored verbatim from the upstream
[`viktmar/FastSRB`](https://github.com/viktmar/FastSRB) `src/expressions.yaml` (MIT).

```python
bench = symbolic_data.load_benchmark("fastsrb", random_state=0)   # package data
dataset = bench.sample("II.38.3", n_points=100)

# opt in to the HF-versioned spec instead, or a local file
bench = symbolic_data.load_benchmark("fastsrb", revision="main")
bench = symbolic_data.load_benchmark("fastsrb", spec_path="expressions.yaml")
```

A couple of upstream FastSRB equations (e.g. `II.24.17`, `B4`) are mostly-non-finite by construction
under their own ranges (a `sqrt` whose argument is usually negative); `sample()` on those may raise,
while `iter_samples()` skips them gracefully per-equation.

## Feynman

`load_benchmark("feynman")` returns the **100-equation Feynman Symbolic Regression Database**
(Udrescu & Tegmark 2020, [AI Feynman](https://arxiv.org/abs/1905.11481)). Formulas and per-variable
sampling ranges come from the canonical `FeynmanEquations.csv` (uniform FSReD ranges). The spec ships
as package data, so no download is needed.

```python
feynman = symbolic_data.load_benchmark("feynman", random_state=0)
feynman.equation_ids()[:3]            # ['I.6.2a', 'I.6.2', 'I.6.2b']
dataset = feynman.sample("I.9.18", n_points=100)   # Newton's gravitation, 9 variables
```

> FastSRB vs Feynman: both draw on the Feynman equation family but are distinct benchmarks. `fastsrb`
> is Martinek's curated set (~120 equations) with physically-motivated ranges (SRSD, Matsubara et al.
> 2024); `feynman` here is the 100-equation core Feynman Symbolic Regression Database
> (`FeynmanEquations.csv`) with the original uniform FSReD ranges.

For equations with a restricted domain (e.g. the two `arcsin` equations), the sampler rejects
out-of-domain draws and resamples, so the realized input distribution is conditioned on the
function's domain rather than uniform over the full declared box.

## Nguyen

`load_benchmark("nguyen")` returns the **12-equation Nguyen suite** (Uy et al. 2011). Formulas and
sampling ranges are derived from the canonical `benchmarks.csv` of the DSO/DSR project (Petersen et
al. 2021, [deep-symbolic-optimization](https://github.com/dso-org/deep-symbolic-optimization)). Ships
as package data.

```python
nguyen = symbolic_data.load_benchmark("nguyen", random_state=0)
dataset = nguyen.sample("Nguyen-5", n_points=20)   # sin(x^2)*cos(x) - 1 on U[-1, 1]
```

The `feynman` and `nguyen` specs (the ones converted from source) are regenerated and **numerically
verified** against their source formulas (a `sympy` oracle: `simplipy(prepared)` vs `sympy(raw)`) by
`tools/build_benchmark_specs.py`; the same oracle runs offline over the shipped specs in the test
suite. `fastsrb` is vendored verbatim, so it is gated on parse + finite-sampling integrity instead.

Pass `spec_path=...` to any loader to read a custom spec file in the same format.

## Provenance

Every benchmark stamps its source on `.provenance`, so a downstream artifact can record exactly
which spec (and dataset revision) produced it:

```python
bench.provenance
# {'source': 'package', 'package': 'symbolic_data',
#  'resource': 'benchmarks/data/feynman.yaml', 'spec_version': '1.0',
#  'benchmark': 'feynman', 'simplipy_engine': 'dev_7-3'}
```

## Registering a benchmark

Benchmarks live in the extensible `BENCHMARKS` registry — add your own in-process or across packages
via entry points. See [Registries & extensibility](registries.md).
