# Benchmarks

`load_benchmark(name)` resolves a named benchmark to a ready-to-sample object. Three loaders ship
built in: `fastsrb` (spec fetched + cached from the `psaegert/ansr-data` HuggingFace dataset), and
the curated `feynman` and `nguyen` sets (specs shipped as package data, no download).

```python
import symbolic_data

bench = symbolic_data.load_benchmark("feynman")          # package data, no download
dataset = bench.sample("I.6.2a", n_points=100, random_state=0)
ids = bench.equation_ids()
```

## FastSRB

v1 ships the **FastSRB** loader (the Fast Symbolic Regression Benchmark by Viktor Martinek,
[arXiv:2508.14481](https://arxiv.org/abs/2508.14481)). Its equation spec is versioned in the
[`psaegert/ansr-data`](https://huggingface.co/datasets/psaegert/ansr-data) HuggingFace dataset and
fetched (and cached) by `huggingface_hub` on first use.

```python
# default: fetch the spec from HF (optionally pin a dataset revision)
bench = symbolic_data.load_benchmark("fastsrb", revision="main", random_state=0)

# or point at a local spec file
bench = symbolic_data.load_benchmark("fastsrb", spec_path="expressions.yaml")
```

The loader records where the spec came from on `bench.provenance` (source, repo id, filename,
revision) — see [reproducibility](#provenance).

Every benchmark object (`FastSRBBenchmark` and the curated `SpecBenchmark` loaders below) exposes
`equation_ids()`, `sample(eq_id, n_points=..., random_state=...)`, `sample_multiple(...)`, and
`iter_samples(...)`.

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

`load_benchmark("nguyen")` returns the **12-equation Nguyen suite** (Uy et al. 2011), with formulas
and sampling ranges pinned to the DSO/DSR standard (Petersen et al. 2021). Nguyen-1..10 are
cross-confirmed against the `psaegert/ansr-data` `nguyen.csv`; Nguyen-11/12 complete the canonical
suite. Ships as package data.

```python
nguyen = symbolic_data.load_benchmark("nguyen", random_state=0)
dataset = nguyen.sample("Nguyen-5", n_points=20)   # sin(x^2)*cos(x) - 1 on U[-1, 1]
```

Both curated specs are regenerated and **numerically verified** against their source formulas (a
`sympy` oracle: `simplipy(prepared)` vs `sympy(raw)`) by `tools/build_benchmark_specs.py`; the same
oracle runs offline over the shipped specs in the test suite.

Pass `spec_path=...` to any loader to read a custom spec file in the same format.

## Provenance

Every benchmark stamps its source on `.provenance`, so a downstream artifact can record exactly
which spec (and dataset revision) produced it:

```python
bench.provenance
# {'source': 'huggingface', 'repo_id': 'psaegert/ansr-data',
#  'filename': 'test_set/fastsrb/expressions.yaml', 'revision': None,
#  'benchmark': 'fastsrb', 'simplipy_engine': 'dev_7-3', ...}
```

## Registering a benchmark

Benchmarks live in the extensible `BENCHMARKS` registry — add your own in-process or across packages
via entry points. See [Registries & extensibility](registries.md).
