# Benchmarks

`load_benchmark(name)` resolves a named benchmark to a ready-to-sample object, fetching its
(HuggingFace-versioned) equation spec on demand.

```python
import symbolic_data

bench = symbolic_data.load_benchmark("fastsrb")          # spec fetched + cached from HF
dataset = bench.sample("II.38.3", n_points=100, random_state=0)
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

`FastSRBBenchmark` exposes `equation_ids()`, `sample(eq_id, n_points=..., random_state=...)`,
`sample_multiple(...)`, and `iter_samples(...)`.

Curated sets (Feynman, Nguyen) are follow-on thin loaders.

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
