"""Tests for ``load_benchmark`` and the benchmark registry.

Uses a self-contained 2-equation FastSRB spec fixture (a slice of the real
``psaegert/ansr-data`` spec) so the loader + simplipy sampling path is exercised
without a network round-trip. The HF-fetch path is covered by a network-gated test.
"""
import os

import pytest

from symbolic_data import BENCHMARKS, FastSRBBenchmark, load_benchmark

FIXTURE = os.path.join(os.path.dirname(__file__), "data", "fastsrb_mini.yaml")


def test_load_fastsrb_from_local_spec():
    bench = load_benchmark("fastsrb", spec_path=FIXTURE)
    assert isinstance(bench, FastSRBBenchmark)
    ids = bench.equation_ids()
    assert "II.38.3" in ids
    assert len(ids) == 2
    assert bench.provenance["source"] == "local"
    assert bench.provenance["benchmark"] == "fastsrb"


def test_fastsrb_samples_a_problem():
    bench = load_benchmark("fastsrb", spec_path=FIXTURE, random_state=0)
    result = bench.sample("II.38.3", n_points=16, random_state=0)
    assert isinstance(result, dict) and result  # non-empty -> full spec+engine path worked


def test_unknown_benchmark_raises_with_helpful_message():
    with pytest.raises(KeyError, match="Unknown benchmark"):
        load_benchmark("does_not_exist")


def test_benchmark_registry_is_extensible():
    @BENCHMARKS.register("dummy_bench")
    def _loader(**kwargs):
        return {"ok": True, **kwargs}

    try:
        assert load_benchmark("dummy_bench", foo=1) == {"ok": True, "foo": 1}
    finally:
        BENCHMARKS._fns.pop("dummy_bench", None)


def test_fastsrb_registered_as_builtin_loader():
    assert "fastsrb" in BENCHMARKS
