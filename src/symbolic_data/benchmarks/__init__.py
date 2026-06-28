"""Benchmark loaders (model-agnostic equation specs + (X, y) sampling)."""
from symbolic_data.benchmarks.spec import SpecBenchmark
from symbolic_data.benchmarks.fastsrb import FastSRBBenchmark

__all__ = ["SpecBenchmark", "FastSRBBenchmark"]
