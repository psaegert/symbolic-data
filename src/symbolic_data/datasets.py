"""Benchmark dataset loaders.

``load_benchmark(name)`` resolves a named benchmark to a ready-to-sample object, fetching
its (HF-versioned) equation spec on demand. Benchmarks live in the ``BENCHMARKS`` registry,
so third parties can add loaders via ``@BENCHMARKS.register`` or ``symbolic_data.benchmarks`` entry
points.

``load_benchmark`` ships three curated loaders:

* ``fastsrb`` -- the FastSRB benchmark (Martinek, arXiv:2508.14481); its spec is fetched from the
  ``psaegert/ansr-data`` HF dataset, or read from a local ``spec_path``.
* ``feynman`` -- the 100-equation Feynman Symbolic Regression Database (Udrescu & Tegmark 2020).
* ``nguyen`` -- the 12-equation Nguyen suite (Uy et al. 2011; DSO/DSR ranges).

The ``feynman`` and ``nguyen`` specs ship as package data (regenerated + numerically verified by
``tools/build_benchmark_specs.py``); ``fastsrb`` resolves its spec from HuggingFace. All three return
a :class:`~symbolic_data.benchmarks.spec.SpecBenchmark` and stamp ``benchmark.provenance``.
"""
from __future__ import annotations

from importlib import resources
from typing import Any

import yaml

from symbolic_data.benchmarks import FastSRBBenchmark, SpecBenchmark
from symbolic_data.registry import Registry

__all__ = ["BENCHMARKS", "load_benchmark", "SpecBenchmark", "FastSRBBenchmark"]

# The HF dataset that versions the canonical benchmark specs.
ANSR_DATA_REPO = "psaegert/ansr-data"
FASTSRB_SPEC = "test_set/fastsrb/expressions.yaml"

# Version stamp for the package-data curated specs (bumped when a spec changes; see CHANGELOG).
CURATED_SPEC_VERSION = "1.0"

BENCHMARKS = Registry("benchmark", entry_point_group="symbolic_data.benchmarks")


def _load_packaged_spec(filename: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load a curated spec shipped as package data; return (spec, provenance)."""
    ref = resources.files("symbolic_data.benchmarks").joinpath("data", filename)
    spec = yaml.safe_load(ref.read_text(encoding="utf-8"))
    provenance = {"source": "package", "package": "symbolic_data", "resource": f"benchmarks/data/{filename}", "spec_version": CURATED_SPEC_VERSION}
    return spec, provenance


def _curated_loader(name: str, filename: str, *, spec_path: str | None, simplipy_engine: Any, random_state: Any, **kwargs: Any) -> SpecBenchmark:
    """Shared body for the package-data curated loaders (feynman, nguyen)."""
    if spec_path is not None:
        spec: Any = str(spec_path)
        provenance = {"source": "local", "path": str(spec_path)}
    else:
        spec, provenance = _load_packaged_spec(filename)
    benchmark = SpecBenchmark(spec, name=name, simplipy_engine=simplipy_engine, random_state=random_state, **kwargs)
    provenance.update({"benchmark": name, "simplipy_engine": str(simplipy_engine)})
    benchmark.provenance = provenance
    return benchmark


def _resolve_spec(
    spec_path: str | None,
    *,
    repo_id: str,
    filename: str,
    revision: str | None,
) -> tuple[str, dict[str, Any]]:
    """Return (path, provenance). Local ``spec_path`` wins; else fetch from the HF dataset."""
    if spec_path is not None:
        return str(spec_path), {"source": "local", "path": str(spec_path)}
    from huggingface_hub import hf_hub_download

    resolved = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset", revision=revision)
    return resolved, {
        "source": "huggingface",
        "repo_id": repo_id,
        "filename": filename,
        "revision": revision,
        "resolved_path": resolved,
    }


@BENCHMARKS.register("fastsrb")
def _load_fastsrb(
    *,
    spec_path: str | None = None,
    simplipy_engine: Any = "dev_7-3",
    random_state: Any = None,
    revision: str | None = None,
    **kwargs: Any,
) -> FastSRBBenchmark:
    resolved, provenance = _resolve_spec(
        spec_path, repo_id=ANSR_DATA_REPO, filename=FASTSRB_SPEC, revision=revision
    )
    benchmark = FastSRBBenchmark(
        resolved, simplipy_engine=simplipy_engine, random_state=random_state, **kwargs
    )
    provenance.update({"benchmark": "fastsrb", "simplipy_engine": str(simplipy_engine)})
    benchmark.provenance = provenance  # stamp source for reproducibility (provenance principle)
    return benchmark


@BENCHMARKS.register("feynman")
def _load_feynman(
    *,
    spec_path: str | None = None,
    simplipy_engine: Any = "dev_7-3",
    random_state: Any = None,
    **kwargs: Any,
) -> SpecBenchmark:
    return _curated_loader("feynman", "feynman.yaml", spec_path=spec_path, simplipy_engine=simplipy_engine, random_state=random_state, **kwargs)


@BENCHMARKS.register("nguyen")
def _load_nguyen(
    *,
    spec_path: str | None = None,
    simplipy_engine: Any = "dev_7-3",
    random_state: Any = None,
    **kwargs: Any,
) -> SpecBenchmark:
    return _curated_loader("nguyen", "nguyen.yaml", spec_path=spec_path, simplipy_engine=simplipy_engine, random_state=random_state, **kwargs)


def load_benchmark(name: str = "fastsrb", **kwargs: Any) -> Any:
    """Load a named benchmark; extra kwargs are forwarded to that benchmark's loader.

    Built-in loaders:

    * ``fastsrb`` -- ``load_benchmark('fastsrb', spec_path=None, simplipy_engine='dev_7-3',
      random_state=None, revision=None)``. With ``spec_path`` unset the FastSRB equation spec is
      fetched (and cached) from the ``psaegert/ansr-data`` HF dataset.
    * ``feynman`` -- ``load_benchmark('feynman', spec_path=None, simplipy_engine='dev_7-3',
      random_state=None)``. 100 equations from the Feynman Symbolic Regression Database, shipped as
      package data.
    * ``nguyen`` -- ``load_benchmark('nguyen', spec_path=None, simplipy_engine='dev_7-3',
      random_state=None)``. The 12-equation Nguyen suite, shipped as package data.

    Pass ``spec_path`` to any curated loader to read a custom spec file instead of the built-in one.
    """
    return BENCHMARKS.get(name)(**kwargs)
