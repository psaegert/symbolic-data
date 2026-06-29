"""Benchmark dataset loaders.

``load_benchmark(name)`` resolves a named benchmark to a ready-to-sample object. Benchmarks live in
the ``BENCHMARKS`` registry, so third parties can add loaders via ``@BENCHMARKS.register`` or
``symbolic_data.benchmarks`` entry points.

``load_benchmark`` ships three curated loaders, each vendored as package data from its canonical
upstream and regenerated + verified by ``tools/build_benchmark_specs.py``:

* ``fastsrb`` -- the FastSRB benchmark (Martinek, arXiv:2508.14481), vendored verbatim from the
  upstream ``viktmar/FastSRB`` (MIT). Pass ``revision`` to instead fetch the HF-versioned spec.
* ``feynman`` -- the 100-equation Feynman Symbolic Regression Database (Udrescu & Tegmark 2020).
* ``nguyen`` -- the 12-equation Nguyen suite (Uy et al. 2011), from the DSO/DSR ``benchmarks.csv``.

All three return a :class:`~symbolic_data.benchmarks.spec.SpecBenchmark` and stamp
``benchmark.provenance``. Pass ``spec_path`` to any loader to read a custom local spec file.
"""
from __future__ import annotations

from importlib import resources
from typing import Any

import yaml

from symbolic_data.benchmarks import FastSRBBenchmark, SpecBenchmark
from symbolic_data.registry import Registry

__all__ = ["BENCHMARKS", "load_benchmark", "load_spec", "SpecBenchmark", "FastSRBBenchmark"]

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


def _load_packaged_header(name: str) -> dict[str, Any] | None:
    """Load a benchmark-spec *header* (``specs/<name>.yaml``) shipped as package data, or None."""
    ref = resources.files("symbolic_data.benchmarks").joinpath("specs", f"{name}.yaml")
    try:
        text = ref.read_text(encoding="utf-8")
    except (FileNotFoundError, OSError):
        return None
    return yaml.safe_load(text)


def _resolve_problems(ref: str, *, spec_path: str | None) -> tuple[Any, dict[str, Any]]:
    """Resolve a spec header's ``problems`` reference to (problem-set, provenance).

    MVP resolver: a local ``spec_path`` wins; else ``ref`` names a package-data problem-set
    (``benchmarks/data/<ref>.yaml``). (HF-versioned ``name@version`` resolution is a later increment.)
    """
    if spec_path is not None:
        return str(spec_path), {"source": "local", "path": str(spec_path)}
    return _load_packaged_spec(f"{ref}.yaml")


def load_spec(
    name: str,
    *,
    spec_path: str | None = None,
    simplipy_engine: Any = "dev_7-3",
    random_state: Any = None,
    **kwargs: Any,
) -> SpecBenchmark:
    """Build a benchmark from its versioned spec *header* (``benchmarks/specs/<name>.yaml``).

    The header (``metadata`` / ``source`` / ``sampling`` / ``problems``) references a problem-set,
    which is resolved (package data, or a local ``spec_path`` override) and wrapped in a
    :class:`SpecBenchmark` carrying the header. The header's ``sampling`` block becomes the canonical
    per-call default (e.g. Nguyen's ``n_points: 20``). ``benchmark.provenance`` records the spec
    version + the resolved problem-set.
    """
    header = _load_packaged_header(name)
    if header is None:
        raise KeyError(f"No benchmark spec for {name!r}")
    meta = header.get("metadata", {})
    problems, problems_prov = _resolve_problems(header.get("problems", name), spec_path=spec_path)
    benchmark = SpecBenchmark(problems, name=meta.get("name", name), header=header, simplipy_engine=simplipy_engine, random_state=random_state, **kwargs)
    benchmark.provenance = {
        "source": "package",
        "benchmark": meta.get("name", name),
        "spec": f"benchmarks/specs/{name}.yaml",
        "spec_version": meta.get("version", CURATED_SPEC_VERSION),
        "problems": problems_prov,
        "simplipy_engine": str(simplipy_engine),
    }
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
    """Load the FastSRB benchmark.

    Defaults to the spec vendored as package data (from the upstream ``viktmar/FastSRB``, MIT). Pass
    ``spec_path`` to read a local file, or ``revision`` to fetch the HF-versioned spec from
    ``psaegert/ansr-data`` instead.
    """
    header = _load_packaged_header("fastsrb")
    if spec_path is not None:
        problems: Any = str(spec_path)
        problems_prov = {"source": "local", "path": str(spec_path)}
    elif revision is not None:
        resolved, problems_prov = _resolve_spec(None, repo_id=ANSR_DATA_REPO, filename=FASTSRB_SPEC, revision=revision)
        problems = resolved
    else:
        problems, problems_prov = _load_packaged_spec("fastsrb.yaml")
    benchmark = FastSRBBenchmark(problems, header=header, simplipy_engine=simplipy_engine, random_state=random_state, **kwargs)
    meta = (header or {}).get("metadata", {})
    benchmark.provenance = {
        "source": "package" if (spec_path is None and revision is None) else problems_prov.get("source"),
        "benchmark": "fastsrb",
        "spec": "benchmarks/specs/fastsrb.yaml",
        "spec_version": meta.get("version", CURATED_SPEC_VERSION),
        "problems": problems_prov,
        "simplipy_engine": str(simplipy_engine),
    }
    return benchmark


@BENCHMARKS.register("feynman")
def _load_feynman(
    *,
    spec_path: str | None = None,
    simplipy_engine: Any = "dev_7-3",
    random_state: Any = None,
    **kwargs: Any,
) -> SpecBenchmark:
    return load_spec("feynman", spec_path=spec_path, simplipy_engine=simplipy_engine, random_state=random_state, **kwargs)


@BENCHMARKS.register("nguyen")
def _load_nguyen(
    *,
    spec_path: str | None = None,
    simplipy_engine: Any = "dev_7-3",
    random_state: Any = None,
    **kwargs: Any,
) -> SpecBenchmark:
    return load_spec("nguyen", spec_path=spec_path, simplipy_engine=simplipy_engine, random_state=random_state, **kwargs)


def load_benchmark(name: str = "fastsrb", **kwargs: Any) -> Any:
    """Load a named benchmark; extra kwargs are forwarded to that benchmark's loader.

    Built-in loaders (all ship their spec as package data):

    * ``fastsrb`` -- ``load_benchmark('fastsrb', spec_path=None, simplipy_engine='dev_7-3',
      random_state=None, revision=None)``. The 120-equation FastSRB spec, vendored from
      ``viktmar/FastSRB``. Pass ``revision`` to fetch the HF-versioned spec instead.
    * ``feynman`` -- ``load_benchmark('feynman', spec_path=None, simplipy_engine='dev_7-3',
      random_state=None)``. 100 equations from the Feynman Symbolic Regression Database.
    * ``nguyen`` -- ``load_benchmark('nguyen', spec_path=None, simplipy_engine='dev_7-3',
      random_state=None)``. The 12-equation Nguyen suite.

    Pass ``spec_path`` to any loader to read a custom spec file instead of the built-in one.
    """
    return BENCHMARKS.get(name)(**kwargs)
