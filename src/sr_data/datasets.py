"""Benchmark dataset loaders.

``load_benchmark(name)`` resolves a named benchmark to a ready-to-sample object, fetching
its (HF-versioned) equation spec on demand. Benchmarks live in the ``BENCHMARKS`` registry,
so third parties can add loaders via ``@BENCHMARKS.register`` or ``sr_data.benchmarks`` entry
points.

v1 ships the FastSRB loader; its spec is versioned in the ``psaegert/ansr-data`` HF dataset and
fetched (and cached) by ``huggingface_hub`` on first use, or read from a local ``spec_path``.
Curated sets (feynman/nguyen) are follow-on thin loaders.
"""
from __future__ import annotations

from typing import Any

from sr_data.benchmarks import FastSRBBenchmark
from sr_data.registry import Registry

__all__ = ["BENCHMARKS", "load_benchmark", "FastSRBBenchmark"]

# The HF dataset that versions the canonical benchmark specs.
ANSR_DATA_REPO = "psaegert/ansr-data"
FASTSRB_SPEC = "test_set/fastsrb/expressions.yaml"

BENCHMARKS = Registry("benchmark", entry_point_group="sr_data.benchmarks")


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


def load_benchmark(name: str = "fastsrb", **kwargs: Any) -> Any:
    """Load a named benchmark; extra kwargs are forwarded to that benchmark's loader.

    fastsrb: ``load_benchmark('fastsrb', spec_path=None, simplipy_engine='dev_7-3',
    random_state=None, revision=None)``. With ``spec_path`` unset the FastSRB equation spec
    is fetched (and cached) from the ``psaegert/ansr-data`` HF dataset.
    """
    return BENCHMARKS.get(name)(**kwargs)
