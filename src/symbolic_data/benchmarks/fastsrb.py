"""Utilities for sampling the FastSRB benchmark equations using SimpliPy.

Translated and adapted from the Julia FastSRB benchmarking code by Viktor Martinek
(https://github.com/viktmar/FastSRB, arXiv:2508.14481), distributed under the MIT License.
The full notice + citation is reproduced in ``THIRD_PARTY_LICENSES`` (FastSRB section).

``FastSRBBenchmark`` is a thin subclass of the general :class:`~symbolic_data.benchmarks.spec.SpecBenchmark`
sampler (the shared engine behind every curated benchmark loader); it only fixes the benchmark
name and preserves the historical ``FastSRBBenchmark(yaml_path, ...)`` constructor signature.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np

from simplipy import SimpliPyEngine

from symbolic_data.benchmarks.spec import SpecBenchmark


class FastSRBBenchmark(SpecBenchmark):
    """Sample datasets from the FastSRB benchmark YAML specification."""

    def __init__(
        self,
        yaml_path: Union[str, Path],
        *,
        simplipy_engine: SimpliPyEngine | str = "dev_7-3",
        random_state: Optional[Union[int, np.random.Generator]] = None,
    ) -> None:
        """Load the FastSRB YAML benchmark specification and prepare a SimpliPy engine."""
        super().__init__(
            yaml_path,
            name="fastsrb",
            simplipy_engine=simplipy_engine,
            random_state=random_state,
        )
