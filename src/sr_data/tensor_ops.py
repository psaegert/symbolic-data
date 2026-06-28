"""Small model-agnostic array helpers for the data layer.

Vendored into sr-data (like ``paths``/``config_io``) so the sampling core has no
upward dependency on a consuming package. flash-ansr keeps its own copy for the
model-input pipeline; the deliberate, contract-tested duplication is the price of
keeping each package's dependency graph clean.
"""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np


def mask_unused_variable_columns(
    arrays: Iterable[np.ndarray],
    *,
    variables: Sequence[str] | None,
    skeleton_tokens: Sequence[str] | None,
) -> None:
    """Zero, in place, the columns of each array for variables absent from the skeleton.

    Column ``i`` corresponds to ``variables[i]``. A variable not appearing in
    ``skeleton_tokens`` carries no signal for the problem, so its column is set to 0
    (the convention shared by training and evaluation for zero-padded inputs). No-op
    when every variable is used, when there are no variables, or for non-2D arrays.
    """
    if not variables:
        return

    variable_to_index = {var: idx for idx, var in enumerate(variables)}
    if not variable_to_index:
        return

    used_variables: set[str] = set()
    if skeleton_tokens:
        for token in skeleton_tokens:
            if token in variable_to_index:
                used_variables.add(token)

    if len(used_variables) == len(variable_to_index):
        return

    unused_indices = [variable_to_index[var] for var in variables if var not in used_variables]
    if not unused_indices:
        return

    for array in arrays:
        if not isinstance(array, np.ndarray):
            continue
        if array.ndim < 2 or array.shape[1] == 0:
            continue
        for idx in unused_indices:
            if idx < array.shape[1]:
                array[:, idx] = 0
