"""Helpers for compiling and evaluating expression programs."""
from typing import Callable

import numpy as np


def safe_f(f: Callable, X: np.ndarray, constants: np.ndarray | None = None) -> np.ndarray:
    """Evaluate ``f`` on ``X`` while normalising scalar outputs to vectors.

    An evaluation numpy refuses outright returns an all-NaN vector instead of
    raising: lambdify folds pure-integer subtrees into arbitrary-precision Python
    ints, and ufuncs raise TypeError on them (OverflowError on the int-to-f64
    boundary). Seen live twice on 2026-08-24 -- np.exp(bigint) through the holdout
    check and np.sinh(bigint) through the data sampler, each killing a streaming
    worker. NaN is exactly the signal every caller already rejects on (box
    resampling, allow_nan gates, the holdout sentinel), so the pathological
    candidate is discarded instead of crashing the producer.
    """
    try:
        if constants is None:
            y = f(*X.T)
        else:
            y = f(*X.T, *constants)
        if not isinstance(y, np.ndarray) or y.ndim == 0 or y.shape[0] == 1:
            y = np.full(X.shape[0], y)
        return y
    except (TypeError, OverflowError):
        return np.full(X.shape[0], np.nan)
