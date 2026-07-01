"""Helpers for compiling and evaluating expression programs."""
from typing import Callable

import numpy as np


def safe_f(f: Callable, X: np.ndarray, constants: np.ndarray | None = None) -> np.ndarray:
    """Evaluate ``f`` on ``X`` while normalising scalar outputs to vectors."""
    if constants is None:
        y = f(*X.T)
    else:
        y = f(*X.T, *constants)
    if not isinstance(y, np.ndarray) or y.ndim == 0 or y.shape[0] == 1:
        y = np.full(X.shape[0], y)
    return y
