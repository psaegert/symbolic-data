"""Holdout management for expression skeleton sampling."""
from dataclasses import dataclass, field
from typing import Callable, Sequence, Tuple
import functools
import warnings

import numpy as np

from symbolic_data.compilation import safe_f
from simplipy import normalize_skeleton


# Fixed seed for the default holdout grid. Previously each HoldoutManager drew a fresh
# unseeded ``np.random.uniform`` grid, so the functional-equivalence holdout (the image-key
# backstop) was non-deterministic across constructions/processes/runs: the SAME config could
# decontaminate different borderline skeletons on different runs (the exact-symbolic path was
# already deterministic; only the 4-dp image-key margin varied). Seeding makes the default
# grid reproducible. NOTE: this seeded default is a reproducible recipe, not a frozen
# artifact; a canonical, version-pinned grid ASSET (shipped ``.npz``) can replace it later.
_DEFAULT_HOLDOUT_GRID_SEED = 20240617


@functools.lru_cache(maxsize=1)
def _default_holdout_grid() -> tuple[np.ndarray, np.ndarray]:
    """Build the deterministic default holdout grid once per process (callers copy)."""
    rng = np.random.default_rng(_DEFAULT_HOLDOUT_GRID_SEED)
    holdout_X = rng.uniform(-10, 10, (512, 100))
    holdout_C = rng.uniform(-10, 10, (100,))
    return holdout_X, holdout_C


@dataclass
class HoldoutManager:
    """Track held-out expressions by both skeleton hash AND functional image (evaluated on a
    fixed grid), so structurally-distinct but functionally-equivalent expressions are also
    excluded."""

    n_variables: int
    allow_nan: bool
    holdout_X: np.ndarray = field(default_factory=lambda: _default_holdout_grid()[0].copy())
    holdout_C: np.ndarray = field(default_factory=lambda: _default_holdout_grid()[1].copy())
    skeleton_hashes: set[Tuple[str, ...]] = field(default_factory=set)
    expression_images: set[Tuple[float, ...] | Tuple[Tuple[float, ...], ...]] = field(default_factory=set)

    def register_skeleton(
        self,
        skeleton_tokens: Sequence[str],
        compiled_fn: Callable[..., np.ndarray | float],
        num_constants: int,
        *,
        n_variables: int | None = None,
    ) -> None:
        skeleton_key = tuple(self._normalize_tokens(skeleton_tokens))
        self.skeleton_hashes.add(skeleton_key)

        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                key = self._evaluate_to_key(compiled_fn, num_constants, n_variables)
        except (OverflowError, NameError) as exc:
            # The structure layer is registered; only the functional-image layer is lost. That
            # is a real degradation of the equivalence backstop for THIS skeleton, so say so
            # instead of silently returning (a swallowed NameError here hid a variable-binding
            # bug for wider-than-catalog holdout laws).
            warnings.warn(
                f"holdout image registration failed for skeleton {skeleton_key!r} "
                f"({type(exc).__name__}: {exc}); only the exact-structure layer covers it",
                RuntimeWarning, stacklevel=2)
            return

        self.expression_images.add(key)

    def is_held_out(
        self,
        skeleton_tokens: Sequence[str],
        compiled_fn: Callable[..., np.ndarray | float],
        num_constants: int,
        *,
        n_variables: int | None = None,
    ) -> bool:
        skeleton_key = tuple(self._normalize_tokens(skeleton_tokens))
        if skeleton_key in self.skeleton_hashes:
            return True

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            key = self._evaluate_to_key(compiled_fn, num_constants, n_variables)
        return key in self.expression_images

    @staticmethod
    def _normalize_tokens(tokens: Sequence[str]) -> list[str]:
        # Canonicalize before hashing: variable renames (v1 -> x1) and numeric literals
        # (3.5 -> <constant>) must not defeat the exact-match layer; literal skeletons
        # otherwise leak through BOTH holdout layers (mirror of the flash-ansr fix).
        return list(normalize_skeleton([str(token) for token in tokens]))

    def _evaluate_to_key(
        self,
        compiled_fn: Callable[..., np.ndarray | float],
        num_constants: int,
        n_variables: int | None = None,
    ) -> Tuple[Tuple[float, ...], ...] | Tuple[float, ...]:
        variable_count = n_variables if n_variables is not None else self.n_variables
        samples = self.holdout_X[:, :variable_count]
        constants_slice = self.holdout_C[:num_constants]
        constants_arg = None if num_constants == 0 else constants_slice

        image = safe_f(compiled_fn, samples, constants_arg)
        image = np.asarray(image, dtype=np.float64)
        image = np.round(image, 4)

        if np.isnan(image).any():
            image = image.copy()
            image[np.isnan(image)] = 0.0

        if image.ndim == 1:
            return tuple(float(value) for value in image.tolist())

        return tuple(tuple(float(value) for value in row.tolist()) for row in image.tolist())
