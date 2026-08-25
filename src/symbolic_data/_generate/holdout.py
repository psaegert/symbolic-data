"""Holdout management for expression skeleton sampling."""
from dataclasses import dataclass, field
from typing import Callable, Sequence, Tuple
import functools
import warnings

import numpy as np

from symbolic_data.compilation import safe_f
from symbolic_data.token_ops import normalize_skeleton


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


@functools.lru_cache(maxsize=1)
def _default_extra_grids() -> tuple[np.ndarray, ...]:
    """A second, positive log-scale probe grid. Half-domain respellings of a held-out law
    (exp(log(x)), sqrt(x)^2) agree with it exactly where BOTH are defined; a grid that
    lives inside that shared domain makes their images collide with the law's instead of
    being pushed apart by the NaN->0 fill on the mixed-sign grid (audit L4, 2026-08-25)."""
    rng = np.random.default_rng(_DEFAULT_HOLDOUT_GRID_SEED + 1)
    positive_X = 10.0 ** rng.uniform(-3, 1, (512, 100))
    return (positive_X,)


@functools.lru_cache(maxsize=1)
def _default_constant_fills() -> np.ndarray:
    """Extra per-slot constant fills for parametric (constant-bearing) skeletons: a pair
    equivalent only under PERMUTING its constant slots images differently under one fill
    but identically under some transposed fill; matching on ANY fill closes that gap
    (audit L8) at zero cost for the constant-free catalog path."""
    rng = np.random.default_rng(_DEFAULT_HOLDOUT_GRID_SEED + 2)
    return rng.uniform(-10, 10, (3, 100))


@dataclass
class HoldoutManager:
    """Track held-out expressions by both skeleton hash AND functional image (evaluated on a
    fixed grid), so structurally-distinct but functionally-equivalent expressions are also
    excluded."""

    n_variables: int
    allow_nan: bool
    holdout_X: np.ndarray = field(default_factory=lambda: _default_holdout_grid()[0].copy())
    holdout_C: np.ndarray = field(default_factory=lambda: _default_holdout_grid()[1].copy())
    extra_grids: tuple = field(default_factory=_default_extra_grids)
    extra_constant_fills: np.ndarray = field(default_factory=lambda: _default_constant_fills().copy())
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
                keys = self._evaluate_to_keys(compiled_fn, num_constants, n_variables)
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

        if all(key[1] == "__unevaluable__" for key in keys):
            # Fail-closed, but LOUD: the sentinel rejects every unevaluable candidate, yet
            # this law now has no discriminating image on any grid -- structure-only
            # protection plus blanket unevaluable rejection (the audit found the silent
            # variant of this state indistinguishable from full coverage).
            warnings.warn(
                f"holdout image for skeleton {skeleton_key!r} is the unevaluable sentinel "
                f"on every probe grid; only the exact-structure layer discriminates it",
                RuntimeWarning, stacklevel=2)
        self.expression_images.update(keys)

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
            keys = self._evaluate_to_keys(compiled_fn, num_constants, n_variables)
        return not keys.isdisjoint(self.expression_images)

    @staticmethod
    def _normalize_tokens(tokens: Sequence[str]) -> list[str]:
        # Canonicalize before hashing: variable renames (v1 -> x1) and numeric literals
        # (3.5 -> <constant>) must not defeat the exact-match layer; literal skeletons
        # otherwise leak through BOTH holdout layers (mirror of the flash-ansr fix).
        return list(normalize_skeleton([str(token) for token in tokens]))

    def _evaluate_to_keys(
        self,
        compiled_fn: Callable[..., np.ndarray | float],
        num_constants: int,
        n_variables: int | None = None,
    ) -> set:
        """Image keys over every probe grid (and, for parametric skeletons, every
        constant fill). One key per (grid, fill); registration stores them all and a
        probe matches on ANY -- strictly more conservative than a single key.

        Each image is standardized (mean 0, std 1, leading-sign canonical) before the
        4-dp rounding: a held-out law's whole output-affine family (a*f + b, either
        sign of a) shares one key, and large-magnitude laws stop failing the ABSOLUTE
        4-dp tolerance on f64 associativity noise (audit L6). A constant image
        standardizes to zeros -- every constant law shares one key, over-rejection in
        the ruled direction. All-NaN / unevaluable images key to a per-grid sentinel
        that matches only other unevaluable images: fail-closed, never a leak.
        """
        variable_count = n_variables if n_variables is not None else self.n_variables
        grids = (self.holdout_X,) + tuple(self.extra_grids)
        if num_constants == 0:
            fills: list[np.ndarray | None] = [None]
        else:
            fills = [self.holdout_C[:num_constants]]
            for row in self.extra_constant_fills:
                if len(row) >= num_constants:
                    fills.append(np.asarray(row[:num_constants]))

        keys: set = set()
        for grid_index, grid in enumerate(grids):
            samples = grid[:, :variable_count]
            for fill in fills:
                try:
                    image = safe_f(compiled_fn, samples, fill)
                    image = np.asarray(image, dtype=np.float64)
                except (TypeError, OverflowError, ValueError):
                    # A pathological integer-structure candidate: lambdify folds pure-int
                    # subtrees into arbitrary-precision Python ints, and numpy ufuncs
                    # refuse them. Such a candidate cannot match any real image; the
                    # sentinel matches only other unevaluable images. Conservative:
                    # over-rejects, never leaks.
                    keys.add((grid_index, "__unevaluable__"))
                    continue
                if np.isnan(image).all():
                    # safe_f signals a refused evaluation as an all-NaN vector; key it to
                    # the sentinel BEFORE the nan->0 fill, which would otherwise collide
                    # with a genuine zero image.
                    keys.add((grid_index, "__unevaluable__"))
                    continue
                # +-inf is masked like NaN: an inf-bearing mean/std is nonfinite and
                # previously dumped every overflowing draw into one shared "const"
                # bucket with every overflowing LAW (a collision class, measured as
                # most of a 7% rejection rate). The finite part of the image still
                # fingerprints the function; all-nonfinite images fall through to the
                # value-keyed constant branch as const-0.
                image = np.where(np.isfinite(image), image, 0.0)
                flat = image.reshape(-1)
                std = float(flat.std())
                mean = float(flat.mean())
                if not (np.isfinite(std) and std > 1e-12 and np.isfinite(mean)):
                    # A CONSTANT image keys by its VALUE, not by a shared canonical form:
                    # a single canonical key made every grid-saturating draw (tanh, atan,
                    # cosh chains flatten to a constant on the probe range) collide with
                    # any constant-imaged law -- measured 7.9% over-rejection, almost all
                    # from this one class. Value-keyed, only draws constant at the SAME
                    # value match: still conservative, no collision class.
                    keys.add((grid_index, "const",
                              round(mean, 4) if np.isfinite(mean) else "nonfinite"))
                    continue
                image = (image - mean) / std
                nonzero = flat[np.abs(flat - mean) > 1e-12 * max(std, 1.0)]
                if nonzero.size and nonzero[0] < mean:
                    image = -image
                image = np.round(image, 4)
                if image.ndim == 1:
                    keys.add((grid_index, tuple(float(v) for v in image.tolist())))
                else:
                    keys.add((grid_index, tuple(tuple(float(v) for v in row)
                                                for row in image.tolist())))
        return keys
