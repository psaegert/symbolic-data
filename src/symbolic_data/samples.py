"""Model-agnostic (X, y) sample generation -- the seam SR methods and eval delegate to.

A :class:`Sample` is one symbolic-regression problem: a ground-truth skeleton plus its
realized constants and sampled support/validation data (optionally noised).

* :func:`sample_from_skeleton` is the per-skeleton core: ``pool.sample_data`` -> split into
  support/validation -> additive noise -> (optional) unused-variable masking -> GT metadata.
* :func:`iter_samples` is the convenience loop over a pool's (or an explicit) skeleton set.

The model-coupling parts (tokenization, prompt serialization, eval bookkeeping) stay with the
consumer; a consumer like srbf wraps each ``Sample`` in its own eval record. The split and
noise semantics match flash-ansr/srbf exactly, so a consumer can delegate without changing
the data it produces.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Sequence

import numpy as np
from simplipy import normalize_expression, normalize_skeleton
from simplipy.utils import substitude_constants

from symbolic_data.skeleton_pool import NoValidSampleFoundError, SkeletonPool
from symbolic_data.tensor_ops import mask_unused_variable_columns

__all__ = ["Sample", "sample_from_skeleton", "iter_samples"]


@dataclass
class Sample:
    """One model-agnostic symbolic-regression problem (skeleton + sampled data)."""

    skeleton: tuple[str, ...]
    expression: list[str] | None       # GT tokens with constants substituted + normalized
    constants: list[float]             # realized constant literals
    variables: list[str]               # pool variable names; column order of the X arrays
    n_variables_used: int              # distinct pool variables appearing in the skeleton
    x_support: np.ndarray
    y_support: np.ndarray
    x_validation: np.ndarray
    y_validation: np.ndarray
    y_support_noisy: np.ndarray
    y_validation_noisy: np.ndarray
    noise_level: float
    complexity: int | None             # token length of the substituted expression
    placeholder: bool = False
    placeholder_reason: str | None = None


def _split_support_and_validation(
    x_all: np.ndarray,
    y_all: np.ndarray,
    n_support: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split the sampled points into a support set (first ``n_support``) and the rest.

    Mirrors the flash-ansr/srbf convention: ``sample_data`` is asked for ``2*n_support``
    points; the first ``n_support`` are support, the remainder validation. Falls back to a
    50/50 split when ``n_support`` is unset or would consume every point.
    """
    total_points = x_all.shape[0]
    n_dim_x = x_all.shape[1]
    n_dim_y = y_all.shape[1]
    if total_points == 0:
        empty_x = np.empty((0, n_dim_x), dtype=np.float32)
        empty_y = np.empty((0, n_dim_y), dtype=np.float32)
        return empty_x, empty_x.copy(), empty_y, empty_y.copy()

    support_count = n_support if n_support is not None else total_points // 2
    support_count = max(1, min(support_count, total_points))
    if support_count == total_points and total_points > 1:
        support_count = total_points // 2

    x_support = x_all[:support_count].astype(np.float32, copy=True)
    y_support = y_all[:support_count].astype(np.float32, copy=True)

    if support_count < total_points:
        x_val = x_all[support_count:].astype(np.float32, copy=True)
        y_val = y_all[support_count:].astype(np.float32, copy=True)
    else:
        x_val = np.empty((0, n_dim_x), dtype=np.float32)
        y_val = np.empty((0, n_dim_y), dtype=np.float32)

    return x_support, x_val, y_support, y_val


def _inject_noise(array: np.ndarray, noise_level: float, rng: np.random.Generator) -> np.ndarray:
    """Add Gaussian noise scaled by ``noise_level * std(array)`` (no-op on empty/constant)."""
    if array.size == 0:
        return array.copy()
    noisy = array.copy()
    y_std = float(np.std(noisy))
    if np.isfinite(y_std) and y_std > 0:
        noise = rng.standard_normal(size=noisy.shape).astype(np.float32)
        noisy = noisy + (noise_level * y_std * noise)
    return noisy


def _gt_metadata(
    skeleton: Sequence[str],
    literals: np.ndarray,
) -> tuple[list[str] | None, int | None]:
    """Normalized GT expression (constants substituted) + its token-length complexity."""
    skeleton_list = normalize_skeleton(skeleton)
    if skeleton_list is None:
        return None, None
    expression_tokens = substitude_constants(list(skeleton_list), values=literals, inplace=False)
    expression = normalize_expression(expression_tokens)
    complexity = len(expression_tokens) if expression_tokens else None
    return expression, complexity


def sample_from_skeleton(
    pool: SkeletonPool,
    skeleton: Sequence[str],
    *,
    n_support: int | None = None,
    noise_level: float = 0.0,
    mask_unused_variables: bool = False,
    rng: np.random.Generator | None = None,
    max_trials: int = 8,
) -> Sample | None:
    """Generate one :class:`Sample` for ``skeleton``, or ``None`` if sampling keeps failing.

    Retries up to ``max_trials`` times on :class:`NoValidSampleFoundError` / empty draws,
    matching the consumer's per-skeleton retry budget.
    """
    skeleton = tuple(skeleton)
    if not pool.skeleton_codes:
        pool.skeleton_codes = pool.compile_codes(verbose=False)
    if skeleton not in pool.skeleton_codes:
        pool.skeleton_codes = pool.compile_codes(verbose=False)
        if skeleton not in pool.skeleton_codes:
            return None

    code, constants_tokens = pool.skeleton_codes[skeleton]
    n_constants = len(constants_tokens)
    n_points = n_support * 2 if n_support is not None else None
    if rng is None:
        rng = np.random.default_rng()

    variable_set = set(pool.variables)
    n_variables_used = len({token for token in skeleton if token in variable_set})

    for _ in range(max_trials):
        try:
            x_all, y_all, literals = pool.sample_data(code, n_constants, n_support=n_points, rng=rng)
        except NoValidSampleFoundError:
            continue
        if x_all.size == 0 or y_all.size == 0:
            continue

        x_support, x_val, y_support, y_val = _split_support_and_validation(x_all, y_all, n_support)
        if x_support.size == 0:
            continue

        if noise_level > 0.0:
            y_support_noisy = _inject_noise(y_support, noise_level, rng)
            y_val_noisy = _inject_noise(y_val, noise_level, rng)
        else:
            y_support_noisy = y_support.copy()
            y_val_noisy = y_val.copy()

        if mask_unused_variables:
            mask_unused_variable_columns(
                (x_support, x_val), variables=pool.variables, skeleton_tokens=skeleton
            )

        expression, complexity = _gt_metadata(skeleton, literals)

        return Sample(
            skeleton=skeleton,
            expression=expression,
            constants=list(np.asarray(literals, dtype=np.float64).ravel().tolist()),
            variables=list(pool.variables),
            n_variables_used=n_variables_used,
            x_support=x_support,
            y_support=y_support,
            x_validation=x_val,
            y_validation=y_val,
            y_support_noisy=y_support_noisy,
            y_validation_noisy=y_val_noisy,
            noise_level=float(noise_level),
            complexity=complexity,
        )

    return None


def _placeholder_sample(pool: SkeletonPool, skeleton: Sequence[str], reason: str) -> Sample:
    empty_x = np.empty((0, pool.n_variables), dtype=np.float32)
    empty_y = np.empty((0, 1), dtype=np.float32)
    return Sample(
        skeleton=tuple(skeleton),
        expression=None,
        constants=[],
        variables=list(pool.variables),
        n_variables_used=0,
        x_support=empty_x,
        y_support=empty_y,
        x_validation=empty_x.copy(),
        y_validation=empty_y.copy(),
        y_support_noisy=empty_y.copy(),
        y_validation_noisy=empty_y.copy(),
        noise_level=0.0,
        complexity=None,
        placeholder=True,
        placeholder_reason=reason,
    )


def iter_samples(
    pool: SkeletonPool,
    *,
    n_support: int | None = None,
    noise_level: float = 0.0,
    mask_unused_variables: bool = False,
    datasets_per_expression: int = 1,
    skeletons: Sequence[Sequence[str]] | None = None,
    rng: np.random.Generator | None = None,
    max_trials: int = 8,
    skip_failed: bool = True,
) -> Iterator[Sample]:
    """Yield :class:`Sample` objects for each skeleton in ``pool`` (or an explicit list).

    One shared :class:`numpy.random.Generator` (seeded by ``seed``) drives the noise across
    the whole iteration, so a fixed ``seed`` gives a reproducible stream. ``skeletons``
    defaults to ``sorted(pool.skeletons)`` (deterministic order). When ``skip_failed`` is
    True, skeletons that exhaust ``max_trials`` are dropped; otherwise a placeholder
    ``Sample`` (``placeholder=True``) is yielded so callers can account for every skeleton.
    """
    if not pool.skeleton_codes:
        pool.skeleton_codes = pool.compile_codes(verbose=False)
    if skeletons is None:
        skeletons = sorted(pool.skeletons)

    rng = rng if rng is not None else np.random.default_rng()

    for skeleton in skeletons:
        for _ in range(datasets_per_expression):
            sample = sample_from_skeleton(
                pool,
                skeleton,
                n_support=n_support,
                noise_level=noise_level,
                mask_unused_variables=mask_unused_variables,
                rng=rng,
                max_trials=max_trials,
            )
            if sample is None:
                if skip_failed:
                    continue
                sample = _placeholder_sample(pool, skeleton, reason="max_trials_exhausted")
            yield sample
