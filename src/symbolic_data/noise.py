"""The measurement-noise mixture: a post-accept capability of every source.

One noise definition shared by training and evaluation: the mixture lives here, next to
the sampling it augments, so a training pipeline and an evaluation harness can never
drift apart on what a noise level means. It is applied strictly AFTER support
acceptance -- the expression prior is shaped by rejection sampling over CLEAN values
only, and noise never feeds back into acceptance. The one deliberate exception: a
noised instance whose float32 cast goes non-finite is rejected outright (measured on
the v24 training prior: 5e-5 of instance-draws, confined to targets already within
~1.3x of the float32 boundary -- the prior perturbation is bounded by that rate).

Config form (``sampling.noise`` as a mapping; every key is REQUIRED -- priors are
pinned explicitly, never defaulted):

.. code-block:: yaml

    sampling:
      noise:
        p_clean: 0.30                  # exact point mass at zero noise
        types: {additive: 0.5, multiplicative: 0.5}
        level: [1.0e-4, 0.3]           # lambda ~ LogUniform(lo, hi)
        outliers:
          p_instance: 0.10             # orthogonal channel: instances contaminated at all
          rate: [0.005, 0.1]           # per-point contamination rate r ~ Uniform(lo, hi)
          magnitude: [3.0, 100.0]      # kappa ~ LogUniform(lo, hi), in robust-scale units

Per-instance semantics: ONE ``(type, lambda)`` draw applies to the support and
validation targets alike. Additive noise is ``N(0, (lambda*s)^2)`` with
``s = 1.4826 * MAD`` of the clean support+validation targets -- the robust scale keeps
contamination relative to the typical spread, not the extremes. Multiplicative noise is
``y * (1 + eps)``, ``eps ~ N(0, lambda^2)``. The outlier channel adds
``sign * kappa * s`` at Bernoulli(r) points on top of either (or of a clean draw), and
is skipped when ``s == 0`` -- a constant target has no spread to deviate from. All
arithmetic runs in float64 and casts to float32 at the end. The scalar ``noise:``
config form keeps its legacy semantics (additive, std-scaled) untouched in
:mod:`symbolic_data.source`.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

__all__ = ["NoiseSpec", "apply_noise"]

_TYPE_NAMES = ("additive", "multiplicative")


def _bounds(raw: Any, key: str, *, positive: bool) -> tuple[float, float]:
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        raise ValueError(f"noise.{key} must be a [lo, hi] pair (got {raw!r})")
    lo, hi = float(raw[0]), float(raw[1])
    if not (np.isfinite(lo) and np.isfinite(hi)) or lo > hi or (positive and lo <= 0.0):
        raise ValueError(f"noise.{key} needs finite {'0 < ' if positive else ''}lo <= hi (got {raw!r})")
    return lo, hi


def _probability(raw: Any, key: str) -> float:
    p = float(raw)
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"noise.{key} must be a probability in [0, 1] (got {raw!r})")
    return p


@dataclass(frozen=True)
class NoiseSpec:
    """The parsed mixture prior (see the module docstring for the config form)."""

    p_clean: float
    type_names: tuple[str, ...]
    type_weights: tuple[float, ...]
    level: tuple[float, float]
    outlier_p_instance: float
    outlier_rate: tuple[float, float]
    outlier_magnitude: tuple[float, float]

    @classmethod
    def parse(cls, raw: Mapping[str, Any]) -> "NoiseSpec":
        """Validate a ``sampling.noise`` mapping; every key required, unknown keys refused."""
        required = {"p_clean", "types", "level", "outliers"}
        if set(raw) != required:
            missing, unknown = required - set(raw), set(raw) - required
            raise ValueError(f"noise spec: missing keys {sorted(missing)}, unknown keys {sorted(unknown)} "
                             f"(all of {sorted(required)} are required; priors are pinned, never defaulted)")
        types = raw["types"]
        if not isinstance(types, Mapping) or not types or not set(types) <= set(_TYPE_NAMES):
            raise ValueError(f"noise.types must be a non-empty mapping over {_TYPE_NAMES} (got {types!r})")
        weights = np.array([float(types[name]) for name in types], dtype=np.float64)
        if np.any(weights <= 0.0) or not np.all(np.isfinite(weights)):
            raise ValueError(f"noise.types weights must be positive (got {types!r})")
        outliers = raw["outliers"]
        required_outliers = {"p_instance", "rate", "magnitude"}
        if not isinstance(outliers, Mapping) or set(outliers) != required_outliers:
            raise ValueError(f"noise.outliers must carry exactly {sorted(required_outliers)} (got {outliers!r})")
        return cls(
            p_clean=_probability(raw["p_clean"], "p_clean"),
            type_names=tuple(types),
            type_weights=tuple(weights / weights.sum()),
            level=_bounds(raw["level"], "level", positive=True),
            outlier_p_instance=_probability(outliers["p_instance"], "outliers.p_instance"),
            outlier_rate=_bounds(outliers["rate"], "outliers.rate", positive=True),
            outlier_magnitude=_bounds(outliers["magnitude"], "outliers.magnitude", positive=True),
        )


def _log_uniform(rng: np.random.Generator, bounds: tuple[float, float], size: Any = None) -> Any:
    return np.exp(rng.uniform(np.log(bounds[0]), np.log(bounds[1]), size=size))


def apply_noise(
        spec: NoiseSpec,
        y_support: np.ndarray,
        y_validation: np.ndarray,
        rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]] | None:
    """One per-instance draw of the mixture, applied to both target arrays.

    Returns ``(y_support_noisy, y_validation_noisy, outlier_mask_support,
    outlier_mask_validation, draw)`` -- float32 noisy targets, bool masks marking
    exactly the points the outlier channel touched (the generative label, not a
    statistical judgment), and the realized-noise provenance ``draw`` stored on the
    Problem. Returns ``None`` when the float32 cast of a noised array goes non-finite:
    the caller rejects the trial.
    """
    ys64 = np.asarray(y_support, dtype=np.float64)
    yv64 = np.asarray(y_validation, dtype=np.float64)
    clean = np.concatenate([ys64.ravel(), yv64.ravel()])
    scale = 1.4826 * float(np.median(np.abs(clean - np.median(clean)))) if clean.size else 0.0

    if rng.random() < spec.p_clean:
        kind, level = "clean", 0.0
    else:
        kind = spec.type_names[int(rng.choice(len(spec.type_names), p=spec.type_weights))]
        level = float(_log_uniform(rng, spec.level))
    rate = 0.0
    if scale > 0.0 and rng.random() < spec.outlier_p_instance:
        rate = float(rng.uniform(*spec.outlier_rate))

    arrays: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    for y64 in (ys64, yv64):
        noisy = y64.copy()
        if kind == "additive":
            noisy = noisy + rng.normal(0.0, level * scale, size=noisy.shape)
        elif kind == "multiplicative":
            noisy = noisy * (1.0 + rng.normal(0.0, level, size=noisy.shape))
        mask = np.zeros(noisy.shape, dtype=bool)
        if rate > 0.0 and noisy.size:
            mask = rng.random(noisy.shape) < rate
            kappa = _log_uniform(rng, spec.outlier_magnitude, size=noisy.shape)
            sign = np.where(rng.random(noisy.shape) < 0.5, -1.0, 1.0)
            noisy = np.where(mask, noisy + sign * kappa * scale, noisy)
        with warnings.catch_warnings():
            # the cast itself flags the overflow we are about to test for
            warnings.simplefilter("ignore", RuntimeWarning)
            noisy32 = noisy.astype(np.float32)
        if noisy32.size and not np.all(np.isfinite(noisy32)):
            return None
        arrays.append(noisy32)
        masks.append(mask)

    draw = {"type": kind, "level": level, "outlier_rate": rate, "scale": scale}
    return arrays[0], arrays[1], masks[0], masks[1], draw
