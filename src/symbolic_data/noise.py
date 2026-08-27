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
          p_instance: 0.10                    # orthogonal channel: instances contaminated at all
          rate: {name: beta, a: 1.0, b: 9.0}  # contamination fraction within the instance
          magnitude: {name: lognormal, median: 5.0, sigma: 1.4}   # kappa, in `scale` units
          scale: neighbour                    # what kappa is measured against
          sign: {mixed: 0.5, up: 0.25, down: 0.25}                # per problem
          min_count: 1                        # a contaminated instance carries >= 1 outlier

``rate`` and ``magnitude`` accept either a named family -- ``uniform``, ``loguniform``,
``beta`` (a, b), ``lognormal`` (median, sigma) -- or the legacy ``[lo, hi]`` pair, which
means Uniform for a rate and LogUniform for a magnitude. ``scale``, ``sign`` and
``min_count`` are optional and default to the pre-2026-08-27 behaviour, so a config
written before then still produces byte-identical data.

The ``scale`` choice is the consequential one. ``mad`` measures kappa against
``1.4826 * MAD(y)`` -- the spread of the SIGNAL -- which makes an outlier's difficulty an
accident of how much ``f`` varies on the sampled support (measured: a 60x swing across
problems, and a median displacement of roughly 3,000 residual sigma, where the robust
statistics literature places outliers at 1-25). ``neighbour`` measures it against what the
nearest neighbour in x fails to predict: it recovers the observation-noise scale when noise
dominates, and stays finite and meaningful on clean data, where it reports how much y moves
between adjacent points. Priors chosen 2026-08-26/27 from Kennedy et al. (2017) and
Oztuerk & Karabatsos (2017) for the rate, Hogg, Bovy & Lang (2010) for a log-scaled
magnitude, and Hoeting, Raftery & Madigan (1996) for the at-least-one convention.

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

__all__ = ["Distribution", "NoiseSpec", "apply_noise"]

_TYPE_NAMES = ("additive", "multiplicative")


#: The distribution families a noise prior may name. Each is (n_params, sampler).
_FAMILIES = {
    "uniform": ("lo", "hi"),
    "loguniform": ("lo", "hi"),
    "beta": ("a", "b"),
    "lognormal": ("median", "sigma"),
}


@dataclass(frozen=True)
class Distribution:
    """A named scalar prior. One internal representation; two config spellings."""

    name: str
    params: tuple[float, float]

    def draw(self, rng: np.random.Generator, size: Any = None) -> Any:
        a, b = self.params
        if self.name == "uniform":
            return rng.uniform(a, b, size=size)
        if self.name == "loguniform":
            return np.exp(rng.uniform(np.log(a), np.log(b), size=size))
        if self.name == "beta":
            return rng.beta(a, b, size=size)
        if self.name == "lognormal":
            return np.exp(np.log(a) + b * rng.standard_normal(size=size))
        raise AssertionError(f"unreachable distribution {self.name!r}")  # pragma: no cover


def _distribution(raw: Any, key: str, *, legacy: str) -> Distribution:
    """Parse a prior given either as ``{name: ..., ...}`` or as a legacy ``[lo, hi]`` pair.

    The pair form is what every config written before 2026-08-27 uses; it maps onto
    ``legacy`` (uniform for rates, log-uniform for magnitudes) so those configs keep
    producing byte-identical data. New configs should name the family explicitly.
    """
    if isinstance(raw, Mapping):
        if "name" not in raw:
            raise ValueError(f"noise.{key} mapping needs a 'name' (got {raw!r})")
        name = str(raw["name"])
        if name not in _FAMILIES:
            raise ValueError(f"noise.{key} unknown distribution {name!r}; "
                             f"expected one of {sorted(_FAMILIES)}")
        wanted = _FAMILIES[name]
        if set(raw) != {"name", *wanted}:
            raise ValueError(f"noise.{key} family {name!r} takes exactly {list(wanted)} "
                             f"(got {sorted(set(raw) - {'name'})})")
        values = tuple(float(raw[k]) for k in wanted)
        if not all(np.isfinite(v) for v in values):
            raise ValueError(f"noise.{key} parameters must be finite (got {raw!r})")
        if name in ("uniform", "loguniform"):
            lo, hi = values
            if lo > hi or (name == "loguniform" and lo <= 0.0):
                raise ValueError(f"noise.{key} needs {'0 < ' if name == 'loguniform' else ''}lo <= hi "
                                 f"(got {raw!r})")
        elif name == "beta":
            if values[0] <= 0.0 or values[1] <= 0.0:
                raise ValueError(f"noise.{key} beta needs a > 0 and b > 0 (got {raw!r})")
        elif name == "lognormal":
            if values[0] <= 0.0 or values[1] <= 0.0:
                raise ValueError(f"noise.{key} lognormal needs median > 0 and sigma > 0 (got {raw!r})")
        return Distribution(name=name, params=values)
    return Distribution(name=legacy, params=_bounds(raw, key, positive=True))


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
    outlier_rate: Distribution
    outlier_magnitude: Distribution
    #: 'mad' = kappa is in units of 1.4826*MAD(y), the spread of the SIGNAL (legacy).
    #: 'neighbour' = units of what the nearest neighbour in x fails to predict, which
    #: recovers the residual scale when noise dominates and stays finite when it does not.
    outlier_scale: str = "mad"
    #: (mixed, up, down) per PROBLEM. Legacy is fully mixed: an independent sign per point.
    outlier_sign: tuple[float, float, float] = (1.0, 0.0, 0.0)
    #: Floor on the realized outlier count once an instance is marked contaminated. 0 keeps
    #: the legacy Bernoulli draw (which frequently yields no outlier at all at small n);
    #: 1 conditions the count on being at least one.
    outlier_min_count: int = 0

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
        optional_outliers = {"scale", "sign", "min_count"}
        if not isinstance(outliers, Mapping) or not required_outliers <= set(outliers):
            raise ValueError(f"noise.outliers must carry at least {sorted(required_outliers)} (got {outliers!r})")
        unknown = set(outliers) - required_outliers - optional_outliers
        if unknown:
            raise ValueError(f"noise.outliers unknown keys {sorted(unknown)}; "
                             f"optional keys are {sorted(optional_outliers)}")
        scale_mode = str(outliers.get("scale", "mad"))
        if scale_mode not in ("mad", "neighbour"):
            raise ValueError(f"noise.outliers.scale must be 'mad' or 'neighbour' (got {scale_mode!r})")
        sign_raw = outliers.get("sign", {"mixed": 1.0})
        if not isinstance(sign_raw, Mapping) or not set(sign_raw) <= {"mixed", "up", "down"} or not sign_raw:
            raise ValueError(f"noise.outliers.sign must be a mapping over "
                             f"{{'mixed','up','down'}} (got {sign_raw!r})")
        sign_w = np.array([float(sign_raw.get(k, 0.0)) for k in ("mixed", "up", "down")], dtype=np.float64)
        if np.any(sign_w < 0.0) or not np.all(np.isfinite(sign_w)) or sign_w.sum() <= 0.0:
            raise ValueError(f"noise.outliers.sign weights must be non-negative and not all zero "
                             f"(got {sign_raw!r})")
        min_count = int(outliers.get("min_count", 0))
        if min_count < 0:
            raise ValueError(f"noise.outliers.min_count must be >= 0 (got {min_count})")
        return cls(
            p_clean=_probability(raw["p_clean"], "p_clean"),
            type_names=tuple(types),
            type_weights=tuple(weights / weights.sum()),
            level=_bounds(raw["level"], "level", positive=True),
            outlier_p_instance=_probability(outliers["p_instance"], "outliers.p_instance"),
            outlier_rate=_distribution(outliers["rate"], "outliers.rate", legacy="uniform"),
            outlier_magnitude=_distribution(outliers["magnitude"], "outliers.magnitude",
                                            legacy="loguniform"),
            outlier_scale=scale_mode,
            outlier_sign=tuple(sign_w / sign_w.sum()),
            outlier_min_count=min_count,
        )


def _log_uniform(rng: np.random.Generator, bounds: tuple[float, float], size: Any = None) -> Any:
    return np.exp(rng.uniform(np.log(bounds[0]), np.log(bounds[1]), size=size))


def _neighbour_scale(x: np.ndarray, y: np.ndarray) -> float:
    """Spread of nearest-neighbour differences in y: what the neighbours fail to predict.

    Measured on CLEAN y, before anything is injected. Nearest neighbour is taken in the
    full x-space (a KD-tree; measured at 0.45% of the per-instance generation budget on the
    median problem), so no ordering or projection is needed and it behaves identically in
    any dimension. The /sqrt(2) undoes the inflation from differencing two independent draws,
    so when observation noise dominates this recovers its standard deviation -- the scale the
    robust-statistics literature measures thresholds in. When the data is clean it does NOT
    collapse to zero; it reports how much y moves between adjacent points, which is exactly
    what limits detectability there.

    Returns 0.0 when the support is too small to have neighbours; the caller falls back.
    """
    x = np.atleast_2d(np.asarray(x, dtype=np.float64))
    y = np.asarray(y, dtype=np.float64).ravel()
    if y.size < 4 or x.shape[0] != y.size:
        return 0.0
    finite = np.isfinite(y) & np.isfinite(x).all(axis=1)
    x, y = x[finite], y[finite]
    if y.size < 4:
        return 0.0
    from scipy.spatial import cKDTree

    _, idx = cKDTree(x).query(x, k=2)
    d = y - y[idx[:, 1]]
    return 1.4826 * float(np.median(np.abs(d - np.median(d)))) / np.sqrt(2.0)


def _outlier_positions(
        n: int, rate: float, min_count: int, rng: np.random.Generator) -> np.ndarray:
    """Which points the outlier channel touches.

    The count is Binomial(n, rate), CONDITIONED on being at least ``min_count`` -- the
    conditioning is a redraw, not a clamp, so it renormalizes the distribution instead of
    piling every empty draw onto a spike at one. Positions are then uniform without
    replacement. Under the legacy ``min_count=0`` this is exactly the old per-point
    Bernoulli in distribution.
    """
    mask = np.zeros(n, dtype=bool)
    if n <= 0:
        return mask
    if min_count <= 0:
        # The original per-point Bernoulli, kept verbatim: same draws, same stream, so a
        # config written before the count was conditioned reproduces bit for bit.
        return rng.random(n) < rate
    k = int(rng.binomial(n, rate))
    if min_count > 0:
        floor = min(min_count, n)
        for _ in range(64):
            if k >= floor:
                break
            k = int(rng.binomial(n, rate))
        k = max(k, floor)
    if k <= 0:
        return mask
    mask[rng.choice(n, size=k, replace=False)] = True
    return mask



def apply_noise(
        spec: NoiseSpec,
        y_support: np.ndarray,
        y_validation: np.ndarray,
        rng: np.random.Generator,
        x_support: np.ndarray | None = None,
        x_validation: np.ndarray | None = None,
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
    # The outlier ruler. 'mad' measures the shove against the spread of the SIGNAL, which
    # makes its difficulty an accident of how much f happens to vary (measured: a 60x swing,
    # and a median displacement of ~3000 residual sigma -- far outside anything the
    # literature calls an outlier). 'neighbour' measures it against what the neighbouring
    # points fail to predict, so kappa means the same thing on every problem.
    outlier_scale = scale
    if spec.outlier_scale == "neighbour":
        nn = _neighbour_scale(x_support, ys64) if x_support is not None else 0.0
        # Too few points to have neighbours: fall back rather than skip the channel.
        outlier_scale = nn if nn > 0.0 else scale

    rate = 0.0
    sign_mode = "mixed"
    if outlier_scale > 0.0 and rng.random() < spec.outlier_p_instance:
        rate = float(np.clip(spec.outlier_rate.draw(rng), 0.0, 1.0))
        # ONE sign decision per problem, drawn here so the support and validation arrays
        # agree: a physical fault has a direction, and every real mechanism the literature
        # measured is one-sided. 'mixed' keeps the independent per-point sign.
        if spec.outlier_sign != (1.0, 0.0, 0.0):
            # Skipped entirely under the legacy all-mixed prior so its RNG stream is untouched.
            sign_mode = ("mixed", "up", "down")[int(rng.choice(3, p=spec.outlier_sign))]

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
            flat = _outlier_positions(noisy.size, rate, spec.outlier_min_count, rng)
            mask = flat.reshape(noisy.shape)
            kappa = np.asarray(spec.outlier_magnitude.draw(rng, size=noisy.shape), dtype=np.float64)
            if sign_mode == "up":
                sign = np.ones(noisy.shape)
            elif sign_mode == "down":
                sign = -np.ones(noisy.shape)
            else:
                sign = np.where(rng.random(noisy.shape) < 0.5, -1.0, 1.0)
            noisy = np.where(mask, noisy + sign * kappa * outlier_scale, noisy)
        with warnings.catch_warnings():
            # the cast itself flags the overflow we are about to test for
            warnings.simplefilter("ignore", RuntimeWarning)
            noisy32 = noisy.astype(np.float32)
        if noisy32.size and not np.all(np.isfinite(noisy32)):
            return None
        arrays.append(noisy32)
        masks.append(mask)

    draw = {"type": kind, "level": level, "outlier_rate": rate, "scale": scale,
            "outlier_scale": outlier_scale, "outlier_sign": sign_mode}
    return arrays[0], arrays[1], masks[0], masks[1], draw
