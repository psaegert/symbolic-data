"""Distribution factories used to sample numerical constants and support points.

Every sampler takes an optional ``rng`` (a :class:`numpy.random.Generator`). Reproducibility is
NEVER obtained from a fixed seed (seeding is bad practice in this project); a caller threads a
single ``Generator`` through a sampling session, and exact reproduction comes from materializing
+ freezing the sampled data, not from re-seeding. When ``rng`` is omitted a process-wide
entropy ``Generator`` is used.

Naming convention: ``log_uniform`` is **natural-log** based (matches ``math``/``numpy``). The
FastSRB benchmark's ``log`` is **base-10**; that semantics is quarantined inside the ``fastsrb``
distribution (:func:`fastsrb_dist`) so it stays faithful to the published benchmark without
contaminating the native ``log_uniform``. See the package README conventions section.
"""
import math
import warnings
from functools import partial
from typing import Any, Callable

import numpy as np

from symbolic_data.registry import Registry

# Process-wide default Generator (entropy-seeded once). Used only when a caller does not thread
# its own Generator; it is NOT a reproducibility mechanism.
_DEFAULT_RNG = np.random.default_rng()


def _resolve_rng(rng: np.random.Generator | None) -> np.random.Generator:
    return rng if rng is not None else _DEFAULT_RNG


def uniform_dist(
    low: float,
    high: float,
    min_value: float | None = None,
    max_value: float | None = None,
    size: Any = 1,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Sample uniformly from ``[low, high]`` with optional clipping."""
    low, high = min(low, high), max(low, high)
    samples = _resolve_rng(rng).uniform(low, high, size=size)
    if min_value is not None and max_value is not None:
        return np.clip(samples, min_value, max_value)
    return samples


def normal_dist(
    loc: float,
    scale: float,
    min_value: float | None = None,
    max_value: float | None = None,
    size: Any = 1,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Sample from a normal distribution with optional clipping."""
    scale = max(scale, 1e-9)
    samples = _resolve_rng(rng).normal(loc, scale, size=size)
    if min_value is not None and max_value is not None:
        return np.clip(samples, min_value, max_value)
    return samples


def log_uniform_dist(
    low: float,
    high: float,
    min_value: float | None = None,
    max_value: float | None = None,
    size: Any = 1,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Sample from a (natural-log) log-uniform distribution with optional clipping."""
    low, high = min(low, high), max(low, high)
    samples = np.exp(_resolve_rng(rng).uniform(np.log(low), np.log(high), size=size))
    if min_value is not None and max_value is not None:
        return np.clip(samples, min_value, max_value)
    return samples


def log_normal_dist(
    mean: float,
    sigma: float,
    min_value: float | None = None,
    max_value: float | None = None,
    size: Any = 1,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Sample from a log-normal distribution with optional clipping."""
    sigma = max(sigma, 1e-9)
    samples = _resolve_rng(rng).lognormal(mean, sigma, size=size)
    if min_value is not None and max_value is not None:
        return np.clip(samples, min_value, max_value)
    return samples


def gamma_dist(
    shape: float,
    scale: float,
    min_value: float | None = None,
    max_value: float | None = None,
    size: Any = 1,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Sample from a gamma distribution with optional clipping."""
    samples = _resolve_rng(rng).gamma(shape, scale, size=size)
    if min_value is not None and max_value is not None:
        return np.clip(samples, min_value, max_value)
    return samples


def cauchy_dist(
    loc: float = 0.0,
    scale: float = 1.0,
    min_value: float | None = None,
    max_value: float | None = None,
    size: Any = 1,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Sample from a Cauchy distribution with optional clipping."""
    scale = max(scale, 1e-9)
    samples = loc + scale * _resolve_rng(rng).standard_cauchy(size=size)
    if min_value is not None and max_value is not None:
        return np.clip(samples, min_value, max_value)
    return samples


def binomial_dist(
    n: int,
    p: float,
    min_value: float | None = None,
    max_value: float | None = None,
    size: Any = 1,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Sample from a binomial distribution with optional clipping."""
    samples = _resolve_rng(rng).binomial(int(n), float(p), size=size)
    if min_value is not None and max_value is not None:
        return np.clip(samples, min_value, max_value)
    return samples


def fastsrb_dist(
    low: float,
    high: float,
    base: str = "uni",
    sign: str = "pos",
    layout: str = "random",
    min_value: float | None = None,
    max_value: float | None = None,
    size: Any = 1,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """FastSRB-faithful variable sampling, exposed as one nestable distribution.

    Reproduces the published FastSRB recipe exactly:

    * ``base``: ``"uni"`` (uniform), ``"log"`` (**base-10** log-uniform -- this is the published
      FastSRB recipe's internal representation; note log-uniform is base-invariant, so it yields
      values identical to the native natural-log ``log_uniform`` for the same Generator), or
      ``"int"`` (uniform then rounded).
    * ``layout``: ``"random"`` (i.i.d. draws) or ``"grid"`` (``linspace`` then shuffle).
    * ``sign``: ``"pos"`` (as-is), ``"neg"`` (``-|x|``), or ``"pos_neg"`` (random sign),
      applied AFTER the base-10 exponentiation.

    ``low == high`` yields a constant fill. ``log`` requires strictly-positive bounds.
    """
    generator = _resolve_rng(rng)
    n = int(size) if np.isscalar(size) else int(np.prod(size))

    distribution = base
    integer = False
    if base == "int":
        distribution = "uni"
        integer = True
    if distribution not in {"uni", "log"}:
        raise ValueError("base must be 'uni', 'log', or 'int'")
    if sign not in {"pos", "neg", "pos_neg"}:
        raise ValueError("sign must be 'pos', 'neg', or 'pos_neg'")
    if layout not in {"random", "grid"}:
        raise ValueError("layout must be 'random' or 'grid'")
    if n < 1:
        raise ValueError("size must be at least 1")
    if layout == "grid" and n == 1:
        warnings.warn(
            "Sampling one point with layout='grid' is degenerate; consider layout='random'",
            RuntimeWarning,
            stacklevel=2,
        )

    low_f = float(low)
    high_f = float(high)
    if low_f > high_f:
        raise ValueError("low must not exceed high")

    if math.isclose(low_f, high_f):
        arr = np.full(n, high_f, dtype=float)
    else:
        if distribution == "log":
            if low_f <= 0 or high_f <= 0:
                raise ValueError("log sampling requires strictly positive bounds")
            low_val = math.log10(low_f)
            high_val = math.log10(high_f)
        else:
            low_val = low_f
            high_val = high_f
        if layout == "random":
            arr = generator.uniform(low_val, high_val, size=n)
        else:
            arr = np.linspace(low_val, high_val, n)
            generator.shuffle(arr)
        if distribution == "log":
            arr = 10.0 ** arr

    if sign == "neg":
        arr = -np.abs(arr)
    elif sign == "pos_neg":
        arr = arr * generator.choice([-1.0, 1.0], size=arr.shape)
    if integer:
        arr = np.rint(arr)

    arr = arr.astype(float, copy=False)
    if min_value is not None and max_value is not None:
        arr = np.clip(arr, min_value, max_value)
    return arr


BASE_DISTRIBUTIONS: dict[str, Callable[..., np.ndarray]] = {
    "uniform": uniform_dist,
    "normal": normal_dist,
    "log_uniform": log_uniform_dist,
    "log_normal": log_normal_dist,
    "gamma": gamma_dist,
    "cauchy": cauchy_dist,
    "binomial": binomial_dist,
    "fastsrb": fastsrb_dist,
}

# Extensible registry of distribution samplers, seeded from the builtins above. Custom
# distributions can be added in-process (``@DISTRIBUTIONS.register``) or across packages via
# ``symbolic_data.distributions`` entry points; either way a registered name drops into the same
# ``{"name": ..., "kwargs": ...}`` config slot as a builtin. ``BASE_DISTRIBUTIONS`` remains the
# source of truth for the builtins.
DISTRIBUTIONS = Registry("distribution", entry_point_group="symbolic_data.distributions")
for _name, _fn in BASE_DISTRIBUTIONS.items():
    DISTRIBUTIONS.register_builtin(_name, _fn)


def sampler_dist(
    base_dist_name: str,
    param_samplers: dict[str, Callable[..., np.ndarray]],
    base_kwargs: dict[str, Any] | None = None,
    size: Any = 1,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Sample from ``base_dist_name`` after drawing its parameters from ``param_samplers``."""
    if base_dist_name not in DISTRIBUTIONS:
        raise ValueError(f"Unknown base_dist_name: {base_dist_name}")

    generator = _resolve_rng(rng)
    final_kwargs = base_kwargs.copy() if base_kwargs else {}
    for param_name, sampler_func in param_samplers.items():
        final_kwargs[param_name] = sampler_func(size=1, rng=generator)[0]  # type: ignore[index]

    base_dist_func = DISTRIBUTIONS.get(base_dist_name)
    return base_dist_func(**final_kwargs, size=size, rng=generator)


def get_distribution(config: dict[str, Any]) -> Callable[..., np.ndarray]:
    """Create a distribution callable ``(size=1, rng=None) -> ndarray`` from ``config``.

    Supports builtins/registered names, the ``constant`` special form, and the ``sampler``
    nesting form (a distribution whose parameters are themselves sampled).
    """
    name = config["name"]
    kwargs = config.get("kwargs", {})

    if name == "constant":
        return lambda size=1, rng=None: np.full(size, kwargs["value"])

    if name in DISTRIBUTIONS:
        return partial(DISTRIBUTIONS.get(name), **kwargs)

    if name == "sampler":
        resolved_samplers = {
            param_name: get_distribution(sampler_config)
            for param_name, sampler_config in kwargs["param_samplers"].items()
        }
        sampler_args = {
            "base_dist_name": kwargs["base_dist_name"],
            "param_samplers": resolved_samplers,
            "base_kwargs": kwargs.get("base_kwargs", {}),
        }
        return partial(sampler_dist, **sampler_args)

    raise ValueError(f"Unknown distribution name: {name}")
