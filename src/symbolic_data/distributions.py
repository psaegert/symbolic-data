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
from decimal import Decimal
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
    """Sample uniformly from ``[low, high]`` with optional clipping.

    ``low``/``high`` may be arrays (one bound pair per column, broadcast against a 2-D
    ``size``): the per-column-params path of :func:`sampler_dist` relies on it.
    """
    low, high = np.minimum(low, high), np.maximum(low, high)
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
    """Sample from a normal distribution with optional clipping.

    ``loc``/``scale`` may be arrays (one pair per column, broadcast against a 2-D
    ``size``): the per-column-params path of :func:`sampler_dist` relies on it.
    """
    scale = np.maximum(scale, 1e-9)
    samples = _resolve_rng(rng).normal(loc, scale, size=size)
    if min_value is not None and max_value is not None:
        return np.clip(samples, min_value, max_value)
    return samples


def choice_dist(
    values: list[Any],
    weights: list[float] | None = None,
    size: Any = 1,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Sample from an explicit weighted set of values.

    The categorical distribution the numeric-literal prior needs: an operator's exponent
    slot draws from a small INTEGER support (``pow``, ``rootn``) rather than a continuum,
    and a training vocabulary is an explicit set. Weights need not be normalized.

    Returns ``float64`` with the caller's ``size`` shape, like every other builtin -- the
    values are indexed into a float array rather than collected in a list comprehension, so
    a tuple ``size`` works and the input dtype does not leak (an integer or string
    ``values`` list used to come back as ``int64`` / ``<U1``).
    """
    if not values:
        raise ValueError("choice distribution needs a non-empty 'values' list")
    probabilities = None
    if weights is not None:
        if len(weights) != len(values):
            raise ValueError(
                f"choice distribution: {len(weights)} weights for {len(values)} values")
        probability_array = np.asarray(weights, dtype=np.float64)
        # NaN passes both tests below (comparisons and sum are False/NaN), so it has to be
        # rejected explicitly or it only fails later, inside numpy's own sampler.
        if not np.isfinite(probability_array).all():
            raise ValueError("choice distribution weights must all be finite")
        if (probability_array < 0).any() or probability_array.sum() <= 0:
            raise ValueError("choice distribution weights must be non-negative and sum to > 0")
        probabilities = probability_array / probability_array.sum()
    value_array = np.asarray(values, dtype=np.float64)
    indices = _resolve_rng(rng).choice(len(values), size=size, p=probabilities)
    return value_array[indices]


def rounded_dist(
    base: Callable[..., np.ndarray] | dict[str, Any],
    precision: Callable[..., np.ndarray] | dict[str, Any] | None = None,
    size: Any = 1,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Draw from ``base`` and round each draw to a sampled number of decimals.

    Precision is a first-class part of a numeric-literal prior, not cosmetics: a
    description-length measure prices a literal by the bit length of its EXACT value, so
    an unrounded float64 draw denotes a rational with a ~17-digit numerator and costs an
    order of magnitude more than a structural node. Sampling the precision spreads
    literal cost across the corpus instead of pinning it at one magnitude.

    ``precision`` omitted means uniform over ``1..d`` where ``d`` is the draw's OWN full
    precision (the decimals in its shortest round-trip repr), so the full range from a
    coarse literal to the raw draw is covered without an arbitrary cap. Precision 0 is
    never used: it would collapse the float branch onto the integer support.
    """
    generator = _resolve_rng(rng)
    base_callable = get_distribution(base) if isinstance(base, dict) else base
    draws = np.asarray(base_callable(size=size, rng=generator), dtype=np.float64)
    shape = draws.shape
    flat = draws.ravel()

    precision_callable = None
    if precision is not None:
        precision_callable = get_distribution(precision) if isinstance(precision, dict) else precision

    out = np.empty(flat.size, dtype=np.float64)
    for index, value in enumerate(flat):
        value = float(value)
        # A non-finite draw has no decimal expansion: `Decimal(repr(inf)).as_tuple().exponent`
        # is the STRING 'F' (or 'n' for nan), so negating it raises. Pass it through.
        if not np.isfinite(value):
            out[index] = value
            continue
        if precision_callable is not None:
            decimals = max(
                int(round(float(np.atleast_1d(precision_callable(size=1, rng=generator))[0]))), 1)
        else:
            full = -Decimal(repr(value)).as_tuple().exponent  # type: ignore[operator]
            decimals = int(generator.integers(1, max(int(full), 1) + 1))
        # `+ 0.0` normalizes -0.0 to 0.0: a rounded small negative draw otherwise reaches
        # a consumer as negative zero, which prints as "-0.0" and denotes nothing useful.
        out[index] = round(value, decimals) + 0.0
    return out.reshape(shape)


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
    "choice": choice_dist,
    "rounded": rounded_dist,
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

# Bases whose parameters may be arrays (one value per column of a 2-D ``size``); the
# per-column-params path of ``sampler_dist`` is limited to these.
VECTOR_PARAM_BASES = frozenset({"uniform", "normal"})

# Builtins whose entries are elementwise-iid for FIXED kwargs: one ``(n, k)`` draw is
# distributionally identical to ``k`` separate ``(n, 1)`` draws. ``fastsrb`` is excluded
# (its 'grid' layout correlates the entries of a single call).
ELEMENTWISE_IID_BASES: frozenset[Callable[..., np.ndarray]] = frozenset({
    uniform_dist, normal_dist, choice_dist, rounded_dist, log_uniform_dist,
    log_normal_dist, gamma_dist, cauchy_dist, binomial_dist,
})


def sampler_dist(
    base_dist_name: str,
    param_samplers: dict[str, Callable[..., np.ndarray]],
    base_kwargs: dict[str, Any] | None = None,
    size: Any = 1,
    rng: np.random.Generator | None = None,
    per_column_params: bool = False,
) -> np.ndarray:
    """Sample from ``base_dist_name`` after drawing its parameters from ``param_samplers``.

    ``per_column_params`` (requires a 2-D ``size=(n_rows, n_columns)`` and a base in
    ``VECTOR_PARAM_BASES``): draw one parameter set PER COLUMN, vectorized, so ``k``
    independent dimensions cost one base draw instead of ``k``. Distributionally identical
    to ``k`` separate ``size=(n_rows, 1)`` calls -- each column still gets its own
    independent parameter draw -- but the rng stream differs from the per-column loop.
    """
    if base_dist_name not in DISTRIBUTIONS:
        raise ValueError(f"Unknown base_dist_name: {base_dist_name}")

    generator = _resolve_rng(rng)
    final_kwargs = base_kwargs.copy() if base_kwargs else {}
    if per_column_params:
        if not (isinstance(size, tuple) and len(size) == 2):
            raise ValueError("per_column_params requires size=(n_rows, n_columns)")
        if base_dist_name not in VECTOR_PARAM_BASES:
            raise ValueError(
                f"per_column_params supports bases {sorted(VECTOR_PARAM_BASES)}; got {base_dist_name!r}")
        for param_name, sampler_func in param_samplers.items():
            final_kwargs[param_name] = np.asarray(
                sampler_func(size=size[1], rng=generator), dtype=np.float64)
    else:
        for param_name, sampler_func in param_samplers.items():
            final_kwargs[param_name] = sampler_func(size=1, rng=generator)[0]  # type: ignore[index]

    base_dist_func = DISTRIBUTIONS.get(base_dist_name)
    return base_dist_func(**final_kwargs, size=size, rng=generator)


def sampler_box_batch(
    prior: Callable[..., np.ndarray],
    n_cols: int,
    n_boxes: int,
    rng: np.random.Generator,
) -> "tuple[Callable[[int], np.ndarray], Callable[[int, int], np.ndarray]] | None":
    """``n_boxes`` BOXES from ``prior`` at once, drawable in row blocks.

    Returns ``(draw_all, draw_one)``: ``draw_all(m) -> (n_boxes, m, n_cols)`` draws an
    ``m``-row block for EVERY box, ``draw_one(j, m) -> (m, n_cols)`` draws a further
    block for box ``j`` alone. For a ``sampler``-form prior the per-column parameter
    sets of all boxes are drawn ONCE here, so every block belongs to its box -- rows
    are iid given the parameters, so any block schedule is distribution-identical to
    drawing each box in one piece. Elementwise-iid builtins need no shared state.
    Returns ``None`` when the prior's shape cannot be block-drawn safely (mixtures,
    unknown callables, non-vectorizable bases).
    """
    if isinstance(prior, partial):
        if prior.func is sampler_dist:
            kw = prior.keywords
            if kw.get("base_dist_name") not in VECTOR_PARAM_BASES:
                return None
            base_kwargs = dict(kw.get("base_kwargs") or {})
            params = {
                name: np.asarray(fn(size=(n_boxes, n_cols), rng=rng), dtype=np.float64)
                for name, fn in kw["param_samplers"].items()
            }
            base_fn = DISTRIBUTIONS.get(kw["base_dist_name"])

            def draw_all_sampler(m: int) -> np.ndarray:
                broadcast = {name: arr[:, None, :] for name, arr in params.items()}
                return base_fn(**base_kwargs, **broadcast, size=(n_boxes, m, n_cols), rng=rng)

            def draw_one_sampler(j: int, m: int) -> np.ndarray:
                box = {name: arr[j] for name, arr in params.items()}
                return base_fn(**base_kwargs, **box, size=(m, n_cols), rng=rng)

            return draw_all_sampler, draw_one_sampler
        if prior.func in ELEMENTWISE_IID_BASES:
            def draw_all_iid(m: int) -> np.ndarray:
                return prior(size=(n_boxes, m, n_cols), rng=rng)

            def draw_one_iid(j: int, m: int) -> np.ndarray:
                return prior(size=(m, n_cols), rng=rng)

            return draw_all_iid, draw_one_iid
    return None


def get_distribution(config: dict[str, Any]) -> Callable[..., np.ndarray]:
    """Create a distribution callable ``(size=1, rng=None) -> ndarray`` from ``config``.

    Supports builtins/registered names, the ``constant`` special form, and the ``sampler``
    nesting form (a distribution whose parameters are themselves sampled).
    """
    if "name" not in config:
        raise ValueError(f"distribution config must include a 'name' key; got keys {sorted(config)}")
    name = config["name"]
    kwargs = config.get("kwargs", {})

    if name == "constant":
        return lambda size=1, rng=None: np.full(size, kwargs["value"])

    if name in DISTRIBUTIONS:
        return partial(DISTRIBUTIONS.get(name), **kwargs)

    if name == "sampler":
        for required_key in ("base_dist_name", "param_samplers"):
            if required_key not in kwargs:
                raise ValueError(f"sampler distribution config must include a '{required_key}' key in kwargs; got kwargs keys {sorted(kwargs)}")
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
