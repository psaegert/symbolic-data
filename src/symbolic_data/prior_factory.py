"""Factory helpers for constructing prior distribution callables."""
from typing import Any, Callable

import numpy as np

from symbolic_data.distributions import _DEFAULT_RNG, get_distribution


class _MixturePrior:
    """A weighted mixture of distributions, as an object rather than a closure.

    Closures do not pickle, and a prior has to reach a spawned data worker. Both
    resolution policies live here: ``per_value=False`` draws one component for the whole
    call (the regime is shared across the draw), ``per_value=True`` draws a component per
    value (each value is its own draw).
    """

    __slots__ = ("distributions", "weights", "per_value")

    def __init__(self, distributions: list, weights: np.ndarray, per_value: bool) -> None:
        self.distributions = distributions
        self.weights = weights
        self.per_value = per_value

    def __call__(self, size: Any = 1, rng: np.random.Generator | None = None) -> Any:
        generator = rng if rng is not None else _DEFAULT_RNG
        if not self.per_value:
            chosen = int(generator.choice(len(self.distributions), p=self.weights))
            return self.distributions[chosen](size=size, rng=generator)
        n = int(size)
        picks = generator.choice(len(self.distributions), size=n, p=self.weights)
        out = np.empty(n, dtype=np.float64)
        for index, distribution in enumerate(self.distributions):
            selected = picks == index
            count = int(selected.sum())
            if count:
                out[selected] = np.asarray(
                    distribution(size=count, rng=generator), dtype=np.float64).reshape(-1)
        return out


def _resolve_mixture(config: list[dict[str, Any]], per_value: bool) -> "_MixturePrior":
    distributions = [get_distribution(sub_config) for sub_config in config]
    weights = np.array([sub_config.get("weight", 1.0) for sub_config in config], dtype=np.float64)
    if weights.sum() == 0:
        raise ValueError("Mixture prior weights must sum to a positive value.")
    weights /= weights.sum()
    return _MixturePrior(distributions, weights, per_value)


def build_prior_callable(config: dict[str, Any] | list[dict[str, Any]]) -> Callable:
    """Create a sampler function from a prior configuration."""
    if isinstance(config, list):
        return _resolve_mixture(config, per_value=False)

    if isinstance(config, dict):
        return get_distribution(config)

    raise TypeError(
        "Prior configuration must be a dict or a list of dicts; "
        f"got {type(config).__name__}."
    )


def build_iid_prior_callable(config: dict[str, Any] | list[dict[str, Any]]) -> Callable:
    """Same distribution as :func:`build_prior_callable`, drawn independently per value.

    :func:`build_prior_callable` resolves a mixture once per CALL, so ``size=n`` returns n
    values from a single component -- correct when the whole draw shares one regime, wrong
    when each value is its own draw. This variant chooses a component per value, which
    makes one ``size=n`` call distributionally identical to n ``size=1`` calls (it is not
    stream-identical: the same seed yields a different sequence).
    """
    if isinstance(config, list):
        return _resolve_mixture(config, per_value=True)

    if isinstance(config, dict):
        return get_distribution(config)

    raise TypeError(
        "Prior configuration must be a dict or a list of dicts; "
        f"got {type(config).__name__}."
    )
