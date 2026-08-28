"""Factory helpers for constructing prior distribution callables."""
from typing import Any, Callable

import numpy as np

from symbolic_data.distributions import _DEFAULT_RNG, get_distribution


def build_prior_callable(config: dict[str, Any] | list[dict[str, Any]]) -> Callable:
    """Create a sampler function from a prior configuration."""
    if isinstance(config, list):
        distributions = [get_distribution(sub_config) for sub_config in config]
        weights = np.array([sub_config.get("weight", 1.0) for sub_config in config], dtype=np.float64)
        if weights.sum() == 0:
            raise ValueError("Mixture prior weights must sum to a positive value.")
        weights /= weights.sum()

        def mixture_distribution(size: Any = 1, rng: np.random.Generator | None = None) -> Any:
            generator = rng if rng is not None else _DEFAULT_RNG
            chosen_index = int(generator.choice(len(distributions), p=weights))
            chosen_dist_callable = distributions[chosen_index]
            return chosen_dist_callable(size=size, rng=generator)

        return mixture_distribution

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
        distributions = [get_distribution(sub_config) for sub_config in config]
        weights = np.array([sub_config.get("weight", 1.0) for sub_config in config], dtype=np.float64)
        if weights.sum() == 0:
            raise ValueError("Mixture prior weights must sum to a positive value.")
        weights /= weights.sum()

        def iid_mixture_distribution(size: Any = 1, rng: np.random.Generator | None = None) -> Any:
            generator = rng if rng is not None else _DEFAULT_RNG
            n = int(size)
            chosen = generator.choice(len(distributions), size=n, p=weights)
            out = np.empty(n, dtype=np.float64)
            for index, distribution in enumerate(distributions):
                selected = chosen == index
                count = int(selected.sum())
                if count:
                    out[selected] = np.asarray(
                        distribution(size=count, rng=generator), dtype=np.float64).reshape(-1)
            return out

        return iid_mixture_distribution

    if isinstance(config, dict):
        return get_distribution(config)

    raise TypeError(
        "Prior configuration must be a dict or a list of dicts; "
        f"got {type(config).__name__}."
    )
