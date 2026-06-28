"""Tests for the extensible registry and its first consumer, ``DISTRIBUTIONS``.

The load-bearing guarantee: routing distribution lookups through the registry is a
behaviour-preserving refactor. Every builtin must resolve to the *same* callable and
produce *byte-identical* samples (for the same seed and config) as before, so no
existing pool config changes meaning. The remaining tests cover the new extensibility
surface (decorator registration, entry-point-style shadowing policy, strict mode) and
that unknown names still raise ``ValueError`` from ``get_distribution`` (error-type
back-compat).
"""
import numpy as np
import pytest

from sr_data import DISTRIBUTIONS, Registry, get_distribution
from sr_data.distributions import BASE_DISTRIBUTIONS, sampler_dist

# One valid config per builtin (kwargs match each factory's signature).
BUILTIN_CONFIGS = {
    "uniform": {"low": -3.0, "high": 3.0},
    "normal": {"loc": 1.0, "scale": 2.0},
    "log_uniform": {"low": 0.1, "high": 10.0},
    "log_normal": {"mean": 0.0, "sigma": 1.0},
    "gamma": {"shape": 2.0, "scale": 1.5},
    "cauchy": {"loc": 0.0, "scale": 1.0},
    "binomial": {"n": 10, "p": 0.3},
}


def test_registry_seeded_faithfully_from_builtins():
    """The registry exposes exactly the builtins, each pointing at the same object."""
    assert set(DISTRIBUTIONS.names(builtins_only=True)) == set(BASE_DISTRIBUTIONS)
    for name, fn in BASE_DISTRIBUTIONS.items():
        assert name in DISTRIBUTIONS
        assert DISTRIBUTIONS.get(name) is fn
        assert DISTRIBUTIONS.get(name, strict_builtins=True) is fn


@pytest.mark.parametrize("name,kwargs", list(BUILTIN_CONFIGS.items()))
def test_builtin_resolution_is_behaviour_preserving(name, kwargs):
    """get_distribution via the registry == calling the builtin factory directly (same seed)."""
    config = {"name": name, "kwargs": kwargs}

    np.random.seed(20240617)
    via_registry = get_distribution(config)(size=64)

    np.random.seed(20240617)
    direct = BASE_DISTRIBUTIONS[name](**kwargs, size=64)

    np.testing.assert_array_equal(via_registry, direct)


def test_constant_and_sampler_special_forms_unchanged():
    """The structural special forms ('constant', nested 'sampler') still resolve."""
    const = get_distribution({"name": "constant", "kwargs": {"value": 7.0}})
    np.testing.assert_array_equal(const(size=5), np.full(5, 7.0))

    np.random.seed(0)
    nested = get_distribution(
        {
            "name": "sampler",
            "kwargs": {
                "base_dist_name": "normal",
                "param_samplers": {"loc": {"name": "uniform", "kwargs": {"low": -1, "high": 1}}},
                "base_kwargs": {"scale": 1.0},
            },
        }
    )
    assert nested(size=3).shape == (3,)


def test_unknown_name_still_raises_valueerror():
    """Error-type back-compat: get_distribution / sampler_dist raise ValueError, not KeyError."""
    with pytest.raises(ValueError, match="Unknown distribution name"):
        get_distribution({"name": "does_not_exist"})
    with pytest.raises(ValueError, match="Unknown base_dist_name"):
        sampler_dist("does_not_exist", {})


def test_custom_distribution_drops_into_a_config_slot():
    """A registered custom distribution is usable from the same config shape as a builtin."""
    reg = Registry("distribution")

    @reg.register("scaled_const")
    def _scaled(value, factor=2.0, size=1):
        return np.full(size, value * factor)

    assert "scaled_const" in reg
    assert reg.get("scaled_const")(value=3.0, size=4).tolist() == [6.0, 6.0, 6.0, 6.0]
    # case-insensitive
    assert reg.get("SCALED_CONST") is reg.get("scaled_const")


def test_collision_warns_and_keeps_existing_unless_overwrite():
    reg = Registry("distribution")
    reg.register("x", lambda size=1: np.zeros(size))

    with pytest.warns(UserWarning, match="already registered"):
        reg.register("x", lambda size=1: np.ones(size))
    np.testing.assert_array_equal(reg.get("x")(size=2), np.zeros(2))  # kept original

    reg.register("x", lambda size=1: np.ones(size), overwrite=True)
    np.testing.assert_array_equal(reg.get("x")(size=2), np.ones(2))  # now shadowed


def test_strict_builtins_excludes_custom_registrations():
    DISTRIBUTIONS.register("ephemeral_custom", lambda size=1: np.zeros(size))
    try:
        assert "ephemeral_custom" in DISTRIBUTIONS
        assert DISTRIBUTIONS.get("ephemeral_custom") is not None
        with pytest.raises(KeyError):
            DISTRIBUTIONS.get("ephemeral_custom", strict_builtins=True)
    finally:
        # keep the module-level singleton clean for other tests
        DISTRIBUTIONS._fns.pop("ephemeral_custom", None)


def test_unknown_in_registry_raises_keyerror_with_helpful_message():
    reg = Registry("distribution")
    with pytest.raises(KeyError, match="Unknown distribution"):
        reg.get("nope")
