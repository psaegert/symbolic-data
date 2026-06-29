"""Stage-a tests for the unified distribution framework + the `fastsrb` distribution.

Per the 0.4.0 design: we do NOT assert bit-equality against 0.3.0's *global-np.random* stream
(seeding is dropped). We DO assert (1) the `fastsrb` distribution faithfully reproduces the
existing `_sample_points` recipe under a shared Generator (faithful porting), (2) the framework
threads a Generator everywhere, and (3) nesting/mixtures compose with `fastsrb`.
"""
import math

import numpy as np
import pytest

from symbolic_data import build_prior_callable, fastsrb_dist, get_distribution
from symbolic_data.benchmarks.spec import SpecBenchmark


# --- (1) faithful porting: fastsrb_dist == SpecBenchmark._sample_points ------------------------

# _sample_points does not touch `self`, so an uninitialised instance is a clean oracle.
_ORACLE = SpecBenchmark.__new__(SpecBenchmark)

_LAYOUT_TO_METHOD = {"random": "random", "grid": "range"}


def _recipe_args(base: str):
    if base == "int":
        return "uni", True
    return base, False


@pytest.mark.parametrize("base", ["uni", "log", "int"])
@pytest.mark.parametrize("sign", ["pos", "neg", "pos_neg"])
@pytest.mark.parametrize("layout", ["random", "grid"])
def test_fastsrb_matches_published_recipe(base, sign, layout):
    low, high = (1e-3, 1e1) if base == "log" else (-3.0, 5.0)
    n = 16
    distribution, integer = _recipe_args(base)

    got = fastsrb_dist(low, high, base=base, sign=sign, layout=layout, size=n,
                       rng=np.random.default_rng(12345))
    ref = _ORACLE._sample_points(low, high, n, method=_LAYOUT_TO_METHOD[layout],
                                 distribution=distribution, sign_mode=sign, integer=integer,
                                 rng=np.random.default_rng(12345))
    assert np.array_equal(got, ref)


# --- (2) invariants ---------------------------------------------------------------------------

def test_fastsrb_log_uniform_is_base_invariant():
    # IMPORTANT (finding): log-uniform sampling is base-INVARIANT. fastsrb's base-10 `log`
    # and the native natural-log `log_uniform` both yield lo*(hi/lo)**r for the same underlying
    # draw r, so for the same Generator they produce IDENTICAL values. The base-10-vs-natural
    # "divergence" flagged during design is therefore only an internal-representation detail,
    # not a difference in the sampled data.
    base10 = fastsrb_dist(1e-3, 1e3, base="log", size=2048, rng=np.random.default_rng(0))
    natural = get_distribution({"name": "log_uniform", "kwargs": {"low": 1e-3, "high": 1e3}})(
        size=2048, rng=np.random.default_rng(0))
    assert np.allclose(base10, natural)
    assert base10.min() >= 1e-3 - 1e-9 and base10.max() <= 1e3 + 1e-6


def test_fastsrb_sign_applied_after_exponentiation():
    neg = fastsrb_dist(1e-2, 1e2, base="log", sign="neg", size=64, rng=np.random.default_rng(1))
    assert np.all(neg <= 0)
    pos = fastsrb_dist(1e-2, 1e2, base="log", sign="pos", size=64, rng=np.random.default_rng(1))
    assert np.all(pos > 0)


def test_fastsrb_int_yields_integers():
    arr = fastsrb_dist(-5, 5, base="int", size=64, rng=np.random.default_rng(2))
    assert np.array_equal(arr, np.rint(arr))


def test_fastsrb_grid_is_shuffled_linspace():
    arr = fastsrb_dist(0.0, 1.0, base="uni", layout="grid", size=11, rng=np.random.default_rng(3))
    assert np.allclose(np.sort(arr), np.linspace(0.0, 1.0, 11))


def test_fastsrb_constant_fill_and_log_positivity():
    assert np.all(fastsrb_dist(2.5, 2.5, size=8) == 2.5)
    with pytest.raises(ValueError):
        fastsrb_dist(-1.0, 1.0, base="log", size=4, rng=np.random.default_rng(4))


# --- (3) rng threading + nesting/mixtures -----------------------------------------------------

def test_same_generator_seed_reproduces_within_session():
    a = get_distribution({"name": "uniform", "kwargs": {"low": 0, "high": 1}})
    out1 = a(size=32, rng=np.random.default_rng(7))
    out2 = a(size=32, rng=np.random.default_rng(7))
    assert np.array_equal(out1, out2)


def test_sampler_nesting_threads_rng():
    cfg = {"name": "sampler", "kwargs": {
        "base_dist_name": "normal",
        "param_samplers": {"loc": {"name": "uniform", "kwargs": {"low": -1, "high": 1}}},
        "base_kwargs": {"scale": 1.0}}}
    dist = get_distribution(cfg)
    out1 = dist(size=16, rng=np.random.default_rng(9))
    out2 = dist(size=16, rng=np.random.default_rng(9))
    assert np.array_equal(out1, out2)
    assert out1.shape == (16,)


def test_fastsrb_nests_inside_mixture():
    mix = build_prior_callable([
        {"name": "fastsrb", "kwargs": {"low": 1e-2, "high": 1e2, "base": "log", "sign": "pos_neg"}, "weight": 1.0},
        {"name": "uniform", "kwargs": {"low": -1, "high": 1}, "weight": 1.0},
    ])
    out = mix(size=8, rng=np.random.default_rng(11))
    assert out.shape == (8,) and np.all(np.isfinite(out))
