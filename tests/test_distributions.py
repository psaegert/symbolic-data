"""Stage-a tests for the unified distribution framework + the `fastsrb` distribution.

Per the 0.4.0 design: we do NOT assert bit-equality against 0.3.0's *global-np.random* stream
(seeding is dropped). We DO assert (1) the `fastsrb` distribution faithfully reproduces the
published `_sample_points` recipe, (2) the framework threads a Generator everywhere, and
(3) nesting/mixtures compose with `fastsrb`.

The faithful-porting check (1) compares against a CHECKED-IN golden fixture
(`tests/golden/fastsrb_recipe.npz`), captured once from `SpecBenchmark._sample_points`. The
fixture is the frozen oracle so the proof survives Stage b deleting `benchmarks/spec.py`.
Regenerate with ``python tests/test_distributions.py`` while `spec.py` still exists.
"""
from pathlib import Path

import numpy as np
import pytest

from symbolic_data import build_prior_callable, fastsrb_dist, get_distribution

# Shared fixture spec (used by both the test and the __main__ generator below) ------------------
SEED = 12345
N = 16
COMBOS = [(base, sign, layout)
          for base in ("uni", "log", "int")
          for sign in ("pos", "neg", "pos_neg")
          for layout in ("random", "grid")]
GOLDEN_PATH = Path(__file__).parent / "golden" / "fastsrb_recipe.npz"
_LAYOUT_TO_METHOD = {"random": "random", "grid": "range"}


def _bounds(base: str) -> tuple[float, float]:
    return (1e-3, 1e1) if base == "log" else (-3.0, 5.0)


def _recipe_args(base: str) -> tuple[str, bool]:
    if base == "int":
        return "uni", True
    return base, False


def _combo_key(base: str, sign: str, layout: str) -> str:
    return f"{base}-{sign}-{layout}"


# --- (1) faithful porting: fastsrb_dist == frozen `_sample_points` golden ----------------------

@pytest.mark.parametrize("base,sign,layout", COMBOS)
def test_fastsrb_matches_published_recipe(base, sign, layout):
    golden = np.load(GOLDEN_PATH)
    low, high = _bounds(base)
    got = fastsrb_dist(low, high, base=base, sign=sign, layout=layout, size=N,
                       rng=np.random.default_rng(SEED))
    ref = golden[_combo_key(base, sign, layout)]
    # Reproduce the golden recipe to floating-point precision, NOT bit-exactly: the seeded Generator
    # stream is version-stable, but a float transform like the base-10 `log`'s ``10**x`` can differ by
    # ~1 ULP across numpy builds, which `array_equal` would (and did, in CI) flag spuriously. A tight
    # tolerance still catches any real recipe change while surviving numpy point releases.
    np.testing.assert_allclose(got, ref, rtol=1e-9, atol=1e-12)


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


# --- golden regeneration (run while benchmarks/spec.py still exists; pre-Stage-b) --------------

if __name__ == "__main__":
    # The committed fixture was ORIGINALLY captured from the historical SpecBenchmark._sample_points
    # oracle (proof that fastsrb_dist faithfully ports the published recipe). That oracle is gone in
    # 0.4.0; fastsrb_dist is the canonical implementation and reproduces it exactly, so a regen now
    # snapshots fastsrb_dist itself as a forward-stability baseline (only re-run intentionally).
    arrays = {}
    for base, sign, layout in COMBOS:
        low, high = _bounds(base)
        arrays[_combo_key(base, sign, layout)] = fastsrb_dist(
            low, high, base=base, sign=sign, layout=layout, size=N, rng=np.random.default_rng(SEED))
    GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(GOLDEN_PATH, **arrays)
    print(f"wrote {len(arrays)} golden arrays -> {GOLDEN_PATH}")
