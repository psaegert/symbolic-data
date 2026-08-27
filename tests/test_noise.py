"""The measurement-noise mixture: parse contract, draw semantics, and source wiring."""
from pathlib import Path

import numpy as np
import pytest

from symbolic_data import load_catalog
from symbolic_data.noise import NoiseSpec, apply_noise
from symbolic_data.source import ProblemSource

NGUYEN = str(Path(__file__).resolve().parent.parent / "assets" / "catalogs" / "nguyen.yaml")

RULED = {
    "p_clean": 0.30,
    "types": {"additive": 0.5, "multiplicative": 0.5},
    "level": [1.0e-4, 0.3],
    "outliers": {"p_instance": 0.10, "rate": [0.005, 0.1], "magnitude": [3.0, 100.0]},
}


@pytest.fixture(scope="module")
def engine():
    from simplipy import SimpliPyEngine
    return SimpliPyEngine.load("acj-4", install=True)


def _spec(**overrides):
    raw = {**RULED, **overrides}
    return NoiseSpec.parse(raw)


# --- parse contract: every key required, priors pinned, never defaulted ----------------

def test_parse_ruled_spec():
    spec = NoiseSpec.parse(RULED)
    assert spec.p_clean == 0.30
    assert set(spec.type_names) == {"additive", "multiplicative"}
    assert spec.type_weights == (0.5, 0.5)
    assert spec.level == (1.0e-4, 0.3)
    assert spec.outlier_p_instance == 0.10
    # The legacy [lo, hi] pair still parses; it names the family it always meant.
    assert (spec.outlier_rate.name, spec.outlier_rate.params) == ("uniform", (0.005, 0.1))
    assert (spec.outlier_magnitude.name, spec.outlier_magnitude.params) == ("loguniform", (3.0, 100.0))
    assert (spec.outlier_scale, spec.outlier_sign, spec.outlier_min_count) == ("mad", (1.0, 0.0, 0.0), 0)


@pytest.mark.parametrize("mutate, needle", [
    (lambda raw: raw.pop("p_clean"), "p_clean"),
    (lambda raw: raw.update(extra=1), "extra"),
    (lambda raw: raw.update(types={"salt-and-pepper": 1.0}), "types"),
    (lambda raw: raw.update(types={}), "types"),
    (lambda raw: raw.update(level=[0.3, 1.0e-4]), "level"),
    (lambda raw: raw.update(level=[0.0, 0.3]), "level"),
    (lambda raw: raw.update(p_clean=1.5), "p_clean"),
    (lambda raw: raw["outliers"].pop("rate"), "outliers"),
    (lambda raw: raw["outliers"].update(surprise=1), "outliers"),
])
def test_parse_refuses_malformed_specs(mutate, needle):
    raw = {**RULED, "types": dict(RULED["types"]), "outliers": dict(RULED["outliers"])}
    mutate(raw)
    with pytest.raises(ValueError, match=needle):
        NoiseSpec.parse(raw)


# --- draw semantics --------------------------------------------------------------------

def test_clean_draw_is_the_identity():
    spec = _spec(p_clean=1.0, outliers={"p_instance": 0.0, "rate": [0.005, 0.1], "magnitude": [3.0, 100.0]})
    ys = np.linspace(-2, 2, 32, dtype=np.float32).reshape(-1, 1)
    yv = np.linspace(3, 4, 8, dtype=np.float32).reshape(-1, 1)
    ys_n, yv_n, mask_s, mask_v, draw = apply_noise(spec, ys, yv, np.random.default_rng(0))
    np.testing.assert_array_equal(ys_n, ys)
    np.testing.assert_array_equal(yv_n, yv)
    assert not mask_s.any() and not mask_v.any()
    assert draw["type"] == "clean" and draw["level"] == 0.0 and draw["outlier_rate"] == 0.0


def test_outlier_mask_marks_exactly_the_touched_points():
    spec = _spec(p_clean=1.0, outliers={"p_instance": 1.0, "rate": [0.2, 0.2], "magnitude": [3.0, 3.0]})
    ys = np.linspace(-2, 2, 256, dtype=np.float32).reshape(-1, 1)
    yv = np.empty((0, 1), dtype=np.float32)
    ys_n, _, mask_s, _, draw = apply_noise(spec, ys, yv, np.random.default_rng(1))
    assert mask_s.any(), "rate 0.2 over 256 points must hit"
    np.testing.assert_array_equal(ys_n[~mask_s], ys[~mask_s])
    assert np.all(ys_n[mask_s] != ys[mask_s])
    assert draw["outlier_rate"] == pytest.approx(0.2)
    assert draw["scale"] == pytest.approx(1.4826 * float(np.median(np.abs(ys - np.median(ys)))), rel=1e-6)


def test_constant_targets_skip_outliers_and_survive_multiplicative():
    # scale == 0: no spread to deviate from -- the outlier channel must stay off, and a
    # multiplicative draw on constant zero leaves zero.
    spec = _spec(p_clean=0.0, types={"multiplicative": 1.0},
                 outliers={"p_instance": 1.0, "rate": [0.1, 0.1], "magnitude": [3.0, 3.0]})
    ys = np.zeros((16, 1), dtype=np.float32)
    ys_n, _, mask_s, _, draw = apply_noise(spec, ys, np.empty((0, 1), np.float32), np.random.default_rng(2))
    np.testing.assert_array_equal(ys_n, ys)
    assert not mask_s.any() and draw["outlier_rate"] == 0.0 and draw["type"] == "multiplicative"


def test_f32_overflow_rejects_the_draw():
    spec = _spec(p_clean=0.0, types={"multiplicative": 1.0}, level=[0.3, 0.3],
                 outliers={"p_instance": 0.0, "rate": [0.005, 0.1], "magnitude": [3.0, 100.0]})
    ys = np.full((64, 1), np.finfo(np.float32).max * 0.99, dtype=np.float32)
    rng = np.random.default_rng(3)
    results = [apply_noise(spec, ys, np.empty((0, 1), np.float32), rng) for _ in range(10)]
    assert any(result is None for result in results), "lambda=0.3 at the f32 boundary must overflow"


def test_draw_statistics_match_the_ruled_prior():
    spec = NoiseSpec.parse(RULED)
    rng = np.random.default_rng(4)
    ys = np.linspace(-1, 1, 32, dtype=np.float32).reshape(-1, 1)
    draws = [apply_noise(spec, ys, np.empty((0, 1), np.float32), rng)[4] for _ in range(2000)]
    clean = sum(d["type"] == "clean" for d in draws) / len(draws)
    contaminated = sum(d["outlier_rate"] > 0 for d in draws) / len(draws)
    assert clean == pytest.approx(0.30, abs=0.04)
    assert contaminated == pytest.approx(0.10, abs=0.03)
    levels = [d["level"] for d in draws if d["type"] != "clean"]
    assert min(levels) >= 1.0e-4 and max(levels) <= 0.3


# --- source wiring ---------------------------------------------------------------------

def test_source_mixture_populates_masks_and_realized_draw(engine):
    src = ProblemSource({"catalog": NGUYEN, "sampling": {"n_support": 12, "n_validation": 4, "noise": RULED}},
                        simplipy_engine=engine, rng=np.random.default_rng(5))
    problems = [p for _, p in zip(range(8), iter(src))]
    assert problems
    for p in problems:
        assert p.y_support_noisy.dtype == np.float32
        assert p.outlier_mask_support.dtype == np.bool_
        assert p.outlier_mask_support.shape == p.y_support.shape
        assert p.outlier_mask_validation.shape == p.y_validation.shape
        assert set(p.noise) == {"type", "level", "outlier_rate", "scale",
                                "outlier_scale", "outlier_sign"}
        assert np.all(np.isfinite(p.y_support_noisy))
        if p.noise["type"] == "clean" and p.noise["outlier_rate"] == 0.0:
            np.testing.assert_array_equal(p.y_support_noisy, p.y_support)


def test_source_scalar_noise_keeps_legacy_semantics(engine):
    src = ProblemSource({"catalog": NGUYEN, "sampling": {"n_support": 12, "n_validation": 4, "noise": 0.1}},
                        simplipy_engine=engine, rng=np.random.default_rng(6))
    p = next(iter(src))
    assert p.noise == 0.1
    assert p.outlier_mask_support is None and p.outlier_mask_validation is None
    assert not np.array_equal(p.y_support_noisy, p.y_support)


def test_frozen_roundtrip_carries_masks(engine, tmp_path):
    src = ProblemSource({"catalog": NGUYEN, "sampling": {"n_support": 8, "n_validation": 2, "noise": RULED}},
                        simplipy_engine=engine, rng=np.random.default_rng(7))
    cat = src.to_catalog(name="nguyen-noisy", n=3)
    path = cat.save(tmp_path / "nguyen-noisy.npz")
    reloaded = load_catalog(str(path))
    assert reloaded.problems and len(reloaded.problems) == 3
    for original, restored in zip(cat.problems, reloaded.problems):
        np.testing.assert_array_equal(original.outlier_mask_support, restored.outlier_mask_support)
        np.testing.assert_array_equal(original.y_support_noisy, restored.y_support_noisy)
        assert restored.noise == original.noise


# ---------------------------------------------------------------------------
# Named distributions, the neighbour ruler, coherent signs, conditioned counts
# (the 2026-08-27 prior; see the module docstring for the provenance of each number)
# ---------------------------------------------------------------------------

RULED_2026 = {
    "p_clean": 0.30,
    "types": {"additive": 0.5, "multiplicative": 0.5},
    "level": [1.0e-4, 0.3],
    "outliers": {
        "p_instance": 1.0,
        "rate": {"name": "beta", "a": 1.0, "b": 9.0},
        "magnitude": {"name": "lognormal", "median": 5.0, "sigma": 1.4},
        "scale": "neighbour",
        "sign": {"mixed": 0.5, "up": 0.25, "down": 0.25},
        "min_count": 1,
    },
}


def _curve(n=96):
    x = np.sort(np.random.default_rng(1).uniform(-4, 4, n)).reshape(-1, 1)
    return x, 2.5 * np.sin(1.3 * x[:, 0]) + 0.4 * x[:, 0] ** 2


class TestNamedDistributions:
    def test_named_families_parse(self):
        spec = NoiseSpec.parse(RULED_2026)
        assert (spec.outlier_rate.name, spec.outlier_rate.params) == ("beta", (1.0, 9.0))
        assert (spec.outlier_magnitude.name, spec.outlier_magnitude.params) == ("lognormal", (5.0, 1.4))
        assert spec.outlier_scale == "neighbour"
        assert spec.outlier_min_count == 1

    @pytest.mark.parametrize("bad, needle", [
        ({"name": "wishart", "a": 1, "b": 2}, "unknown distribution"),
        ({"a": 1.0, "b": 9.0}, "needs a 'name'"),
        ({"name": "beta", "a": 1.0}, "takes exactly"),
        ({"name": "beta", "a": 0.0, "b": 9.0}, "a > 0"),
        ({"name": "lognormal", "median": -1.0, "sigma": 1.0}, "median > 0"),
        ({"name": "loguniform", "lo": 0.0, "hi": 1.0}, "lo <= hi"),
    ])
    def test_malformed_distributions_are_refused(self, bad, needle):
        raw = {**RULED_2026, "outliers": {**RULED_2026["outliers"], "rate": bad}}
        with pytest.raises(ValueError, match=needle):
            NoiseSpec.parse(raw)

    @pytest.mark.parametrize("bad, needle", [
        ({"scale": "vibes"}, "scale"),
        ({"sign": {"sideways": 1.0}}, "sign"),
        ({"sign": {}}, "sign"),
        ({"min_count": -1}, "min_count"),
    ])
    def test_malformed_options_are_refused(self, bad, needle):
        raw = {**RULED_2026, "outliers": {**RULED_2026["outliers"], **bad}}
        with pytest.raises(ValueError, match=needle):
            NoiseSpec.parse(raw)


class TestNeighbourScale:
    def test_recovers_the_noise_scale_when_noise_dominates(self):
        from symbolic_data.noise import _neighbour_scale
        rng = np.random.default_rng(0)
        x = np.sort(rng.uniform(-4, 4, 512)).reshape(-1, 1)
        clean = 2.5 * np.sin(1.3 * x[:, 0])
        eps = 0.2
        got = _neighbour_scale(x, clean + rng.normal(0, eps, 512))
        assert 0.7 * eps < got < 1.4 * eps

    def test_stays_finite_on_clean_data(self):
        # The whole reason for this ruler: sigma_eps is exactly 0 on a clean instance,
        # so a residual-scale definition would inject nothing at all there.
        from symbolic_data.noise import _neighbour_scale
        x, y = _curve(256)
        assert _neighbour_scale(x, y) > 0.0

    def test_too_few_points_returns_zero_so_the_caller_falls_back(self):
        from symbolic_data.noise import _neighbour_scale
        assert _neighbour_scale(np.zeros((3, 1)), np.zeros(3)) == 0.0


class TestOutlierChannel:
    def test_min_count_guarantees_a_positive(self):
        x, y = _curve()
        spec = NoiseSpec.parse(RULED_2026)
        for seed in range(60):
            out = apply_noise(spec, y, y[:4], np.random.default_rng(seed),
                              x_support=x, x_validation=x[:4])
            if out is None:
                continue
            assert out[2].sum() >= 1, "a contaminated instance must carry an outlier"

    def test_conditioning_does_not_spike_at_one(self):
        # max(k, 1) would pile every empty draw onto k == 1; conditioning renormalizes.
        x, y = _curve()
        spec = NoiseSpec.parse(RULED_2026)
        counts = []
        for seed in range(400):
            out = apply_noise(spec, y, y[:4], np.random.default_rng(seed),
                              x_support=x, x_validation=x[:4])
            if out is not None:
                counts.append(int(out[2].sum()))
        assert np.mean(np.array(counts) == 1) < 0.35

    def test_sign_is_coherent_within_a_problem(self):
        x, y = _curve()
        # p_clean = 1.0: no Gaussian noise to confound the direction. On a noisy instance
        # a sub-sigma outlier (12.5% of the lognormal's mass sits below kappa = 1, by
        # design) can be masked by the wobble, so the NET value need not move up.
        spec = NoiseSpec.parse({**RULED_2026, "p_clean": 1.0,
                                "outliers": {**RULED_2026["outliers"], "sign": {"up": 1.0}}})
        for seed in range(40):
            out = apply_noise(spec, y, y[:4], np.random.default_rng(seed),
                              x_support=x, x_validation=x[:4])
            if out is None:
                continue
            noisy, _, mask, _, draw = out
            assert draw["outlier_sign"] == "up"
            if mask.any():
                # every touched point moved the same way
                assert np.all(np.asarray(noisy)[mask] > np.asarray(y)[mask])

    def test_the_draw_records_which_ruler_was_used(self):
        x, y = _curve()
        out = apply_noise(NoiseSpec.parse(RULED_2026), y, y[:4], np.random.default_rng(3),
                          x_support=x, x_validation=x[:4])
        assert out is not None
        draw = out[4]
        assert draw["outlier_sign"] in ("mixed", "up", "down")
        assert draw["outlier_scale"] > 0.0
        # the neighbour ruler is a different number from the signal spread
        assert draw["outlier_scale"] != draw["scale"]
