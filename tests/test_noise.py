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
    assert spec.outlier_rate == (0.005, 0.1)
    assert spec.outlier_magnitude == (3.0, 100.0)


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
        assert set(p.noise) == {"type", "level", "outlier_rate", "scale"}
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
