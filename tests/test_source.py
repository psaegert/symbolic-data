"""Stage-c tests for the SET/FIXED-mode ProblemSource (catalog -> Problems)."""
import numpy as np
import pytest

from symbolic_data import Problem, ProblemSource
from symbolic_data import resolver as R


@pytest.fixture(scope="module")
def engine():
    from simplipy import SimpliPyEngine
    return SimpliPyEngine.load("dev_7-3", install=True)


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    # Resolve curated catalogs from the vendored package data (no network).
    monkeypatch.setattr(R, "fetch_manifest", lambda **kw: {})


def _source(engine, **sampling):
    return ProblemSource({"catalog": "nguyen", "sampling": sampling}, simplipy_engine=engine)


def test_set_mode_iterates_catalog_into_problems(engine):
    src = _source(engine, n_support=8, n_validation=4, method="iterate", noise=0.0)
    assert src.mode == "set" and src.size_hint() == 12
    problems = list(src)
    assert len(problems) == 12
    eq_ids = set()
    for p in problems:
        assert isinstance(p, Problem) and not p.is_placeholder
        nv = len(p.variables)
        assert p.x_support.shape == (8, nv) and p.y_support.shape == (8, 1)
        assert p.x_validation.shape == (4, nv) and p.y_validation.shape == (4, 1)
        assert p.is_finite() and p.expression and p.n_variables_used >= 1
        assert p.x_support.dtype == np.float32
        eq_ids.add(p.eq_id)
    assert len(eq_ids) == 12  # all distinct nguyen equations


def test_problems_per_expression_multiplies(engine):
    src = _source(engine, n_support=6, n_validation=2, problems_per_expression=3)
    assert src.size_hint() == 36
    assert len(list(src)) == 36


def test_noise_zero_means_noisy_equals_clean(engine):
    p = next(iter(_source(engine, n_support=6, n_validation=2, noise=0.0)))
    assert np.array_equal(p.y_support, p.y_support_noisy)


def test_noise_positive_perturbs_targets(engine):
    p = next(iter(_source(engine, n_support=12, n_validation=4, noise=0.1)))
    assert not np.array_equal(p.y_support, p.y_support_noisy)
    assert p.x_support.shape[0] == 12  # X is never noised; just check shape


def test_random_without_replacement_covers_all_once(engine):
    src = _source(engine, n_support=6, n_validation=2, method="random_without_replacement")
    eq_ids = [p.eq_id for p in src]
    assert len(eq_ids) == 12 and len(set(eq_ids)) == 12


def test_filter_max_complexity_drops_complex_problems(engine):
    base = list(_source(engine, n_support=6, n_validation=2))
    src = ProblemSource(
        {"catalog": "nguyen", "sampling": {"n_support": 6, "n_validation": 2},
         "holdouts": [{"filter": {"max_complexity": 5}}]},
        simplipy_engine=engine,
    )
    kept = list(src)
    assert all(p.complexity is None or p.complexity <= 5 for p in kept)
    assert len(kept) < len(base)  # nguyen has higher-complexity entries that get filtered


def test_exclude_holdout_not_yet_implemented(engine):
    src = ProblemSource(
        {"catalog": "nguyen", "holdouts": [{"exclude": "feynman"}]}, simplipy_engine=engine,
    )
    with pytest.raises(NotImplementedError):
        list(src)


def test_fixed_mode_roundtrips_inline_problems(engine):
    problems = list(_source(engine, n_support=6, n_validation=2))[:3]
    fixed = ProblemSource({"problems": [p.to_dict() for p in problems]})
    assert fixed.mode == "fixed" and fixed.size_hint() == 3
    out = list(fixed)
    assert len(out) == 3 and out[0].eq_id == problems[0].eq_id
    assert np.array_equal(out[0].x_support, problems[0].x_support)


def test_generate_mode_stub_raises():
    src = ProblemSource({"generator": {"operators": {}}})
    assert src.mode == "generate"
    with pytest.raises(NotImplementedError):
        list(src)
