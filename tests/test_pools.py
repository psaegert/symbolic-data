"""Data catalogs: problems that hold pools of rows, drawn afresh by the ProblemSource (symbolic_data.pools)."""
import math

import numpy as np
import pytest

from symbolic_data import ProblemSource, data_problem, draw_from_pools
from symbolic_data.catalog import ProblemCatalog
from symbolic_data.pools import has_pools, split_rule
from symbolic_data.problem import Problem


def _rows(n, start=0, n_vars=2):
    """Rows whose first column is their own index, so a draw can be traced back to the pool."""
    x = np.column_stack([np.arange(start, start + n, dtype=float)] + [np.full(n, 0.5)] * (n_vars - 1))
    return x, 2.0 * x[:, 0] + 1.0


def _freeze(problems, tmp_path, name="pools"):
    path = ProblemCatalog.from_problems(problems, name=name).save(tmp_path / name)
    return str(path)


def _ids(a):
    return set(np.asarray(a)[:, 0].astype(int).tolist())


def test_a_realized_frozen_problem_is_used_as_stored():
    x, y = _rows(1000)
    realized = Problem(x_support=x, y_support=y.reshape(-1, 1), y_support_noisy=y.reshape(-1, 1),
                       x_validation=x[:10], y_validation=y[:10].reshape(-1, 1), y_validation_noisy=y[:10].reshape(-1, 1),
                       skeleton=("+", "x1"), expression=["+", "x1"], constants=[], variables=["x1", "x2"], complexity=2)
    assert not has_pools(realized)
    assert draw_from_pools(realized, 512, 512, np.random.default_rng(0)) is realized


def test_given_split_draws_capped_rows_from_each_pool(tmp_path):
    x, y = _rows(2000)
    xt, yt = _rows(700, start=10_000)
    p = data_problem(x, y, x_test=xt, y_test=yt, variables=["x1", "x2"], test_split="given", eq_id="g", gt_kind="none")
    src = ProblemSource({"catalog": _freeze([p], tmp_path), "sampling": {"n_support": 512, "n_validation": 512}})
    a, b = next(iter(src)), next(iter(src))
    assert a.x_support.shape == (512, 2) and a.x_validation.shape == (512, 2)
    assert _ids(a.x_support) <= set(range(2000)) and _ids(a.x_validation) <= set(range(10_000, 10_700))
    np.testing.assert_array_equal(a.y_support.ravel(), 2.0 * a.x_support[:, 0] + 1.0)        # rows stay aligned
    np.testing.assert_array_equal(a.y_support_noisy, a.y_support)
    assert _ids(a.x_support) != _ids(b.x_support)                                              # a fresh draw per pass
    assert a.meta["n_fit_pool"] == 2000 and a.meta["n_test_pool"] == 700 and a.meta["test_split"] == "given"


def test_pools_no_larger_than_the_caps_are_used_whole_in_order(tmp_path):
    x, y = _rows(300)
    xt, yt = _rows(40, start=5000)
    p = data_problem(x, y, x_test=xt, y_test=yt, variables=["x1", "x2"], test_split="given", eq_id="small", gt_kind="none")
    out = next(iter(ProblemSource({"catalog": _freeze([p], tmp_path), "sampling": {"n_support": 512, "n_validation": 512}})))
    np.testing.assert_array_equal(out.x_support, p.x_support)
    np.testing.assert_array_equal(out.x_validation, p.x_validation)


def test_no_split_means_no_test(tmp_path):
    x, y = _rows(900)
    p = data_problem(x, y, variables=["x1", "x2"], test_split="none", eq_id="n", gt_kind="none")
    out = next(iter(ProblemSource({"catalog": _freeze([p], tmp_path), "sampling": {"n_support": 512, "n_validation": 512}})))
    assert out.x_support.shape == (512, 2) and out.x_validation.shape == (0, 2) and out.y_validation.shape == (0, 1)
    assert out.meta["n_test_pool"] == 0
    with pytest.raises(ValueError, match="only stored for test_split 'given'"):
        data_problem(x, y, x_test=x[:5], y_test=y[:5], variables=["x1", "x2"], test_split="none", gt_kind="none")
    bad = Problem(x_support=x, y_support=y.reshape(-1, 1), y_support_noisy=y.reshape(-1, 1), x_validation=x[:3],
                  y_validation=y[:3].reshape(-1, 1), y_validation_noisy=y[:3].reshape(-1, 1), skeleton=None,
                  expression=None, constants=[], variables=["x1", "x2"], complexity=None, meta={"test_split": "none"})
    with pytest.raises(ValueError, match="keeps its validation arrays empty"):
        draw_from_pools(bad, 512, 512, np.random.default_rng(0))


def test_a_random_split_is_cut_anew_every_draw_as_sklearn_counts_it():
    x, y = _rows(1001)
    p = data_problem(x, y, variables=["x1", "x2"], test_split={"random": 0.25}, eq_id="r", gt_kind="none")
    rng = np.random.default_rng(1)
    a = draw_from_pools(p, None, None, rng)                      # uncapped: the split alone
    n_test = math.ceil(0.25 * 1001)
    assert a.x_validation.shape[0] == n_test and a.x_support.shape[0] == 1001 - n_test
    assert _ids(a.x_support).isdisjoint(_ids(a.x_validation)) and _ids(a.x_support) | _ids(a.x_validation) == set(range(1001))
    b = draw_from_pools(p, 512, 128, rng)                        # capped after the split
    assert b.x_support.shape[0] == 512 and b.x_validation.shape[0] == 128
    assert _ids(b.x_support).isdisjoint(_ids(b.x_validation))
    assert b.meta["n_fit_pool"] == 1001 - n_test and b.meta["n_test_pool"] == n_test
    c = draw_from_pools(p, None, None, np.random.default_rng(2))
    assert _ids(c.x_validation) != _ids(a.x_validation)          # the split itself is drawn per pass


def test_seeded_draws_reproduce():
    x, y = _rows(3000)
    xt, yt = _rows(3000, start=9000)
    p = data_problem(x, y, x_test=xt, y_test=yt, variables=["x1", "x2"], test_split="given", gt_kind="none")
    a = draw_from_pools(p, 512, 512, np.random.default_rng(7))
    b = draw_from_pools(p, 512, 512, np.random.default_rng(7))
    np.testing.assert_array_equal(a.x_support, b.x_support)
    np.testing.assert_array_equal(a.x_validation, b.x_validation)


def test_reference_predictions_are_cut_with_their_rows():
    x, y = _rows(1500)
    xt, yt = _rows(800, start=4000)
    p = data_problem(x, y, x_test=xt, y_test=yt, variables=["x1", "x2"], test_split="given", eq_id="ref",
                     expression=["+", "x1", "1"], gt_kind="reference")
    p.y_reference_support = (3.0 * x[:, 0]).reshape(-1, 1)
    p.y_reference_validation = (3.0 * xt[:, 0]).reshape(-1, 1)
    out = draw_from_pools(p, 256, 64, np.random.default_rng(3))
    np.testing.assert_array_equal(out.y_reference_support.ravel(), 3.0 * out.x_support[:, 0])
    np.testing.assert_array_equal(out.y_reference_validation.ravel(), 3.0 * out.x_validation[:, 0])


def test_the_split_rule_and_the_rows_are_checked():
    assert split_rule("given") == "given" and split_rule("none") == "none" and split_rule({"random": 0.25}) == 0.25
    for bad in ("75/25", {"random": 1.0}, {"random": 0.2, "seed": 1}):
        with pytest.raises(ValueError):
            split_rule(bad)
    x, y = _rows(20)
    with pytest.raises(ValueError, match="needs the authors' test rows"):
        data_problem(x, y, variables=["x1", "x2"], test_split="given", gt_kind="none")
    with pytest.raises(ValueError, match="do not match"):
        data_problem(x, y[:5], variables=["x1", "x2"], test_split="none", gt_kind="none")


def test_n_support_prior_cannot_cap_pools(tmp_path):
    x, y = _rows(50)
    p = data_problem(x, y, variables=["x1", "x2"], test_split="none", gt_kind="none")
    src = ProblemSource({"catalog": _freeze([p], tmp_path), "sampling": {"n_support": "prior", "n_validation": 0}})
    with pytest.raises(ValueError, match="cannot cap a data catalog"):
        list(src)
