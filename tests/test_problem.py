"""Stage-a tests for the central :class:`Problem` data unit."""
import numpy as np

from symbolic_data import Problem


def _make_problem(**overrides):
    n_vars = 2
    xs = np.zeros((4, n_vars), dtype=np.float32)
    ys = np.zeros((4, 1), dtype=np.float32)
    xv = np.ones((2, n_vars), dtype=np.float32)
    yv = np.ones((2, 1), dtype=np.float32)
    kwargs = dict(
        x_support=xs, y_support=ys, y_support_noisy=ys.copy(),
        x_validation=xv, y_validation=yv, y_validation_noisy=yv.copy(),
        skeleton=("add", "x1", "mul", "c", "x2"),
        expression=["add", "x1", "mul", "1.5", "x2"],
        constants=[1.5],
        variables=["x1", "x2", "x3"],
        complexity=5,
        noise=0.0,
        eq_id="demo-1",
    )
    kwargs.update(overrides)
    return Problem(**kwargs)


def test_n_variables_used_is_derived():
    p = _make_problem()
    # x1 and x2 appear in the skeleton; x3 does not.
    assert p.n_variables_used == 2
    assert _make_problem(skeleton=None).n_variables_used == 0


def test_is_finite_gate():
    assert _make_problem().is_finite()
    bad = _make_problem()
    bad.y_support = bad.y_support.copy()
    bad.y_support[0, 0] = np.nan
    assert not bad.is_finite()
    inf = _make_problem()
    inf.x_validation = inf.x_validation.copy()
    inf.x_validation[0, 0] = np.inf
    assert not inf.is_finite()


def test_to_from_dict_roundtrip():
    p = _make_problem(meta={"units": [1, 0, 0], "moniker": "demo"})
    d = p.to_dict()
    q = Problem.from_dict(d)
    assert q.eq_id == p.eq_id
    assert q.skeleton == p.skeleton
    assert q.expression == p.expression
    assert q.constants == p.constants
    assert q.complexity == p.complexity
    assert q.meta == p.meta
    assert np.array_equal(q.x_support, p.x_support)
    assert np.array_equal(q.y_validation_noisy, p.y_validation_noisy)
    # to_dict copies meta (mutating the dict must not touch the Problem)
    d["meta"]["moniker"] = "changed"
    assert p.meta["moniker"] == "demo"


def test_placeholder_constructor():
    p = Problem.placeholder(["x1", "x2"], reason="max_trials_exhausted", eq_id="e")
    assert p.is_placeholder and p.placeholder_reason == "max_trials_exhausted"
    assert p.eq_id == "e"
    assert p.x_support.shape == (0, 2) and p.y_support.shape == (0, 1)
    assert p.n_variables_used == 0
    assert p.is_finite()  # empty arrays are vacuously finite
