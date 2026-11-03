"""Input specs of declarative catalog entries: dist configs and mixtures, joint samplers, derived variables
and row constraints; the fastsrb shorthand draws exactly as before."""
import numpy as np
import pytest

from symbolic_data.catalog import ProblemCatalog
from symbolic_data.distributions import fastsrb_dist
from symbolic_data.errors import CatalogEntryError
from symbolic_data.inputs import JOINT_INPUTS, constraint_mask, draw_rows, draw_variable


@pytest.fixture(scope="module")
def engine():
    from simplipy import SimpliPyEngine
    return SimpliPyEngine.load("acj-4", install=True)


def _catalog(entry: dict) -> ProblemCatalog:
    return ProblemCatalog.from_yaml({"metadata": {"name": "t", "version": 1}, "expressions": {"e": entry}})


def test_the_fastsrb_shorthand_draws_the_same_stream_as_before():
    variables = {"v1": {"sample_range": [-1.0, 1.0], "sample_type": ["uni", "pos"]},
                 "v2": {"sample_range": [0.1, 10.0], "sample_type": ["log", "pos_neg"]}}
    a = draw_rows(variables, {}, ["v1", "v2"], 64, np.random.default_rng(7))
    rng = np.random.default_rng(7)
    b = np.column_stack([fastsrb_dist(-1.0, 1.0, base="uni", sign="pos", layout="random", size=64, rng=rng),
                         fastsrb_dist(0.1, 10.0, base="log", sign="pos_neg", layout="random", size=64, rng=rng)])
    assert np.array_equal(a, b)


def test_a_dist_config_and_a_mixture_per_value():
    rng = np.random.default_rng(0)
    x = draw_variable({"dist": {"name": "log_uniform", "kwargs": {"low": 0.1, "high": 10.0}}}, 4000, rng)
    assert x.min() >= 0.1 and x.max() <= 10.0 and abs(np.median(np.log10(x))) < 0.1
    mix = draw_variable({"dist": [{"name": "uniform", "kwargs": {"low": 0.0, "high": 1.0}, "weight": 3},
                                  {"name": "uniform", "kwargs": {"low": 10.0, "high": 11.0}, "weight": 1}]}, 8000, rng)
    assert set(np.unique(np.floor(mix / 10.0))) == {0.0, 1.0}
    assert abs((mix >= 10.0).mean() - 0.25) < 0.02


def test_a_derived_variable_is_a_push_forward_of_latent_variables():
    meta = {"latent": {"x1": {"sample_range": [1.0, 5.0], "sample_type": ["uni", "pos"]},
                       "x2": {"sample_range": [1.0, 5.0], "sample_type": ["uni", "pos"]}}}
    rows = draw_rows({"v1": {"expr": "x2 / x1"}, "v2": {"expr": "x1"}}, meta, ["v1", "v2"], 1000, np.random.default_rng(1))
    assert rows.min() >= 0.2 - 1e-12 and rows[:, 0].max() <= 5.0 + 1e-12
    assert not np.allclose(rows[:, 0], rows[:, 1])


def test_a_joint_sampler_draws_all_inputs_at_once():
    JOINT_INPUTS.register("_test_diagonal", lambda size, n_variables, rng, scale=1.0: np.tile(rng.uniform(0, scale, (size, 1)), (1, n_variables)), overwrite=True)
    rows = draw_rows({"v1": {}, "v2": {}}, {"inputs": {"name": "_test_diagonal", "kwargs": {"scale": 2.0}}}, ["v1", "v2"], 50, np.random.default_rng(2))
    assert rows.shape == (50, 2) and np.array_equal(rows[:, 0], rows[:, 1]) and rows.max() <= 2.0


def test_constraints_mask_rows():
    x = np.array([[2.0, 1.0], [1.0, 2.0], [3.0, 3.0]])
    assert constraint_mask({"constraints": ["v1 > v2"]}, x, ["v1", "v2"]).tolist() == [True, False, False]
    assert constraint_mask({}, x, ["v1", "v2"]).all()


def test_realize_keeps_only_rows_that_satisfy_the_constraints(engine):
    box = {"sample_range": [0.0, 1.0], "sample_type": ["uni", "pos"]}
    cat = _catalog({"prepared": "v1 - v2", "vars": {"v1": dict(box), "v2": dict(box)}, "constraints": ["v1 > v2"]})
    rex = cat.realize(cat["e"], 200, np.random.default_rng(3), engine=engine)
    assert rex.x.shape == (200, 2) and (rex.x[:, 0] > rex.x[:, 1]).all() and (rex.y > 0).all()


def test_realize_draws_from_a_dist_and_rejects_a_malformed_spec(engine):
    cat = _catalog({"prepared": "v1 * v1", "vars": {"v1": {"dist": {"name": "normal", "kwargs": {"loc": 0.0, "scale": 1.0}}}}})
    rex = cat.realize(cat["e"], 500, np.random.default_rng(4), engine=engine)
    assert abs(rex.x.mean()) < 0.2 and np.allclose(rex.y.ravel(), rex.x.ravel() ** 2)
    bad = _catalog({"prepared": "v1", "vars": {"v1": {"dist": {"name": "no_such_distribution"}}}})
    with pytest.raises(CatalogEntryError):
        bad.realize(bad["e"], 10, np.random.default_rng(5), engine=engine)
