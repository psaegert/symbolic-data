"""Input specs of declarative catalog entries: dist configs and mixtures, joint samplers, derived variables
and row constraints; the fastsrb shorthand draws exactly as before."""
import numpy as np
import pytest

from symbolic_data.catalog import ProblemCatalog
from symbolic_data.distributions import fastsrb_dist
from symbolic_data.errors import CatalogEntryError
from symbolic_data.inputs import JOINT_INPUTS, constraint_mask, draw_rows, draw_variable, spec_for_test_rows
from symbolic_data.source import ProblemSource


@pytest.fixture(scope="module")
def engine():
    from simplipy import SimpliPyEngine
    return SimpliPyEngine.load("acj2-4-llm", install=True)


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
    assert constraint_mask({"row_constraints": ["v1 > v2"]}, x, ["v1", "v2"]).tolist() == [True, False, False]
    assert constraint_mask({}, x, ["v1", "v2"]).all()


def test_realize_keeps_only_rows_that_satisfy_the_constraints(engine):
    box = {"sample_range": [0.0, 1.0], "sample_type": ["uni", "pos"]}
    cat = _catalog({"prepared": "v1 - v2", "vars": {"v1": dict(box), "v2": dict(box)}, "row_constraints": ["v1 > v2"]})
    rex = cat.realize(cat["e"], 200, np.random.default_rng(3), engine=engine)
    assert rex.x.shape == (200, 2) and (rex.x[:, 0] > rex.x[:, 1]).all() and (rex.y > 0).all()


def test_realize_draws_from_a_dist_and_rejects_a_malformed_spec(engine):
    cat = _catalog({"prepared": "v1 * v1", "vars": {"v1": {"dist": {"name": "normal", "kwargs": {"loc": 0.0, "scale": 1.0}}}}})
    rex = cat.realize(cat["e"], 500, np.random.default_rng(4), engine=engine)
    assert abs(rex.x.mean()) < 0.2 and np.allclose(rex.y.ravel(), rex.x.ravel() ** 2)
    bad = _catalog({"prepared": "v1", "vars": {"v1": {"dist": {"name": "no_such_distribution"}}}})
    with pytest.raises(CatalogEntryError):
        bad.realize(bad["e"], 10, np.random.default_rng(5), engine=engine)


def test_a_folded_normal_is_a_normal_folded_onto_the_half_line():
    x = draw_variable({"dist": {"name": "folded_normal", "kwargs": {"loc": 2.0, "scale": 3.0 ** 0.5}}}, 20000, np.random.default_rng(6))
    folded = np.abs(np.random.default_rng(6).normal(2.0, 3.0 ** 0.5, 20000))
    assert (x >= 0).all() and np.allclose(x, folded)


def test_the_e2e_mixture_draws_whitened_clusters_of_the_functions_shape():
    rng = np.random.default_rng(8)
    for shape in ("gaussian", "uniform"):
        x = JOINT_INPUTS.get("e2e_mixture")(size=600, n_variables=3, rng=rng, shape=shape, k=2)
        assert x.shape == (600, 3)
        assert np.allclose(x.mean(axis=0), 0.0, atol=1e-9) and np.allclose(x.std(axis=0), 1.0)
    one = JOINT_INPUTS.get("e2e_mixture")(size=200, n_variables=1, rng=rng, shape="uniform", k=1, scale=4.0)
    assert np.isclose(one.std(), 4.0)
    with pytest.raises(ValueError):
        JOINT_INPUTS.get("e2e_mixture")(size=5, n_variables=1, rng=rng, shape="laplace")


def test_fastsrb_constraints_are_notes_not_row_constraints():
    # FastSRB entries carry their own `constraints` (e.g. ['v4 != 0'], with ^ for powers): never enforced
    assert constraint_mask({"constraints": ["v1 ^ 2 > 0"]}, np.ones((3, 1)), ["v1"]).all()


def _first_problem(cat: ProblemCatalog, engine, n_support: int, n_validation: int, seed: int = 0):
    src = ProblemSource({"catalog": cat, "sampling": {"n_support": n_support, "n_validation": n_validation, "noise": 0.0}},
                        simplipy_engine=engine, rng=np.random.default_rng(seed))
    return next(iter(src))


def test_the_test_rows_spec_replaces_only_how_a_variable_is_drawn():
    normal = {"name": "normal", "kwargs": {"loc": 0.0, "scale": 1.0}}
    v, m = spec_for_test_rows({"v1": {"name": "x", "units": [1, 0, 0, 0, 0], "sample_range": [0.0, 1.0], "sample_type": ["uni", "pos"]}},
                              {"test": {"vars": {"v1": {"dist": normal}}, "row_constraints": ["v1 > 0"]}})
    assert v == {"v1": {"name": "x", "units": [1, 0, 0, 0, 0], "dist": normal}} and m["row_constraints"] == ["v1 > 0"]
    with pytest.raises(ValueError):
        spec_for_test_rows({}, {"test": {"noise": 0.1}})


def test_a_test_block_draws_the_test_rows_from_its_own_spec(engine):
    cat = _catalog({"prepared": "v1 * 2", "vars": {"v1": {"sample_range": [0.0, 1.0], "sample_type": ["uni", "pos"]}},
                    "test": {"vars": {"v1": {"sample_range": [5.0, 6.0], "sample_type": ["uni", "pos"]}}}})
    p = _first_problem(cat, engine, 64, 32)
    assert p.x_support.shape == (64, 1) and p.x_validation.shape == (32, 1)
    assert p.x_support.max() <= 1.0 and p.x_validation.min() >= 5.0 and np.allclose(p.y_validation, 2 * p.x_validation)


def test_an_entry_without_a_test_block_splits_one_draw_as_before(engine):
    box = {"sample_range": [0.0, 1.0], "sample_type": ["uni", "pos"]}
    cat = _catalog({"prepared": "v1 + v2", "vars": {"v1": dict(box), "v2": dict(box)}})
    p = _first_problem(cat, engine, 48, 16, seed=3)
    one = cat.realize(cat["e"], 64, np.random.default_rng(3), engine=engine)
    assert np.array_equal(p.x_support, one.x[:48]) and np.array_equal(p.x_validation, one.x[48:])


def _counting_sampler():
    calls = []

    def sampler(size, n_variables, rng):
        calls.append(size)
        return np.column_stack([np.full(size, float(len(calls))), rng.uniform(-1.0, 1.0, size)])
    return sampler, calls


def test_test_rows_of_an_empty_test_block_are_a_fresh_point_set(engine):
    sampler, calls = _counting_sampler()
    JOINT_INPUTS.register("_test_counting", sampler, overwrite=True)
    cat = _catalog({"prepared": "v1 + v2", "vars": {"v1": {}, "v2": {}}, "inputs": {"name": "_test_counting"}, "test": {}})
    p = _first_problem(cat, engine, 40, 20)
    assert set(p.x_support[:, 0]) == {1.0} and set(p.x_validation[:, 0]) == {2.0} and calls == [40, 20]


def test_a_joint_sampler_short_of_valid_rows_is_drawn_again_as_a_whole(engine):
    sampler, calls = _counting_sampler()
    JOINT_INPUTS.register("_test_counting", sampler, overwrite=True)
    cat = _catalog({"prepared": "v1 + sqrt(v2)", "vars": {"v1": {}, "v2": {}}, "inputs": {"name": "_test_counting"}})
    rex = cat.realize(cat["e"], 100, np.random.default_rng(9), engine=engine)
    assert rex.x.shape == (100, 2) and (rex.x[:, 1] >= 0).all()
    assert len(set(rex.x[:, 0])) == 1 and len(calls) >= 2 and calls[-1] > 100     # one point set, the last and larger


def test_the_erbench_mixture_fits_in_the_window_and_tests_outside_it():
    erb = JOINT_INPUTS.get("erbench_mixture")
    rng = np.random.default_rng(11)
    for alpha, window in ((0.0, (0.0, 50.0)), (0.5, (25.0, 75.0)), (1.0, (50.0, 100.0))):
        fit = erb(size=500, n_variables=2, rng=rng, supports=[[0.0, 100.0], [3.0, 3.0]], alpha=alpha, region="fit")
        test = erb(size=500, n_variables=2, rng=rng, supports=[[0.0, 100.0], [3.0, 3.0]], alpha=alpha, region="test")
        assert window[0] <= fit[:, 0].min() and fit[:, 0].max() <= window[1]
        assert ((test[:, 0] <= window[0]) | (test[:, 0] >= window[1])).all()
        assert (fit[:, 1] == 3.0).all() and (test[:, 1] == 3.0).all()
    middle = erb(size=500, n_variables=1, rng=rng, supports=[[0.0, 100.0]], alpha=0.5, region="test")
    assert int((middle[:, 0] <= 25.0).sum()) == 250                     # the two sides split by length
    ints = erb(size=300, n_variables=1, rng=rng, supports=[[1, 500]], alpha=0.0, region="fit", dtype="int")
    assert np.array_equal(ints, np.round(ints)) and ints.min() >= 1 and ints.max() <= 251
    with pytest.raises(ValueError):
        erb(size=5, n_variables=1, rng=rng, supports=[[0.0, 1.0]], region="eval")


def test_an_erbench_dummy_takes_the_point_sets_own_values():
    x = JOINT_INPUTS.get("erbench_mixture")(size=200, n_variables=3, rng=np.random.default_rng(12),
                                            supports=[[0.0, 1.0], [10.0, 20.0]], alpha=0.0, dummies=1)
    assert x.shape == (200, 3) and set(x[:, 2]) <= set(x[:, :2].reshape(-1))
    assert (x[:, 2] < 1.0).any() and (x[:, 2] > 10.0).any()           # from both variables' values


def test_a_grid_draws_the_mesh_points_uniformly():
    x = draw_variable({"dist": {"name": "grid", "kwargs": {"low": -3.0, "high": 3.0, "step": 0.01}}}, 60000, np.random.default_rng(13))
    assert np.allclose(x * 100, np.round(x * 100)) and x.min() == -3.0 and x.max() == 3.0
    assert len(np.unique(x)) == 601 and abs(x.mean()) < 0.03


_UNIT = {"sample_range": [0.0, 1.0], "sample_type": ["uni", "pos"]}


def _numbers(tokens):
    out = []
    for t in tokens:
        try:
            out.append(float(t))
        except (TypeError, ValueError):
            pass
    return out


def test_per_run_values_are_drawn_anew_for_every_run_and_written_into_its_ground_truth(engine):
    cat = _catalog({"prepared": "1.5 * v1", "vars": {"v1": {"sample_range": [1.0, 2.0], "sample_type": ["uni", "pos"]}},
                    "per_run": {"params": {"c1": {"sample_range": [1.0, 2.0], "sample_type": ["uni", "pos"]}},
                                "prepared": "c1 * v1"}})
    rng, seen = np.random.default_rng(0), set()
    for _ in range(5):
        r = cat.realize(cat["e"], 32, rng, engine=engine)
        c1 = r.meta["per_run_values"]["c1"]
        assert 1.0 <= c1 <= 2.0 and np.allclose(r.y.ravel(), c1 * r.x[:, 0])      # one value for the whole run
        assert any(abs(v - c1) < 1e-9 for v in _numbers(r.expression))          # the run's ground truth holds it
        seen.add(round(c1, 9))
    assert len(seen) == 5                                                            # a new value every run


def test_per_run_values_drive_derived_inputs_and_are_shared_by_the_fit_and_test_rows(engine):
    # an input whose range the authors draw per run: lo ~ U(-10, 9), hi ~ U(lo + 1, 10)
    box = {"params": {"lo": {"sample_range": [-10.0, 9.0], "sample_type": ["uni", "pos"]}, "w": dict(_UNIT)},
           "latent": {"t": dict(_UNIT)}, "vars": {"v1": {"expr": "lo + (1 + w * (9 - lo)) * t"}}}
    cat = _catalog({"prepared": "v1", "vars": {"v1": {"name": "x", "sample_range": [0.0, 1.0], "sample_type": ["uni", "pos"]}},
                    "per_run": box, "test": {}})
    p = _first_problem(cat, engine, 64, 32, seed=4)
    lo, w = p.meta["per_run_values"]["lo"], p.meta["per_run_values"]["w"]
    hi = lo + 1 + w * (9 - lo)
    for x in (p.x_support, p.x_validation):                                         # the test rows use the same box
        assert x.min() >= lo - 1e-12 and x.max() <= hi + 1e-12
    assert hi <= 10.0 + 1e-12 and not (p.x_support.min() >= 0.0 and p.x_support.max() <= 1.0)


def test_a_per_run_block_is_checked(engine):
    for bad in ({"params": {"v1": dict(_UNIT)}}, {"params": {"exp": dict(_UNIT)}}, {"draws": {}}):
        cat = _catalog({"prepared": "v1", "vars": {"v1": dict(_UNIT)}, "per_run": bad})
        with pytest.raises(CatalogEntryError, match="per_run"):
            cat.realize(cat["e"], 8, np.random.default_rng(0), engine=engine)
