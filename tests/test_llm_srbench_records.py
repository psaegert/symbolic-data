"""LLM-SRBench's records (tools/catalog_build/llm_srbench_records.py): the notation repairs of the authors' formulas,
the exactness check that allows for float32 storage, and the data builder keeping the ground-truth kind a record
states."""
import importlib.util
import sys
from pathlib import Path

import numpy as np

BUILD = Path(__file__).resolve().parents[1] / "tools" / "catalog_build"


def _module(name):
    sys.path.insert(0, str(BUILD))
    spec = importlib.util.spec_from_file_location(name, BUILD / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


L = _module("llm_srbench_records")


def test_the_time_series_notation_becomes_plain_inputs_and_named_constants():
    text, unknown = L.spelled("-0.19*A(t)**2 + 0.19_z*A(t)**2/(0.75*A(t)**4 + 1) - 0.19_s*sin(t)", ["dA_dt", "t", "A"])
    assert text == "-0.19*A**2 + k_z*A**2/(0.75*A**4 + 1) - k_s*sin(t)"
    assert unknown == ["k_s", "k_z"]
    text, unknown = L.spelled("F0*sin(t) - beta*Abs(v(t))**0.333333333333333 - omega0**2*x(t)", ["dv_dt", "x", "t", "v"])
    assert text == "F0*sin(t) - beta*Abs(v)**(1/3) - omega0**2*x"
    assert unknown == ["F0", "beta", "omega0"]


def test_fitted_values_replace_the_named_constants():
    assert L.with_values("k_z*A**2 - omega0**2*x", {"k_z": 0.5, "omega0": 1.25}) == "(0.5)*A**2 - (1.25)**2*x"
    assert L.with_values("A**2", {}) == "A**2"


def test_rows_stored_as_float32_count_as_reproduced_where_the_formula_is_steep():
    rng = np.random.default_rng(0)
    x = np.concatenate([rng.uniform(2.0, 3.0, 999), [1 + 3e-7]])       # one row next to the pole of 1/(x - 1)
    rows = np.column_stack([1.0 / (x - 1), x]).astype(np.float32).astype(np.float64)   # target computed, then stored
    f = lambda x1: 1.0 / (x1 - 1)  # noqa: E731
    assert L.fvu(rows[:, 0], L.predict(f, rows, [])) > np.finfo(np.float32).eps     # the plain test fails ...
    assert L.unexplained(f, rows, []) == 0                                           # ... rounding explains every row
    g = lambda x1: 1.0 / (x1 - 1) + 1e-3  # noqa: E731
    assert L.unexplained(g, rows, []) == 999                                         # a wrong formula is caught


def test_a_record_that_states_its_ground_truth_kind_keeps_it():
    B = _module("build_data_catalogs")
    R = sys.modules["data_catalog_readers"]
    rng = np.random.default_rng(1)
    a = rng.uniform(1.0, 2.0, 200)
    y = a ** 2 * (1 + 1e-3 * rng.standard_normal(200))                # the formula misses by more than float32 eps

    def kind(prob):
        rows = R.Rows(a[:, None], y, ["a"], R.NONE, "none")
        rows.finish()
        return B.with_ground_truth(rows, prob, B.Formulas())["gt_kind"]
    assert kind({"id": "t", "formula": "a**2"}) == "reference"
    assert kind({"id": "t", "formula": "a**2", "gt_kind": "exact"}) == "exact"
