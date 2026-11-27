"""The catalog builders' gates and the 2026-11-28 audit fixes (tools/catalog_build): the numeric raw-vs-prepared check
of the formula builders, the reference-law gate of the data builder, and the transcription fixes (ERBench's
two-argument log, HeuristicLab's AircraftLift, cp3-bench's reference formulas, HEP LLM-SR's target)."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

BUILD = Path(__file__).resolve().parents[1] / "tools" / "catalog_build"


def _module(name, where=BUILD):
    sys.path.insert(0, str(BUILD))
    sys.path.insert(0, str(BUILD.parent))
    spec = importlib.util.spec_from_file_location(name, where / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


G = _module("formula_gate")
P = _module("paths")


# ------------------------------------------------------------------------------------------------ the formula gate
BUGGY = "0.4342944819032518*log(v1*(1-v2)/(v2*(1-v1)))/0.4342944819032518*log(v3)"     # the six ERBench problems
RAW = "log(x_0*(1 - x_1)/(x_1*(1 - x_0)), 10)/log(x_2, 10)"
BOX = [(0.01, 0.99), (0.01, 0.99), (1.01, 10.0)]


def test_the_gate_refuses_a_prepared_formula_that_lost_its_parentheses():
    with pytest.raises(G.GateError, match="differs from the raw formula"):
        G.gate("erbench-eponymous", "eponymous_191_alpha0", RAW, ["x_0", "x_1", "x_2"], BUGGY, BOX)


def test_the_fixed_erbench_rewrite_passes_the_gate():
    E = _module("build_erbench_catalogs")
    prepared = E.prepared_of(RAW)
    assert prepared == "(0.4342944819032518*log(v1*(1-v2)/(v2*(1-v1))))/(0.4342944819032518*log(v3))"
    result = G.gate("erbench-eponymous", "eponymous_191_alpha0", RAW, ["x_0", "x_1", "x_2"], prepared, BOX)
    assert result["ok"] and result["checked"] > 100


def test_the_gate_lets_a_listed_difference_through_with_its_reason(monkeypatch):
    monkeypatch.setattr(G, "allowed", lambda: {("cat", "p"): "the authors fit ln y"})
    result = G.gate("cat", "p", "x**2", ["x"], "2*log(v1)", [(1.0, 3.0)])
    assert result["ok"] is False and result["allowed"] == "the authors fit ln y"


def test_the_gate_reports_an_entry_it_cannot_evaluate_instead_of_passing_it():
    result = G.compare("sqrt(-1 - x**2)", ["x"], "sqrt(-1-v1**2)", [(0.0, 1.0)])
    assert result["ok"] is None and result["checked"] == 0


def test_the_gate_reads_inputs_that_repeat_a_name_as_unused():
    result = G.compare("B*x", ["B", "x", "B"], "v1*v2", [(1.0, 2.0)] * 3)
    assert result["ok"]


# ------------------------------------------------------------------------------- the formula builder's record fixes
def test_heuristiclab_aircraft_lift_has_the_authors_sign_and_hep_poses_the_authors_target():
    F = _module("build_formula_catalogs")
    lift = F.fixed({"set": "fresh-46", "id": "AircraftLift", "formula": "x1*(x2 - 2) + x3*x4*x5/x6"})
    assert lift["formula"] == "x1*(x2 + 2) + x3*x4*x5/x6" and "why" not in lift
    hep = F.fixed({"set": "fresh-127", "id": "ee_mumu_angular_id", "formula": "alpha**2/(4*s)*(1 + cos_theta**2)",
                   "constants": {"alpha": 0.00755, "s": 1e6}})
    prepared = F.prepared_of(hep["formula"], ["cos_theta"], F.named_constants(hep))
    value = eval(prepared.replace("v1", "0.0"))                                             # noqa: S307
    assert value == pytest.approx(2 * np.pi * 0.3893793721e9 * 0.00755 ** 2 / 4e6, rel=1e-12)
    assert value == pytest.approx(0.0349, rel=2e-3)                                       # the authors' fitted 0.03492


# ---------------------------------------------------------------------------------- the data builder's ground truth
B = _module("build_data_catalogs")
R = _module("data_catalog_readers")


def _noisy(prob, catalog="cat", noise=1e-2):
    rng = np.random.default_rng(0)
    a = rng.uniform(1.0, 2.0, 200)
    rows = R.Rows(a[:, None], a ** 2 * (1 + noise * rng.standard_normal(200)), ["a"], R.NONE, "none", eq_id="t")
    rows.finish()
    return B.with_ground_truth(rows, {"id": "t", "formula": "a**2", **prob}, B.Formulas(), catalog), rows.meta


def test_a_formula_that_misses_its_rows_without_a_stated_reason_is_refused():
    with pytest.raises(ValueError, match="no reason states it as a reference law"):
        _noisy({})


def test_a_reference_law_needs_its_reason_and_meets_its_bound(monkeypatch):
    gt, meta = _noisy({"gt_reason": "the authors add 1% noise", "gt_fvu_bound": 0.01})
    assert gt["gt_kind"] == "reference" and meta["reference_reason"] == "the authors add 1% noise"
    with pytest.raises(ValueError, match="above its bound"):
        _noisy({"gt_reason": "the authors add 1% noise", "gt_fvu_bound": 1e-6})
    monkeypatch.setattr(B.paths, "reference_laws", lambda: {("cat", "t"): {"reason": "listed", "fvu_bound": 0.01}})
    assert _noisy({})[0]["gt_kind"] == "reference"


def test_an_exact_formula_needs_no_reason():
    assert _noisy({}, noise=0.0)[0]["gt_kind"] == "exact"


def test_the_three_cp3_bench_formulas_are_corrected_and_every_listed_reference_law_has_a_bound():
    assert {k for k in B.FORMULA_FIXES} == {("fresh-54", "C3h"), ("fresh-54", "C6a"), ("fresh-54", "C6c")}
    t = np.linspace(0.0, 3.0, 7)
    m1 = np.linspace(0.1, 1.5, 7)
    import sympy
    c6c = sympy.lambdify(["t", "m1"], sympy.sympify(B.FORMULA_FIXES[("fresh-54", "C6c")][0]))
    M, eta = 2 * m1, 0.25                              # the authors' data: m2 = m1
    phase = -3 * t ** 2 / M + 20 * t / M
    amp = -4 * M ** 2 * (-3 * t + 10) ** (-10 / 3) + 2 * (-3 * t + 10) ** (2 / 3)
    want = -(2 * M * eta / 5) * (amp * np.cos(phase) + 4 * M * (-3 * t + 10) ** (-4 / 3) * np.sin(phase))
    np.testing.assert_allclose(c6c(t, m1), want, rtol=1e-12)
    for (catalog, eq_id), law in P.reference_laws().items():
        assert law["reason"] and 0 < law["fvu_bound"] < 1, (catalog, eq_id)
