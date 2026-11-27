"""A ground truth is kept as written: ``sqrt`` stays ``sqrt`` when the SimpliPy engine's config defines it.

SimpliPy 0.15 configs may declare DEFINED FUNCTIONS (``defined_functions:``): ``sqrt``, ``cot``, ``sec``, ``csc``,
``acot``, ``asec`` and ``acsc``. Such an engine reads a defined function at its own head, prices it as written and
evaluates it as its definition (``sqrt`` as ``rootn(u, 2)``). A truth compiled with it keeps the formula's ``sqrt``;
with an engine whose config defines nothing, the square root is written ``rootn(u, 2)`` as before. The skeleton and
the targets are the same either way. Training generation is unchanged: it writes the generating vocabulary.

The two engines below are the small test engine (acj2-4-llm, its rules unchanged), with and without the section.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest
import yaml

from symbolic_data import LampleChartonCatalog, Problem, ProblemCatalog, ProblemSource
from symbolic_data._evaluation import compile_expression
from symbolic_data.errors import NoValidSampleFoundError
from symbolic_data.token_ops import desugar_sqrt, spell_square_roots, written_arity

ROOT = Path(__file__).resolve().parent.parent
CATALOGS = ROOT / "assets" / "catalogs"

#: The section as SimpliPy 0.15's released engine configs declare it.
DEFINED_FUNCTIONS = {
    "sqrt": {"arity": 1, "definition": "rootn(_0, 2)"},
    "cot": {"arity": 1, "definition": "inv(tan(_0))"},
    "acot": {"arity": 1, "definition": "atan(inv(_0))"},
    "sec": {"arity": 1, "definition": "inv(cos(_0))"},
    "csc": {"arity": 1, "definition": "inv(sin(_0))"},
    "asec": {"arity": 1, "definition": "acos(inv(_0))"},
    "acsc": {"arity": 1, "definition": "asin(inv(_0))"},
}


def _engine_config(tmp_path_factory, defined: bool) -> str:
    """A copy of the small test engine whose config carries the section (``defined``) or not; its config path."""
    from simplipy import asset_manager

    source = Path(asset_manager.get_path("acj2-4-llm", install=True)).parent
    target = tmp_path_factory.mktemp("defined" if defined else "undefined") / "engine"
    shutil.copytree(source, target)
    config = yaml.safe_load((target / "config.yaml").read_text())
    config.pop("defined_functions", None)
    if defined:
        config["defined_functions"] = DEFINED_FUNCTIONS
    (target / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    return str(target / "config.yaml")


@pytest.fixture(scope="module")
def defining_config(tmp_path_factory) -> str:
    return _engine_config(tmp_path_factory, defined=True)


@pytest.fixture(scope="module")
def engine_defining_sqrt(defining_config):
    from simplipy import SimpliPyEngine
    return SimpliPyEngine.from_config(defining_config)


@pytest.fixture(scope="module")
def engine_without_definitions(tmp_path_factory):
    from simplipy import SimpliPyEngine
    return SimpliPyEngine.from_config(_engine_config(tmp_path_factory, defined=False))


def _fastsrb_sqrt_entries() -> list[tuple[str, dict]]:
    with open(CATALOGS / "fastsrb.yaml") as file:
        entries = yaml.safe_load(file)["expressions"]
    return [(eq_id, entry) for eq_id, entry in entries.items() if "sqrt" in entry["prepared"]]


def _values(compiled: dict, seed: int) -> list[np.ndarray]:
    """The compiled truth on a positive and on a mixed-sign draw (outside the domain the values are NaN)."""
    rng = np.random.default_rng(seed)
    n = len(compiled["variable_order"])
    out = []
    for x in (rng.uniform(0.1, 5.0, (64, n)), rng.uniform(-5.0, 5.0, (64, n))):
        with np.errstate(all="ignore"):
            out.append(np.broadcast_to(np.asarray(compiled["callable"](*x.T), dtype=float), (64,)))
    return out


# ---- the engine's heads ---------------------------------------------------------------------------------------------
def test_written_arity_adds_the_defined_functions(engine_defining_sqrt, engine_without_definitions):
    assert written_arity(engine_without_definitions) == engine_without_definitions.operator_arity
    defined = written_arity(engine_defining_sqrt)
    assert defined == {**engine_defining_sqrt.operator_arity, **{name: 1 for name in DEFINED_FUNCTIONS}}
    assert "sqrt" not in engine_defining_sqrt.operator_arity        # a defined function is no operator


def test_an_engine_before_the_section_reads_its_operators():
    class Engine:     # a SimpliPy without `defined_functions`
        operator_arity = {"rootn": 2, "*": 2}

    assert written_arity(Engine()) == {"rootn": 2, "*": 2}


def test_spell_square_roots_keeps_sqrt_only_where_the_engine_defines_it(engine_defining_sqrt, engine_without_definitions):
    tokens = ["/", "exp", "x1", "sqrt", "*", "2", "x2"]
    assert spell_square_roots(tokens, written_arity(engine_defining_sqrt)) == tokens
    rewritten = spell_square_roots(tokens, written_arity(engine_without_definitions))
    assert rewritten == ["/", "exp", "x1", "rootn", "*", "2", "x2", "2"]
    assert rewritten == desugar_sqrt(tokens, engine_without_definitions.operator_arity)


def test_spell_square_roots_walks_as_desugar_sqrt_without_the_definition(engine_without_definitions):
    with pytest.raises(ValueError):
        spell_square_roots(["sqrt", "x1", "x2"], written_arity(engine_without_definitions))


# ---- the truth path ---------------------------------------------------------------------------------------------------
def test_a_truth_keeps_sqrt_as_written_with_a_defining_engine(engine_defining_sqrt):
    compiled = compile_expression(engine_defining_sqrt, "t", "v1 * sqrt(v2 + 3)", {"v1": {}, "v2": {}})
    assert compiled["expression"] == ["*", "x1", "sqrt", "+", "x2", "3"]
    assert compiled["constants"] == [3.0]                    # a written sqrt has no root-index literal
    assert "sqrt" not in compiled["prefix"]                  # the skeleton is the canonical form


def test_a_truth_writes_rootn_without_the_definition(engine_without_definitions):
    compiled = compile_expression(engine_without_definitions, "t", "v1 * sqrt(v2 + 3)", {"v1": {}, "v2": {}})
    assert compiled["expression"] == ["*", "x1", "rootn", "+", "x2", "3", "2"]
    assert compiled["constants"] == [3.0, 2.0]


def test_written_truths_keep_the_skeleton_and_every_value(engine_defining_sqrt, engine_without_definitions):
    """Every fastsrb law spelled with sqrt: the skeleton, its infix and the targets are identical with either engine
    (bit for bit, NaN where the law is undefined); only the spelling of the truth differs."""
    entries = _fastsrb_sqrt_entries()
    assert len(entries) > 25
    for eq_id, entry in entries:
        written = compile_expression(engine_defining_sqrt, eq_id, entry["prepared"], entry["vars"])
        rootn = compile_expression(engine_without_definitions, eq_id, entry["prepared"], entry["vars"])
        assert "sqrt" in written["expression"] and "sqrt" not in rootn["expression"], eq_id
        assert desugar_sqrt(written["expression"], engine_without_definitions.operator_arity) == rootn["expression"], eq_id
        assert written["prefix"] == rootn["prefix"], eq_id
        assert written["normalized_infix"] == rootn["normalized_infix"], eq_id
        n_roots = written["expression"].count("sqrt")
        assert len(rootn["constants"]) - len(written["constants"]) == n_roots, eq_id
        for a, b in zip(_values(written, 0), _values(rootn, 0)):
            assert np.array_equal(a, b, equal_nan=True), eq_id


def test_a_truth_without_defined_heads_is_compiled_as_before(engine_defining_sqrt, engine_without_definitions):
    for prepared in ("v1 ** 2 + sin(v2)", "exp(-v1) / (1 + v2)"):
        a = compile_expression(engine_defining_sqrt, "t", prepared, {"v1": {}, "v2": {}})
        b = compile_expression(engine_without_definitions, "t", prepared, {"v1": {}, "v2": {}})
        assert {k: v for k, v in a.items() if k != "callable"} == {k: v for k, v in b.items() if k != "callable"}
        for x, y in zip(_values(a, 1), _values(b, 1)):
            assert np.array_equal(x, y, equal_nan=True)


def test_a_realized_problem_keeps_the_written_truth(engine_defining_sqrt, engine_without_definitions):
    """The set-mode source (what the frozen draws render): same rows and targets, the truth as written."""
    nguyen = str(CATALOGS / "nguyen.yaml")
    sampling = {"n_support": 16, "n_validation": 8, "method": "iterate", "noise": 0.0}
    written = list(ProblemSource({"catalog": nguyen, "sampling": sampling}, simplipy_engine=engine_defining_sqrt,
                                 rng=np.random.default_rng(3)))
    rootn = list(ProblemSource({"catalog": nguyen, "sampling": sampling}, simplipy_engine=engine_without_definitions,
                               rng=np.random.default_rng(3)))
    assert len(written) == len(rootn) == 12
    for a, b in zip(written, rootn):
        assert a.eq_id == b.eq_id
        for field in ("x_support", "y_support", "x_validation", "y_validation"):
            assert np.array_equal(getattr(a, field), getattr(b, field)), (a.eq_id, field)
        assert a.skeleton == b.skeleton
    a, b = next((a, b) for a, b in zip(written, rootn) if "sqrt" in a.expression)      # Nguyen-8, sqrt(v1)
    assert list(a.expression) == ["sqrt", "x1"] and list(b.expression) == ["rootn", "x1", "2"]
    assert a.complexity == b.complexity - 1


# ---- training generation stays in the generating vocabulary -------------------------------------------------------
def _training_catalog(engine_config: str) -> LampleChartonCatalog:
    with open(ROOT / "configs" / "test" / "catalog_train.yaml", encoding="utf-8") as file:
        cfg = yaml.safe_load(file)
    cfg["simplipy_engine"] = engine_config
    return LampleChartonCatalog.from_config(cfg)


def test_training_skeletons_never_carry_a_defined_head(defining_config):
    catalog = _training_catalog(defining_config)
    rng = np.random.default_rng(0)
    sampled = 0
    for _ in range(200):
        try:
            skeleton, _, _ = catalog.sample_skeleton(new=True, decontaminate=False, rng=rng)
        except NoValidSampleFoundError:
            continue
        assert not set(skeleton) & set(DEFINED_FUNCTIONS), skeleton
        sampled += 1
    assert sampled >= 100


def test_the_training_holdout_writes_rootn_with_a_defining_engine(defining_config, tmp_path):
    """A frozen truth that keeps sqrt (written with a defining engine) is held out in the generating vocabulary."""
    x = np.linspace(0.5, 2.5, 16)
    problem = Problem.from_data(x, np.sqrt(x), expression=["sqrt", "x1"], eq_id="rooted",
                                meta={"alternate_renderings": ["sqrt(v1)"]})
    npz = str(ProblemCatalog.from_problems([problem], name="sqrt-probe").save(tmp_path / "sqrt-probe"))
    catalog = _training_catalog(defining_config)
    catalog.register_holdout_pool(npz)
    try:
        assert ("x1",) in catalog.holdout_skeletons
        assert not {token for proto in catalog.holdout_skeletons for token in proto} & set(DEFINED_FUNCTIONS)
        assert len(catalog.holdout_y) >= 1
    finally:
        catalog.clear_holdouts()
