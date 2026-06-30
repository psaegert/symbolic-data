"""ProblemSource holdout decontamination is SKELETON-LEVEL + variable-canonical (sd 0.8.0).

Decontamination replaces the old internal skeleton-pool holdout: a generated problem is dropped when
its SKELETON (constants collapsed, variables canonicalized by ``normalize_skeleton``) matches one in
the excluded catalog -- which may be declarative OR generative, resolved by ref. The critical case is
cross-namespace: a held-out FastSRB-style ``v1..`` expression must drop a generated ``x1..`` skeleton
of the same structure, or training silently leaks the eval set.
"""
import os
import tempfile
from itertools import islice
from pathlib import Path

import yaml

from symbolic_data import ProblemSource

RECIPE = Path(__file__).resolve().parent.parent / "configs" / "test" / "catalog_train.yaml"


def _frozen_spec(skeletons: list[list[str]], name: str = "probe") -> str:
    cfg = yaml.safe_load(RECIPE.read_text(encoding="utf-8"))
    cfg["type"] = "lample_charton"
    cfg["name"] = name
    cfg["skeletons"] = skeletons
    path = os.path.join(tempfile.mkdtemp(), f"{name}.yaml")
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle)
    return path


def _declarative(expressions: dict, name: str = "decl") -> str:
    doc = {"metadata": {"name": name, "version": 1}, "expressions": expressions}
    path = os.path.join(tempfile.mkdtemp(), f"{name}.yaml")
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(doc, handle)
    return path


def test_self_exclude_generative_drops_everything():
    # A frozen catalog excluded against ITSELF: every sampled problem's skeleton is in the exclude set.
    spec = _frozen_spec([["sin", "x1"], ["+", "x1", "x2"]], name="selfx")
    src = ProblemSource({"catalog": spec, "sampling": {"n_support": 8, "n_validation": 0, "noise": 0.0},
                         "holdouts": [{"exclude": spec}]})
    kept = [p for p in islice(iter(src), 8) if not p.is_placeholder]
    assert kept == []


def test_cross_namespace_declarative_exclude_drops_matching_structure():
    # The leak-risk case: a declarative catalog holding sin(v1) must drop a generated ('sin','x1')
    # skeleton -- normalize_skeleton canonicalizes v1 -> x1, so the structures match across namespaces.
    spec = _frozen_spec([["sin", "x1"], ["cos", "x1"]], name="crossns")
    excl = _declarative({"E1": {"raw": "sin(v1)", "prepared": "sin(v1)", "vars": {"v1": {}}}}, name="declv")
    src = ProblemSource({"catalog": spec, "sampling": {"n_support": 8, "n_validation": 0, "noise": 0.0},
                         "holdouts": [{"exclude": excl}]})
    kept_skeletons = {tuple(p.skeleton) for p in islice(iter(src), 8) if not p.is_placeholder}
    assert ("sin", "x1") not in kept_skeletons          # the v1.. exclusion dropped the x1.. match
    assert ("cos", "x1") in kept_skeletons              # the non-matching structure survives


def test_open_generative_ref_infers_generate_mode():
    cfg = yaml.safe_load(RECIPE.read_text(encoding="utf-8"))
    cfg["type"] = "lample_charton"
    path = os.path.join(tempfile.mkdtemp(), "open.yaml")
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle)
    bounded = ProblemSource({"catalog": path, "sampling": {"size": 4}})
    assert bounded.mode == "generate" and bounded.size_hint() == 4
    unbounded = ProblemSource({"catalog": path, "sampling": {}})
    assert unbounded.mode == "generate" and unbounded.size_hint() is None


def test_frozen_generative_ref_infers_set_mode():
    spec = _frozen_spec([["sin", "x1"], ["+", "x1", "x2"], ["cos", "x2"]], name="frozenset")
    src = ProblemSource({"catalog": spec, "sampling": {"n_support": 8, "n_validation": 0}})
    assert src.mode == "set" and src.size_hint() == 3      # bounded by the 3 frozen skeletons


def test_frozen_set_iteration_is_BOUNDED():
    # Regression: a frozen generative catalog in set mode must iterate its fixed skeleton set ONCE and
    # TERMINATE (each skeleton once) -- not stream unbounded. Otherwise list(source) / srbf val eval
    # would hang despite a finite size_hint.
    spec = _frozen_spec([["sin", "x1"], ["+", "x1", "x2"], ["cos", "x2"]], name="bounded")
    src = ProblemSource({"catalog": spec, "sampling": {"n_support": 8, "n_validation": 0, "noise": 0.0}})
    problems = list(src)                                   # must terminate
    assert len(problems) == 3 == src.size_hint()
    assert {tuple(p.skeleton) for p in problems} == {("sin", "x1"), ("+", "x1", "x2"), ("cos", "x2")}
