"""The registration-side sign closure (2026-11-05): a held-out law's family leaves its fittable constants free, so a
law registered with one sign of a constant must also hold out the variant with the other sign -- even where the
canonical form writes that sign into structure (here a sum becomes a difference)."""
import os
import warnings

import numpy as np
import pytest
import yaml

import symbolic_data.generative as generative
from symbolic_data.catalog import ProblemCatalog
from symbolic_data.generative import HOLDOUT_EFFORT, HOLDOUT_SIMPLIFY_MODE, LampleChartonCatalog

# SOOSE-WC-031. Measured before the closure: flipping the sign of 2.88 was not held out.
LAW = "v3 + 1/(2.8807879935653626*v2 - log(4.901810635918965*v1 + v2))"
VARIANT = "x3 - 1/(2.8807879935653626*x2 + log(4.901810635918965*x1 + x2))"


@pytest.fixture(scope="module")
def catalog():
    cfg_path = os.path.join(os.path.dirname(__file__), "..", "configs", "test", "catalog_train.yaml")
    config = yaml.safe_load(open(cfg_path, encoding="utf-8"))
    config["holdout_pools"] = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return LampleChartonCatalog.from_config(config)


def _pool(prepared: str) -> ProblemCatalog:
    entry = {"prepared": prepared, "vars": {f"v{i}": {"sample_range": [1, 5]} for i in (1, 2, 3)}}
    return ProblemCatalog.from_yaml({"metadata": {"name": "sign-closure", "version": 1}, "expressions": {"law": entry}})


def _held_out(catalog, infix: str) -> bool:
    engine = catalog.simplipy_engine
    tokens = ["pow" if t in ("**", "^") else t for t in engine.infix_to_prefix(infix)]
    canonical = list(engine.simplify(tokens, mode=HOLDOUT_SIMPLIFY_MODE, effort=HOLDOUT_EFFORT))
    return catalog.is_held_out(canonical, [], n_variables=3)


@pytest.mark.parametrize("patterns, variant_held_out", [(0, False), (generative.HOLDOUT_SIGN_PATTERNS, True)])
def test_a_sign_variant_of_a_held_out_law_is_held_out(catalog, monkeypatch, patterns, variant_held_out):
    """Without the closure (0 patterns) the variant escapes: the gate can fail. With it, the variant is held out."""
    monkeypatch.setattr(generative, "HOLDOUT_SIGN_PATTERNS", patterns)
    catalog.clear_holdouts()
    try:
        catalog.register_holdout_pool(_pool(LAW))
        assert _held_out(catalog, LAW.replace("v", "x"))
        assert _held_out(catalog, VARIANT) is variant_held_out
    finally:
        catalog.clear_holdouts()


def test_only_fittable_constants_change_sign(catalog):
    """The exponent 2 and the reciprocal's 1 are structure; only 2.88... is a fittable constant, so the one variant
    is the canonical form with that constant negated and nothing else changed."""
    engine = catalog.simplipy_engine
    tokens = engine.infix_to_prefix("x1**2 + 2.8807879935653626*x2 + 1/x3")
    canonical = list(engine.simplify(["pow" if t == "**" else t for t in tokens],
                                     mode=HOLDOUT_SIMPLIFY_MODE, effort=HOLDOUT_EFFORT))
    site = canonical.index("2.8807879935653626")
    assert catalog.holdout_sign_variants(tokens) == [canonical[:site] + ["neg"] + canonical[site:]]
    assert catalog.holdout_sign_variants(engine.infix_to_prefix("x1**2 + 1/x3")) == []


def test_many_constants_get_a_seeded_bounded_sample(catalog):
    """Past 2**k - 1 > HOLDOUT_SIGN_PATTERNS the patterns are every single flip plus seeded random ones: bounded, and
    the same in every process."""
    law = " + ".join(f"{1.5 + i}*x1**{i + 1}" for i in range(9))
    tokens = catalog.simplipy_engine.infix_to_prefix(law)
    first = catalog.holdout_sign_variants(tokens)
    assert len(first) == generative.HOLDOUT_SIGN_PATTERNS
    assert first == catalog.holdout_sign_variants(tokens)
    assert len({tuple(v) for v in first}) == len(first)


def test_the_closure_adds_keys_only_at_registration(catalog):
    """Registering with the closure holds out strictly more than without it; the law's own key is in both."""
    catalog.clear_holdouts()
    try:
        catalog.register_holdout_pool(_pool(LAW))
        with_closure = set(catalog.holdout_manager.skeleton_hashes)
    finally:
        catalog.clear_holdouts()
    previous = generative.HOLDOUT_SIGN_PATTERNS
    generative.HOLDOUT_SIGN_PATTERNS = 0
    try:
        catalog.register_holdout_pool(_pool(LAW))
        without = set(catalog.holdout_manager.skeleton_hashes)
    finally:
        generative.HOLDOUT_SIGN_PATTERNS = previous
        catalog.clear_holdouts()
    assert without < with_closure


def test_a_variant_that_loses_a_variable_is_not_registered(catalog):
    """Flipping one of the law's two 8.43652s cancels (a - x1)/(x1 - a) to -1 and leaves a multiple of x2: a collapse
    no member of the family with independent constants reaches. The flips produce it; the closure does not register
    it."""
    law = "(8.43652 - v1)/(v1 + 8.43652) * v2"
    variants = catalog.holdout_sign_variants(catalog.simplipy_engine.infix_to_prefix(law))
    keys = [catalog.holdout_family_prototype(variant, strict=True) for variant in variants]
    assert any(generative._count_variables(key) == 1 for key in keys)
    catalog.clear_holdouts()
    try:
        catalog.register_holdout_pool(_pool(law))
        assert all(generative._count_variables(list(key)) == 2 for key in catalog.holdout_manager.skeleton_hashes)
    finally:
        catalog.clear_holdouts()


def _value(catalog, tokens, x1):
    """A prefix expression over x1 at one point, through the engine's own realization."""
    from simplipy.utils import codify
    engine = catalog.simplipy_engine
    code = codify(engine.prefix_to_infix(engine.operators_to_realizations(list(tokens)), realization=True), ["x1"])
    return float(engine.code_to_lambda(code)(x1))


@pytest.mark.parametrize("law, negated", [
    ("exp(-2)*x1", lambda x: -np.exp(-2) * x),
    ("cos(-2)*x1", lambda x: -np.cos(2) * x),
    ("cosh(-2)*x1", lambda x: -np.cosh(2) * x),
    ("x1 + exp(-2)", lambda x: x - np.exp(-2)),
])
def test_a_ground_constant_changes_sign_as_a_whole(catalog, law, negated):
    """The canonical form can keep a ground constant whole (exp(-2), cos(2)): masking makes it ONE <constant>, so its
    sign variant negates the whole constant. Flipping the literal inside left cos/cosh unchanged and exp positive."""
    variants = catalog.holdout_sign_variants(catalog.simplipy_engine.infix_to_prefix(law))
    assert len(variants) == 1
    assert _value(catalog, variants[0], 1.3) == pytest.approx(negated(1.3))


def test_any_leaf_that_is_not_a_literal_is_a_variable():
    """alpha + exp(-2): the site is exp(-2), not the whole sum. A name pattern (x/v + digits) read alpha as ground and
    flipped the sum, which leaves the family."""
    arity = {"+": 2, "*": 2, "exp": 1}
    tokens = ["+", "alpha", "exp", "-2"]
    assert generative._ground_sign_sites(tokens, arity, literals={3}, fittable={3}) == [2]
    # a typed literal (here: not fittable) is still a literal, not a variable: its ground subtree is no site
    assert generative._ground_sign_sites(["*", "alpha", "exp", "2"], arity, literals={3}, fittable=set()) == []
