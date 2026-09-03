"""The 2026-08-27 holdout fold: simplify BEFORE masking, at permissive/effort=4, with the
'__unevaluable__' sentinel retired.

Each case is a family the OLD mask-then-simplify order split into two prototypes, so a
generated draw of one spelling escaped a holdout registered under the other.
"""
import os
import warnings

import pytest
import yaml

from symbolic_data.generative import (
    HOLDOUT_EFFORT, HOLDOUT_SIMPLIFY_MODE, LampleChartonCatalog)


@pytest.fixture(scope="module")
def catalog():
    cfg_path = os.path.join(os.path.dirname(__file__), "..", "configs", "test", "catalog_train.yaml")
    config = yaml.safe_load(open(cfg_path, encoding="utf-8"))
    config["holdout_pools"] = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return LampleChartonCatalog.from_config(config)


def test_fold_runs_at_the_ruled_strength():
    assert HOLDOUT_SIMPLIFY_MODE == "permissive"
    assert HOLDOUT_EFFORT == 4


@pytest.mark.parametrize("label,left,right", [
    # like-term / like-factor collection: unreachable once literals are masked, because the
    # AC core can no longer do exact rational arithmetic on a symbolic <constant>
    ("x+x == 2x", ["+", "x1", "x1"], ["*", "2", "x1"]),
    ("x*x == x^2", ["*", "x1", "x1"], ["pow", "x1", "2"]),
    ("x-x == 0", ["-", "x1", "x1"], ["0"]),
    ("(x*y)/(x*y) == 1", ["/", "*", "x1", "x2", "*", "x1", "x2"], ["1"]),
    # distributivity: recovered ONLY by simplifying before masking
    ("2(x+1) == 2x+2", ["*", "2", "+", "x1", "1"], ["+", "*", "2", "x1", "2"]),
    # already held under the old order; must not regress
    ("c*x == x*c", ["*", "3.0", "x1"], ["*", "x1", "7.5"]),
    ("sin(2x) == sin(x)", ["sin", "*", "2.0", "x1"], ["sin", "x1"]),
    ("AC commuted", ["/", "*", "x1", "x2", "x3"], ["/", "*", "x2", "x1", "x3"]),
])
def test_same_family_shares_one_prototype(catalog, label, left, right):
    assert catalog.holdout_family_prototype(left) == catalog.holdout_family_prototype(right), label


def test_subfamily_is_not_merged(catalog):
    """(x+c)^2 vs x^2+ax+b is a CONTAINMENT, not an equality, and must stay separate.

    Closing it is impossible here in two independent ways: the miner's mu(T) < mu(S) bound
    forbids an expansion rule (no rule in the library grows), and the factoring direction
    needs the constraint b = a^2/4 between two constant leaves, which the matcher binds
    independently. Measured residual leak from this class: <= 0.33% of accepted draws.
    """
    square = catalog.holdout_family_prototype(["pow", "+", "x1", "1.0", "2"])
    expanded = catalog.holdout_family_prototype(
        ["+", "+", "pow", "x1", "2", "*", "2.0", "x1", "1.0"])
    assert square is not None and expanded is not None
    assert square != expanded


def test_registration_emits_no_unevaluable_sentinel(catalog):
    """The sentinel identifies 'unevaluable on grid g', not any law: registering it made every
    candidate that also fails on grid g match (measured: 2 keys of 9,499 -> 4.57% of draws)."""
    config = dict(catalog.config) if hasattr(catalog, "config") else None
    cfg_path = os.path.join(os.path.dirname(__file__), "..", "configs", "test", "catalog_train.yaml")
    config = yaml.safe_load(open(cfg_path, encoding="utf-8"))
    config["holdout_pools"] = ["fastsrb"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        built = LampleChartonCatalog.from_config(config)
    sentinels = [k for k in built.holdout_manager.expression_images
                 if isinstance(k, tuple) and len(k) > 1 and k[1] == "__unevaluable__"]
    assert sentinels == []


def test_prototype_is_idempotent(catalog):
    """Registration and probing both call this; a non-idempotent key would desynchronize them."""
    for tokens in (["+", "x1", "x1"], ["/", "*", "x1", "x2", "x3"], ["sin", "*", "2.0", "x1"]):
        once = catalog.holdout_family_prototype(tokens)
        assert catalog.holdout_family_prototype(once) == once


# --- the power spelling (2026-09-03) -------------------------------------------------------------
# infix_to_prefix spells powers as ``**``; the engine's simplify rejects that token, and the
# registration side used to swallow the error and register the prototype of the RAW spelling.
# A sampler draw of the same law arrives AC-canonical, so the two prototypes differed wherever
# the canon restructures the law: 584 of 1,602 power-bearing holdout laws were not held out.

@pytest.mark.parametrize("label,raw,canonical_spelling", [
    ("x**2 == pow(x, 2)", ["**", "x1", "2"], ["pow", "x1", "2"]),
    ("1/2 m (v^2 + u^2) 1/2 w^2 (I.24.6)", ["*", "*", "*", "/", "1", "2", "x1", "+", "**", "x2", "2", "**", "x3", "2", "*", "/", "1", "2", "**", "x4", "2"],
     ["*", "*", "*", "/", "1", "2", "x1", "+", "pow", "x2", "2", "pow", "x3", "2", "*", "/", "1", "2", "pow", "x4", "2"]),
    ("exp(-(x/y)^2 / 2) (I.6.2)", ["exp", "/", "neg", "**", "/", "x1", "x2", "2", "2"], ["exp", "/", "neg", "pow", "/", "x1", "x2", "2", "2"]),
])
def test_power_spelling_registers_the_canonical_family(catalog, label, raw, canonical_spelling):
    engine = catalog.simplipy_engine
    canonical = list(engine.simplify(canonical_spelling, mode=HOLDOUT_SIMPLIFY_MODE, effort=HOLDOUT_EFFORT))
    assert catalog.holdout_family_prototype(raw) == catalog.holdout_family_prototype(canonical, assume_canonical=True), label
    assert catalog.holdout_family_prototype(raw) == catalog.holdout_family_prototype(canonical_spelling), label


def test_strict_prototype_refuses_an_uncanonicalizable_law(catalog):
    """On the registration path a canonicalization failure must not fall back to the raw spelling."""
    alien = ["frobnicate", "x1"]
    assert catalog.holdout_family_prototype(alien, strict=True) is None


@pytest.mark.parametrize("name", ["nguyen", "keijzer"])
def test_every_catalog_law_is_held_out_in_canonical_form(catalog, name):
    """The leak test: register a curated catalog, then canonicalize each of its laws exactly as
    the sampler would and ask is_held_out. Every law must be held out."""
    from symbolic_data.catalog import load_catalog
    from symbolic_data.token_ops import desugar_sqrt
    import numpy as np
    import re
    engine = catalog.simplipy_engine
    catalog.clear_holdouts()
    catalog.register_holdout_pool(name)
    missed = []
    for entry in load_catalog(name).iter_entries(np.random.default_rng(0)):
        prefix = desugar_sqrt(engine.infix_to_prefix(entry.prepared), engine.operator_arity_compat)
        prefix = [("x" + t[1:]) if re.fullmatch(r"v\d+", t) else ("pow" if t in ("**", "^") else t) for t in prefix]
        canonical = list(engine.simplify(prefix, mode=HOLDOUT_SIMPLIFY_MODE, effort=HOLDOUT_EFFORT))
        if not catalog.is_held_out(canonical, [], assume_canonical=catalog._skeleton_is_holdout_canonical):
            missed.append((entry.id, entry.prepared))
    catalog.clear_holdouts()
    assert not missed, f"{len(missed)} laws of {name} are not held out in canonical form: {missed[:5]}"
