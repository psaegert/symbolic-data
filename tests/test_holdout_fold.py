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
