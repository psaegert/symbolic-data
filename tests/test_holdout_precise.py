"""The precise holdout family key (``holdout_family: precise``, 2026-11-20).

The legacy key masks every literal and then folds every operator with a ``<constant>`` operand into its other
operand. With typed exponent slots that also strips exponents and root indices, and it folds ``c/X`` into ``X``, so
x^2, x^3, sqrt(x), c/(x+a) and a*x+b all reduce to the core ``x``. The precise key masks by role (simplipy's
``mask_fittable``: exponents and root indices are structure) and folds a constant only where it reparametrizes its
operand affinely.
"""
import os
import re
import warnings

import numpy as np
import pytest
import yaml

from simplipy import masking
from simplipy.utils import explicit_constant_placeholders

from symbolic_data._generate.holdout import HoldoutManager
from symbolic_data.generative import HOLDOUT_EFFORT, HOLDOUT_SIMPLIFY_MODE, LampleChartonCatalog
from symbolic_data.token_ops import desugar_sqrt

CONFIG = os.path.join(os.path.dirname(__file__), "..", "configs", "test", "catalog_train.yaml")


def _build(holdout_family=None, pools=()):
    config = yaml.safe_load(open(CONFIG, encoding="utf-8"))
    config["holdout_pools"] = list(pools)
    if holdout_family is not None:
        config["holdout_family"] = holdout_family
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return LampleChartonCatalog.from_config(config)


@pytest.fixture(scope="module")
def precise():
    return _build("precise")


@pytest.fixture(scope="module")
def legacy():
    return _build()


def _held(catalog, tokens):
    engine = catalog.simplipy_engine
    _, constants = explicit_constant_placeholders(engine.operators_to_realizations(list(tokens)), inplace=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return catalog.is_held_out(list(tokens), constants)


def _prefix(catalog, infix):
    engine = catalog.simplipy_engine
    tokens = ["pow" if token in ("**", "^") else token for token in engine.infix_to_prefix(infix)]
    tokens = desugar_sqrt(tokens, engine.operator_arity_compat)
    return list(engine.simplify(tokens, mode=HOLDOUT_SIMPLIFY_MODE, effort=HOLDOUT_EFFORT))


# --- configuration -------------------------------------------------------------------------------------------------

def test_legacy_is_the_default(legacy):
    assert legacy.holdout_family == "legacy"
    assert legacy.holdout_manager.keep_literals is False


def test_unknown_key_is_refused():
    with pytest.raises(ValueError, match="holdout_family"):
        _build("loose")


def test_precise_catalog_keeps_literals_in_the_hash_layer(precise):
    assert precise.holdout_manager.keep_literals is True
    assert precise.holdout_manager.finite_standardization is True
    precise.clear_holdouts()
    assert precise.holdout_manager.keep_literals is True
    assert precise.holdout_manager.finite_standardization is True


def test_finite_standardization_is_affine_invariant_where_the_image_is_partly_nan():
    """Legacy zeroes non-finite entries before standardizing, so f + 1 and f key differently wherever f is NaN on
    part of the grid; the finite-standardized key is exactly invariant under a*f + b."""
    rng = np.random.default_rng(0)
    image = rng.normal(size=50)
    image[::3] = np.nan
    finite_key = HoldoutManager._finite_image_key
    assert finite_key(0, image) == finite_key(0, 2.5 * image + 1.0) == finite_key(0, -0.5 * image - 3.0)
    assert finite_key(0, image) != finite_key(0, image ** 3)


def test_unpickled_catalog_without_the_key_is_legacy(legacy):
    state = legacy.__getstate__()
    state.pop("holdout_family")
    restored = LampleChartonCatalog.__new__(LampleChartonCatalog)
    restored.__setstate__(state)
    assert restored.holdout_family == "legacy"


def test_manager_hash_keeps_numerals_only_with_keep_literals():
    loose = HoldoutManager(n_variables=2, allow_nan=False)
    exact = HoldoutManager(n_variables=2, allow_nan=False, keep_literals=True)
    assert loose._normalize_tokens(["pow", "x1", "2"]) == loose._normalize_tokens(["pow", "x1", "3"])
    assert exact._normalize_tokens(["pow", "x1", "2"]) != exact._normalize_tokens(["pow", "x1", "3"])
    assert exact._normalize_tokens(["+", "v1", "<c>"]) == ["+", "x1", "<constant>"]


# --- the precise prototype -----------------------------------------------------------------------------------------

@pytest.mark.parametrize("infix,prototype", [
    ("x1", ["x1"]),
    ("2.3*x1 + 0.7", ["x1"]),                               # scale and shift fold, as in the legacy key
    ("1.3*x1**2 + 0.4", ["pow", "x1", "2"]),                # the exponent is structure
    ("1.3*x1**3 + 0.4", ["pow", "x1", "3"]),
    ("1.5/(x1 + 0.3)", ["inv", "x1"]),                      # a reciprocal is not a rescaling
    ("2**x1", ["exp", "x1"]),                               # c^X = exp(X log c)
    ("1.2*exp(-0.3*x1)", ["exp", "x1"]),
])
def test_precise_prototypes(precise, infix, prototype):
    assert precise.holdout_family_prototype(_prefix(precise, infix)) == prototype


@pytest.mark.parametrize("left,right", [
    ("x1 - x2", "x1 + x2"),                                 # additive term signs are spelling
    ("1.3*x1**2 + 0.4", "x1**2"),
    ("1.5/(x1 + 0.3)", "2.2/(x1 - 1.7)"),
    ("2**x1", "exp(-1.3*x1)"),
    ("0.5*x1**3 - 2.1*x1 + 4", "x1**3 + x1"),
])
def test_same_family_shares_a_precise_prototype(precise, left, right):
    assert precise.holdout_family_prototype(_prefix(precise, left)) == \
        precise.holdout_family_prototype(_prefix(precise, right))


@pytest.mark.parametrize("left,right", [
    ("x1**2", "x1**3"),
    ("x1**2", "x1"),
    ("1.5/(x1 + 0.3)", "x1"),
    ("x1**4 + x1", "x1**3 + x1**2 + x1"),
    ("x1**2", "sqrt(x1)"),
])
def test_different_families_stay_apart(precise, legacy, left, right):
    """Each pair shares the legacy prototype or image; the precise key separates them."""
    assert precise.holdout_family_prototype(_prefix(precise, left)) != \
        precise.holdout_family_prototype(_prefix(precise, right))


def test_prototype_pass_never_loses_a_variable(precise):
    """(x+1)/(x+2) folds to x/x; simplifying that pass to a constant would hold out every constant law."""
    prototype = precise.holdout_family_prototype(_prefix(precise, "(x1 + 1)/(x1 + 2)"))
    assert prototype is not None and "x1" in prototype


def test_rational_exponent_is_kept_whole(precise):
    prototype = precise.holdout_family_prototype(_prefix(precise, "3*x1**2.5"))
    assert prototype is not None and prototype[0] == "pow" and "<constant>" not in prototype
    assert prototype != ["x1"]


@pytest.mark.parametrize("label,left,right", [
    # the legacy same-family cases (test_holdout_fold) hold under the precise key too
    ("x+x == 2x", ["+", "x1", "x1"], ["*", "2", "x1"]),
    ("x*x == x^2", ["*", "x1", "x1"], ["pow", "x1", "2"]),
    ("2(x+1) == 2x+2", ["*", "2", "+", "x1", "1"], ["+", "*", "2", "x1", "2"]),
    ("c*x == x*c", ["*", "3.0", "x1"], ["*", "x1", "7.5"]),
    ("sin(2x) == sin(x)", ["sin", "*", "2.0", "x1"], ["sin", "x1"]),
    ("AC commuted", ["/", "*", "x1", "x2", "x3"], ["/", "*", "x2", "x1", "x3"]),
    ("renumbered", ["/", "x2", "x1"], ["/", "x1", "x2"]),
])
def test_legacy_family_equalities_survive(precise, label, left, right):
    assert precise.holdout_family_prototype(left) == precise.holdout_family_prototype(right), label


def test_subfamily_is_not_merged(precise):
    square = precise.holdout_family_prototype(["pow", "+", "x1", "1.0", "2"])
    expanded = precise.holdout_family_prototype(["+", "+", "pow", "x1", "2", "*", "2.0", "x1", "1.0"])
    assert square is not None and expanded is not None and square != expanded


@pytest.mark.parametrize("infix", ["x1**2", "1.5/(x1 + 0.3)", "2**x1", "x1**3 + x1", "3*x1**2.5", "x1/(x2 + 1)"])
def test_precise_prototype_is_idempotent(precise, infix):
    once = precise.holdout_family_prototype(_prefix(precise, infix))
    assert precise.holdout_family_prototype(once) == once


def test_legacy_prototypes_are_unchanged(legacy):
    """The legacy key is byte-identical: exponents and reciprocals still fold to the core."""
    assert legacy.holdout_family_prototype(_prefix(legacy, "1.3*x1**3 + 0.4")) == ["x1"]
    assert legacy.holdout_family_prototype(_prefix(legacy, "1.5/(x1 + 0.3)")) == ["x1"]


# --- registration and probing --------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def nguyen_pair():
    return _build("precise", ["nguyen"]), _build(None, ["nguyen"])


def test_released_draw_trains_under_precise_only(nguyen_pair):
    """c1 x^4 + c2 x + c3 is in no nguyen family; the legacy key held it out through sqrt(x), whose legacy core is x."""
    precise, legacy = nguyen_pair
    tokens = _prefix(precise, "1.3*x1**4 + 0.4*x1 + 2.1")
    assert not _held(precise, tokens)
    assert _held(legacy, tokens)


@pytest.mark.parametrize("infix", ["1.7*x1**3 - 0.2*x1**2 + 3.1*x1 + 0.5", "2.5*sqrt(x1) - 1", "4*x1**6 + x1**5 - x1**4 + x1**3 + x1**2 + 2*x1"])
def test_family_members_stay_held_out(nguyen_pair, infix):
    precise, _ = nguyen_pair
    engine = precise.simplipy_engine
    tokens = desugar_sqrt(["pow" if t in ("**", "^") else t for t in engine.infix_to_prefix(infix)],
                          engine.operator_arity_compat)
    canonical = list(engine.simplify(tokens, mode=HOLDOUT_SIMPLIFY_MODE, effort=HOLDOUT_EFFORT))
    assert _held(precise, canonical), infix


@pytest.mark.parametrize("name", ["nguyen", "keijzer"])
def test_every_catalog_law_and_its_family_is_held_out(name):
    """The leak test under the precise key: every law in canonical form, and three members of its family with the
    fittable constants redrawn, must be held out."""
    from symbolic_data.catalog import load_catalog
    catalog = _build("precise", [name])
    engine = catalog.simplipy_engine
    rng = np.random.default_rng(0)
    missed = []
    for entry in load_catalog(name).iter_entries(np.random.default_rng(0)):
        prefix = desugar_sqrt(engine.infix_to_prefix(entry.prepared), engine.operator_arity_compat)
        prefix = [("x" + t[1:]) if re.fullmatch(r"v\d+", t) else ("pow" if t in ("**", "^") else t) for t in prefix]
        canonical = list(engine.simplify(prefix, mode=HOLDOUT_SIMPLIFY_MODE, effort=HOLDOUT_EFFORT))
        members = [canonical]
        for _ in range(3):
            member = list(canonical)
            for index, value, role in masking.literal_sites(member, engine):
                if masking.mask_fittable(value, role) is not None:
                    member[index] = repr(round(float(rng.choice([-1, 1]) * rng.uniform(0.2, 5.0)), 3))
            members.append(list(engine.simplify(member, mode=HOLDOUT_SIMPLIFY_MODE, effort=HOLDOUT_EFFORT)))
        for member in members:
            if not _held(catalog, member):
                missed.append((entry.id, " ".join(member)))
    assert not missed, f"{len(missed)} family members of {name} are not held out: {missed[:5]}"
