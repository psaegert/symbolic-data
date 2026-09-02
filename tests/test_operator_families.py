"""Per-expression operator FAMILIES: one independent coin per family (owner ruling 2026-09-02)."""
from pathlib import Path

import numpy as np
import pytest
import yaml

from simplipy import SimpliPyEngine
from symbolic_data._generate.skeleton_sampling import SkeletonSampler

CONFIG = Path(__file__).resolve().parent.parent / "configs" / "test" / "catalog_train.yaml"
TRIG = {"sin", "cos", "tan", "asin", "acos", "atan"}
HYP = {"sinh", "cosh", "tanh", "asinh", "acosh", "atanh"}
EXPLOG = {"exp", "log"}
POW = {"pow", "rootn"}
ARITH = {"+", "-", "*", "/"}

STRUCT = {"abs", "inv", "neg"}   # the catalog's structural unaries ride in the base family
FAMILIES = [
    {"name": "arith", "p": 1.0, "operators": sorted(ARITH | STRUCT)},
    {"name": "pow", "p": 0.30, "operators": sorted(POW)},
    {"name": "trig", "p": 0.20, "operators": sorted(TRIG)},
    {"name": "explog", "p": 0.20, "operators": sorted(EXPLOG)},
    {"name": "hyp", "p": 0.10, "operators": sorted(HYP)},
]


@pytest.fixture(scope="module")
def cfg():
    return yaml.safe_load(open(CONFIG))


@pytest.fixture(scope="module")
def engine(cfg):
    return SimpliPyEngine.load(cfg.get("simplipy_engine", "acj-4"), install=True)


def make_sampler(engine, cfg, families=None, profiles=None):
    return SkeletonSampler(simplipy_engine=engine, sample_strategy=cfg["sample_strategy"], variables=cfg["variables"],
                           operator_weights=cfg["operator_weights"], literal_prior=cfg["literal_prior"],
                           typed_slots=cfg["typed_slots"], operator_families=families, operator_profiles=profiles)


def _ops(engine, sk):
    return {t for t in sk if t in engine.operator_arity}


def test_families_restrict_operators_to_the_drawn_subset(engine, cfg):
    s = make_sampler(engine, cfg, families=FAMILIES)
    rng = np.random.default_rng(1)
    for _ in range(300):
        ops = _ops(engine, s.sample(6, rng))
        # every operator belongs to a family, and arithmetic is always allowed
        assert ops <= ARITH | STRUCT | POW | TRIG | EXPLOG | HYP, ops


def test_coin_rates_match_p(engine, cfg):
    # 1,500 draws of 8-operator trees: with equal-mass families nearly every present family
    # places at least one node, so the presence rate tracks its coin.
    s = make_sampler(engine, cfg, families=FAMILIES)
    rng = np.random.default_rng(2)
    n = 1500
    present = {"pow": 0, "trig": 0, "explog": 0, "hyp": 0}
    for _ in range(n):
        ops = _ops(engine, s.sample(8, rng))
        present["pow"] += bool(ops & POW)
        present["trig"] += bool(ops & TRIG)
        present["explog"] += bool(ops & EXPLOG)
        present["hyp"] += bool(ops & HYP)
    rates = {k: v / n for k, v in present.items()}
    # a present family can still miss a node in a short tree, so the rate sits at or below p
    assert 0.20 < rates["pow"] <= 0.33, rates
    assert 0.12 < rates["trig"] <= 0.23, rates
    assert 0.12 < rates["explog"] <= 0.23, rates
    assert 0.05 < rates["hyp"] <= 0.13, rates


def test_present_family_carries_the_base_mass_uniformly(engine, cfg):
    s = make_sampler(engine, cfg, families=FAMILIES)
    base_mass = sum(cfg["operator_weights"][op] for op in ARITH | STRUCT)
    profile = s._family_profile((0, 2))   # arith + trig
    unary = dict(zip(profile["unary_operators"], profile["unary_probs"]))
    assert TRIG <= set(unary) <= TRIG | STRUCT
    trig_share = sum(v for k, v in unary.items() if k in TRIG)
    # the trig family carries the whole base mass, split uniformly over its six members
    assert all(abs(unary[k] - trig_share / len(TRIG)) < 1e-12 for k in TRIG)
    assert s._families[2]["mass"] == base_mass


def test_base_only_subset_draws_only_base_operators(engine, cfg):
    s = make_sampler(engine, cfg, families=FAMILIES)
    profile = s._family_profile((0,))
    assert set(profile["unary_operators"]) <= STRUCT and set(profile["binary_operators"]) == ARITH


def test_every_weighted_operator_needs_a_family(engine, cfg):
    # Dropping an operator by omission is an error, not a silent prior change.
    with pytest.raises(ValueError, match="belong to no family"):
        make_sampler(engine, cfg, families=[{"p": 1.0, "operators": sorted(ARITH)}])


def test_family_profiles_are_cached_per_subset(engine, cfg):
    s = make_sampler(engine, cfg, families=FAMILIES)
    a = s._family_profile((0, 1))
    b = s._family_profile((0, 1))
    assert a is b


def test_validation(engine, cfg):
    with pytest.raises(ValueError, match="mutually exclusive"):
        make_sampler(engine, cfg, families=FAMILIES, profiles=[{"weight": 1.0, "operators": ["+", "*"]}])
    with pytest.raises(ValueError, match="p = 1"):
        make_sampler(engine, cfg, families=[{"p": 0.5, "operators": ["+", "*"]}])
    with pytest.raises(ValueError, match="unknown operators"):
        make_sampler(engine, cfg, families=[{"p": 1.0, "operators": ["+", "frobnicate"]}])
    with pytest.raises(ValueError, match="0 < p <= 1"):
        make_sampler(engine, cfg, families=[{"p": 1.0, "operators": ["+"]}, {"p": 0.0, "operators": ["sin"]}])


def test_no_families_consumes_no_rng(engine, cfg):
    # `None` is the legacy sampler: the same seed yields the same skeleton with or without the key.
    legacy = make_sampler(engine, cfg)
    also = make_sampler(engine, cfg, families=None)
    a = legacy.sample(5, np.random.default_rng(9))
    b = also.sample(5, np.random.default_rng(9))
    assert a == b
