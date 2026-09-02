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


def test_present_families_share_the_catalog_mass_of_the_operators_they_cover(engine, cfg):
    # Families choose the vocabulary; the catalog keeps the tree shape: the optional families
    # present carry exactly the catalog mass of all optional operators, split equally per
    # present family and uniformly within, so the unary/binary balance the recursion sees is
    # the catalog's whatever subset is drawn.
    s = make_sampler(engine, cfg, families=FAMILIES)
    w = cfg["operator_weights"]
    optional_mass = sum(w[op] for op in POW | TRIG | EXPLOG | HYP)
    profile = s._family_profile((0, 2))   # arith + trig: trig alone carries the whole optional mass
    unary = dict(zip(profile["unary_operators"], profile["unary_probs"]))
    assert TRIG <= set(unary) <= TRIG | STRUCT
    total = sum(w[op] for op in ARITH | STRUCT) + optional_mass
    assert all(abs(unary[op] - unary["sin"]) < 1e-12 for op in TRIG)   # uniform within the family
    # the unary mass equals what the catalog gives ALL unary-effective operators
    u_mass = profile["n_unary"] / (profile["n_unary"] + profile["n_binary"]) * total
    assert abs(u_mass - (optional_mass + sum(w[op] for op in STRUCT))) < 1e-9
    two = s._family_profile((0, 2, 3))    # trig + explog split the same optional mass
    unary2 = dict(zip(two["unary_operators"], two["unary_probs"]))
    assert abs(unary2["exp"] / unary2["sin"] - len(TRIG) / len(EXPLOG)) < 1e-9
    assert abs(two["n_unary"] - profile["n_unary"]) < 1e-12


def test_operators_per_coin_scales_mixing_with_length(engine, cfg):
    # One coin per block of m operators: a long expression mixes more, a short one stays pure.
    s = make_sampler(engine, cfg, families=FAMILIES)
    s5 = SkeletonSampler(simplipy_engine=engine, sample_strategy=cfg["sample_strategy"], variables=cfg["variables"],
                         operator_weights=cfg["operator_weights"], literal_prior=cfg["literal_prior"],
                         typed_slots=cfg["typed_slots"], operator_families=FAMILIES, operators_per_coin=5)
    rng = np.random.default_rng(5)
    def families_present(sampler, n_ops, draws=800):
        cnt = 0
        for _ in range(draws):
            ops = _ops(engine, sampler.sample(n_ops, rng))
            cnt += sum(bool(ops & fam) for fam in (POW, TRIG, EXPLOG, HYP))
        return cnt / draws
    short_one, long_one = families_present(s, 3), families_present(s, 15)
    short_five, long_five = families_present(s5, 3), families_present(s5, 15)
    assert abs(short_five - short_one) < 0.12          # <= 5 operators: a single coin either way
    assert long_five > long_one + 0.3                  # 15 operators: three coins per family


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


def test_operator_subset_draws_only_from_the_drawn_subset(engine, cfg):
    # The operator-side twin of the variable draw: k distinct operators (k <= n_ops), then every
    # node draws from that subset -- so a k-operator expression never uses more than k kinds.
    s = SkeletonSampler(simplipy_engine=engine, sample_strategy=cfg["sample_strategy"], variables=cfg["variables"],
                        operator_weights=cfg["operator_weights"], literal_prior=cfg["literal_prior"],
                        typed_slots=cfg["typed_slots"], operator_subset=True)
    rng = np.random.default_rng(3)
    for n_ops in (1, 2, 4, 8):
        for _ in range(100):
            ops = [t for t in s.sample(n_ops, rng) if t in engine.operator_arity]
            assert len(ops) == n_ops and len(set(ops)) <= n_ops


def test_operator_subset_is_exclusive_with_families(engine, cfg):
    with pytest.raises(ValueError, match="exclusive"):
        SkeletonSampler(simplipy_engine=engine, sample_strategy=cfg["sample_strategy"], variables=cfg["variables"],
                        operator_weights=cfg["operator_weights"], literal_prior=cfg["literal_prior"],
                        typed_slots=cfg["typed_slots"], operator_families=FAMILIES, operator_subset=True)
