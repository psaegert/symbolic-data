"""Per-expression operator profiles (2026-09-02): a profile is drawn once per expression, nodes
sample from its operator subset, and the unary/binary tree recursion runs on the profile's
weight mass per arity class. The legacy sampler (no `operator_profiles`) must stay byte-identical."""
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from simplipy import SimpliPyEngine
from symbolic_data._generate.skeleton_sampling import SkeletonSampler

CONFIG = Path(__file__).resolve().parent.parent / "configs" / "test" / "catalog_train.yaml"
FIXTURE = Path(__file__).resolve().parent / "fixtures" / "skeleton_sampler_legacy.json"
TRANS = {"sin", "cos", "tan", "asin", "acos", "atan", "sinh", "cosh", "tanh", "asinh", "acosh", "atanh", "exp", "log"}


@pytest.fixture(scope="module")
def cfg():
    return yaml.safe_load(open(CONFIG))


@pytest.fixture(scope="module")
def engine(cfg):
    return SimpliPyEngine.load(cfg.get("simplipy_engine", "acj-4"), install=True)


def make_sampler(engine, cfg, profiles=None, n_vars_prior=None):
    return SkeletonSampler(simplipy_engine=engine, sample_strategy=cfg["sample_strategy"], variables=cfg["variables"],
                           operator_weights=cfg["operator_weights"], literal_prior=cfg["literal_prior"],
                           typed_slots=cfg["typed_slots"], operator_profiles=profiles, n_unique_variables_prior=n_vars_prior)


def test_legacy_sampler_is_byte_identical(engine, cfg):
    fixture = json.load(open(FIXTURE))
    s = make_sampler(engine, cfg)
    for seed, expected in fixture.items():
        rng = np.random.default_rng(1000 + int(seed))
        got = [s.sample(n, rng) for n in (2, 4, 7, 11)]
        assert got == expected, f"seed {seed}: legacy sampling changed"


def test_profile_restricts_operators(engine, cfg):
    s = make_sampler(engine, cfg, profiles=[{"weight": 1.0, "operators": ["+", "-", "*", "/"]}])
    rng = np.random.default_rng(7)
    ops = set()
    for _ in range(200):
        ops |= {t for t in s.sample(6, rng) if t in engine.operator_arity}
    assert ops <= {"+", "-", "*", "/"}, ops


def test_binary_only_profile_grows_binary_trees(engine, cfg):
    # No unary operators in the profile -> unary mass 0 -> the tree recursion never places a unary node.
    s = make_sampler(engine, cfg, profiles=[{"weight": 1.0, "operators": ["+", "*"]}])
    rng = np.random.default_rng(11)
    for _ in range(100):
        sk = s.sample(5, rng)
        assert all(engine.operator_arity[t] == 2 for t in sk if t in engine.operator_arity)


def test_mixture_weights_set_class_shares(engine, cfg):
    s = make_sampler(engine, cfg, profiles=[
        {"name": "arith", "weight": 0.7, "operators": ["+", "-", "*", "/"]},
        {"name": "trig", "weight": 0.3, "operators": {"+": 10, "*": 10, "sin": 10, "cos": 10}},  # profile-local weights
    ])
    rng = np.random.default_rng(3)
    with_trans = sum(any(t in TRANS for t in s.sample(6, rng)) for _ in range(600)) / 600
    # The trig profile is drawn 30% of the time and, with balanced local weights, nearly every
    # 6-operator tree it grows carries a sin/cos.
    assert 0.18 < with_trans < 0.35, with_trans


def test_validation(engine, cfg):
    with pytest.raises(ValueError, match="unknown operators"):
        make_sampler(engine, cfg, profiles=[{"weight": 1.0, "operators": ["+", "nope"]}])
    with pytest.raises(ValueError, match="weight > 0"):
        make_sampler(engine, cfg, profiles=[{"weight": 0.0, "operators": ["+"]}])


def test_n_unique_variables_prior_caps_distinct_symbols(engine, cfg):
    # A prior concentrated on 1-2 distinct leaf symbols must never produce an expression with 3+.
    s = make_sampler(engine, cfg, n_vars_prior={"name": "choice", "kwargs": {"values": [1, 2], "weights": [1, 1]}})
    rng = np.random.default_rng(5)
    for _ in range(200):
        sk = s.sample(8, rng)
        leaves = {t for t in sk if t not in engine.operator_arity}
        assert len(leaves) <= 2, sk
