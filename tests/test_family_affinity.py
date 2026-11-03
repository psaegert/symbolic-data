"""The coherence prior (family_affinity) on the nesting sampler: a Dirichlet-multinomial relabeling of the
function nodes. The tree shape is the depth prior's, each family keeps its weight share on average, and two
function nodes of one expression share a family with probability (1 + alpha * sum pi^2) / (1 + alpha)."""
import os
import warnings
from collections import Counter

import numpy as np
import pytest
import yaml

from symbolic_data.generative import LampleChartonCatalog

FAMILIES = {"trig": ["sin", "cos", "tan"], "inverse_trig": ["asin", "acos", "atan"],
            "hyperbolic": ["sinh", "cosh", "tanh"], "inverse_hyperbolic": ["asinh", "acosh", "atanh"],
            "exp": ["exp"], "log": ["log"], "abs": ["abs"]}
FAMILY_OF = {op: name for name, ops in FAMILIES.items() for op in ops}
SPEC = {"families": FAMILIES, "independent": ["pow", "rootn"]}
PI = {name: len(ops) / 15 for name, ops in FAMILIES.items()}            # every function weighs 1 in the test catalog


def make(**extra):
    cfg_path = os.path.join(os.path.dirname(__file__), "..", "configs", "test", "catalog_train.yaml")
    config = yaml.safe_load(open(cfg_path, encoding="utf-8"))
    config["holdout_pools"] = []
    config.update({"unary_mass": 0.5, "nesting_decay": 0.25, "nesting_transparent": ["neg", "inv"]}, **extra)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return LampleChartonCatalog.from_config(config)


def functions(cat, n_ops, seed, count):
    rng = np.random.default_rng(seed)
    return [[tok for tok in cat.skeleton_sampler.sample(n_ops, rng) if tok in FAMILY_OF] for _ in range(count)]


def p_same_family(draws):
    same = pairs = 0
    for fns in draws:
        fams = Counter(FAMILY_OF[f] for f in fns)
        k = len(fns)
        if k >= 2:
            pairs += k * (k - 1) / 2
            same += sum(n * (n - 1) / 2 for n in fams.values())
    return same / pairs


def test_bad_specs_are_refused():
    with pytest.raises(ValueError, match="requires nesting_decay"):
        cfg_path = os.path.join(os.path.dirname(__file__), "..", "configs", "test", "catalog_train.yaml")
        config = yaml.safe_load(open(cfg_path, encoding="utf-8"))
        config["holdout_pools"] = []
        config["family_affinity"] = {"concentration": 1.0, **SPEC}
        LampleChartonCatalog.from_config(config)
    with pytest.raises(ValueError, match="needs a family; missing"):
        make(family_affinity={"concentration": 1.0, "families": {k: v for k, v in FAMILIES.items() if k != "abs"}, "independent": ["pow", "rootn"]})
    with pytest.raises(ValueError, match="in both"):
        make(family_affinity={"concentration": 1.0, "families": {**FAMILIES, "also": ["sin"]}, "independent": ["pow", "rootn"]})
    with pytest.raises(ValueError, match="not a counted unary operator"):
        make(family_affinity={"concentration": 1.0, "families": {**FAMILIES, "abs": ["abs", "neg"]}, "independent": ["pow", "rootn"]})
    with pytest.raises(ValueError, match="needs a family; missing"):
        make(family_affinity={"concentration": 1.0, "families": FAMILIES})                     # pow/rootn unplaced
    with pytest.raises(ValueError, match="must be positive"):
        make(family_affinity={"concentration": 0.0, **SPEC})
    with pytest.raises(ValueError, match="concentration: alpha"):
        make(family_affinity={"families": FAMILIES, "independent": ["pow", "rootn"]})


@pytest.mark.parametrize("alpha", [0.2, 5.0])
def test_two_function_nodes_share_a_family_as_the_urn_says(alpha):
    cat = make(family_affinity={"concentration": alpha, **SPEC})
    draws = functions(cat, 8, seed=1, count=1500)
    expected = (1 + alpha * sum(p * p for p in PI.values())) / (1 + alpha)
    assert abs(p_same_family(draws) - expected) < 0.04


def test_independent_operators_keep_their_share():
    plain = make()
    coherent = make(family_affinity={"concentration": 0.2, **SPEC})

    def power_share(cat, seed):
        rng = np.random.default_rng(seed)
        toks = [t for _ in range(2000) for t in cat.skeleton_sampler.sample(8, rng)]
        fn = [t for t in toks if t in FAMILY_OF or t in ("pow", "rootn")]
        return sum(1 for t in fn if t in ("pow", "rootn")) / len(fn)
    assert abs(power_share(plain, 6) - power_share(coherent, 7)) < 0.02


def test_without_the_key_the_labels_are_independent():
    draws = functions(make(), 8, seed=2, count=1500)
    assert abs(p_same_family(draws) - sum(p * p for p in PI.values())) < 0.03


def test_each_family_keeps_its_share_and_the_shape_is_unchanged():
    plain = functions(make(), 8, seed=3, count=2000)
    coherent = functions(make(family_affinity={"concentration": 0.2, **SPEC}), 8, seed=4, count=2000)
    n_plain = np.mean([len(f) for f in plain])
    n_coherent = np.mean([len(f) for f in coherent])
    assert abs(n_plain - n_coherent) < 0.08 * n_plain                     # same number of function nodes
    shares = Counter(FAMILY_OF[f] for fns in coherent for f in fns)
    total = sum(shares.values())
    for name, pi in PI.items():
        assert abs(shares[name] / total - pi) < 0.035                    # exchangeable: the share stays pi


def test_operator_counts_stay_exact():
    cat = make(family_affinity={"concentration": 0.2, **SPEC})
    arity = cat.simplipy_engine.operator_arity
    rng = np.random.default_rng(5)
    for n_ops in (1, 4, 9):
        for _ in range(50):
            sk = cat.skeleton_sampler.sample(n_ops, rng)
            assert sum(1 for t in sk if t in arity) == n_ops
