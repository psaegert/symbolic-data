"""The width prior (alternation_decay) on the nesting sampler: label mode is a pure relabeling of the
depth prior's draw (delta = 1 is byte-identical), shape mode keeps exact operator counts, and both
make class switches rarer along a backbone."""
import os
import warnings

import numpy as np
import pytest
import yaml

from symbolic_data.generative import LampleChartonCatalog

ADD, MUL = {"+", "-"}, {"*", "/"}


def make(**extra):
    cfg_path = os.path.join(os.path.dirname(__file__), "..", "configs", "test", "catalog_train.yaml")
    config = yaml.safe_load(open(cfg_path, encoding="utf-8"))
    config["holdout_pools"] = []
    config.update({"unary_mass": 0.5, "nesting_decay": 0.25, "nesting_transparent": ["neg", "inv"]}, **extra)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return LampleChartonCatalog.from_config(config)


def draws(cat, n_ops, seed, count=60):
    rng = np.random.default_rng(seed)
    return [tuple(cat.skeleton_sampler.sample(n_ops, rng)) for _ in range(count)]


def switch_rate(cat, skeletons):
    """Share of binary nodes whose class differs from the parent backbone class (through neg/inv)."""
    arity = cat.simplipy_engine.operator_arity
    switches = binaries = 0
    for sk in skeletons:
        pos = [0]

        def walk(parent_class):
            nonlocal switches, binaries
            tok = sk[pos[0]]; pos[0] += 1
            if tok not in arity:
                return
            if arity[tok] == 2 and tok in ADD | MUL:
                cls = "A" if tok in ADD else "M"
                binaries += 1
                switches += int(parent_class is not None and cls != parent_class)
                walk(cls); walk(cls)
            elif tok in ("neg", "inv"):
                walk(parent_class)
            else:
                for _ in range(arity[tok]):
                    walk(None)
        walk(None)
    return switches / max(binaries, 1)


@pytest.mark.parametrize("mode", ["label", "shape"])
def test_bad_values_are_refused(mode):
    with pytest.raises(ValueError):
        make(alternation_decay=0.0, alternation_mode=mode)
    with pytest.raises(ValueError):
        make(alternation_decay=0.5, alternation_mode="sideways")


@pytest.mark.parametrize("mode", ["label", "shape"])
def test_delta_one_is_the_plain_nesting_draw(mode):
    plain = make()
    sticky = make(alternation_decay=1.0, alternation_mode=mode)
    for n in (3, 8, 14):
        assert draws(plain, n, 11) == draws(sticky, n, 11)


@pytest.mark.parametrize("mode", ["label", "shape"])
def test_exact_operator_count_and_fewer_switches(mode):
    plain = make()
    sticky = make(alternation_decay=0.25, alternation_mode=mode)
    arity = plain.simplipy_engine.operator_arity
    for n in (6, 12):
        sk = draws(sticky, n, 3, count=200)
        assert all(sum(1 for t in s if t in arity) == n for s in sk)
        assert switch_rate(sticky, sk) < 0.6 * switch_rate(plain, draws(plain, n, 3, count=200))


def test_label_mode_keeps_the_shape():
    """Relabeling: the same seed gives the same unary/binary layout, only the binary identities move."""
    plain, label = make(), make(alternation_decay=0.25, alternation_mode="label")
    arity = plain.simplipy_engine.operator_arity
    for a, b in zip(draws(plain, 10, 5), draws(label, 10, 5)):
        assert len(a) == len(b)
        assert [arity.get(t, 0) for t in a] == [arity.get(t, 0) for t in b]
        assert [t for t in a if t in arity and arity[t] == 1] == [t for t in b if t in arity and arity[t] == 1]
