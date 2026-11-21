"""The research term prior as a generative catalog (``type: term_prior``, design/term-prior).

The full equivalence with the prototype (draw for draw under one seed) is checked by
``design/term-prior/code`` against the T8.1 config; these tests cover the catalog contract.
"""
import os
import pickle
import warnings

import numpy as np
import pytest
import yaml

from symbolic_data import build_catalog
from symbolic_data.generative import _GENERATIVE_CATALOGS
from symbolic_data.term_prior import TermPriorCatalog

CONFIG = os.path.join(os.path.dirname(__file__), "..", "configs", "test", "catalog_train.yaml")
FAMILIES = {"abs": ["abs"], "exp": ["exp"], "hyperbolic": ["sinh", "cosh", "tanh"], "inverse_hyperbolic": ["asinh", "acosh", "atanh"],
            "inverse_trig": ["asin", "acos", "atan"], "log": ["log"], "trig": ["sin", "cos", "tan"]}


def _config():
    config = yaml.safe_load(open(CONFIG, encoding="utf-8"))
    config.update(unary_mass=0.5, nesting_decay=0.25, nesting_transparent=["neg", "inv"], alternation_decay=0.5,
                  family_affinity={"concentration": 0.36, "families": FAMILIES, "independent": ["pow", "rootn"]},
                  term_prior={})
    config["sample_strategy"] = dict(config["sample_strategy"], max_length=35, max_tries=50)   # T8's cap; long draws are resampled
    return config


@pytest.fixture(scope="module")
def catalog():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return TermPriorCatalog.from_config(_config())


def test_registered():
    assert _GENERATIVE_CATALOGS["term_prior"] is TermPriorCatalog


def test_build_catalog_by_type():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        built = build_catalog(dict(_config(), type="term_prior"))
    assert isinstance(built, TermPriorCatalog)


def test_deterministic_and_capped(catalog):
    def draws(seed):
        rng = np.random.default_rng(seed)
        return [catalog.sample_skeleton(new=True, decontaminate=False, rng=rng)[0] for _ in range(20)]
    first, second = draws(0), draws(0)
    assert first == second
    assert all(len(s) <= catalog.sample_strategy["max_length"] for s in first)
    assert any(t in catalog.variables for s in first for t in s)


def test_pickle_roundtrip(catalog):
    restored = pickle.loads(pickle.dumps(catalog))
    rng = np.random.default_rng(1)
    skeleton, code, constants = restored.sample_skeleton(new=True, decontaminate=False, rng=rng)
    assert len(skeleton) > 0 and code is not None
