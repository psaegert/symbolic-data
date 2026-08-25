"""Regression tests for the 2026-08-25 holdout hardening (adversarial-audit findings).

Every case here is a REPRODUCED leak from the audit: an expression that entered
training while its family was declared held out. The ruled doctrine: over-rejection
over contamination; families, not spellings.
"""
import os
import warnings

import numpy as np
import pytest
import yaml

from symbolic_data.generative import LampleChartonCatalog


@pytest.fixture(scope="module")
def fastsrb_catalog():
    cfg_path = os.path.join(os.path.dirname(__file__), "..", "configs", "test", "catalog_train.yaml")
    config = yaml.safe_load(open(cfg_path, encoding="utf-8"))
    config["holdout_pools"] = ["fastsrb"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return LampleChartonCatalog.from_config(config)


def test_full_pool_registers_or_raises(fastsrb_catalog):
    # Audit L1: 64 of fastsrb's 120 power-bearing laws silently vanished behind a bare
    # `continue` (`**` missing from the plain arity table). Registration now either
    # covers the whole pool or raises; the quotiented family count must reflect all
    # 120 laws, not the 33 families of the power-free half.
    assert len(fastsrb_catalog.holdout_skeletons) > 60


@pytest.mark.parametrize("name,tokens", [
    # L1: power-bearing laws (II.27.18, II.8.31 shapes)
    ("exact power law", ['*', '8.854e-12', 'pow', 'x1', '2']),
    ("pow(x1,2)", ['pow', 'x1', '2']),
    ("pow(x1,2)/2", ['/', 'pow', 'x1', '2', '2']),
    # L2: the literal-affine family of registered x1*x2 (I.12.1)
    ("scaled", ['*', '3.5', '*', 'x1', 'x2']),
    ("shifted", ['+', '*', 'x1', 'x2', '1.2']),
    ("scaled+shifted", ['+', '*', '3.5', '*', 'x1', 'x2', '1.2']),
    # L3: variable renumbering
    ("renumbered variable", ['x2']),
    ("swapped ratio", ['/', 'x2', 'x1']),
    # L4: half-domain respellings of the law x1 (positive-grid backstop)
    ("exp(log(x1))", ['exp', 'log', 'x1']),
    ("sqrt(x1)^2", ['pow', 'rootn', 'x1', '2', '2']),
    # L5/L6: re-association of a registered product law (II.38.3)
    ("re-associated product", ['/', '*', 'x1', '*', 'x2', 'x3', 'x4']),
])
def test_family_members_are_held_out(fastsrb_catalog, name, tokens):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert fastsrb_catalog.is_held_out(list(tokens), []), name


def test_ordinary_draws_mostly_pass(fastsrb_catalog):
    # The conservative direction must stay affordable: measured 0.30% rejection on
    # 1000 ordinary draws of the v24.0-T16 config at hardening time, ~5% on THIS
    # simpler test-config prior (more of its mass genuinely falls in fastsrb
    # families). Bound well above baseline so only a collision-class regression
    # (the const-key and nonfinite-mean classes added +7-8 points) fails.
    rng = np.random.default_rng(7)
    rejected = n = 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        while n < 300:
            try:
                sk, code, consts = fastsrb_catalog.sample_skeleton(
                    rng=rng, new=True, decontaminate=False)
            except Exception:
                continue
            n += 1
            rejected += bool(fastsrb_catalog.is_held_out(list(sk), list(consts), code))
    assert rejected / n < 0.10


def test_uncanonicalizable_probe_fails_closed(fastsrb_catalog):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert fastsrb_catalog.is_held_out(['completely', 'alien', 'tokens'], [])
