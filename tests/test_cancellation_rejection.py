"""sample_skeleton rejects cancellation artefacts (owner ruling 2026-09-02): a draw whose
simplification removes every variable it placed is resampled; a draw made without variables
(the constant slot alone) is a deliberate constant law and passes."""
import os
import warnings

import numpy as np
import pytest
import yaml

from symbolic_data.generative import LampleChartonCatalog


@pytest.fixture(scope="module")
def catalog():
    cfg_path = os.path.join(os.path.dirname(__file__), "..", "configs", "test", "catalog_train.yaml")
    config = yaml.safe_load(open(cfg_path, encoding="utf-8"))
    config["holdout_pools"] = []
    config["sample_strategy"]["max_tries"] = 8   # the shipped test recipe allows ONE try; rejection needs a retry budget
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return LampleChartonCatalog.from_config(config)


def _scripted(catalog, monkeypatch, draws):
    it = iter(draws)
    monkeypatch.setattr(catalog.skeleton_sampler, "sample", lambda n_operators, rng=None: list(next(it)))


def test_cancellation_artefact_is_resampled(catalog, monkeypatch):
    _scripted(catalog, monkeypatch, [["-", "x1", "x1"], ["/", "x2", "x2"], ["+", "x1", "2.5"]])
    skeleton, _, _ = catalog.sample_skeleton(new=True, decontaminate=False, rng=np.random.default_rng(0))
    assert "x1" in skeleton, skeleton
    assert not (set(skeleton) & {"x2"})


def test_deliberate_constant_draw_passes(catalog, monkeypatch):
    _scripted(catalog, monkeypatch, [["+", "1.5", "2"]])
    skeleton, _, _ = catalog.sample_skeleton(new=True, decontaminate=False, rng=np.random.default_rng(0))
    assert not any(t.startswith("x") for t in skeleton), skeleton
