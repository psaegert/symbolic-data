"""The public GenerativeCatalog contract (the direct-use API flash-ansr + srbf baselines depend on)."""
import os

import numpy as np
import pytest
import yaml

from symbolic_data import Catalog, GenerativeCatalog, LampleChartonCatalog, RealizedExpression, build_catalog
from symbolic_data.errors import NoValidSampleFoundError


def _cfg():
    cfg_path = os.path.join(os.path.dirname(__file__), "..", "configs", "test", "skeleton_pool_train.yaml")
    return yaml.safe_load(open(cfg_path, encoding="utf-8"))


@pytest.fixture(scope="module")
def catalog():
    return LampleChartonCatalog.from_config(_cfg())


def test_is_a_generative_catalog(catalog):
    assert isinstance(catalog, GenerativeCatalog) and isinstance(catalog, Catalog)
    assert catalog.is_finite() is False
    assert catalog.name == "lample_charton"


def test_sample_skeleton_direct(catalog):
    # The feature-extractor / baseline path: raw random structures on demand, no support data.
    rng = np.random.default_rng(0)
    skeleton, code, constants = catalog.sample_skeleton(new=True, decontaminate=False, rng=rng)
    assert isinstance(skeleton, tuple) and len(skeleton) >= 1
    assert all(isinstance(t, str) for t in skeleton)
    assert isinstance(constants, list)


def test_clear_holdouts_is_callable(catalog):
    catalog.clear_holdouts()  # srbf baselines call this; must not raise
    assert catalog.holdout_pools == []


def test_iter_entries_then_realize(catalog):
    rng = np.random.default_rng(1)
    entries = list(catalog.iter_entries(rng, size=2))
    assert len(entries) == 2
    realized_any = False
    for entry in entries:
        try:
            realized = catalog.realize(entry, n_points=12, rng=rng)
        except NoValidSampleFoundError:
            continue
        assert isinstance(realized, RealizedExpression)
        assert realized.x.shape[0] == 12 and realized.y.shape[0] == 12
        assert realized.skeleton == tuple(entry.skeleton)
        realized_any = True
    assert realized_any, "expected at least one entry to realize"


def test_build_catalog_dispatch():
    # mapping with a type -> generative; the registry resolves it.
    spec = {**_cfg(), "type": "lample_charton"}
    cat = build_catalog(spec)
    assert isinstance(cat, LampleChartonCatalog)
    with pytest.raises(ValueError, match="type"):
        build_catalog({"simplipy_engine": "dev_7-3"})  # mapping without a type
    with pytest.raises(ValueError, match="unknown generative catalog"):
        build_catalog({"type": "does_not_exist"})
