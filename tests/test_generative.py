"""The public GenerativeCatalog contract (the direct-use API flash-ansr + srbf baselines depend on)."""
import os

import numpy as np
import pytest
import yaml

from symbolic_data import Catalog, GenerativeCatalog, LampleChartonCatalog, RealizedExpression, build_catalog
from symbolic_data.errors import NoValidSampleFoundError


def _cfg():
    cfg_path = os.path.join(os.path.dirname(__file__), "..", "configs", "test", "catalog_train.yaml")
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


def test_iter_entries_default_is_bounded_and_raises_on_open():
    # ISSUE-002: the default method='iterate' matches Catalog.iter_entries -> list() is BOUNDED on a
    # fixed skeleton set; an OPEN catalog with no set and no size raises instead of a silent unbounded
    # stream a caller might list() (the footgun), while explicit method='procedural' still streams.
    import itertools
    rng = np.random.default_rng(3)
    fixed = LampleChartonCatalog.from_config(_cfg())
    fixed.create(3, rng=rng)
    assert len(list(fixed.iter_entries(rng))) == 3            # default 'iterate' -> bounded, no hang

    open_cat = LampleChartonCatalog.from_config(_cfg())
    with pytest.raises(ValueError):
        list(open_cat.iter_entries(rng))                      # open, no set, no size -> raise (no hang)

    streamed = list(itertools.islice(open_cat.iter_entries(rng, method="procedural"), 3))
    assert len(streamed) == 3                                 # explicit procedural still streams


def test_build_catalog_dispatch():
    # mapping with a type -> generative; the registry resolves it.
    spec = {**_cfg(), "type": "lample_charton"}
    cat = build_catalog(spec)
    assert isinstance(cat, LampleChartonCatalog)
    with pytest.raises(ValueError, match="type"):
        build_catalog({"simplipy_engine": "dev_7-3"})  # mapping without a type
    with pytest.raises(ValueError, match="unknown generative catalog"):
        build_catalog({"type": "does_not_exist"})


def test_register_holdout_pool_frozen_catalog_is_not_a_silent_noop(tmp_path):
    # A FROZEN ProblemCatalog (materialized .npz, e.g. a measured-data import with reference laws)
    # has `entries == {}`; the pre-fix registration iterated entries and silently registered NOTHING,
    # leaving every frozen benchmark OUT of the two-layer training holdout. The prototypes must come
    # from `.problems` (skeleton, falling back to expression tokens); gt_kind="none" (black-box)
    # problems carry neither and contribute nothing, by definition.
    from symbolic_data import Problem, ProblemCatalog

    x = np.linspace(0.5, 2.5, 16)
    with_law = Problem.from_data(x, np.sin(x), expression=["sin", "x1"], eq_id="lawful")
    black_box = Problem.from_data(x, np.exp(x), eq_id="blackbox")
    assert with_law.gt_kind == "reference" and black_box.gt_kind == "none"
    frozen = ProblemCatalog.from_problems([with_law, black_box], name="frozen-probe")
    npz = str(frozen.save(tmp_path / "frozen-probe"))

    catalog = LampleChartonCatalog.from_config(_cfg())
    catalog.register_holdout_pool(npz)
    try:
        assert ("sin", "x1") in catalog.holdout_skeletons     # pre-fix: empty set (silent no-op)
        assert len(catalog.holdout_skeletons) == 1            # the black-box problem contributed nothing
        assert len(catalog.holdout_y) >= 1                    # the grid-image layer registered too
    finally:
        catalog.clear_holdouts()
