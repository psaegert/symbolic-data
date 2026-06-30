"""By-name/by-path resolution of GENERATIVE catalogs (sd 0.8.0 C-core).

A string catalog ref (local path or HF ``name[@version]``) is resolved, then dispatched on content:
a ``type:`` spec builds a GenerativeCatalog (open, or FROZEN if it carries inline ``skeletons:``);
anything else is a declarative ProblemCatalog. This is what lets a generative recipe / a frozen
validation set be distributed as a single self-contained yaml and resolved by name.
"""
import os
import tempfile
from pathlib import Path

import numpy as np
import yaml

from symbolic_data import ProblemCatalog, ProblemSource
from symbolic_data.generative import GenerativeCatalog, LampleChartonCatalog, build_catalog

ASSETS = Path(__file__).resolve().parent.parent / "assets" / "catalogs"
RECIPE = Path(__file__).resolve().parent.parent / "configs" / "test" / "skeleton_pool_train.yaml"


def _write_spec(extra: dict) -> str:
    cfg = yaml.safe_load(RECIPE.read_text(encoding="utf-8"))
    cfg["type"] = "lample_charton"
    cfg.update(extra)
    path = os.path.join(tempfile.mkdtemp(), "spec.yaml")
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle)
    return path


def test_build_catalog_open_generative_spec_by_path():
    cat = build_catalog(_write_spec({"name": "open-recipe"}))
    assert isinstance(cat, GenerativeCatalog) and not cat.skeletons   # open: no frozen skeletons


def test_build_catalog_frozen_generative_spec_inlines_skeletons():
    # A frozen catalog distributed as ONE yaml (recipe + inline skeletons), e.g. a validation set.
    seed = LampleChartonCatalog.from_config(yaml.safe_load(RECIPE.read_text(encoding="utf-8")))
    seed.create(4, rng=np.random.default_rng(0))
    frozen = [list(s) for s in seed.skeletons]
    cat = build_catalog(_write_spec({"name": "frozen-val", "skeletons": frozen}))
    assert isinstance(cat, GenerativeCatalog)
    assert cat.skeletons == {tuple(s) for s in frozen}        # exactly the inlined set, no generation


def test_problemsource_over_frozen_spec_samples_only_frozen_skeletons():
    seed = LampleChartonCatalog.from_config(yaml.safe_load(RECIPE.read_text(encoding="utf-8")))
    seed.create(4, rng=np.random.default_rng(1))
    frozen = {tuple(s) for s in seed.skeletons}
    spec = _write_spec({"name": "frozen-val", "skeletons": [list(s) for s in frozen]})
    src = ProblemSource({"catalog": spec, "sampling": {"n_support": 16, "n_validation": 0, "noise": 0.0}})
    from itertools import islice
    problems = [p for p in islice(iter(src), 12) if not p.is_placeholder]
    assert problems and all(tuple(p.skeleton) in frozen for p in problems)


def test_build_catalog_declarative_path_still_builds_problem_catalog():
    cat = build_catalog(str(ASSETS / "nguyen.yaml"))
    assert isinstance(cat, ProblemCatalog) and len(cat) == 12


def test_v23_val_frozen_catalog_resolves_with_its_1000_skeletons():
    # The shipped v23 validation set: a single self-contained generative spec (recipe + inline frozen
    # skeletons). Resolving it yields the fixed 1000-skeleton set; a ProblemSource samples X/y per the
    # recipe over those fixed skeletons (the held-out eval set, no generation).
    cat = build_catalog(str(ASSETS / "v23-val.yaml"))
    assert isinstance(cat, GenerativeCatalog)
    assert len(cat.skeletons) == 1000
    src = ProblemSource({"catalog": str(ASSETS / "v23-val.yaml"),
                         "sampling": {"n_support": 32, "n_validation": 0, "noise": 0.0}})
    from itertools import islice
    frozen = set(cat.skeletons)
    probs = [p for p in islice(iter(src), 5) if not p.is_placeholder]
    assert probs and all(tuple(p.skeleton) in frozen for p in probs)
