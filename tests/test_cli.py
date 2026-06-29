import os

import pytest
import yaml

from symbolic_data import load_catalog
from symbolic_data.__main__ import build_parser, main

REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
GEN_CONFIG = os.path.join(REPO_ROOT, "configs", "test", "skeleton_pool_train.yaml")


def test_cli_requires_a_subcommand():
    with pytest.raises(SystemExit):
        build_parser().parse_args([])


def test_materialize_generate_to_frozen_catalog(tmp_path):
    # A ProblemSource config (generate mode) -> materialize -> frozen .npz catalog.
    gen_cfg = yaml.safe_load(open(GEN_CONFIG, encoding="utf-8"))
    gen_cfg["size"] = 3
    cfg_path = tmp_path / "source.yaml"
    cfg_path.write_text(yaml.safe_dump({"generator": gen_cfg, "sampling": {"n_support": 8, "n_validation": 4}}), encoding="utf-8")
    out = tmp_path / "frozen.npz"

    main(["materialize", "-c", str(cfg_path), "-o", str(out)])

    assert out.exists()
    cat = load_catalog(str(out))
    assert cat.frozen and cat.problems is not None and len(cat.problems) >= 1
