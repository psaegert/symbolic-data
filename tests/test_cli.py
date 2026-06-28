import os

import pytest

from symbolic_data.__main__ import build_parser, main

REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
FASTSRB_FIXTURE = os.path.join(os.path.dirname(__file__), "data", "fastsrb_mini.yaml")
BASE_CONFIG = os.path.join(REPO_ROOT, "configs", "test", "skeleton_pool_train.yaml")


def test_cli_requires_a_subcommand():
    with pytest.raises(SystemExit):
        build_parser().parse_args([])


def test_generate_skeleton_pool(tmp_path):
    out = str(tmp_path / "pool")
    main(["generate-skeleton-pool", "-c", BASE_CONFIG, "-o", out, "-s", "3"])
    assert os.path.exists(os.path.join(out, "skeletons.pkl"))


def test_import_data_fastsrb(tmp_path):
    pytest.importorskip("pandas")
    out = str(tmp_path / "fastsrb_pool")
    main(["import-data", "-i", FASTSRB_FIXTURE, "-b", BASE_CONFIG, "-p", "fastsrb", "-e", "dev_7-3", "-o", out])
    assert os.path.exists(os.path.join(out, "skeletons.pkl"))


def test_split_skeleton_pool(tmp_path):
    pool_dir = str(tmp_path / "pool")
    main(["generate-skeleton-pool", "-c", BASE_CONFIG, "-o", pool_dir, "-s", "4"])

    main(["split-skeleton-pool", "-i", pool_dir, "-t", "0.5", "-r", "0"])
    assert os.path.exists(os.path.join(pool_dir, "train", "skeletons.pkl"))
    assert os.path.exists(os.path.join(pool_dir, "val", "skeletons.pkl"))
