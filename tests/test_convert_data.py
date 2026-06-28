import os

import pytest

from simplipy import SimpliPyEngine

from symbolic_data import SkeletonPool, ParserFactory
from symbolic_data.convert_data import FastSRBParser, NguyenParser, SOOSEParser, FeynmanParser

REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
FASTSRB_FIXTURE = os.path.join(os.path.dirname(__file__), "data", "fastsrb_mini.yaml")
BASE_CONFIG = os.path.join(REPO_ROOT, "configs", "test", "skeleton_pool_train.yaml")


def test_parser_factory_returns_expected_parsers():
    assert isinstance(ParserFactory.get_parser("fastsrb"), FastSRBParser)
    assert isinstance(ParserFactory.get_parser("nguyen"), NguyenParser)
    assert isinstance(ParserFactory.get_parser("soose"), SOOSEParser)
    assert isinstance(ParserFactory.get_parser("feynman"), FeynmanParser)


def test_parser_factory_unknown_raises():
    with pytest.raises(ValueError, match="Unknown parser"):
        ParserFactory.get_parser("does-not-exist")


def test_fastsrb_ingest_round_trip(tmp_path):
    pd = pytest.importorskip("pandas")
    import yaml

    with open(FASTSRB_FIXTURE, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    records = []
    for identifier, payload in raw.items():
        if not isinstance(payload, dict):
            continue
        record = {"id": identifier}
        record.update(payload)
        if record.get("prepared") is None:
            record["prepared"] = ""
        records.append(record)
    df = pd.DataFrame.from_records(records)

    base = SkeletonPool.from_config(BASE_CONFIG)
    engine = SimpliPyEngine.load("dev_7-3", install=True)
    pool = ParserFactory.get_parser("fastsrb").parse_data(df, engine, base)

    assert len(pool.skeletons) == len(records)
    assert len(pool.skeleton_codes) == len(records)

    # Headline observable: ingest produces a pool that round-trips through disk and stays usable.
    out = str(tmp_path / "fastsrb_pool")
    pool.save(out, config=BASE_CONFIG)
    _, loaded = SkeletonPool.load(out, verbose=False)
    assert len(loaded.skeletons) == len(records)
