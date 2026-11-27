"""catalog_metadata: a catalog's metadata without its problems; and every release catalog in the repository names its
benchmark (srbf averages by benchmark: the -restricted, -share-alike, -points and formula/data parts of one benchmark
share one slug; tools/catalog_build/tags/benchmarks.json)."""
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from symbolic_data import Problem, ProblemCatalog, catalog_metadata

REPO = Path(__file__).resolve().parents[1]
BENCHMARKS = json.loads((REPO / "tools/catalog_build/tags/benchmarks.json").read_text())


def test_a_yaml_catalogs_metadata_is_read_without_its_expressions(tmp_path):
    doc = {"metadata": {"name": "toy", "version": 3, "benchmark": "toy-bench", "tags": {"origin": "x", "tags": []}},
           "expressions": {"e1": {"raw": "x", "prepared": "v1", "n_variables": 1, "vars": {"v1": {"name": "x"}}}}}
    path = tmp_path / "toy.yaml"
    path.write_text(yaml.safe_dump(doc, sort_keys=False) + "  this line would not parse: [\n")   # never read
    assert catalog_metadata(path) == doc["metadata"]


def test_a_frozen_catalogs_metadata_is_its_catalog_record(tmp_path):
    x = np.linspace(1.0, 2.0, 8).reshape(-1, 1)
    p = Problem.from_data(x, 2.0 * x[:, 0], eq_id="p")
    path = ProblemCatalog.from_problems([p], name="frozen-toy", version=2, meta={"benchmark": "toy-bench"}).save(
        tmp_path / "frozen-toy.npz")
    m = catalog_metadata(path)
    assert m["name"] == "frozen-toy" and m["version"] == 2 and m["benchmark"] == "toy-bench"


def _release_files():
    for folder in ("assets/catalogs", "assets_sa/catalogs"):
        for path in sorted((REPO / folder).glob("*")):
            if path.suffix == ".yaml" and not path.stem.startswith("lample-charton") and path.name != "fastsrb.yaml":
                yield path
            elif path.suffix == ".npz" and path.stem.endswith("-points"):
                yield path


def test_every_release_catalog_in_the_repository_names_its_benchmark():
    of = {c: b["benchmark"] for b in BENCHMARKS for c in b["catalogs"]}
    files = list(_release_files())
    assert len(files) > 100
    for path in files:
        meta = catalog_metadata(path)
        assert meta.get("benchmark") == of.get(meta["name"]), path.name
        assert "source_record" not in meta, path.name


@pytest.mark.parametrize("parts", [("gecco-2013-2017-real", "gecco-2013-2017-real-restricted",
                                    "gecco-2013-2017-real-share-alike"),
                                   ("keijzer", "keijzer-points"), ("pysr-empiricalbench", "pysr-empiricalbench-formulas")])
def test_the_parts_of_one_benchmark_share_its_slug(parts):
    of = {c: b["benchmark"] for b in BENCHMARKS for c in b["catalogs"]}
    assert len({of[p] for p in parts}) == 1
