"""Tests for the level-1 ProblemCatalog + load_catalog (curated, vendored)."""
import pytest

from symbolic_data import CatalogEntry, ProblemCatalog, load_catalog
from symbolic_data import resolver as R

CURATED_COUNTS = {"fastsrb": 120, "feynman": 100, "nguyen": 12}


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    # Force the vendored package-data path (no network / manifest) for deterministic tests.
    monkeypatch.setattr(R, "fetch_manifest", lambda **kw: {})


@pytest.mark.parametrize("name,count", CURATED_COUNTS.items())
def test_load_curated_catalog_counts_and_metadata(name, count):
    cat = load_catalog(name)
    assert isinstance(cat, ProblemCatalog)
    assert cat.name == name
    assert cat.version == 1
    assert cat.source == "vendored"
    assert len(cat) == count
    assert len(list(cat.iter_expressions())) == count
    # metadata carried through (incl. sampling defaults + conventions note)
    assert cat.meta.get("name") == name
    assert "sampling_defaults" in cat.meta
    assert "conventions" in cat.meta


def test_entry_structure_and_vars():
    cat = load_catalog("feynman")
    entry = cat["I.6.2a"]
    assert isinstance(entry, CatalogEntry)
    assert entry.raw and entry.prepared
    assert "v1" in entry.variables
    v1 = entry.variables["v1"]
    assert "sample_range" in v1 and "sample_type" in v1
    # extra (non-structured) keys land in meta
    assert "sources" in entry.meta


def test_nguyen_sampling_default_is_20():
    cat = load_catalog("nguyen")
    assert cat.meta["sampling_defaults"]["n_points"] == 20


def test_to_from_yaml_roundtrip(tmp_path):
    cat = load_catalog("nguyen")
    out = tmp_path / "nguyen.yaml"
    cat.to_yaml(out)
    reloaded = ProblemCatalog.from_yaml(out)
    assert reloaded.name == cat.name and reloaded.version == cat.version
    assert set(reloaded.entries) == set(cat.entries)
    e0 = next(iter(cat.entries))
    assert reloaded[e0].raw == cat[e0].raw
    assert reloaded[e0].variables == cat[e0].variables


def test_flat_yaml_form_is_accepted():
    cat = ProblemCatalog.from_yaml({"Eq1": {"raw": "x1", "prepared": "v1", "vars": {"v1": {}}}}, name="adhoc")
    assert cat.name == "adhoc" and len(cat) == 1 and cat["Eq1"].raw == "x1"
