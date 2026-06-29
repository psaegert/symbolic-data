"""Tests for the repo-agnostic versioned-artifact resolver."""
import hashlib
import json

import pytest

from symbolic_data import resolver as R


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def catalog_file(tmp_path):
    f = tmp_path / "feynman_v1.yaml"
    f.write_text("metadata: {name: feynman, version: 1}\nI.6.2a: {raw: x}\n", encoding="utf-8")
    return f


@pytest.fixture
def manifest(catalog_file):
    return {
        "feynman": {
            "type": "problem-catalog",
            "repo_id": "psaegert/symbolic-data-assets",
            "default_version": 2,
            "versions": {
                "1": {"directory": "catalogs/feynman/v1", "files": ["catalog.yaml"],
                      "sha256": {"catalog.yaml": _sha256(catalog_file)}, "revision": "abc123"},
                "2": {"directory": "catalogs/feynman/v2", "files": ["catalog.yaml"],
                      "sha256": {"catalog.yaml": _sha256(catalog_file)}, "revision": "def456"},
            },
        }
    }


# --- ref parsing / local-path detection (the shadowing bug-fix) --------------------------------

def test_looks_like_path_only_for_explicit_paths():
    assert not R._looks_like_path("feynman")              # bare manifest name: NOT a path
    assert not R._looks_like_path("feynman@2")
    assert R._looks_like_path("./feynman.yaml")
    assert R._looks_like_path("/abs/feynman.yaml")
    assert R._looks_like_path("dir/feynman.yaml")
    assert R._looks_like_path("feynman.yaml")             # bare but has a known suffix


def test_parse_ref_forms():
    assert R._parse_ref("feynman") == (None, "feynman", None)
    assert R._parse_ref("feynman@3") == (None, "feynman", 3)
    assert R._parse_ref("user/repo:feynman@3") == ("user/repo", "feynman", 3)
    assert R._parse_ref("user/repo:feynman") == ("user/repo", "feynman", None)


# --- local path -------------------------------------------------------------------------------

def test_resolve_local_path(catalog_file):
    art = R.resolve(str(catalog_file))
    assert art.source == "local" and art.path == str(catalog_file) and art.name == "feynman_v1"


def test_resolve_missing_local_path_raises(tmp_path):
    with pytest.raises(R.ResolverError):
        R.resolve(str(tmp_path / "nope.yaml"))


# --- manifest resolution + integrity ----------------------------------------------------------

def test_resolve_name_at_version(monkeypatch, manifest, catalog_file):
    monkeypatch.setattr(R, "fetch_manifest", lambda **kw: manifest)
    monkeypatch.setattr("huggingface_hub.hf_hub_download", lambda **kw: str(catalog_file))
    art = R.resolve("feynman@1")
    assert art.source == "huggingface" and art.version == 1 and art.revision == "abc123"
    assert art.path == str(catalog_file)


def test_resolve_bare_name_uses_default_version(monkeypatch, manifest, catalog_file):
    monkeypatch.setattr(R, "fetch_manifest", lambda **kw: manifest)
    monkeypatch.setattr("huggingface_hub.hf_hub_download", lambda **kw: str(catalog_file))
    art = R.resolve("feynman")
    assert art.version == 2 and art.revision == "def456"


def test_sha256_mismatch_raises(monkeypatch, manifest, catalog_file):
    manifest["feynman"]["versions"]["1"]["sha256"]["catalog.yaml"] = "0" * 64
    monkeypatch.setattr(R, "fetch_manifest", lambda **kw: manifest)
    monkeypatch.setattr("huggingface_hub.hf_hub_download", lambda **kw: str(catalog_file))
    with pytest.raises(R.ResolverError, match="sha256 mismatch"):
        R.resolve("feynman@1", vendored_fallback=False)


def test_third_party_repo_override_is_used(monkeypatch, manifest, catalog_file):
    seen = {}

    def fake_fetch(repo_id=None, manifest_filename=None):
        seen["repo_id"] = repo_id
        return manifest

    monkeypatch.setattr(R, "fetch_manifest", fake_fetch)
    monkeypatch.setattr("huggingface_hub.hf_hub_download", lambda **kw: str(catalog_file))
    R.resolve("someuser/their-assets:feynman@1")
    assert seen["repo_id"] == "someuser/their-assets"


# --- vendored fallback ------------------------------------------------------------------------

def test_vendored_fallback_when_manifest_empty(monkeypatch, catalog_file):
    monkeypatch.setattr(R, "fetch_manifest", lambda **kw: {})
    monkeypatch.setattr(R, "_vendored_path", lambda name: str(catalog_file))
    art = R.resolve("feynman")
    assert art.source == "vendored" and art.path == str(catalog_file)


def test_unknown_name_no_vendored_raises(monkeypatch):
    monkeypatch.setattr(R, "fetch_manifest", lambda **kw: {})
    monkeypatch.setattr(R, "_vendored_path", lambda name: None)
    with pytest.raises(R.ResolverError):
        R.resolve("does_not_exist")
