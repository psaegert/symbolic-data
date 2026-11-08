"""The catalog publish (tools/publish_catalogs.py): the gates a publish must pass before anything uploads, run against
a fake hosted manifest (no Hugging Face call, the catalog lint skipped)."""
import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools"
_SPEC = importlib.util.spec_from_file_location("publish_catalogs", TOOLS / "publish_catalogs.py")
pc = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(pc)


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """A curated catalog in git (keijzer@2), two data catalogs and a license pass that covers all three."""
    (tmp_path / "assets").mkdir()
    (tmp_path / "assets" / "keijzer.yaml").write_text("metadata:\n  name: keijzer\n  version: 2\n")
    (tmp_path / "build").mkdir()
    for name in ("esr-cosmology", "treloar-rubber"):
        np.savez(tmp_path / "build" / f"{name}.npz", _meta=np.array(json.dumps({"catalog": {"version": 1}})))
    licenses = [{"catalog": n, "redistribute": "yes", "license": "MIT", "upstream": "x"}
                for n in ("keijzer", "esr-cosmology", "treloar-rubber")]
    (tmp_path / "licenses.json").write_text(json.dumps(licenses))
    monkeypatch.setattr(pc, "LICENSES", str(tmp_path / "licenses.json"))
    monkeypatch.setitem(sys.modules, "lint_catalogs", types.SimpleNamespace(lint_paths=lambda: True))
    return tmp_path


def _publish(repo, monkeypatch, hosted, build="build", catalogs=None, **kw):
    (repo / "manifest.json").write_text(json.dumps(hosted))

    class FakeApi:
        def hf_hub_download(self, **_):
            return str(repo / "manifest.json")

    monkeypatch.setattr(pc, "HfApi", FakeApi)
    catalogs = catalogs or {"keijzer": ("keijzer.yaml", 7, "problem_catalog")}
    built = {n: (f"{n}.npz", 1, "problem_catalog") for n in ("esr-cosmology", "treloar-rubber")}
    pc.publish("fake/repo", str(repo / "assets"), catalogs, {}, built=built, build_dir=str(repo / build), **kw)


def test_built_data_catalogs_publish_as_new_versions(repo, monkeypatch, capsys):
    _publish(repo, monkeypatch, {})
    out = capsys.readouterr().out
    assert "new: esr-cosmology@1, keijzer@2, treloar-rubber@1" in out and "dry run" in out


def test_a_data_catalog_neither_built_nor_hosted_is_refused(repo, monkeypatch):
    with pytest.raises(SystemExit, match=r"not built and not hosted: \['esr-cosmology', 'treloar-rubber'\]"):
        _publish(repo, monkeypatch, {}, build="nothing-built")


def test_a_data_catalog_not_built_keeps_its_hosted_versions(repo, monkeypatch, capsys):
    hosted = {n: {"versions": {"1": {}}} for n in ("esr-cosmology", "treloar-rubber")}
    _publish(repo, monkeypatch, hosted, build="nothing-built")
    out = capsys.readouterr().out
    assert "not built here (hosted versions kept): ['esr-cosmology', 'treloar-rubber']" in out and "dry run" in out


def test_a_catalog_the_license_pass_does_not_clear_is_refused(repo, monkeypatch):
    with pytest.raises(SystemExit, match="the license pass does not clear"):
        _publish(repo, monkeypatch, {}, shareable=("yes, share-alike repo",))


def test_a_new_version_needs_a_license_row_or_a_hand_written_notice(repo, monkeypatch, capsys):
    (repo / "assets" / "fastsrb.yaml").write_text("metadata:\n  name: fastsrb\n  version: 3\n")
    catalogs = {"fastsrb": ("fastsrb.yaml", 120, "problem_catalog")}
    with pytest.raises(SystemExit, match=r"no license-pass row and no hand-written notice for \['fastsrb'\]"):
        _publish(repo, monkeypatch, {}, catalogs=catalogs)
    _publish(repo, monkeypatch, {}, catalogs=catalogs, hand_noticed=("fastsrb",))
    assert "dry run" in capsys.readouterr().out


def test_the_notices_name_every_catalog_they_vouch_for():
    text = (Path(pc.HF_CARD) / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
    assert pc.notices(text) == text, "run python tools/publish_catalogs.py --notices"
    assert set(pc.NOTICED_BEFORE_2026_11) <= pc.hand_noticed(text)
    assert {"fastsrb", "lample-charton-v24-bench"} <= pc.hand_noticed(text)        # '(@1, @2, @3)' dropped
    passed = pc.license_pass()
    published = {**pc.CATALOGS, **pc.DATA_CATALOGS}
    assert all(n in passed or n in pc.NOTICED_BEFORE_2026_11 for n in published)
