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
    assert set(pc.NOTICED_BEFORE_2026_11 + pc.RETIRED_HOSTED) <= pc.hand_noticed(text)
    assert {"fastsrb", "lample-charton-v24-bench"} <= pc.hand_noticed(text)        # '(@1, @2, @3)' dropped
    passed = pc.license_pass()
    published = {**pc.CATALOGS, **pc.DATA_CATALOGS}
    assert all(n in passed or n in pc.NOTICED_BEFORE_2026_11 for n in published)


def test_every_mit_bsd_or_apache_source_ships_its_copyright_notice_and_license_text(tmp_path, monkeypatch):
    assert pc.without_copyright() == []
    for source in pc.copyrights():
        assert source["license"] in pc.NOTICE_LICENSES and source["catalogs"], source["source"]
        assert source["copyright"] or source.get("note"), source["source"]       # a notice, or why there is none
    assert all((Path(pc.HF_CARD) / "LICENSES" / f"{lic}.txt").is_file() for lic in pc.NOTICE_LICENSES)
    # a catalog whose MIT source loses its entry is caught
    entries = [s for s in pc.copyrights() if s["source"] != "IBM/AI-Hilbert"]
    (tmp_path / "copyrights.json").write_text(json.dumps(entries))
    monkeypatch.setattr(pc, "COPYRIGHTS", str(tmp_path / "copyrights.json"))
    assert pc.without_copyright() == ["ai-hilbert-problem-set", "ai-hilbert-real-data"]
    # so is a hand-noticed catalog, and a retired one whose versions stay hosted
    entries = [s for s in pc.copyrights() if s["source"] not in ("viktmar/FastSRB", "dso-org/deep-symbolic-optimization")]
    (tmp_path / "copyrights.json").write_text(json.dumps(entries))
    missing = pc.without_copyright()
    assert "fastsrb" in missing and "srsd-dummy" not in missing                 # srsd-dummy keeps its SRSD entry
    assert {"grammarvae", "nonic", "pagie", "r-rationals", "sine", "keijzer"} <= set(missing)


def test_a_hosted_entry_the_notices_do_not_know_is_refused(repo, monkeypatch, capsys):
    hosted = {"old-catalog": {"versions": {"1": {}}}}
    with pytest.raises(SystemExit, match=r"hosted entries \['old-catalog'\] are neither published here nor listed"):
        _publish(repo, monkeypatch, hosted)
    _publish(repo, monkeypatch, hosted, retired=("old-catalog",))
    assert "preserved retired entries: ['old-catalog']" in capsys.readouterr().out


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sa = _load("publish_catalogs_sa", TOOLS / "publish_catalogs_sa.py")
restricted = _load("publish_catalogs_restricted", TOOLS / "publish_catalogs_restricted.py")


def test_each_repo_publishes_exactly_what_the_license_pass_routes_to_it():
    # owner 2026-11-09: licensed data in the main repo, share-alike and copyleft in the share-alike repo, data under
    # non-commercial terms or with no license in the restricted repo
    passed = pc.license_pass()
    tables = {"main": {**pc.CATALOGS, **pc.DATA_CATALOGS}, "sa": {**sa.CATALOGS, **sa.DATA_CATALOGS},
              "restricted": restricted.DATA_CATALOGS}
    routes = {"main": pc.SHAREABLE, "sa": sa.ROUTE, "restricted": restricted.ROUTE}
    for repo, names in tables.items():
        assert all(passed[n]["redistribute"] in routes[repo] for n in names if n in passed), repo
    for repo in ("sa", "restricted"):                                    # nothing routed to a repo is left behind
        assert {n for n, r in passed.items() if r["redistribute"] in routes[repo]} <= set(tables[repo]), repo
    assert not set(tables["main"]) & set(tables["restricted"]) and not set(tables["main"]) & set(tables["sa"])


@pytest.mark.parametrize("route, shareable", [("yes, restricted repo", pc.SHAREABLE),
                                              ("yes", ("yes, restricted repo",)),
                                              ("yes, share-alike repo", ("yes, restricted repo",))])
def test_a_repo_refuses_a_catalog_routed_to_another(repo, monkeypatch, route, shareable):
    rows = json.loads((repo / "licenses.json").read_text())
    rows = [dict(r, redistribute=route if r["catalog"] == "esr-cosmology" else r["redistribute"]) for r in rows]
    (repo / "licenses.json").write_text(json.dumps(rows))
    with pytest.raises(SystemExit, match=r"the license pass does not clear \['esr-cosmology"):
        _publish(repo, monkeypatch, {}, shareable=shareable)


def test_a_first_publish_starts_from_an_empty_manifest(repo, monkeypatch, capsys):
    from huggingface_hub.errors import RepositoryNotFoundError

    class NoRepoApi:
        def hf_hub_download(self, **_):
            raise RepositoryNotFoundError.__new__(RepositoryNotFoundError)        # as the hub answers a missing repo

    monkeypatch.setattr(pc, "HfApi", NoRepoApi)
    built = {n: (f"{n}.npz", 1, "problem_catalog") for n in ("esr-cosmology", "treloar-rubber")}
    pc.publish("fake/new-repo", str(repo / "assets"), {}, {}, built=built, build_dir=str(repo / "build"))
    out = capsys.readouterr().out
    assert "hosted manifest (fake/new-repo): 0 entries" in out and "new: esr-cosmology@1, treloar-rubber@1" in out


def test_the_restricted_notices_list_every_dataset_with_its_terms():
    text = (Path(restricted.CARD_DIR) / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
    assert pc.notices(text, names=restricted.DATA_CATALOGS, terms=True) == text, \
        "run python tools/publish_catalogs_restricted.py --notices"
    assert pc.without_copyright(names=restricted.DATA_CATALOGS) == []
    passed = pc.license_pass()
    for name in restricted.DATA_CATALOGS:
        row = passed[name]
        for mid in row.get("members") or {"all": None}:
            assert f"| {name} | {mid} |" in text, (name, mid)
    for card in (Path(restricted.CARD_DIR) / "README.md", Path(pc.HF_CARD) / "README.md"):
        assert "open an issue at\nhttps://github.com/psaegert/symbolic-data" in card.read_text().replace(
            "open an issue at https://", "open an issue at\nhttps://")


def test_restricted_members_carry_their_terms_and_the_statement():
    build = _load("build_data_catalogs", TOOLS / "catalog_build" / "build_data_catalogs.py")
    problems = [types.SimpleNamespace(eq_id=i, meta={}) for i in ("ok", "sa", "nc", "out")]
    members = {"ok": {"license": "MIT", "share": "yes"}, "sa": {"license": "GPL-3.0", "share": "share-alike"},
               "nc": {"license": "CC BY-NC 4.0 (x)", "share": "restricted", "terms": "non-commercial use only"},
               "out": {"license": "NIST SRD", "share": "no", "reason": "terms forbid redistribution"}}
    parts, left_out = build.split_members(problems, {"catalog": "c", "members": members})
    assert [[p.eq_id for p in parts[k]] for k in ("", "-share-alike", "-restricted")] == [["ok"], ["sa"], ["nc"]]
    assert left_out == ["out: terms forbid redistribution"]
    nc = parts["-restricted"][0].meta
    assert nc["terms"] == "non-commercial use only" and "CC BY-NC 4.0 (x)" in nc["statement"]
    assert "github.com/psaegert/symbolic-data" in nc["statement"] and "terms" not in parts[""][0].meta
    whole = [types.SimpleNamespace(eq_id="a", meta={})]
    parts, _ = build.split_members(whole, {"catalog": "w", "redistribute": "yes, restricted repo", "terms": "none stated",
                                           "license": "no licence", "upstream": "somewhere"})
    assert parts[""][0].meta["terms"] == "none stated" and "Source: somewhere" in parts[""][0].meta["statement"]


def _built(repo, name, problems):
    np.savez(repo / "build" / f"{name}.npz",
             _meta=np.array(json.dumps({"catalog": {"version": 1}, "problems": problems})))


def _route(repo, name, **row):
    rows = [r for r in json.loads((repo / "licenses.json").read_text()) if r["catalog"] != name]
    (repo / "licenses.json").write_text(json.dumps(rows + [{"catalog": name, "license": "x", "upstream": "y", **row}]))


def _publish_built(repo, monkeypatch, names, shareable=pc.SHAREABLE):
    (repo / "manifest.json").write_text("{}")
    monkeypatch.setattr(pc, "HfApi", lambda: types.SimpleNamespace(hf_hub_download=lambda **_: str(repo / "manifest.json")))
    pc.publish("fake/repo", str(repo / "assets"), {}, {}, built={n: (f"{n}.npz", 1, "problem_catalog") for n in names},
               build_dir=str(repo / "build"), shareable=shareable)


def test_a_stale_unsplit_catalog_is_refused(repo, monkeypatch, capsys):
    # an 11-05-style build of a set whose datasets the license pass now splits: every member in the set's own catalog
    members = {"a": {"license": "MIT", "share": "yes"}, "b": {"license": "none", "share": "restricted", "terms": "none stated"}}
    _route(repo, "esr-cosmology", redistribute="yes, these members", members=members)
    _built(repo, "esr-cosmology", [{"eq_id": "a", "meta": {}}, {"eq_id": "b", "meta": {}}])
    with pytest.raises(SystemExit, match=r"routes elsewhere.*'esr-cosmology': \['b: routed to restricted'\]"):
        _publish_built(repo, monkeypatch, ["esr-cosmology"])
    _built(repo, "esr-cosmology", [{"eq_id": "a", "meta": {}}])                          # split as the pass routes it
    _publish_built(repo, monkeypatch, ["esr-cosmology"])
    assert "dry run" in capsys.readouterr().out


def test_restricted_data_carry_their_statement_and_stay_in_the_restricted_repo(repo, monkeypatch, capsys):
    _route(repo, "treloar-rubber", redistribute="yes, restricted repo", terms="none stated")
    restricted_route = ("yes, restricted repo",)
    _built(repo, "treloar-rubber", [{"eq_id": "t", "meta": {"terms": "none stated"}}])
    with pytest.raises(SystemExit, match="restricted data without terms and statement"):
        _publish_built(repo, monkeypatch, ["treloar-rubber"], restricted_route)
    _built(repo, "treloar-rubber", [{"eq_id": "t", "meta": {"terms": "none stated", "statement": "srbf republishes ..."}}])
    _publish_built(repo, monkeypatch, ["treloar-rubber"], restricted_route)
    assert "dry run" in capsys.readouterr().out
    # restricted data in a catalog routed to the main repo
    _built(repo, "esr-cosmology", [{"eq_id": "e", "meta": {"terms": "non-commercial use only", "statement": "..."}}])
    with pytest.raises(SystemExit, match=r"'esr-cosmology': \['e: restricted data \(non-commercial use only\)'\]"):
        _publish_built(repo, monkeypatch, ["esr-cosmology"])
