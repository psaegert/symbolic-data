"""What srbf's release leaves out by rule (tools/catalog_build/tags/dropped.json; scope 2026-11-27: y = f(x) on one data
set; formulas as their authors wrote them, 2026-11-26): every entry has its reason, a set's parts go with it, and no
dropped catalog is left in the repository."""
import importlib.util
import sys
from pathlib import Path

BUILD = Path(__file__).resolve().parents[1] / "tools" / "catalog_build"
sys.path.insert(0, str(BUILD))
_SPEC = importlib.util.spec_from_file_location("paths", BUILD / "paths.py")
P = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(P)


def test_the_drop_list_names_every_reason_and_parts_go_with_their_set():
    rows = P.dropped()
    assert all(r["reason"] and r["kind"] in ("scope", "formula as written") for r in rows.values())
    assert P.dropped_whole("sldbench") and P.dropped_whole("mvsr-2024")
    assert P.dropped_whole("physo-class") and P.dropped_whole("newtonbench")
    assert not P.dropped_whole("phye2e-space-physics") and not P.dropped_whole("phye2e-space-physics-restricted")
    assert len(P.dropped_members("phye2e-space-physics")) == 12
    assert P.dropped_members("heuristiclab-regression-instance-providers") == {"Breiman-I"}
    assert not P.dropped_whole("cvgp-trigonometric-control-variable-datasets")      # CVGP stays (2026-11-27)


def test_no_dropped_catalog_is_left_in_the_repository():
    repo = BUILD.parents[1]
    files = {p.stem for folder in ("assets/catalogs", "assets_sa/catalogs") for p in (repo / folder).glob("*.*")}
    assert not {c for c in files if P.dropped_whole(c)}
