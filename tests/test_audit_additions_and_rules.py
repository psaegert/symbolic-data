"""The 2026-11-27 audit's rulings in the catalog builders (tools/catalog_build): compilations kept whole (residue
members copied from the identical srbf problem), the rule for paper-vs-code conflicts (follow what produced the authors'
published results), the added sets (LLM-SR oscillators, BMS Rössler, Brum et al.), the overlap fingerprint, the problem
flags, and the metadata catalog_tags.py writes (stars from the deviations, notes, flags)."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

BUILD = Path(__file__).resolve().parents[1] / "tools" / "catalog_build"
REPO = BUILD.parents[1]


def _module(name):
    sys.path.insert(0, str(BUILD))
    spec = importlib.util.spec_from_file_location(name, BUILD / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


F = _module("build_formula_catalogs")
P = _module("paths")
T = _module("catalog_tags")


# ------------------------------------------------------------------------------------------ compilations kept whole
def test_every_residue_member_copies_an_existing_problem_under_its_own_id_and_subset():
    sets = F.residues()
    assert {s["catalog"] for s in sets.values()} == {
        "scrbenchmark", "dgsr-additional-feynman", "dgsr-feynman-d-2-d-5-subsets", "shape-constrained-sr-synthetic",
        "kan-feynman-no-units-subset", "rsrm-constant-optimization-benchmark"}
    member = sets["interrupted-4"]["members"][0]
    entry = F.copied(member, REPO / "assets/catalogs")
    source = yaml.safe_load((REPO / "assets/catalogs/feynman.yaml").read_text())["expressions"][member["copy_of"].split(":")[1]]
    assert entry["prepared"] == source["prepared"] and entry["vars"] == source["vars"]
    assert entry["subset"] == member["subset"] and entry["meta"]["same_as"] == member["copy_of"]


@pytest.mark.parametrize("catalog, total", [("scrbenchmark", 120), ("dgsr-additional-feynman", 32),
                                            ("dgsr-feynman-d-2-d-5-subsets", 15), ("shape-constrained-sr-synthetic", 19),
                                            ("rsrm-constant-optimization-benchmark", 97)])
def test_the_compilations_hold_their_published_size(catalog, total):
    doc = yaml.safe_load((REPO / f"assets/catalogs/{catalog}.yaml").read_text())
    assert len(doc["expressions"]) == total


# ------------------------------------------------------------------------------------------- paper vs code (data)
def _record(tag, eq_id):
    return next(p for p in P.final_problems() if p["set"] == tag and p["id"] == eq_id)


def test_dgsr_a32_and_ai_feynman_i_16_6_follow_the_code_and_the_data():
    a32 = F.fixed(_record("interrupted-5", "Feynman-A-32"))
    assert {(v["low"], v["high"]) for v in a32["variables"]} == {(1.0, 3.0)}
    i166 = F.fixed(_record("interrupted-6", "I.16.6"))
    variables, latent, _ = F.inputs_of(i166)
    assert variables["v1"]["expr"] == "u/c" and variables["v2"]["expr"] == "v/c"
    assert latent["c"]["sample_range"] == [1.0, 5.0]
    korns = F.fixed(_record("fresh-30", "Korns-15"))
    assert {(v["low"], v["high"]) for v in korns["variables"]} == {(0.0, 50.0)}


def test_a_patch_naming_an_input_the_record_lacks_is_refused(monkeypatch):
    monkeypatch.setitem(F.RULE_FIXES, ("fresh-30", "Korns-15"), {"variables": {"nope": {"low": 0.0}}})
    with pytest.raises(KeyError, match="no inputs"):
        F.fixed(_record("fresh-30", "Korns-15"))


def test_weber_fechner_keeps_the_authors_half_grid():
    from symbolic_data._evaluation import load_engine
    problem = F.points_problem(F.fixed(_record("critic-2", "weber_fechner_law")), "t-points", "t", load_engine(None))
    x = np.asarray(problem.x_support)
    assert len(x) == 5050 and np.all(x[:, 0] <= x[:, 1])


# ------------------------------------------------------------------------------------------------- the added sets
def test_the_added_sets_are_read_as_survey_records():
    added = P.additions()
    assert set(added["sets"]) == {"llm-sr-oscillators", "bms-rossler", "brum-2025"}
    by_set = {}
    for p in P.final_problems():
        by_set.setdefault(p["set"], []).append(p)
    assert len(by_set["llm-sr-oscillators"]) == 2 and len(by_set["bms-rossler"]) == 2 and len(by_set["brum-2025"]) == 28
    assert P.data_set_record("bms-rossler")["datasets"][0]["noise_sd"] == 5.0
    assert all(p.get("gt_reason") and p.get("gt_fvu_bound") for p in by_set["bms-rossler"])
    assert P.licenses()["bms-rossler"]["redistribute"] == "yes, restricted repo"


def test_the_dynamical_systems_generator_regenerates_the_recorded_bytes():
    import hashlib
    G = _module("sr_dynamical_systems")
    rows = G.generate()
    assert len(rows) == 28
    for rec in P.data_set_record("brum-2025")["datasets"]:
        for role, want in rec["arrays"].items():
            assert hashlib.sha256(np.ascontiguousarray(rows[rec["id"]][role]).tobytes()).hexdigest() == want["sha256"]


# ---------------------------------------------------------------------------------------------- overlap and flags
def test_the_law_fingerprint_ignores_renaming_and_rewriting_but_not_constants():
    ov = _module("overlap")
    assert ov.law_key("v1*v2 + sin(v1)") == ov.law_key("sin(v2) + v2*v1")
    assert ov.law_key("2*(v1 + v2)") == ov.law_key("2*v1 + 2*v2")
    assert ov.law_key("v1*v2 + sin(v1)") != ov.law_key("v1*v2 + 2*sin(v1)")


def test_flag_helpers():
    x = np.linspace(0.0, 1.0, 50)[:, None]
    assert T._single_input_r2(np.column_stack([x, np.ones_like(x)]), 3 * x[:, 0] + 1) > 0.999
    assert T._single_input_r2(x, np.sin(20 * x[:, 0])) < 0.5
    assert T._ordered(x, {"random": 0.2}) and not T._ordered(x, "given")
    assert set(T.FLAGS) == {"trivial", "low-signal", "target-leak", "tiny", "ordered"}


# ---------------------------------------------------------------------------- the metadata catalog_tags.py writes
def _yaml(tmp_path, display=None):
    head = "metadata:\n  name: toy\n" + (f"  display_name: {display}\n" if display else "") + "  version: 1\n"
    path = tmp_path / "toy.yaml"
    path.write_text(head + "  source_record: catalog-prep/sets/fresh-1.json\n  description: a toy\nexpressions:\n  e: {}\n")
    return path


def test_apply_writes_benchmark_star_notes_and_flags_and_removes_record_ids(tmp_path):
    path = _yaml(tmp_path)
    dev = {"changes": [{"kind": "filled", "problems": 1, "text": "x"}], "follows_first_publication": ["as Keijzer"],
           "paper_vs_code": ["the code's range"]}
    row = {"origin": "classic benchmark", "tags": ["srbf-modified"], "benchmark": "toy-bench", "srbf_changes": dev["changes"],
           "srbf_notes": T.notes_of(dev), "star": T.starred_by(dev), "problem_flags": {"trivial": ["e"]}}
    T.apply_yaml(path, row, {})
    md = yaml.safe_load(path.read_text())["metadata"]
    assert md["benchmark"] == "toy-bench" and md["display_name"] == "toy*" and md["problem_flags"] == {"trivial": ["e"]}
    assert md["srbf_notes"] == ["Follows the defining publication: as Keijzer", "Paper and code differ: the code's range"]
    assert "source_record" not in md and md["description"] == "a toy"
    T.apply_yaml(path, {**row, "star": False, "srbf_notes": [], "problem_flags": {}}, {})
    md = yaml.safe_load(path.read_text())["metadata"]
    assert "display_name" not in md and "srbf_notes" not in md and "problem_flags" not in md


def test_running_part_of_a_benchmark_earns_no_star():
    assert not T.starred_by({"changes": [{"kind": "part", "problems": 3, "text": "x"}]})
    assert all(T.starred_by({"changes": [{"kind": k, "problems": 1, "text": "x"}]}) for k in ("changed", "filled", "kept"))
