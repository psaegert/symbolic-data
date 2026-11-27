"""Frozen draws (symbolic_data.draws): every problem's data rendered once, reproducibly, with a manifest."""
import json

import numpy as np
import pytest
import yaml

from symbolic_data import ProblemSource, data_problem
from symbolic_data.__main__ import main
from symbolic_data.catalog import ProblemCatalog
from symbolic_data.draws import (ATTEMPTS_FORMAT, FORMAT, MANIFEST, CatalogDraws, DrawsSettings, load_attempts,
                                 needs_check, read_meta, render, render_catalog, seed_sequence, sha256_file, verify)
from symbolic_data.source import _inject_noise

UNIT = {"sample_range": [1.0, 2.0], "sample_type": ["uni", "pos"]}
SETTINGS = DrawsSettings(n_support=16, n_validation=8, simplipy_engine="acj-4")


@pytest.fixture(scope="module")
def engine():
    from simplipy import SimpliPyEngine
    return SimpliPyEngine.load("acj-4", install=True)


@pytest.fixture
def formula_catalog(tmp_path):
    """Three entries: one drawn whole, one with its own test rows, one with a per-run constant."""
    doc = {"metadata": {"name": "toy", "version": 3}, "expressions": {
        "a": {"prepared": "v1 * v2 + 2.5", "vars": {"v1": dict(UNIT), "v2": dict(UNIT)}},
        "b": {"prepared": "sin(v1)", "vars": {"v1": dict(UNIT)}, "test": {"vars": {"v1": {"sample_range": [5.0, 6.0], "sample_type": ["uni", "pos"]}}}},
        "c": {"prepared": "1.5 * v1", "vars": {"v1": dict(UNIT)},
              "per_run": {"params": {"c1": dict(UNIT)}, "prepared": "c1 * v1"}, "test": {}},
    }}
    path = tmp_path / "toy.yaml"
    path.write_text(yaml.safe_dump(doc), encoding="utf-8")
    return str(path)


def _rows(n, start=0):
    x = np.column_stack([np.arange(start, start + n, dtype=float), np.full(n, 0.5)])
    return x, 2.0 * x[:, 0] + 1.0


@pytest.fixture
def data_catalog(tmp_path):
    """Data problems under each split rule, with pools larger than the caps."""
    x, y = _rows(40)
    xt, yt = _rows(30, start=1000)
    problems = [
        data_problem(x, y, x_test=xt, y_test=yt, variables=["x1", "x2"], test_split="given", eq_id="given", gt_kind="none"),
        data_problem(x, y, variables=["x1", "x2"], test_split={"random": 0.25}, eq_id="random", gt_kind="none"),
        data_problem(x, y, variables=["x1", "x2"], test_split={"folds": [list(range(0, 20)), list(range(20, 40))]},
                     eq_id="folds", gt_kind="none"),
        data_problem(x[:10], y[:10], variables=["x1", "x2"], test_split="none", eq_id="small", gt_kind="none"),
    ]
    return str(ProblemCatalog.from_problems(problems, name="toy-data").save(tmp_path / "toy-data.npz"))


def _arrays(path, index):
    with np.load(path, allow_pickle=False) as data:
        return {k.split("__", 1)[1]: data[k] for k in data.files if k.startswith(f"p{index}__")}


def test_a_render_writes_one_draw_per_problem_and_a_manifest(formula_catalog, tmp_path):
    manifest = render([formula_catalog], tmp_path / "draws", SETTINGS, log=lambda _: None)
    entry = manifest["catalogs"]["toy"]
    assert manifest["format"] == FORMAT and manifest["repeats"] == [1] and list(entry["files"]) == ["1"]
    assert manifest["settings"]["n_support"] == 16 and manifest["settings"]["simplipy_engine"] == "acj-4"
    assert entry["version"] == 3 and entry["problems"] == 3 and entry["kinds"] == {"formula": 3}
    assert entry["source_sha256"] == sha256_file(formula_catalog) and entry["source"] == "toy.yaml"
    for repeat in ("1",):
        info = entry["files"][repeat]
        path = tmp_path / "draws" / info["path"]
        assert info["path"] == f"toy/repeat-{repeat}.npz" and sha256_file(path) == info["sha256"]
        meta = read_meta(path)
        assert meta["repeat"] == int(repeat) and meta["catalog"]["source_sha256"] == entry["source_sha256"]
        assert [p["eq_id"] for p in meta["problems"]] == ["a", "b", "c"] and [p["row"] for p in meta["problems"]] == [0, 1, 2]
        arrays = _arrays(path, 0)
        assert arrays["x_support"].shape == (16, 2) and arrays["x_validation"].shape == (8, 2)
        assert arrays["noise_support"].shape == (16, 1) and arrays["noise_validation"].shape == (8, 1)
        assert all(a.dtype == np.float64 for a in arrays.values())
        np.testing.assert_allclose(arrays["y_support"].ravel(), np.prod(arrays["x_support"], axis=1) + 2.5)
    on_disk = json.loads((tmp_path / "draws" / MANIFEST).read_text())
    assert on_disk == manifest


def test_a_second_render_writes_the_same_bytes(formula_catalog, data_catalog, tmp_path):
    first = render([formula_catalog, data_catalog], tmp_path / "one", SETTINGS, log=lambda _: None)
    second = render([formula_catalog, data_catalog], tmp_path / "two", SETTINGS, jobs=2, log=lambda _: None)
    assert first == second
    assert (tmp_path / "one" / MANIFEST).read_bytes() == (tmp_path / "two" / MANIFEST).read_bytes()


def test_the_per_run_values_are_frozen_with_the_rows(formula_catalog, tmp_path):
    render([formula_catalog], tmp_path, SETTINGS, log=lambda _: None)
    path = tmp_path / "toy" / "repeat-1.npz"
    record = read_meta(path)["problems"][2]
    c1 = record["meta"]["per_run_values"]["c1"]
    arrays = _arrays(path, 2)
    # the fitting and the test rows of the draw share its value, and its ground truth is written with it
    assert 1.0 <= c1 <= 2.0
    np.testing.assert_allclose(arrays["y_support"].ravel(), c1 * arrays["x_support"][:, 0])
    np.testing.assert_allclose(arrays["y_validation"].ravel(), c1 * arrays["x_validation"][:, 0])
    assert repr(c1) in record["expression"] and record["constants"] == [c1]


def test_a_further_repeat_is_a_draw_of_its_own(formula_catalog, data_catalog, tmp_path):
    """The format keeps a repeat index: a render that asks for repeat 2 adds an independent draw beside repeat 1."""
    manifest = render([formula_catalog, data_catalog], tmp_path, DrawsSettings(n_support=16, n_validation=8, repeats=(1, 2),
                                                                               simplipy_engine="acj-4"), log=lambda _: None)
    assert manifest["repeats"] == [1, 2] and list(manifest["catalogs"]["toy"]["files"]) == ["1", "2"]
    one, two = (read_meta(tmp_path / f"toy/repeat-{r}.npz")["problems"][2] for r in (1, 2))
    assert one["meta"]["per_run_values"] != two["meta"]["per_run_values"]
    first, second = _arrays(tmp_path / "toy/repeat-1.npz", 0), _arrays(tmp_path / "toy/repeat-2.npz", 0)
    assert not np.array_equal(first["x_support"], second["x_support"])
    assert not np.array_equal(first["noise_support"], second["noise_support"])
    split_one, split_two = (_arrays(tmp_path / f"toy-data/repeat-{r}.npz", 1) for r in (1, 2))
    assert not np.array_equal(split_one["x_validation"], split_two["x_validation"])     # each repeat its own split
    alone = render([formula_catalog], tmp_path / "one", SETTINGS, log=lambda _: None)
    assert alone["catalogs"]["toy"]["files"]["1"]["sha256"] == manifest["catalogs"]["toy"]["files"]["1"]["sha256"]


def test_a_test_block_draws_the_held_out_rows_apart(formula_catalog, tmp_path):
    render([formula_catalog], tmp_path, SETTINGS, log=lambda _: None)
    arrays = _arrays(tmp_path / "toy/repeat-1.npz", 1)
    assert arrays["x_support"].min() >= 1.0 and arrays["x_support"].max() <= 2.0
    assert arrays["x_validation"].min() >= 5.0 and arrays["x_validation"].max() <= 6.0


def test_a_problem_draws_the_same_whatever_else_is_rendered(formula_catalog, tmp_path):
    render([formula_catalog], tmp_path / "all", SETTINGS, log=lambda _: None)
    manifest = render([formula_catalog], tmp_path / "one", SETTINGS, ids={"toy": ["c"]}, log=lambda _: None)
    assert manifest["catalogs"]["toy"]["problems"] == 1 and manifest["catalogs"]["toy"]["ids"]["count"] == 1
    alone, together = _arrays(tmp_path / "one/toy/repeat-1.npz", 0), _arrays(tmp_path / "all/toy/repeat-1.npz", 2)
    assert alone.keys() == together.keys()
    for key in alone:
        np.testing.assert_array_equal(alone[key], together[key])
    assert read_meta(tmp_path / "one/toy/repeat-1.npz")["problems"][0]["row"] == 2


def test_the_draw_is_the_source_draw_from_the_derived_seed(formula_catalog, tmp_path, engine):
    render([formula_catalog], tmp_path, SETTINGS, log=lambda _: None)
    points, noise = (np.random.default_rng(s) for s in seed_sequence(SETTINGS.seed_root, "toy", "a", 1).spawn(2))
    problem = ProblemSource({"catalog": formula_catalog, "sampling": SETTINGS.sampling()}, simplipy_engine=engine).draw(0, points)
    arrays = _arrays(tmp_path / "toy/repeat-1.npz", 0)
    np.testing.assert_array_equal(arrays["x_support"], problem.x_support)
    np.testing.assert_array_equal(arrays["y_validation"], problem.y_validation)
    np.testing.assert_array_equal(arrays["noise_support"], noise.standard_normal((16, 1)))


def test_the_noise_vector_reproduces_the_scalar_noise_of_a_source():
    """A noise level s scales the stored vector as the source scales a fresh one: y + s * std(y) * noise."""
    y = np.linspace(-3.0, 7.0, 32).reshape(-1, 1)
    for level in (0.0, 0.01, 0.1):
        vector = np.random.default_rng(5).standard_normal(y.shape)
        expected = y.copy() if level == 0.0 else y + level * float(np.std(y)) * vector
        np.testing.assert_array_equal(_inject_noise(y, level, np.random.default_rng(5)), expected)


def test_data_problems_freeze_one_draw_of_their_pools(data_catalog, tmp_path):
    manifest = render([data_catalog], tmp_path, SETTINGS, log=lambda _: None)
    assert manifest["catalogs"]["toy-data"]["kinds"] == {"data": 4}
    records = read_meta(tmp_path / "toy-data/repeat-1.npz")["problems"]
    given = _arrays(tmp_path / "toy-data/repeat-1.npz", 0)
    assert given["x_support"].shape == (16, 2) and given["x_validation"].shape == (8, 2)
    assert set(given["x_validation"][:, 0]) <= set(range(1000, 1030))                    # the authors' test rows
    assert "noise_support" not in given and "y_support_noisy" not in given                # data is not noised
    np.testing.assert_array_equal(given["y_support"].ravel(), 2.0 * given["x_support"][:, 0] + 1.0)
    folds = _arrays(tmp_path / "toy-data/repeat-1.npz", 2)
    test_rows = set(folds["x_validation"][:, 0].astype(int))
    assert test_rows <= (set(range(0, 20)) if records[2]["meta"]["fold"] == 1 else set(range(20, 40)))
    fit_rows = set(folds["x_support"][:, 0].astype(int))
    assert not fit_rows & test_rows and len(fit_rows) == 16                              # the other fold, capped
    split = _arrays(tmp_path / "toy-data/repeat-1.npz", 1)
    assert split["x_validation"].shape == (8, 2) and records[1]["meta"]["n_test_pool"] == 10   # ceil(0.25 * 40)
    small = _arrays(tmp_path / "toy-data/repeat-1.npz", 3)
    np.testing.assert_array_equal(small["x_support"], _rows(10)[0])                       # a pool under its cap: whole
    assert small["x_validation"].shape == (0, 2)


def test_a_stored_problem_keeps_its_noisy_targets(tmp_path):
    from symbolic_data.problem import Problem
    x, y = _rows(6)
    y = y.reshape(-1, 1)
    stored = Problem(x_support=x, y_support=y, y_support_noisy=y + 0.5, x_validation=x[:2], y_validation=y[:2],
                     y_validation_noisy=y[:2].copy(), skeleton=None, expression=None, constants=[], variables=["x1", "x2"],
                     complexity=None, eq_id="s")
    path = str(ProblemCatalog.from_problems([stored], name="stored").save(tmp_path / "stored.npz"))
    manifest = render([path], tmp_path / "draws", SETTINGS, log=lambda _: None)
    assert manifest["catalogs"]["stored"]["kinds"] == {"stored": 1}
    arrays = _arrays(tmp_path / "draws/stored/repeat-1.npz", 0)
    np.testing.assert_array_equal(arrays["y_support_noisy"], y + 0.5)
    assert "y_validation_noisy" not in arrays


def test_a_placeholder_is_recorded_without_arrays(tmp_path):
    doc = {"metadata": {"name": "bad"}, "expressions": {"nan": {"prepared": "log(v1)", "vars": {
        "v1": {"sample_range": [-2.0, -1.0], "sample_type": ["uni", "pos"]}}}}}
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(doc), encoding="utf-8")
    manifest = render([str(path)], tmp_path / "draws", DrawsSettings(n_support=4, n_validation=2, max_trials=2,
                                                                     simplipy_engine="acj-4"), log=lambda _: None)
    assert manifest["catalogs"]["bad"]["placeholders"] == {"1": 1}
    record = read_meta(tmp_path / "draws/bad/repeat-1.npz")["problems"][0]
    assert record["is_placeholder"] and record["placeholder_reason"] and _arrays(tmp_path / "draws/bad/repeat-1.npz", 0) == {}


def test_verify_finds_a_changed_or_missing_file_and_reproduces_a_render(formula_catalog, data_catalog, tmp_path):
    render([formula_catalog, data_catalog], tmp_path, SETTINGS, log=lambda _: None)
    assert verify(tmp_path) == []
    assert verify(tmp_path, [formula_catalog, data_catalog]) == []
    assert not any(p.name.startswith(".verify") for p in tmp_path.iterdir())
    target = tmp_path / "toy/repeat-1.npz"
    content = bytearray(target.read_bytes())
    content[len(content) // 2] ^= 0xFF
    target.write_bytes(bytes(content))
    (tmp_path / "toy-data/repeat-1.npz").unlink()
    assert sorted(verify(tmp_path)) == ["toy repeat 1: toy/repeat-1.npz does not match its sha256",
                                        "toy-data repeat 1: toy-data/repeat-1.npz is missing"]


def test_verify_reports_a_render_that_does_not_reproduce(formula_catalog, tmp_path):
    render([formula_catalog], tmp_path / "draws", SETTINGS, log=lambda _: None)
    changed = yaml.safe_load(open(formula_catalog))
    changed["expressions"]["a"]["prepared"] = "v1 * v2 + 2.75"
    other = tmp_path / "toy.yaml"
    other.write_text(yaml.safe_dump(changed), encoding="utf-8")
    problems = verify(tmp_path / "draws", [str(other)])
    assert "toy: the catalog's source differs from the rendered one" in problems
    assert any("a second render writes other bytes" in p for p in problems)


def test_a_rendered_root_is_refused_unless_updated_with_its_settings(formula_catalog, data_catalog, tmp_path):
    render([formula_catalog], tmp_path, SETTINGS, log=lambda _: None)
    with pytest.raises(FileExistsError):
        render([data_catalog], tmp_path, SETTINGS, log=lambda _: None)
    with pytest.raises(ValueError, match="settings"):
        render([data_catalog], tmp_path, DrawsSettings(n_support=32, n_validation=8, simplipy_engine="acj-4"),
               update=True, log=lambda _: None)
    manifest = render([data_catalog], tmp_path, SETTINGS, update=True, log=lambda _: None)
    assert sorted(manifest["catalogs"]) == ["toy", "toy-data"] and verify(tmp_path) == []


def test_two_catalogs_of_one_name_are_refused(formula_catalog, tmp_path):
    copy = tmp_path / "copy" / "toy.yaml"
    copy.parent.mkdir()
    copy.write_text(open(formula_catalog).read(), encoding="utf-8")
    with pytest.raises(ValueError, match="two catalogs are named 'toy'"):
        render([formula_catalog, str(copy)], tmp_path / "draws", SETTINGS, log=lambda _: None)


def test_unknown_ids_and_generative_catalogs_are_refused(formula_catalog, tmp_path):
    with pytest.raises(ValueError, match="ids not in the catalog"):
        render_catalog(formula_catalog, tmp_path, SETTINGS, ids=["zzz"])
    generative = tmp_path / "gen.yaml"
    generative.write_text(open("configs/test/catalog_train.yaml").read() + "\ntype: lample_charton\n", encoding="utf-8")
    with pytest.raises(ValueError):
        render_catalog(str(generative), tmp_path, SETTINGS)


def test_settings_are_checked():
    with pytest.raises(ValueError):
        DrawsSettings(repeats=(1, 1))
    with pytest.raises(ValueError):
        DrawsSettings(repeats=(0,))
    with pytest.raises(ValueError):
        DrawsSettings(n_support=0)


def test_source_draw_needs_one_problem_per_expression_and_a_finite_catalog(formula_catalog, engine):
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="problems_per_expression"):
        ProblemSource({"catalog": formula_catalog, "sampling": {"problems_per_expression": 2}}, simplipy_engine=engine).draw(0, rng)
    same = [ProblemSource({"catalog": formula_catalog, "sampling": {"n_support": 8}}, simplipy_engine=engine)
            .draw(1, np.random.default_rng(11)) for _ in range(2)]
    np.testing.assert_array_equal(same[0].x_support, same[1].x_support)
    assert same[0].eq_id == "b"


def test_the_cli_renders_and_verifies(formula_catalog, tmp_path, capsys):
    ids = tmp_path / "ids.json"
    ids.write_text(json.dumps({"ids": ["a", "c"]}), encoding="utf-8")
    root = tmp_path / "draws"
    main(["render-draws", formula_catalog, "-o", str(root), "--n-support", "16", "--n-validation", "8",
          "--simplipy-engine", "acj-4", "--ids", f"toy={ids}"])
    manifest = json.loads((root / MANIFEST).read_text())
    assert manifest["repeats"] == [1] and manifest["catalogs"]["toy"]["problems"] == 2
    main(["verify-draws", str(root), formula_catalog, "--ids", f"toy={ids}"])
    assert "sound and reproduced" in capsys.readouterr().out
    (root / "toy/repeat-1.npz").unlink()
    with pytest.raises(SystemExit):
        main(["verify-draws", str(root)])


# ------------------------------------------------------------------------------------------------ attempts
def _attempts(catalog_path, choices, *, name="toy", settings=SETTINGS):
    return {"format": ATTEMPTS_FORMAT, "format_version": 1, "settings": settings.record(), "repeats": list(settings.repeats),
            "rule": "the test's rule", "catalogs": {name: {"source_sha256": sha256_file(catalog_path), "attempts": {"1": choices}}}}


def test_an_attempt_changes_what_is_left_to_chance_and_nothing_else(formula_catalog, engine):
    draws = CatalogDraws(formula_catalog, SETTINGS, simplipy_engine=engine)
    assert draws.name == "toy" and draws.ids == ["a", "b", "c"] and draws.kinds == ["formula"] * 3
    first, second = draws.draw(0, 1, 1), draws.draw(0, 1, 2)
    assert first.problem.expression == second.problem.expression and first.problem.constants == second.problem.constants
    assert not np.array_equal(first.problem.x_support, second.problem.x_support)
    assert not np.array_equal(first.noise_support, second.noise_support)
    np.testing.assert_array_equal(draws.draw(0, 1, 2).problem.x_support, second.problem.x_support)   # seeded
    one, two = draws.draw(2, 1, 1).problem, draws.draw(2, 1, 2).problem                           # per-run values
    assert one.meta["per_run_values"] != two.meta["per_run_values"] and one.skeleton == two.skeleton
    with pytest.raises(ValueError):
        seed_sequence("s", "toy", "a", 1, 0)


def test_needs_check_skips_constants_by_construction_and_data(formula_catalog, data_catalog, tmp_path, engine):
    doc = {"metadata": {"name": "k"}, "expressions": {"k": {"prepared": "2.5", "vars": {"v1": dict(UNIT)}}}}
    path = tmp_path / "k.yaml"
    path.write_text(yaml.safe_dump(doc), encoding="utf-8")
    constant = CatalogDraws(str(path), SETTINGS, simplipy_engine=engine).draw(0)
    assert tuple(constant.problem.skeleton) == ("<constant>",) and not needs_check(constant.kind, constant.problem)
    formula = CatalogDraws(formula_catalog, SETTINGS, simplipy_engine=engine).draw(1)
    assert needs_check(formula.kind, formula.problem)
    data = CatalogDraws(data_catalog, SETTINGS, simplipy_engine=engine).draw(0)
    assert data.kind == "data" and not needs_check(data.kind, data.problem)


def test_each_problem_keeps_the_attempt_the_document_names(formula_catalog, tmp_path, engine):
    doc = _attempts(formula_catalog, {"a": {"attempt": 3, "formula_inherent": False, "failures": ["numeric", "symbolic"]},
                                      "b": {"attempt": 1, "formula_inherent": True, "failures": ["symbolic"] * 4}})
    path = tmp_path / "attempts.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    manifest = render([formula_catalog], tmp_path / "draws", SETTINGS, attempts=str(path), log=lambda _: None)
    records = read_meta(tmp_path / "draws/toy/repeat-1.npz")["problems"]
    assert [(r["attempt"], r["checked"], r["formula_inherent"]) for r in records] == [(3, True, False), (1, True, True), (1, False, False)]
    assert [r["failures"] for r in records] == [["numeric", "symbolic"], ["symbolic"] * 4, []]
    third = CatalogDraws(formula_catalog, SETTINGS, simplipy_engine=engine).draw(0, 1, 3)
    arrays = _arrays(tmp_path / "draws/toy/repeat-1.npz", 0)
    np.testing.assert_array_equal(arrays["x_support"], third.problem.x_support)
    np.testing.assert_array_equal(arrays["noise_validation"], third.noise_validation)
    assert manifest["acceptance"] == {"attempts_sha256": sha256_file(path), "rule": "the test's rule"}
    assert manifest["catalogs"]["toy"]["acceptance"] == {"1": {"checked": 2, "later_attempt": 1, "formula_inherent": ["b"],
                                                               "failures": {"numeric": 1, "symbolic": 5}}}
    plain = render([formula_catalog], tmp_path / "plain", SETTINGS, log=lambda _: None)
    assert plain["acceptance"] is None and plain["catalogs"]["toy"]["acceptance"] is None
    assert [r["attempt"] for r in read_meta(tmp_path / "plain/toy/repeat-1.npz")["problems"]] == [1, 1, 1]


def test_a_render_with_attempts_reproduces(formula_catalog, data_catalog, tmp_path):
    doc = _attempts(formula_catalog, {"c": {"attempt": 2, "formula_inherent": False, "failures": ["numeric"]}})
    doc["catalogs"]["toy-data"] = {"source_sha256": sha256_file(data_catalog), "attempts": {"1": {}}}
    render([formula_catalog, data_catalog], tmp_path, SETTINGS, attempts=doc, log=lambda _: None)
    assert verify(tmp_path, [formula_catalog, data_catalog]) == []


def test_an_attempts_document_that_does_not_fit_is_refused(formula_catalog, data_catalog, tmp_path):
    good = _attempts(formula_catalog, {})
    with pytest.raises(ValueError, match="not in the attempts document"):
        render([data_catalog], tmp_path / "a", SETTINGS, attempts=good, log=lambda _: None)
    with pytest.raises(ValueError, match="other settings"):
        load_attempts(good, DrawsSettings(n_support=32, n_validation=8, simplipy_engine="acj-4"))
    with pytest.raises(ValueError, match="no repeat"):
        load_attempts(good, DrawsSettings(n_support=16, n_validation=8, repeats=(1, 2), simplipy_engine="acj-4"))
    with pytest.raises(ValueError, match="draw-attempts"):
        load_attempts({**good, "format": "other"}, SETTINGS)
    stale = {**good, "catalogs": {"toy": {"source_sha256": "0" * 64, "attempts": {"1": {}}}}}
    with pytest.raises(ValueError, match="another version of the catalog"):
        render([formula_catalog], tmp_path / "b", SETTINGS, attempts=stale, log=lambda _: None)
    render([formula_catalog], tmp_path / "c", SETTINGS, attempts=good, log=lambda _: None)
    with pytest.raises(ValueError, match="attempts it was rendered with"):
        render([formula_catalog], tmp_path / "c", SETTINGS, update=True, log=lambda _: None)


def test_the_cli_renders_with_an_attempts_document(formula_catalog, tmp_path):
    path = tmp_path / "attempts.json"
    path.write_text(json.dumps(_attempts(formula_catalog, {"b": {"attempt": 2, "formula_inherent": False}})), encoding="utf-8")
    main(["render-draws", formula_catalog, "-o", str(tmp_path / "draws"), "--n-support", "16", "--n-validation", "8",
          "--simplipy-engine", "acj-4", "--attempts", str(path)])
    assert [r["attempt"] for r in read_meta(tmp_path / "draws/toy/repeat-1.npz")["problems"]] == [1, 2, 1]
