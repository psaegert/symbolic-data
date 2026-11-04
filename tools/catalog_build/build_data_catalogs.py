"""Build srbf's data catalogs: one frozen pool catalog (symbolic_data.pools) per survey set with data problems.

srbf's data-catalog rules (srbf ruling 2026-10-17): a data problem is the benchmark's rows as published. Each run
draws at most 512 fitting and 512 test rows afresh (the ProblemSource's job, not the builder's: pools hold every
row). There is a test split ONLY where the benchmark's authors give one:

* ``"given"``: the authors pin the test rows: published files, row indices, a deterministic rule (by value, time,
  group), or one seeded split whose rows are known;
* ``{"random": f}``: the authors' procedure redraws a random split per run (holdout fraction ``f``; k-fold CV is
  ``1/k``; repeated random splits);
* ``"none"``: no split and no procedure ("if people fail to tell us what to test on, we don't"): the test metrics
  stay empty and the catalog is marked.

Where the authors give several test sets (in- and out-of-domain), each is a problem of its own in its own subset
(srbf ruling 9, 2026-10-05). The SRBench tracks are built by build_srbench_catalogs.py.

Inputs are the survey's records (paths.py): ``final_problems.json`` says which data problems srbf keeps after every
exclusion, ``data_sets.json`` holds each dataset's files, sha256, columns and split as published. The files themselves
are not in the repository: the readers take them from SYMBOLIC_DATA_CATALOG_RAW (the survey's raw/ folder) or download
them from their recorded URL, and a file whose sha256 differs from its record stops that set. How each set's files
become rows is in data_catalog_readers.py, one reader per set. The catalogs go to build/data-catalogs (git-ignored):
they are published to Hugging Face only.

    build_data_catalogs.py [--only fresh-60,fresh-94] [--out build/data-catalogs] [--list]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data_catalog_readers as R  # noqa: E402
import paths  # noqa: E402  the survey's records and where the raw files are
from symbolic_data import data_problem  # noqa: E402
from symbolic_data.catalog import ProblemCatalog  # noqa: E402

BUILT_ELSEWHERE = {"done-0", "done-7", "done-8"}  # the SRBench tracks: build_srbench_catalogs.py
RULES = ("srbf data catalogs (2026-10-17): pools; <= 512 fitting + 512 test rows drawn per run; a test split only "
         "where the authors give one")


def wanted() -> dict[str, list[dict]]:
    """The data problems srbf keeps, by survey set, in the final list's order."""
    by_set: dict[str, list[dict]] = defaultdict(list)
    for p in paths.final_problems():
        if p["kind"] == "D" and p["set"] not in BUILT_ELSEWHERE:
            by_set[p["set"]].append(p)
    return dict(by_set)


def formula_of(prob: dict) -> str | None:
    """The problem's formula in sympy spelling, or None for black-box data ('none (...)' texts included)."""
    f = prob.get("formula")
    if not isinstance(f, str) or not f.strip() or f.strip().lower().startswith("none"):
        return None
    return f


class Formulas:
    """Compiles a data problem's formula over its v1..vN inputs with the release SimpliPy engine (lazily)."""

    def __init__(self) -> None:
        self._engine = None

    def engine(self):
        if self._engine is None:
            from symbolic_data._evaluation import load_engine
            self._engine = load_engine(None)
        return self._engine

    def compile(self, eq_id: str, formula: str, names: list[str]) -> dict:
        from symbolic_data._evaluation import compile_expression
        rename = {n: f"v{i + 1}" for i, n in enumerate(names)}
        # sympy spellings SimpliPy reads differently: Abs -> abs, the constant pi -> its value
        formula = re.sub(r"\bpi\b", repr(math.pi), re.sub(r"\bAbs\(", "abs(", formula))
        prepared = re.sub(r"\b(" + "|".join(map(re.escape, sorted(names, key=len, reverse=True))) + r")\b",
                          lambda m: rename[m.group(1)], formula)
        return compile_expression(self.engine(), eq_id, prepared, {v: {} for v in rename.values()}, name="data")


def with_ground_truth(rows: R.Rows, prob: dict, formulas: Formulas) -> dict:
    """data_problem keyword arguments for the problem's formula: exact when it reproduces the fitting rows to float32
    eps, a reference law otherwise; a formula SimpliPy cannot read yet is kept as text and the problem is black-box."""
    f = formula_of(prob)
    if f is None:
        return {"gt_kind": "none"}
    from symbolic_data._evaluation import evaluate
    names = rows.formula_names or rows.features
    try:
        compiled = formulas.compile(prob["id"], f, names)
        pred = np.asarray(evaluate(compiled, {f"v{i + 1}": rows.x[:, i] for i in range(rows.x.shape[1])}),
                          dtype=np.float64)
        pred = np.broadcast_to(pred, rows.y.shape).astype(np.float64)
    except Exception as exc:  # an operator the release SimpliPy lacks (erf, Mod, besselj, ...): rebuild with 0.15
        rows.meta["formula_pending"] = {"formula": f, "error": f"{type(exc).__name__}: {exc}"[:300]}
        return {"gt_kind": "none"}
    fvu = float(np.mean((rows.y - pred) ** 2) / np.var(rows.y)) if np.var(rows.y) > 0 else float("nan")
    rows.meta["formula"] = f
    rows.meta["formula_fvu_on_fit_rows"] = fvu
    kind = "exact" if fvu <= float(np.finfo(np.float32).eps) else "reference"
    return {"gt_kind": kind, "expression": compiled["expression"], "skeleton": compiled["prefix"],
            "constants": compiled["constants"]}


def problem(rows: R.Rows, prob: dict, catalog: str, display: str, record: dict, formulas: Formulas):
    gt = with_ground_truth(rows, prob, formulas)
    meta = {
        "catalog": catalog, "catalog_display": display, "subset": rows.subset or prob["unit"].split(":", 1)[1],
        "source_id": prob["id"], "url": record.get("url"), "sha256": record.get("sha256"),
        "feature_names": rows.features, "split_as_published": record.get("split"),
        "split_reading": rows.split_note, "n_rows": int(len(rows.y) + (0 if rows.y_test is None else len(rows.y_test))),
        **({"transforms": rows.transforms} if rows.transforms else {}),
        **rows.meta,
    }
    test = {"x_test": rows.x_test, "y_test": rows.y_test} if R.rule_of(rows.split) == "given" else {}
    return data_problem(rows.x, rows.y, variables=[f"v{i + 1}" for i in range(rows.x.shape[1])],
                        test_split=rows.split, eq_id=rows.eq_id or prob["id"], meta=meta, **test, **gt)


def build(tag: str, probs: list[dict], out: Path, formulas: Formulas) -> dict:
    spec = R.SETS[tag]
    rec = paths.data_set_record(tag)
    records = {str(q.get("id")): q for q in rec["datasets"] if isinstance(q, dict)}
    problems, notes, skipped = [], [], []
    for prob in probs:
        record = records[prob["id"]]
        try:
            read = R.as_list(spec.read(record, prob, rec))
        except R.SkipDataset as why:
            skipped.append(str(why))
            continue
        for rows in read:
            rows.finish()
            problems.append(problem(rows, prob, spec.slug, spec.display, record, formulas))
            notes.append(rows.summary())
    if not problems:
        return {"tag": tag, "catalog": spec.slug, "problems": 0, "skipped": skipped}
    meta = {"display_name": spec.display, "kind": "data", "source_record": f"catalog-prep/sets/{tag}.json",
            "source_name": rec["name"], "built_from": "build_data_catalogs.py", "rules": RULES,
            "test_splits": sorted({json.dumps(p.meta["test_split"]) for p in problems})}
    path = ProblemCatalog.from_problems(problems, name=spec.slug, version=1, meta=meta).save(out / f"{spec.slug}.npz")
    return {"tag": tag, "catalog": spec.slug, "display": spec.display, "problems": len(problems),
            "path": str(path), "bytes": os.path.getsize(path),
            "splits": dict(sorted(_count(json.dumps(p.meta["test_split"]) for p in problems).items())),
            "gt_kinds": dict(_count(p.gt_kind for p in problems)),
            "formula_pending": sum(1 for p in problems if "formula_pending" in p.meta),
            "rows_fit": [int(min(p.x_support.shape[0] for p in problems)), int(max(p.x_support.shape[0] for p in problems))],
            "rows_test": [int(min(p.x_validation.shape[0] for p in problems)),
                          int(max(p.x_validation.shape[0] for p in problems))],
            "dropped_nonfinite_rows": int(sum(p.meta.get("dropped_nonfinite_rows", 0) for p in problems)),
            "skipped": skipped, "notes": notes[:3]}


def _count(items) -> dict:
    out: dict = defaultdict(int)
    for i in items:
        out[i] += 1
    return dict(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(paths.DATA_BUILD))
    ap.add_argument("--only", default=None, help="comma-separated survey set tags")
    ap.add_argument("--list", action="store_true", help="list the sets, their readers and problem counts")
    a = ap.parse_args()
    sets = wanted()
    if a.list:
        for tag, probs in sorted(sets.items(), key=lambda kv: -len(kv[1])):
            spec = R.SETS.get(tag)
            print(f"{tag:14s} {len(probs):4d}  {spec.slug if spec else '-- no reader --'}")
        return
    out = Path(os.path.expanduser(a.out))
    out.mkdir(parents=True, exist_ok=True)
    tags = a.only.split(",") if a.only else [t for t in sets if t in R.SETS]
    formulas = Formulas()
    report_path = out / "build_data_report.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else {}
    failed = []
    for tag in tags:
        try:
            report[tag] = build(tag, sets[tag], out, formulas)
            print(json.dumps(report[tag]), flush=True)
        except Exception as exc:  # one set's failure must not stop the others; it is reported
            failed.append(tag)
            report[tag] = {"tag": tag, "error": f"{type(exc).__name__}: {exc}"[:600]}
            print(f"FAILED {tag}: {type(exc).__name__}: {exc}", flush=True)
            traceback.print_exc()
        report_path.write_text(json.dumps(report, indent=1))
    if failed:
        raise SystemExit(f"failed: {','.join(failed)}")


if __name__ == "__main__":
    main()
