"""Build srbf's data catalogs: one frozen pool catalog (symbolic_data.pools) per survey set with data problems.

srbf's data-catalog rules (srbf ruling 2026-10-17): a data problem is the benchmark's rows as published. Each run
draws at most 512 fitting and 512 test rows afresh (the ProblemSource's job, not the builder's: pools hold every
row). There is a test split ONLY where the benchmark's authors give one:

* ``"given"``: the authors pin the test rows: published files, row indices, a deterministic rule (by value, time,
  group), or one seeded split whose rows are known;
* ``{"random": f}``: the authors' procedure redraws a random split per run (holdout fraction ``f``; k-fold CV is
  ``1/k`` where its folds are not known; repeated random splits);
* ``{"folds": [[rows], ...]}``: the authors test on fixed folds whose rows are known (published, or drawn again with
  the authors' code and seeds): every run tests on one fold and fits on the others (srbf ruling 2026-11-05);
* ``"none"``: no split and no procedure ("if people fail to tell us what to test on, we don't"): the test metrics
  stay empty and the catalog is marked.

Where the authors give several test sets (in- and out-of-domain), each is a problem of its own in its own subset
(srbf ruling 9, 2026-10-05). The SRBench tracks are built by build_srbench_catalogs.py.

The license pass (tags/licenses.json, paths.licenses) routes every set and member (owner rulings 2026-11-05 and
2026-11-09): licensed data go to the main assets repo; share-alike and copyleft data to the share-alike repo; data
under non-commercial terms or with no license are republished, with a statement of their source, their terms and why
srbf republishes them (paths.statement), in the restricted repo; only data whose terms restrict redistribution are
left out (listed on the benchmarks page), and a stale catalog of such a set is removed. A set whose datasets differ
keeps each member in its metadata with its license, and its share-alike and restricted members go to catalogs of
their own, ``<slug>-share-alike`` and ``<slug>-restricted``.

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
# Transcription fixes to the survey's records (the 2026-11-28 audit: three cp3-bench reference formulas did not fit their
# own data, FVU 68 / 0.09 / 1.4; each corrected form reproduces every row of its file to float precision). The reason
# goes into the problem's metadata.
FORMULA_FIXES = {
    ("fresh-54", "C3h"): ("300*(a_av_tt/a_av + 0.07158985**2*Omega_m0/(2*a_av**3))",
                          "the record's prefactor 10^2*(-6) is the authors' -6 Q_D convention applied twice; the published "
                          "target 10^2 Q_D is 300 (a_D_ddot/a_D + H_D0^2 Omega_m0/(2 a_D^3)), exact on every row"),
    ("fresh-54", "C6a"): ("-(2*1*0.25/5)*((-4*1**2*(-3*t + 10)**(-10/3) + 2*(-3*t + 10)**(2/3))*cos(-3*t**2/1 + 2*10*t/1) "
                          "+ 4*1*(-3*t + 10)**(-4/3)*sin(-3*t**2/1 + 2*10*t/1))",
                          "the sine's phase is -3 t^2/M + 2 v0 t/M, as in the cosine (the record had v0 t/M); exact on every "
                          "row of the authors' file"),
    ("fresh-54", "C6c"): ("-(2*(m1 + m1)*(m1*m1/(m1 + m1)**2)/5)*((-4*(m1 + m1)**2*(-3*t + 10)**(-10/3) + 2*(-3*t + 10)**(2/3))"
                          "*cos(-3*t**2/(m1 + m1) + 2*10*t/(m1 + m1)) + 4*(m1 + m1)*(-3*t + 10)**(-4/3)*sin(-3*t**2/(m1 + m1) "
                          "+ 2*10*t/(m1 + m1)))",
                          "the authors' data set the second mass equal to the first (m2 = m1, not 0.5) and the sine's phase "
                          "is -3 t^2/M + 2 v0 t/M; exact on every row of the authors' file"),
}
RULES = ("srbf data catalogs (2026-10-17): pools; <= 512 fitting + 512 test rows drawn per run; a test split only "
         "where the authors give one")


def _split_key(p) -> str:
    """A problem's split rule for the reports; a folds rule is named by its number of folds, not its rows."""
    rule = p.meta["test_split"]
    return json.dumps({"folds": len(rule["folds"])} if isinstance(rule, dict) and "folds" in rule else rule)


def wanted() -> dict[str, list[dict]]:
    """The data problems srbf keeps, by survey set, in the final list's order."""
    by_set: dict[str, list[dict]] = defaultdict(list)
    for p in paths.final_problems():
        if p["kind"] == "D" and p["set"] not in BUILT_ELSEWHERE:
            fix = FORMULA_FIXES.get((p["set"], p["id"]))
            by_set[p["set"]].append({**p, "formula": fix[0], "formula_fix": fix[1]} if fix else p)
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


def stated_kind(rows: R.Rows, prob: dict) -> str | None:
    """The ground-truth kind a record states, honoured only with the record's own check over the fitting rows
    (``gt_check``: ``rows``, ``rows_within_input_rounding``, ``fit_fvu``). A stated ``exact`` needs every fitting row
    within float32 input rounding: the authors computed each target before storing its row as float32, so where a
    formula is steep a row misses by more than eps while agreeing within that rounding (LLM-SRBench;
    llm_srbench_records.py). Anything else a record states is refused, not trusted."""
    kind = prob.get("gt_kind")
    if kind is None:
        return None
    if kind not in ("exact", "reference"):
        raise ValueError(f"{prob['id']}: stated gt_kind {kind!r} is neither 'exact' nor 'reference'")
    check, n = prob.get("gt_check") or {}, len(rows.y)
    if check.get("rows") != n:
        raise ValueError(f"{prob['id']}: states gt_kind {kind!r} without a check over its {n} fitting rows")
    if kind == "exact" and check.get("rows_within_input_rounding") != n:
        raise ValueError(f"{prob['id']}: states gt_kind 'exact', but its check puts "
                         f"{check.get('rows_within_input_rounding')} of {n} fitting rows within float32 input rounding")
    return kind


def reference_of(prob: dict, catalog: str, eq_id: str | None) -> dict | None:
    """Why the problem's formula may miss its rows and by how much (FVU): the record's own ``gt_reason`` and
    ``gt_fvu_bound``, else its row in tags/reference_laws.json (paths.reference_laws), else None."""
    if prob.get("gt_reason") and prob.get("gt_fvu_bound") is not None:
        return {"reason": prob["gt_reason"], "fvu_bound": float(prob["gt_fvu_bound"])}
    laws = paths.reference_laws()
    return laws.get((catalog, str(eq_id))) or laws.get((catalog, str(prob["id"])))


def with_ground_truth(rows: R.Rows, prob: dict, formulas: Formulas, catalog: str = "") -> dict:
    """data_problem keyword arguments for the problem's formula: exact when it reproduces the fitting rows to float32
    eps; a reference law only with a stated reason and an FVU bound it meets (reference_of), and refused otherwise
    (the 2026-11-28 audit: three transcription errors stood as 'reference' laws); a formula SimpliPy cannot read yet is
    kept as text and the problem is black-box. A record may state the kind with its own check (stated_kind); the check
    goes into the problem's meta."""
    f = formula_of(prob)
    if f is None:
        return {"gt_kind": "none"}
    stated = stated_kind(rows, prob)
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
    if stated:
        rows.meta["gt_kind_stated"] = True
        rows.meta["gt_check"] = dict(prob["gt_check"])
    if prob.get("formula_fix"):
        rows.meta["formula_corrected"] = prob["formula_fix"]
    kind = stated or ("exact" if fvu <= float(np.finfo(np.float32).eps) else "reference")
    if kind == "reference":
        ref = reference_of(prob, catalog, rows.eq_id)
        if ref is None:
            raise ValueError(f"{catalog}:{rows.eq_id or prob['id']}: the formula misses its rows (FVU {fvu:.3g}) and "
                             "no reason states it as a reference law (tags/reference_laws.json)")
        if not fvu <= ref["fvu_bound"]:
            raise ValueError(f"{catalog}:{rows.eq_id or prob['id']}: the reference law misses its rows by FVU {fvu:.3g}, "
                             f"above its bound {ref['fvu_bound']:g}")
        rows.meta["reference_reason"], rows.meta["reference_fvu_bound"] = ref["reason"], ref["fvu_bound"]
    return {"gt_kind": kind, "expression": compiled["expression"], "skeleton": compiled["prefix"],
            "constants": compiled["constants"]}


def problem(rows: R.Rows, prob: dict, catalog: str, display: str, record: dict, formulas: Formulas):
    gt = with_ground_truth(rows, prob, formulas, catalog)
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


PARTS = {"yes": "", "share-alike": "-share-alike", "restricted": "-restricted"}


def split_members(problems: list, lic: dict) -> tuple[dict[str, list], list[str]]:
    """The problems by the catalog they go to (``""`` the set's own, ``-share-alike``, ``-restricted``) and those left
    out, as the license pass routes them. Each kept member carries its license; restricted data (a member marked
    ``restricted``, or every problem of a set routed whole to the restricted repo) also carry their terms and the
    statement of why srbf republishes them."""
    parts: dict[str, list] = {suffix: [] for suffix in PARTS.values()}
    left_out = []
    whole_restricted = lic.get("redistribute") == "yes, restricted repo"
    whole_elsewhere = whole_restricted or lic.get("redistribute") == "yes, share-alike repo"
    for p in problems:
        member = (lic.get("members") or {}).get(p.eq_id)
        if member is None and lic.get("members"):
            raise ValueError(f"{p.eq_id}: not in the license pass's members of {lic['catalog']}")
        if member is not None and member["share"] == "no":
            left_out.append(f"{p.eq_id}: {member['reason']}")
            continue
        if member is not None:
            p.meta["license"] = member["license"]
        elif whole_elsewhere:                                          # a set routed whole to another repo
            p.meta["license"] = lic["license"]
        if whole_restricted or (member is not None and member["share"] == "restricted"):
            terms = member["terms"] if member is not None else lic["terms"]
            p.meta["terms"] = terms
            p.meta["statement"] = paths.statement(p.meta["license"] if member is not None else lic["upstream"], terms)
        parts[PARTS[member["share"]] if member is not None else ""].append(p)
    return parts, left_out


def build(tag: str, probs: list[dict], out: Path, formulas: Formulas) -> dict:
    spec = R.SETS[tag]
    if paths.dropped_whole(spec.slug):          # left out by rule (tags/dropped.json): no catalog, no stale part
        for name in (spec.slug, f"{spec.slug}-share-alike", f"{spec.slug}-restricted"):
            (out / f"{name}.npz").unlink(missing_ok=True)
        return {"tag": tag, "catalog": spec.slug, "problems": 0, "dropped": paths.dropped()[spec.slug]["reason"]}
    gone = paths.dropped_members(spec.slug)
    probs = [p for p in probs if p["id"] not in gone]
    lic = paths.licenses().get(spec.slug) or {}
    if lic and not lic["redistribute"].startswith("yes"):
        stale = out / f"{spec.slug}.npz"
        if stale.exists():
            stale.unlink()
        return {"tag": tag, "catalog": spec.slug, "problems": 0, "left_out": f"license: {lic['verdict']}"}
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
    parts, left_out = split_members(problems, lic)
    problems, share_alike, restricted = parts[""], parts["-share-alike"], parts["-restricted"]
    for name in (spec.slug, f"{spec.slug}-share-alike", f"{spec.slug}-restricted"):
        (out / f"{name}.npz").unlink(missing_ok=True)                 # a part that lost its members leaves no stale file
    meta = {"display_name": spec.display, "kind": "data", "source_record": f"catalog-prep/sets/{tag}.json",
            "source_name": rec["name"], "built_from": "build_data_catalogs.py", "rules": RULES}
    path = None
    for suffix, part, label, flag in (("", problems, "", {}), ("-share-alike", share_alike, " (share-alike)", {"share_alike": True}),
                                      ("-restricted", restricted, " (restricted)", {"restricted": True})):
        if part:
            saved = ProblemCatalog.from_problems(part, name=spec.slug + suffix, version=1, meta={
                **meta, **flag, "display_name": spec.display + label,
                "test_splits": sorted({_split_key(p) for p in part})}).save(out / f"{spec.slug}{suffix}.npz")
            path = path or saved
    if path is None:
        return {"tag": tag, "catalog": spec.slug, "problems": 0, "skipped": skipped, "left_out": left_out}
    problems = problems or share_alike or restricted                  # the set's own catalog, else its first part
    return {"tag": tag, "catalog": spec.slug, "display": spec.display, "problems": len(parts[""]),
            "share_alike": len(share_alike), "restricted": len(restricted), "left_out": left_out,
            "path": str(path), "bytes": os.path.getsize(path),
            "splits": dict(sorted(_count(_split_key(p) for p in problems).items())),
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
