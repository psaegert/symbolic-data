"""Tags for every catalog of the next srbf release: what kind of benchmark it is and what it tests.

srbf ruling 2026-11-03 ("a bunte Mischung: noise levels, outliers, extrapolation, ... small tags on the catalogs") and
2026-11-04 (the recommendations accepted, then "roll with it"): the tags describe the version srbf runs, not the purpose
of the authors' paper (srbf sets the noise of every formula catalog itself, ruling 4), and live in each catalog's
metadata.

Vocabulary (three kinds):
- origin, one per catalog: science formula, classic benchmark, generated formulas, measured data, synthetic data
  (simulator or DFT output, model output, data files the authors generated);
- what the test checks: extrapolation, several tests, no test split (no tag when the test is the training law);
- what the data holds: dummy inputs, integer inputs, fixed points, noisy data, outliers, srbf-modified.
A catalog carries a tag when it holds for at least half of its problems; the counts are kept with the tags.

``facts`` reads what the catalog files state (test blocks, inputs absent from the formula, integer laws, pinned points,
splits, the '*'); the judgement tags (origin; extrapolation, noise and outliers of data catalogs) were read from the
catalog-prep set records once, each with its source sentence (judged.json); ``merge`` applies the rules above and
``apply`` writes ``tags: {origin, tags}`` into each catalog's metadata, and its ``license`` from the license pass
(tags/licenses.json: the license of what each catalog took, with its evidence; a catalog that may not be shared gets
none), its ``citation`` (tags/citations.json: its paper and its dataset, where they exist), and ``srbf_changes``
(tags/deviations.json: each way srbf's version differs from the authors' benchmark, empty where it does not; srbf
ruling 2026-11-05) (a published catalog whose metadata changes gets its next version). Run ``apply`` after every
build or patch of the catalogs, then count_release.py.

    catalog_tags.py facts --counts <release_counts.json> --out <facts.json>
    catalog_tags.py merge --facts <facts.json> --judged <judged.json> --out <tags.json>
    catalog_tags.py apply --tags <tags.json> [--licenses <licenses.json>] [--citations <citations.json>]
                          [--published <manifest.json> ...]
"""
from __future__ import annotations

import argparse
import sys
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths  # noqa: E402  the curated inputs
ORIGINS = ("science formula", "classic benchmark", "generated formulas", "measured data", "synthetic data")
TAGS = ("extrapolation", "several tests", "no test split", "fit only", "generator in-distribution", "dummy inputs",
        "integer inputs", "fixed points", "noisy data", "outliers", "srbf-modified")
# per-problem flags for filtering (owner 2026-11-27), written as ``problem_flags`` {flag: [ids]} into the catalog's
# metadata from tags/problem_flags.json (``flags`` computes them; FLAGS defines each)
FLAGS = {
    "trivial": "the target is linear in one input: a straight-line fit to one input explains it (R^2 >= 0.999 on the "
               "fitting rows; formula problems on 256 points of the declared ranges)",
    "low-signal": "the inputs barely predict the target: a random forest and a linear fit (shuffled 5-fold CV) both "
                  "reach R^2 < 0.05 on the fitting rows (data without an exact law)",
    "target-leak": "the target is an exact function of other inputs it is computed from (e.g. a total and its parts)",
    "tiny": "fewer than 20 fitting rows, or no more rows than inputs plus one (underdetermined even for a linear fit)",
    "ordered": "the rows are sorted by an input and the authors' split is random or by folds, so a test row usually sits "
               "between two fitting rows (interpolation)",
}
TARGET_LEAKS = {"evogp-uci": ["daily_demand_forecasting_orders"],
                "srbench-2021-blackbox-restricted": ["527_analcatdata_election2000"]}
# the reading pass used "physics law" for science formulas; data that is neither measured nor drawn by srbf from a
# formula became "synthetic data" (its origin proposals: simulator output, synthetic data, fitted model output,
# mixed real and synthetic data with most rows synthetic)
RENAMED = {"physics law": "science formula"}
SYNTHETIC = ("simulator output", "synthetic data", "fitted model output", "mixed real and synthetic data")
# shares the reading pass stated in other words than "N of M problems"; None: not shown to hold for half
SHARES = {("cp3-bench", "noisy data"): (3, 28), ("cp3-bench", "dummy inputs"): (8, 28),
          ("srbench-2.0-first-principles", "noisy data"): (5, 12), ("mcdermott-predictive-modelling", "several tests"): (2, 6),
          ("scilaws-real", "extrapolation"): None}
# a property the benchmark's generator gives every problem with some probability: the tag holds whatever share a draw
# lands on (ERBench adds one dummy column with probability 1/2, src/benchmark.py generate_problem)
BY_DESIGN = {"dummy inputs": ("erbench-",)}
_SHARE = re.compile(r"^(\d+) of (\d+) problems")
_INT = re.compile(r"int|binomial|poisson", re.I)


def _int_input(spec: dict[str, Any]) -> bool:
    if any(_INT.search(str(t)) for t in spec.get("sample_type") or []):
        return True
    dist = spec.get("dist") or {}
    if not isinstance(dist, dict):
        return False
    name, kw = str(dist.get("name") or ""), dist.get("kwargs") or {}
    if _INT.search(name):
        return True
    if name == "grid":                                      # a lattice of whole numbers
        return all(float(kw.get(k, 0.5)).is_integer() for k in ("low", "step"))
    if name == "choice":
        return all(float(v).is_integer() for v in kw.get("values") or [0.5])
    return False


def yaml_facts(path: Path) -> dict[str, Any]:
    doc = paths.load_yaml(path)
    md, entries = doc["metadata"], doc["expressions"]
    dummy = integer = tested = fresh = suffixed = 0
    bases = {str(k).split(".", 1)[0] for k, e in entries.items() if e.get("test_source") and "." in str(k)}
    for key, e in entries.items():
        keys = [k for k in (e.get("vars") or {}) if k != "v0"]
        prepared = str(e.get("prepared") or "")
        if any(not re.search(rf"\b{k}\b", prepared) for k in keys):
            dummy += 1
        joint = (e.get("inputs") or {}).get("kwargs") or {}
        if joint.get("dtype") == "int" or any(_int_input(v) for k, v in (e.get("vars") or {}).items() if k != "v0"):
            integer += 1
        if isinstance(e.get("test"), dict):
            if e["test"]:
                tested += 1
            else:
                fresh += 1
        if e.get("test_source") and "." in str(key):
            suffixed += 1
    several = sum(1 for k in entries if str(k).split(".", 1)[0] in bases)
    return {"entries": len(entries), "with_dummy_input": dummy, "with_integer_input": integer,
            "with_test_block": tested, "with_fresh_test_instance": fresh, "extra_test_entries": suffixed,
            "in_several_tests": several,
            "joint_inputs": sorted({(e.get("inputs") or {}).get("name") for e in entries.values()} - {None}),
            "modified_by_srbf": md.get("modified_by_srbf"), "display_name": md.get("display_name") or md["name"]}


def npz_facts(path: Path) -> dict[str, Any]:
    meta = json.loads(str(np.load(path)["_meta"]))
    md = meta["catalog"].get("meta") or {}
    rules = [(p.get("meta") or {}).get("test_split") or "none" for p in meta["problems"]]
    fit_only = sum(1 for p, r in zip(meta["problems"], rules) if r == "none" and p.get("gt_kind") in (None, "none"))
    split = Counter(str({"folds": len(r["folds"])} if isinstance(r, dict) and "folds" in r else r) for r in rules)
    readings = Counter(str((p.get("meta") or {}).get("split_reading") or "")[:120] for p in meta["problems"])
    return {"entries": len(meta["problems"]), "test_split": dict(split),
            "split_readings": dict(readings.most_common(4)),
            "ground_truth": dict(Counter(str(p.get("gt_kind")) for p in meta["problems"])),
            "pinned_points": path.stem.endswith("-points"), "modified_by_srbf": md.get("modified_by_srbf"),
            "display_name": md.get("display_name") or meta["catalog"]["name"], "fit_only": fit_only}


def facts(counts: Path) -> list[dict[str, Any]]:
    deviations = {d["catalog"]: d for d in json.loads((Path(__file__).resolve().parent / "tags" / "deviations.json").read_text())}
    inventory: dict[str, Counter[str]] = {}
    for p in paths.testspecs():
        inventory.setdefault(p["catalog"], Counter())[p["test_kind"]] += 1
    out = []
    for r in json.load(open(counts))["rows"]:
        path = Path(r["file"])
        f = yaml_facts(path) if path.suffix == ".yaml" else npz_facts(path)
        tag = r.get("set") or (re.findall(r"sets/([\w-]+)\.json", str(r.get("source_record") or "")) or [None])[0]
        sure = []                                           # tags the files settle
        if f.get("with_dummy_input"):
            sure.append("dummy inputs")
        if f.get("with_integer_input"):
            sure.append("integer inputs")
        if f.get("pinned_points"):
            sure.append("fixed points")
        if f.get("fit_only") and 2 * f["fit_only"] >= f["entries"]:    # neither a ground truth nor a test split
            sure.append("fit only")
        if starred_by(deviations.get(r["name"]) or {}):     # derived from the deviations (2026-11-27), as the '*' is
            sure.append("srbf-modified")
        if f.get("extra_test_entries") or (r["kind"] == "formula" and len(r["subsets"]) > 1 and f.get("with_test_block")
                                           and set(r["subsets"]) & {"iid", "ood", "in-range", "wide", "beyond", "id"}):
            sure.append("several tests")
        split = f.get("test_split") or {}
        if split and set(split) == {"none"}:
            sure.append("no test split")
        out.append({"catalog": r["name"], "display_name": f["display_name"], "kind": r["kind"],
                    "problems": r["problems"], "subsets": r["subsets"], "set": tag, "source": r["source"],
                    "file": _relative(path),
                    "test_inventory": dict(inventory.get(tag, {})) if tag else None, "facts": f, "tags_from_files": sure})
    return out


def _relative(path: Path) -> str:
    """A catalog file's path relative to the repository, so the tables carry no local paths (outside: as given)."""
    try:
        return path.resolve().relative_to(paths.REPO).as_posix()
    except ValueError:
        return str(path)


def _file_share(tag: str, f: dict[str, Any]) -> tuple[int, int]:
    x, n = f["facts"], f["facts"]["entries"]
    if tag == "dummy inputs":
        return x["with_dummy_input"], n
    if tag == "fit only":
        return x["fit_only"], n
    if tag == "integer inputs":
        return x["with_integer_input"], n
    if tag == "several tests" and x.get("extra_test_entries"):
        return x["in_several_tests"], n
    return n, n                                             # a catalog-wide property (pinned points, no split, '*', subsets)


def merge(facts_path: Path, judged_path: Path) -> list[dict[str, Any]]:
    judged = {j["catalog"]: j for j in json.load(open(judged_path))}
    rows = []
    for f in json.load(open(facts_path)):
        k, j = f["catalog"], judged.get(f["catalog"]) or {}
        proposal = (j.get("origin_proposal") or "").lower()
        origin = "synthetic data" if proposal.startswith(SYNTHETIC) else RENAMED.get(j.get("origin"), j.get("origin"))
        found: dict[str, dict[str, Any]] = {}
        for tag in f["tags_from_files"]:
            m, n = _file_share(tag, f)
            found[tag] = {"share": [m, n], "source": "the catalog files"}
        for t in j.get("tags") or []:
            if t["tag"] in found:
                continue
            m = _SHARE.match(t.get("source") or "")
            if (k, t["tag"]) in SHARES:
                share = SHARES[(k, t["tag"])]
            else:
                share = (int(m.group(1)), int(m.group(2))) if m else (f["problems"], f["problems"])
            found[t["tag"]] = {"share": list(share) if share else None, "source": t.get("source")}
        kept = {t: v for t, v in found.items() if v["share"] and (2 * v["share"][0] >= v["share"][1]
                                                                  or k.startswith(BY_DESIGN.get(t, ("\0",))))}
        dropped = {t: v for t, v in found.items() if t not in kept}
        rows.append({"catalog": k, "display_name": f["display_name"], "kind": f["kind"], "problems": f["problems"],
                     "file": f["file"], "origin": origin, "origin_source": j.get("origin_source"),
                     "origin_proposal": j.get("origin_proposal"),
                     "tags": [t for t in TAGS if t in kept], "detail": kept, "below_half": dropped, "notes": j.get("notes")})
    return rows


# ---------------------------------------------------------------------------------------------------------- apply
# the metadata keys apply writes, in their order after ``version`` (each one flow line); the keys it removes from the
# public metadata (internal record ids; the 2026-11-27 audit)
KEYS = ("benchmark", "tags", "license", "citation", "srbf_changes", "srbf_notes", "problem_flags")
REMOVED = ("source_record",)
OPTIONAL = ("srbf_notes", "problem_flags")      # written only where they hold something (srbf_changes: [] says "none")


def _block(text: str, key: str) -> re.Match | None:
    """A top-level metadata key's line and its continuation lines (deeper-indented, as yaml wraps a long value)."""
    return re.search(rf"(?m)^  {key}:(?: .*)?\n(?:(?:    | {{5,}}).*\n)*", text)


def _set_line(text: str, key: str, line: str, after: str) -> str:
    """The metadata's ``key`` (with its continuation lines) replaced by ``line``, or inserted after its ``after``
    key; an empty ``line`` removes it."""
    old = _block(text, key)
    if old:
        return text[:old.start()] + line + text[old.end():]
    if not line:
        return text
    anchor = _block(text, after)
    return text[:anchor.end()] + line + text[anchor.end():] if anchor else text


def _flow(key: str, value: Any) -> str:
    return f"  {key}: " + json.dumps(value, ensure_ascii=False) + "\n"


def starred(display: str, star: bool | None) -> str:
    """The display name with srbf's '*' (it changed or filled something; deviations.json) or without it."""
    base = display.rstrip("*").rstrip()
    return base + "*" if star else base


def values(row: dict[str, Any]) -> dict[str, Any]:
    """The metadata values a row sets (None: leave the key as it is)."""
    return {"benchmark": row.get("benchmark"), "tags": {"origin": row["origin"], "tags": row["tags"]},
            "license": row.get("license"), "citation": row.get("citation"), "srbf_changes": row.get("srbf_changes"),
            "srbf_notes": row.get("srbf_notes"), "problem_flags": row.get("problem_flags")}


def apply_yaml(path: Path, row: dict[str, Any], published: dict[str, int]) -> str:
    """Write each key of KEYS as one flow line after the metadata's ``version``, remove REMOVED, and set the display
    name's star; bump a published version once."""
    before = path.read_text()
    text, after = before, "version"
    for key, value in values(row).items():
        if value is not None:
            text = _set_line(text, key, "" if value in ([], {}) and key in OPTIONAL else _flow(key, value), after)
        if _block(text, key):
            after = key
    for key in REMOVED:
        text = _set_line(text, key, "", "version")
    if row.get("star") is not None:
        head = yaml.safe_load(text.split("\nexpressions:", 1)[0])["metadata"]
        display = head.get("display_name") or head["name"]
        new = starred(display, row["star"])
        if new != head["name"] or head.get("display_name"):
            text = _set_line(text, "display_name", _flow("display_name", new) if new != head["name"] else "", "name")
    if text == before:
        return "unchanged"
    m = re.search(r"(?m)^  name: (\S+)$", text)
    v = re.search(r"(?m)^  version: (\d+)$", text)
    name, version = (m.group(1) if m else None), int(v.group(1))
    note = "written"
    if name in published and version <= published[name]:
        text = re.sub(r"(?m)^(  version: )(\d+)$", lambda mm: f"{mm.group(1)}{published[name] + 1}", text, count=1)
        note += f", version {version} -> {published[name] + 1}"
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)
    return note


def apply_npz(path: Path, row: dict[str, Any]) -> str:
    z = np.load(path)
    arrays = {k: z[k] for k in z.files}
    meta = json.loads(str(arrays["_meta"]))
    md = meta["catalog"].setdefault("meta", {})
    before = json.dumps(md, sort_keys=True)
    for key, value in values(row).items():
        if value in ([], {}) and key in OPTIONAL:
            md.pop(key, None)
        elif value is not None:
            md[key] = value
    for key in REMOVED:
        md.pop(key, None)
    if row.get("star") is not None:
        md["display_name"] = starred(md.get("display_name") or meta["catalog"]["name"], row["star"])
    if json.dumps(md, sort_keys=True) == before:
        return "unchanged"
    arrays["_meta"] = np.array(json.dumps(meta))
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez(tmp, **arrays)
    os.replace(tmp, path)
    return "written"


def _rows(path: Path | None) -> list[dict[str, Any]]:
    return json.load(open(path)) if path and path.exists() else []


def apply(tags_path: Path, published_paths: list[Path], licenses_path: Path | None = None,
          citations_path: Path | None = None, deviations_path: Path | None = None,
          benchmarks_path: Path | None = None) -> None:
    published: dict[str, int] = {}
    for p in published_paths:
        for name, v in json.load(open(p)).items():
            if isinstance(v, dict) and v.get("default_version") is not None:
                published[name] = int(max(int(x) for x in (v.get("versions") or {"1": None})))
    # the license pass (tags/licenses.json): the license line of every catalog that may be shared (not where unclear)
    licenses = {r["catalog"]: r["license"] for r in _rows(licenses_path) if r["redistribute"].startswith("yes")}
    # the citations (tags/citations.json): the paper and the dataset of each catalog, where they exist
    citations = {c["catalog"]: {k: c[k] for k in ("paper", "data") if c.get(k)} for c in _rows(citations_path)}
    # the deviation audit (tags/deviations.json): how srbf's version differs from the authors' benchmark; every
    # deviation, running part of it included, earns the '*'; a version that follows the defining publication, a
    # compilation that follows its own code or data, and the rule for paper-vs-code conflicts are visible notes without
    # a star (owner 2026-11-27)
    deviations = {c["catalog"]: c for c in _rows(deviations_path)}
    # the benchmark each catalog belongs to (tags/benchmarks.json): srbf averages by benchmark (owner 2026-11-27)
    benchmark = {c: b["benchmark"] for b in _rows(benchmarks_path) for c in b["catalogs"]}
    # the per-problem flags (tags/problem_flags.json, ``flags``)
    flags_path = Path(__file__).resolve().parent / "tags" / "problem_flags.json"
    flags = json.loads(flags_path.read_text())["flags"] if flags_path.exists() else {}
    done = Counter()
    for row in json.load(open(tags_path)):
        dev = deviations.get(row["catalog"])
        row = {**row, "license": licenses.get(row["catalog"]), "citation": citations.get(row["catalog"]) or None,
               "srbf_changes": dev["changes"] if dev else None, "srbf_notes": notes_of(dev) if dev else None,
               "star": starred_by(dev) if dev else None, "benchmark": benchmark.get(row["catalog"]),
               "problem_flags": flags.get(row["catalog"], {})}
        if benchmarks_path and row["benchmark"] is None:
            raise SystemExit(f"{row['catalog']}: no benchmark in {benchmarks_path}")
        path = Path(row["file"]) if Path(row["file"]).is_absolute() else paths.REPO / row["file"]
        note = apply_yaml(path, row, published) if path.suffix == ".yaml" else apply_npz(path, row)
        done[note.split(",")[0]] += 1
        if "version" in note:
            print(f"{row['catalog']}: {note}", flush=True)
    print(dict(done))


# every kind earns the '*': srbf changes, fills in, keeps-and-flags something, or runs part of the benchmark (owner
# 2026-11-27: "incomplete runs over benchmarks count as stars")
STAR_KINDS = ("part", "changed", "filled", "kept")


def starred_by(dev: dict[str, Any]) -> bool:
    """Whether srbf's version carries the '*': it runs part of the benchmark, or changes, fills in or keeps-as-published-
    but-flags something."""
    return any(c["kind"] in STAR_KINDS for c in dev.get("changes") or [])


def notes_of(dev: dict[str, Any]) -> list[str]:
    """The catalog's visible notes without a star: where srbf follows the defining publication, and where the rule
    for paper-vs-code conflicts applied (follow what produced the authors' published results; the paper's version
    noted)."""
    return [*(f"Follows the defining publication: {t}" for t in dev.get("follows_first_publication") or []),
            *(f"Follows the compilation's own code or data: {t}" for t in dev.get("compilation_code") or []),
            *(f"Paper and code differ: {t}" for t in dev.get("paper_vs_code") or [])]


# ---------------------------------------------------------------------------------------------------- problem flags
def _single_input_r2(x: np.ndarray, y: np.ndarray) -> float:
    """The best R^2 of a straight-line fit of ``y`` to one column of ``x``."""
    best, vy = 0.0, float(np.var(y))
    if vy <= 0 or len(y) < 3:
        return 0.0
    for j in range(x.shape[1]):
        a = np.column_stack([x[:, j], np.ones(len(y))])
        if np.var(x[:, j]) <= 0:
            continue
        coef, *_ = np.linalg.lstsq(a, y, rcond=None)
        best = max(best, 1.0 - float(np.mean((y - a @ coef) ** 2)) / vy)
    return best


def _low_signal(x: np.ndarray, y: np.ndarray) -> bool:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import KFold, cross_val_score
    if len(y) < 10 or np.var(y) <= 0:
        return False
    take = np.random.default_rng(0).permutation(len(y))[:1000]
    x, y = x[take], y[take]
    folds = KFold(min(5, len(y) // 2), shuffle=True, random_state=0)     # shuffled: rows sorted by an input
    ols = float(np.mean(cross_val_score(LinearRegression(), x, y, cv=folds, scoring="r2")))
    if ols >= 0.05:
        return False
    rf = float(np.mean(cross_val_score(RandomForestRegressor(n_estimators=50, random_state=0, n_jobs=4), x, y, cv=folds,
                                       scoring="r2")))
    return rf < 0.05


def _ordered(x: np.ndarray, rule: Any) -> bool:
    if not (isinstance(rule, dict) and ("random" in rule or "folds" in rule)) or len(x) < 10:
        return False
    d = np.diff(x, axis=0)
    return bool(any((np.all(d[:, j] >= 0) or np.all(d[:, j] <= 0)) and np.ptp(x[:, j]) > 0 for j in range(x.shape[1])))


def flags(counts: Path) -> dict[str, dict[str, list[str]]]:
    """Each release catalog's problems that carry a flag of FLAGS (data catalogs from their rows; formula catalogs'
    'trivial' from 256 points of each entry's declared ranges)."""
    import formula_gate
    out: dict[str, dict[str, list[str]]] = {}
    frozen = {n: set(json.loads(paths.frozen_path(f).read_text())["ids"]) for n, f in
              {"erbench-syneq": "release__erbench-syneq.json", "erbench-oeis": "erbench-oeis.json"}.items()}
    for r in json.load(open(counts))["rows"]:
        found: dict[str, list[str]] = {k: [] for k in FLAGS}
        found["target-leak"] = list(TARGET_LEAKS.get(r["name"], []))
        path = Path(r["file"])
        if path.suffix == ".npz":
            z = np.load(path)
            meta = json.loads(str(z["_meta"]))
            for i, p in enumerate(meta["problems"]):
                x = np.asarray(z[f"p{i}__x_support"], dtype=np.float64)
                y = np.asarray(z[f"p{i}__y_support"], dtype=np.float64).reshape(-1)
                eq = str(p.get("eq_id"))
                if _single_input_r2(x, y) >= 0.999:
                    found["trivial"].append(eq)
                if len(y) < 20 or len(y) <= x.shape[1] + 1:
                    found["tiny"].append(eq)
                if _ordered(x, (p.get("meta") or {}).get("test_split")):
                    found["ordered"].append(eq)
                if p.get("gt_kind") != "exact" and _low_signal(x, y):       # an exact law's rows carry the full signal
                    found["low-signal"].append(eq)
        else:
            doc = paths.load_yaml(path)
            rng = np.random.default_rng(0)
            for key, e in doc["expressions"].items():
                if r["name"] in frozen and str(key) not in frozen[r["name"]]:
                    continue
                keys = sorted((k for k in (e.get("vars") or {}) if k != "v0"), key=lambda k: int(k[1:]))
                try:
                    f = formula_gate._function(str(e["prepared"]), keys)
                    cols = [rng.uniform(*(formula_gate.box_of(e["vars"][k]) or (0.5, 2.0)), 256) for k in keys]
                    y = formula_gate._values(f, cols)
                except Exception:
                    continue
                ok = np.isfinite(y)
                if ok.sum() >= 16 and _single_input_r2(np.column_stack(cols)[ok], y[ok]) >= 0.999:
                    found["trivial"].append(str(key))
        found = {k: v for k, v in found.items() if v}
        if found:
            out[r["name"]] = found
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a0 = sub.add_parser("flags")
    a0.add_argument("--counts", required=True)
    a0.add_argument("--out", default=str(Path(__file__).resolve().parent / "tags" / "problem_flags.json"))
    a1 = sub.add_parser("facts")
    a1.add_argument("--counts", required=True)
    a1.add_argument("--out", required=True)
    a2 = sub.add_parser("merge")
    a2.add_argument("--facts", required=True)
    a2.add_argument("--judged", required=True)
    a2.add_argument("--out", required=True)
    a3 = sub.add_parser("apply")
    a3.add_argument("--tags", required=True)
    a3.add_argument("--published", nargs="*", default=[], help="hosted manifest.json files: the published versions")
    a3.add_argument("--licenses", default=str(Path(__file__).resolve().parent / "tags" / "licenses.json"))
    a3.add_argument("--citations", default=str(Path(__file__).resolve().parent / "tags" / "citations.json"))
    a3.add_argument("--deviations", default=str(Path(__file__).resolve().parent / "tags" / "deviations.json"))
    a3.add_argument("--benchmarks", default=str(Path(__file__).resolve().parent / "tags" / "benchmarks.json"))
    a = ap.parse_args()
    if a.cmd == "flags":
        found = flags(Path(a.counts))
        Path(a.out).write_text(json.dumps({"definitions": FLAGS, "flags": found}, indent=1) + "\n")
        print({k: sum(len(v.get(k, [])) for v in found.values()) for k in FLAGS}, "->", a.out)
    elif a.cmd == "facts":
        rows = facts(Path(a.counts))
        Path(a.out).write_text(json.dumps(rows, indent=1) + "\n")
        c = Counter(t for r in rows for t in r["tags_from_files"])
        print(f"{len(rows)} catalogs; tags from the files: {dict(c)} -> {a.out}")
    elif a.cmd == "merge":
        rows = merge(Path(a.facts), Path(a.judged))
        Path(a.out).write_text(json.dumps(rows, indent=1) + "\n")
        print(f"{len(rows)} catalogs; origins {dict(Counter(r['origin'] for r in rows))}; "
              f"tags {dict(Counter(t for r in rows for t in r['tags']))}; "
              f"below half {dict(Counter(t for r in rows for t in r['below_half']))} -> {a.out}")
    else:
        apply(Path(a.tags), [Path(p) for p in a.published], Path(a.licenses), Path(a.citations), Path(a.deviations),
              Path(a.benchmarks))


if __name__ == "__main__":
    main()
