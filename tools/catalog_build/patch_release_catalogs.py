"""Apply the source audit's fixes to srbf's release formula catalogs (symbolic-data assets/catalogs/*.yaml).

The release catalogs are DSO transcriptions (benchmarks.csv); the 2026-10-05 source audit
(workflow_archive/source-audit/CONSOLIDATED.md, section 1) checked them against each benchmark's own authors, and
catalog-prep recorded the authors' definitions of the members srbf gets wrong in its family sets. This script writes
those definitions into the release catalogs under the same keys and adds the members the release lacks. It edits
only the affected entries' text (each entry is re-dumped where it stands; new entries are appended), so a catalog's
other entries keep their bytes, and bumps the catalog's version.

Pinned-grid members (Keijzer-6, Neat-6, the authors' Sine, Vladislavleva-2/3's mesh designs) are data problems
(ruling 9(2)). They leave their release catalog for a companion '<catalog>-points' data catalog (srbf ruling 2026-11-03,
call (d), then the source audit's other exact-training-point members; build_formula_catalogs.py --family-points). A
catalog left without entries (sine, nonic, r-rationals) is removed. Const-Test-1/2 are dropped.

    patch_release_catalogs.py --catalogs <symbolic-data>/assets/catalogs [--dry-run]
"""
from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_formula_catalogs as B  # noqa: E402

# family set -> release catalog; ids that REPLACE a release entry or are ADDED; ids left for later
FAMILIES = {
    "fresh-10": ("keijzer", {"Keijzer-5"}),
    "fresh-11": ("korns", {"Korns-5", "Korns-6", "Korns-7", "Korns-8", "Korns-9", "Korns-13", "Korns-14", "Korns-15"}),
    "fresh-13": ("vladislavleva", {"Vladislavleva-7"}),
    "fresh-18": ("poly", {"Poly-10"}),
}
# members the DSO transcription lacks (added, not replacing; stable across re-runs of this script)
ADDED = {"Korns-13", "Korns-14", "Korns-15"}
# Uy et al. 2011 Table 3: Nguyen-9 and -10 on [-1, 1]^2 (DSO: [0, 1]^2); fresh-5's Uy2011-F9 is the fixed Nguyen-9.
# The nguyen catalog stays Uy et al.'s 12: fresh-5's NGGP and SPL variants (Nguyen-12*, Nguyen-2c/9c) are other
# authors' problems (Nguyen-2c/9c live in fresh-7's catalog).
RANGE_FIXES = {("nguyen", "Nguyen-9"): [-1.0, 1.0], ("nguyen", "Nguyen-10"): [-1.0, 1.0]}
# the pinned-grid members that move to '<catalog>-points'
MOVED = {"keijzer": ["Keijzer-1", "Keijzer-2", "Keijzer-3", "Keijzer-4", "Keijzer-6", "Keijzer-7", "Keijzer-8", "Keijzer-9"],
         "neat": ["Neat-6"], "vladislavleva": ["Vladislavleva-2", "Vladislavleva-3"], "sine": ["Sine"], "nonic": ["Nonic"],
         "r-rationals": ["R1", "R2", "R3", "R1a", "R2a", "R3a"]}
# dropped outright (srbf ruling 2026-11-03): DSO's repository sanity tests, not a published benchmark
DROPPED = {"constant": ["Const-Test-1", "Const-Test-2"]}
# the versions published on the Hub; a patched catalog becomes the next one (once per release, however often this runs)
RELEASED = {"keijzer": 1, "korns": 1, "nguyen": 1, "poly": 1, "vladislavleva": 1, "fastsrb.v2": 2, "neat": 1, "sine": 1,
            "jin": 1, "soose-nc": 1, "soose-wc": 1, "soose-fc": 1, "nonic": 1, "r-rationals": 1, "constant": 1}
# the release catalogs with authors' test sets (rulings 9(1), 9(7); authors_tests.py); pinned-training members wait
TEST_CATALOGS = ["vladislavleva", "keijzer", "nguyen", "neat", "jin", "soose-nc", "soose-wc", "soose-fc"]


def _fastsrb_fixes() -> dict[str, Any]:
    """fastsrb.v2 entry -> (edit, note): the audit's FastSRB/SRSD fixes, following SRSD's code and published data where
    its own artifacts disagree (source-audit C2)."""
    def sign_plus(e):
        e["raw"] = e["raw"].replace("- 2 * x1 * x2 * cos", "+ 2 * x1 * x2 * cos")
        e["prepared"] = e["prepared"].replace("- 2 * v1 * v2 * cos", "+ 2 * v1 * v2 * cos")

    def eps_linear(e):
        e["raw"] = e["raw"].replace("2 * epsilon ^ 2 * E_n", "2 * epsilon * E_n")
        e["prepared"] = e["prepared"].replace("2 * v1 ^ 2 * v2", "2 * v1 * v2")

    def negative(*keys):
        def edit(e):
            for k in keys:
                e["vars"][k]["sample_type"] = ["log", "neg"]
        return edit

    def choice(key, values):
        def edit(e):
            spec = {k: v for k, v in e["vars"][key].items() if k not in ("sample_range", "sample_type")}
            e["vars"][key] = {**spec, "dist": {"name": "choice", "kwargs": {"values": [float(v) for v in values]}}}
        return edit

    def box(key, low, high):
        def edit(e):
            e["vars"][key]["sample_range"] = [low, high]
        return edit

    def integers(*keys):
        """SRSD's IntegerSampling(lo, hi) is numpy's randint: uniform on {lo, ..., hi - 1}; the fastsrb `int` base
        rounds a uniform draw instead (both ends at half weight, hi included)."""
        def edit(e):
            for k in keys:
                if "dist" in e["vars"][k]:
                    continue
                lo, hi = (int(v) for v in e["vars"][k]["sample_range"])
                spec = {kk: v for kk, v in e["vars"][k].items() if kk not in ("sample_range", "sample_type")}
                e["vars"][k] = {**spec, "dist": {"name": "grid", "kwargs": {"low": float(lo), "high": float(hi - 1), "step": 1.0}}}
        return edit

    def accept(*subs):
        """The entry's `accept` forms re-spelled for its corrected law by (old, new) text substitutions, applied to the
        forms that are not yet the formula (so a second run changes nothing)."""
        def edit(e):
            forms = e.get("accept") or []
            bad = set(accept_mismatches(e))
            out = []
            for form in forms:
                if form in bad:
                    for old, new in subs:
                        form = form.replace(old, new)
                out.append(form)
            e["accept"] = out
        return edit

    def all_of(*edits):
        def edit(e):
            for ed in edits:
                ed(e)
        return edit
    ints = "integer inputs on {lo..hi-1}, as SRSD's IntegerSampling (numpy randint) and published data"
    return {
        "I.29.16": (all_of(sign_plus, accept(("cos(", "(-1.0)*cos("))),
                    "the cosine term's sign as SRSD's code, true_eq and published data (+)"),
        "B6": (all_of(eps_linear, accept(("(v1^2.0)", "v1")), integers("v5", "v6")),
               "epsilon not squared, as SRSD's code, true_eq and published data; " + ints),
        "III.21.20": (negative("v1", "v2"), "rho_c_0 and q negative only (SRSD code uses_positive=False; FastSRB bcd094b)"),
        "B7": (choice("v2", [-1, 0, 1]), "the curvature in {-1, 0, 1}, each 1/3 (SRSD code, table and published data)"),
        "III.15.27": (all_of(choice("v1", [*range(-99, 0), *range(1, 100)]), integers("v2")),
                      "alpha in +-{1..99} (SRSD IntegerSampling(1, 100) with a random sign; no 0); " + ints),
        "I.44.4": (box("v1", 1e24, 1e26), "n on [1e24, 1e26], as SRSD's problem table and published data (code: "
                                          "[1e25, 1e27]; kept, 2026-11-03)"),
        "III.12.43": (integers("v1"), ints), "I.30.5": (integers("v2"), ints), "III.19.51": (integers("v3"), ints),
        "I.30.3": (integers("v2"), ints), "B1": (integers("v1", "v2"), ints), "B11": (integers("v3"), ints),
    }


def _audit_note(note: str | None, entry: dict[str, Any]) -> str:
    """The entry's source-audit note: its fix, and the accept forms dropped (kept in ``accept_dropped``)."""
    dropped = entry.get("accept_dropped") or []
    parts = ([note] if note else []) + ([f"{len(dropped)} accept form(s) dropped: not equal to the formula"] if dropped else [])
    return "; ".join(parts)


def accept_mismatches(entry: dict[str, Any], n: int = 64) -> list[str]:
    """The entry's `accept` forms that are not its `prepared`: relative difference above 1e-6 on n points drawn from its
    declared inputs, where the formula's value is finite and above 1e-100 (below, equal spellings underflow apart, e.g.
    sqrt(exp(-x)) and exp(-x/2)). Both are evaluated by symbolic_data's own engine."""
    from symbolic_data._evaluation import compile_expression, evaluate, load_engine
    from symbolic_data.catalog import ProblemCatalog
    from symbolic_data.inputs import draw_rows
    if not entry.get("accept"):
        return []
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = load_engine(None)
    cat = ProblemCatalog.from_yaml({"metadata": {"name": "accept", "version": 1}, "expressions": {"e": entry}})
    e = cat.entries["e"]
    keys = sorted((k for k in e.variables if k != "v0"), key=lambda k: int(k[1:]))
    x = draw_rows(e.variables, e.meta, keys, n, np.random.default_rng(0))

    def f(text):
        c = compile_expression(_ENGINE, "accept", text, {k: {} for k in keys}, name="accept")
        y = np.asarray(evaluate(c, {k: x[:, i] for i, k in enumerate(keys)}), dtype=float)
        return np.broadcast_to(y.reshape(-1) if y.size == n else y, (n,))
    with np.errstate(all="ignore"):
        y = f(entry["prepared"])
        bad = []
        for form in entry["accept"]:
            z = f(form)
            ok = np.isfinite(y) & (np.abs(y) > 1e-100)
            if not ok.any() or not np.all(np.isfinite(z[ok])) or np.max(np.abs(z[ok] - y[ok]) / np.abs(y[ok])) > 1e-6:
                bad.append(form)
    return bad


_ENGINE = None


def entry_block(text: str, eq_id: str) -> tuple[int, int] | None:
    """The line span [start, end) of an entry under ``expressions:`` (2-space key)."""
    lines = text.splitlines(keepends=True)
    key = re.compile(r"^  (['\"]?)" + re.escape(eq_id) + r"\1:\s*$")
    start = next((i for i, ln in enumerate(lines) if key.match(ln)), None)
    if start is None:
        return None
    end = next((j for j in range(start + 1, len(lines)) if re.match(r"^  \S|^\S", lines[j])), len(lines))
    return start, end


def dump_entry(eq_id: str, entry: dict[str, Any]) -> str:
    block = yaml.safe_dump({eq_id: entry}, sort_keys=False, allow_unicode=True, width=80)
    return "".join("  " + ln if ln.strip() else ln for ln in block.splitlines(keepends=True))


def put_entry(text: str, eq_id: str, entry: dict[str, Any]) -> str:
    lines = text.splitlines(keepends=True)
    span = entry_block(text, eq_id)
    if span is None:
        return text + ("" if text.endswith("\n") else "\n") + dump_entry(eq_id, entry)
    return "".join(lines[:span[0]]) + dump_entry(eq_id, entry) + "".join(lines[span[1]:])


def set_version(text: str, version: int) -> str:
    return re.sub(r"(?m)^(  version: )(\d+)$", lambda m: f"{m.group(1)}{version}", text, count=1)


def drop_entry(text: str, eq_id: str) -> str:
    span = entry_block(text, eq_id)
    if span is None:
        return text
    lines = text.splitlines(keepends=True)
    return "".join(lines[:span[0]] + lines[span[1]:])


def note_companion(text: str, catalog: str, ids: list[str]) -> str:
    """One metadata line after ``version`` naming the companion and every member that moved to it."""
    old = re.search(r'(?m)^  companion: "[^"]*?: (.*), the authors pinned points \(a data catalog\)"\n', text)
    moved = set(ids) | (set(old.group(1).split(", ")) if old else set())
    order = sorted(moved, key=lambda i: [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", i)])
    line = f"  companion: {json.dumps(f'{catalog}-points: ' + ', '.join(order) + ', the authors pinned points (a data catalog)')}\n"
    if old:
        return text.replace(old.group(0), line)
    return re.sub(r"(?m)^(  version: \d+\n)", lambda m: m.group(1) + line, text, count=1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalogs", default=str(B.paths.CATALOGS))
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    root = Path(a.catalogs)
    from symbolic_data._evaluation import load_engine
    engine = load_engine(None)
    problems = {(p["set"], p["id"]): p for p in B.paths.final_problems()}
    changed: dict[str, list[str]] = {}
    texts: dict[str, str] = {}
    for tag, (catalog, ids) in FAMILIES.items():
        text = texts.setdefault(catalog, (root / f"{catalog}.yaml").read_text())
        release = yaml.safe_load(text)["expressions"]
        sources = release[next(iter(release))].get("sources")
        built = {}
        for eq_id in sorted(ids):
            entry = B.entry_of(problems[(tag, eq_id)])
            old = release.get(eq_id) if eq_id not in ADDED else None
            if old:   # the release entry's names and sources stay; the authors' formula and ranges replace DSO's
                for key, spec in entry["vars"].items():
                    if key in old.get("vars", {}):
                        spec["name"] = old["vars"][key]["name"]
                entry["source_audit"] = f"the authors' definition (catalog-prep {tag}) replaces the DSO transcription"
            else:
                entry["source_audit"] = f"the authors' member (catalog-prep {tag}), missing from the DSO transcription"
            rest = {k: v for k, v in entry.items()
                    if k not in ("raw", "prepared", "n_variables", "subset", "input_distributions")}
            entry = {"raw": entry["raw"], "prepared": entry["prepared"], "n_variables": entry["n_variables"],
                     "sources": (old or {}).get("sources", sources), **rest}
            built[eq_id] = entry
        failed = B.check(built, 64, engine)
        if failed:
            raise SystemExit(f"{catalog}: {failed}")
        for eq_id, entry in built.items():
            text = put_entry(text, eq_id, entry)
            changed.setdefault(catalog, []).append(eq_id)
        texts[catalog] = text
    for (catalog, eq_id), box in RANGE_FIXES.items():
        text = texts.setdefault(catalog, (root / f"{catalog}.yaml").read_text())
        doc = yaml.safe_load(text)["expressions"][eq_id]
        for key, spec in doc["vars"].items():
            if key != "v0":
                spec["sample_range"] = list(box)
        doc["source_audit"] = "range of Uy et al. 2011 Table 3 ([-1, 1]^2; DSO: [0, 1]^2)"
        doc.get("meta", {}).pop("finite_fraction", None)
        if B.check({eq_id: doc}, 64, engine):
            raise SystemExit(f"{catalog}:{eq_id} fails its check")
        texts[catalog] = put_entry(text, eq_id, doc)
        changed.setdefault(catalog, []).append(eq_id)
    text = (root / "fastsrb.v2.yaml").read_text()
    entries = yaml.safe_load(text)["expressions"]
    fixed = {}
    for eq_id, (edit, note) in _fastsrb_fixes().items():
        e = entries[eq_id]
        edit(e)
        e["source_audit"] = _audit_note(note, e)
        e.get("meta", {}).pop("finite_fraction", None)
        fixed[eq_id] = e
    for eq_id, e in entries.items():                    # accept forms that are not the formula (any entry)
        bad = accept_mismatches(e)
        if bad:
            e["accept"] = [a for a in e["accept"] if a not in bad]
            e["accept_dropped"] = sorted(set(e.get("accept_dropped") or []) | set(bad))
            base = _fastsrb_fixes()[eq_id][1] if eq_id in fixed else None
            e["source_audit"] = _audit_note(base, e)
            e.get("meta", {}).pop("finite_fraction", None)
            fixed[eq_id] = e
    failed = B.check(fixed, 64, engine)
    if failed:
        raise SystemExit(f"fastsrb.v2: {failed}")
    for eq_id, e in fixed.items():
        text = put_entry(text, eq_id, e)
    texts["fastsrb.v2"], changed["fastsrb.v2"] = text, list(fixed)
    for catalog, ids in MOVED.items():
        path = root / f"{catalog}.yaml"
        if not path.exists():
            print(f"{catalog}: removed (already patched)")
            continue
        text = texts.get(catalog) or path.read_text()
        dropped = [eq_id for eq_id in ids if entry_block(text, eq_id) is not None]
        for eq_id in dropped:
            text = drop_entry(text, eq_id)
        if not dropped:                                     # never in the release (neat has no Neat-6)
            continue
        if not yaml.safe_load(text).get("expressions"):     # every member moved: the catalog goes
            texts.pop(catalog, None)
            if not a.dry_run:
                path.unlink()
            print(f"{catalog}: removed, its members moved to {catalog}-points")
            continue
        texts[catalog] = note_companion(text, catalog, dropped)
        changed.setdefault(catalog, []).extend(f"{i} -> {catalog}-points" for i in dropped)
    for catalog, ids in DROPPED.items():
        text = texts.get(catalog) or (root / f"{catalog}.yaml").read_text()
        dropped = [eq_id for eq_id in ids if entry_block(text, eq_id) is not None]
        if not dropped:
            continue
        for eq_id in dropped:
            text = drop_entry(text, eq_id)
        line = f"  dropped: {json.dumps(', '.join(dropped) + ': DSO repository sanity tests, not a benchmark (srbf ruling 2026-11-03)')}\n"
        texts[catalog] = text if line in text else re.sub(r"(?m)^(  version: \d+\n)", lambda m: m.group(1) + line, text, count=1)
        changed.setdefault(catalog, []).extend(f"{i} dropped" for i in dropped)
    for catalog in TEST_CATALOGS:
        text = texts.get(catalog) or (root / f"{catalog}.yaml").read_text()
        added = []
        for eq_id, e in yaml.safe_load(text)["expressions"].items():
            keys = sorted((k for k in e["vars"] if k != "v0"), key=lambda k: int(k[1:]))
            tests = B.authors_tests.tests_for(f"release/{catalog}", eq_id, [e["vars"][k]["name"] for k in keys])
            if tests == [("", None, None)]:
                continue
            source = B.paths.public(B.authors_tests.inventory()[(f"release/{catalog}", eq_id)]["source"])[:400]
            built = {}
            for suffix, subset, block in tests:
                x = {k: v for k, v in copy.deepcopy(e).items() if k not in ("test", "test_source", "subset")}
                if subset:
                    x["subset"] = subset
                if block is not None:
                    x["test"], x["test_source"] = block, source
                built[eq_id + suffix] = x
            failed = B.check(built, 64, engine)
            if failed:
                raise SystemExit(f"{catalog}: {failed}")
            for k, x in built.items():
                text = put_entry(text, k, x)
                added.append(k)
        if added:
            texts[catalog] = text
            changed.setdefault(catalog, []).append(f"authors' tests on {len(added)} entries")
    for catalog, text in texts.items():
        text = set_version(text, RELEASED[catalog] + 1)
        if text == (root / f"{catalog}.yaml").read_text():
            print(f"{catalog}: unchanged (already patched)")
            continue
        yaml.safe_load(text)                                # still a valid catalog
        if not a.dry_run:
            (root / f"{catalog}.yaml").write_text(text)
        print(f"{catalog}: {', '.join(changed[catalog])}")


if __name__ == "__main__":
    main()
