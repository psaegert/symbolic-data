"""Structural lint for declarative catalog yamls — the publish-time gate.

Born from the FastSRB III.21.20 lesson (2026-07-10 audit + upstream PR viktmar/FastSRB#1): an
upstream spec carried a leaked token inside a sample_range list and YAML-string number literals,
which the strict loader only surfaced at REALIZE time (a permanently-placeholder entry in a
published catalog). This lint moves that discovery to import/publish time.

Checks per entry (structural only — fast; realize-validation stays a separate step):
- sample_range: exactly 2 elements, both numeric (int/float, NOT str), low <= high
- sample_type: [base, sign] with base in {uni, log, int}, sign in {pos, neg, pos_neg};
  log requires same-sign nonzero bounds
- n_variables present and == count of v-keys excluding v0
- prepared present and non-empty; eq_ids unique (dict keys are inherently unique -- checked
  against duplicate-normalized collisions only if requested)
- metadata: name, version present; license key present (warning for pre-audit catalogs,
  error for anything published after 2026-07-10 -- see GRANDFATHERED)

Usage:
    PYTHONPATH=src python tools/lint_catalogs.py [paths...]      # default: all catalog yamls
Exit code 0 = clean (grandfather warnings allowed), 1 = errors.
publish_catalogs.py / publish_catalogs_sa.py call `lint_paths` as a hard pre-publish gate.
"""
from __future__ import annotations

import glob
import os
import sys

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_GLOBS = [
    os.path.join(HERE, "..", "assets", "catalogs", "*.yaml"),
    os.path.join(HERE, "..", "assets_sa", "catalogs", "*.yaml"),
]
# published before the lint existed; missing license keys are WARNINGS here (fold in at the
# next version bump per AUDIT_2026-07-10.md), errors everywhere else
GRANDFATHERED_NO_LICENSE = {
    "constant", "fastsrb", "feynman", "grammarvae", "jin", "keijzer", "korns", "koza",
    "lample-charton-v23", "livermore", "livermore2", "meier", "neat", "nguyen", "nonic",
    "pagie", "poly", "r-rationals", "sine", "srsd-dummy", "v23-val", "vladislavleva",
}
# immutable published version files superseded by a newer version: kept byte-identical forever
# so name@1 pins keep resolving; their defects are the REASON the newer version exists.
# Errors downgrade to warnings here -- and only here.
PINNED_SUPERSEDED = {"fastsrb.yaml"}
BASES = {"uni", "log", "int"}
SIGNS = {"pos", "neg", "pos_neg"}


def lint_file(path: str) -> tuple[list[str], list[str]]:
    errors, warnings = [], []
    name = os.path.basename(path).rsplit(".yaml", 1)[0].removesuffix(".v2")
    doc = yaml.safe_load(open(path, encoding="utf-8"))
    if not isinstance(doc, dict):
        return [f"{path}: not a mapping"], []

    # generative specs (v23-val / lample-charton) have a different schema; only check loadability
    if "skeletons" in doc or doc.get("type") in ("lample_charton",):
        if "license" not in doc and name not in GRANDFATHERED_NO_LICENSE:
            errors.append(f"{name}: generative spec without license key")
        return errors, warnings

    meta = doc.get("metadata", {})
    exprs = doc.get("expressions", doc if "metadata" not in doc else {})
    for key in ("name", "version"):
        if key not in meta:
            errors.append(f"{name}: metadata.{key} missing")
    if "license" not in meta:
        (warnings if name in GRANDFATHERED_NO_LICENSE else errors).append(
            f"{name}: metadata.license missing"
            + (" (grandfathered; fold in at next version)" if name in GRANDFATHERED_NO_LICENSE else ""))

    for eq_id, entry in exprs.items():
        if not isinstance(entry, dict):
            errors.append(f"{name}:{eq_id}: entry not a mapping")
            continue
        where = f"{name}:{eq_id}"
        prepared = entry.get("prepared")
        if not isinstance(prepared, str) or not prepared.strip():
            errors.append(f"{where}: prepared missing/empty")
        variables = entry.get("vars") or {}
        n_inputs = len([k for k in variables if k != "v0"])
        if "n_variables" not in entry:
            errors.append(f"{where}: n_variables missing (registry identity keys on it)")
        elif int(entry["n_variables"]) != n_inputs:
            errors.append(f"{where}: n_variables={entry['n_variables']} != {n_inputs} input vars")
        for vk, spec in variables.items():
            if vk == "v0" or not isinstance(spec, dict):
                continue
            vwhere = f"{where}.{vk}"
            sr = spec.get("sample_range")
            st = spec.get("sample_type")
            if sr is None and st is None:
                continue                      # spec-less var (some sources omit; loader handles)
            if not isinstance(sr, list) or len(sr) != 2:
                errors.append(f"{vwhere}: sample_range must be exactly [low, high], got {sr!r}")
                continue
            bad_type = [b for b in sr if not isinstance(b, (int, float)) or isinstance(b, bool)]
            if bad_type:
                errors.append(f"{vwhere}: non-numeric sample_range element(s) {bad_type!r} "
                              f"(YAML-string literals or leaked tokens)")
                continue
            low, high = float(sr[0]), float(sr[1])
            if low > high:
                errors.append(f"{vwhere}: sample_range low > high ({low} > {high})")
            if not (isinstance(st, list) and len(st) == 2 and st[0] in BASES and st[1] in SIGNS):
                errors.append(f"{vwhere}: sample_type must be [base in {sorted(BASES)}, "
                              f"sign in {sorted(SIGNS)}], got {st!r}")
                continue
            if st[0] == "log" and (low * high <= 0):
                errors.append(f"{vwhere}: log base requires same-sign nonzero bounds, got [{low}, {high}]")
    return errors, warnings


def lint_paths(paths: list[str] | None = None) -> bool:
    """Lint the given yamls (default: all catalogs). Prints findings; returns True iff clean."""
    if not paths:
        paths = sorted(p for g in DEFAULT_GLOBS for p in glob.glob(g))
    all_errors, all_warnings = [], []
    for path in paths:
        errors, warnings = lint_file(path)
        if os.path.basename(path) in PINNED_SUPERSEDED:
            warnings = warnings + [f"{e} [pinned-superseded artifact; fixed in newer version]"
                                   for e in errors]
            errors = []
        all_errors += errors
        all_warnings += warnings
    for w in all_warnings:
        print(f"WARN  {w}")
    for e in all_errors:
        print(f"ERROR {e}")
    print(f"lint: {len(paths)} files, {len(all_errors)} errors, {len(all_warnings)} warnings")
    return not all_errors


if __name__ == "__main__":
    sys.exit(0 if lint_paths(sys.argv[1:] or None) else 1)
