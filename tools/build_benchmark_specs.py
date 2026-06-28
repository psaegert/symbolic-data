#!/usr/bin/env python
"""Regenerate the curated benchmark spec YAMLs (Feynman, Nguyen) shipped as package data.

This is *build-time* tooling, not shipped in the wheel. It converts canonical, machine-readable
sources into the ``prepared``/``vars`` spec format consumed by
:class:`symbolic_data.benchmarks.spec.SpecBenchmark`, and verifies every generated equation with a
numerical oracle before writing:

* **Feynman** -- the Feynman Symbolic Regression Database (Udrescu & Tegmark 2020, AI Feynman). The
  authoritative ``FeynmanEquations.csv`` (Formula + per-variable name/low/high) is fetched from the
  versioned ``psaegert/ansr-data`` HuggingFace dataset. Six rows carry a wrong ``# variables`` count
  (a known FSReD typo); the count is taken from the populated columns instead and the correction is
  reported.
* **Nguyen** -- the standard Nguyen suite (Uy et al. 2011), formulas + sampling ranges pinned to the
  DSO/DSR convention (Petersen et al. 2021, deep-symbolic-optimization ``benchmarks.csv``). Nguyen-1
  through Nguyen-10 are cross-confirmed against ``psaegert/ansr-data`` ``test_set/nguyen/nguyen.csv``.

The oracle (requires ``sympy``) evaluates the original source Formula via ``sympy.lambdify`` and the
generated ``prepared`` expression via SimpliPy on the *same* sampled inputs and asserts ``allclose``;
this checks faithfulness through a fully independent parse path. The script exits non-zero if any
equation fails to build, sample, or match.

Usage::

    python tools/build_benchmark_specs.py            # regenerate + verify
    python tools/build_benchmark_specs.py --check     # verify only (fail if output would change)
"""
from __future__ import annotations

import argparse
import csv
import re
import warnings
from pathlib import Path

import numpy as np
import yaml

from simplipy import SimpliPyEngine

ANSR_DATA_REPO = "psaegert/ansr-data"
FEYNMAN_CSV = "test_set/feynman/FeynmanEquations.csv"
DATA_DIR = Path(__file__).resolve().parent.parent / "src" / "symbolic_data" / "benchmarks" / "data"

PI_LITERAL = repr(float(np.pi))
# Source function name -> SimpliPy-parseable name. ``pi`` is substituted with a numeric literal.
FUNC_TO_SIMPLIPY = {"arcsin": "asin", "arccos": "acos", "arctan": "atan", "ln": "log"}
KNOWN_FUNCS = {"sin", "cos", "exp", "log", "asin", "acos", "atan", "sinh", "cosh", "tanh", "sqrt", "abs"}

FEYNMAN_SOURCES = [
    "Feynman Symbolic Regression Database (Udrescu & Tegmark 2020, AI Feynman)",
    "formulas + sampling ranges from FeynmanEquations.csv (psaegert/ansr-data)",
]

# Canonical Nguyen suite: (id, original infix formula, (low, high)). Verified against the DSO/DSR
# benchmarks.csv; Nguyen-1..10 also match psaegert/ansr-data test_set/nguyen/nguyen.csv.
NGUYEN = [
    ("Nguyen-1", "x1**3 + x1**2 + x1", (-1.0, 1.0)),
    ("Nguyen-2", "x1**4 + x1**3 + x1**2 + x1", (-1.0, 1.0)),
    ("Nguyen-3", "x1**5 + x1**4 + x1**3 + x1**2 + x1", (-1.0, 1.0)),
    ("Nguyen-4", "x1**6 + x1**5 + x1**4 + x1**3 + x1**2 + x1", (-1.0, 1.0)),
    ("Nguyen-5", "sin(x1**2) * cos(x1) - 1", (-1.0, 1.0)),
    ("Nguyen-6", "sin(x1) + sin(x1 + x1**2)", (-1.0, 1.0)),
    ("Nguyen-7", "log(x1 + 1) + log(x1**2 + 1)", (0.0, 2.0)),
    ("Nguyen-8", "sqrt(x1)", (0.0, 4.0)),
    ("Nguyen-9", "sin(x1) + sin(x2**2)", (0.0, 1.0)),
    ("Nguyen-10", "2 * sin(x1) * cos(x2)", (0.0, 1.0)),
    ("Nguyen-11", "x1**x2", (0.0, 1.0)),
    ("Nguyen-12", "x1**4 - x1**3 + x2**2 / 2 - x2", (0.0, 1.0)),
]
NGUYEN_SOURCES = [
    "Nguyen benchmark (Uy et al. 2011)",
    "formulas + ranges from the DSO/DSR standard (Petersen et al. 2021, deep-symbolic-optimization)",
    "Nguyen-1..10 cross-confirmed against psaegert/ansr-data test_set/nguyen/nguyen.csv",
]


def _remap_var_names(formula: str, names: list[str]) -> str:
    """Replace physical variable names with ``v1..vn`` (whole-word, longest-first, collision-safe)."""
    out = formula
    for k in sorted(range(len(names)), key=lambda i: -len(names[i])):
        out = re.sub(r"(?<![\w.])" + re.escape(names[k]) + r"(?![\w.])", f"\x00{k + 1}\x00", out)
    return re.sub(r"\x00(\d+)\x00", lambda m: f"v{m.group(1)}", out)


def _prepared_from_source(formula: str, names: list[str]) -> str:
    prepared = _remap_var_names(formula, names)
    for fn, repl in FUNC_TO_SIMPLIPY.items():
        prepared = re.sub(r"\b" + fn + r"\b", repl, prepared)
    prepared = re.sub(r"\bpi\b", PI_LITERAL, prepared)
    return prepared


def _validate_var_set(eq_id: str, prepared: str, nvars: int) -> None:
    tokens = set(re.findall(r"[A-Za-z_]\w*", prepared))
    non_funcs = {t for t in tokens if t not in KNOWN_FUNCS}
    v_vars = {t for t in non_funcs if re.fullmatch(r"v\d+", t)}
    unmapped = non_funcs - v_vars
    if unmapped:
        raise ValueError(f"{eq_id}: unmapped tokens {sorted(unmapped)} in prepared {prepared!r}")
    expected = {f"v{k}" for k in range(1, nvars + 1)}
    if v_vars != expected:
        raise ValueError(f"{eq_id}: variable set {sorted(v_vars)} != expected {sorted(expected)} (prepared {prepared!r})")


def build_feynman() -> tuple[dict, list[str]]:
    from huggingface_hub import hf_hub_download

    csv_path = hf_hub_download(repo_id=ANSR_DATA_REPO, filename=FEYNMAN_CSV, repo_type="dataset")
    rows = [r for r in csv.DictReader(open(csv_path, encoding="utf-8-sig")) if r.get("Filename", "").strip()]

    spec: dict = {}
    corrections: list[str] = []
    for r in rows:
        eq_id = r["Filename"].strip()
        formula = r["Formula"].strip()
        # The CSV "# variables" column is unreliable (6 known typos); use the populated columns.
        nvars = 0
        while r.get(f"v{nvars + 1}_name", "").strip() != "":
            nvars += 1
        declared = r.get("# variables", "").strip()
        if str(declared) != str(nvars):
            corrections.append(f"{eq_id}: CSV #variables={declared} -> {nvars} (from populated columns)")
        names = [r[f"v{k}_name"].strip() for k in range(1, nvars + 1)]
        ranges = [(float(r[f"v{k}_low"]), float(r[f"v{k}_high"])) for k in range(1, nvars + 1)]
        prepared = _prepared_from_source(formula, names)
        _validate_var_set(eq_id, prepared, nvars)
        vb = {
            f"v{k + 1}": {"name": names[k], "sample_range": [ranges[k][0], ranges[k][1]], "sample_type": ["uni", "pos"]}
            for k in range(nvars)
        }
        vb["v0"] = {"name": r["Output"].strip()}
        spec[eq_id] = {"raw": formula, "prepared": prepared, "n_variables": nvars, "sources": list(FEYNMAN_SOURCES), "vars": vb}
    return spec, corrections


def build_nguyen() -> dict:
    spec: dict = {}
    for eq_id, formula, (lo, hi) in NGUYEN:
        nvars = max((int(m.group(1)) for m in re.finditer(r"x(\d+)", formula)), default=1)
        prepared = re.sub(r"x(\d+)", lambda m: f"v{m.group(1)}", formula)
        _validate_var_set(eq_id, prepared, nvars)
        vb = {
            f"v{k + 1}": {"name": f"x{k + 1}", "sample_range": [float(lo), float(hi)], "sample_type": ["uni", "pos"]}
            for k in range(nvars)
        }
        vb["v0"] = {"name": "y"}
        spec[eq_id] = {"raw": formula, "prepared": prepared, "n_variables": nvars, "sources": list(NGUYEN_SOURCES), "vars": vb}
    return spec


def oracle(name: str, spec: dict, engine: SimpliPyEngine) -> None:
    """Assert simplipy(prepared) == sympy(original) on shared sampled inputs, for every equation."""
    import sympy as sp

    from symbolic_data.benchmarks.spec import SpecBenchmark

    fn_locals = {
        "arcsin": sp.asin, "arccos": sp.acos, "arctan": sp.atan, "ln": sp.log,
        "exp": sp.exp, "sqrt": sp.sqrt, "sin": sp.sin, "cos": sp.cos, "tanh": sp.tanh, "pi": sp.pi,
    }
    bench = SpecBenchmark(spec, name=name, simplipy_engine=engine, random_state=0)
    failures = []
    for eq_id in bench.equation_ids():
        entry = spec[eq_id]
        formula = entry["raw"]
        names = [entry["vars"][f"v{k}"]["name"] for k in range(1, entry["n_variables"] + 1)]
        nvars = entry["n_variables"]
        loc = {n: sp.Symbol(n) for n in names}
        loc.update(fn_locals)
        f = sp.lambdify([sp.Symbol(n) for n in names], sp.sympify(formula, locals=loc), modules=["numpy"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            s = bench.sample(eq_id, n_points=200, random_state=7)
        X, y_engine = s["data"]["X"], s["data"]["y"]
        with np.errstate(all="ignore"):
            y_sympy = np.broadcast_to(np.asarray(f(*[X[:, k] for k in range(nvars)]), dtype=float), y_engine.shape)
        finite = np.isfinite(y_sympy) & np.isfinite(y_engine)
        # Pure relative bound (tiny absolute floor only for genuine zeros) so the oracle cannot be
        # loosened by small output magnitudes.
        if finite.sum() < 2 or not np.allclose(y_sympy[finite], y_engine[finite], rtol=1e-9, atol=1e-12):
            failures.append(eq_id)
    if failures:
        raise SystemExit(f"ORACLE FAILED for {name}: {failures}")
    print(f"  oracle: {name} {len(bench.equation_ids())}/{len(bench.equation_ids())} equations match (rtol=1e-9)")


def _dump(spec: dict) -> str:
    return yaml.safe_dump(spec, sort_keys=False, default_flow_style=False, allow_unicode=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="verify only; fail if the output files would change")
    args = ap.parse_args()

    engine = SimpliPyEngine.load("dev_7-3", install=True)

    feynman, corrections = build_feynman()
    nguyen = build_nguyen()
    print(f"built feynman={len(feynman)} nguyen={len(nguyen)}")
    for c in corrections:
        print(f"  FSReD correction: {c}")

    oracle("feynman", feynman, engine)
    oracle("nguyen", nguyen, engine)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for fname, spec in [("feynman.yaml", feynman), ("nguyen.yaml", nguyen)]:
        out = DATA_DIR / fname
        text = _dump(spec)
        if args.check:
            current = out.read_text(encoding="utf-8") if out.exists() else ""
            if current != text:
                raise SystemExit(f"{out} is stale; re-run tools/build_benchmark_specs.py")
            print(f"  check: {out} up to date")
        else:
            out.write_text(text, encoding="utf-8")
            print(f"  wrote {out} ({len(spec)} equations)")


if __name__ == "__main__":
    main()
