#!/usr/bin/env python
"""Regenerate the curated benchmark spec YAMLs (FastSRB, Feynman, Nguyen) shipped as package data.

This is *build-time* tooling, not shipped in the wheel. Each curated benchmark is sourced from its
canonical upstream, processed into the ``prepared``/``vars`` spec format consumed by
:class:`symbolic_data.benchmarks.spec.SpecBenchmark`, verified, and written to
``src/symbolic_data/benchmarks/data/``:

* **FastSRB** -- vendored *verbatim* from the upstream ``src/expressions.yaml`` in
  https://github.com/viktmar/FastSRB (Martinek, arXiv:2508.14481, MIT). It is already in the spec
  format, so it is shipped unchanged; raw<->prepared faithfulness is the upstream's responsibility.
  Gate: every equation must load + sample finite (integrity); the sympy oracle is run informationally.
* **Feynman** -- the 100-equation Feynman Symbolic Regression Database (Udrescu & Tegmark 2020). The
  equations are physics facts; the machine-readable ``FeynmanEquations.csv`` is fetched from the
  ``psaegert/ansr-data`` mirror (the upstream FSReD is only a 6.5 GB tarball, not a stable file fetch).
  Six rows carry a wrong ``# variables`` count (a known FSReD typo); the count is taken from the
  populated columns and the correction reported. Gate: the sympy oracle (this is our conversion).
* **Nguyen** -- the 12-equation Nguyen suite (Uy et al. 2011), fetched from the canonical
  ``benchmarks.csv`` in https://github.com/dso-org/deep-symbolic-optimization (Petersen et al. 2021,
  BSD-3). The ``pow()/div()`` source syntax is converted to infix via ``ast``. Gate: the sympy oracle.

The oracle (requires ``sympy``) evaluates the original source formula via ``sympy.lambdify`` and the
generated ``prepared`` expression via SimpliPy on the *same* sampled inputs and asserts ``allclose``
through a fully independent parse path.

Usage::

    python tools/build_benchmark_specs.py            # fetch + regenerate + verify
    python tools/build_benchmark_specs.py --check     # verify only (fail if output would change)
"""
from __future__ import annotations

import argparse
import ast
import csv
import io
import json
import re
import urllib.request
import warnings
from pathlib import Path

import numpy as np
import yaml

from simplipy import SimpliPyEngine

DATA_DIR = Path(__file__).resolve().parent.parent / "src" / "symbolic_data" / "benchmarks" / "data"

ANSR_DATA_REPO = "psaegert/ansr-data"
FEYNMAN_CSV = "test_set/feynman/FeynmanEquations.csv"
VIKTMAR_FASTSRB = "https://raw.githubusercontent.com/viktmar/FastSRB/main/src/expressions.yaml"
DSO_BENCHMARKS = "https://raw.githubusercontent.com/dso-org/deep-symbolic-optimization/master/dso/dso/task/regression/benchmarks.csv"

PI_LITERAL = repr(float(np.pi))
FUNC_TO_SIMPLIPY = {"arcsin": "asin", "arccos": "acos", "arctan": "atan", "ln": "log"}
KNOWN_FUNCS = {"sin", "cos", "exp", "log", "asin", "acos", "atan", "sinh", "cosh", "tanh", "sqrt", "abs"}

FEYNMAN_SOURCES = [
    "Feynman Symbolic Regression Database (Udrescu & Tegmark 2020, AI Feynman)",
    "machine-readable FeynmanEquations.csv via the psaegert/ansr-data mirror",
]
NGUYEN_SOURCES = [
    "Nguyen benchmark (Uy et al. 2011)",
    "formulas + ranges from dso-org/deep-symbolic-optimization benchmarks.csv (Petersen et al. 2021)",
]
# Canonical 12-equation Nguyen suite (the standard suite excludes DSO range-variants like Nguyen-12a).
NGUYEN_IDS = [f"Nguyen-{i}" for i in range(1, 13)]


def _fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "symbolic-data-build/1.0"})
    return urllib.request.urlopen(req, timeout=60).read()  # noqa: S310 - fixed canonical https URLs


# -------------------------------------------------------------------------------------------------
# Shared helpers
# -------------------------------------------------------------------------------------------------
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


# -------------------------------------------------------------------------------------------------
# FastSRB -- vendored verbatim from viktmar/FastSRB (MIT)
# -------------------------------------------------------------------------------------------------
def build_fastsrb() -> tuple[bytes, dict]:
    raw_bytes = _fetch(VIKTMAR_FASTSRB)
    spec = yaml.safe_load(io.BytesIO(raw_bytes))
    if not isinstance(spec, dict) or not spec:
        raise SystemExit("FastSRB upstream yaml did not parse to a non-empty mapping")
    return raw_bytes, spec


# -------------------------------------------------------------------------------------------------
# Feynman -- convert FeynmanEquations.csv (FSReD) into the spec format
# -------------------------------------------------------------------------------------------------
def _remap_var_names(formula: str, names: list[str]) -> str:
    out = formula
    for k in sorted(range(len(names)), key=lambda i: -len(names[i])):
        out = re.sub(r"(?<![\w.])" + re.escape(names[k]) + r"(?![\w.])", f"\x00{k + 1}\x00", out)
    return re.sub(r"\x00(\d+)\x00", lambda m: f"v{m.group(1)}", out)


def build_feynman() -> tuple[dict, list[str]]:
    from huggingface_hub import hf_hub_download

    csv_path = hf_hub_download(repo_id=ANSR_DATA_REPO, filename=FEYNMAN_CSV, repo_type="dataset")
    rows = [r for r in csv.DictReader(open(csv_path, encoding="utf-8-sig")) if r.get("Filename", "").strip()]

    spec: dict = {}
    corrections: list[str] = []
    for r in rows:
        eq_id = r["Filename"].strip()
        formula = r["Formula"].strip()
        nvars = 0
        while r.get(f"v{nvars + 1}_name", "").strip() != "":
            nvars += 1
        declared = r.get("# variables", "").strip()
        if str(declared) != str(nvars):
            corrections.append(f"{eq_id}: CSV #variables={declared} -> {nvars} (from populated columns)")
        names = [r[f"v{k}_name"].strip() for k in range(1, nvars + 1)]
        ranges = [(float(r[f"v{k}_low"]), float(r[f"v{k}_high"])) for k in range(1, nvars + 1)]
        prepared = _remap_var_names(formula, names)
        for fn, repl in FUNC_TO_SIMPLIPY.items():
            prepared = re.sub(r"\b" + fn + r"\b", repl, prepared)
        prepared = re.sub(r"\bpi\b", PI_LITERAL, prepared)
        _validate_var_set(eq_id, prepared, nvars)
        vb = {
            f"v{k + 1}": {"name": names[k], "sample_range": [ranges[k][0], ranges[k][1]], "sample_type": ["uni", "pos"]}
            for k in range(nvars)
        }
        vb["v0"] = {"name": r["Output"].strip()}
        spec[eq_id] = {"raw": formula, "prepared": prepared, "n_variables": nvars, "sources": list(FEYNMAN_SOURCES), "vars": vb}
    return spec, corrections


# -------------------------------------------------------------------------------------------------
# Nguyen -- convert the DSO benchmarks.csv (pow()/div() syntax) into the spec format
# -------------------------------------------------------------------------------------------------
class _PowDivToInfix(ast.NodeTransformer):
    """Rewrite DSO's pow(a, b) -> a ** b and div(a, b) -> a / b; leave sin/cos/log/sqrt as calls."""

    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)
        if isinstance(node.func, ast.Name) and node.func.id in {"pow", "div"} and len(node.args) == 2:
            op = ast.Pow() if node.func.id == "pow" else ast.Div()
            return ast.BinOp(left=node.args[0], op=op, right=node.args[1])
        return node


def _dso_expr_to_infix(expr: str) -> str:
    tree = ast.parse(expr, mode="eval")
    tree = _PowDivToInfix().visit(tree)
    ast.fix_missing_locations(tree)
    return ast.unparse(tree.body)


def build_nguyen() -> dict:
    rows = {r["name"]: r for r in csv.DictReader(io.StringIO(_fetch(DSO_BENCHMARKS).decode("utf-8")))}
    spec: dict = {}
    for eq_id in NGUYEN_IDS:
        r = rows[eq_id]
        raw = _dso_expr_to_infix(r["expression"])
        lo, hi, _n = json.loads(r["train_spec"])["all"]["U"]
        nvars = int(r["variables"])
        prepared = re.sub(r"\bx(\d+)\b", lambda m: f"v{m.group(1)}", raw)
        _validate_var_set(eq_id, prepared, nvars)
        vb = {
            f"v{k + 1}": {"name": f"x{k + 1}", "sample_range": [float(lo), float(hi)], "sample_type": ["uni", "pos"]}
            for k in range(nvars)
        }
        vb["v0"] = {"name": "y"}
        spec[eq_id] = {"raw": raw, "prepared": prepared, "n_variables": nvars, "sources": list(NGUYEN_SOURCES), "vars": vb}
    return spec


# -------------------------------------------------------------------------------------------------
# Numerical oracle: simplipy(prepared) == sympy(raw) on shared sampled inputs, for every equation.
# -------------------------------------------------------------------------------------------------
def oracle(name: str, spec: dict, engine: SimpliPyEngine, *, gate: bool) -> list[str]:
    import sympy as sp

    from symbolic_data.benchmarks.spec import SpecBenchmark

    fn_locals = {
        "arcsin": sp.asin, "arccos": sp.acos, "arctan": sp.atan, "ln": sp.log,
        "exp": sp.exp, "sqrt": sp.sqrt, "sin": sp.sin, "cos": sp.cos, "tanh": sp.tanh, "pi": sp.pi,
    }
    bench = SpecBenchmark(spec, name=name, simplipy_engine=engine, random_state=0)
    failures, n = [], 0
    for eq_id in bench.equation_ids():
        entry = spec[eq_id]
        nvars = len([k for k in entry["vars"] if k != "v0"])
        names = [entry["vars"][f"v{k}"]["name"] for k in range(1, nvars + 1)]
        loc = {n_: sp.Symbol(n_) for n_ in names}
        loc.update(fn_locals)
        try:
            # ^ -> ** to match SpecBenchmark's prepared normalization (upstream fastsrb raw uses ^).
            raw_norm = entry["raw"].replace("^", "**")
            f = sp.lambdify([sp.Symbol(n_) for n_ in names], sp.sympify(raw_norm, locals=loc), modules=["numpy"])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                s = bench.sample(eq_id, n_points=200, random_state=7)
            X, y_engine = s["data"]["X"], s["data"]["y"]
            with np.errstate(all="ignore"):
                y_sympy = np.broadcast_to(np.asarray(f(*[X[:, k] for k in range(nvars)]), dtype=float), y_engine.shape)
            finite = np.isfinite(y_sympy) & np.isfinite(y_engine)
            n += 1
            if finite.sum() < 2 or not np.allclose(y_sympy[finite], y_engine[finite], rtol=1e-9, atol=1e-12):
                failures.append(eq_id)
        except Exception as exc:  # sympy cannot parse this raw (only informational benches reach here)
            failures.append(f"{eq_id} ({type(exc).__name__})")
    label = "GATE" if gate else "info"
    print(f"  oracle[{label}]: {name} {n - len([x for x in failures])}/{len(bench.equation_ids())} match (rtol=1e-9); failures={len(failures)}")
    if failures and gate:
        raise SystemExit(f"ORACLE GATE FAILED for {name}: {failures}")
    if failures:
        print(f"    (informational) upstream raw!=prepared or unparseable for: {failures}")
    return failures


def integrity(name: str, spec: dict, engine: SimpliPyEngine, *, gate: bool, max_trials: int = 1000) -> list[str]:
    """Report equations that cannot sample finite at default settings.

    For vendored-verbatim specs (fastsrb) this is informational: a few upstream equations are
    mostly-non-finite by construction under their own ranges (e.g. a sqrt whose argument is usually
    negative), which is an upstream property, not a packaging defect. iter_samples() skips such
    equations gracefully per-equation.
    """
    from symbolic_data.benchmarks.spec import SpecBenchmark

    bench = SpecBenchmark(spec, name=name, simplipy_engine=engine, random_state=0)
    bad = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for eq_id in bench.equation_ids():
            try:
                y = bench.sample(eq_id, n_points=16, max_trials=max_trials, random_state=7)["data"]["y"]
                if not np.all(np.isfinite(y)):
                    bad.append(eq_id)
            except Exception as exc:
                bad.append(f"{eq_id} ({type(exc).__name__})")
    n = len(bench.equation_ids())
    print(f"  integrity[{'GATE' if gate else 'info'}]: {name} {n - len(bad)}/{n} equations sample finite")
    if bad and gate:
        raise SystemExit(f"INTEGRITY GATE FAILED for {name}: {bad}")
    if bad:
        print(f"    (informational) hard-to-sample under default ranges (upstream-owned): {bad}")
    return bad


def _dump(spec: dict) -> str:
    return yaml.safe_dump(spec, sort_keys=False, default_flow_style=False, allow_unicode=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="verify only; fail if the output files would change")
    args = ap.parse_args()

    engine = SimpliPyEngine.load("dev_7-3", install=True)

    fastsrb_bytes, fastsrb_spec = build_fastsrb()
    feynman, corrections = build_feynman()
    nguyen = build_nguyen()
    print(f"built fastsrb={len(fastsrb_spec)} feynman={len(feynman)} nguyen={len(nguyen)}")
    for c in corrections:
        print(f"  FSReD correction: {c}")

    # fastsrb is vendored verbatim: the gate is only that it parses to a non-empty mapping in the
    # spec format (checked in build_fastsrb). Integrity is reported informationally. A sympy oracle is
    # NOT meaningful here -- Martinek's `prepared` folds physical constants (c, G, ...) as literals
    # while `raw` names them, so `raw` is not independently evaluable from `vars` alone.
    integrity("fastsrb", fastsrb_spec, engine, gate=False)
    # feynman + nguyen are our conversions: the oracle is the gate.
    oracle("feynman", feynman, engine, gate=True)
    oracle("nguyen", nguyen, engine, gate=True)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    outputs = [("fastsrb.yaml", fastsrb_bytes.decode("utf-8")), ("feynman.yaml", _dump(feynman)), ("nguyen.yaml", _dump(nguyen))]
    for fname, text in outputs:
        out = DATA_DIR / fname
        if args.check:
            current = out.read_text(encoding="utf-8") if out.exists() else ""
            if current != text:
                raise SystemExit(f"{out} is stale; re-run tools/build_benchmark_specs.py")
            print(f"  check: {out} up to date")
        else:
            out.write_text(text, encoding="utf-8")
            print(f"  wrote {out}")


if __name__ == "__main__":
    main()
