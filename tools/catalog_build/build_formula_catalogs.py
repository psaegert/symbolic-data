"""Build srbf's new formula catalogs: one declarative symbolic-data catalog (yaml) per catalog-prep set.

A formula problem is the benchmark's formula, its constants (or their distributions) and its input distributions;
srbf draws the points (512 + 512 per run) and sets the noise (srbf rulings 4 and 9, 2026-10-01/05). Encoding
(agreed with the SRBF tech session, 2026-11-03):

* an input is the fastsrb shorthand where it is a uniform or log-uniform range, a ``dist`` config otherwise
  (normal, a discrete set), derived from ``latent`` variables where the authors give a push-forward, and drawn by a
  joint sampler where the authors' inputs are not a product over variables (symbolic_data.inputs);
* a constant the authors draw from a distribution is drawn ONCE per problem (ruling 9(6)), seeded by the problem's
  identity, and written into ``prepared``; the spec and the seed stay in the entry's meta;
* the formula is parsed with sympy and written in SimpliPy's infix: cot -> 1/tan, acot(u) -> atan(1/u), a Piecewise
  -> its exact floor/Abs indicator form (piecewise.py), finite sums expanded.

Inputs are the catalog-prep records (final_problems.json, kind F, sets outside release/). A problem this builder
cannot encode yet is reported, not guessed (build_formula_report.json).

    build_formula_catalogs.py --out <symbolic-data>/assets/catalogs [--only fresh-91,...] [--check 64]

Needs symbolic-data with symbolic_data.inputs on the path, and sympy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import sympy
import yaml
from sympy.parsing.sympy_parser import parse_expr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths  # noqa: E402  the curated inputs (assets/upstream/catalog-prep) and the public-text filter
import piecewise  # noqa: E402  the exact Piecewise -> floor/Abs rewrite
import authors_tests  # noqa: E402  the authors' test sets (rulings 9(1), 9(7))

RULES = ("srbf formula catalogs (srbf ruling 2026-10-01/05): the benchmark's formulas, constants and input distributions; "
         "srbf draws 512 + 512 points per run and sets the noise; constants with a distribution are drawn once per "
         "problem (ruling 9(6))")


# The operators of SimpliPy 0.15 (srbf ruling 2026-11-02: whole classes) as sympy prints them.
NEEDS_015 = re.compile(r"\b(Mod|floor|ceiling|Max|Min|erf|erfc|besselj|bessely|besseli|besselk|harmonic|atan2)\(")


# catalog-prep's family sets: the authors' definitions of release-catalog members and their missing members. They are
# written INTO the release catalogs (patch_release_catalogs.py), not as catalogs of their own (their names would
# collide). Their pinned-grid members (Keijzer-6, Neat-6, Sine, Vladislavleva-2/3) are data problems that wait for
# srbf's call on where they live; fresh-5's other members are other authors' variants (Nguyen-12* NGGP;
# Nguyen-2c/9c are fresh-7's).
FAMILY_SETS = {"fresh-10": "keijzer", "fresh-11": "korns", "fresh-13": "vladislavleva", "fresh-18": "sine, poly",
               "fresh-27": "neat", "fresh-5": "nguyen"}
# Sets whose authors' input distribution is unpublished, approximated by srbf and marked with '*' (srbf ruling 2026-11-03,
# as for uDSR*). EQL robot arm (Martius & Lampert 2016, results.tex l.159-169): "sinusoidal joint target angles with
# amplitude in [-pi/2, pi/2], each joint with a different frequency"; extrapolation: amplitude [-pi, pi]. The
# generator (Kinematics-dataset.ipynb) was never released (workflow_archive/test-specs/EQL_ARM_2026-11-03.md).
# Sinusoids with distinct frequencies, sampled uniformly in time, are independent and arcsine-distributed on [-A, A]:
# each joint is A sin(u), u ~ U[0, 2 pi).
ARM = "fresh-153"
MODIFIED = {ARM: ("the authors' input generator is unpublished: srbf draws each joint angle as (pi/2) sin(u), "
                  "u ~ U[0, 2 pi) (the time-sampled limit of their sinusoidal joint targets), and (pi) sin(u) for the "
                  "extrapolation test"),
            "fresh-64": ("the authors give the noisy Bessel problem's range, x in [0, 4], but not how its points are "
                         "placed: srbf draws x uniformly on [0, 4] and the order nu uniformly from {0, ..., 4}")}
# Inputs srbf chooses where the authors state only a range (srbf ruling 2026-11-04, the same '*' rule): BMS (Guimera et al.
# 2020, SM Fig. S6A-C) fits J_nu(x), nu = 0..4, on 100 noisy points per order in [0, x_max], x_max = 4; srbf sets the
# noise itself (ruling 4).
SRBF_INPUTS = {("fresh-64", "bessel-J0-4-xmax4"): {
    "x": {"dist": {"name": "uniform", "kwargs": {"low": 0.0, "high": 4.0}}},
    "nu": {"dist": {"name": "choice", "kwargs": {"values": [0.0, 1.0, 2.0, 3.0, 4.0]}}}}}


def arm_inputs(p: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    variables, latent = {}, {}
    for i, v in enumerate(p["variables"]):
        variables[f"v{i + 1}"] = {"name": v["name"], "expr": f"{math.pi / 2!r}*sin(u{i + 1})"}
        latent[f"u{i + 1}"] = {"dist": {"name": "uniform", "kwargs": {"low": 0.0, "high": 2 * math.pi}}}
    return variables, latent


# The family sets' pinned-grid members: they leave their release catalog for a companion '<catalog>-points' data
# catalog (srbf ruling 2026-11-03, call (d)); fresh-5's Nguyen-2c/9c are fresh-7's and built there.
FAMILY_POINTS = {("fresh-10", "Keijzer-6"): "keijzer", ("fresh-27", "Neat-6"): "neat", ("fresh-18", "Sine"): "sine",
                 ("fresh-13", "Vladislavleva-2"): "vladislavleva", ("fresh-13", "Vladislavleva-3"): "vladislavleva"}


# A set whose data problems build a data catalog of the same name (build_data_catalogs.py): its formula half is named
# apart, since srbf needs unique catalog names. PySR's EmpiricalBench (Cranmer 2023): hubble, kepler, leavitt and bode
# are measured data, newton, planck, schechter and ideal_gas are formulas.
SLUGS = {"fresh-2": "pysr-empiricalbench-formulas"}


class Unsupported(Exception):
    """A problem the builder cannot encode yet (reported)."""


# ---------------------------------------------------------------------------------------------------------- inputs
_NUM = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"


def _num(s: str) -> float:
    return float(sympy.sympify(s.replace("pi", "*pi" if re.match(r"^\d", s) and "pi" in s and "*" not in s else "pi")))


def _bounds(text: str) -> tuple[float, float] | None:
    m = re.search(rf"[\[(]\s*({_NUM}|[-\d.*/ ]*pi[\d.*/ ]*)\s*,\s*({_NUM}|[-\d.*/ ]*pi[\d.*/ ]*)\s*[\])]", text)
    return (_num(m.group(1).replace(" ", "")), _num(m.group(2).replace(" ", ""))) if m else None


def input_spec(v: dict[str, Any]) -> dict[str, Any]:
    """A variable's spec from its record: the distribution text, its low/high, step and discreteness."""
    text = str(v.get("distribution") or "")
    t = text.lower()
    lo, hi = v.get("low"), v.get("high")
    if v.get("derived_from") or v.get("dependent_on") or v.get("conditional_on") or "derived" in t:
        raise Unsupported(f"derived input {v['name']}: {text[:80]}")
    if "mixture" in t or "regime" in t or "empirical" in t or "trajectory" in t:
        raise Unsupported(f"needs a joint sampler: {text[:80]}")
    lattice = ("uniform on {" in t or "drawn uniformly with replacement" in t) and v.get("step")
    if lattice and lo is not None and hi is not None:
        n = int(round((float(hi) - float(lo)) / float(v["step"]))) + 1
        return {"dist": {"name": "choice", "kwargs": {"values": [float(x) for x in np.linspace(float(lo), float(hi), n)]}}}
    if "grid" in t or re.search(r"\be\[", t) or "evenly spaced" in t or "equidistant" in t:
        raise Unsupported(f"pinned grid (a data problem): {text[:80]}")
    if lo is not None and hi is not None and float(lo) == float(hi):     # a constant column the authors pass as an input
        return {"dist": {"name": "choice", "kwargs": {"values": [float(lo)]}}}
    if v.get("discrete") or "discrete" in t or "integer" in t:
        values = re.search(r"\{([^}]*)\}", text)
        if values and "..." not in values.group(1):
            vals = [float(sympy.sympify(x.strip())) for x in values.group(1).split(",") if x.strip()]
        elif lo is not None and hi is not None and v.get("step"):
            vals = list(np.round(np.arange(float(lo), float(hi) + float(v["step"]) / 2, float(v["step"])), 12))
        else:
            raise Unsupported(f"discrete input without values: {text[:80]}")
        return {"dist": {"name": "choice", "kwargs": {"values": [float(x) for x in vals]}}}
    if "normal" in t and "uniform" not in t:
        m = re.search(rf"n\(\s*(?:mean\s*=\s*)?({_NUM})\s*,\s*(?:var\s*=\s*)?({_NUM})\s*\)", t)
        if not m:
            raise Unsupported(f"normal without parameters: {text[:80]}")
        return {"dist": {"name": "normal", "kwargs": {"loc": float(m.group(1)), "scale": math.sqrt(float(m.group(2)))}}}
    if "log" in t and ("uniform" in t or "lu[" in t):
        if "sign" in t and ("+ or -" in t or "both" in t):
            mag = re.search(rf"magnitude[^\[]*\[\s*({_NUM})\s*,\s*({_NUM})\s*\]", t)
            if not mag:
                raise Unsupported(f"signed log-uniform without magnitudes: {text[:80]}")
            return {"sample_range": [float(mag.group(1)), float(mag.group(2))], "sample_type": ["log", "pos_neg"]}
        b = _bounds(text) or ((float(lo), float(hi)) if lo is not None and hi is not None else None)
        if not b or min(b) <= 0:
            raise Unsupported(f"log-uniform without positive bounds: {text[:80]}")
        return {"sample_range": [float(b[0]), float(b[1])], "sample_type": ["log", "pos"]}
    if re.search(r"x_min|x_max|\bdrawn per variable\b", t):
        raise Unsupported(f"a per-problem random box: {text[:80]}")
    if lo is not None and hi is not None:
        # "t ~ U[0, h]" names a range set by another input; "uniform U[a, b] ..., i.e. f ~ ..." only restates the law
        if re.search(r"\b[a-z]\s*~", t) and "u[" not in t[:6] and not t.startswith("uniform u["):
            raise Unsupported(f"conditional range: {text[:80]}")
        return {"sample_range": [float(lo), float(hi)], "sample_type": ["uni", "pos"]}
    raise Unsupported(f"no range: {text[:80]}")


# ------------------------------------------------------------------------------------------------------- constants
def _seed(*parts: str) -> int:
    return int(hashlib.sha256(":".join(("srbf-constant",) + parts).encode()).hexdigest()[:16], 16)


def baked_constants(p: dict[str, Any]) -> tuple[dict[str, float], dict[str, Any]]:
    """The constants the authors draw from a distribution, drawn once for this problem (seeded by its identity), and
    a record of each draw. Constants given as numbers or exact expressions are in the formula already."""
    values, record = {}, {}
    for name, spec in sorted((p.get("constants") or {}).items()):
        text = spec.get("distribution") if isinstance(spec, dict) else spec
        if not isinstance(text, str):
            continue
        m = re.match(rf"^\s*(?:uniform\s+)?U[\[(]\s*({_NUM})\s*,\s*({_NUM})\s*[\])]", text.strip(), re.I)
        if not m:
            continue                                  # a description of a literal the formula holds (pi, 1/3, ...)
        seed = _seed(p["set"], p["id"], name)
        lo, hi = float(m.group(1)), float(m.group(2))
        values[name] = float(np.random.default_rng(seed).uniform(lo, hi))
        record[name] = {"distribution": text.strip(), "seed": seed, "value": values[name]}
    return values, record


# --------------------------------------------------------------------------------------------------------- formula
def _rewrite(e: sympy.Expr) -> sympy.Expr:
    """Forms SimpliPy has no operator for, written exactly with ones it has."""
    if e.has(sympy.Piecewise):
        e = piecewise.rewrite(e)
    if e.has(sympy.Sum):
        e = e.doit()
    e = e.replace(sympy.cot, lambda u: 1 / sympy.tan(u))
    e = e.replace(sympy.acot, lambda u: sympy.atan(1 / u))
    return e


def named_constants(p: dict[str, Any]) -> dict[str, float]:
    """Constants the formula names by a symbol and the record gives a number for (G = 1, alpha = 0.00755, ...)."""
    out = {}
    for name, value in (p.get("constants") or {}).items():
        if isinstance(value, bool) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", str(name)):
            continue
        if isinstance(value, (int, float)):
            out[name] = float(value)
        elif isinstance(value, str) and re.fullmatch(r"[-+0-9.eE/* ()pi]+", value.strip()):
            try:
                out[name] = float(sympy.sympify(value))
            except (sympy.SympifyError, TypeError):
                continue
    return out


def prepared_of(formula: str, names: list[str], constants: dict[str, float]) -> str:
    """The formula over v1..vN with the baked constants, in SimpliPy's infix. Parsed WITHOUT evaluation, so the
    authors' structure reaches SimpliPy as written (its simplification is the judge's, as for the release catalogs)."""
    def read(real: bool) -> sympy.Expr:
        local = {n: sympy.Symbol(n, real=True) if real and n in names else sympy.Symbol(n) for n in [*names, *constants]}
        e = parse_expr(formula, local_dict=local, evaluate=False)
        if constants:
            e = e.xreplace({sympy.Symbol(k): sympy.Float(repr(v)) for k, v in constants.items()})
        free = {s.name for s in e.free_symbols} - set(names)
        if free:
            raise Unsupported(f"free symbols {sorted(free)} in {formula[:80]}")
        return _rewrite(e)

    def over_v(real: bool) -> sympy.Expr:
        e = read(real)
        if e.has(sympy.hyper):
            raise Unsupported("the formula needs the hypergeometric function, which SimpliPy does not have")
        return e.xreplace({sym: sympy.Symbol(f"v{names.index(sym.name) + 1}", **({"real": True} if real else {}))
                           for sym in e.free_symbols})
    e = over_v(real=False)
    if e.has(sympy.re, sympy.im, sympy.I):
        # sympy takes a symbol to be complex and rewrites |exp(x)| as exp(re(x)): the inputs are real, so read them so
        e = over_v(real=True)
    s = sympy.sstr(e, full_prec=False)
    s = re.sub(r"\bAbs\(", "abs(", s)
    s = re.sub(r"\bpi\b", repr(math.pi), s)
    s = re.sub(r"\bE\b", repr(math.e), s)
    return s


# ---------------------------------------------------------------------------------------- pinned points: data
_GRID_N = re.compile(r"(?:N\s*=\s*(\d+)\s*points|(\d+)\s*(?:equally spaced|evenly spaced|equidistant)\s*"
                     r"(?:values|points|base points))")
_TEST_GRID = re.compile(rf"test inputs E\[\s*({_NUM})\s*,\s*({_NUM})\s*\],\s*(\d+) evenly spaced points")
# RSRM (Xu, Liu & Sun, ICLR 2024 App. F; srbf ruling 2026-11-03): everything RSRM saw, x = 0..100 (its training [30, 80] and
# its validation [0, 100]; arXiv Sec. 4.3: the fit range is [0, 100]), is the fit; the extrapolation x = -100..-1 is
# the test. The appendix's test [10, 25] lies inside the fit and drops.
SPLIT_RANGES = {("fresh-111", f"Curve-{k}"): ((0.0, 100.0), (-100.0, -1.0)) for k in (1, 2, 3)}
# Tests the authors state without their points, drawn by srbf; their catalog is marked '*' (srbf ruling 2026-11-04, the rule
# given for EQL's arm). BMS (Guimera et al. 2020, SM Fig. S6H) fits the Bessel problem on [0, 2] and measures its error
# "in the range [0, 4]": srbf draws (low, high, per order) = 100 points per order nu uniformly on [0, 4], once, seeded.
# The input order is (x, nu), as recorded.
SRBF_TESTS = {("fresh-64", "bessel-J0-4-xmax2"): (0.0, 4.0, 100)}
MODIFIED_POINTS = {"fresh-64": "the authors measure the Bessel problem's error on [0, 4] without stating its points: srbf "
                               "tests on 100 points per order nu drawn uniformly on [0, 4], once, seeded",
                   "fresh-145": "the authors test F1-F3 on 200 random points without stating their law: srbf draws them "
                                "uniformly on the training range, once, seeded"}
# Vladislavleva et al. 2009 Table I: the authors' test mesh per input, (start:step:stop); the GP-benchmarks table
# writes it "testing 'x: E[start, stop, step]'"
_TEST_MESH = re.compile(rf"test mesh \w+\s*=\s*\(({_NUM}):({_NUM}):({_NUM})\)")
_TESTING_E = re.compile(rf"testing '\w+:\s*E\[\s*({_NUM})\s*,\s*({_NUM})\s*,\s*({_NUM})\s*\]'")
# McDermott et al. 2012 (DyFor): the series is fitted on its first 14 values and forecast on the rest
SPLIT_COUNT = {("fresh-48", "dyfor-simulated"): 14}
# Tests the authors run on random points without stating their law: srbf draws them uniformly on the training range
# (marked '*'). Schmidt & Lipson's coevolved fitness predictors test F1-F3 on 200 additional random points.
SRBF_RANDOM_TESTS = {("fresh-145", f): 200 for f in ("F1", "F2", "F3")}


def grid_values(v: dict[str, Any]) -> np.ndarray:
    """The values a pinned input takes: N points over [low, high], or low, low + step, ..., high."""
    text, lo, hi, step = str(v.get("distribution") or ""), v.get("low"), v.get("high"), v.get("step")
    values = re.search(r"\{([^}]*)\}", text)
    if values and "..." not in values.group(1):
        return np.array([float(sympy.sympify(x.strip())) for x in values.group(1).split(",") if x.strip()])
    n = _GRID_N.search(text)
    if n and lo is not None and hi is not None:
        return np.linspace(float(lo), float(hi), int(n.group(1) or n.group(2)))
    if step and lo is not None and hi is not None:
        return np.round(np.arange(float(lo), float(hi) + float(step) / 2, float(step)), 12)
    raise Unsupported(f"grid without its points: {text[:80]}")


def _numeric(formula: str, names: list[str], constants: dict[str, float]):
    """The formula as a numpy function of its inputs (harmonic numbers through the digamma function)."""
    import scipy.special
    local = {n: sympy.Symbol(n) for n in [*names, *constants]}
    e = parse_expr(formula, local_dict=local).subs({sympy.Symbol(k): v for k, v in constants.items()})
    e = e.doit() if e.has(sympy.Sum) else e
    harmonic = {"harmonic": lambda n: scipy.special.digamma(np.asarray(n, float) + 1) + np.euler_gamma}
    return sympy.lambdify([sympy.Symbol(n) for n in names], e, modules=[harmonic, "scipy", "numpy"])


def points_problem(p: dict[str, Any], slug: str, display: str, engine) -> Any:
    """A formula problem whose points the authors pin: the formula evaluated on them, a data problem
    (ruling 9(2)). Fit on the pinned grid; a test only where the authors give one (Keijzer-6 and Neat-6: 1..120, the
    release fix; Vladislavleva-2/3: their test meshes; a test grid next to random training draws), none otherwise
    (srbf ruling 2026-11-03, call (c))."""
    from symbolic_data import data_problem
    from symbolic_data._evaluation import compile_expression
    names = [v["name"] for v in p["variables"]]
    constants = {**named_constants(p), **baked_constants(p)[0]}
    f = _numeric(p["formula"], names, constants)
    split, x_test, note = "none", None, "none: the authors pin the points and give no test"
    if p["id"] in ("Keijzer-6", "Neat-6"):
        x = np.arange(1.0, 51.0)[:, None]
        x_test, split, note = np.arange(1.0, 121.0)[:, None], "given", "given: fit on x = 1..50, test on x = 1..120"
    elif all(_TEST_GRID.search(str(v.get("distribution"))) for v in p["variables"]):
        # random training draws and a test grid per input; with several inputs the grids are paired, point i of each
        # (SPL's make_datasets.py: np.linspace per variable)
        rng = np.random.default_rng(_seed(p["set"], p["id"], "fit-pool"))
        x = np.column_stack([rng.uniform(float(v["low"]), float(v["high"]), 10000) for v in p["variables"]])
        grids = [_TEST_GRID.search(str(v.get("distribution"))) for v in p["variables"]]
        x_test = np.column_stack([np.linspace(float(m.group(1)), float(m.group(2)), int(m.group(3))) for m in grids])
        split, note = "given", "given: the authors' test grid; fit rows from the training range (a seeded pool of 10,000)"
    elif (p["set"], p["id"]) in SPLIT_RANGES:
        (fa, fb), (ta, tb) = SPLIT_RANGES[(p["set"], p["id"])]
        grid = grid_values(p["variables"][0])[:, None]
        x, x_test = grid[(grid[:, 0] >= fa) & (grid[:, 0] <= fb)], grid[(grid[:, 0] >= ta) & (grid[:, 0] <= tb)]
        split, note = "given", f"given: fit on the authors' grid points in [{fa:g}, {fb:g}], test on [{ta:g}, {tb:g}]"
    else:
        axes = [grid_values(v) for v in p["variables"]]
        x = np.array(np.meshgrid(*axes, indexing="ij")).reshape(len(axes), -1).T
        meshes = [_TEST_MESH.search(str(v.get("notes") or "")) for v in p["variables"]]
        steps = [(float(m.group(1)), float(m.group(2)), float(m.group(3))) for m in meshes] if all(meshes) else None
        e_meshes = [_TESTING_E.search(str(v.get("notes") or "")) for v in p["variables"]]
        if steps is None and all(e_meshes):                # E[start, stop, step]
            steps = [(float(m.group(1)), float(m.group(3)), float(m.group(2))) for m in e_meshes]
        if steps:                                          # the authors' test mesh (Vladislavleva-2/3)
            axes = [np.round(np.arange(a, b + st / 2, st), 12) for a, st, b in steps]
            x_test = np.array(np.meshgrid(*axes, indexing="ij")).reshape(len(axes), -1).T
            split, note = "given", "given: fit on the authors' training mesh, test on their test mesh"
        if (p["set"], p["id"]) in SPLIT_COUNT:
            k = SPLIT_COUNT[(p["set"], p["id"])]
            x, x_test = x[:k], x[k:]
            split, note = "given", f"given: the authors fit the first {k} values and forecast the rest"
        if (p["set"], p["id"]) in SRBF_RANDOM_TESTS:
            per = SRBF_RANDOM_TESTS[(p["set"], p["id"])]
            rng = np.random.default_rng(_seed(p["set"], p["id"], "test-pool"))
            x_test = np.column_stack([rng.uniform(x[:, i].min(), x[:, i].max(), per) for i in range(x.shape[1])])
            split, note = "given", (f"srbf: the authors test on {per} random points without stating their law; "
                                    "uniform on the training range, seeded")
    if (p["set"], p["id"]) in SRBF_TESTS:
        lo, hi, per = SRBF_TESTS[(p["set"], p["id"])]
        orders = grid_values(p["variables"][1])
        rng = np.random.default_rng(_seed(p["set"], p["id"], "test-pool"))
        x_test = np.column_stack([rng.uniform(lo, hi, per * len(orders)), np.repeat(orders, per)])
        split, note = "given", (f"srbf: the authors measure the error on [{lo:g}, {hi:g}] without stating its points; "
                                f"{per} points per order, uniform, seeded")
    y = np.asarray(f(*x.T), dtype=float).reshape(-1) * np.ones(len(x))
    yt = None if x_test is None else np.asarray(f(*x_test.T), dtype=float).reshape(-1) * np.ones(len(x_test))
    gt: dict[str, Any] = {"gt_kind": "exact"}
    meta = paths.public({"catalog": slug, "catalog_display": display, "subset": p["unit"].split(":", 1)[1],
                         "source_id": p["id"], "formula": p["formula"],
                         "input_distributions": {v["name"]: v.get("distribution") for v in p["variables"]},
                         "split_reading": note})
    try:
        compiled = compile_expression(engine, p["id"], prepared_of(p["formula"], names, constants),
                                      {f"v{i + 1}": {} for i in range(len(names))}, name=slug)
        gt.update(expression=compiled["expression"], skeleton=compiled["prefix"], constants=compiled["constants"])
    except Exception as exc:                     # an operator the release SimpliPy lacks (harmonic, erf, ...)
        gt = {"gt_kind": "none"}
        meta["formula_pending"] = {"formula": p["formula"], "error": f"{type(exc).__name__}: {exc}"[:200]}
    keep = np.isfinite(y)
    if not keep.all():                     # points where the formula is undefined: not a row of the problem
        meta["undefined_points_left_out"] = int((~keep).sum())
    test = {"x_test": x_test[np.isfinite(yt)], "y_test": yt[np.isfinite(yt)]} if split == "given" else {}
    return data_problem(x[keep], y[keep], variables=[f"v{i + 1}" for i in range(len(names))], test_split=split,
                        eq_id=p["id"], meta=meta, **test, **gt)


# ---------------------------------------------------------------------------------------------------------- build
def slug_of(name: str) -> str:
    s = re.sub(r"\([^)]*\)", "", name.lower())
    return re.sub(r"[^a-z0-9]+", "-", s).strip("-")[:60]


_DERIVED = re.compile(r"derived:\s*([A-Za-z_]\w*)\s*=\s*(.+?)\s+with\s+(.+?)(?:\s*\(|,\s*i\.e\.|$)")
# "integer uniform on {n_1+1, ..., 7}, conditional on n_1" (PySR EmpiricalBench's Rydberg formula)
_INT_AFTER = re.compile(r"integer uniform on \{\s*(\w+)\s*\+\s*(\d+)\s*,\s*\.\.\.\s*,\s*(\d+)\s*\},\s*conditional on (\w+)")
_LATENT = re.compile(rf"([A-Za-z_]\w*)\s*~\s*U[\[(]\s*({_NUM})\s*,\s*({_NUM})\s*[\])]")


def _uniform(lo: float, hi: float) -> dict[str, Any]:
    return {"sample_range": [float(lo), float(hi)], "sample_type": ["uni", "pos"]}


def joint_sampler_inputs(p: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """(vars, latent) of a problem whose authors draw a few base quantities and derive its inputs from them (SciMED's
    settling sphere: rhoa, nu, Re and rhop - rhoa uniform; d, V, ... derived): each input is its formula in the base
    quantities, the derived ones substituted."""
    js = p["joint_sampler"]
    base = {name: sympy.Symbol(name) for name in js["base"]}
    latent = {}
    for name, text in js["base"].items():
        b = _bounds(text)
        if not b:
            raise Unsupported(f"joint sampler base without a range: {name}: {text[:60]}")
        latent[name] = _uniform(*b)
    derived: dict[str, sympy.Expr] = {}
    for name, expr in js.get("derived", {}).items():          # in the authors' order: each may use the ones before
        derived[name] = sympy.sympify(expr, locals={**base, **{k: sympy.Symbol(k) for k in derived}}).xreplace(
            {sympy.Symbol(k): v for k, v in derived.items()})
    variables = {}
    for i, v in enumerate(p["variables"]):
        if v["name"] in base:
            e: sympy.Expr = base[v["name"]]
        elif v["name"] in derived:
            e = derived[v["name"]]
        else:
            e = sympy.sympify(v.get("derived_from") or "", locals={**base, **{k: sympy.Symbol(k) for k in derived}})
            e = e.xreplace({sympy.Symbol(k): d for k, d in derived.items()})
        free = {x.name for x in e.free_symbols} - set(base)
        if free:
            raise Unsupported(f"joint sampler: {v['name']} needs {sorted(free)}")
        variables[f"v{i + 1}"] = {"name": v["name"], "expr": re.sub(r"\bAbs\(", "abs(", sympy.sstr(e))}
    return variables, latent


def row_constraints(p: dict[str, Any]) -> list[str]:
    """The authors' constraints on a row (GeoBench: the triangle inequality, angles summing below pi), over v1..vN."""
    names = [v["name"] for v in p["variables"]]
    pattern = re.compile(r"\b(" + "|".join(map(re.escape, sorted(names, key=len, reverse=True))) + r")\b")
    return [re.sub(r"\bpi\b", repr(math.pi), pattern.sub(lambda m: f"v{names.index(m.group(1)) + 1}", c))
            for c in p.get("constraints") or []]


def inputs_of(p: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], dict[str, Any]]:
    """(vars, latent, baked): each input's spec. A push-forward ("derived: var1 = x2/x1 with x1 ~ U[1,5], ...") draws
    its latent variables and derives the input; an input conditional on another ("t | h ~ U[0, sqrt(2*h/9.81)]",
    "U[0, R]") draws a latent U[0,1] and scales it to the range the other input sets, which is drawn as a latent
    variable too; a box the authors draw per problem and variable is drawn once, seeded."""
    variables: dict[str, dict[str, Any]] = {}
    latent: dict[str, dict[str, Any]] = {}
    baked: dict[str, Any] = {}
    by_name = {v["name"]: v for v in p["variables"]}
    conditioned = {c for v in p["variables"]
                   for c in (v.get("conditional_on") or ([v["dependent_on"]] if v.get("dependent_on") else []))}
    conditioned |= {m.group(4) for v in p["variables"] if (m := _INT_AFTER.search(str(v.get("distribution") or "")))}
    phye2e = None
    for i, v in enumerate(p["variables"]):
        key, text = f"v{i + 1}", str(v.get("distribution") or "")
        if text.startswith("mixture, branch drawn once per problem: p=0.5 |N(2, variance 3)|"):
            # PhyE2E: one branch per problem for all its inputs, |N(2, 3)| or U[2 - sqrt 3, 2 + sqrt 3] (drawn once, seeded)
            if phye2e is None:
                folded = np.random.default_rng(_seed(p["set"], p["id"], "branch")).random() < 0.5
                phye2e = ({"dist": {"name": "folded_normal", "kwargs": {"loc": 2.0, "scale": math.sqrt(3.0)}}} if folded
                          else _uniform(2.0 - math.sqrt(3.0), 2.0 + math.sqrt(3.0)))
                baked["branch"] = {"distribution": text[:120], "drawn": "folded normal" if folded else "uniform"}
            variables[key] = {"name": v["name"], **phye2e}
            continue
        d = _DERIVED.search(text)
        if d:
            for name, lo, hi in _LATENT.findall(d.group(3)):
                latent.setdefault(name, _uniform(float(lo), float(hi)))
            variables[key] = {"name": v["name"], "expr": d.group(2).strip()}
            continue
        after = _INT_AFTER.search(text)
        if after:                                     # uniform on {a + k, ..., top}: a, plus k, plus floor(u * count)
            a, k, top = after.group(4), int(after.group(2)), int(after.group(3))
            latent.setdefault(a, input_spec(by_name[a]))
            latent[f"u_{v['name']}"] = _uniform(0.0, 1.0)
            variables[key] = {"name": v["name"], "expr": f"{a} + {k} + floor(u_{v['name']} * ({top} - {a} - {k} + 1))"}
            continue
        cond = v.get("conditional_on") or ([v["dependent_on"]] if v.get("dependent_on") else [])
        if cond:
            high = v.get("high_expr") or cond[0]
            low = v.get("low") or 0.0
            for c in cond:
                latent.setdefault(c, input_spec(by_name[c]))
            latent[f"u_{v['name']}"] = _uniform(0.0, 1.0)
            variables[key] = {"name": v["name"], "expr": f"{low} + u_{v['name']} * (({high}) - {low})"}
            continue
        if v["name"] in conditioned:                  # set by its latent twin, which the conditional input reads
            latent.setdefault(v["name"], input_spec(v))
            variables[key] = {"name": v["name"], "expr": v["name"]}
            continue
        if re.search(r"x_min_j ~ U\(-10, 9\) and x_max_j ~ U\(x_min_j \+ 1, 10\)", text):
            rng = np.random.default_rng(_seed(p["set"], p["id"], "box", v["name"]))
            lo = float(rng.uniform(-10.0, 9.0))
            hi = float(rng.uniform(lo + 1.0, 10.0))
            variables[key] = {"name": v["name"], **_uniform(lo, hi)}
            baked[v["name"]] = {"distribution": text, "drawn": [lo, hi]}
            continue
        variables[key] = {"name": v["name"], **input_spec(v)}
    return variables, latent, baked


_E2E = re.compile(r"^E2E mixture ([01]) k=(\d+)$")


def joint_inputs_of(p: dict[str, Any]) -> dict[str, Any] | None:
    """A joint input sampler for the problem, where the authors' inputs are not a product over variables: E2E's
    mixture (shape 0 Gaussian / 1 uniform and k clusters, fixed per function; the clusters drawn per point set)."""
    found = {_E2E.match(str(v.get("distribution") or "")) for v in p["variables"]}
    if found and None not in found:
        (m,) = {(x.group(1), x.group(2)) for x in found}
        return {"name": "e2e_mixture", "kwargs": {"shape": "gaussian" if m[0] == "0" else "uniform", "k": int(m[1])}}
    return None


def distinct_names(variables: dict[str, dict[str, Any]]) -> None:
    """An input that repeats an earlier input's name gets the next free capital letter (LaSR names its added unused
    inputs A, B, C, ..., which can repeat a formula input such as B or F; the formula reads the first one)."""
    used = {v["name"] for v in variables.values()}
    seen: set[str] = set()
    for key in sorted(variables, key=lambda k: int(k[1:])):
        name = variables[key]["name"]
        if name in seen:
            new = next(c for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" if c not in used)
            used.add(new)
            variables[key]["name"] = new
        seen.add(variables[key]["name"])


def entry_of(p: dict[str, Any]) -> dict[str, Any]:
    names = [v["name"] for v in p["variables"]]
    if not names:
        raise Unsupported("no inputs")
    if not p.get("formula"):
        raise Unsupported("no formula")
    constants, record = baked_constants(p)
    constants = {**named_constants(p), **constants}
    joint = joint_inputs_of(p)
    if joint:
        variables, latent, boxes = {f"v{i + 1}": {"name": v["name"]} for i, v in enumerate(p["variables"])}, {}, {}
    elif p["set"] == ARM:
        (variables, latent), boxes = arm_inputs(p), {}
    elif (p["set"], p["id"]) in SRBF_INPUTS:
        spec = SRBF_INPUTS[(p["set"], p["id"])]
        variables, latent, boxes = {f"v{i + 1}": {"name": v["name"], **spec[v["name"]]} for i, v in enumerate(p["variables"])}, {}, {}
    elif isinstance(p.get("joint_sampler"), dict) and p["joint_sampler"].get("base"):
        (variables, latent), boxes = joint_sampler_inputs(p), {}
    else:
        variables, latent, boxes = inputs_of(p)
    distinct_names(variables)
    variables["v0"] = {"name": "y"}
    # the target the authors fit, where it is not the published law itself (PySR EmpiricalBench fits ln y)
    target = p.get("target_as_evaluated") or p["formula"]
    entry = {"raw": p["formula"], "prepared": prepared_of(target, names, constants), "n_variables": len(names),
             "subset": p["unit"].split(":", 1)[1], "vars": variables,
             "input_distributions": {v["name"]: v.get("distribution") for v in p["variables"]}}
    if target != p["formula"]:
        entry["target"] = f"the authors fit {target}, not the published law itself"
    if p.get("constraints"):
        entry["row_constraints"] = row_constraints(p)
    if joint:
        entry["inputs"] = joint
        if joint["name"] == "e2e_mixture":
            entry["test"] = {}     # E2E's test points: a fresh instance of the mixture (Kamienny et al. 2022, Sec. 3.1)
    if latent:
        entry["latent"] = latent
    if record:
        entry["constants_drawn"] = record
    if boxes:
        entry["input_ranges_drawn"] = boxes
    return entry


N_MC = 20_000   # the release catalogs' audit size (tools/audit_finite_fraction.py)


def with_tests(p: dict[str, Any], entry: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """The problem's entries: one per authors' test (authors_tests.tests_for); the one whose test is the training
    distribution keeps the id, another test adds a suffix and its own subset."""
    names = [v["name"] for v in p["variables"]]
    rec = authors_tests.inventory().get((p["set"], p["id"]))
    out = {}
    for suffix, subset, block in authors_tests.tests_for(p["set"], p["id"], names):
        e = dict(entry)
        if subset:
            e["subset"] = subset if e["subset"] == "all" else f"{e['subset']}-{subset}"
        if block is not None:
            e["test"] = block
            e["test_source"] = paths.public(rec["source"])[:400]
        out[p["id"] + suffix] = e
    return out


def finite_fraction(cat, entry, engine) -> float:
    """The share of rows drawn from the entry's declared inputs on which the formula is finite (and the row meets
    the entry's constraints): how far the accepted rows' law sits from the declared one."""
    import warnings
    from symbolic_data._evaluation import evaluate
    from symbolic_data.inputs import constraint_mask, draw_rows
    compiled = cat._compiled(entry, engine)
    order = compiled["variable_order"]
    rng = np.random.default_rng(20260710)                 # the audit's seed; never used for benchmark draws
    x = draw_rows(entry.variables, entry.meta, order, N_MC, rng)
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        y = np.asarray(evaluate(compiled, {k: x[:, i] for i, k in enumerate(order)}), dtype=float)
    y = np.broadcast_to(y.reshape(-1) if y.size == N_MC else y, (N_MC,))
    return float((np.isfinite(x).all(axis=1) & np.isfinite(y) & constraint_mask(entry.meta, x, order)).mean())


def too_long(engine, prepared: str) -> None:
    """Raise when the judge's parser refuses the ground truth for its length (SimpliPy caps a prefix at 4,096 tokens):
    such a formula cannot be read back by srbf. An operator the release SimpliPy lacks is left to the compile check."""
    try:
        engine.read_infix(prepared)
    except ValueError as exc:
        if "too long" in str(exc):
            raise ValueError(f"the judge's parser refuses it: {exc}") from exc


def real_anywhere(entry: dict[str, Any], n: int = 4096) -> bool:
    """Whether the published formula takes a real, finite value anywhere on its inputs' ranges (numpy, no complex
    numbers): a formula such as sqrt(-0.2 * |x|) has none, and sympy reads it with the imaginary unit."""
    names = [k for k in entry["vars"] if k != "v0"]             # the prepared formula: v1..vN, constants as numbers
    rng = np.random.default_rng(0)
    cols = []
    for k, v in entry["vars"].items():
        if k == "v0":
            continue
        lo, hi = (v.get("sample_range") or [-1.0, 1.0])[:2]
        cols.append(rng.uniform(float(lo), float(hi), n))
    with np.errstate(all="ignore"):
        f = sympy.lambdify([sympy.Symbol(x) for x in names],
                           parse_expr(entry["prepared"], local_dict={x: sympy.Symbol(x) for x in names}), modules="numpy")
        y = np.asarray(f(*cols), dtype=complex).reshape(-1)
    return bool((np.isfinite(y) & (y.imag == 0)).any())


def check(entries: dict[str, dict[str, Any]], n: int, engine) -> dict[str, str]:
    """Compile and draw ``n`` rows of every entry, and record its valid-domain fraction; the failures by id."""
    from symbolic_data.catalog import ProblemCatalog
    cat = ProblemCatalog.from_yaml({"metadata": {"name": "check", "version": 1}, "expressions": entries})
    failed = {}
    rng = np.random.default_rng(0)
    for eq_id, entry in cat.entries.items():
        try:
            too_long(engine, entries[eq_id]["prepared"])
            cat.realize(entry, n, rng, engine=engine)
            if "test" in entry.meta:
                try:
                    cat.realize(entry, n, rng, engine=engine, part="test")
                except Exception as exc:   # the authors' test region has no valid point: keep the problem, test in range
                    del entries[eq_id]["test"]
                    entries[eq_id]["authors_test_unrealizable"] = f"{type(exc).__name__}: {exc}"[:200]
            f = round(finite_fraction(cat, entry, engine), 4)
            entries[eq_id].setdefault("meta", {})["finite_fraction"] = f
            if f < 0.05:
                entries[eq_id]["meta"]["low_validity"] = True
        except Exception as exc:
            failed[eq_id] = f"{type(exc).__name__}: {exc}"[:200]
            if ("name 'I' is not defined" in str(exc) or "I" in re.findall(r"\b\w+\b", entries[eq_id]["prepared"])) \
                    and not real_anywhere(entries[eq_id]):
                failed[eq_id] = "no real value: the formula is complex everywhere on the authors' input range"
    return failed


# The release members whose authors train on pinned points (source audit: 20 problems with exact training points; srbf
# 2026-11-03: they move to companion '-points' catalogs, as Keijzer-6 did). Formula as in the release catalog; inputs
# as (start, stop, step) grids, ("linspace", a, b, n), "uniform" (a seeded pool of 10,000 on the training range) or
# "train" (the test is the training points). Keijzer 2003 Table 3; Krawiec & Pawlak 2013 Table 2 (Nonic, R1-R3:
# 20 equidistant points on [-1, 1], tested on random points); Mundhenk et al. 2021 (R1a-R3a: E(-10, 10, 20), test =
# train). Pagie & Hogeweg 1997: Pagie-1 on the 26 x 26 grid E[-5, 5, 0.4] per input, no test stated (their 100 x 100 test
# grid has no stated step); Kusner et al. 2017: GrammarVAE-1 on 1,000 evenly spaced points of [-10, 10], no test. A
# grid with several inputs is given per input and meshed. Const-Test-1/2 (DSO repository sanity tests) are dropped, not
# moved.
_K1 = "0.3*v1*sin(2*3.141592653589793*v1)"
GRAMMARVAE_1 = "1./3+v1+sin(((v1)**(2)))"
_R = {"R1": "((((v1+1)**(3)))/(((v1)**(2))-v1+1))",
      "R2": "(((((v1)**(5))-3*((v1)**(3))+1))/((((v1)**(2))+1)))",
      "R3": "(((((v1)**(6))+((v1)**(5))))/((((v1)**(4))+((v1)**(3))+((v1)**(2))+v1+1)))"}
RELEASE_POINTS = {
    ("keijzer", "Keijzer-1"): (_K1, (-1, 1, 0.1), (-1, 1, 0.001)),
    ("keijzer", "Keijzer-2"): (_K1, (-2, 2, 0.1), (-2, 2, 0.001)),
    ("keijzer", "Keijzer-3"): (_K1, (-3, 3, 0.1), (-3, 3, 0.001)),
    ("keijzer", "Keijzer-4"): ("((v1)**(3))*exp(-v1)*cos(v1)*sin(v1)*(((sin(v1))**(2))*cos(v1)-1)", (0, 10, 0.05), (0.05, 10.05, 0.05)),
    ("keijzer", "Keijzer-7"): ("log(v1)", (1, 100, 1), (1, 100, 0.1)),
    ("keijzer", "Keijzer-8"): ("sqrt(v1)", (0, 100, 1), (0, 100, 0.1)),
    ("keijzer", "Keijzer-9"): ("log(v1+sqrt(((v1)**(2))+1))", (0, 100, 1), (0, 100, 0.1)),
    ("nonic", "Nonic"): ("+".join(f"((v1)**({k}))" for k in range(9, 1, -1)) + "+v1", ("linspace", -1, 1, 20), "uniform"),
    **{("r-rationals", r): (f, ("linspace", -1, 1, 20), "uniform") for r, f in _R.items()},
    **{("r-rationals", f"{r}a"): (f, ("linspace", -10, 10, 20), "train") for r, f in _R.items()},
    ("pagie", "Pagie-1"): ("((1)/((1+((v1)**(-4)))))+((1)/((1+((v2)**(-4)))))", ((-5, 5, 0.4), (-5, 5, 0.4)), None),
    ("grammarvae", "GrammarVAE-1"): (GRAMMARVAE_1, ("linspace", -10, 10, 1000), None),
}
DROPPED = {"constant": ["Const-Test-1", "Const-Test-2"]}


def _points(spec: Any) -> np.ndarray:
    if spec[0] == "linspace":
        return np.linspace(float(spec[1]), float(spec[2]), int(spec[3]))
    a, b, step = map(float, spec)
    return np.round(a + step * np.arange(int(round((b - a) / step)) + 1), 12)


def release_points_problem(catalog: str, eq_id: str, engine) -> Any:
    """A release member whose authors pin its points, as a data problem (fit on their training points, test on theirs)."""
    from symbolic_data import data_problem
    from symbolic_data._evaluation import compile_expression, evaluate
    prepared, train, test = RELEASE_POINTS[(catalog, eq_id)]
    axes = [_points(t) for t in train] if isinstance(train[0], tuple) else [_points(train)]
    keys = [f"v{i + 1}" for i in range(len(axes))]
    compiled = compile_expression(engine, eq_id, prepared, {k: {} for k in keys}, name=f"{catalog}-points")
    x = np.array(np.meshgrid(*axes, indexing="ij")).reshape(len(axes), -1).T
    if test is None:
        x_test, note = None, "none: the authors pin the points and state no test"
    elif test == "train":
        x_test, note = x.copy(), "given: the authors test on their training points"
    elif test == "uniform":
        rng = np.random.default_rng(_seed(f"release/{catalog}", eq_id, "test-pool"))
        x_test = rng.uniform(x.min(), x.max(), (10000, 1))
        note = "given: the authors test on random points of the training range (a seeded pool of 10,000)"
    else:
        x_test, note = _points(test)[:, None], "given: the authors' test grid"
    y = np.asarray(evaluate(compiled, {k: x[:, i] for i, k in enumerate(keys)}), dtype=float).reshape(-1) * np.ones(len(x))
    meta = {"catalog": f"{catalog}-points", "catalog_display": f"{catalog} (pinned points)", "source_id": eq_id,
            "moved_from": catalog, "split_reading": note}
    keep = np.isfinite(y)
    test: dict[str, Any] = {}
    if x_test is not None:
        yt = np.asarray(evaluate(compiled, {k: x_test[:, i] for i, k in enumerate(keys)}), dtype=float).reshape(-1) * np.ones(len(x_test))
        keep_t = np.isfinite(yt)
        test = {"x_test": x_test[keep_t], "y_test": yt[keep_t]}
    return data_problem(x[keep], y[keep], variables=keys, test_split="given" if test else "none", eq_id=eq_id, meta=meta,
                        gt_kind="exact", expression=compiled["expression"], skeleton=compiled["prefix"],
                        constants=compiled["constants"], **test)


def build_family_points(problems: list[dict[str, Any]], out: Path, engine) -> dict[str, list[str]]:
    """The companion '<catalog>-points' data catalogs of the release catalogs' pinned-grid members."""
    from symbolic_data.catalog import ProblemCatalog
    by_catalog: dict[str, list[Any]] = defaultdict(list)
    for p in problems:
        catalog = FAMILY_POINTS.get((p["set"], p["id"]))
        if catalog:
            by_catalog[catalog].append(points_problem(p, f"{catalog}-points", f"{catalog} (pinned points)", engine))
    for catalog, eq_id in RELEASE_POINTS:
        by_catalog[catalog].append(release_points_problem(catalog, eq_id, engine))
    for catalog, pinned in sorted(by_catalog.items()):
        ProblemCatalog.from_problems(pinned, name=f"{catalog}-points", version=1, meta={
            "display_name": f"{catalog} (pinned points)", "kind": "data", "rules": RULES,
            "companion_of": catalog, "source_record": ", ".join(
                [f"catalog-prep {t}" for t in sorted({t for (t, i), c in FAMILY_POINTS.items() if c == catalog})]
                + (["the release catalog (RELEASE_POINTS)"] if any(c == catalog for c, _ in RELEASE_POINTS) else []))
        }).save(out / f"{catalog}-points.npz")
        print(f"{catalog}-points: {', '.join(q.eq_id for q in pinned)}", flush=True)
    return {c: [q.eq_id for q in v] for c, v in by_catalog.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(paths.CATALOGS), help="where the catalogs go (default: assets/catalogs)")
    ap.add_argument("--only", default=None)
    ap.add_argument("--check", type=int, default=64, help="rows drawn per entry to check it (0: no check)")
    ap.add_argument("--family-points", action="store_true", help="build only the release catalogs' companion -points catalogs")
    a = ap.parse_args()
    problems = [p for p in paths.final_problems()
                if p["kind"] == "F" and not p["set"].startswith("release/")]
    if a.family_points:
        from symbolic_data._evaluation import load_engine
        build_family_points(problems, Path(a.out), load_engine(None))
        return
    by_set: dict[str, list[dict]] = defaultdict(list)
    for p in problems:
        by_set[p["set"]].append(p)
    tags = [t for t in (a.only.split(",") if a.only else sorted(by_set)) if t not in FAMILY_SETS]
    engine = None
    if a.check:
        from symbolic_data._evaluation import load_engine
        engine = load_engine(None)
    out = Path(a.out)
    report_path = out / "build_formula_report.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else {}
    for tag in tags:
        rec = paths.set_record(tag)
        entries, skipped, pinned = {}, {}, []
        slug = SLUGS.get(tag) or slug_of(rec["name"])
        points_display = f"{rec['name']}{'*' if tag in MODIFIED_POINTS else ''} (pinned points)"
        for p in by_set[tag]:
            try:
                entries.update(with_tests(p, entry_of(p)))
            except authors_tests.Pending as why:
                skipped[p["id"]] = f"test: {why}"
            except Unsupported as why:
                if str(why).startswith("pinned grid") and "mapped per model" not in str(p["variables"]):
                    try:
                        pinned.append(points_problem(p, f"{slug}-points", points_display, engine))
                        continue
                    except Exception as exc:
                        why = Unsupported(f"pinned grid: {type(exc).__name__}: {exc}"[:200])
                skipped[p["id"]] = str(why)
            except Exception as exc:                      # a formula sympy cannot read, ...
                skipped[p["id"]] = f"{type(exc).__name__}: {exc}"[:200]
        if pinned:
            from symbolic_data.catalog import ProblemCatalog
            modified = {"modified_by_srbf": MODIFIED_POINTS[tag]} if tag in MODIFIED_POINTS else {}
            ProblemCatalog.from_problems(pinned, name=f"{slug}-points", version=1, meta={
                "display_name": points_display, **modified, "kind": "data", "rules": RULES,
                "source_record": f"catalog-prep/sets/{tag}.json"}).save(out / f"{slug}-points.npz")
        failed = check(entries, a.check, engine) if a.check and entries else {}
        pending = {}
        for eq_id, why in list(failed.items()):
            if "compile_failed" in why and NEEDS_015.search(entries[eq_id]["prepared"]):
                entries[eq_id]["needs_simplipy"] = "0.15"   # an operator the release SimpliPy lacks: kept, checked with 0.15
                pending[eq_id] = failed.pop(eq_id)
            else:                    # no valid point on the authors' domain, or a formula that cannot be compiled
                entries.pop(eq_id)
        if entries:
            modified = {"display_name": f"{rec['name']}*", "modified_by_srbf": MODIFIED[tag]} if tag in MODIFIED else {}
            doc = {"metadata": {"name": slug, "version": 1, "description": rec["name"], **modified, "sources": rec.get("sources"),
                                "sampling_defaults": {"n_points": 512, "method": "random", "noise": 0.0},
                                "source_kind": "set", "source_record": f"catalog-prep/sets/{tag}.json",
                                "conventions": {"rules": RULES}},
                   "expressions": entries}
            (out / f"{slug}.yaml").write_text(yaml.safe_dump(paths.public(doc), sort_keys=False, allow_unicode=True, width=120))
        report[tag] = {"catalog": slug, "problems": len(by_set[tag]), "built": len(entries), "pinned_points": len(pinned),
                       "skipped": skipped,
                       "check_failed": failed, "needs_simplipy_015": sorted(pending)}
        print(f"{tag:14s} {slug:50s} built {len(entries):5d}+{len(pinned)}/{len(by_set[tag]):5d} skipped {len(skipped):4d} "
              f"check-failed {len(failed):4d}", flush=True)
        report_path.write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
