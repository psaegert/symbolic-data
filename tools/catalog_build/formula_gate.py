"""The formula builders' numeric gate: a catalog entry's ``prepared`` formula must compute its ``raw`` formula.

A builder rewrites each formula the authors wrote (sympy spelling: named constants, two-argument logs, Piecewise,
cot, pi, ...) into the infix SimpliPy reads. A rewrite that loses a pair of parentheses poses another law without any
error (the 2026-11-28 audit found ERBench's log(u, 10) / log(w, 10) written as 0.434*log(u)/0.434*log(w), a product).
So every builder evaluates both on the same points and refuses an entry whose prepared formula differs from its raw
formula, unless the difference is expected and listed with its reason in tags/raw_vs_prepared_allowed.json (catalog,
problem ids, reason).

The points: each input drawn from its declared range where the entry states one, else from a few fixed boxes; only
points where both formulas are finite and real are compared. An entry with no such point is reported as unchecked,
never passed silently.
"""
from __future__ import annotations

import json
import re
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

import numpy as np
import sympy
from sympy.parsing.sympy_parser import parse_expr

RTOL = 1e-8          # relative agreement asked of the two formulas (both evaluated in float64)
N_POINTS = 256
BOXES = ((0.5, 2.0), (0.1, 10.0), (-2.0, 2.0), (1.0, 100.0))      # tried in turn where no range is declared
ALLOWED_FILE = Path(__file__).resolve().parent / "tags" / "raw_vs_prepared_allowed.json"


@lru_cache(maxsize=1)
def allowed() -> dict[tuple[str, str], str]:
    """(catalog, problem id) -> the reason its prepared formula may differ from its raw formula."""
    if not ALLOWED_FILE.exists():
        return {}
    out = {}
    for row in json.loads(ALLOWED_FILE.read_text()):
        for eq_id in row["ids"]:
            out[(row["catalog"], str(eq_id))] = row["reason"]
    return out


def _function(text: str, names: list[str], constants: dict[str, float] | None = None) -> Callable:
    local = {n: sympy.Symbol(n) for n in names}
    local.update({"Abs": sympy.Abs, "abs": sympy.Abs, "log": sympy.log, "floor": sympy.floor})
    e = parse_expr(text.replace("^", "**"), local_dict=local)
    if constants:
        e = e.xreplace({sympy.Symbol(k): sympy.Float(repr(v)) for k, v in constants.items()})
    free = {s.name for s in e.free_symbols} - set(names)
    if free:
        raise ValueError(f"free symbols {sorted(free)}")
    if e.has(sympy.Sum):
        e = e.doit()
    return sympy.lambdify([local[n] for n in names], e, modules=["numpy", "scipy", {"floor": np.floor}])


def _values(f: Callable, columns: list[np.ndarray]) -> np.ndarray:
    n = len(columns[0])
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        y = np.asarray(f(*columns), dtype=complex)
    y = np.broadcast_to(y.reshape(-1) if y.size == n else y, (n,))
    return np.where(y.imag == 0, y.real, np.nan)


def compare(raw: str, raw_names: list[str], prepared: str, boxes: list[tuple[float, float] | None],
            constants: dict[str, float] | None = None, seed: int = 0) -> dict[str, Any]:
    """Evaluate ``raw`` (over ``raw_names``, ``constants`` substituted) and ``prepared`` (over v1..vN, the same inputs in
    the same order) on shared points; each input is drawn from its ``boxes`` entry, or the fixed BOXES where None.
    Returns {"checked": points compared, "max_rel": the largest relative difference, "ok": agreement}."""
    keys = [f"v{i + 1}" for i in range(len(raw_names))]
    # an input that repeats an earlier name is not in the formula (the formula reads the first one): a placeholder
    unique = [n if n not in raw_names[:i] else f"_repeated_{i}" for i, n in enumerate(raw_names)]
    f_raw = _function(raw, unique, constants)
    f_prep = _function(prepared, keys)
    rng = np.random.default_rng(seed)
    best: dict[str, Any] = {"checked": 0, "max_rel": float("nan"), "ok": None}
    declared = all(b is not None for b in boxes)
    for fallback, n in ([(None, N_POINTS), (None, 16 * N_POINTS)] if declared else [(f, N_POINTS) for f in BOXES]):
        cols = [rng.uniform(*(box if box is not None else fallback), n) for box in boxes]
        y_raw, y_prep = _values(f_raw, cols), _values(f_prep, cols)
        both = np.isfinite(y_raw) & np.isfinite(y_prep)
        if both.sum() >= 8:
            scale = np.maximum(np.abs(y_raw[both]), np.abs(y_prep[both]))
            rel = np.abs(y_raw[both] - y_prep[both]) / np.where(scale > 0, scale, 1.0)
            one_sided = int((np.isfinite(y_raw) != np.isfinite(y_prep)).sum())
            return {"checked": int(both.sum()), "max_rel": float(rel.max()), "one_sided": one_sided,
                    "ok": bool(rel.max() <= RTOL)}
    return best


class GateError(ValueError):
    """A prepared formula that does not compute its raw formula, with no listed reason: the build stops."""


def gate(catalog: str, eq_id: str, raw: str, raw_names: list[str], prepared: str,
         boxes: list[tuple[float, float] | None], constants: dict[str, float] | None = None) -> dict[str, Any]:
    """``compare`` with the allow-list applied: raises GateError for a difference no listed reason covers; returns the
    comparison (with ``allowed`` set to the reason where one applies)."""
    result = compare(raw, raw_names, prepared, boxes, constants)
    reason = allowed().get((catalog, str(eq_id)))
    if reason:
        result["allowed"] = reason
    elif result["ok"] is False:
        raise GateError(f"{catalog}:{eq_id}: prepared formula differs from the raw formula (max relative difference "
                        f"{result['max_rel']:.3g} on {result['checked']} points): {prepared[:120]!r} vs {raw[:120]!r}")
    return result


def box_of(spec: dict[str, Any]) -> tuple[float, float] | None:
    """The range an input's catalog spec declares: a fastsrb ``sample_range``, or a choice/uniform/normal ``dist``."""
    if spec.get("sample_range"):
        lo, hi = (float(v) for v in spec["sample_range"][:2])
        return (lo, hi) if spec.get("sample_type", ["uni"])[1:2] != ["pos_neg"] else (-hi, hi)
    dist = spec.get("dist") or {}
    kw = dist.get("kwargs") or {}
    if dist.get("name") == "choice" and kw.get("values"):
        return (float(min(kw["values"])), float(max(kw["values"])))
    if dist.get("name") == "uniform":
        return (float(kw.get("low", 0.0)), float(kw.get("high", 1.0)))
    if dist.get("name") == "normal":
        loc, scale = float(kw.get("loc", 0.0)), float(kw.get("scale", 1.0))
        return (loc - 2 * scale, loc + 2 * scale)
    return None


def names_in(text: str, names: list[str]) -> list[str]:
    """The names of ``names`` that ``text`` uses (whole words)."""
    return [n for n in names if re.search(rf"(?<![\w.]){re.escape(n)}(?!\w)", text)]
