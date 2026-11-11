"""The authors' test sets as symbolic-data ``test`` blocks.

srbf ruling 9(1) (2026-10-05): "validation from the authors' val spec if given, else drawn from the training spec";
9(7): several authors' tests (in range and out of range) are kept as separate subsets. The specs come from the
test-spec inventory (2026-11-03; workflow_archive/test-specs/testspecs.json): one record per problem with the
authors' test(s), each a law per input (uniform, grid, or a quoted "other" form) and the source it was read from.

A test the inventory calls the training distribution drawn again needs no block (an entry without ``test`` already
draws its test rows from its training spec). Every other test becomes a block; a problem with several tests becomes
one entry per test, and the one that matches the training distribution keeps the problem's id (its results stay
paired with earlier runs). What cannot be written as a block yet is reported, not guessed: ``Pending``.
"""
from __future__ import annotations

import sys
import math
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths  # noqa: E402  the curated inputs

# Sets whose tests wait for a ruling or for a sampler: they keep the training-spec test until then.
DEFERRED = {
    "fresh-111": "RSRM: pinned grid points, split in build_formula_catalogs.SPLIT_RANGES (srbf ruling 2026-11-03)",
    "fresh-64": "BMS Bessel: pinned points; srbf's test on [0, 4] in build_formula_catalogs.SRBF_TESTS (srbf ruling 2026-11-04)",
}

_NUM = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"


class Pending(Exception):
    """A test the inventory gives in a form this module cannot write yet."""


@lru_cache(maxsize=1)
def inventory() -> dict[tuple[str, str], dict[str, Any]]:
    return {(p["catalog"], p["id"]): p for p in paths.testspecs()}


def _uniform(lo: float, hi: float) -> dict[str, Any]:
    return {"name": "uniform", "kwargs": {"low": float(lo), "high": float(hi)}}


def _two_bands(a: tuple[float, float], b: tuple[float, float]) -> list[dict[str, Any]]:
    """Uniform on the union of two intervals: a per-value mixture weighted by length."""
    return [{**_uniform(*a), "weight": a[1] - a[0]}, {**_uniform(*b), "weight": b[1] - b[0]}]


def _grid(s: dict[str, Any]) -> dict[str, Any]:
    if s.get("values"):
        return {"name": "choice", "kwargs": {"values": [float(v) for v in s["values"]]}}
    return {"name": "grid", "kwargs": {"low": float(s["low"]), "high": float(s["high"]), "step": float(s["step"])}}


def _other(tag: str, s: dict[str, Any]) -> dict[str, Any] | list[dict[str, Any]]:
    text = s.get("text") or ""
    m = re.search(rf"U\(({_NUM}),\s*({_NUM})\) or U\(({_NUM}),\s*({_NUM})\)", text)
    if m and tag == "fresh-79":                                   # SymbolicGPT XT: each coordinate in one of two bands
        a, b, c, d = map(float, m.groups())
        return _two_bands((a, b), (c, d))
    m = re.search(rf"uniform on \[({_NUM}),\s*({_NUM})\[ U \]({_NUM}),\s*({_NUM})\]", text)
    if m:                                                         # Decomposable NSR extrapolation
        a, b, c, d = map(float, m.groups())
        return _two_bands((a, b), (c, d))
    m = re.search(rf"uniform on \]({_NUM}),\s*({_NUM})\] only", text)
    if m:
        return _uniform(*map(float, m.groups()))
    m = re.search(rf"\|v\| ~ U\[b \+ ({_NUM}), 2b\] with b = .*?\(= ({_NUM})\)", text)
    if m:                                                         # SurfaceBench OOD band, either sign
        gap, b = map(float, m.groups())
        return _two_bands((-2 * b, -(b + gap)), (b + gap, 2 * b))
    m = re.search(rf"integers with ({_NUM}) <= \|v\| <= ({_NUM})", text)
    if m:
        lo, hi = int(float(m.group(1))), int(float(m.group(2)))
        mags = list(range(lo, hi + 1))
        return {"name": "choice", "kwargs": {"values": [float(-v) for v in reversed(mags)] + [float(v) for v in mags]}}
    raise Pending(f"{tag}: unreadable test law {text[:100]!r}")


def _shell(t: dict[str, Any], keys: list[str], names: list[str]) -> dict[str, Any] | None:
    """A cube shell (EQL's extrapolation sets): [-R, R]^d without [-r, r]^d, i.e. a uniform box with max-norm > r. An
    input stored on another scale (the cart-pendulum's angle, theta = u pi/2) keeps its own box, and its share of the
    norm is divided by that scale."""
    texts = " ".join([t.get("joint_text") or ""] + [s.get("text") or "" for s in t["per_variable"].values()])
    m = re.search(rf"\[({_NUM}),\s*({_NUM})\]\^\d+ \\ \[({_NUM}),\s*({_NUM})\]\^\d+", texts)
    if m:
        outer, inner = float(m.group(2)), float(m.group(4))
        if float(m.group(1)) != -outer or float(m.group(3)) != -inner:
            raise Pending(f"an asymmetric shell {m.group(0)}")
    else:
        m = re.search(rf"on \[({_NUM}),\s*({_NUM})\]\^\d+ with max-norm > ({_NUM})", texts)
        if not m:
            return None
        outer, inner = float(m.group(2)), float(m.group(3))
    box, terms = {}, []
    for key, name in zip(keys, names):
        spec = t["per_variable"].get(name)
        hi = float(spec["high"]) if spec and spec.get("high") is not None else outer
        if spec and spec.get("low") is not None and float(spec["low"]) != -hi:
            raise Pending(f"an asymmetric box for {name}")
        box[key] = {"dist": _uniform(-hi, hi)}
        scale = hi / outer
        terms.append(f"abs({key})" if scale == 1.0 else f"abs({key}) / {scale!r}")
    norm = terms[0] if len(terms) == 1 else (f"maximum({terms[0]}, {terms[1]})" if len(terms) == 2
                                             else f"maximum.reduce([{', '.join(terms)}])")
    return {"vars": box, "row_constraints": [f"{norm} > {inner!r}"]}


def _one_out(t: dict[str, Any], keys: list[str]) -> dict[str, Any] | None:
    """EvoMO-SR's axis-mode test (Rossetti et al. 2026, generate_random_datasets.py ``sample_axis_ood``; srbf
    2026-11-04): exactly one input per row moved out of the box [-h, h]^d, to +-U(a, b) x h, the others in the box.
    Written as the box [-b h, b h]^d and two row constraints, one input beyond a h and the others inside [-h, h]. Which
    input moves is random per row, where the authors balance the inputs exactly; no '*' (the law is the authors')."""
    m = re.search(rf"centre \+/- U\(({_NUM}),\s*({_NUM})\) x half-width", t.get("joint_text") or "")
    if not m:
        return None
    a, b = map(float, m.groups())
    texts = " ".join(s.get("text") or "" for s in t["per_variable"].values())
    box = re.search(rf"otherwise LHS on \[({_NUM}),\s*({_NUM})\]", texts)
    if not box or float(box.group(1)) != -float(box.group(2)):
        raise Pending(f"an axis-mode test without a symmetric box: {texts[:100]!r}")
    h = float(box.group(2))
    out = " + ".join(f"1*(abs({k}) > {a * h!r})" for k in keys)
    inside = " + ".join(f"1*(abs({k}) <= {h!r})" for k in keys)
    return {"vars": {k: {"dist": _uniform(-b * h, b * h)} for k in keys},
            "row_constraints": [f"{out} == 1", f"{inside} == {len(keys) - 1}"]}


def _block(tag: str, t: dict[str, Any], names: list[str]) -> dict[str, Any]:
    keys = [f"v{i + 1}" for i in range(len(names))]
    if tag == "fresh-153":                  # EQL robot arm*: the extrapolation amplitude pi on srbf's arcsine joints
        return {"vars": {k: {"expr": f"{math.pi!r}*sin(u{i + 1})"} for i, k in enumerate(keys)}}
    shell = _shell(t, keys, names)
    if shell is not None:
        return shell
    one_out = _one_out(t, keys)
    if one_out is not None:
        return one_out
    out: dict[str, Any] = {}
    for key, name in zip(keys, names):
        s = t["per_variable"].get(name)
        if s is None:
            continue                                              # an input the test does not restate keeps its law
        if s["law"] == "uniform":
            out[key] = {"dist": _uniform(s["low"], s["high"])}
        elif s["law"] == "grid":
            out[key] = {"dist": _grid(s)}
        elif s["law"] == "log_uniform":
            out[key] = {"dist": {"name": "log_uniform", "kwargs": {"low": float(s["low"]), "high": float(s["high"])}}}
        else:
            out[key] = {"dist": _other(tag, s)}
    return {"vars": out}


# test labels that need a name of their own (Jin et al. 2019 Sec. 4: three test sets; EvoMO-SR's ID test)
LABELS = {"test U[-3,3] (in-range)": "in-range", "test U[-6,6]": "wide", "test U[3,6]": "beyond", "ID test": "id"}


def _slug(label: str) -> str:
    if label in LABELS:
        return LABELS[label]
    s = label.lower()
    for key, short in (("out-of-domain", "ood"), ("ood", "ood"), ("in-distribution", "id"), ("iid", "iid"),
                       ("extrapolation far", "extrap-far"), ("extrapolation near", "extrap-near"),
                       ("extrapolation", "extrap"), ("interpolation", "interp"), ("validation", "val")):
        if key in s:
            return short
    return re.sub(r"[^a-z0-9]+", "-", s).strip("-")[:24]


def _same_as_training(t: dict[str, Any]) -> bool:
    text = " ".join([t.get("label") or ""] + [s.get("text") or "" for s in t["per_variable"].values()]).lower()
    return any(k in text for k in ("same box and law as training", "as training", "in-distribution", "(in-range)",
                                   "training support", "interpolation range", "same distribution"))


def tests_for(tag: str, eq_id: str, names: list[str]) -> list[tuple[str, str | None, dict[str, Any] | None]]:
    """``[(id suffix, subset suffix, test block or None)]`` for one problem: ``[("", None, None)]`` when its test is
    its training distribution (or it has no authors' test, or the set waits: DEFERRED)."""
    rec = inventory().get((tag, eq_id))
    if rec is None or tag in DEFERRED or rec["test_kind"] in ("same_distribution", "unclear"):
        return [("", None, None)]
    if rec["test_kind"] == "pinned_points" and not rec["training_spec_same_as_record"]:
        return [("", None, None)]          # pinned training points: a data problem (srbf asked 2026-11-03)
    tests = rec["tests"]
    if rec["test_kind"] == "pinned_points" and rec["training_spec_same_as_record"] and len(tests) == 1:
        return [("", None, _block(tag, tests[0], names))]   # random training, the authors' test mesh (Keijzer-10..15)
    if rec["test_kind"] == "different_region" and len(tests) == 1:
        return [("", None, _block(tag, tests[0], names))]
    out: list[tuple[str, str | None, dict[str, Any] | None]] = []
    for t in tests:                                               # several: one subset per test (ruling 9(7))
        slug = _slug(t["label"])
        if _same_as_training(t) and not any(s == "" for s, _, _ in out):
            out.append(("", slug, None))
        else:
            out.append((f".{slug}", slug, _block(tag, t, names)))
    if len({s for s, _, _ in out}) != len(out):
        raise Pending(f"{tag}:{eq_id}: two tests share the label {[s for s, _, _ in out]}")
    return out
