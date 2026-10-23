"""Data catalogs: problems that hold POOLS of rows, drawn afresh for every run.

A frozen problem built from data -- a benchmark's measured or published rows, or points its authors pinned -- holds
every row available for fitting (the ``support`` arrays) and, when the authors give one, their test rows (the
``validation`` arrays). Its ``meta["test_split"]`` says which:

* ``"given"``: the validation arrays are the authors' test rows;
* ``"none"``: the authors give no test split or procedure, so there is no test: the validation arrays are empty and
  stay empty ("if people fail to tell us what to test on, we don't", owner 2026-10-17);
* ``{"random": f}``: the authors' procedure is a random split with a test fraction ``f`` (SRBench's 75/25 is
  ``{"random": 0.25}``); the support arrays hold all rows and every draw splits them anew, ``ceil(f * N)`` test
  rows as scikit-learn's ``train_test_split`` counts them.

Every draw then takes at most ``n_support`` fitting rows and ``n_validation`` test rows (srbf: 512 each), without
replacement and in the pool's row order; a pool no larger than its cap is used whole, in its stored order. The draw
is the :class:`~symbolic_data.source.ProblemSource`'s usage policy (:func:`draw_from_pools`), so each run of each
method sees its own rows, as a formula catalog's problems are sampled afresh.

A frozen problem without ``test_split`` is a realized problem (a materialized formula catalog, a frozen evaluation
subset) and is used exactly as stored, whatever the counts.
"""
from __future__ import annotations

import dataclasses
import math
from typing import Any, Mapping, Sequence

import numpy as np

from symbolic_data.numeric import STORAGE_DTYPE
from symbolic_data.problem import Problem

__all__ = ["TEST_SPLIT_KEY", "data_problem", "draw_from_pools", "has_pools", "split_rule"]

#: The ``Problem.meta`` key of a data problem's split rule.
TEST_SPLIT_KEY = "test_split"

# the arrays of each side of a problem, in the order they are cut together
_SUPPORT = ("x_support", "y_support", "y_support_noisy", "y_reference_support", "outlier_mask_support")
_VALIDATION = ("x_validation", "y_validation", "y_validation_noisy", "y_reference_validation", "outlier_mask_validation")


def has_pools(problem: Problem) -> bool:
    """Whether ``problem`` is a data problem (it declares a split rule) rather than a realized one."""
    return TEST_SPLIT_KEY in (problem.meta or {})


def split_rule(value: Any) -> str | float:
    """The split rule of a ``meta["test_split"]`` value: ``"given"``, ``"none"`` or the random test fraction."""
    if value in ("given", "none"):
        return value
    if isinstance(value, Mapping) and set(value) == {"random"}:
        fraction = float(value["random"])
        if not 0.0 < fraction < 1.0:
            raise ValueError(f"a random test fraction must lie strictly between 0 and 1 (got {fraction})")
        return fraction
    raise ValueError(f"test_split must be 'given', 'none' or {{'random': f}} (got {value!r})")


def _rows(problem: Problem, fields: Sequence[str]) -> int:
    return int(np.asarray(getattr(problem, fields[0])).shape[0])


def _cut(problem: Problem, fields: Sequence[str], index: np.ndarray) -> dict[str, Any]:
    """The given side's arrays at ``index`` (absent optional arrays stay absent)."""
    out: dict[str, Any] = {}
    for name in fields:
        value = getattr(problem, name)
        out[name] = None if value is None else np.asarray(value)[index]
    return out


def _pooled(problem: Problem) -> dict[str, Any]:
    """Support and validation rows stacked into one pool per array (a random rule splits the pool itself)."""
    out: dict[str, Any] = {}
    for s, v in zip(_SUPPORT, _VALIDATION):
        a, b = getattr(problem, s), getattr(problem, v)
        if a is None and b is None:
            out[s] = None
            continue
        if (a is None) != (b is None):
            raise ValueError(f"{problem.eq_id}: {s} and {v} must both be present or both absent")
        out[s] = np.concatenate([np.asarray(a), np.asarray(b)], axis=0)
    return out


def _take(n_pool: int, cap: int | None, rng: np.random.Generator) -> np.ndarray:
    """Row indices of one draw: the whole pool when it fits under ``cap``, else ``cap`` rows without replacement."""
    if cap is None or n_pool <= cap:
        return np.arange(n_pool)
    return np.sort(rng.choice(n_pool, size=int(cap), replace=False))


def draw_from_pools(problem: Problem, n_support: int | None, n_validation: int | None,
                    rng: np.random.Generator) -> Problem:
    """One run's draw of a data problem: its split rule applied, then at most ``n_support`` fitting and
    ``n_validation`` test rows. A realized problem (no split rule) comes back unchanged. ``meta`` records the pool
    sizes the draw was taken from (``n_fit_pool``, ``n_test_pool``)."""
    if not has_pools(problem) or problem.is_placeholder:
        return problem
    rule = split_rule(problem.meta[TEST_SPLIT_KEY])
    if isinstance(rule, float):
        pool = _pooled(problem)
        n_pool = int(pool["x_support"].shape[0])
        order = rng.permutation(n_pool)
        n_test = int(math.ceil(rule * n_pool))
        test_rows, fit_rows = np.sort(order[:n_test]), np.sort(order[n_test:])
        fit_pick = fit_rows[_take(fit_rows.size, n_support, rng)]
        test_pick = test_rows[_take(test_rows.size, n_validation, rng)]
        arrays = {name: None if value is None else value[fit_pick] for name, value in pool.items()}
        arrays.update({v: None if pool[s] is None else pool[s][test_pick] for s, v in zip(_SUPPORT, _VALIDATION)})
        n_fit_pool, n_test_pool = int(fit_rows.size), int(test_rows.size)
    else:
        n_fit_pool = _rows(problem, _SUPPORT)
        n_test_pool = _rows(problem, _VALIDATION)
        if rule == "none" and n_test_pool:
            raise ValueError(f"{problem.eq_id}: test_split 'none' holds {n_test_pool} test rows; a problem without a "
                             "test split keeps its validation arrays empty")
        arrays = _cut(problem, _SUPPORT, _take(n_fit_pool, n_support, rng))
        arrays.update(_cut(problem, _VALIDATION, _take(n_test_pool, n_validation, rng)))
    meta = dict(problem.meta)
    meta.update(n_fit_pool=n_fit_pool, n_test_pool=n_test_pool)
    return dataclasses.replace(problem, meta=meta, **arrays)


def data_problem(x_fit: np.ndarray, y_fit: np.ndarray, *, variables: Sequence[str], test_split: str | Mapping[str, float],
                 x_test: np.ndarray | None = None, y_test: np.ndarray | None = None, eq_id: str | None = None,
                 expression: Sequence[str] | None = None, skeleton: Sequence[str] | None = None,
                 constants: Sequence[float] = (), gt_kind: str | None = None,
                 meta: Mapping[str, Any] | None = None) -> Problem:
    """A data problem: its fitting rows, the authors' test rows when ``test_split="given"``, and its ground truth
    when there is one (``gt_kind`` "exact" for pinned points of a known formula, "reference" for an accepted law
    with measured data, "none" for black-box data). The targets are used as given: no noise is added."""
    rule = split_rule(test_split)
    x_fit = np.asarray(x_fit, dtype=STORAGE_DTYPE).reshape(len(x_fit), -1)
    y_fit = np.asarray(y_fit, dtype=STORAGE_DTYPE).reshape(-1, 1)
    if x_fit.shape[0] != y_fit.shape[0] or x_fit.shape[1] != len(variables):
        raise ValueError(f"{eq_id}: x_fit {x_fit.shape} and y_fit {y_fit.shape} do not match {len(variables)} variables")
    if rule == "given":
        if x_test is None or y_test is None or len(x_test) == 0:
            raise ValueError(f"{eq_id}: test_split 'given' needs the authors' test rows")
        x_test = np.asarray(x_test, dtype=STORAGE_DTYPE).reshape(len(x_test), -1)
        y_test = np.asarray(y_test, dtype=STORAGE_DTYPE).reshape(-1, 1)
        if x_test.shape[0] != y_test.shape[0] or x_test.shape[1] != x_fit.shape[1]:
            raise ValueError(f"{eq_id}: x_test {x_test.shape} and y_test {y_test.shape} do not match x_fit {x_fit.shape}")
    else:
        if x_test is not None or y_test is not None:
            raise ValueError(f"{eq_id}: test rows are only stored for test_split 'given' (a random split cuts the "
                             "fitting rows itself; 'none' has no test)")
        x_test = np.empty((0, x_fit.shape[1]), dtype=STORAGE_DTYPE)
        y_test = np.empty((0, 1), dtype=STORAGE_DTYPE)
    merged = dict(meta or {})
    merged[TEST_SPLIT_KEY] = test_split if isinstance(test_split, str) else {"random": float(rule)}
    return Problem(
        x_support=x_fit, y_support=y_fit, y_support_noisy=y_fit.copy(),
        x_validation=x_test, y_validation=y_test, y_validation_noisy=y_test.copy(),
        skeleton=tuple(skeleton) if skeleton is not None else None,
        expression=list(expression) if expression is not None else None,
        constants=[float(c) for c in constants], variables=[str(v) for v in variables],
        complexity=len(expression) if expression is not None else None, noise=0.0, eq_id=eq_id,
        meta=merged, gt_kind=gt_kind)
