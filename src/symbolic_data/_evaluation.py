"""Compile a catalog expression to a numeric callable and evaluate it (set-mode internals).

Ported from the (now-removed) spec sampler: parse the `prepared` infix over variables ``v1..vn``
with a SimpliPy engine, simplify, realize operators, codify to a lambda, and evaluate on sampled
inputs. ``v0`` carries target metadata only. These are the simplipy-dependent helpers
``ProblemSource`` uses for set-mode (X, y) generation; the X *sampling* itself is the
``fastsrb`` distribution.
"""
from __future__ import annotations

import warnings
from typing import Any, Dict, List, Mapping

import numpy as np
from simplipy import SimpliPyEngine, normalize_expression
from simplipy.utils import codify


def _is_number_token(token: Any) -> bool:
    """True iff ``token`` is a numeric literal (a concrete constant), not a variable/operator."""
    if not isinstance(token, str):
        return False
    try:
        float(token)
        return True
    except (TypeError, ValueError):
        return False


def load_engine(engine: SimpliPyEngine | str | None) -> SimpliPyEngine:
    """Resolve an engine id / None to a loaded :class:`SimpliPyEngine` (instances pass through)."""
    if isinstance(engine, SimpliPyEngine):
        return engine
    return SimpliPyEngine.load(engine or "dev_7-3", install=True)


def resolve_variable_order(vars_info: Mapping[str, Mapping[str, Any]]) -> List[str]:
    """Ordered input variables ``v1..vn`` (excludes the target ``v0``); contiguous, 1-indexed."""
    candidate_keys = [key for key in vars_info.keys() if key.startswith("v") and key != "v0"]
    if not candidate_keys:
        raise ValueError("Entry does not define any input variables")
    try:
        indices = sorted(int(key[1:]) for key in candidate_keys)
    except ValueError as exc:
        raise ValueError("Variable identifiers must follow the 'v<int>' pattern") from exc
    variable_order: List[str] = []
    for idx in range(1, indices[-1] + 1):
        key = f"v{idx}"
        if key not in vars_info:
            raise KeyError(f"Missing sampling specification for {key}")
        variable_order.append(key)
    return variable_order


def _normalize_prepared(expression: str) -> str:
    return expression.replace("^", "**")


def compile_expression(
    engine: SimpliPyEngine,
    eq_id: str,
    prepared: str,
    vars_info: Mapping[str, Mapping[str, Any]],
    *,
    name: str = "catalog",
) -> Dict[str, Any]:
    """Compile ``prepared`` to a callable over the ordered input variables.

    Returns ``{callable, variable_order, prefix, normalized_infix}``.
    """
    if not isinstance(prepared, str) or not prepared.strip():
        raise ValueError(f"Entry {eq_id} has no prepared expression")
    if not isinstance(vars_info, Mapping):
        raise ValueError(f"Entry {eq_id} has no variable definitions")

    prepared_text = _normalize_prepared(prepared)
    variable_order = resolve_variable_order(vars_info)

    prefix_parsed = engine.parse(prepared_text, mask_numbers=False)
    try:
        prefix_simplified = engine.simplify(prefix_parsed, max_pattern_length=4)
    except Exception as exc:  # pragma: no cover - defensive against SimpliPy regressions
        warnings.warn(
            f"Failed to simplify {name} expression {eq_id}: {exc}. Falling back to unsimplified prefix.",
            RuntimeWarning,
        )
        prefix_simplified = prefix_parsed

    used_variables = {tok for tok in prefix_simplified if isinstance(tok, str) and tok.startswith("v")}
    unknown = used_variables - set(variable_order) - {"v0"}
    if unknown:
        raise KeyError(f"Prepared expression for {eq_id} references undefined variables: {', '.join(sorted(unknown))}")

    # Evaluate the CONCRETE `prefix_parsed` (numeric literals intact) to produce y. Do NOT realize
    # `prefix_simplified` instead: `engine.simplify(..., max_pattern_length=4)` also constantifies
    # literals into `<constant>` placeholders (it yields the normalized SKELETON, returned as `prefix`),
    # which is not directly evaluable -- realizing it would corrupt y (turn valid entries into
    # placeholders). Three distinct, deliberate objects are returned: `prefix` = the masked SKELETON
    # (the structural / recovery form); `expression` + `constants` = the CONCRETE ground truth (the
    # actual formula with its literal values, matching the generative catalog's RealizedExpression);
    # `callable` evaluates that concrete formula.
    prefix_realized = engine.operators_to_realizations(prefix_parsed)
    code = codify(engine.prefix_to_infix(prefix_realized, realization=True), variable_order)
    return {
        "callable": engine.code_to_lambda(code),
        "variable_order": variable_order,
        "prefix": tuple(prefix_simplified),
        "normalized_infix": engine.prefix_to_infix(prefix_simplified, realization=False),
        "expression": normalize_expression(list(prefix_parsed)),
        "constants": [float(tok) for tok in prefix_parsed if _is_number_token(tok)],
    }


def evaluate(compiled: Dict[str, Any], value_map: Mapping[str, Any]) -> Any:
    """Evaluate the compiled callable on ``value_map`` (var-name -> column / scalar)."""
    ordered_inputs = [value_map[name] for name in compiled["variable_order"]]
    with np.errstate(all="ignore"):
        return compiled["callable"](*ordered_inputs)


def broadcast_target(target: Any, n_points: int, eq_id: str) -> np.ndarray:
    """Coerce an evaluated target to shape ``(n_points,)`` (handles scalars / broadcastable)."""
    target_arr = np.asarray(target, dtype=float)
    if target_arr.shape == (n_points,):
        return target_arr
    if target_arr.size == 1:
        return np.full(n_points, float(target_arr), dtype=float)
    squeezed = np.squeeze(target_arr)
    if squeezed.shape == (n_points,):
        return squeezed
    try:
        return np.broadcast_to(target_arr, (n_points,)).copy()
    except ValueError as exc:
        raise ValueError(f"Could not broadcast target values to length {n_points} for {eq_id}") from exc
