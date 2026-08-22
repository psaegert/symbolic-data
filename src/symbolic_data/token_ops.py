"""Token-level helpers for manipulating prefix expressions."""
import re
from typing import Any, Sequence

import numpy as np
from simplipy import normalize_variable_token
from simplipy.utils import is_numeric_string


def normalize_expression(tokens: Sequence[str | Any] | None) -> list[str] | None:
    """Canonical variable names, everything else verbatim -- a positional token walk.

    Until simplipy 0.13 this lived in simplipy as ``normalize_expression``; 0.14
    replaced that surface with ``to_expression``, which canonicalises through the
    engine's AC state (constants fold, an engine is required). This site's subject is
    the CONCRETE ground truth -- the recorded expression and its literal values must
    stay exactly as parsed -- so the positional walk moves here, to the consumer,
    built on the two per-token primitives simplipy keeps (``normalize_variable_token``,
    ``is_numeric_string``).
    """
    if tokens is None:
        return None
    normalized: list[str] = []
    for token in tokens:
        token_str = str(token)
        normalized_token, is_var = normalize_variable_token(token_str)
        normalized.append(normalized_token if is_var else token_str)
    return normalized


def normalize_skeleton(tokens: Sequence[str | Any] | None) -> list[str] | None:
    """The positional skeleton key: variables canonicalised, numerics masked.

    The decontamination / holdout matching key (same 0.14 provenance as
    ``normalize_expression`` above). simplipy's ``to_skeleton`` is NOT this: it
    re-runs the engine, so its skeleton depends on the loaded rule artifact --
    the wrong property for an exclusion key that must be stable across engine
    versions. Numerics are classified by the normative token grammar
    (``is_numeric_string``), never a bare ``float()`` probe: the reserved
    spellings (``inf``/``nan``/underscore groupings) pass through verbatim and can
    never compare equal to a genuinely numeric site.
    """
    if tokens is None:
        return None
    normalized: list[str] = []
    for token in tokens:
        token_str = str(token)
        normalized_token, is_var = normalize_variable_token(token_str)
        if is_var:
            normalized.append(normalized_token)
            continue
        if token_str in {"<constant>", "<c>"}:
            normalized.append("<constant>")
            continue
        if is_numeric_string(token_str):
            normalized.append("<constant>")
            continue
        normalized.append(token_str)
    return normalized


def substitute_constants(
    prefix_expression: list[str],
    values: list | np.ndarray,
    constants: list[str] | None = None,
    inplace: bool = False,
) -> list[str]:
    """Fill ``<constant>`` placeholders (and known constant names) with ``values``."""
    modified_prefix_expression = prefix_expression if inplace else prefix_expression.copy()

    constant_index = 0
    constants = [] if constants is None else list(constants)

    for i, token in enumerate(prefix_expression):
        if token == "<constant>" or re.match(r"C_\d+", token) or token in constants:
            modified_prefix_expression[i] = str(values[constant_index])
            constant_index += 1

    return modified_prefix_expression


def apply_variable_mapping(prefix_expression: list[str], variable_mapping: dict[str, str]) -> list[str]:
    """Return a new prefix expression with variables remapped via ``variable_mapping``."""
    return [variable_mapping.get(token, token) for token in prefix_expression]


def identify_constants(
    prefix_expression: list[str],
    constants: list[str] | None = None,
    inplace: bool = False,
    convert_numbers_to_constant: bool = True,
) -> tuple[list[str], list[str]]:
    """Rename ``<constant>`` tokens (optionally numeric literals) to ``C_i`` symbols."""
    modified_prefix_expression = prefix_expression if inplace else prefix_expression.copy()

    constant_index = 0
    constants = [] if constants is None else list(constants)

    for i, token in enumerate(prefix_expression):
        matches_constant = token == "<constant>" or (
            convert_numbers_to_constant and (re.match(r"C_\d+", token) or token.isnumeric())
        )
        if matches_constant:
            if len(constants) > constant_index:
                modified_prefix_expression[i] = constants[constant_index]
            else:
                modified_prefix_expression[i] = f"C_{constant_index}"
                constants.append(f"C_{constant_index}")
            constant_index += 1

    return modified_prefix_expression, constants


def flatten_nested_list(nested_list: list[Any] | Any, reverse: bool = False) -> list[str]:
    """Flatten a nested structure of lists into a flat list of tokens."""
    flat_list: list[str] = []
    stack: list[Any] = [nested_list]
    while stack:
        current = stack.pop()
        if isinstance(current, list):
            stack.extend(current)
        else:
            flat_list.append(current)
    if reverse:
        flat_list.reverse()
    return flat_list


def desugar_sqrt(tokens: list[str], operator_arity: dict[str, int]) -> list[str]:
    """Rewrite every ``sqrt`` call to the engine's spelling: ``sqrt u`` -> ``rootn u 2``.

    Curated formulas (and SymPy's printer) spell the square root as ``sqrt(...)``, but the
    engine's vocabulary has no ``sqrt`` operator -- its parser is lenient with unknown
    function names and passes the bare token through, which every downstream prefix walk
    then rejects as malformed. Tokens other than ``sqrt`` that are missing from
    ``operator_arity`` are treated as leaves, preserving the parser's lenient behavior
    (a genuinely alien token still fails downstream, with the same error as before).
    """
    out: list[str] = []

    def walk(index: int) -> int:
        token = tokens[index]
        if token == "sqrt":
            out.append("rootn")
            end = walk(index + 1)
            out.append("2")
            return end
        out.append(token)
        end = index + 1
        for _ in range(operator_arity.get(token, 0)):
            end = walk(end)
        return end

    end = walk(0)
    if end != len(tokens):
        raise ValueError(f"sqrt desugaring consumed {end} of {len(tokens)} tokens")
    return out
