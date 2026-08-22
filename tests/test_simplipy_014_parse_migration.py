"""simplipy 0.14 removed ``SimpliPyEngine.parse``. Its replacement per the 0.14
migration table is ``read_infix`` -- the SAME raw reader under the name that states its
contract ("renamed from ``parse``", simplipy ruling 2026-08-18): TOLERANT of unknown
vocabulary and SPELLING-PRESERVING. Both call sites are MIGRATED
(``_evaluation.compile_expression`` and ``generative._sympy_simplify_skeleton``), and
these tests pin the migration's acceptance criteria GREEN over the same frozen corpora
that once recorded the sites as blocked.

THE HISTORY THIS FILE CARRIES. An earlier revision measured ``to_prefix`` as the
candidate and recorded both sites BLOCKED -- correctly, for that candidate:
``to_prefix`` reads through the canonical state, which refuses undeclared vocabulary
(``sqrt``, 46.6% of the curated corpus) and folds concrete literals (the recorded
constants -- the ground truth -- changed, and one ill-conditioned fold changed y
itself). Those measurements were sound; the candidate was wrong. The distinction
tests are KEPT below: they document why ``to_prefix`` is a canonicaliser and not a
reader, which is exactly the line 0.14 drew between the two surfaces.

The removed method was a pure delegation to the Rust raw reader
(``self._core.parse``); ``read_infix`` delegates to the same reader. The oracle tests
compare the two spellings in one process via ``_core.parse`` -- its subject IS the
removed behaviour; production code must not use it.
"""
from __future__ import annotations

import glob
import os
from typing import Any

import numpy as np
import pytest
import yaml
from simplipy import SimpliPyEngine

from symbolic_data._evaluation import compile_expression
from symbolic_data.token_ops import desugar_sqrt


CATALOGS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'assets', 'catalogs')

# Real `simplified_infix` values at the generative.py site, captured from the site's own
# pipeline: skeleton -> prefix_to_infix(power='**') -> uniform(-10, 10) substitution ->
# sympy.simplify -> Abs->abs. Frozen, so the test needs neither sympy nor a fork.
SYMPY_OUTPUTS_AGREEING = [
    '-1.38635147178003', 'atan(3/x0)', '0.415786836673858', 'cos(x1 + 1.64872127070013)',
    '1.26919635484101', 'x1 + log(tanh(x0))', 'sin(x0)', '-2', '6',
    'sin(6.317071*x0 + 2)', 'x0 + x1', '-1/(atan(x0) - 3)',
]
SYMPY_OUTPUTS_WHERE_TO_PREFIX_DIVERGED = [
    # (sympy output, raw-reader prefix, to_prefix's canonicalised prefix) -- both after
    # desugar_sqrt, as the site runs it. The middle column is what read_infix must keep.
    ('-6.21770200000000', ['-6.21770200000000'], ['-6.217702']),
    ('2.00000000000000', ['2.00000000000000'], ['2']),
    ('sin(x1) - 0.5',
     ['-', 'sin', 'x1', '0.5'],
     ['-', 'sin', 'x1', '/', '1', '2']),
    ('0.5/((-1)**x1 + 0.25)',
     ['/', '0.5', '+', 'pow', '-1', 'x1', '0.25'],
     ['/', 'inv', '+', 'pow', '-1', 'x1', '/', '1', '4', '2']),
    ('1/atan(log(x1))',
     ['/', '1', 'atan', 'log', 'x1'],
     ['inv', 'atan', 'log', 'x1']),
    ('1.25/tanh(3)',
     ['/', '1.25', 'tanh', '3'],
     ['/', '*', '5', 'inv', 'tanh', '3', '4']),
]


@pytest.fixture(scope='module')
def engine() -> SimpliPyEngine:
    return SimpliPyEngine.load('acj-4-3', install=True)


def _entries(filename: str) -> dict[str, Any]:
    with open(os.path.join(CATALOGS, filename)) as file:
        return yaml.safe_load(file)['expressions']


def _prepared_corpus() -> list[str]:
    """Every prepared expression across every curated catalog in the repository."""
    corpus = []
    for path in sorted(glob.glob(os.path.join(CATALOGS, '*.yaml'))):
        with open(path) as file:
            document = yaml.safe_load(file)
        if not isinstance(document, dict):
            continue
        expressions = document.get('expressions', document)
        if not isinstance(expressions, dict):
            continue
        for entry in expressions.values():
            if not isinstance(entry, dict):
                continue
            prepared = entry.get('prepared')
            if isinstance(prepared, str) and prepared.strip() and isinstance(entry.get('vars'), dict):
                corpus.append(prepared)
    return corpus


# --------------------------------------------------------------------------------------
# THE MIGRATION, PROVEN: read_infix IS the removed reader
# --------------------------------------------------------------------------------------

def test_read_infix_is_the_removed_reader_corpus_wide(engine: SimpliPyEngine) -> None:
    """Byte identity against the raw-reader oracle over every curated prepared
    expression (6,780 entries, 46.6% spelled with ``sqrt``) and every frozen SymPy
    output -- the whole input language of both migrated sites."""
    corpus = _prepared_corpus()
    assert len(corpus) > 6000
    probes = corpus + SYMPY_OUTPUTS_AGREEING + [row[0] for row in SYMPY_OUTPUTS_WHERE_TO_PREFIX_DIVERGED]
    for text in probes:
        assert engine.read_infix(text) == engine._core.parse(text, True, False), text


def test_read_infix_keeps_the_vocabulary_tolerance_desugar_sqrt_needs(engine: SimpliPyEngine) -> None:
    """``sqrt`` is not in the engine's vocabulary; the reader must emit it as a bare
    leaf so ``desugar_sqrt`` can rewrite it afterwards."""
    raw = engine.read_infix('sqrt(v1 * v2 / v3)')
    assert raw == ['sqrt', '/', '*', 'v1', 'v2', 'v3']
    assert desugar_sqrt(raw, engine.operator_arity) == ['rootn', '/', '*', 'v1', 'v2', 'v3', '2']


def test_evaluation_site_concrete_ground_truth_is_preserved(engine: SimpliPyEngine) -> None:
    """The two entries whose recorded constants the ``to_prefix`` candidate moved, plus
    the ill-conditioned fold that changed y itself -- all preserved under read_infix."""
    entries = _entries('constant.yaml')
    entry = entries['Constant-2']            # sin(v1**2)*cos(v1) - 0.75
    compiled = compile_expression(engine, 'Constant-2', entry['prepared'], entry['vars'])
    assert compiled['constants'] == [2.0, 0.75]

    entry = entries['Const-Test-1']          # 3.14159...*v1*v1
    compiled = compile_expression(engine, 'Const-Test-1', entry['prepared'], entry['vars'])
    assert compiled['constants'] == [3.141592653589793]

    entry = _entries('fastsrb.yaml')['III.8.54']
    compiled = compile_expression(engine, 'III.8.54', entry['prepared'], entry['vars'])
    assert compiled['constants'] == [6.626e-34, 2.0, 3.1415926535897, 2.0]
    v1, v2 = np.array([1.5]), np.array([1.3])
    with np.errstate(all='ignore'):
        y = float(compiled['callable'](v1, v2)[0])
    assert y == pytest.approx(0.000267705857155472)


def test_evaluation_site_compiles_the_sqrt_half_of_the_corpus(engine: SimpliPyEngine) -> None:
    """The 46.6% of the corpus the ``to_prefix`` candidate refused outright: a sqrt
    entry compiles, and its simplified prefix is the explicit dialect (no tags) as the
    downstream consumers require -- 0.14 simplify is dialect-preserving, no form= knob."""
    entries = _entries('fastsrb.yaml')
    checked = 0
    for eq_id, entry in entries.items():
        if 'sqrt' not in entry['prepared']:
            continue
        compiled = compile_expression(engine, eq_id, entry['prepared'], entry['vars'])
        assert not any(str(tok).startswith('<add') or str(tok).startswith('<mul')
                       for tok in compiled['prefix']), eq_id
        checked += 1
        if checked >= 25:
            break
    assert checked == 25


def test_generative_site_prefix_is_unchanged_on_real_sympy_output(engine: SimpliPyEngine) -> None:
    """The migrated ``_sympy_simplify_skeleton`` parse stage: read_infix reproduces the
    raw reader on every frozen SymPy output, including every row where ``to_prefix``
    diverged."""
    for expression in SYMPY_OUTPUTS_AGREEING:
        assert (desugar_sqrt(engine.read_infix(expression), engine.operator_arity)
                == desugar_sqrt(engine._core.parse(expression, True, False), engine.operator_arity))
    for expression, raw_expected, _ in SYMPY_OUTPUTS_WHERE_TO_PREFIX_DIVERGED:
        got = desugar_sqrt(engine.read_infix(expression), engine.operator_arity)
        assert got == raw_expected, expression


# --------------------------------------------------------------------------------------
# THE DISTINCTION, KEPT: to_prefix is a canonicaliser, not a reader
# --------------------------------------------------------------------------------------

def test_to_prefix_refuses_undeclared_vocabulary(engine: SimpliPyEngine) -> None:
    """Why ``to_prefix`` was never the replacement, half one: it parses into the
    canonical state, which only knows declared operators."""
    with pytest.raises(ValueError):
        engine.to_prefix('sqrt(v1 * v2 / v3)')


def test_to_prefix_no_longer_moves_the_spelling_either(engine: SimpliPyEngine) -> None:
    """The blocked-era divergence corpus, re-measured under 0.14's conversion split:
    ``to_prefix`` is now a PURE syntactic conversion (nothing folds, nothing
    re-spells), so the canonicalisation half of the old blocker is gone from the
    library itself -- on in-vocabulary input the two spellings now agree, and the
    refusal above is the ONLY remaining distinction. The third column of the frozen
    corpus records what 0.13's to_prefix did; it is history, not a contract."""
    for expression, raw_expected, _canonical_0_13 in SYMPY_OUTPUTS_WHERE_TO_PREFIX_DIVERGED:
        raw = desugar_sqrt(engine._core.parse(expression, True, False), engine.operator_arity)
        pure = desugar_sqrt(engine.to_prefix(expression), engine.operator_arity)
        assert raw == raw_expected, expression
        assert pure == raw, expression
