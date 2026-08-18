"""simplipy 0.14 removed ``SimpliPyEngine.parse``. These tests MEASURE what the two call
sites that used it would become under the replacement surface (``to_prefix``), over the
real curated catalogs and over real SymPy output, and pin the result.

The verdict they record: NEITHER site can be migrated, so neither is migrated.
``_evaluation.compile_expression`` and ``generative._sympy_simplify_skeleton`` still call
``engine.parse`` and are BLOCKED pending an owner decision.

WHY THE OLD BEHAVIOUR IS STILL REACHABLE HERE. The removed ``engine.parse`` was a pure
delegation to the Rust raw reader (``self._core.parse(infix, convert_expression,
mask_numbers)``), which simplipy 0.14 keeps as the conversion hub's INTERNAL entry with no
compatibility promise. These tests use it as the ORACLE for the removed public method --
the only way to compare both spellings in one process, and its subject IS the removed
behaviour. Production code must not use it.

WHAT BREAKS.

1. VOCABULARY TOLERANCE. The raw reader passed an unknown function name through as a
   leaf; that is precisely what :func:`symbolic_data.token_ops.desugar_sqrt` documents and
   relies on -- ``sqrt`` is not in the engine's vocabulary, so both sites parse first and
   rewrite ``sqrt u`` -> ``rootn u 2`` afterwards. ``to_prefix`` reads through the AC
   parser, which REFUSES the name, and no rewrite can run after a raise. 3159 of the 6780
   curated catalog entries (46.6%) are affected.
2. EXACT-RATIONAL CANONICALISATION. The raw reader preserved the literal spelling;
   ``to_prefix`` reads into the canonical state, where a decimal becomes an exact rational
   and constant subtrees fold. The site returns ``expression`` and ``constants`` -- the
   CONCRETE ground truth -- so the recorded constants change (``0.75`` -> ``3``, ``4``;
   ``v1*v1`` -> ``pow v1 2`` adds a structural ``2`` to the constant list), and in
   ill-conditioned expressions the folded f64 result is a different NUMBER.
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
SYMPY_OUTPUTS_DIVERGING = [
    # (sympy output, old prefix, new prefix) -- both after desugar_sqrt, as the site runs it
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


class _OldEngine:
    """The pre-0.14 spelling: ``engine.parse`` == the raw reader."""

    def __init__(self, inner: SimpliPyEngine) -> None:
        self._inner = inner

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def parse(self, infix_expression: str, convert_expression: bool = True, mask_numbers: bool = False) -> list[str]:
        return self._inner._core.parse(infix_expression, convert_expression, mask_numbers)


class _NewEngine(_OldEngine):
    """The 0.14 migration candidate: ``to_prefix``."""

    def parse(self, infix_expression: str, convert_expression: bool = True, mask_numbers: bool = False) -> list[str]:
        assert mask_numbers is False, 'the symbolic-data sites never masked at parse time'
        return self._inner.to_prefix(infix_expression)


def _entries(filename: str) -> dict[str, Any]:
    with open(os.path.join(CATALOGS, filename)) as file:
        return yaml.safe_load(file)['expressions']


@pytest.fixture(scope='module')
def curated_census() -> tuple[int, int]:
    """(entries with a prepared expression, of which spelled with ``sqrt``) over every
    curated catalog in the repository."""
    total = with_sqrt = 0
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
                total += 1
                with_sqrt += 'sqrt' in prepared
    return total, with_sqrt


# --------------------------------------------------------------------------------------
# BLOCKER 1 -- the vocabulary tolerance both sites are built on
# --------------------------------------------------------------------------------------

def test_desugar_sqrt_depends_on_a_reader_that_passes_unknown_names_through(engine: SimpliPyEngine) -> None:
    """``sqrt`` is not in the engine's vocabulary. The removed reader emitted it as a bare
    token, which ``desugar_sqrt`` then rewrote; ``to_prefix`` raises before that can run."""
    raw = engine._core.parse('sqrt(v1 * v2 / v3)', True, False)
    assert raw == ['sqrt', '/', '*', 'v1', 'v2', 'v3']
    assert desugar_sqrt(raw, engine.operator_arity) == ['rootn', '/', '*', 'v1', 'v2', 'v3', '2']

    with pytest.raises(ValueError):
        engine.to_prefix('sqrt(v1 * v2 / v3)')


def test_sqrt_is_half_the_curated_corpus(curated_census: tuple[int, int]) -> None:
    """The blast radius of BLOCKER 1 at the ``_evaluation`` site: these entries raise
    instead of compiling, and ``compile_expression`` does not guard the parse call."""
    total, with_sqrt = curated_census
    assert total > 6000
    assert with_sqrt / total > 0.4, (total, with_sqrt)


# --------------------------------------------------------------------------------------
# BLOCKER 2 -- the _evaluation site's concrete ground truth
# --------------------------------------------------------------------------------------

def test_evaluation_site_is_value_preserving_on_the_well_conditioned_entries(engine: SimpliPyEngine) -> None:
    """The bar the site's numeric use sets: the compiled callable must give the same y.
    It holds for every sqrt-free FastSRB entry except the ill-conditioned one pinned
    below (measured corpus-wide: 3616 of the 3621 entries both spellings can compile)."""
    old_engine, new_engine = _OldEngine(engine), _NewEngine(engine)
    rng = np.random.default_rng(0)
    checked = 0
    for eq_id, entry in _entries('fastsrb.yaml').items():
        prepared = entry['prepared']
        if 'sqrt' in prepared or eq_id == 'III.8.54':
            continue
        compiled_old = compile_expression(old_engine, eq_id, prepared, entry['vars'])
        compiled_new = compile_expression(new_engine, eq_id, prepared, entry['vars'])
        inputs = [rng.uniform(0.5, 2.0, size=8) for _ in compiled_old['variable_order']]
        with np.errstate(all='ignore'):
            y_old = np.asarray(compiled_old['callable'](*inputs), dtype=float)
            y_new = np.asarray(compiled_new['callable'](*inputs), dtype=float)
        assert np.allclose(y_old, y_new, rtol=1e-9, atol=1e-12, equal_nan=True), eq_id
        checked += 1
    assert checked > 80, 'corpus slice too small to be a proof'


def test_evaluation_site_value_preservation_fails_on_ill_conditioned_folding(engine: SimpliPyEngine) -> None:
    """BLOCKER 2a. Exact-rational folding rewrites ``2 * 3.1415926535897 / 6.626e-34`` into
    a 38-digit integer over 3313. The f64 product then differs in the low bits -- which is
    the whole value of ``sin`` at that magnitude, so y is a DIFFERENT NUMBER, not a
    rounding difference. 5 of the 3621 shared entries diverge this way."""
    entry = _entries('fastsrb.yaml')['III.8.54']
    compiled_old = compile_expression(_OldEngine(engine), 'III.8.54', entry['prepared'], entry['vars'])
    compiled_new = compile_expression(_NewEngine(engine), 'III.8.54', entry['prepared'], entry['vars'])

    assert compiled_old['constants'] == [6.626e-34, 2.0, 3.1415926535897, 2.0]
    assert compiled_new['constants'] == [3.1415926535897e+37, 3313.0, 2.0]

    v1, v2 = np.array([1.5]), np.array([1.3])
    with np.errstate(all='ignore'):
        y_old = float(compiled_old['callable'](v1, v2)[0])
        y_new = float(compiled_new['callable'](v1, v2)[0])
    assert y_old == pytest.approx(0.000267705857155472)
    assert y_new == pytest.approx(0.8669423214155054)


def test_evaluation_site_concrete_constants_change(engine: SimpliPyEngine) -> None:
    """BLOCKER 2b. ``constants`` is the recorded ground truth of a catalog entry. Under
    canonicalisation a decimal splits into an exact rational's two literals, and a
    collected power contributes its structural exponent -- neither is a fitted constant."""
    old_engine, new_engine = _OldEngine(engine), _NewEngine(engine)
    entries = _entries('constant.yaml')

    entry = entries['Constant-2']            # sin(v1**2)*cos(v1) - 0.75
    old = compile_expression(old_engine, 'Constant-2', entry['prepared'], entry['vars'])
    new = compile_expression(new_engine, 'Constant-2', entry['prepared'], entry['vars'])
    assert old['constants'] == [2.0, 0.75]
    assert new['constants'] == [2.0, 3.0, 4.0]

    entry = entries['Const-Test-1']          # 3.14159...*v1*v1
    old = compile_expression(old_engine, 'Const-Test-1', entry['prepared'], entry['vars'])
    new = compile_expression(new_engine, 'Const-Test-1', entry['prepared'], entry['vars'])
    assert old['constants'] == [3.141592653589793]
    assert new['constants'] == [3.141592653589793, 2.0]


@pytest.mark.xfail(strict=True, reason='BLOCKED: to_prefix refuses sqrt and moves the concrete '
                                       'ground truth; this is the acceptance criterion, red until '
                                       'the owner rules')
def test_evaluation_site_migration_acceptance_criterion(engine: SimpliPyEngine) -> None:
    """Migrating ``compile_expression`` requires: every entry the old spelling compiled
    still compiles, with the same concrete constants."""
    old_engine, new_engine = _OldEngine(engine), _NewEngine(engine)
    for eq_id, entry in _entries('fastsrb.yaml').items():
        old = compile_expression(old_engine, eq_id, entry['prepared'], entry['vars'])
        new = compile_expression(new_engine, eq_id, entry['prepared'], entry['vars'])
        assert old['constants'] == new['constants'], eq_id


# --------------------------------------------------------------------------------------
# BLOCKER 3 -- the generative site (SymPy output back to prefix)
# --------------------------------------------------------------------------------------

def test_generative_site_agrees_where_sympy_prints_the_canonical_state(engine: SimpliPyEngine) -> None:
    """The majority case (measured 164 of 300 real SymPy outputs)."""
    for expression in SYMPY_OUTPUTS_AGREEING:
        old = desugar_sqrt(engine._core.parse(expression, True, False), engine.operator_arity)
        new = desugar_sqrt(engine.to_prefix(expression), engine.operator_arity)
        assert old == new, expression


def test_generative_site_diverges_on_real_sympy_output(engine: SimpliPyEngine) -> None:
    """BLOCKER 3. The catalog stores CONCRETE literals, so a changed spelling is a changed
    catalog row: SymPy's ``-6.21770200000000`` is recorded verbatim by the old reader and
    as ``-6.217702`` by ``to_prefix``, ``0.5`` becomes the two-token rational ``/ 1 2``,
    and ``1/u`` becomes ``inv u``. Measured 136 of 300 real SymPy outputs.
    """
    for expression, old_expected, new_expected in SYMPY_OUTPUTS_DIVERGING:
        old = desugar_sqrt(engine._core.parse(expression, True, False), engine.operator_arity)
        new = desugar_sqrt(engine.to_prefix(expression), engine.operator_arity)
        assert old == old_expected, expression
        assert new == new_expected, expression
        assert old != new


@pytest.mark.xfail(strict=True, reason='BLOCKED: to_prefix moves SymPy output; this is the '
                                       'acceptance criterion for migrating the generative site')
def test_generative_site_migration_acceptance_criterion(engine: SimpliPyEngine) -> None:
    """Migrating ``_sympy_simplify_skeleton`` requires the emitted prefix to be unchanged."""
    for expression, _, _ in SYMPY_OUTPUTS_DIVERGING:
        assert (desugar_sqrt(engine._core.parse(expression, True, False), engine.operator_arity)
                == desugar_sqrt(engine.to_prefix(expression), engine.operator_arity))
