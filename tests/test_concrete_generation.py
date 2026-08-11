"""The 0.14.0 generation surface: concrete numeric literals, typed exponent slots, the
`choice` / `rounded` literal distributions, `sqrt` desugaring, and the holdout bridge
between concrete-literal skeletons and placeholder-form structures."""
from pathlib import Path

import numpy as np
import pytest
import yaml

from simplipy import SimpliPyEngine

from symbolic_data._generate.holdout import HoldoutManager
from symbolic_data._generate.skeleton_sampling import SkeletonSampler
from symbolic_data.distributions import choice_dist, rounded_dist
from symbolic_data.errors import NoValidSampleFoundError
from symbolic_data.generative import LampleChartonCatalog, _gt_metadata
from symbolic_data.token_ops import desugar_sqrt

CONFIG = Path(__file__).resolve().parent.parent / "configs" / "test" / "catalog_train.yaml"
SLOT_PRIOR = {"name": "choice", "kwargs": {"values": [2, -2, 3, -3], "weights": [2, 2, 1, 1]}}


@pytest.fixture(scope="module")
def engine():
    return SimpliPyEngine.load("acj-4-3", install=True)


def _cfg() -> dict:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def _sampler(engine, typed_slots=..., literal_prior=...) -> SkeletonSampler:
    cfg = _cfg()
    return SkeletonSampler(
        simplipy_engine=engine,
        sample_strategy=cfg["sample_strategy"],
        variables=cfg["variables"],
        operator_weights=cfg["operator_weights"],
        literal_prior=cfg["literal_prior"] if literal_prior is ... else literal_prior,
        typed_slots=cfg["typed_slots"] if typed_slots is ... else typed_slots,
    )


def _operator_argument_spans(tokens, arity):
    """Yield (operator, [child token slices]) for every operator occurrence (prefix walk)."""
    def walk(i):
        end, kids = i + 1, []
        for _ in range(arity.get(tokens[i], 0)):
            start = end
            end = walk(start)[0]
            kids.append(tokens[start:end])
        return end, kids

    out, i = [], 0
    while i < len(tokens):
        if tokens[i] in arity and arity[tokens[i]] >= 1:
            out.append((tokens[i], walk(i)[1]))
        i += 1
    return out


def test_typed_slots_validation_fails_closed(engine):
    with pytest.raises(ValueError, match="unknown operator"):
        _sampler(engine, typed_slots={"frobnicate": {"argument": 0, "prior": SLOT_PRIOR}})
    with pytest.raises(ValueError, match="arity"):
        _sampler(engine, typed_slots={"sin": {"argument": 0, "prior": SLOT_PRIOR}})
    with pytest.raises(ValueError, match="argument"):
        _sampler(engine, typed_slots={"pow": {"argument": 5, "prior": SLOT_PRIOR}})
    with pytest.raises(ValueError, match="prior"):
        _sampler(engine, typed_slots={"pow": {"argument": 1}})


def test_sampler_yields_concrete_expressions_with_constrained_slots(engine):
    sampler = _sampler(engine)
    rng = np.random.default_rng(20260811)
    arity = engine.operator_arity
    cfg = _cfg()
    rootn_support = {float(v) for v in cfg["typed_slots"]["rootn"]["prior"]["kwargs"]["values"]}
    pow_int_support = {float(v) for v in cfg["typed_slots"]["pow"]["prior"][0]["kwargs"]["values"]}

    saw_pow = saw_rootn = 0
    for _ in range(400):
        tokens = sampler.sample(int(rng.integers(1, 11)), rng=rng)
        assert "<constant>" not in tokens                       # concrete literals only
        for token in tokens:
            try:
                value = float(token)
            except ValueError:
                continue
            if value.is_integer():                              # integral values take the
                assert "." not in token and "e" not in token    # integer spelling
        for operator, kids in _operator_argument_spans(tokens, arity):
            if operator not in ("pow", "rootn"):
                continue
            slot = kids[1]
            assert len(slot) == 1, f"{operator} slot must be a literal, got {slot}"
            value = float(slot[0])                              # never a subtree / symbol
            if operator == "rootn":
                saw_rootn += 1
                assert value in rootn_support
            else:
                saw_pow += 1
                if value.is_integer() and "." not in slot[0]:
                    assert value in pow_int_support
    assert saw_pow > 0 and saw_rootn > 0


def test_choice_dist_contract():
    rng = np.random.default_rng(0)
    draws = choice_dist([2, 3], weights=[1.0, 0.0], size=200, rng=rng)
    assert draws.dtype == np.float64
    assert set(np.unique(draws)) == {2.0}                        # weight 0 is never drawn
    assert choice_dist([1, 2, 3], size=(4, 5), rng=rng).shape == (4, 5)
    with pytest.raises(ValueError, match="non-empty"):
        choice_dist([], rng=rng)
    with pytest.raises(ValueError, match="weights"):
        choice_dist([1, 2], weights=[1.0], rng=rng)
    with pytest.raises(ValueError, match="finite"):
        choice_dist([1, 2], weights=[float("nan"), 1.0], rng=rng)
    with pytest.raises(ValueError, match="non-negative"):
        choice_dist([1, 2], weights=[-1.0, 2.0], rng=rng)


def test_rounded_dist_contract():
    rng = np.random.default_rng(0)
    tiny = rounded_dist({"name": "normal", "kwargs": {"loc": 0, "scale": 0.001}}, size=500, rng=rng)
    assert not np.signbit(tiny[tiny == 0.0]).any()               # -0.0 normalized away

    infs = rounded_dist(lambda size, rng: np.full(size, np.inf), size=3, rng=rng)
    assert np.isinf(infs).all()                                  # non-finite passes through

    one_decimal = rounded_dist(
        {"name": "normal", "kwargs": {"loc": 0, "scale": 5}},
        precision=lambda size, rng: np.asarray([1.0]), size=50, rng=rng)
    assert all(round(float(v), 1) == float(v) for v in one_decimal)


def test_desugar_sqrt_rewrites_to_rootn(engine):
    arity = engine.operator_arity
    assert desugar_sqrt(["sqrt", "x1"], arity) == ["rootn", "x1", "2"]
    assert desugar_sqrt(["sqrt", "sqrt", "x1"], arity) == ["rootn", "rootn", "x1", "2", "2"]
    assert desugar_sqrt(["/", "exp", "x1", "sqrt", "*", "2", "x2"], arity) == \
        ["/", "exp", "x1", "rootn", "*", "2", "x2", "2"]
    assert desugar_sqrt(["+", "x1", "x2"], arity) == ["+", "x1", "x2"]


def test_holdout_structure_layer_bridges_concrete_and_placeholder_forms():
    # Registered skeletons and generated skeletons meet in NORMALIZED form, so a corpus of
    # concrete literals still decontaminates against placeholder-form holdout structures.
    manager = HoldoutManager(n_variables=1, allow_nan=False)

    def image_layer_unavailable(*args, **kwargs):
        raise NameError("image layer unavailable in this test")

    with pytest.warns(RuntimeWarning, match="exact-structure layer"):
        manager.register_skeleton(["*", "x1", "<constant>"], image_layer_unavailable, 1)

    assert tuple(manager._normalize_tokens(["*", "x1", "2.0"])) in manager.skeleton_hashes
    assert manager.is_held_out(["*", "x1", "-3"], image_layer_unavailable, 1)
    assert manager.is_held_out(["*", "v1", "0.25"], image_layer_unavailable, 1)


def test_gt_metadata_concrete_and_placeholder_paths():
    # Concrete skeleton, zero literals: the tokens ARE the ground truth.
    expression, complexity = _gt_metadata(["*", "x1", "2.5"], [])
    assert complexity == 3 and expression is not None and "<constant>" not in expression

    # Placeholder-form skeleton (frozen spec): sampled literals substitute in.
    expression, complexity = _gt_metadata(["*", "x1", "<constant>"], np.asarray([2.5]))
    assert complexity == 3 and "<constant>" not in expression and "2.5" in expression


def test_catalog_generates_concrete_prefix_expressions_end_to_end():
    catalog = LampleChartonCatalog.from_config(str(CONFIG))
    rng = np.random.default_rng(20260811)
    got, attempts = 0, 0
    while got < 10:
        attempts += 1
        assert attempts < 1000, "generation acceptance collapsed"
        try:
            sample = catalog.sample(n_support=16, rng=rng)
        except NoValidSampleFoundError:
            continue
        got += 1
        skeleton = list(sample["skeleton_hash"])
        assert sample["skeleton_constants"] == []               # nothing left to fit
        assert np.asarray(sample["literals"]).size == 0
        assert not any(str(t).startswith("<") for t in skeleton)  # concrete, untagged prefix
        assert np.isfinite(sample["x_support"]).all()
        assert np.isfinite(sample["y_support"]).all()


def test_simplify_call_signature_errors_propagate_instead_of_spinning():
    # Twice in this project's history a removed simplify keyword produced an infinite
    # full-CPU loop: the TypeError was wrapped as a retryable NoValidSampleFoundError and
    # the rejection loop resampled forever. Programming errors must escape the loop.
    catalog = LampleChartonCatalog.from_config(str(CONFIG))

    def broken_simplify(*args, **kwargs):
        raise TypeError("simplify() got an unexpected keyword argument 'inplace'")

    catalog.simplipy_engine.simplify = broken_simplify
    with pytest.raises(TypeError, match="unexpected keyword"):
        for _ in range(50):  # a rejection loop would spin past this in microseconds
            catalog.sample(n_support=8, rng=np.random.default_rng(0))
