"""gt_kind + measured-data schema (0.11.0): inference, validation, from_data, npz round-trip."""
import numpy as np
import pytest

from symbolic_data import Problem, ProblemCatalog
from symbolic_data.problem import GT_KINDS
from symbolic_data.tensor_ops import mask_unused_variable_columns


def _arrays(n=8, d=2):
    x = np.arange(n * d, dtype=np.float32).reshape(n, d)
    y = x[:, :1] * 2.0
    return x, y


def _synthetic(**over):
    x, y = _arrays()
    base = dict(x_support=x, y_support=y, y_support_noisy=y.copy(),
                x_validation=x[:2], y_validation=y[:2], y_validation_noisy=y[:2].copy(),
                skeleton=("*", "x1", "<constant>"), expression=["*", "x1", "2.0"],
                constants=[2.0], variables=["x1", "x2"], complexity=3)
    base.update(over)
    return Problem(**base)


def test_gt_kind_inference_and_validation():
    assert _synthetic().gt_kind == "exact"                                   # structure => exact
    assert _synthetic(expression=None).gt_kind == "exact"                    # skeleton-only (generative edge)
    x, y = _arrays()
    blackbox = Problem.from_data(x, y)
    assert blackbox.gt_kind == "none" and blackbox.skeleton is None
    with pytest.raises(ValueError, match="gt_kind"):
        _synthetic(gt_kind="best")
    with pytest.raises(ValueError, match="empty"):
        _synthetic(gt_kind="none")                                           # none + structure
    with pytest.raises(ValueError, match="skeleton or expression"):
        Problem.from_data(x, y, gt_kind="reference")                         # reference w/o structure
    assert set(GT_KINDS) == {"exact", "reference", "none"}


def test_placeholder_exempt_from_structure_validation():
    p = Problem.placeholder(variables=["x1"], reason="test")
    assert p.gt_kind in GT_KINDS                                             # inferred, no raise


def test_from_data_reference_conventions():
    x, y = _arrays()
    ref = y + 0.1
    p = Problem.from_data(x, y, expression=["*", "x1", "2.0"], y_reference_support=ref,
                          eq_id="hubble")
    assert p.gt_kind == "reference"
    assert p.skeleton is not None                                            # derived via simplipy
    assert p.noise is None                                                   # unknown measurement noise
    np.testing.assert_array_equal(p.y_support, p.y_support_noisy)            # measured y IS the target
    assert p.y_support_noisy is not p.y_support                              # copy, not alias
    np.testing.assert_allclose(p.y_reference_support, ref)


def test_from_data_normalizes_reference_arrays():
    # reference predictions get the SAME normalization as y: float32 column vectors, shape-checked
    # (a raw float64 1-d law prediction must not survive to the npz as-is).
    x, y = _arrays()
    p = Problem.from_data(x, y, expression=["*", "x1", "2.0"],
                          y_reference_support=np.asarray(y, dtype=np.float64).ravel())
    assert p.y_reference_support.dtype == np.float32
    assert p.y_reference_support.shape == p.y_support.shape
    with pytest.raises(ValueError, match="y_reference_support shape"):
        Problem.from_data(x, y, expression=["*", "x1", "2.0"], y_reference_support=y[:3])


def test_round_trip_mixed_catalog_npz(tmp_path):
    x, y = _arrays()
    problems = [
        _synthetic(eq_id="syn1"),
        Problem.from_data(x, y, expression=["*", "x1", "2.0"], y_reference_support=y + 1.0,
                          y_reference_validation=np.zeros((0, 1), np.float32), eq_id="ref1"),
        Problem.from_data(x, y, eq_id="bb1"),
    ]
    cat = ProblemCatalog.from_problems(problems, name="mixed-smoke")
    path = cat.save(tmp_path / "mixed.npz")
    loaded = ProblemCatalog.from_npz(path)
    kinds = [p.gt_kind for p in loaded.problems]
    assert kinds == ["exact", "reference", "none"]
    np.testing.assert_array_equal(loaded.problems[1].y_reference_support,
                                  problems[1].y_reference_support)
    assert loaded.problems[0].y_reference_support is None
    assert loaded.problems[2].expression is None


def test_legacy_scalar_blob_infers_gt_kind():
    d = _synthetic().to_dict()
    for k in ("gt_kind", "y_reference_support", "y_reference_validation"):
        d.pop(k)                                                             # 0.10-era dict
    assert Problem.from_dict(d).gt_kind == "exact"


def test_mask_unused_variable_columns_noop_without_skeleton():
    x, _ = _arrays()
    before = x.copy()
    mask_unused_variable_columns([x], variables=["x1", "x2"], skeleton_tokens=None)
    np.testing.assert_array_equal(x, before)                                 # black-box: keep all columns
