"""Tests for the model-agnostic sample seam (``Sample`` / ``iter_samples``).

Two layers: deterministic unit tests for the load-bearing pure helpers (the split + noise
semantics that must match flash-ansr/srbf exactly, no sampling involved), and an
end-to-end pass over a real (small) SkeletonPool built from the test config.
"""
import os

import numpy as np
import pytest

from symbolic_data.skeleton_pool import SkeletonPool
from symbolic_data.samples import Sample, iter_samples, sample_from_skeleton
from symbolic_data.samples import _inject_noise, _split_support_and_validation

CONFIG = os.path.join(os.path.dirname(__file__), "..", "configs", "test", "skeleton_pool_train.yaml")


# --- pure-helper semantics (deterministic, no sampling) -----------------------
def test_split_first_n_support_then_rest():
    x_all = np.arange(32, dtype=np.float64).reshape(16, 2)
    y_all = np.arange(16, dtype=np.float64).reshape(16, 1)

    x_sup, x_val, y_sup, y_val = _split_support_and_validation(x_all, y_all, n_support=8)

    assert x_sup.shape == (8, 2) and x_val.shape == (8, 2)
    assert y_sup.shape == (8, 1) and y_val.shape == (8, 1)
    assert x_sup.dtype == np.float32 and y_val.dtype == np.float32
    np.testing.assert_array_equal(x_sup, x_all[:8])
    np.testing.assert_array_equal(x_val, x_all[8:])
    # full coverage, no overlap
    assert x_sup.shape[0] + x_val.shape[0] == x_all.shape[0]


def test_split_none_n_support_is_half():
    x_all = np.zeros((10, 3), dtype=np.float64)
    y_all = np.zeros((10, 1), dtype=np.float64)
    x_sup, x_val, _, _ = _split_support_and_validation(x_all, y_all, n_support=None)
    assert x_sup.shape[0] == 5 and x_val.shape[0] == 5


def test_split_support_equal_total_falls_back_to_half():
    x_all = np.zeros((6, 2), dtype=np.float64)
    y_all = np.zeros((6, 1), dtype=np.float64)
    # asking for all 6 as support must not leave an empty validation set
    x_sup, x_val, _, _ = _split_support_and_validation(x_all, y_all, n_support=6)
    assert x_sup.shape[0] == 3 and x_val.shape[0] == 3


def test_split_empty_input():
    x_all = np.empty((0, 4), dtype=np.float64)
    y_all = np.empty((0, 1), dtype=np.float64)
    x_sup, x_val, y_sup, y_val = _split_support_and_validation(x_all, y_all, n_support=8)
    assert x_sup.shape == (0, 4) and x_val.shape == (0, 4)
    assert y_sup.shape == (0, 1) and y_val.shape == (0, 1)


def test_inject_noise_zero_and_nonzero():
    rng = np.random.default_rng(0)
    y = np.linspace(-1, 1, 20, dtype=np.float32).reshape(20, 1)

    # noise scales with std and is deterministic for a fixed rng
    a = _inject_noise(y, noise_level=0.1, rng=np.random.default_rng(42))
    b = _inject_noise(y, noise_level=0.1, rng=np.random.default_rng(42))
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, y)
    assert a.shape == y.shape and a.dtype == np.float32

    # a constant array (std 0) is returned unchanged even with noise requested
    const = np.full((10, 1), 3.0, dtype=np.float32)
    np.testing.assert_array_equal(_inject_noise(const, 0.5, rng), const)

    # empty array -> empty copy
    assert _inject_noise(np.empty((0, 1), np.float32), 0.5, rng).shape == (0, 1)


# --- end-to-end over a real pool ----------------------------------------------
@pytest.fixture(scope="module")
def pool():
    p = SkeletonPool.from_config(CONFIG)
    np.random.seed(20240617)
    p.create(12, verbose=False)
    p.skeleton_codes = p.compile_codes(verbose=False)
    return p


def test_iter_samples_end_to_end(pool):
    n_support = 8
    samples = list(iter_samples(pool, n_support=n_support, rng=np.random.default_rng(0), max_trials=16))
    assert samples, "expected at least one non-failing sample from a 12-skeleton pool"

    for s in samples:
        assert isinstance(s, Sample) and not s.placeholder
        assert s.x_support.ndim == 2 and s.x_support.shape[1] == pool.n_variables
        assert s.x_support.dtype == np.float32 and s.y_support.dtype == np.float32
        assert 1 <= s.x_support.shape[0] <= n_support
        assert s.x_validation.shape[1] == pool.n_variables
        assert isinstance(s.expression, list) and len(s.expression) > 0
        assert s.complexity is not None and s.complexity > 0
        assert len(s.constants) == len(pool.skeleton_codes[s.skeleton][1])
        assert 0 <= s.n_variables_used <= pool.n_variables
        # no noise requested -> noisy is an identical copy
        np.testing.assert_array_equal(s.y_support_noisy, s.y_support)


def test_iter_samples_noise_changes_y(pool):
    clean = list(iter_samples(pool, n_support=8, noise_level=0.0, rng=np.random.default_rng(1), max_trials=16))
    noisy = list(iter_samples(pool, n_support=8, noise_level=0.1, rng=np.random.default_rng(1), max_trials=16))
    # at least one sample whose noisy support differs from clean support
    assert any(
        n.y_support.size > 0 and not np.array_equal(n.y_support_noisy, n.y_support) for n in noisy
    )
    # clean run leaves y untouched
    assert all(np.array_equal(c.y_support_noisy, c.y_support) for c in clean)


def test_sample_from_skeleton_unknown_returns_none(pool):
    assert sample_from_skeleton(pool, ("definitely", "not", "a", "skeleton"), n_support=8) is None


def test_mask_unused_variable_columns_zeros_absent_vars(pool):
    # find a skeleton that uses a strict subset of the pool's variables
    target = None
    for sk in pool.skeletons:
        used = {t for t in sk if t in set(pool.variables)}
        if 0 < len(used) < pool.n_variables:
            target = sk
            break
    if target is None:
        pytest.skip("no partial-variable skeleton in this pool draw")

    s = sample_from_skeleton(pool, target, n_support=8, mask_unused_variables=True, max_trials=16)
    if s is None:
        pytest.skip("sampling failed for the chosen skeleton")
    used_idx = {pool.variables.index(t) for t in target if t in set(pool.variables)}
    for col in range(pool.n_variables):
        if col not in used_idx and s.x_support.shape[0] > 0:
            np.testing.assert_array_equal(s.x_support[:, col], 0)
