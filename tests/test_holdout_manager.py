import numpy as np

from symbolic_data.holdout import HoldoutManager


def _make_manager(n_variables: int = 2, allow_nan: bool = False) -> HoldoutManager:
    base_points = np.array(
        [
            [-1.0, -0.5, 0.0],
            [-0.5, 0.0, 0.5],
            [0.0, 0.5, 1.0],
            [0.5, 1.0, 1.5],
        ],
        dtype=np.float64,
    )
    constants = np.array([1.0, -1.0, 0.5], dtype=np.float64)
    return HoldoutManager(
        n_variables=n_variables,
        allow_nan=allow_nan,
        holdout_X=base_points.copy(),
        holdout_C=constants.copy(),
    )


def test_duplicate_skeleton_detected_by_hash():
    manager = _make_manager()

    tokens = ["add", "x0", "x1"]

    def fn(x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
        return x0 + x1

    manager.register_skeleton(tokens, fn, num_constants=0)

    assert tuple(tokens) in manager.skeleton_hashes
    assert manager.is_held_out(tokens, fn, num_constants=0)


def test_functionally_equivalent_skeleton_detected():
    manager = _make_manager()

    tokens_a = ["add", "x0", "x1"]
    tokens_b = ["add", "x1", "x0"]

    def fn(x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
        return x0 + x1

    manager.register_skeleton(tokens_a, fn, num_constants=0)

    assert manager.is_held_out(tokens_b, fn, num_constants=0)


def test_constants_included_in_image_key():
    manager = _make_manager(n_variables=1)

    tokens = ["add", "x0", "<constant>"]

    def fn(x0: np.ndarray, c0: float) -> np.ndarray:
        return x0 + c0

    manager.register_skeleton(tokens, fn, num_constants=1)

    assert manager.is_held_out(tokens, fn, num_constants=1)


def test_nan_outputs_are_zeroed_before_comparison():
    manager = _make_manager(n_variables=1)

    def nan_fn(x0: np.ndarray) -> np.ndarray:
        return np.where(x0 > 0.0, np.nan, 0.0)

    manager.register_skeleton(["identity", "x0"], nan_fn, num_constants=0)

    def zero_fn(x0: np.ndarray) -> np.ndarray:
        return np.zeros_like(x0)

    assert manager.is_held_out(["identity", "x0"], zero_fn, num_constants=0)


def test_default_grid_is_deterministic():
    # Two managers built with NO grid args must share an identical default grid, so the
    # functional-equivalence holdout (image-key backstop) is reproducible across
    # constructions/processes/runs. Pre-fix this drew a fresh unseeded np.random grid.
    a = HoldoutManager(n_variables=3, allow_nan=False)
    b = HoldoutManager(n_variables=3, allow_nan=False)

    assert np.array_equal(a.holdout_X, b.holdout_X)
    assert np.array_equal(a.holdout_C, b.holdout_C)
    assert a.holdout_X.shape == (512, 100)
    assert a.holdout_C.shape == (100,)

    # Each manager owns an independent copy: mutating one must not affect the other
    # (nor the shared lru_cache'd master).
    a.holdout_X[0, 0] += 1.0
    assert not np.array_equal(a.holdout_X, b.holdout_X)
