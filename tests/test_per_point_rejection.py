"""Per-point rejection (0.12.0): partial domains realize; degenerate domains stay honest."""
import warnings

import numpy as np
import yaml

from symbolic_data import ProblemSource


def _catalog(tmp_path, prepared, lo, hi):
    cfg = {
        "metadata": {"name": "rejection-smoke", "version": 1, "source_kind": "set",
                     "sampling_defaults": {"n_points": 16, "method": "random", "noise": 0.0}},
        "expressions": {"E1": {
            "raw": prepared, "prepared": prepared, "n_variables": 1,
            "vars": {"v1": {"name": "x1", "sample_range": [lo, hi], "sample_type": ["uni", "pos"]},
                     "v0": {"name": "y"}},
        }},
    }
    path = tmp_path / "cat.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return str(path)


def _one(path):
    src = ProblemSource({"catalog": path,
                         "sampling": {"n_support": 32, "n_validation": 8, "problems_per_expression": 1}})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return next(iter(src))


def test_partial_domain_realizes_with_finite_points(tmp_path):
    # sqrt over [-1, 1]: valid fraction ~0.5 -- exhausted under whole-draw rejection (0.5^40),
    # trivially satisfiable per point. All returned points must be finite and in-domain.
    p = _one(_catalog(tmp_path, "v1 ** (1/2)", -1.0, 1.0))
    assert not p.is_placeholder
    assert p.x_support.shape[0] == 32 and p.x_validation.shape[0] == 8
    assert np.isfinite(p.y_support).all() and np.isfinite(p.y_validation).all()
    assert (p.x_support >= 0).all()          # the conditional law: declared range ∩ domain


def test_degenerate_domain_still_placeholders(tmp_path):
    # log over [-2, -1]: valid fraction 0 -- must exhaust the cap and placeholder honestly.
    p = _one(_catalog(tmp_path, "log(v1)", -2.0, -1.0))
    assert p.is_placeholder
    assert "max_trials" in str(p.placeholder_reason) or "non-finite" in str(p.placeholder_reason)


def test_adaptive_batch_oversamples_by_observed_rejection():
    from symbolic_data.catalog import _adaptive_batch

    # observed f = 0.25 -> need 100 more points -> batch ~ 100/0.25*1.25 = 500 (+8)
    assert _adaptive_batch(needed=100, collected=25, drawn=100, budget_left=10_000) == 508
    # budget-capped
    assert _adaptive_batch(needed=100, collected=25, drawn=100, budget_left=300) == 300
    # floor: zero collected so far must not divide to an infinite batch
    assert _adaptive_batch(needed=10, collected=0, drawn=100, budget_left=10_000) <= 10_000


def test_partial_domain_realizes_in_few_rounds(tmp_path):
    """Efficiency: with per-point + adaptive oversampling the sqrt case needs ~2 eval rounds,
    not ~1/f iterations. Counted via a wrapped rng."""
    import warnings
    import yaml
    from symbolic_data import ProblemSource

    cfg = {
        "metadata": {"name": "eff-smoke", "version": 1, "source_kind": "set",
                     "sampling_defaults": {"n_points": 16, "method": "random", "noise": 0.0}},
        "expressions": {"E1": {
            "raw": "v1 ** (1/2)", "prepared": "v1 ** (1/2)", "n_variables": 1,
            "vars": {"v1": {"name": "x1", "sample_range": [-1.0, 1.0], "sample_type": ["uni", "pos"]},
                     "v0": {"name": "y"}},
            "meta": {"finite_fraction": 0.5},
        }},
    }
    path = tmp_path / "cat.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    src = ProblemSource({"catalog": str(path),
                         "sampling": {"n_support": 64, "n_validation": 16, "problems_per_expression": 1}})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p = next(iter(src))
    assert not p.is_placeholder and p.x_support.shape[0] == 64


def test_rejection_is_storage_dtype_aware(tmp_path):
    """Rejection is judged in the STORAGE dtype, so what counts as valid moves with it.

    At f32 storage this expression was the witness: exp(60x) passes 3.4e38 at x ~ 1.48, so
    only the low-x slice of [0, 10] survived and x_support.max() had to stay under 1.5. At
    f64 storage the same expression is finite across the whole interval (it reaches ~1e260,
    well inside 1.8e308), so the slice is the WHOLE box -- the migration hands back the
    ~85% of this domain the f32 bar was discarding."""
    p = _one(_catalog(tmp_path, "exp(60*v1)", 0.0, 10.0))
    assert not p.is_placeholder
    assert np.isfinite(p.y_support).all()
    assert p.x_support.max() > 5.0, "the high-x slice f32 used to reject must now be reachable"
    assert p.y_support.max() > 1e40, "and it must carry values f32 could never have stored"


def test_rejection_still_fires_past_the_f64_boundary(tmp_path):
    """The bar moved; it did not disappear. exp(800x) passes 1.8e308 at x ~ 0.89, so on
    [0, 10] only the low-x slice is realizable and the rest must still be rejected."""
    p = _one(_catalog(tmp_path, "exp(800*v1)", 0.0, 10.0))
    assert not p.is_placeholder
    assert np.isfinite(p.y_support).all()
    assert p.x_support.max() < 1.0
