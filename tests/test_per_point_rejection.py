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
