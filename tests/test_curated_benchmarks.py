"""Tests for the curated package-data benchmark loaders (feynman, nguyen).

Two layers of verification:

* **Integrity** (no extra deps): every shipped equation registers, parses through SimpliPy, and
  samples finite ``(X, y)``. This is the runtime regression guard.
* **Faithfulness** (sympy-gated, offline): for every equation, ``simplipy(prepared)`` is compared
  against ``sympy(raw)`` -- the original source formula stored in the spec -- on the *same* sampled
  inputs. This locks the specs against silent corruption through a fully independent parse path,
  using only the shipped package data (no network).
"""
import warnings

import numpy as np
import pytest

from symbolic_data import BENCHMARKS, SpecBenchmark, load_benchmark

CURATED = {"feynman": 100, "nguyen": 12}


@pytest.fixture(scope="module")
def benches():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return {name: load_benchmark(name, random_state=0) for name in CURATED}


@pytest.mark.parametrize("name,count", CURATED.items())
def test_curated_registered_with_expected_count(benches, name, count):
    bench = benches[name]
    assert name in BENCHMARKS
    assert isinstance(bench, SpecBenchmark)
    assert len(bench.equation_ids()) == count


@pytest.mark.parametrize("name", CURATED)
def test_curated_provenance_stamped(benches, name):
    prov = benches[name].provenance
    assert prov["source"] == "package"
    assert prov["benchmark"] == name
    assert prov["spec_version"]
    assert prov["resource"].endswith(f"{name}.yaml")


@pytest.mark.parametrize("name", CURATED)
def test_every_equation_samples_finite(benches, name):
    bench = benches[name]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for eq_id in bench.equation_ids():
            sample = bench.sample(eq_id, n_points=16, random_state=7)
            X, y = sample["data"]["X"], sample["data"]["y"]
            assert X.shape[0] == 16, eq_id
            assert np.all(np.isfinite(X)), eq_id
            assert np.all(np.isfinite(y)), eq_id
            # the variable count in the spec matches the sampled columns
            assert X.shape[1] == sample["metadata"]["n_variables"], eq_id


def test_nguyen_known_equations(benches):
    bench = benches["nguyen"]
    ids = bench.equation_ids()
    assert ids[0] == "Nguyen-1" and ids[-1] == "Nguyen-12"
    # Nguyen-1 = x^3 + x^2 + x sampled on [-1, 1]
    eq = bench._entries["Nguyen-1"]
    assert eq["raw"] == "x1**3 + x1**2 + x1"
    assert eq["vars"]["v1"]["sample_range"] == [-1.0, 1.0]


def test_feynman_known_equation(benches):
    bench = benches["feynman"]
    assert "I.6.2a" in bench.equation_ids()
    eq = bench._entries["I.6.2a"]
    assert eq["raw"] == "exp(-theta**2/2)/sqrt(2*pi)"
    assert eq["vars"]["v1"]["name"] == "theta"


# --------------------------------------------------------------------------------------------------
# Faithfulness oracle: simplipy(prepared) == sympy(raw) on shared inputs. Offline (package data only).
# This is sympy-gated; the integrity tests above run without sympy.
# --------------------------------------------------------------------------------------------------
try:
    import sympy  # noqa: F401
    _HAS_SYMPY = True
except ImportError:  # pragma: no cover - sympy is in the [dev] extra
    _HAS_SYMPY = False


@pytest.mark.skipif(not _HAS_SYMPY, reason="sympy not installed (install symbolic-data[dev])")
@pytest.mark.parametrize("name", CURATED)
def test_prepared_matches_source_formula(benches, name):
    """Every shipped equation: SimpliPy(prepared) agrees with sympy(raw) on sampled inputs.

    Pure relative agreement (rtol-only with a tiny absolute floor for genuine zeros), so the check
    cannot be loosened by small output magnitudes.
    """
    fn_locals = {
        "arcsin": sympy.asin, "arccos": sympy.acos, "arctan": sympy.atan, "ln": sympy.log,
        "exp": sympy.exp, "sqrt": sympy.sqrt, "sin": sympy.sin, "cos": sympy.cos, "tanh": sympy.tanh,
        "pi": sympy.pi,
    }
    bench = benches[name]
    mismatches = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for eq_id in bench.equation_ids():
            entry = bench._entries[eq_id]
            nvars = entry["n_variables"]
            names = [entry["vars"][f"v{k}"]["name"] for k in range(1, nvars + 1)]
            local = {n: sympy.Symbol(n) for n in names}
            local.update(fn_locals)
            f = sympy.lambdify(
                [sympy.Symbol(n) for n in names],
                sympy.sympify(entry["raw"], locals=local),
                modules=["numpy"],
            )
            sample = bench.sample(eq_id, n_points=128, random_state=11)
            X, y_engine = sample["data"]["X"], sample["data"]["y"]
            with np.errstate(all="ignore"):
                y_sympy = np.broadcast_to(
                    np.asarray(f(*[X[:, k] for k in range(nvars)]), dtype=float), y_engine.shape
                )
            finite = np.isfinite(y_sympy) & np.isfinite(y_engine)
            if finite.sum() < 2 or not np.allclose(y_sympy[finite], y_engine[finite], rtol=1e-9, atol=1e-12):
                mismatches.append(eq_id)
    assert not mismatches, f"{name}: prepared != source formula for {mismatches}"
