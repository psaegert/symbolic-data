"""Tests for the curated package-data benchmark loaders (fastsrb, feynman, nguyen).

All three ship their spec as package data. Two layers of verification:

* **Integrity** (no extra deps): every shipped equation registers and samples finite ``(X, y)``.
  For ``feynman``/``nguyen`` (specs we convert from source) this holds for every equation; for
  ``fastsrb`` (vendored verbatim from upstream) a couple of equations are mostly-non-finite by
  construction under their own ranges, so it is checked tolerantly.
* **Faithfulness** (sympy-gated, offline): for the converted specs, ``simplipy(prepared)`` is
  compared against ``sympy(raw)`` on the *same* sampled inputs -- an independent parse path that locks
  the specs against silent corruption, using only the shipped package data (no network). ``fastsrb``
  is excluded: its upstream ``prepared`` folds physical constants as literals, so ``raw`` is not
  independently evaluable from ``vars`` alone.
"""
import warnings

import numpy as np
import pytest

from symbolic_data import BENCHMARKS, SpecBenchmark, load_benchmark, load_spec

# All curated loaders + their equation counts (registration / provenance).
CURATED = {"fastsrb": 120, "feynman": 100, "nguyen": 12}
# Specs we convert from source: every equation must sample finite AND pass the faithfulness oracle.
GATED = {"feynman": 100, "nguyen": 12}


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
    # provenance is now spec-header-aware: a `spec` (the header) referencing a `problems` set
    assert prov["spec"].endswith(f"specs/{name}.yaml")
    assert prov["problems"]["resource"].endswith(f"data/{name}.yaml")


@pytest.mark.parametrize("name", CURATED)
def test_spec_header_carried(benches, name):
    """Each curated benchmark carries its versioned spec header (metadata/source/sampling)."""
    header = benches[name].header
    assert header["metadata"]["name"] == name
    assert header["source"]["kind"] == "set"
    assert "n_points" in header["sampling"]


def test_canonical_n_points_from_header_applied():
    """The spec header's sampling.n_points is the no-arg default (Nguyen's canonical 20),
    while an explicit n_points still overrides it."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ng = load_spec("nguyen", random_state=0)
        assert ng.header["sampling"]["n_points"] == 20
        assert ng.sample("Nguyen-1", random_state=0)["data"]["X"].shape[0] == 20  # header default
        assert ng.sample("Nguyen-1", n_points=5, random_state=0)["data"]["X"].shape[0] == 5  # override


def test_load_spec_unknown_raises():
    with pytest.raises(KeyError):
        load_spec("does_not_exist")


@pytest.mark.parametrize("name", GATED)
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


def test_fastsrb_mostly_samples_and_iterates(benches):
    """fastsrb is vendored verbatim; a couple of upstream equations are mostly-non-finite by
    construction. The vast majority sample finite, and iter_samples skips failures gracefully."""
    bench = benches["fastsrb"]
    finite = 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for eq_id in bench.equation_ids():
            try:
                y = bench.sample(eq_id, n_points=16, max_trials=1000, random_state=7)["data"]["y"]
                finite += int(np.all(np.isfinite(y)))
            except RuntimeError:
                pass
        # iter_samples must complete without raising (it skips per-equation failures)
        iterated = sum(1 for _ in bench.iter_samples(count=1, n_points=8))
    assert finite >= 117, f"only {finite}/120 fastsrb equations sampled finite"
    assert iterated >= 117


def test_nguyen_known_equations(benches):
    bench = benches["nguyen"]
    ids = bench.equation_ids()
    assert ids[0] == "Nguyen-1" and ids[-1] == "Nguyen-12"
    # Nguyen-1 = x^3 + x^2 + x sampled on [-1, 1] (raw is ast.unparse-formatted from the DSO source)
    eq = bench._entries["Nguyen-1"]
    assert eq["raw"] == "x1 ** 3 + x1 ** 2 + x1"
    assert eq["vars"]["v1"]["sample_range"] == [-1.0, 1.0]


def test_feynman_known_equation(benches):
    bench = benches["feynman"]
    assert "I.6.2a" in bench.equation_ids()
    eq = bench._entries["I.6.2a"]
    assert eq["raw"] == "exp(-theta**2/2)/sqrt(2*pi)"
    assert eq["vars"]["v1"]["name"] == "theta"


# --------------------------------------------------------------------------------------------------
# Faithfulness oracle: simplipy(prepared) == sympy(raw) on shared inputs. Offline (package data only).
# Sympy-gated; the integrity tests above run without sympy. Only the converted specs are checked.
# --------------------------------------------------------------------------------------------------
try:
    import sympy  # noqa: F401
    _HAS_SYMPY = True
except ImportError:  # pragma: no cover - sympy is in the [dev] extra
    _HAS_SYMPY = False


@pytest.mark.skipif(not _HAS_SYMPY, reason="sympy not installed (install symbolic-data[dev])")
@pytest.mark.parametrize("name", GATED)
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
