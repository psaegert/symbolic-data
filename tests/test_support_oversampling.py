"""Domain-aware support oversampling (v24): a failed try doubles the per-try draw and
keeps the first ``n_support`` in-domain rows, so a domain-restricted expression no
longer needs EVERY row of a draw to land inside its domain at once. The default
(``support_oversampling_max`` absent or 1) keeps whole-draw semantics; its rng
stream is pinned below (re-captured 2026-08-23 for the vectorized support draw)."""
import hashlib
from pathlib import Path

import numpy as np
import pytest
import yaml

from simplipy import SimpliPyEngine
from simplipy.utils import codify

from symbolic_data.errors import NoValidSampleFoundError
from symbolic_data.generative import LampleChartonCatalog
from symbolic_data.prior_factory import build_prior_callable

CONFIG = Path(__file__).resolve().parent.parent / "configs" / "test" / "catalog_train.yaml"

# sha256(x || y || literals), first 16 hex digits, of sample_data draws captured on the
# try-batched box draw + f64 evaluation (2026-08-23): the default path must consume the
# rng identically. (History: 2026-08-22 pinned the pre-oversampling per-column loop;
# then, each deliberately: the vectorized per-column-params draw changed the stream,
# f64 end-to-end evaluation changed y's low bits at the f32 boundary cast, and
# try-batching now draws every box's parameters and probe block up front.)
GOLDEN_DRAWS = {7: "99fecd30e20cc50f", 20260822: "27ddc7083a9101c0", 424242: "5f96eb693231aa09"}

# A fixed symmetric support prior. The test config's meta-sampler draws its own
# low/high per call, which sometimes lands entirely positive -- that would let the
# whole-draw path succeed by luck and un-red the starvation test.
SYMMETRIC_SUPPORT = build_prior_callable({"name": "uniform", "kwargs": {"low": -30.0, "high": 30.0}})


@pytest.fixture(scope="module")
def engine() -> SimpliPyEngine:
    return SimpliPyEngine.load("acj-4", install=True)


def _catalog(engine: SimpliPyEngine, **sample_strategy_overrides):
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    catalog = LampleChartonCatalog(
        simplipy_engine=engine,
        sample_strategy={**cfg["sample_strategy"], "max_tries": 64, **sample_strategy_overrides},
        literal_prior=cfg["literal_prior"],
        variables=cfg["variables"],
        support_sampler_config=cfg["support_sampler"],
        typed_slots=cfg["typed_slots"],
        operator_weights=cfg["operator_weights"],
        allow_nan=False,
        simplify=True,
        decontaminate=False,
    )
    return catalog, cfg


def test_the_default_path_rng_stream_is_pinned(engine: SimpliPyEngine) -> None:
    catalog, cfg = _catalog(engine)
    code = codify("x1 * c0 + c1", cfg["variables"] + ["c0", "c1"])
    for seed, expected in GOLDEN_DRAWS.items():
        x, y, literals = catalog.sample_data(code, n_constants=2, n_support=48, rng=np.random.default_rng(seed))
        digest = hashlib.sha256(x.tobytes() + y.tobytes() + np.asarray(literals).tobytes()).hexdigest()[:16]
        assert digest == expected, f"seed {seed}: rng consumption of the default path changed"


def test_whole_draw_rejection_starves_a_domain_restricted_expression(engine: SimpliPyEngine) -> None:
    # log(x1) under a symmetric prior: a whole draw of 48 rows is valid only when every
    # row lands positive (P = 2^-48 per try), so 64 tries always fail. This is the
    # night-2 corpus concern in miniature.
    catalog, cfg = _catalog(engine)
    code = codify("np.log(x1)", cfg["variables"])
    with pytest.raises(NoValidSampleFoundError):
        catalog.sample_data(code, n_constants=0, n_support=48,
                            support_prior=SYMMETRIC_SUPPORT, rng=np.random.default_rng(1))


def test_oversampling_recovers_the_same_expression(engine: SimpliPyEngine) -> None:
    catalog, cfg = _catalog(engine, support_oversampling_max=16)
    code = codify("np.log(x1)", cfg["variables"])
    x, y, literals = catalog.sample_data(code, n_constants=0, n_support=48,
                                         support_prior=SYMMETRIC_SUPPORT, rng=np.random.default_rng(1))
    assert x.shape == (48, len(cfg["variables"]))
    assert y.shape == (48, 1)
    assert literals.shape == (0,)
    assert np.isfinite(x).all()
    assert np.isfinite(y).all()
    # The kept rows are exactly the in-domain ones, in draw order.
    assert (x[:, 0] > 0).all()
