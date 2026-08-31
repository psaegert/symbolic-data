

def test_simplify_mode_reaches_the_target_canonicalization(monkeypatch):
    """The target-canon mode is a config choice: data generates FROM the simplified
    skeleton (target == data), so the corpus tier is licensed -- and after the 0.14
    silent-default drift, the mode must provably reach the engine call."""
    import numpy as np
    import yaml
    from symbolic_data.generative import LampleChartonCatalog

    import os
    import pytest
    ref = os.path.expanduser("~/Projects/flash-ansr/configs/v25.0-T1/catalog_train.yaml")
    if not os.path.exists(ref):
        pytest.skip("reference run config not present in this checkout")
    cfg = dict(yaml.safe_load(open(ref)))
    cfg["holdout_pools"] = []
    cfg["simplify_mode"] = "permissive"
    cat = LampleChartonCatalog.from_config(cfg)
    assert cat.simplify_mode == "permissive"
    seen = []
    real_simplify = cat.simplipy_engine.simplify

    def spy(tokens, *args, **kwargs):
        seen.append(kwargs.get("mode"))
        return real_simplify(tokens, *args, **kwargs)

    monkeypatch.setattr(cat.simplipy_engine, "simplify", spy)
    rng = np.random.default_rng(0)
    for _ in range(3):
        try:
            cat.sample_skeleton(new=True, decontaminate=False, rng=rng)
        except Exception:
            continue
    assert "permissive" in seen, f"target canonicalization never saw the configured mode: {seen}"


def test_simplipy_engine_modes_reaches_the_loader(monkeypatch):
    """`simplipy_engine_modes` is the worker memory profile: it must arrive at
    SimpliPyEngine.load as `modes=`, and its absence must keep the historical
    call (no kwarg at all, so simplipy < 0.14.2 keeps working without the key)."""
    import os

    import pytest
    import yaml

    import symbolic_data.generative as generative

    ref = os.path.expanduser("~/Projects/flash-ansr/configs/v25.0-T1/catalog_train.yaml")
    if not os.path.exists(ref):
        pytest.skip("reference run config not present in this checkout")
    cfg = dict(yaml.safe_load(open(ref)))
    cfg["holdout_pools"] = []

    seen: list[dict] = []
    real_load = generative.SimpliPyEngine.load

    def spy(name, *args, **kwargs):
        seen.append(dict(kwargs))
        kwargs.pop("modes", None)  # delegate on the pre-0.14.2 signature; the spy already recorded it
        return real_load(name, *args, **kwargs)

    monkeypatch.setattr(generative.SimpliPyEngine, "load", staticmethod(spy))

    cfg["simplipy_engine_modes"] = ["f64", "permissive"]
    generative.LampleChartonCatalog.from_config(dict(cfg))
    assert seen and seen[-1].get("modes") == ("f64", "permissive")

    cfg.pop("simplipy_engine_modes")
    generative.LampleChartonCatalog.from_config(dict(cfg))
    assert "modes" not in seen[-1], f"absent key must not send a modes kwarg: {seen[-1]}"


def test_the_fold_skip_is_provably_free(monkeypatch):
    """S1 (task #92): when the target canon IS the holdout canon, the fold's opening
    canonicalization is skipped. Golden proof: same seed => byte-identical skeletons
    and identical family prototypes, with exactly one fewer simplify per holdout check."""
    import os

    import numpy as np
    import pytest
    import yaml

    import symbolic_data.generative as gen

    ref = os.path.expanduser("~/Projects/flash-ansr/configs/v25.0-T1/catalog_train.yaml")
    if not os.path.exists(ref):
        pytest.skip("reference run config not present in this checkout")
    cfg = dict(yaml.safe_load(open(ref)))
    cfg["holdout_pools"] = []
    cfg["simplify_mode"] = "permissive"

    def draws(force_off, n=12):
        cat = gen.LampleChartonCatalog.from_config(dict(cfg))
        assert cat._skeleton_is_holdout_canonical is True
        if force_off:
            real_probe = cat.is_held_out
            monkeypatch.setattr(
                cat, "is_held_out",
                lambda sk, cs, **kw: real_probe(sk, cs, **{**kw, "assume_canonical": False}))
        calls = []
        real_simplify = type(cat.simplipy_engine).simplify
        monkeypatch.setattr(
            type(cat.simplipy_engine), "simplify",
            lambda self, tokens, *a, **kw: calls.append(1) or real_simplify(self, tokens, *a, **kw))
        rng = np.random.default_rng(1234)
        out, protos = [], []
        for _ in range(n):
            try:
                skeleton, _code, constants = cat.sample_skeleton(new=True, decontaminate=True, rng=rng)
            except Exception:
                continue
            out.append((tuple(skeleton), tuple(constants)))
            protos.append(tuple(cat.holdout_family_prototype(list(skeleton)) or ()))
        monkeypatch.undo()
        return out, protos, len(calls)

    fast_out, fast_protos, fast_calls = draws(force_off=False)
    slow_out, slow_protos, slow_calls = draws(force_off=True)

    assert fast_out == slow_out, "the skip changed WHICH skeletons are drawn"
    assert fast_protos == slow_protos, "the skip changed the family KEY"
    assert fast_calls < slow_calls, f"no call was saved ({fast_calls} vs {slow_calls})"


def test_the_fold_skip_key_is_identical_on_canonical_inputs():
    """Direct key identity: for an already-canonical input, the assume_canonical path
    returns the same prototype as the full fold."""
    import os

    import numpy as np
    import pytest
    import yaml

    import symbolic_data.generative as gen

    ref = os.path.expanduser("~/Projects/flash-ansr/configs/v25.0-T1/catalog_train.yaml")
    if not os.path.exists(ref):
        pytest.skip("reference run config not present in this checkout")
    cfg = dict(yaml.safe_load(open(ref)))
    cfg["holdout_pools"] = []
    cfg["simplify_mode"] = "permissive"
    cat = gen.LampleChartonCatalog.from_config(cfg)
    rng = np.random.default_rng(99)
    checked = 0
    for _ in range(20):
        try:
            skeleton, _code, _constants = cat.sample_skeleton(new=True, decontaminate=False, rng=rng)
        except Exception:
            continue
        canonical = cat.simplipy_engine.simplify(
            list(skeleton), mode=gen.HOLDOUT_SIMPLIFY_MODE, effort=gen.HOLDOUT_EFFORT)
        assert (cat.holdout_family_prototype(canonical, assume_canonical=True)
                == cat.holdout_family_prototype(canonical))
        checked += 1
    assert checked >= 5
