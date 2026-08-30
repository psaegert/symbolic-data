

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
