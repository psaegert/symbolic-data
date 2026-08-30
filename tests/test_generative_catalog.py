

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
