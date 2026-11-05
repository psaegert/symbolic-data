"""ERBench's five catalogs on ERBench's own generator, and the re-frozen SynEq and OEIS subsets.

srbf ruling 2026-11-03: inputs from ERBench's own spec (call (a)); one problem per formula and alpha, re-frozen over formula x
alpha (call 1 after the briefing); dummy variables come from the benchmark (ruling 4). ERBench (Kahlmeyer et al. 2026;
src/benchmark.py ``generate_dataset``, ``generate_problem``, ``get_samples``) makes three regression problems of every
formula, alpha in {0, 0.5, 1}: the inputs are drawn from a 1-3 part mixture per variable, the fitting rows in a window
half the support's width placed by alpha, the test rows on the rest of the support, and 0 or 1 dummy column is added
(drawn per problem). symbolic-data's ``erbench_mixture`` sampler draws exactly that; srbf sets the number of points and
the noise (ruling 4), so ERBench's 100 points per dimension and its noise draw are not taken over, and srbf's per-point
rejection replaces ERBench's resampling of invalid targets.

Each catalog is rebuilt from its published version 1 (the base ref), so the script can run any number of times:

    build_erbench_catalogs.py --symbolic-data <worktree> [--base origin/feat/data-catalog-pools] [--check 32]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import freeze as F  # noqa: E402  the 1,024-problem subsamples
import paths  # noqa: E402  the curated inputs
CATALOGS = {"erbench-syneq": "assets/catalogs", "erbench-phybench": "assets/catalogs",
            "erbench-densities": "assets/catalogs", "erbench-eponymous": "assets_sa/catalogs",
            "erbench-oeis": "assets_sa/catalogs"}
ALPHAS = ("0", "0.5", "1")
# subsets above 1024 problems get one frozen subsample (ruling 5); the set names keep the earlier salts' form
FROZEN = {"erbench-syneq": "release/erbench-syneq", "erbench-oeis": "erbench-oeis"}
SAMPLING = ("ERBench's own generator (src/benchmark.py generate_problem / get_samples; srbf ruling 2026-11-03): one problem per "
            "formula and alpha in {0, 0.5, 1}; every variable a fresh 1-3 part mixture (uniform sub-interval or normal "
            "with sigma = distance to the nearer edge / 3; integers: uniform-integer or binomial), clipped to the "
            "support; fitting rows in the window [a1, a1 + (b - a)/2], a1 = a + alpha (b - a)/2; test rows on the rest "
            "of the support, split by length; 0 or 1 dummy column per problem (drawn once, seeded), its values drawn "
            "from the point set's own values. srbf sets the number of points and the noise (ruling 4); invalid rows are "
            "redrawn (per-point rejection) where ERBench resamples invalid targets.")


def dummies_of(catalog: str, eq_id: str) -> int:
    """ERBench draws ``randint(0, 2)`` dummy columns per problem: drawn once here, seeded by the problem's identity."""
    return hashlib.sha256(f"erbench-dummy:{catalog}:{eq_id}".encode()).digest()[0] & 1


def convert(catalog: str, doc: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for key, e in doc["expressions"].items():
        keys = sorted((k for k in e["vars"] if k != "v0"), key=lambda k: int(k[1:]))
        supports = [[float(v) for v in e["vars"][k]["sample_range"]] for k in keys]
        bases = {e["vars"][k]["sample_type"][0] for k in keys}
        dtype = "int" if bases == {"int"} else "float"
        for alpha in ALPHAS:
            eq_id = f"{key}_alpha{alpha}"
            d = dummies_of(catalog, eq_id)
            variables = {k: {kk: vv for kk, vv in e["vars"][k].items() if kk not in ("sample_range", "sample_type")} for k in keys}
            if d:
                variables[f"v{len(keys) + 1}"] = {"name": "dummy"}
            variables["v0"] = e["vars"].get("v0", {"name": "y"})
            kwargs = {"supports": supports, "alpha": float(alpha), "dtype": dtype, "dummies": d}
            meta = {k: v for k, v in (e.get("meta") or {}).items() if k not in ("finite_fraction", "meta")}
            entry = {k: v for k, v in e.items() if k not in ("vars", "meta", "n_variables")}
            out[eq_id] = {**entry, "n_variables": len(keys) + d, "vars": variables,
                          "inputs": {"name": "erbench_mixture", "kwargs": {**kwargs, "region": "fit"}},
                          "test": {"inputs": {"name": "erbench_mixture", "kwargs": {**kwargs, "region": "test"}}},
                          "meta": {**meta, "erbench_formula": str(key), "alpha": float(alpha), "dummies": d}}
    md = dict(doc["metadata"])
    md["version"] = int(md["version"]) + 1
    md["description"] = (md.get("description") or "").rstrip(".") + "; one problem per formula and alpha (ERBench's generator)."
    md["conventions"] = {**(md.get("conventions") or {}), "sampling": SAMPLING}
    md["sampling_defaults"] = {**(md.get("sampling_defaults") or {}), "n_points": 512}
    return {"metadata": md, "expressions": out}


def check(catalog: str, doc: dict[str, Any], n: int) -> dict[str, Any]:
    """Draw ``n`` fitting and test rows of every entry; report failures and constant targets."""
    from symbolic_data._evaluation import load_engine
    from symbolic_data.catalog import ProblemCatalog
    engine = load_engine(None)
    cat = ProblemCatalog.from_yaml(doc)
    rng = np.random.default_rng(0)
    failed, constant = {}, []
    for eq_id, entry in cat.entries.items():
        try:
            fit = cat.realize(entry, n, rng, engine=engine)
            cat.realize(entry, n, rng, engine=engine, part="test")
            if np.ptp(fit.y) == 0:
                constant.append(eq_id)
        except Exception as exc:
            failed[eq_id] = f"{type(exc).__name__}: {exc}"[:200]
    return {"failed": failed, "constant_target": constant}


def freeze(catalog: str, doc: dict[str, Any]) -> dict[str, Any]:
    ids = list(doc["expressions"])
    salt, take, sizes, frozen = F.draw(FROZEN[catalog], "all", [(i, "all") for i in ids])
    return {"set": FROZEN[catalog], "name": f"{catalog} (ERBench, formula x alpha)", "subset": "all",
            "source_file": f"symbolic-data {CATALOGS[catalog]}/{catalog}.yaml v{doc['metadata']['version']}",
            "n_source": len(ids), "k": F.K, "salt": salt, "groups": {g: [sizes[g], take[g]] for g in sizes},
            "ids": frozen, "rule": "srbf ruling 2026-11-03: re-frozen over formula x alpha (supersedes the 2026-10-05 list)"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbolic-data", default=str(paths.REPO))
    ap.add_argument("--base", default="origin/feat/data-catalog-pools")
    ap.add_argument("--check", type=int, default=32)
    a = ap.parse_args()
    root = Path(a.symbolic_data)
    report = {}
    for catalog, where in CATALOGS.items():
        text = subprocess.run(["git", "-C", str(root), "show", f"{a.base}:{where}/{catalog}.yaml"],
                              capture_output=True, text=True, check=True).stdout
        doc = convert(catalog, yaml.safe_load(text))
        (root / where / f"{catalog}.yaml").write_text(yaml.safe_dump(doc, sort_keys=False, allow_unicode=True, width=120))
        rec = {"formulas": len(yaml.safe_load(text)["expressions"]), "problems": len(doc["expressions"]),
               "with_dummy": sum(e["meta"]["dummies"] for e in doc["expressions"].values())}
        if a.check:
            rec.update(check(catalog, doc, a.check))
        if catalog in FROZEN:
            frozen = freeze(catalog, doc)
            target = paths.frozen_path(f"{FROZEN[catalog].replace('/', '__')}.json")
            target.write_text(json.dumps(frozen, indent=1) + "\n")
            rec["frozen"] = {"k": len(frozen["ids"]), "formulas": len({i.rsplit("_alpha", 1)[0] for i in frozen["ids"]}),
                             "file": str(target.relative_to(paths.PREP.parent))}
        report[catalog] = rec
        print(catalog, {k: (v if not isinstance(v, (dict, list)) else len(v)) for k, v in rec.items()}, flush=True)
    paths.frozen_path("ERBENCH_2026-11-03.json").write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
