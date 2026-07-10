"""Cross-suite equivalence registry (P1 close-out).

Canonical identity = normalized skeleton (dev_7-3 parse of `prepared`, literals masked) +
n_variables. Suites keep their entries (source separation; citable as published); the registry
records which entries are the same underlying problem so (a) the v24 union holdout deduplicates
and (b) cross-suite analyses can collapse duplicates.
"""
from __future__ import annotations

import json
import warnings
from collections import defaultdict
from pathlib import Path

import yaml

from simplipy import SimpliPyEngine, normalize_skeleton

CATALOGS = ["constant", "grammarvae", "jin", "keijzer", "korns", "koza", "livermore",
            "livermore2", "meier", "neat", "nonic", "pagie", "poly", "r-rationals",
            "sine", "vladislavleva", "nguyen", "fastsrb", "feynman", "srsd-dummy"]
# FROZEN catalogs (materialized .npz, measured data): identity from each problem's stored
# skeleton (already normalize_skeleton-canonical) + its x width.
FROZEN_CATALOGS = ["first-principles", "cp3-cosmo", "ai-descartes"]


def main() -> None:
    engine = SimpliPyEngine.load("dev_7-3", install=True)
    groups: dict[tuple, list[str]] = defaultdict(list)
    for name in CATALOGS:
        path = Path("assets/catalogs") / f"{name}.yaml"
        if not path.exists():
            print(f"skip {name} (no local yaml)")
            continue
        cfg = yaml.safe_load(path.read_text())
        for eq_id, entry in cfg["expressions"].items():
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    prefix = engine.parse(entry["prepared"], mask_numbers=True)
                    skel = tuple(normalize_skeleton([str(t) for t in prefix]))
            except Exception as exc:
                skel = ("UNPARSED", str(exc)[:40])
            groups[(skel, int(entry.get("n_variables", 0)))].append(f"{name}:{eq_id}")

    for name in FROZEN_CATALOGS:
        path = Path("assets/catalogs") / f"{name}.npz"
        if not path.exists():
            print(f"skip {name} (no local npz)")
            continue
        from symbolic_data import ProblemCatalog
        for problem in ProblemCatalog.from_npz(path).problems or []:
            # identity must live in the SAME space as the yaml branch (parse of a prepared infix,
            # literals masked): the stored problem.skeleton comes from compile_expression's
            # simplify pass, which reassociates constants differently from engine.parse -- keying
            # on it silently under-clusters equivalent entries. meta.prepared_infix (written by
            # frozen builders) goes through the identical pipeline as yaml `prepared`.
            prepared = (problem.meta or {}).get("prepared_infix")
            if prepared:
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        prefix = engine.parse(prepared, mask_numbers=True)
                        skel = tuple(normalize_skeleton([str(t) for t in prefix]))
                except Exception as exc:
                    skel = ("UNPARSED", str(exc)[:40])
            elif problem.skeleton is not None:
                skel = tuple(problem.skeleton)      # fallback: stored-skeleton space (may under-cluster)
            else:
                continue           # black-box: nothing to identify (and nothing to hold out)
            groups[(skel, int(problem.x_support.shape[1]))].append(f"{name}:{problem.eq_id}")

    clusters = {f"eq{i:04d}": sorted(members)
                for i, ((skel, n), members) in enumerate(sorted(groups.items(), key=lambda kv: kv[1][0]))
                if len(members) > 1}

    # Explicit provenance links (meta.base_catalog/base_eq_id): variant catalogs that renumber
    # variables (e.g. srsd-dummy's dummy insertions) defeat index-sensitive skeleton identity,
    # so their 1:1 base links are recorded as explicit equivalences instead of inferred ones.
    explicit = {}
    for name in CATALOGS:
        path = Path("assets/catalogs") / f"{name}.yaml"
        if not path.exists():
            continue
        cfg = yaml.safe_load(path.read_text())
        for eq_id, entry in cfg["expressions"].items():
            meta = entry.get("meta") or {}
            base_cat, base_eq = meta.get("base_catalog"), meta.get("base_eq_id")
            if base_cat and base_eq:
                explicit[f"{name}:{eq_id}"] = f"{base_cat.split('@')[0]}:{base_eq}"
    for name in FROZEN_CATALOGS:
        path = Path("assets/catalogs") / f"{name}.npz"
        if not path.exists():
            continue
        from symbolic_data import ProblemCatalog
        for problem in ProblemCatalog.from_npz(path).problems or []:
            meta = problem.meta or {}
            base_cat, base_eq = meta.get("base_catalog"), meta.get("base_eq_id")
            if base_cat and base_eq:
                explicit[f"{name}:{problem.eq_id}"] = f"{base_cat.split('@')[0]}:{base_eq}"
    out = {"identity": "normalized skeleton (dev_7-3, literals masked) + n_variables; "
                       "PLUS explicit meta.base_eq_id links (variable-index-insensitive)",
           "explicit_variant_links": explicit,
           "known_limit": "skeleton identity is variable-index-sensitive (canonicalization of "
                          "variable order is a deferred decision); explicit links cover variant "
                          "catalogs, and every rendering joins the union holdout independently",
           "n_entries_total": sum(len(m) for m in groups.values()),
           "n_unique_problems": len(groups),
           "n_cross_or_intra_duplicate_clusters": len(clusters),
           "clusters": clusters}
    Path("assets/catalogs/DEDUP_REGISTRY.json").write_text(json.dumps(out, indent=1))
    print(f"entries={out['n_entries_total']} unique={out['n_unique_problems']} "
          f"duplicate clusters={len(clusters)}")
    for cid, members in list(clusters.items())[:12]:
        print(f"  {cid}: {members}")


if __name__ == "__main__":
    main()
