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
            "sine", "vladislavleva", "nguyen", "fastsrb", "feynman"]


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

    clusters = {f"eq{i:04d}": sorted(members)
                for i, ((skel, n), members) in enumerate(sorted(groups.items(), key=lambda kv: kv[1][0]))
                if len(members) > 1}
    out = {"identity": "normalized skeleton (dev_7-3, literals masked) + n_variables",
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
