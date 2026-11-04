"""The next srbf release's problem count, read from the catalogs srbf will run.

The totals come from the built catalogs and their frozen subsamples (srbf ruling 2026-11-04). The release is:

- the curated catalogs in assets/catalogs and assets_sa/catalogs (YAML and the pinned-point .npz companions) except the
  generative ones, FastSRB at its default version, and ERBench SynEq and OEIS at their frozen 1,024;
- the older data catalogs (.npz) in assets/catalogs only where a newer build replaces none (SUPERSEDED);
- the data catalogs, built outside git (--catalogs, default build/data-catalogs: its *.npz, or data/*.npz and
  srbench-*.npz) and published to Hugging Face only.

A problem's subset is its entry's ``subset``, else release_subsets.json's tiers (FastSRB, SRSD), else ``all``. Subsets
with fewer than 20 problems in a catalog with several subsets are listed (the floor was lifted 2026-11-04: a benchmark
keeps the subsets its authors define).

    count_release.py [--symbolic-data <repo>] [--catalogs <locally built data catalogs>] [--out <json>]
"""
from __future__ import annotations

import argparse
import sys
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths  # noqa: E402  the curated inputs
GENERATIVE = {"lample-charton-v24", "lample-charton-v24-bench"}
VERSIONED = {"fastsrb.yaml": "fastsrb.v2.yaml"}              # publish_catalogs.MULTI_VERSION: the default version only
FROZEN = {"erbench-syneq": "release__erbench-syneq.json", "erbench-oeis": "erbench-oeis.json"}
# symbolic-data's frozen data catalogs that a local build replaces (the 2026-10-17 drops: AI Descartes FSRD_noise,
# SRBench 2.0's supernovae_zg; CP3-bench as one catalog)
SUPERSEDED = {"ai-descartes.npz": "ai-descartes-real-data.npz", "cp3-blackbox.npz": "cp3-bench.npz",
              "cp3-cosmo.npz": "cp3-bench.npz", "first-principles.npz": "srbench-2.0-first-principles.npz",
              "physo-streams.npz": "physo-milky-way-streams.npz", "srbench2-blackbox.npz": "srbench-2.0-blackbox.npz"}
FLOOR = 20


def tiers() -> dict[str, dict[str, str]]:
    rel = paths.release_subsets()
    return {cat: {str(k): s["name"] for s in rel[cat].get("subsets") or [] for k in s.get("keys") or []}
            for cat in ("fastsrb", "srsd-dummy") if cat in rel}


def from_yaml(path: Path, tier: dict[str, dict[str, str]]) -> dict[str, Any]:
    doc = yaml.safe_load(open(path))
    name = doc["metadata"]["name"]
    keep = set(json.load(open(paths.frozen_path(FROZEN[name])))["ids"]) if name in FROZEN else None
    subsets: Counter[str] = Counter()
    for key, e in doc["expressions"].items():
        if keep is not None and str(key) not in keep:
            continue
        subsets[e.get("subset") or tier.get(name, {}).get(str(key), "all")] += 1
    md = doc["metadata"]
    return {"name": name, "kind": "formula", "file": str(path), "version": md.get("version"),
            "source_record": md.get("source_record"),
            "display_name": md.get("display_name") or name, "of": len(doc["expressions"]) if keep is not None else None,
            "subsets": dict(subsets)}


def from_npz(path: Path) -> dict[str, Any]:
    meta = json.loads(str(np.load(path)["_meta"]))
    cat = meta["catalog"]
    subsets = Counter((p.get("meta") or {}).get("subset") or "all" for p in meta["problems"])
    return {"name": cat["name"], "kind": "data", "file": str(path), "version": cat.get("version"),
            "source_record": (cat.get("meta") or {}).get("source_record"),
            "display_name": (cat.get("meta") or {}).get("display_name") or cat["name"], "subsets": dict(subsets)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbolic-data", default=str(paths.REPO))
    ap.add_argument("--catalogs", default=str(paths.DATA_BUILD), help="the built data catalogs (*.npz), if any")
    ap.add_argument("--out", default=str(paths.REPO / "build/release_counts.json"))
    a = ap.parse_args()
    sd = Path(a.symbolic_data)
    tier = tiers()
    rows = []
    for folder in (sd / "assets/catalogs", sd / "assets_sa/catalogs"):
        for path in sorted(folder.glob("*.yaml")):
            if path.stem in GENERATIVE or path.name in VERSIONED.values():
                continue
            rows.append({**from_yaml(folder / VERSIONED.get(path.name, path.name), tier), "source": "symbolic-data"})
        for path in sorted(folder.glob("*.npz")):
            if path.name not in SUPERSEDED:
                rows.append({**from_npz(path), "source": "symbolic-data"})
    local = Path(a.catalogs)
    for path in sorted([*local.glob("*.npz"), *(local / "data").glob("*.npz")]):
        rows.append({**from_npz(path), "source": "data build"})
    names = Counter(r["name"] for r in rows)
    clash = sorted(n for n, c in names.items() if c > 1)   # srbf needs unique names: reported, both counted
    for r in rows:
        r["problems"] = sum(r["subsets"].values())
    empty = sorted(r["file"] for r in rows if not r["problems"])   # a build that kept no problem is not in the release
    rows = [r for r in rows if r["problems"]]
    small = sorted((r["name"], s, n) for r in rows if len(r["subsets"]) > 1 for s, n in r["subsets"].items() if n < FLOOR)
    by_kind = Counter()
    for r in rows:
        by_kind[r["kind"]] += r["problems"]
    total = sum(r["problems"] for r in rows)
    report = {"rule": __doc__.split("\n\n")[1].strip(), "catalogs": len(rows), "problems": total,
              "subsets": sum(len(r["subsets"]) for r in rows), "problems_by_kind": dict(by_kind),
              "subsets_below_floor": [{"catalog": c, "subset": s, "problems": n} for c, s, n in small],
              "name_clashes": {n: [r["file"] for r in rows if r["name"] == n] for n in clash}, "empty_catalogs": empty,
              "superseded_in_symbolic_data": SUPERSEDED, "rows": sorted(rows, key=lambda r: -r["problems"])}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(report, indent=1))
    print(f"{len(rows)} catalogs, {total:,} problems ({', '.join(f'{k} {v:,}' for k, v in by_kind.items())}), "
          f"{report['subsets']} subsets ({len(small)} below {FLOOR}) -> {a.out}")
    if clash:
        print(f"name clashes (srbf needs unique catalog names): {clash}")


if __name__ == "__main__":
    main()
