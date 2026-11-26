"""The frozen draws of srbf's release: every problem of every release catalog drawn once (``symbolic_data.draws``).

The catalogs are the release's, by count_release.py's rules: the curated formula catalogs (YAML; FastSRB at its
default version, the generative recipes left out), ERBench SynEq and OEIS on their frozen 1,024 problems, the
pinned-point ``.npz`` companions not superseded by a newer build, and the locally built data catalogs. Run it after
the catalogs and the engine are final; a second run with the same software writes the same bytes, which
``symbolic-data verify-draws <root> <catalogs>`` checks.

    render_draws.py --out <root> [--formula-only] [--catalogs build/data-catalogs] [--only <name> ...] [--jobs N]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import count_release  # noqa: E402  the release's catalog rules
import paths  # noqa: E402  the curated inputs


def release_catalogs(symbolic_data: Path, data_build: Path | None, *, formula_only: bool = False) -> list[Path]:
    """The release's catalog files, formula catalogs first."""
    formula: list[Path] = []
    data: list[Path] = []
    for folder in (symbolic_data / "assets/catalogs", symbolic_data / "assets_sa/catalogs"):
        for path in sorted(folder.glob("*.yaml")):
            if path.stem in count_release.GENERATIVE or path.name in count_release.VERSIONED.values():
                continue
            formula.append(folder / count_release.VERSIONED.get(path.name, path.name))
        data.extend(p for p in sorted(folder.glob("*.npz")) if p.name not in count_release.SUPERSEDED)
    if data_build is not None:
        data.extend(sorted([*data_build.glob("*.npz"), *(data_build / "data").glob("*.npz")]))
    return formula if formula_only else formula + data


def frozen_ids() -> dict[str, list[str]]:
    """The frozen 1,024-problem subsamples srbf evaluates, by catalog name."""
    out = {}
    for name, file in count_release.FROZEN.items():
        with open(paths.frozen_path(file), encoding="utf-8") as handle:
            out[name] = [str(i) for i in json.load(handle)["ids"]]
    return out


def main() -> None:
    from symbolic_data.draws import DrawsSettings, render

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True, help="the root to render into")
    ap.add_argument("--symbolic-data", default=str(paths.REPO))
    ap.add_argument("--catalogs", default=str(paths.DATA_BUILD), help="the built data catalogs (*.npz), if any")
    ap.add_argument("--formula-only", action="store_true", help="the formula catalogs alone")
    ap.add_argument("--only", nargs="+", default=None, help="these catalogs (file stems) alone")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--update", action="store_true", help="add to a root that holds draws")
    a = ap.parse_args()
    files = release_catalogs(Path(a.symbolic_data), Path(a.catalogs), formula_only=a.formula_only)
    if a.only:
        files = [f for f in files if f.stem in set(a.only)]
    manifest = render([str(f) for f in files], a.out, DrawsSettings(), ids=frozen_ids(), jobs=a.jobs, update=a.update)
    catalogs = manifest["catalogs"].values()
    print(f"{len(manifest['catalogs'])} catalogs, {sum(c['problems'] for c in catalogs):,} problems, "
          f"{sum(f['bytes'] for c in catalogs for f in c['files'].values()) / 1e9:.2f} GB -> {a.out}")


if __name__ == "__main__":
    main()
