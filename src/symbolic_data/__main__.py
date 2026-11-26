"""Command-line interface for the symbolic-data data layer.

Three commands:

* ``materialize`` -- sample a :class:`~symbolic_data.source.ProblemSource` (from a config) once and
  FREEZE it to a versioned catalog (``.npz``), the shareable + exactly-reproducible form. The
  config is a ProblemSource spec (a ``catalog`` ref, a ``generator`` block, or inline ``problems``,
  plus a ``sampling`` block).
* ``render-draws`` -- render catalogs' frozen draws (:mod:`symbolic_data.draws`): every problem's data
  drawn once, with a manifest of sha256s.
* ``verify-draws`` -- check a rendered root against its manifest and, for the catalogs named, that a
  second render writes the same bytes.

Invoke as ``symbolic-data materialize ...`` (console script) or ``python -m symbolic_data ...``.

(The pre-0.4.1 generate / import-data / split data-CLI commands were removed: catalog generation is
now ProblemSource's private engine, curated test sets are HF artifacts (resolved by name, not
bundled), and decontamination is a ProblemSource ``holdouts: [{exclude: <catalog>}]``.)
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Sequence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="symbolic-data",
        description="symbolic-data: model-agnostic symbolic-regression data layer (Problem / ProblemCatalog / ProblemSource).",
    )
    subparsers = parser.add_subparsers(dest="command_name", required=True)

    materialize = subparsers.add_parser(
        "materialize",
        help="Sample a ProblemSource (from a config) once and freeze it to a catalog (.npz).",
    )
    materialize.add_argument("-c", "--config", type=str, required=True, help="ProblemSource config (yaml): a catalog/generator/problems spec + sampling.")
    materialize.add_argument("-o", "--output", type=str, required=True, help="Output catalog path (a .npz frozen catalog).")
    materialize.add_argument("-n", "--n", type=int, default=None, help="Cap the number of problems (required for an unbounded generator without `size`).")
    materialize.add_argument("--name", type=str, default=None, help="Catalog name stamped in metadata.")

    from symbolic_data.draws import DEFAULT_ENGINE, DEFAULT_REPEATS, DEFAULT_SEED_ROOT

    render = subparsers.add_parser("render-draws", help="Render catalogs' frozen draws: every problem's data drawn once.")
    render.add_argument("catalogs", nargs="+", help="Catalog references (local paths or name[@version]).")
    render.add_argument("-o", "--output", type=str, required=True, help="The root to render into (it gets a manifest.json).")
    render.add_argument("--repeats", type=int, nargs="+", default=list(DEFAULT_REPEATS), help="The repeats to render (default: 1, one draw per problem).")
    render.add_argument("--n-support", type=int, default=512, help="Fitting points per formula problem; the most fitting rows of a data problem.")
    render.add_argument("--n-validation", type=int, default=512, help="Held-out points per formula problem; the most test rows of a data problem.")
    render.add_argument("--ids", action="append", default=[], metavar="CATALOG=FILE",
                        help="Render only these problems of CATALOG: FILE is a JSON list of ids or a mapping with an `ids` list. Repeatable.")
    render.add_argument("--seed-root", type=str, default=DEFAULT_SEED_ROOT, help="The string every problem's seed is derived from.")
    render.add_argument("--simplipy-engine", type=str, default=DEFAULT_ENGINE, help="The engine the ground truth is compiled with.")
    render.add_argument("--jobs", type=int, default=1, help="Catalogs rendered at once.")
    render.add_argument("--update", action="store_true", help="Add the catalogs to a root that holds draws (same settings).")

    verify = subparsers.add_parser("verify-draws", help="Check a rendered root; with catalogs, render them again and compare.")
    verify.add_argument("root", help="The rendered root (holding manifest.json).")
    verify.add_argument("catalogs", nargs="*", help="Catalogs to render again; their bytes must match the manifest.")
    verify.add_argument("--ids", action="append", default=[], metavar="CATALOG=FILE", help="As for render-draws.")
    verify.add_argument("--jobs", type=int, default=1, help="Catalogs rendered at once.")

    return parser


def _materialize(args: argparse.Namespace) -> None:
    from symbolic_data.source import ProblemSource

    source = ProblemSource.from_config(args.config)
    catalog = source.to_catalog(name=args.name, n=args.n)
    path = catalog.save(args.output)
    print(f"Materialized {len(catalog.problems or [])} problems -> {path}")


def _ids(pairs: Sequence[str]) -> dict[str, list[str]]:
    """``CATALOG=FILE`` pairs -> the id list of each catalog."""
    out: dict[str, list[str]] = {}
    for pair in pairs:
        catalog, sep, path = pair.partition("=")
        if not sep or not catalog or not path:
            raise SystemExit(f"--ids expects CATALOG=FILE, got {pair!r}")
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
        ids = payload["ids"] if isinstance(payload, dict) else payload
        out[catalog] = [str(i) for i in ids]
    return out


def _render_draws(args: argparse.Namespace) -> None:
    from symbolic_data.draws import DrawsSettings, render

    settings = DrawsSettings(n_support=args.n_support, n_validation=args.n_validation, repeats=tuple(args.repeats),
                             seed_root=args.seed_root, simplipy_engine=args.simplipy_engine)
    manifest = render(args.catalogs, args.output, settings, ids=_ids(args.ids), jobs=args.jobs, update=args.update)
    print(f"Rendered {len(manifest['catalogs'])} catalog(s) -> {args.output}/manifest.json")


def _verify_draws(args: argparse.Namespace) -> None:
    from symbolic_data.draws import verify

    problems = verify(args.root, args.catalogs, ids=_ids(args.ids), jobs=args.jobs)
    for line in problems:
        print(line)
    if problems:
        sys.exit(1)
    print(f"{args.root}: sound{' and reproduced' if args.catalogs else ''}")


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    match args.command_name:
        case "materialize":
            _materialize(args)
        case "render-draws":
            _render_draws(args)
        case "verify-draws":
            _verify_draws(args)


if __name__ == "__main__":
    main()
