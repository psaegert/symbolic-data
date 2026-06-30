"""Command-line interface for the symbolic-data data layer.

One command:

* ``materialize`` -- sample a :class:`~symbolic_data.source.ProblemSource` (from a config) once and
  FREEZE it to a versioned catalog (``.npz``), the shareable + exactly-reproducible form. The
  config is a ProblemSource spec (a ``catalog`` ref, a ``generator`` block, or inline ``problems``,
  plus a ``sampling`` block).

Invoke as ``symbolic-data materialize ...`` (console script) or ``python -m symbolic_data ...``.

(The pre-0.4.1 skeleton-pool commands -- generate-skeleton-pool / import-data / split-skeleton-pool
-- were removed: skeleton pools are now ProblemSource's private engine, curated test sets are HF
artifacts (resolved by name, not bundled), and decontamination is a ProblemSource
``holdouts: [{exclude: <catalog>}]``.)
"""
from __future__ import annotations

import argparse
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

    return parser


def _materialize(args: argparse.Namespace) -> None:
    from symbolic_data.source import ProblemSource

    source = ProblemSource.from_config(args.config)
    catalog = source.to_catalog(name=args.name, n=args.n)
    path = catalog.save(args.output)
    print(f"Materialized {len(catalog.problems or [])} problems -> {path}")


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    match args.command_name:
        case "materialize":
            _materialize(args)


if __name__ == "__main__":
    main()
