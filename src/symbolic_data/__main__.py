"""Command-line interface for the symbolic-data data layer.

Replaces the skeleton-pool data CLI that flash-ansr shipped before 0.7 (when the data layer was
carved out). Three commands:

* ``generate-skeleton-pool`` -- sample a canonical skeleton pool from a config.
* ``import-data``            -- ingest a raw benchmark spec (CSV/YAML) into a (holdout) skeleton
                                pool. Needs the ``[ingest]`` extra (pandas).
* ``split-skeleton-pool``    -- split a pool into train/val subsets.

Invoke as ``symbolic-data <command>`` (console script) or ``python -m symbolic_data <command>``.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from typing import Sequence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="symbolic-data",
        description="symbolic-data: model-agnostic symbolic-regression data layer (skeleton pools + benchmark ingest).",
    )
    subparsers = parser.add_subparsers(dest="command_name", required=True)

    generate = subparsers.add_parser("generate-skeleton-pool", help="Sample a skeleton pool from a config.")
    generate.add_argument("-s", "--size", type=int, required=True, help="Number of skeletons to sample.")
    generate.add_argument("-o", "--output-dir", type=str, required=True, help="Path to the output directory.")
    generate.add_argument("-c", "--config", type=str, required=True, help="Path to the skeleton-pool configuration file.")
    generate.add_argument("-v", "--verbose", action="store_true", help="Print a progress bar.")
    generate.add_argument("--output-reference", type=str, default="relative", help="Reference type for saved config paths.")
    generate.add_argument("--no-output-recursive", dest="output_recursive", action="store_false", default=True, help="Do not recursively save referenced configs.")

    import_data = subparsers.add_parser("import-data", help="Ingest a raw benchmark spec (CSV/YAML) into a skeleton pool (needs the [ingest] extra).")
    import_data.add_argument("-i", "--input", type=str, required=True, help="Path to the benchmark file (CSV or YAML).")
    import_data.add_argument("-b", "--base-skeleton-pool", type=str, required=True, help="Path to a base skeleton-pool config to import into.")
    import_data.add_argument("-p", "--parser", type=str, required=True, choices=["soose", "feynman", "nguyen", "fastsrb"], help="Which benchmark parser to use.")
    import_data.add_argument("-e", "--simplipy-engine", type=str, required=True, help="SimpliPy engine name/path for parsing and simplifying.")
    import_data.add_argument("-o", "--output-dir", type=str, required=True, help="Path to the output directory.")
    import_data.add_argument("-v", "--verbose", action="store_true", help="Print a progress bar.")

    split = subparsers.add_parser("split-skeleton-pool", help="Split a skeleton pool into train/val subsets.")
    split.add_argument("-i", "--input", type=str, required=True, help="Path to the skeleton pool to split.")
    split.add_argument("-t", "--train-size", type=float, default=0.8, help="Fraction of skeletons in the training split.")
    split.add_argument("-r", "--random-state", type=int, default=None, help="Random seed for the split.")
    split.add_argument("-v", "--verbose", action="store_true", help="Print progress information.")

    return parser


def _generate_skeleton_pool(args: argparse.Namespace) -> None:
    from symbolic_data import SkeletonPool

    if args.verbose:
        print(f"Generating skeleton pool from {args.config}")
    skeleton_pool = SkeletonPool.from_config(args.config)
    skeleton_pool.create(size=args.size, verbose=args.verbose)

    if args.verbose:
        print(f"Saving skeleton pool to {args.output_dir}")
    skeleton_pool.save(directory=args.output_dir, config=args.config, reference=args.output_reference, recursive=args.output_recursive)


def _import_data(args: argparse.Namespace) -> None:
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - exercised via the [ingest] extra
        raise SystemExit("import-data requires pandas; install it with: pip install 'symbolic-data[ingest]'") from exc

    from pathlib import Path

    import yaml

    from simplipy import SimpliPyEngine

    from symbolic_data import SkeletonPool, substitute_root_path
    from symbolic_data.convert_data import ParserFactory

    if args.verbose:
        print(f"Importing data from {args.input}")

    simplipy_engine = SimpliPyEngine.load(args.simplipy_engine, install=True)
    base_skeleton_pool = SkeletonPool.from_config(args.base_skeleton_pool)
    input_path = substitute_root_path(args.input)
    path_obj = Path(input_path)

    if path_obj.suffix.lower() in {".yaml", ".yml"}:
        with open(input_path, "r", encoding="utf-8") as handle:
            raw_data = yaml.safe_load(handle)

        if not isinstance(raw_data, dict):
            raise ValueError("Expected YAML benchmark file to contain a mapping of equation identifiers to entries.")

        records = []
        for identifier, payload in raw_data.items():
            if not isinstance(payload, dict):
                continue

            record = {"id": identifier}
            record.update(payload)
            if "prepared" in record and record["prepared"] is None:
                # Normalise missing prepared expressions to empty strings for downstream filtering.
                record["prepared"] = ""
            records.append(record)

        df = pd.DataFrame.from_records(records)
    else:
        df = pd.read_csv(input_path)

    data_parser = ParserFactory.get_parser(args.parser)
    test_skeleton_pool = data_parser.parse_data(df, simplipy_engine, base_skeleton_pool, verbose=args.verbose)

    if args.verbose:
        print(f"Saving imported skeleton pool to {args.output_dir}")
    test_skeleton_pool.save(directory=args.output_dir, config=args.base_skeleton_pool, reference="relative", recursive=True)


def _split_skeleton_pool(args: argparse.Namespace) -> None:
    import os

    from symbolic_data import SkeletonPool

    print(f"Loading skeleton pool from {args.input}")
    config, skeleton_pool = SkeletonPool.load(args.input)
    train_skeleton_pool, val_skeleton_pool = skeleton_pool.split(train_size=args.train_size, random_state=args.random_state)

    train_path = os.path.join(args.input, "train")
    val_path = os.path.join(args.input, "val")

    print(f"Saving training pool to {train_path}")
    train_skeleton_pool.save(directory=train_path, config=deepcopy(config), reference="relative", recursive=True)
    print(f"Saving validation pool to {val_path}")
    val_skeleton_pool.save(directory=val_path, config=deepcopy(config), reference="relative", recursive=True)


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    match args.command_name:
        case "generate-skeleton-pool":
            _generate_skeleton_pool(args)
        case "import-data":
            _import_data(args)
        case "split-skeleton-pool":
            _split_skeleton_pool(args)


if __name__ == "__main__":
    main()
