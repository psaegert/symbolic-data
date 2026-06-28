# Command-line interface

`symbolic-data` ships a small CLI for the data-preparation steps that flash-ansr used to own before
its data layer was carved out (flash-ansr 0.7). Invoke it as `symbolic-data <command>` (console
script) or `python -m symbolic_data <command>`.

The relative paths in the examples below (`./configs/...`, `./data/...`) resolve against your
**current working directory**, so run the commands from your project root. Separately, any path you
write with a literal `{{ROOT}}` token is expanded to `$SYMBOLIC_DATA_ROOT` when that variable is set,
otherwise to the installed package's own checkout root (usually not what you want); prefer plain
relative or absolute paths.

```bash
symbolic-data --help
```

There are three commands.

## generate-skeleton-pool

Sample a canonical skeleton pool from a config (the CLI wrapper around
`SkeletonPool.from_config(...).create(...).save(...)`).

```bash
symbolic-data generate-skeleton-pool \
  -c "./configs/my_model/skeleton_pool_val.yaml" \
  -o "./data/my_model/skeleton_pool_val" \
  -s 1000 -v   # sample 1000 skeletons, show a progress bar
```

`--output-reference` controls how saved config paths are written (`relative` by default) and
`--no-output-recursive` disables recursively saving referenced configs.

## import-data

Ingest a raw benchmark spec into a skeleton pool, e.g. to build a **holdout pool** that training
then excludes (via `holdout_pools:` in a `skeleton_pool_*.yaml`). You supply two existing files:
`-i` a benchmark spec you provide (a YAML mapping such as the FastSRB `expressions.yaml`, or a CSV
of equations) and `-b` a base skeleton-pool config you author or copy (it defines the variables,
operators, and engine the imported skeletons must fit). `-p` selects the parser.

```bash
symbolic-data import-data \
  -i "./benchmarks/fastsrb/expressions.yaml" \
  -b "./configs/test_set/skeleton_pool.yaml" \
  -p fastsrb -e dev_7-3 \
  -o "./data/test_set/fastsrb/skeleton_pool" -v
```

Built-in parsers (`-p`): `fastsrb`, `feynman`, `nguyen`, `soose`. Each parses, validates, simplifies,
standardizes variable names, and codifies every equation into the base pool.

`-e` names a SimpliPy engine (here `dev_7-3`). It is loaded via `SimpliPyEngine.load(..., install=True)`,
which downloads and installs the engine if it is not already present; use the same engine your base
config and training pipeline use. See the SimpliPy documentation for available engine names.

> **Needs the `[ingest]` extra.** `import-data` builds the input table with pandas, which is an
> optional dependency: `pip install "symbolic-data[ingest]"`. The parser API itself
> (`symbolic_data.ParserFactory`) does not import pandas; only this CLI command (which constructs the
> DataFrame from your file) needs it.

## split-skeleton-pool

Split an existing pool into `train/` and `val/` subdirectories.

```bash
symbolic-data split-skeleton-pool \
  -i "./data/my_model/skeleton_pool" \
  -t 0.8 -r 0   # 80% train, seed 0
```

## Programmatic use

Everything the CLI does is available as a library. Pool generation is
[`SkeletonPool`](sampling.md); ingest is `ParserFactory`:

```python
from symbolic_data import SkeletonPool, ParserFactory

base = SkeletonPool.from_config("./configs/test_set/skeleton_pool.yaml")
parser = ParserFactory.get_parser("fastsrb")
pool = parser.parse_data(df, simplipy_engine, base)   # df: a pandas DataFrame of the raw spec
pool.save("./data/test_set/fastsrb/skeleton_pool")
```
