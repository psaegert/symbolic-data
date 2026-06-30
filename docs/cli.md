# Command-line interface

`symbolic-data` ships a small CLI with **one command**, `materialize`. Invoke it as
`symbolic-data materialize ...` (console script) or `python -m symbolic_data materialize ...`.

```bash
symbolic-data --help
```

The relative paths in the examples below resolve against your **current working directory**, so run
the command from your project root. Separately, any path you write with a literal `{{ROOT}}` token is
expanded to `$SYMBOLIC_DATA_ROOT` when that variable is set, otherwise to the installed package's own
checkout root (usually not what you want); prefer plain relative or absolute paths.

## materialize

Sample a [`ProblemSource`](sampling.md#problemsource-level-2) (from a config) once and **freeze** it
to a versioned `.npz` catalog — the shareable, exactly-reproducible form. The config is a
`ProblemSource` spec: a `catalog` ref (declarative or generative), or inline `problems`, plus a
`sampling` block and optional `holdouts`.

```bash
symbolic-data materialize \
  -c "./configs/my_val.yaml" \
  -o "./data/my_val.npz" \
  -n 1000 \
  --name my_val
```

| flag | meaning |
|---|---|
| `-c`, `--config` | ProblemSource config (yaml): a catalog / problems spec + sampling. |
| `-o`, `--output` | output catalog path (a `.npz` frozen catalog). |
| `-n`, `--n` | cap the number of problems (required for an unbounded generator without `size`). |
| `--name` | catalog name stamped in metadata. |

An example config (`configs/my_val.yaml`) — generate 1000 fresh expressions and decontaminate them
against a held-out set:

```yaml
catalog: lample-charton-v23          # an open generative recipe (resolved by name)
sampling:
  n_support: prior                   # draw the support size per sample from the catalog's prior
  n_validation: 0                    # required when n_support: prior
  size: 1000                         # number of expressions to draw
holdouts:
  - exclude: v23-val                 # drop any problem whose skeleton matches the v23 validation set
```

The frozen `.npz` it writes is loaded back with `load_catalog("./data/my_val.npz")` (or
`ProblemSource({"catalog": "./data/my_val.npz"})`), and re-iterating yields byte-identical
`Problem`s.

> The pre-0.4.1 `generate-skeleton-pool` / `import-data` / `split-skeleton-pool` data-CLI commands
> were **removed**. Catalog generation is now the `ProblemSource` engine, curated test sets are
> Hugging Face artifacts (resolved by name, not bundled — see [Benchmarks](benchmarks.md)), and
> decontamination is a `ProblemSource` `holdouts: [{exclude: <catalog>}]` (see
> [Holdout & leak-safety](holdout.md)).

## Programmatic use

`materialize` is a thin wrapper around the library:

```python
from symbolic_data import ProblemSource

source = ProblemSource.from_config("./configs/my_val.yaml")
catalog = source.to_catalog(name="my_val", n=1000)   # a FROZEN ProblemCatalog
path = catalog.save("./data/my_val.npz")
print(f"Materialized {len(catalog.problems or [])} problems -> {path}")
```
