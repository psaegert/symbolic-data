# Frozen draws

A catalog holds problem templates; a [`ProblemSource`](sampling.md#problemsource-level-2) draws each problem's
points afresh on every pass. **Frozen draws** fix that draw ahead of time: every problem of a catalog is drawn once,
written to disk with a manifest of sha256 sums, and an evaluation reads it instead of sampling. Every method, budget,
shard and machine that reads the same render evaluates on the same data, bit for bit.

symbolic-data renders (`symbolic-data render-draws`, `symbolic_data.draws`); an evaluation harness reads. srbf
reads them through `data_source.draws` and refuses absent or mismatched draws (see its documentation).

## What one draw holds

| problem | what is frozen |
|---|---|
| formula problem (a declarative entry) | the `n_support` fitting points and `n_validation` held-out points, drawn together or from the entry's `test` block exactly as a `ProblemSource` draws them; the values the entry's `per_run` block draws (`meta["per_run_values"]`), with the ground truth (`expression`, `constants`, `skeleton`) compiled with them; one standard normal noise vector per side |
| data problem (pools, `symbolic_data.pools`) | the rows of one draw from its pools: its split rule applied (a random split cut, or one of the authors' folds picked), at most `n_support` fitting and `n_validation` test rows taken; an authors' test split is used as it is |
| realized problem (a frozen problem without a split rule) | the problem as stored |

**Noise.** A formula draw stores `noise_support` and `noise_validation`, standard normal vectors of the shape of
the targets. A noise level `s` makes the noisy targets

    y_noisy = y + s * std(y) * noise          (per side; y unchanged when s = 0, y is empty or std(y) = 0)

which is the scalar `sampling.noise` of a `ProblemSource` with the stored vector in place of a fresh draw. Every
noise level scales the same vector, so a noise sweep varies the level and nothing else. Data problems are not
noised; a stored problem keeps the noisy targets it was stored with.

**One draw per problem.** The default render holds one draw per problem, repeat `r = 1`. The repeat index is
part of the layout, the seeds and the manifest, so a later render can add repeats `2, 3, ...` beside it without a
change of format, and repeat 1 stays byte-identical.

## Layout

```
<root>/manifest.json
<root>/<catalog name>/repeat-1.npz
```

Each `repeat-<r>.npz` is a standard NumPy archive (`np.load`):

| member | content |
|---|---|
| `_meta` | UTF-8 JSON as a `uint8` array: `format`, `format_version`, `catalog` (`name`, `version`, `source` file name, `source_sha256`), `repeat`, `settings`, and `problems`: one record per problem, in catalog order (`row`, `eq_id`, `kind` `formula`/`data`/`stored`, `skeleton`, `expression`, `constants`, `variables`, `complexity`, `gt_kind`, `noise`, `meta`, `is_placeholder`, `placeholder_reason`) |
| `p<i>__x_support`, `p<i>__y_support`, `p<i>__x_validation`, `p<i>__y_validation` | the points of problem `i` (float64; targets as `(n, 1)` columns) |
| `p<i>__noise_support`, `p<i>__noise_validation` | the noise vectors (formula problems) |
| `p<i>__y_support_noisy`, `p<i>__y_validation_noisy` | stored noisy targets that differ from the clean ones (stored problems) |
| `p<i>__y_reference_*`, `p<i>__outlier_mask_*` | where the problem has them |

A placeholder (an entry with no valid draw within `max_trials`, or a malformed entry) is a record without arrays.
The file is written member by member, uncompressed, with every zip entry dated 1980-01-01, so the same draws give the
same bytes.

`manifest.json` records the format, the settings every catalog of the root shares (`n_support`, `n_validation`,
`layout`, `max_trials`, `seed_root`, `simplipy_engine`), the `repeats`, the versions it was rendered with
(`rendered_with`: Python, NumPy, symbolic-data, SimpliPy) and, per catalog: `name`, `version`, `source`,
`source_sha256` (the sha256 of the catalog file the draws were rendered from), `problems`, `kinds`, `ids` (the count
and sha256 of the id list when only some problems were rendered, else `null`), `placeholders` per repeat, and
`files`: per repeat its `path`, `bytes` and `sha256`. It is written with sorted keys and no timestamp, so a second
render writes the same manifest.

## Seeds

Problem `eq_id` of catalog `name` in repeat `r` draws from

    SeedSequence(int(sha256("<seed_root>\x1f<name>\x1f<eq_id>\x1f<r>")))

spawned into two streams: the first draws the points and the per-run values (passed to `ProblemSource.draw`), the
second the noise vectors. A problem's draw therefore depends on nothing but its catalog name, its id, the repeat and
the seed root (`symbolic-data/frozen-draws/1` by default): not on the other problems, their order, or which of them
are rendered, and the noise vector does not move when a sampler needs more attempts. Ids must be distinct within a
catalog.

## Rendering

```bash
symbolic-data render-draws ./assets/catalogs/nguyen.yaml ./assets/catalogs/feynman.yaml -o ./draws --jobs 4
symbolic-data verify-draws ./draws ./assets/catalogs/nguyen.yaml ./assets/catalogs/feynman.yaml
```

`render-draws` takes any catalog reference (a local `.yaml` or `.npz`, a `name[@version]`), renders with 512
fitting and 512 held-out points by default, compiles the ground truth with `acj-5-4-llm` (`--simplipy-engine`), and
refuses a root that already holds draws unless `--update` adds catalogs with the same settings. `--ids CATALOG=FILE`
renders a frozen subset (a JSON list of ids, or a mapping with an `ids` list). `verify-draws` checks every file
against the manifest and, for the catalogs it is given, renders them again into a scratch folder and compares the
bytes: a render is reproducible when that passes.

Formula evaluation may differ in the last bits between SimpliPy versions and machines, which is why a render is made
once and read everywhere rather than redrawn from the seeds. The release's draws are rendered by
`tools/catalog_build/render_draws.py` once the catalogs and the engine are final; on one machine with 8 workers the
13,418 formula problems of srbf's release take about 3 minutes and 0.8 GB (float64, one draw each).

From Python:

```python
from symbolic_data.draws import DrawsSettings, render, verify

manifest = render(["./assets/catalogs/nguyen.yaml"], "./draws", DrawsSettings(n_support=512, n_validation=512))
assert verify("./draws", ["./assets/catalogs/nguyen.yaml"]) == []
```

## What a reader checks

A reader keys a catalog by the sha256 of the catalog file it resolves, not by its name, and refuses:

- a root without `manifest.json`, or a manifest of another format or version;
- a catalog the manifest does not hold (another version of the catalog has another sha256), or a repeat it does not hold;
- a draws file that is missing, whose sha256 differs from the manifest, or whose `_meta` names another catalog,
  repeat or settings;
- sampling that differs from the render (`n_support`, `n_validation`, `layout`, `max_trials`), more than one problem
  per expression, another visiting order, a noise mixture (only the scalar noise is defined over the stored
  vector), and holdouts or filters (the render is the problem set);
- optionally, a manifest whose sha256 differs from a pinned one.

A result should record the draws file's sha256 and the repeat, so that two results can be shown to have read the
same data.
