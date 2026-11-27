# catalog_build: the benchmark catalogs of srbf's 2026-11 release

These scripts build symbolic-data catalogs from the curated records of srbf's benchmark survey
(`assets/upstream/catalog-prep`, see its NOTICE): every newly collected benchmark set as a formula catalog, the
problems whose authors fix their points as companion `-points` data catalogs, the authors' test sets, ERBench on its
own generator, and the release catalogs' corrections. Run them from the repository root, in this order:

1. `python tools/catalog_build/patch_release_catalogs.py` — the release catalogs' corrections and authors' test sets
   (Keijzer, Vladislavleva, Nguyen, Neat, Jin, SOOSE, FastSRB), the pinned-point members moved out, Const-Test dropped.
   Idempotent: a second run changes nothing.
2. `python tools/catalog_build/build_erbench_catalogs.py` — ERBench's five catalogs on ERBench's generator (one problem
   per formula and alpha) and the frozen SynEq/OEIS subsamples; rebuilt from the published version 1 (`--base`).
3. `python tools/catalog_build/build_formula_catalogs.py [--only <set tags>]` — one catalog per collected formula set,
   and `--family-points` for the release catalogs' companion `-points` catalogs. Writes into `assets/catalogs` and a
   `build_formula_report.json` (what was built, skipped, and why).
4. `python tools/build_srsd_dummy.py` — srsd-dummy on FastSRB's corrected laws with SRSD's published dummy columns.
5. `python tools/catalog_build/catalog_tags.py flags ...`, `facts ...`, `merge ...`, `apply ...` — the catalog tags
   (origin and what the catalog tests) and the catalog-level metadata srbf reads into every catalog: its `benchmark`
   (`tags/benchmarks.json`: srbf averages by benchmark; the -restricted, -share-alike, -points parts and the formula/data
   halves of one benchmark share one slug), its `license`, `citation`, `srbf_changes` (`tags/deviations.json`; a change
   of kind changed, filled or kept earns the '*' and the srbf-modified tag, running part of a benchmark does not), its
   `srbf_notes` (where srbf follows the defining publication, and where the rule for paper-vs-code conflicts applied:
   srbf follows what produced the authors' published results, their code or data, and notes the paper's version;
   every instance in `tags/rule_applications.json`) and its `problem_flags` (`tags/problem_flags.json`, from `flags`:
   trivial, low-signal, target-leak, tiny, ordered). `tags/judged.json` is the reading of the set records each judgement
   tag rests on; `tags/global_notes.json` holds the notes that apply to every catalog. A rebuild drops a catalog's tags,
   so run `apply` after every build.
6. `python tools/catalog_build/build_data_catalogs.py` and `build_srbench_catalogs.py` — the data catalogs: each
   benchmark's published data as frozen pools (`symbolic_data.pools`), one reader per set in `data_catalog_readers.py`,
   and the four SRBench tracks from PMLB. They are written to `build/data-catalogs` (git-ignored) and published to
   Hugging Face only. Their input files are not in the repository: `data_sets.json` names each file, its source and
   its sha256, and `SYMBOLIC_DATA_CATALOG_RAW` points at a folder holding them (files with a URL are downloaded).
   Run `catalog_tags.py apply` on them too. LLM-SRBench's records are not the survey's: `llm_srbench_records.py
   --write` makes them from the authors' frozen release (its files under `SYMBOLIC_DATA_CATALOG_RAW/llm-srbench`),
   with the formulas repaired and the constants the authors left without values fitted to the training rows, each
   formula checked against every row. The license pass (`tags/licenses.json`) routes every set and member:
   licensed data to `tools/publish_catalogs.py` (the main repo), share-alike and copyleft data to
   `publish_catalogs_sa.py`, and data under non-commercial terms or with no license, each carrying a statement of
   its source, its terms and why srbf republishes it (owner 2026-11-09), to `publish_catalogs_restricted.py`; a set
   whose datasets differ is split into `<slug>`, `<slug>-share-alike` and `<slug>-restricted`. All three publish from
   `build/data-catalogs` (their `DATA_CATALOGS`), and each refuses what the pass routes elsewhere. A data catalog not
   built keeps its hosted versions. A rebuild writes the same npz bytes where nothing changed (np.savez dates every zip entry
   1980-01-01; the builds of 2026-11-05 and 2026-11-08 match by sha256), so a published catalog stays unchanged; a
   catalog whose content changes needs its version bumped, since the publish refuses new bytes under a published
   version.
7. `python tools/catalog_build/count_release.py` — the release's problem, subset and benchmark counts (the curated
   catalogs and the data build), with what the rules and the license pass leave out. `overlap.py` then measures, per
   catalog, the problems whose law srbf also runs in another benchmark and the identical ones (`tags/overlap.json`).
8. `python tools/catalog_build/render_draws.py --out <root> --jobs 8` — the release's frozen draws
   (`symbolic_data.draws`, docs/draws.md): every problem of every release catalog drawn once (ERBench SynEq and OEIS
   on their frozen 1,024), for srbf to read. Run it after the catalogs and the engine are final; `symbolic-data
   verify-draws <root> <catalogs>` shows that a second render writes the same bytes.

Records added after the survey (2026-11-27 audit) sit next to it: `additions_2026-11.json` (the LLM-SR oscillators,
the Bayesian machine scientist's Rössler experiment, recovered from its supplementary figures by
`bms_rossler_figures.py` and `bms_rossler_rows.py`, and Brum et al.'s dynamical systems, regenerated by
`sr_dynamical_systems.py`), and `residues_2026-11.json`: the members of six compilations the survey cut to what srbf did
not hold yet, which build_formula_catalogs.py copies back from the identical srbf problem (every compilation is kept
whole as published).

Gates (2026-11-27 audit): every formula builder checks numerically that each entry's prepared formula computes the
formula the authors wrote (`formula_gate.py`; an expected difference needs its row in `tags/raw_vs_prepared_allowed.json`
with the reason), and stops otherwise; the data builder keeps a formula that misses its rows as a `reference` ground
truth only with a stated reason and an FVU bound it meets (`tags/reference_laws.json`, or `gt_reason` and
`gt_fvu_bound` in the record), and refuses it otherwise. Transcription fixes to the survey's records are explicit, each
with its reason (`RECORD_FIXES` in build_formula_catalogs.py, `FORMULA_FIXES` in build_data_catalogs.py). What the
rules leave out of the release (scope, formulas as written) is `tags/dropped.json`, with each reason; the builders skip
it and count_release lists it.

`SYMBOLIC_DATA_CATALOG_PREP` points the builders at another copy of the survey's records (`paths.py`). Text copied
from the records into a catalog passes `paths.public` (no local paths, no internal wording).
