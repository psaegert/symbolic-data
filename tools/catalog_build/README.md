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
5. `python tools/catalog_build/catalog_tags.py facts ...`, `merge ...`, `apply ...` — the catalog tags (origin and
   what the catalog tests) into every catalog's metadata; `tags/judged.json` is the reading of the set records each
   judgement tag rests on. A rebuild drops a catalog's tags, so run `apply` after every build.
6. `python tools/catalog_build/build_data_catalogs.py` and `build_srbench_catalogs.py` — the data catalogs: each
   benchmark's published data as frozen pools (`symbolic_data.pools`), one reader per set in `data_catalog_readers.py`,
   and the four SRBench tracks from PMLB. They are written to `build/data-catalogs` (git-ignored) and published to
   Hugging Face only. Their input files are not in the repository: `data_sets.json` names each file, its source and
   its sha256, and `SYMBOLIC_DATA_CATALOG_RAW` points at a folder holding them (files with a URL are downloaded).
   Run `catalog_tags.py apply` on them too. `tools/publish_catalogs.py` publishes them from `build/data-catalogs`
   (`DATA_CATALOGS`; the share-alike members through `publish_catalogs_sa.py`). A data catalog not built keeps its
   hosted versions. A rebuild writes the same npz bytes where nothing changed (np.savez dates every zip entry
   1980-01-01; the builds of 2026-11-05 and 2026-11-08 match by sha256), so a published catalog stays unchanged; a
   catalog whose content changes needs its version bumped, since the publish refuses new bytes under a published
   version.
7. `python tools/catalog_build/count_release.py` — the release's problem and subset counts (the curated catalogs and
   the data build).

`SYMBOLIC_DATA_CATALOG_PREP` points the builders at another copy of the survey's records (`paths.py`). Text copied
from the records into a catalog passes `paths.public` (no local paths, no internal wording).
