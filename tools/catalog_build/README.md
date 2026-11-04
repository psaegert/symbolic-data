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
6. `python tools/catalog_build/count_release.py` — the release's problem and subset counts.

`SYMBOLIC_DATA_CATALOG_PREP` points the builders at another copy of the survey's records (`paths.py`). Text copied
from the records into a catalog passes `paths.public` (no local paths). The data catalogs (measured data, the SRBench
tracks) are built outside the repository for now; `count_release.py --catalogs <dir>` counts them.
