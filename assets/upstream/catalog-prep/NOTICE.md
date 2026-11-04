# catalog-prep: the curated inputs of the catalog builders

srbf's benchmark survey (2026-10..11) recorded, for every collected benchmark set, its problems as the authors define
them: formula, constants, input distributions, subsets, the authors' test specification, and the sources each was
read from (papers, supplements, repositories; cited per record). tools/catalog_build/ builds symbolic-data catalogs
from these records.

- `final_problems.json.gz`: the problems of the newly collected sets, formula (kind F) and data (kind D), with their
  subset (`unit`).
- `sets.json.gz`: each set's name and sources.
- `data_sets.json.gz`: the data sets' full records: each dataset's files (by their place in the survey's working copy,
  `catalog-prep/raw/...`), source URL, sha256, columns, and the split as published. The files are not vendored.
- `testspecs.json.gz`: the authors' test sets per problem (law per input, the source it was read from).
- `frozen/`: the fixed 1,024-problem subsamples (rank = sha256 of "srbf-1024:<set>:<subset>:<id>").
- `release_subsets.json`, `excluded.json`: the release catalogs' subsets and what was left out, with reasons.

The formulas, distributions and data belong to their benchmarks' authors under each benchmark's license, as recorded
per set; local file paths of the survey's working copy are removed.
