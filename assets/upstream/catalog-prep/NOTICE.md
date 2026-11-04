# catalog-prep: the curated inputs of the catalog builders

srbf's benchmark survey (2026-10..11) recorded, for every collected benchmark set, its problems as the authors define
them: formula, constants, input distributions, subsets, the authors' test specification, and the sources each was
read from (papers, supplements, repositories; cited per record). tools/catalog_build/ builds symbolic-data catalogs
from these records.

- `final_problems.json.gz`: the formula problems (kind F) of the newly collected sets, with their subset (`unit`).
- `sets.json.gz`: each set's name and sources.
- `testspecs.json.gz`: the authors' test sets per problem (law per input, the source it was read from).
- `frozen/`: the fixed 1,024-problem subsamples (rank = sha256 of "srbf-1024:<set>:<subset>:<id>").
- `release_subsets.json`, `excluded.json`: the release catalogs' subsets and what was left out, with reasons.

The formulas and distributions belong to their benchmarks' authors under each benchmark's license, as recorded per
set; local file paths of the survey's working copy are removed.
