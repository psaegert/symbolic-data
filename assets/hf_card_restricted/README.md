---
license: other
license_name: per-dataset-terms
license_link: THIRD_PARTY_NOTICES.md
---

# symbolic-data assets (restricted data)

Benchmark data for the [`symbolic-data`](https://pypi.org/project/symbolic-data/) package whose sources set
non-commercial or scientific-use-only terms, or state no license at all. They are kept apart from the licensed
catalogs in [psaegert/symbolic-data-assets](https://huggingface.co/datasets/psaegert/symbolic-data-assets) and the
share-alike and copyleft ones in
[psaegert/symbolic-data-assets-sa](https://huggingface.co/datasets/psaegert/symbolic-data-assets-sa), so that those
repos' license tags stay true.

## Why these data are here

srbf republishes these data unchanged so that every run uses exactly the same bytes, pinned by sha256, and does not
depend on a third-party host that may change, move or disappear. Republishing grants no rights: every dataset stays
under the terms of its source.

- **Non-commercial or scientific-use-only terms:** use these data only as their terms allow.
- **No license stated:** the source has granted no license. srbf republishes the data for the reason above and names
  their source.

THIRD_PARTY_NOTICES.md lists every dataset's source and terms. Each problem also carries them in its metadata
(`license`, `terms`, `statement`).

## Removal on request

If you hold rights to any of these data and object to their republication, open an issue at
https://github.com/psaegert/symbolic-data and they will be removed.

Resolve by explicit repo syntax: `load_catalog("psaegert/symbolic-data-assets-restricted:<name>@<version>")`.
Curated in [psaegert/symbolic-data](https://github.com/psaegert/symbolic-data) (builders in `tools/catalog_build`).
