---
license:
- mit
- bsd-3-clause
- apache-2.0
- cc-by-4.0
- cc-by-3.0
- cc-by-2.5
- cc0-1.0
- etalab-2.0
---

# symbolic-data assets (benchmark catalogs)

Versioned, sha256-pinned symbolic-regression benchmark catalogs for the
[`symbolic-data`](https://pypi.org/project/symbolic-data/) package. Resolve by name:
`load_catalog("fastsrb")` / `load_catalog("fastsrb@1")` (pinned). `manifest.json` maps
`name@version -> files + revision + sha256`; versioning is FORWARD-ONLY (published version
files never change bytes; fixes ship as new versions — e.g. `fastsrb@2` repairs defects
found by the 2026-07-10 program audit while `fastsrb@1` keeps resolving unchanged).

## Licensing

No single license covers this repo: **each catalog stays under the license of the source it took
content from**, and the `license` tags list every license that occurs. THIRD_PARTY_NOTICES.md names
each catalog's source and license and every MIT, BSD-3-Clause and Apache-2.0 source's copyright
notice; LICENSES/ holds those licenses' texts, and the notices link the Creative Commons and Etalab
legal codes. Every catalog of srbf's 2026-11 release also names its license, its citations and what
srbf changed in its own metadata. Share-alike (CC BY-SA) catalogs are deliberately NOT in this repo:
they live in the separate
[psaegert/symbolic-data-assets-sa](https://huggingface.co/datasets/psaegert/symbolic-data-assets-sa)
repo behind a license firewall.

The data catalogs (a benchmark's published data as frozen pools) are built from their authors'
files by the source repo's builders (`tools/catalog_build`) and published only here.

Curated in [psaegert/symbolic-data](https://github.com/psaegert/symbolic-data) (builders,
vendored upstream files with NOTICEs, validation reports).
