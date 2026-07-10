---
license: mit
---

# symbolic-data assets (benchmark catalogs)

Versioned, sha256-pinned symbolic-regression benchmark catalogs for the
[`symbolic-data`](https://pypi.org/project/symbolic-data/) package. Resolve by name:
`load_catalog("fastsrb")` / `load_catalog("fastsrb@1")` (pinned). `manifest.json` maps
`name@version -> files + revision + sha256`; versioning is FORWARD-ONLY (published version
files never change bytes; fixes ship as new versions — e.g. `fastsrb@2` repairs defects
found by the 2026-07-10 program audit while `fastsrb@1` keeps resolving unchanged).

## Licensing

The repo-level `license: mit` tag reflects the predominant licensing; **each catalog carries
its own provenance** — see THIRD_PARTY_NOTICES.md for the per-catalog upstream sources,
licenses, and copyright notices (MIT and BSD-3-Clause upstreams; plus self-generated
artifacts). Share-alike (CC BY-SA 4.0) catalogs are deliberately NOT in this repo: they live
in the separate [psaegert/symbolic-data-assets-sa](https://huggingface.co/datasets/psaegert/symbolic-data-assets-sa)
repo behind a license firewall.

Curated in [psaegert/symbolic-data](https://github.com/psaegert/symbolic-data) (builders,
vendored upstream files with NOTICEs, validation reports).
