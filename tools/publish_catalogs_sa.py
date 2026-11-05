"""Publish the CC-BY-SA-4.0 catalogs to their own, SEPARATE HuggingFace dataset repo.

The share-alike firewall (verified against the OEIS EULA + CC BY-SA 4.0 legal code, 2026-07-10):
OEIS-/Wikipedia-derived catalogs are Adapted Material under CC BY-SA 4.0 and must carry that
license. A SEPARATE repo (rather than a subdirectory of the main assets repo) is mandatory so
(a) the HF license card metadata is exact and machine-readable, (b) no repo-level
Adapted-Material reading (CC BY-SA 4.0 Sec 4(b)) can encumber the permissive catalogs, and
(c) the SA artifact is self-contained: LICENSE (full legal code + derivation statement) ships
inside the repo alongside the catalogs and manifest.

Consumers resolve these by explicit repo syntax:
    load_catalog("psaegert/symbolic-data-assets-sa:erbench-oeis@1")
(the DEFAULT repo's manifest intentionally does NOT list them).

Run from the repo root: ``python tools/publish_catalogs_sa.py`` (dry run; add ``--execute`` to publish,
which requires HF auth). The same incremental, forward-only publish as publish_catalogs.py: a file publishes
as the version its metadata states, published versions are immutable and stay resolvable.
"""
from __future__ import annotations

import os

from huggingface_hub import CommitOperationAdd

REPO = "psaegert/symbolic-data-assets-sa"
REPO_TYPE = "dataset"
HERE = os.path.dirname(os.path.abspath(__file__))
SA_DIR = os.path.join(HERE, "..", "assets_sa")

CATALOGS = {
    # P5 ERBench SA families (OEIS-/Wikipedia-derived, CC-BY-SA-4.0; see assets_sa/LICENSE)
    "erbench-oeis": ("catalogs/erbench-oeis.yaml", 10497, "problem_catalog"),
    "erbench-eponymous": ("catalogs/erbench-eponymous.yaml", 585, "problem_catalog"),
}

CARD = """---
license: cc-by-sa-4.0
---

# symbolic-data assets (CC BY-SA 4.0 catalogs)

Share-alike-licensed benchmark catalogs for the `symbolic-data` package, kept SEPARATE from the
permissive catalogs in `psaegert/symbolic-data-assets`. Everything in this repo is licensed
**CC BY-SA 4.0** (see LICENSE: full legal code + derivation statement).

Derived from The On-Line Encyclopedia of Integer Sequences (OEIS, (c) The OEIS Foundation Inc.,
CC-BY-SA-4.0; per-entry A-number attribution in catalog meta) and Wikipedia ("List of scientific
equations named after people", CC-BY-SA-4.0), via the ERBench benchmark (arXiv:2606.09276).

Resolve by explicit repo syntax: `load_catalog("psaegert/symbolic-data-assets-sa:erbench-oeis@<version>")`.
"""


def main(execute: bool = False) -> None:
    from publish_catalogs import publish
    card = [CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=CARD.encode("utf-8")),
            CommitOperationAdd(path_in_repo="LICENSE", path_or_fileobj=os.path.join(SA_DIR, "LICENSE"))]
    publish(REPO, SA_DIR, CATALOGS, {}, execute=execute, extra_ops=tuple(card))


if __name__ == "__main__":
    import sys
    main(execute="--execute" in sys.argv)
