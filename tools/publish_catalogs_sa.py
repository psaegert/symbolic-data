"""Publish the share-alike and copyleft catalogs to their own, SEPARATE HuggingFace dataset repo.

The share-alike firewall (verified against the OEIS EULA + CC BY-SA 4.0 legal code, 2026-07-10):
OEIS-/Wikipedia-derived catalogs are Adapted Material under CC BY-SA 4.0 and must carry that
license. A SEPARATE repo (rather than a subdirectory of the main assets repo) is mandatory so
(a) the HF license card metadata is exact and machine-readable, (b) no repo-level
Adapted-Material reading (CC BY-SA 4.0 Sec 4(b)) can encumber the permissive catalogs, and
(c) the SA artifact is self-contained: LICENSE (full legal code + derivation statement) ships
inside the repo alongside the catalogs and manifest. The same holds for copyleft data (GPL, ODbL;
owner 2026-11-09): copyleft with copyleft, each catalog naming its own license, the license texts in
LICENSES/. The data catalogs routed here by the license pass ("yes, share-alike repo"; built into
build/data-catalogs) publish here too; the publish refuses anything it routes elsewhere.

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
# srbf 2026-11 release: the share-alike members of the data catalogs (publish_catalogs.DATA_BUILD)
DATA_CATALOGS = {
    "dimensionless-learning": ("dimensionless-learning.npz", 5, "problem_catalog"),
    "gecco-2013-2017-real-share-alike": ("gecco-2013-2017-real-share-alike.npz", 1, "problem_catalog"),
    "scilaws-real-share-alike": ("scilaws-real-share-alike.npz", 4, "problem_catalog"),
}
ROUTE = ("yes, share-alike repo",)
COPYLEFT_TEXTS = ("GPL-2.0-only", "GPL-3.0-only", "ODbL-1.0")
# catalogs without a license-pass row: their notices are the derivation statement of assets_sa/LICENSE
NOTICED_BEFORE_2026_11 = ("erbench-eponymous", "erbench-oeis")
LEGAL_CODE = "The full CC BY-SA 4.0 legal code follows."

CARD = """---
license:
- cc-by-sa-4.0
- cc-by-sa-3.0
- gpl-3.0
- gpl-2.0
- odbl
---

# symbolic-data assets (share-alike and copyleft catalogs)

Share-alike- and copyleft-licensed benchmark catalogs for the `symbolic-data` package, kept SEPARATE
from the catalogs in `psaegert/symbolic-data-assets` (licensed) and
`psaegert/symbolic-data-assets-restricted` (non-commercial terms or no license). Each catalog, and each
problem's metadata, names its license; LICENSE holds the CC BY-SA 4.0 legal code and the derivation
statement, LICENSES/ the GPL-2.0, GPL-3.0 and ODbL 1.0 texts.

- CC BY-SA 4.0: `erbench-oeis` and `erbench-eponymous`, derived from The On-Line Encyclopedia of Integer
  Sequences (OEIS, (c) The OEIS Foundation Inc.; per-entry A-number attribution in catalog meta) and
  Wikipedia ("List of scientific equations named after people"), via the ERBench benchmark
  (arXiv:2606.09276); and SciLaws-Bench's `running_endurance_iaaf__velocity`.
- CC BY-SA 3.0 ([legal code](https://creativecommons.org/licenses/by-sa/3.0/legalcode)): SciLaws-Bench's
  `baseball_pythagorean_winpct_lahman__win_pct`.
- GPL-3.0-or-later: SciLaws-Bench's `binary_pulsar_post_keplerian_atnf__PBDOT`, from the ATNF Pulsar
  Catalogue (psrcat v2.8.0; its programs and databases remain the property of the Australia Telescope
  National Facility, CSIRO).
- GPL-3.0: `dimensionless-learning`, the dataset files of PyDimension v1.0.0 (Xie et al. 2022).
- GPL-2: `gecco-2013-2017-real-share-alike`, the Ozone data of the R package mlbench (Friedrich Leisch and
  Evgenia Dimitriadou).
- ODbL 1.0: SciLaws-Bench's `wind_turbine_power_curve_engie__power_kW`.

The SciLaws tasks are `scilaws-real-share-alike` (SciLaws-Bench, Huang et al. 2026, arXiv:2609.01552;
per-task licenses in its dataset/LICENSES.md). Data catalogs store the rows their authors published as
arrays; no value is changed.

Resolve by explicit repo syntax: `load_catalog("psaegert/symbolic-data-assets-sa:erbench-oeis@<version>")`.
"""


def main(execute: bool = False) -> None:
    from publish_catalogs import publish
    with open(os.path.join(SA_DIR, "LICENSE"), encoding="utf-8") as handle:
        statement = handle.read().partition(LEGAL_CODE)[0]
    unlisted = [name for name in NOTICED_BEFORE_2026_11 if name not in statement]
    if unlisted:
        raise SystemExit(f"REFUSED: NOTICED_BEFORE_2026_11 names {unlisted}, which assets_sa/LICENSE does not")
    card = [CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=CARD.encode("utf-8")),
            CommitOperationAdd(path_in_repo="LICENSE", path_or_fileobj=os.path.join(SA_DIR, "LICENSE"))]
    card += [CommitOperationAdd(path_in_repo=f"LICENSES/{lic}.txt", path_or_fileobj=os.path.join(SA_DIR, "LICENSES", f"{lic}.txt"))
             for lic in COPYLEFT_TEXTS]
    publish(REPO, SA_DIR, CATALOGS, {}, execute=execute, extra_ops=tuple(card), built=DATA_CATALOGS,
            shareable=ROUTE, hand_noticed=NOTICED_BEFORE_2026_11)


if __name__ == "__main__":
    import sys
    main(execute="--execute" in sys.argv)
