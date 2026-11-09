"""Publish the restricted catalogs to their own Hugging Face dataset repo, apart from the licensed ones.

Owner ruling 2026-11-09: data whose source sets non-commercial or scientific-use-only terms, or states no license at
all, are not left out of srbf's benchmark but republished, each with a statement of where they come from, the terms
at their source, why srbf republishes them (every run uses exactly the same bytes, pinned by sha256, independent of a
third-party host that may change, move or disappear) and how a rights holder has them removed. They live in a repo
of their own so that the main repo's license tags stay true: nothing here is under a license the main repo names.
The tags/licenses.json routing ("yes, restricted repo") decides what publishes here; the publish refuses anything
the license pass routes elsewhere, and the main and share-alike tools refuse what it routes here.

Consumers resolve these by explicit repo syntax:
    load_catalog("psaegert/symbolic-data-assets-restricted:mvsr-2024@1")

Run from the repo root: ``python tools/publish_catalogs_restricted.py`` (dry run; ``--execute`` publishes, which
requires HF auth; ``--notices`` rewrites the generated part of the card's notices). The same incremental,
forward-only publish as publish_catalogs.py.
"""
from __future__ import annotations

import os

from huggingface_hub import CommitOperationAdd

REPO = "psaegert/symbolic-data-assets-restricted"
HERE = os.path.dirname(os.path.abspath(__file__))
CARD_DIR = os.path.join(HERE, "..", "assets", "hf_card_restricted")
ROUTE = ("yes, restricted repo",)

# built into build/data-catalogs by tools/catalog_build (git-ignored; published here only): the sets routed here whole
# and the <slug>-restricted parts of sets whose datasets differ
DATA_CATALOGS = {
    "bms-real-data": ("bms-real-data.npz", 3, "problem_catalog"),
    "dow-chemical-evostar-2010": ("dow-chemical-evostar-2010.npz", 1, "problem_catalog"),
    "dow-distillation-tower": ("dow-distillation-tower.npz", 1, "problem_catalog"),
    "dow-emulator": ("dow-emulator.npz", 1, "problem_catalog"),
    "drsr-debcat": ("drsr-debcat.npz", 2, "problem_catalog"),
    "eql-double-pendulum": ("eql-double-pendulum.npz", 4, "problem_catalog"),
    "ffx-cmos-ota": ("ffx-cmos-ota.npz", 6, "problem_catalog"),
    "ffx-high-dimensional-circuits": ("ffx-high-dimensional-circuits.npz", 12, "problem_catalog"),
    "gecco-2013-2017-real-restricted": ("gecco-2013-2017-real-restricted.npz", 3, "problem_catalog"),
    "gp-challenge-protein": ("gp-challenge-protein.npz", 54, "problem_catalog"),
    "gp-gomea-real-world-restricted": ("gp-gomea-real-world-restricted.npz", 3, "problem_catalog"),
    "gplearn-diabetes": ("gplearn-diabetes.npz", 1, "problem_catalog"),
    "heuristiclab-real-world": ("heuristiclab-real-world.npz", 14, "problem_catalog"),
    "ideasearch-pdf": ("ideasearch-pdf.npz", 6, "problem_catalog"),
    "implied-volatility-slices": ("implied-volatility-slices.npz", 34, "problem_catalog"),
    "langlaw-perovskites-restricted": ("langlaw-perovskites-restricted.npz", 1, "problem_catalog"),
    "materials-laws-multi-agent": ("materials-laws-multi-agent.npz", 1, "problem_catalog"),
    "mcdermott-predictive-modelling-restricted": ("mcdermott-predictive-modelling-restricted.npz", 3, "problem_catalog"),
    "mvsr-2024": ("mvsr-2024.npz", 501, "problem_catalog"),
    "mvsr-2025": ("mvsr-2025.npz", 51, "problem_catalog"),
    "nuclear-misr-restricted": ("nuclear-misr-restricted.npz", 1, "problem_catalog"),
    "oxygen-ion-conductors": ("oxygen-ion-conductors.npz", 2, "problem_catalog"),
    "phye2e-space-physics-restricted": ("phye2e-space-physics-restricted.npz", 1, "problem_catalog"),
    "scilaws-real-restricted": ("scilaws-real-restricted.npz", 6, "problem_catalog"),
    "shadow-economies": ("shadow-economies.npz", 2, "problem_catalog"),
    "shape-constrained-real-world-restricted": ("shape-constrained-real-world-restricted.npz", 3, "problem_catalog"),
    "sldbench": ("sldbench.npz", 77, "problem_catalog"),
    "sparc-radial-acceleration": ("sparc-radial-acceleration.npz", 1, "problem_catalog"),
    "srbench-2.0-blackbox-restricted": ("srbench-2.0-blackbox-restricted.npz", 7, "problem_catalog"),
    "srbench-2.0-first-principles-restricted": ("srbench-2.0-first-principles-restricted.npz", 2, "problem_catalog"),
    "srbench-2021-blackbox-restricted": ("srbench-2021-blackbox-restricted.npz", 38, "problem_catalog"),
    "white-2013-proposed": ("white-2013-proposed.npz", 1, "problem_catalog"),
}


def main(execute: bool = False, write_notices: bool = False) -> None:
    from publish_catalogs import HF_CARD, NOTICE_LICENSES, notices, publish, without_copyright
    path = os.path.join(CARD_DIR, "THIRD_PARTY_NOTICES.md")
    with open(path, encoding="utf-8") as handle:
        text = handle.read()
    fresh = notices(text, names=DATA_CATALOGS, terms=True)
    if write_notices:
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(fresh)
        return
    if fresh != text:
        raise SystemExit("REFUSED: assets/hf_card_restricted/THIRD_PARTY_NOTICES.md is stale -- run "
                         "python tools/publish_catalogs_restricted.py --notices")
    uncovered = without_copyright(names=DATA_CATALOGS)
    if uncovered:
        raise SystemExit(f"REFUSED: no copyright entry for {uncovered} (tools/catalog_build/tags/copyrights.json)")
    card = [CommitOperationAdd(path_in_repo=fn, path_or_fileobj=os.path.join(CARD_DIR, fn))
            for fn in ("README.md", "THIRD_PARTY_NOTICES.md")]
    card += [CommitOperationAdd(path_in_repo=f"LICENSES/{lic}.txt", path_or_fileobj=os.path.join(HF_CARD, "LICENSES", f"{lic}.txt"))
             for lic in NOTICE_LICENSES]
    publish(REPO, CARD_DIR, {}, {}, execute=execute, extra_ops=tuple(card), built=DATA_CATALOGS, shareable=ROUTE)


if __name__ == "__main__":
    import sys
    main(execute="--execute" in sys.argv, write_notices="--notices" in sys.argv)
