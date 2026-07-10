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

Run from the repo root: ``python tools/publish_catalogs_sa.py`` (requires HF auth).
Forward-only versioning discipline as in publish_catalogs.py.
"""
from __future__ import annotations

import hashlib
import json
import os

from huggingface_hub import CommitOperationAdd, HfApi

REPO = "psaegert/symbolic-data-assets-sa"
REPO_TYPE = "dataset"
HERE = os.path.dirname(os.path.abspath(__file__))
SA_DIR = os.path.join(HERE, "..", "assets_sa")

CATALOGS = {
    # P5 ERBench SA families (OEIS-/Wikipedia-derived, CC-BY-SA-4.0; see assets_sa/LICENSE)
    "erbench-oeis": ("catalogs/erbench-oeis.yaml", 3499, "problem_catalog"),
    "erbench-eponymous": ("catalogs/erbench-eponymous.yaml", 195, "problem_catalog"),
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

Resolve by explicit repo syntax: `load_catalog("psaegert/symbolic-data-assets-sa:erbench-oeis@1")`.
"""


def sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    from lint_catalogs import lint_paths
    assert lint_paths(), "catalog lint failed -- fix errors before publishing"
    api = HfApi()
    print(f"HF user: {api.whoami()['name']}")
    api.create_repo(REPO, repo_type=REPO_TYPE, exist_ok=True, private=False)

    ops = [CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=CARD.encode("utf-8")),
           CommitOperationAdd(path_in_repo="LICENSE",
                              path_or_fileobj=os.path.join(SA_DIR, "LICENSE"))]
    for _name, (rel, _cnt, _type) in CATALOGS.items():
        local = os.path.join(SA_DIR, rel)
        assert os.path.isfile(local), local
        ops.append(CommitOperationAdd(path_in_repo=os.path.basename(rel), path_or_fileobj=local))
    commit = api.create_commit(repo_id=REPO, repo_type=REPO_TYPE, operations=ops,
                               commit_message="Publish CC-BY-SA catalogs v1 (erbench-oeis, erbench-eponymous)")
    revision = commit.oid
    print(f"files commit: {revision}")

    manifest = {}
    for name, (rel, _cnt, ctype) in CATALOGS.items():
        fn = os.path.basename(rel)
        manifest[name] = {
            "type": ctype, "repo_id": REPO, "default_version": 1,
            "versions": {"1": {"repo_id": REPO, "directory": "", "files": [fn],
                               "revision": revision,
                               "sha256": {fn: sha256(os.path.join(SA_DIR, rel))}}},
        }
    api.upload_file(path_or_fileobj=(json.dumps(manifest, indent=2) + "\n").encode("utf-8"),
                    path_in_repo="manifest.json", repo_id=REPO, repo_type=REPO_TYPE,
                    commit_message="Publish SA manifest v1")
    print("manifest.json uploaded")


if __name__ == "__main__":
    main()
