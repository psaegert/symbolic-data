"""Publish the curated catalogs to the Hugging Face assets dataset repo + a versioned manifest.

This is the distribution step for ``symbolic_data``'s curated catalogs: they live as artifacts on
Hugging Face (the artifact store), NOT in the PyPI wheel. The wheel ships NO catalog copies (pure-HF
since 0.8.0); the canonical, versioned, sha256-integrity-checked source of truth is the HF dataset
repo, and the repo keeps the source-of-truth yamls under ``assets/catalogs/`` (publish source + the
tests' local fixtures).

Layout (flat repo root): ``<name>.yaml`` for each catalog + ``manifest.json``. Each manifest entry
pins the catalog's content by ``revision`` (the git commit sha of the files commit) and per-file
``sha256``, exactly what ``symbolic_data.resolver.resolve`` verifies on download.

Run from the repo root: ``python tools/publish_catalogs.py``  (requires HF auth: ``huggingface_hub.whoami``).

Versioning discipline (forward-only): re-running this script overwrites the v1 manifest entry to point
at a fresh files commit. That is fine for the INITIAL publish, but a later CONTENT change to a catalog
should be published as a NEW version (add ``"2": {...}`` and bump ``default_version``) rather than
re-running v1, so a pinned ``name@1`` always resolves to identical bytes.
"""
from __future__ import annotations

import hashlib
import json
import os

from huggingface_hub import HfApi, CommitOperationAdd

REPO = "psaegert/symbolic-data-assets"          # MUST match resolver.HF_MANIFEST_REPO
REPO_TYPE = "dataset"
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "assets", "catalogs")

# Forward-only multi-version catalogs: name -> {version: filename}, plus the default version a
# bare name resolves to. v1 files stay byte-identical forever (pinned refs keep resolving);
# content fixes ship as NEW versions. fastsrb@2 = the 2026-07-10 audit repair (III.21.20
# unrealizable, string bounds, missing n_variables); see the changelog inside fastsrb.v2.yaml.
MULTI_VERSION = {
    "fastsrb": {"versions": {"1": "fastsrb.yaml", "2": "fastsrb.v2.yaml"}, "default": 2},
}

# logical name -> (filename, entry/skeleton count for the log, manifest type)
CATALOGS = {
    "fastsrb": ("fastsrb.yaml", 120, "problem_catalog"),
    "feynman": ("feynman.yaml", 100, "problem_catalog"),
    "nguyen": ("nguyen.yaml", 12, "problem_catalog"),
    "v23-val": ("v23-val.yaml", 1000, "generative_catalog"),                 # frozen validation set
    "lample-charton-v23": ("lample-charton-v23.yaml", None, "generative_catalog"),  # open training recipe
    # P1 GP-toy suites (DSO benchmarks.csv, BSD-3; validated 2026-07-10, 16/16)
    "constant": ("constant.yaml", 10, "problem_catalog"),
    "grammarvae": ("grammarvae.yaml", 1, "problem_catalog"),
    "jin": ("jin.yaml", 6, "problem_catalog"),
    "keijzer": ("keijzer.yaml", 15, "problem_catalog"),
    "korns": ("korns.yaml", 12, "problem_catalog"),
    "koza": ("koza.yaml", 2, "problem_catalog"),
    "livermore": ("livermore.yaml", 25, "problem_catalog"),
    "livermore2": ("livermore2.yaml", 150, "problem_catalog"),
    "meier": ("meier.yaml", 2, "problem_catalog"),
    "neat": ("neat.yaml", 8, "problem_catalog"),
    "nonic": ("nonic.yaml", 1, "problem_catalog"),
    "pagie": ("pagie.yaml", 1, "problem_catalog"),
    "poly": ("poly.yaml", 6, "problem_catalog"),
    "r-rationals": ("r-rationals.yaml", 6, "problem_catalog"),
    "sine": ("sine.yaml", 1, "problem_catalog"),
    "vladislavleva": ("vladislavleva.yaml", 8, "problem_catalog"),
    # P2 Feynman variant 3/3 (fastsrb + feynman already published)
    "srsd-dummy": ("srsd-dummy.yaml", 120, "problem_catalog"),
    # P3 real-world GT: SRBench-2.0 phenomenological track (PMLB first_principles_*, MIT).
    # FROZEN measured-data catalog (.npz): 13 datasets + refit reference laws, gt_kind=reference.
    "first-principles": ("first-principles.npz", 13, "problem_catalog"),
    # P3 real-world GT: known-GT cosmology subset of cp3-bench (Things-to-bench, MIT). FROZEN.
    "cp3-cosmo": ("cp3-cosmo.npz", 17, "problem_catalog"),
    # P3 real-world GT: AI-Descartes (IBM, MIT) — FSRD_noise (81 frozen 10-point noisy Feynman
    # renderings + clean validation) + 6 measured real-world datasets. FROZEN.
    "ai-descartes": ("ai-descartes.npz", 87, "problem_catalog"),
    # P3 real-world GT: PhySO (MIT) — Class-SR MW streams (frozen), paper astro panel + Class-SR
    # Table 1 (declarative; isochrone-action excluded, see physo-astro conventions).
    "physo-streams": ("physo-streams.npz", 29, "problem_catalog"),
    "physo-astro": ("physo-astro.yaml", 2, "problem_catalog"),
    "physo-class": ("physo-class.yaml", 8, "problem_catalog"),
    # P4 neural-SR bespoke: SOOSE NC/WC/FC (NeSymReS, MIT; WC/FC recovered from deleted history
    # @0cfff79 — the only surviving concrete instantiation). SSDNC (no license + no canonical
    # artifact), TPSR-400 (protocol, not a file), SymbolicGPT (generator configs only) = skips.
    "soose-nc": ("soose-nc.yaml", 200, "problem_catalog"),
    "soose-wc": ("soose-wc.yaml", 200, "problem_catalog"),
    "soose-fc": ("soose-fc.yaml", 200, "problem_catalog"),
    # P5 ERBench permissive novel families (MIT/BSD-3; arXiv:2606.09276). The CC-BY-SA families
    # (erbench-oeis, erbench-eponymous) publish to the SEPARATE SA repo via publish_catalogs_sa.py.
    "erbench-syneq": ("erbench-syneq.yaml", 5301, "problem_catalog"),
    "erbench-phybench": ("erbench-phybench.yaml", 90, "problem_catalog"),
    "erbench-densities": ("erbench-densities.yaml", 33, "problem_catalog"),
    # P6 black-box tier (gt_kind='none', eval-only): GT-free cp3-bench discovery sets.
    "cp3-blackbox": ("cp3-blackbox.npz", 11, "problem_catalog"),
    # P6: SRBench 2.0 black-box 12-selection (PMLB, MIT; fri_c0 pair = verified z-scored
    # Friedman-1 references) + the AI-Feynman BONUS set (aifeynman-original rendering, 3rd of 3).
    "srbench2-blackbox": ("srbench2-blackbox.npz", 12, "problem_catalog"),
    "feynman-bonus": ("feynman-bonus.yaml", 20, "problem_catalog"),
}


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
    who = api.whoami()["name"]
    print(f"HF user: {who}")

    api.create_repo(REPO, repo_type=REPO_TYPE, exist_ok=True, private=False)
    print(f"repo ready: {REPO} ({REPO_TYPE}, public)")

    # 1. upload all catalog files in ONE commit so a single revision pins the whole v1 set
    ops = []
    filenames = [fn for (fn, _cnt, _type) in CATALOGS.values()]
    for spec in MULTI_VERSION.values():
        filenames += [fn for fn in spec["versions"].values() if fn not in filenames]
    for extra in ("README.md", "THIRD_PARTY_NOTICES.md"):
        path = os.path.abspath(os.path.join(DATA_DIR, "..", "hf_card", extra))
        assert os.path.isfile(path), f"missing {extra}: {path}"
        ops.append(CommitOperationAdd(path_in_repo=extra, path_or_fileobj=path))
    for fn in filenames:
        local = os.path.abspath(os.path.join(DATA_DIR, fn))
        assert os.path.isfile(local), f"missing catalog file: {local}"
        ops.append(CommitOperationAdd(path_in_repo=fn, path_or_fileobj=local))
    commit = api.create_commit(
        repo_id=REPO, repo_type=REPO_TYPE, operations=ops,
        commit_message=f"Publish {len(filenames)} catalog artifacts + card/notices",
    )
    revision = commit.oid
    print(f"files commit: {revision}")

    # 2. build the manifest pinning revision + per-file sha256, then upload it
    manifest: dict = {}
    for name, (fn, _cnt, ctype) in CATALOGS.items():
        spec = MULTI_VERSION.get(name)
        versions = spec["versions"] if spec else {"1": fn}
        manifest[name] = {
            "type": ctype,
            "repo_id": REPO,
            "default_version": spec["default"] if spec else 1,
            "versions": {
                v: {
                    "repo_id": REPO,
                    "directory": "",
                    "files": [vfn],
                    "revision": revision,
                    "sha256": {vfn: sha256(os.path.abspath(os.path.join(DATA_DIR, vfn)))},
                }
                for v, vfn in versions.items()
            },
        }
    manifest_bytes = (json.dumps(manifest, indent=2) + "\n").encode("utf-8")
    api.upload_file(
        path_or_fileobj=manifest_bytes, path_in_repo="manifest.json",
        repo_id=REPO, repo_type=REPO_TYPE, commit_message="Publish manifest v1",
    )
    print("manifest.json uploaded")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
