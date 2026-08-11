"""Publish the curated catalogs to the Hugging Face assets dataset repo + a versioned manifest.

This is the distribution step for ``symbolic_data``'s curated catalogs: they live as artifacts on
Hugging Face (the artifact store), NOT in the PyPI wheel. The wheel ships NO catalog copies (pure-HF
since 0.8.0); the canonical, versioned, sha256-integrity-checked source of truth is the HF dataset
repo, and the repo keeps the source-of-truth yamls under ``assets/catalogs/`` (publish source + the
tests' local fixtures).

Layout (flat repo root): ``<name>.yaml`` for each catalog + ``manifest.json``. Each manifest entry
pins the catalog's content by ``revision`` (the git commit sha of the files commit) and per-file
``sha256``, exactly what ``symbolic_data.resolver.resolve`` verifies on download.

Run from the repo root: ``python tools/publish_catalogs.py`` (dry run; add ``--execute`` to
publish — requires HF auth: ``huggingface_hub.whoami``).

Versioning discipline (forward-only, enforced): the publish is INCREMENTAL against the hosted
manifest. Unchanged published versions are skipped; a version absent from the hosted entry is
uploaded and the entry extended; hosted entries absent from CATALOGS (retired catalogs) are
preserved verbatim so pinned legacy stacks keep resolving; and a content change to a published
version is refused — ship it as a NEW version (add ``"2": {...}`` to ``MULTI_VERSION`` and bump
``default``), so a pinned ``name@1`` always resolves to identical bytes.
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
    # Generation-2 open training recipe + its unsimplified benchmark twin (0.14.0). The
    # generation-1 pair (lample-charton-v23, v23-val) left the repo with the hyper-operator
    # vocabulary; their hosted, revision-pinned manifest entries are PRESERVED by the merge
    # logic below so the legacy stack (symbolic-data < 0.14 + simplipy < 0.12) keeps resolving.
    "lample-charton-v24": ("lample-charton-v24.yaml", None, "generative_catalog"),
    "lample-charton-v24-bench": ("lample-charton-v24-bench.yaml", None, "generative_catalog"),
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


def main(execute: bool = False) -> None:
    """Incremental, forward-only publish: hosted state is the baseline, never rebuilt.

    * a CATALOGS name absent from the hosted manifest -> its file uploads and a new entry
      is added (revision = the fresh files commit);
    * a hosted name absent from CATALOGS (a retired catalog) -> its entry is PRESERVED
      verbatim, so pinned legacy stacks keep resolving;
    * a hosted name whose local file bytes differ from the pinned sha256 -> REFUSED:
      published versions are immutable, ship the change as a NEW version in MULTI_VERSION.

    Dry-run by default; pass ``--execute`` to publish.
    """
    from lint_catalogs import lint_paths
    assert lint_paths(), "catalog lint failed -- fix errors before publishing"
    import json as _json

    api = HfApi()
    manifest_path = api.hf_hub_download(repo_id=REPO, repo_type=REPO_TYPE, filename="manifest.json")
    with open(manifest_path, "r", encoding="utf-8") as handle:
        hosted = _json.load(handle)
    print(f"hosted manifest: {len(hosted)} entries")

    # Per-VERSION comparison: an unchanged published version skips, a version absent from
    # the hosted entry publishes (extending the entry), a content change to a published
    # version is refused. `new_versions[name]` maps version -> filename to upload.
    new_versions: dict[str, dict[str, str]] = {}
    unchanged, conflicts = [], []
    for name, (fn, _cnt, _ctype) in CATALOGS.items():
        spec = MULTI_VERSION.get(name)
        local_versions = spec["versions"] if spec else {"1": fn}
        for vfn in local_versions.values():
            assert os.path.isfile(os.path.abspath(os.path.join(DATA_DIR, vfn))), f"missing catalog file: {vfn}"
        hosted_versions = hosted.get(name, {}).get("versions", {})
        missing, changed = {}, []
        for v, vfn in local_versions.items():
            local_sha = sha256(os.path.abspath(os.path.join(DATA_DIR, vfn)))
            hosted_version = hosted_versions.get(str(v))
            if hosted_version is None:
                missing[str(v)] = vfn
            elif hosted_version["sha256"].get(vfn) != local_sha:
                changed.append(v)
        if changed:
            conflicts.append(name)
        elif missing:
            new_versions[name] = missing
        else:
            unchanged.append(name)
    new_names = sorted(new_versions)
    preserved = [name for name in hosted if name not in CATALOGS]

    print(f"new: {sorted(new_names)}")
    print(f"unchanged (skipped): {len(unchanged)}")
    print(f"preserved retired entries: {sorted(preserved)}")
    if conflicts:
        raise SystemExit(
            f"REFUSED: content changed for published catalogs {sorted(conflicts)} -- published "
            f"versions are immutable; ship the change as a NEW version via MULTI_VERSION.")
    if not new_names:
        print("nothing to publish")
        return
    if not execute:
        print("dry run (pass --execute to publish)")
        return

    who = api.whoami()["name"]
    print(f"HF user: {who}")

    # 1. upload only the MISSING-version files, in one commit so a single revision pins them
    ops = []
    filenames: list[str] = []
    for versions in new_versions.values():
        for vfn in versions.values():
            if vfn not in filenames:
                filenames.append(vfn)
    for fn in filenames:
        local = os.path.abspath(os.path.join(DATA_DIR, fn))
        ops.append(CommitOperationAdd(path_in_repo=fn, path_or_fileobj=local))
    commit = api.create_commit(
        repo_id=REPO, repo_type=REPO_TYPE, operations=ops,
        commit_message=f"Publish {len(filenames)} catalog artifacts: {', '.join(new_names)}",
    )
    revision = commit.oid
    print(f"files commit: {revision}")

    # 2. merge: hosted entries verbatim (incl. retired names); a new name gets a fresh
    # entry, an existing multi-version name is EXTENDED (published versions untouched)
    manifest = dict(hosted)
    for name, missing in new_versions.items():
        fn, _cnt, ctype = CATALOGS[name]
        spec = MULTI_VERSION.get(name)
        entry = dict(hosted.get(name) or {"type": ctype, "repo_id": REPO, "versions": {}})
        entry["default_version"] = spec["default"] if spec else 1
        entry["versions"] = dict(entry["versions"])
        for v, vfn in missing.items():
            entry["versions"][v] = {
                "repo_id": REPO,
                "directory": "",
                "files": [vfn],
                "revision": revision,
                "sha256": {vfn: sha256(os.path.abspath(os.path.join(DATA_DIR, vfn)))},
            }
        manifest[name] = entry
    manifest_bytes = (json.dumps(manifest, indent=2) + "\n").encode("utf-8")
    api.upload_file(
        path_or_fileobj=manifest_bytes, path_in_repo="manifest.json",
        repo_id=REPO, repo_type=REPO_TYPE,
        commit_message=f"Manifest: add {', '.join(sorted(new_names))} (hosted entries preserved)",
    )
    print(f"manifest.json uploaded: {len(manifest)} entries "
          f"({len(new_names)} new, {len(preserved)} preserved-retired)")


if __name__ == "__main__":
    import sys
    main(execute="--execute" in sys.argv)
