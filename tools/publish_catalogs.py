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
manifest. A local file publishes as the version its own metadata states (``metadata.version`` of a
yaml, ``catalog.version`` of an npz). Unchanged published versions are skipped; a version absent
from the hosted entry is uploaded and the entry extended (its newest version becomes the default);
hosted versions and entries absent locally (older versions, retired catalogs) are preserved
verbatim, so pinned stacks keep resolving; and a content change to a published version is refused:
edit the file AND bump its ``metadata.version``, so a pinned ``name@1`` always resolves to
identical bytes. ``MULTI_VERSION`` lists catalogs that keep several versions as local files.
"""
from __future__ import annotations

import hashlib
import json
import os
import re

from huggingface_hub import HfApi, CommitOperationAdd

REPO = "psaegert/symbolic-data-assets"          # MUST match resolver.HF_MANIFEST_REPO
REPO_TYPE = "dataset"
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "assets", "catalogs")

# Forward-only multi-version catalogs: name -> {version: filename}, plus the default version a
# bare name resolves to. v1 files stay byte-identical forever (pinned refs keep resolving);
# content fixes ship as NEW versions. fastsrb@2 = the 2026-07-10 audit repair (III.21.20
# unrealizable, string bounds, missing n_variables); see the changelog inside fastsrb.v2.yaml.
# fastsrb.v2.yaml now holds fastsrb@3 (the 2026-11 source-audit fixes); fastsrb@2 stays hosted.
MULTI_VERSION = {
    "fastsrb": {"versions": {"1": "fastsrb.yaml", "3": "fastsrb.v2.yaml"}, "default": 3},
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
    "constant": ("constant.yaml", 8, "problem_catalog"),
    "grammarvae": ("grammarvae.yaml", 1, "problem_catalog"),
    "jin": ("jin.yaml", 18, "problem_catalog"),
    "keijzer": ("keijzer.yaml", 7, "problem_catalog"),
    "korns": ("korns.yaml", 15, "problem_catalog"),
    "koza": ("koza.yaml", 2, "problem_catalog"),
    "livermore": ("livermore.yaml", 25, "problem_catalog"),
    "livermore2": ("livermore2.yaml", 150, "problem_catalog"),
    "meier": ("meier.yaml", 2, "problem_catalog"),
    "neat": ("neat.yaml", 8, "problem_catalog"),
    # nonic (1) and r-rationals (6) retired 2026-11: their authors pin the points, so they are nonic-points and
    # r-rationals-points (data catalogs)
    "pagie": ("pagie.yaml", 1, "problem_catalog"),
    "poly": ("poly.yaml", 6, "problem_catalog"),
    # sine (1 entry) retired 2026-11: the authors pin Sine's points, so it is the data catalog sine-points
    "vladislavleva": ("vladislavleva.yaml", 6, "problem_catalog"),
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
    "soose-nc": ("soose-nc.yaml", 400, "problem_catalog"),
    "soose-wc": ("soose-wc.yaml", 400, "problem_catalog"),
    "soose-fc": ("soose-fc.yaml", 400, "problem_catalog"),
    # P5 ERBench permissive novel families (MIT/BSD-3; arXiv:2606.09276). The CC-BY-SA families
    # (erbench-oeis, erbench-eponymous) publish to the SEPARATE SA repo via publish_catalogs_sa.py.
    "erbench-syneq": ("erbench-syneq.yaml", 15903, "problem_catalog"),
    "erbench-phybench": ("erbench-phybench.yaml", 270, "problem_catalog"),
    "erbench-densities": ("erbench-densities.yaml", 99, "problem_catalog"),
    # P6 black-box tier (gt_kind='none', eval-only): GT-free cp3-bench discovery sets.
    "cp3-blackbox": ("cp3-blackbox.npz", 11, "problem_catalog"),
    # P6: SRBench 2.0 black-box 12-selection (PMLB, MIT; fri_c0 pair = verified z-scored
    # Friedman-1 references) + the AI-Feynman BONUS set (aifeynman-original rendering, 3rd of 3).
    "srbench2-blackbox": ("srbench2-blackbox.npz", 12, "problem_catalog"),
    "feynman-bonus": ("feynman-bonus.yaml", 20, "problem_catalog"),
    # srbf 2026-11 release: the newly collected benchmark sets as formula catalogs and the companion '-points' data
    # catalogs of problems whose authors fix their points (built by tools/catalog_build/, see its README).
    "ai-feynman-2-0-modular-equations": ("ai-feynman-2-0-modular-equations.yaml", 1, "problem_catalog"),
    "ai-feynman-database-100-main-equations": ("ai-feynman-database-100-main-equations.yaml", 74, "problem_catalog"),
    "ai-hilbert-problem-set": ("ai-hilbert-problem-set.yaml", 9, "problem_catalog"),
    "autora-synthetic-behavioural-science-models-points": ("autora-synthetic-behavioural-science-models-points.npz", 7, "problem_catalog"),
    "bms-synthetic-prior-expression-and-bessel-functions": ("bms-synthetic-prior-expression-and-bessel-functions.yaml", 2, "problem_catalog"),
    "bms-synthetic-prior-expression-and-bessel-functions-points": ("bms-synthetic-prior-expression-and-bessel-functions-points.npz", 1, "problem_catalog"),
    "breiman-i": ("breiman-i.yaml", 1, "problem_catalog"),
    "chen-luo-jiang-block-building-test-cases": ("chen-luo-jiang-block-building-test-cases.yaml", 13, "problem_catalog"),
    "coevolved-fitness-predictors-test-problems-points": ("coevolved-fitness-predictors-test-problems-points.npz", 8, "problem_catalog"),
    "cp3-bench-standard-tuning-datasets-points": ("cp3-bench-standard-tuning-datasets-points.npz", 4, "problem_catalog"),
    "cvgp-trigonometric-control-variable-datasets": ("cvgp-trigonometric-control-variable-datasets.yaml", 1024, "problem_catalog"),
    "ddrsr-simple-test-and-complex-test": ("ddrsr-simple-test-and-complex-test.yaml", 20, "problem_catalog"),
    "decomposable-nsr-synthetic-e1-e13": ("decomposable-nsr-synthetic-e1-e13.yaml", 24, "problem_catalog"),
    "dgsr-additional-feynman": ("dgsr-additional-feynman.yaml", 8, "problem_catalog"),
    "dgsr-feynman-d-2-d-5-subsets": ("dgsr-feynman-d-2-d-5-subsets.yaml", 2, "problem_catalog"),
    "dgsr-synthetic-d-12": ("dgsr-synthetic-d-12.yaml", 7, "problem_catalog"),
    "dso-benchmark-registry": ("dso-benchmark-registry.yaml", 4, "problem_catalog"),
    "dso-benchmark-registry-points": ("dso-benchmark-registry-points.npz", 1, "problem_catalog"),
    "e2e-in-domain-synthetic-validation-set": ("e2e-in-domain-synthetic-validation-set.yaml", 1024, "problem_catalog"),
    "eql-cart-pendulum": ("eql-cart-pendulum.yaml", 8, "problem_catalog"),
    "eql-formulas-f-1-f-4-and-division-formula": ("eql-formulas-f-1-f-4-and-division-formula.yaml", 13, "problem_catalog"),
    "eql-pendulum-vector-field": ("eql-pendulum-vector-field.yaml", 6, "problem_catalog"),
    "eql-robotic-arm-forward-kinematics": ("eql-robotic-arm-forward-kinematics.yaml", 28, "problem_catalog"),
    "erbench-public-legacy-formulas": ("erbench-public-legacy-formulas.yaml", 5, "problem_catalog"),
    "errless-synthetic-posterior-set": ("errless-synthetic-posterior-set.yaml", 7, "problem_catalog"),
    "esr-cosmology-datasets": ("esr-cosmology-datasets.yaml", 1, "problem_catalog"),
    "evomo-sr-random-expression-datasets-p1-p4": ("evomo-sr-random-expression-datasets-p1-p4.yaml", 8, "problem_catalog"),
    "friedman-functions": ("friedman-functions.yaml", 2, "problem_catalog"),
    "friedman-random-function-generator-instances": ("friedman-random-function-generator-instances.yaml", 12, "problem_catalog"),
    "geobench": ("geobench.yaml", 71, "problem_catalog"),
    "gfn-sr-noisy-near-twin-set": ("gfn-sr-noisy-near-twin-set.yaml", 6, "problem_catalog"),
    "gp-benchmarks-sr-table": ("gp-benchmarks-sr-table.yaml", 11, "problem_catalog"),
    "gp-benchmarks-sr-table-points": ("gp-benchmarks-sr-table-points.npz", 3, "problem_catalog"),
    "gplearn-documentation-examples": ("gplearn-documentation-examples.yaml", 1, "problem_catalog"),
    "gpu-lm-inner-constant-recovery-problems": ("gpu-lm-inner-constant-recovery-problems.yaml", 18, "problem_catalog"),
    "gsr-symset": ("gsr-symset.yaml", 17, "problem_catalog"),
    "hep-llm-sr-angular-distribution-problems": ("hep-llm-sr-angular-distribution-problems.yaml", 2, "problem_catalog"),
    "heuristiclab-physics-instances": ("heuristiclab-physics-instances.yaml", 4, "problem_catalog"),
    "heuristiclab-regression-instance-providers": ("heuristiclab-regression-instance-providers.yaml", 14, "problem_catalog"),
    "heuristiclab-regression-instance-providers-points": ("heuristiclab-regression-instance-providers-points.npz", 1, "problem_catalog"),
    "hierbosss-single-atom-catalysis-and-synthetic-polynomial": ("hierbosss-single-atom-catalysis-and-synthetic-polynomial.yaml", 1, "problem_catalog"),
    "kan-feynman-no-units-subset": ("kan-feynman-no-units-subset.yaml", 1, "problem_catalog"),
    "kan-toy-functions": ("kan-toy-functions.yaml", 5, "problem_catalog"),
    "keijzer-points": ("keijzer-points.npz", 8, "problem_catalog"),
    "kim-et-al-eql-benchmark-functions": ("kim-et-al-eql-benchmark-functions.yaml", 12, "problem_catalog"),
    "knowledge-integration-newtonian-cases": ("knowledge-integration-newtonian-cases.yaml", 3, "problem_catalog"),
    "lasr-noisy-feynman-with-distractors": ("lasr-noisy-feynman-with-distractors.yaml", 100, "problem_catalog"),
    "lasr-synthetic-equations": ("lasr-synthetic-equations.yaml", 41, "problem_catalog"),
    "llm-lex-random-mathematica-functions": ("llm-lex-random-mathematica-functions.yaml", 3, "problem_catalog"),
    "llm-lex-random-mathematica-functions-points": ("llm-lex-random-mathematica-functions-points.npz", 20, "problem_catalog"),
    "mcdermott-table-1-predictive-modelling-sets-points": ("mcdermott-table-1-predictive-modelling-sets-points.npz", 1, "problem_catalog"),
    "multi-view-sr-synthetic-benchmarks": ("multi-view-sr-synthetic-benchmarks.yaml", 8, "problem_catalog"),
    "multi-view-sr-synthetic-benchmarks-points": ("multi-view-sr-synthetic-benchmarks-points.npz", 8, "problem_catalog"),
    "neat-points": ("neat-points.npz", 1, "problem_catalog"),
    "newtonbench": ("newtonbench.yaml", 108, "problem_catalog"),
    "nguyen-c-points": ("nguyen-c-points.npz", 1, "problem_catalog"),
    "nonic-points": ("nonic-points.npz", 1, "problem_catalog"),
    "nsrwh-only-five-variables-nc": ("nsrwh-only-five-variables-nc.yaml", 301, "problem_catalog"),
    "nsrwh-only-five-variables-wc": ("nsrwh-only-five-variables-wc.yaml", 301, "problem_catalog"),
    "nsrwh-train-nc": ("nsrwh-train-nc.yaml", 301, "problem_catalog"),
    "nsrwh-train-wc": ("nsrwh-train-wc.yaml", 299, "problem_catalog"),
    "occamnet-analytic-and-non-analytic-suites": ("occamnet-analytic-and-non-analytic-suites.yaml", 22, "problem_catalog"),
    "phye2e-llm-generated-synthetic-formulas": ("phye2e-llm-generated-synthetic-formulas.yaml", 600, "problem_catalog"),
    "physo-astrophysical-case-studies": ("physo-astrophysical-case-studies.yaml", 9, "problem_catalog"),
    "physo-class-sr-benchmark": ("physo-class-sr-benchmark.yaml", 8, "problem_catalog"),
    "pse-high-dimensional-distractor-set": ("pse-high-dimensional-distractor-set.yaml", 20, "problem_catalog"),
    "pysr-empiricalbench-formulas": ("pysr-empiricalbench-formulas.yaml", 4, "problem_catalog"),
    "r-rationals-points": ("r-rationals-points.npz", 6, "problem_catalog"),
    "reproduction-bias-not-included-test-sets": ("reproduction-bias-not-included-test-sets.yaml", 297, "problem_catalog"),
    "rils-rols-random-instances": ("rils-rols-random-instances.yaml", 235, "problem_catalog"),
    "rsrm-constant-optimization-benchmark": ("rsrm-constant-optimization-benchmark.yaml", 93, "problem_catalog"),
    "rsrm-gaussian-cdf-extrapolation-test-points": ("rsrm-gaussian-cdf-extrapolation-test-points.npz", 3, "problem_catalog"),
    "scimed-settling-sphere-configurations": ("scimed-settling-sphere-configurations.yaml", 2, "problem_catalog"),
    "scrbenchmark": ("scrbenchmark.yaml", 15, "problem_catalog"),
    "shape-constrained-sr-synthetic": ("shape-constrained-sr-synthetic.yaml", 7, "problem_catalog"),
    "sine-points": ("sine-points.npz", 1, "problem_catalog"),
    "spl-noise-data-scarcity-grid": ("spl-noise-data-scarcity-grid.yaml", 1, "problem_catalog"),
    "surfacebench-explicit-surfaces": ("surfacebench-explicit-surfaces.yaml", 256, "problem_catalog"),
    "sym-q-ai-feynman-with-drift-terms": ("sym-q-ai-feynman-with-drift-terms.yaml", 15, "problem_catalog"),
    "symantic-synthetic-set": ("symantic-synthetic-set.yaml", 10, "problem_catalog"),
    "symbolicgpt-bivariate-set": ("symbolicgpt-bivariate-set.yaml", 999, "problem_catalog"),
    "symbolicgpt-test-sets": ("symbolicgpt-test-sets.yaml", 987, "problem_catalog"),
    "symbolicregression-com-toy-test-suite-1": ("symbolicregression-com-toy-test-suite-1.yaml", 1, "problem_catalog"),
    "symformer-in-domain-synthetic-evaluation": ("symformer-in-domain-synthetic-evaluation.yaml", 1024, "problem_catalog"),
    "symtree-benchmark-functions": ("symtree-benchmark-functions.yaml", 16, "problem_catalog"),
    "torchsisso-synthetic-expressions": ("torchsisso-synthetic-expressions.yaml", 9, "problem_catalog"),
    "vladislavleva-points": ("vladislavleva-points.npz", 2, "problem_catalog"),
    "white-et-al-2013-proposed-sr-set-points": ("white-et-al-2013-proposed-sr-set-points.npz", 1, "problem_catalog"),
}


def sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_version(path: str) -> int:
    """The version a catalog file states about itself: a yaml's ``metadata.version``, an npz's ``catalog.version``."""
    if path.endswith(".npz"):
        import numpy as np
        return int(json.loads(str(np.load(path)["_meta"]))["catalog"].get("version") or 1)
    with open(path, "r", encoding="utf-8") as handle:
        head = handle.read(1 << 16)                      # the metadata block opens the file
    m = re.search(r"(?m)^  version: (\d+)\s*$", head)
    return int(m.group(1)) if m else 1


def publish(repo: str, data_dir: str, catalogs: dict, multi_version: dict, execute: bool = False,
            extra_ops: tuple = ()) -> None:
    """Incremental, forward-only publish of ``catalogs`` ({name: (path under data_dir, count, type)}) to ``repo``;
    the files land flat at the repo root.

    * a version absent from the hosted entry (or a name absent from the manifest) -> its file uploads and
      the entry is extended (revision = the fresh files commit; default = the newest version);
    * hosted versions and names absent locally -> PRESERVED verbatim, so pinned stacks keep resolving;
    * a version whose local bytes differ from the pinned sha256 -> REFUSED: published versions are
      immutable; bump the file's metadata.version.

    Dry-run by default; ``execute=True`` publishes.
    """
    from lint_catalogs import lint_paths
    assert lint_paths(), "catalog lint failed -- fix errors before publishing"

    def local(rel: str) -> str:
        return os.path.abspath(os.path.join(data_dir, rel))

    api = HfApi()
    manifest_path = api.hf_hub_download(repo_id=repo, repo_type=REPO_TYPE, filename="manifest.json")
    with open(manifest_path, "r", encoding="utf-8") as handle:
        hosted = json.load(handle)
    print(f"hosted manifest ({repo}): {len(hosted)} entries")

    new_versions: dict[str, dict[str, str]] = {}
    unchanged, conflicts = [], []
    for name, (fn, _cnt, _ctype) in catalogs.items():
        spec = multi_version.get(name)
        local_versions = spec["versions"] if spec else {str(file_version(local(fn))): fn}
        for v, rel in local_versions.items():
            assert os.path.isfile(local(rel)), f"missing catalog file: {rel}"
            stated = file_version(local(rel))
            assert stated == int(v), f"{name}: {rel} states version {stated}, MULTI_VERSION says {v}"
        hosted_versions = hosted.get(name, {}).get("versions", {})
        missing, changed = {}, []
        for v, rel in local_versions.items():
            hosted_version = hosted_versions.get(str(v))
            if hosted_version is None:
                missing[str(v)] = rel
            elif hosted_version["sha256"].get(os.path.basename(rel)) != sha256(local(rel)):
                changed.append(v)
        if changed:
            conflicts.append(f"{name}@{','.join(changed)}")
        elif missing:
            new_versions[name] = missing
        else:
            unchanged.append(name)
    new_names = sorted(new_versions)
    preserved = [name for name in hosted if name not in catalogs]

    print(f"new: {', '.join(f'{n}@{v}' for n in new_names for v in new_versions[n])}")
    print(f"unchanged (skipped): {len(unchanged)}")
    print(f"preserved retired entries: {sorted(preserved)}")
    if conflicts:
        raise SystemExit(
            f"REFUSED: content changed for published versions {sorted(conflicts)} -- published "
            f"versions are immutable; bump the file's metadata.version.")
    if not new_names:
        print("nothing to publish")
        return
    if not execute:
        print("dry run (pass --execute to publish)")
        return

    who = api.whoami()["name"]
    print(f"HF user: {who}")

    # 1. upload only the MISSING-version files, in one commit so a single revision pins them
    ops = list(extra_ops)
    uploaded: list[str] = []
    for versions in new_versions.values():
        for rel in versions.values():
            if rel not in uploaded:
                uploaded.append(rel)
                ops.append(CommitOperationAdd(path_in_repo=os.path.basename(rel), path_or_fileobj=local(rel)))
    commit = api.create_commit(
        repo_id=repo, repo_type=REPO_TYPE, operations=ops,
        commit_message=f"Publish {len(uploaded)} catalog artifacts: {', '.join(new_names)}",
    )
    revision = commit.oid
    print(f"files commit: {revision}")

    # 2. merge: hosted entries verbatim (incl. retired names and older versions); a new name gets a
    # fresh entry, an existing one is EXTENDED (published versions untouched)
    manifest = dict(hosted)
    for name, missing in new_versions.items():
        _fn, _cnt, ctype = catalogs[name]
        spec = multi_version.get(name)
        entry = dict(hosted.get(name) or {"type": ctype, "repo_id": repo, "versions": {}})
        entry["versions"] = dict(entry["versions"])
        for v, rel in missing.items():
            entry["versions"][v] = {
                "repo_id": repo,
                "directory": "",
                "files": [os.path.basename(rel)],
                "revision": revision,
                "sha256": {os.path.basename(rel): sha256(local(rel))},
            }
        entry["default_version"] = spec["default"] if spec else max(int(v) for v in entry["versions"])
        manifest[name] = entry
    manifest_bytes = (json.dumps(manifest, indent=2) + "\n").encode("utf-8")
    api.upload_file(
        path_or_fileobj=manifest_bytes, path_in_repo="manifest.json",
        repo_id=repo, repo_type=REPO_TYPE,
        commit_message=f"Manifest: add {', '.join(sorted(new_names))} (hosted entries preserved)",
    )
    print(f"manifest.json uploaded: {len(manifest)} entries "
          f"({len(new_names)} new, {len(preserved)} preserved-retired)")


def main(execute: bool = False) -> None:
    publish(REPO, DATA_DIR, CATALOGS, MULTI_VERSION, execute=execute)


if __name__ == "__main__":
    import sys
    main(execute="--execute" in sys.argv)
