"""Build srbf's SRBench data catalogs: frozen pool catalogs (symbolic_data.pools), one per SRBench track.

srbf's data-catalog rules (srbf ruling 2026-10-17): SRBench is a SOURCE, not a protocol -- srbf takes its datasets, not its
10 or 30 seeds or its 10,000-row training cap. Each run draws at most 512 fitting and 512 test rows afresh. The test
split is the authors' procedure: SRBench 2021 and 2.0 both split every run 75/25 with scikit-learn's
train_test_split, so every problem's rule is {"random": 0.25}. The tracks are named apart (srbf 2026-10-17) so that
nobody reads Feynman into "SRBench": srbf's Feynman is its own formula catalogs.

Inputs are the survey's records (paths.data_set_record): each dataset's pinned URL,
the sha256 of its .tsv.gz (from PMLB's Git LFS pointer), its row count and feature names. Every file is checked
against them before it enters a catalog; a mismatch stops the build.

    build_srbench_catalogs.py [--out build/data-catalogs] [--only srbench-2021-blackbox,...]
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from symbolic_data import data_problem
from symbolic_data.catalog import ProblemCatalog

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths  # noqa: E402  the survey's records

CACHE = paths.CACHE / "pmlb"
SPLIT = {"random": 0.25}

TRACKS = {
    # name: (survey set tag, display name, authors' split as published)
    "srbench-2021-blackbox": ("done-0", "SRBench 2021 black-box",
                              "train_test_split(train_size=0.75, test_size=0.25), one split per seed (SRBench 2021)"),
    "srbench-2.0-blackbox": ("done-7", "SRBench 2.0 black-box",
                             "train_test_split(train_size=0.75, test_size=0.25), one split per run (SRBench 2.0)"),
    "srbench-2.0-first-principles": ("done-8", "SRBench 2.0 first-principles",
                                     "train_test_split(train_size=0.75, test_size=0.25), one split per run (SRBench 2.0)"),
}
# SRBench 2021's Strogatz problems (srbf 2026-10-17: Strogatz as people usually use it):
# PMLB's 14 strogatz_* files, added in one commit and untouched since; the formula is each file's metadata.yaml.
STROGATZ_COMMIT = "00f6bd3f970c63771ebc42688b00711363734f31"
STROGATZ = ["strogatz_bacres1", "strogatz_bacres2", "strogatz_barmag1", "strogatz_barmag2", "strogatz_glider1",
            "strogatz_glider2", "strogatz_lv1", "strogatz_lv2", "strogatz_predprey1", "strogatz_predprey2",
            "strogatz_shearflow1", "strogatz_shearflow2", "strogatz_vdp1", "strogatz_vdp2"]
STROGATZ_SPLIT = "train_test_split(train_size=0.75, test_size=0.25), one split per seed (SRBench 2021 ground-truth problems)"
# srbf 2026-10-17: SRBench 2.0 first-principles without supernovae_zg (redundant with its sibling)
LEAVE_OUT = {"srbench-2.0-first-principles": {"first_principles_supernovae_zg"}}
# Datasets whose source permits redistribution only with its permission note attached: the note travels in the
# problem's metadata (and in the dataset card's notices). Tecator's statement, section 1 of the StatLib file
# (OpenML 505's description quotes it).
PERMISSION_NOTES = {
    "505_tecator": (
        "Statement of permission from Tecator (the original data source): These data are recorded on a Tecator "
        "Infratec Food and Feed Analyzer working in the wavelength range 850 - 1050 nm by the Near Infrared "
        "Transmission (NIT) principle. Each sample contains finely chopped pure meat with different moisture, fat and "
        "protein contents. If results from these data are used in a publication we want you to mention the "
        "instrument and company name (Tecator) in the publication. In addition, please send a preprint of your "
        "article to Karin Thente, Tecator AB, Box 70, S-263 21 Hoganas, Sweden. The data are available in the public "
        "domain with no responsability from the original data source. The data can be redistributed as long as this "
        "permission note is attached. For more information about the instrument - call Perstorp Analytical's "
        "representative in your area."),
}


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fetch(entry: dict) -> bytes:
    """The dataset's .tsv.gz bytes: the recorded local file, else the cache, else the pinned URL; checked by sha256."""
    want = entry["sha256"]
    local = paths.input_file(entry["file"]) if entry.get("file") else None
    for path in [local, CACHE / f"{want}.tsv.gz"]:
        if path and os.path.exists(path):
            data = Path(path).read_bytes()
            if _sha256(data) == want:
                return data
    with urllib.request.urlopen(entry["url"], timeout=120) as resp:
        data = resp.read()
    got = _sha256(data)
    if got != want:
        raise SystemExit(f"{entry['id']}: sha256 {got} != recorded {want} ({entry['url']})")
    CACHE.mkdir(parents=True, exist_ok=True)
    (CACHE / f"{want}.tsv.gz").write_bytes(data)
    return data


def problem_of(entry: dict, track: str, display: str, split_text: str, set_meta: dict) -> object:
    data = fetch(entry)
    frame = pd.read_csv(io.BytesIO(gzip.decompress(data)), sep="\t")
    if "target" not in frame.columns:
        raise SystemExit(f"{entry['id']}: no 'target' column")
    features = [c for c in frame.columns if c != "target"]
    if entry.get("feature_names") and list(entry["feature_names"]) != features:
        raise SystemExit(f"{entry['id']}: features {features[:5]}... differ from the record {entry['feature_names'][:5]}...")
    if entry.get("n_rows") and int(entry["n_rows"]) != len(frame):
        raise SystemExit(f"{entry['id']}: {len(frame)} rows, record says {entry['n_rows']}")
    x = frame[features].to_numpy(dtype=np.float64)
    y = frame["target"].to_numpy(dtype=np.float64)
    if not (np.isfinite(x).all() and np.isfinite(y).all()):
        raise SystemExit(f"{entry['id']}: non-finite values")
    meta = {
        "catalog": track, "catalog_display": display, "source": entry.get("source"), "url": entry["url"],
        "sha256": entry["sha256"], "pmlb_name": entry.get("pmlb_name"), "feature_names": features,
        "subset": entry.get("subset"), "split_as_published": split_text, "n_rows": int(len(frame)),
        **({"permission_note": PERMISSION_NOTES[entry["id"]]} if entry["id"] in PERMISSION_NOTES else {}),
    }
    if entry.get("reference_law"):
        # information only (the authors' accepted hypothesis, not used to score runs): kept for readers
        law = entry["reference_law"]
        meta["reference_law"] = {k: law.get(k) for k in ("formula_sympy", "formula_as_published_code", "role", "notes")}
    return data_problem(x, y, variables=[f"v{i + 1}" for i in range(len(features))], test_split=SPLIT,
                        eq_id=entry["id"], gt_kind="none", meta=meta)


def _get(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=120) as resp:
        return resp.read()


def rewrite_cot(text: str) -> str:
    """cot(u) -> (1/tan(u)), exactly (srbf's operator list, 2026-10-21: cot needs no operator of its own)."""
    out, i = [], 0
    while True:
        j = text.find("cot(", i)
        if j < 0 or (j > 0 and (text[j - 1].isalnum() or text[j - 1] == "_")):
            if j < 0:
                return "".join(out) + text[i:]
            out.append(text[i:j + 4])
            i = j + 4
            continue
        depth, k = 1, j + 4
        while depth:
            depth += {"(": 1, ")": -1}.get(text[k], 0)
            k += 1
        out.append(text[i:j] + "(1/tan(" + rewrite_cot(text[j + 4:k - 1]) + "))")
        i = k


def strogatz_problem(name: str, engine) -> object:
    """One Strogatz problem: the states (x, y) and the rate of change, with the formula of its metadata.yaml."""
    import re

    import yaml
    from symbolic_data._evaluation import compile_expression, evaluate
    base = f"https://raw.githubusercontent.com/EpistasisLab/pmlb/{STROGATZ_COMMIT}/datasets/{name}"
    pointer = _get(f"{base}/{name}.tsv.gz").decode()
    want = re.search(r"oid sha256:([0-9a-f]{64})", pointer).group(1)
    entry = {"id": name, "sha256": want,
             "url": f"https://media.githubusercontent.com/media/EpistasisLab/pmlb/{STROGATZ_COMMIT}/datasets/{name}/{name}.tsv.gz"}
    data = fetch(entry)
    frame = pd.read_csv(io.BytesIO(gzip.decompress(data)), sep="\t")
    features = [c for c in frame.columns if c != "target"]
    meta_yaml = yaml.safe_load(_get(f"https://raw.githubusercontent.com/EpistasisLab/pmlb/master/datasets/{name}/metadata.yaml"))
    line = [ln.strip() for ln in meta_yaml["description"].splitlines() if re.match(r"^[xy]' *=", ln.strip())]
    if len(line) != 1:
        raise SystemExit(f"{name}: expected one formula line in metadata.yaml, got {line}")
    formula = rewrite_cot(line[0].split("=", 1)[1].strip())
    rename = {f: f"v{i + 1}" for i, f in enumerate(features)}
    prepared = re.sub(r"\b(" + "|".join(map(re.escape, features)) + r")\b", lambda m: rename[m.group(1)], formula)
    compiled = compile_expression(engine, name, prepared, {v: {} for v in rename.values()}, name="srbench-2021-strogatz")
    x = frame[features].to_numpy(dtype=np.float64)
    y = frame["target"].to_numpy(dtype=np.float64)
    pred = np.asarray(evaluate(compiled, {rename[f]: x[:, i] for i, f in enumerate(features)}), dtype=np.float64).ravel()
    fvu = float(np.mean((y - pred) ** 2) / np.var(y))
    # the targets are the formula evaluated on the simulated states (exact) or a numerical estimate of the rate
    gt_kind = "exact" if fvu <= float(np.finfo(np.float32).eps) else "reference"
    meta = {"catalog": "srbench-2021-strogatz", "catalog_display": "SRBench 2021 Strogatz", "url": entry["url"],
            "sha256": want, "pmlb_name": name, "pmlb_commit": STROGATZ_COMMIT, "feature_names": features,
            "formula_as_published": line[0], "formula_fvu_on_file": fvu, "split_as_published": STROGATZ_SPLIT,
            "n_rows": int(len(frame)), "source": meta_yaml.get("source")}
    problem = data_problem(x, y, variables=list(rename.values()), test_split=SPLIT, eq_id=name,
                           expression=compiled["expression"], skeleton=compiled["prefix"], constants=compiled["constants"],
                           gt_kind=gt_kind, meta=meta)
    if gt_kind == "reference":
        problem.y_reference_support = pred.reshape(-1, 1)
        problem.y_reference_validation = np.empty((0, 1))
    print(f"  {name}: {x.shape} formula {formula!r} -> FVU on the file {fvu:.3g} ({gt_kind})", flush=True)
    return problem


def build_strogatz(out: Path) -> dict:
    from symbolic_data._evaluation import load_engine
    engine = load_engine(None)
    problems = [strogatz_problem(n, engine) for n in STROGATZ]
    meta = {"display_name": "SRBench 2021 Strogatz", "kind": "data", "test_split": SPLIT, "split_as_published": STROGATZ_SPLIT,
            "pmlb_commit": STROGATZ_COMMIT, "built_from": "build_srbench_catalogs.py",
            "rules": "srbf data catalogs (2026-10-17): pools, <= 512 fitting + 512 test rows drawn per run"}
    path = ProblemCatalog.from_problems(problems, name="srbench-2021-strogatz", version=1, meta=meta).save(out / "srbench-2021-strogatz.npz")
    return {"track": "srbench-2021-strogatz", "problems": len(problems), "rows": int(sum(p.x_support.shape[0] for p in problems)),
            "path": str(path), "bytes": os.path.getsize(path), "gt_kinds": sorted({p.gt_kind for p in problems})}


def build(track: str, out: Path) -> dict:
    tag, display, split_text = TRACKS[track]
    rec = paths.data_set_record(tag)
    entries = [e for e in rec["datasets"] if e["id"] not in LEAVE_OUT.get(track, set())]
    problems = []
    for i, e in enumerate(entries):
        problems.append(problem_of(e, track, display, split_text, rec))
        print(f"  {track} {i + 1}/{len(entries)} {e['id']}: {problems[-1].x_support.shape}", flush=True)
    meta = {"display_name": display, "kind": "data", "test_split": SPLIT, "split_as_published": split_text,
            "source_record": f"catalog-prep/sets/{tag}.json", "built_from": "build_srbench_catalogs.py",
            "rules": "srbf data catalogs (2026-10-17): pools, <= 512 fitting + 512 test rows drawn per run"}
    path = ProblemCatalog.from_problems(problems, name=track, version=1, meta=meta).save(out / f"{track}.npz")
    return {"track": track, "problems": len(problems), "rows": int(sum(p.x_support.shape[0] for p in problems)),
            "path": str(path), "bytes": os.path.getsize(path)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(paths.DATA_BUILD))
    ap.add_argument("--only", default=None)
    a = ap.parse_args()
    out = Path(os.path.expanduser(a.out))
    out.mkdir(parents=True, exist_ok=True)
    tracks = a.only.split(",") if a.only else list(TRACKS) + ["srbench-2021-strogatz"]
    report = [build_strogatz(out) if t == "srbench-2021-strogatz" else build(t, out) for t in tracks]
    print(json.dumps(report, indent=1))
    (out / "build_srbench_report.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    sys.exit(main())
