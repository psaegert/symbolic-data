"""Build the ``cp3-blackbox`` catalog: the GT-FREE discovery datasets of cp3-bench (P6 tier).

11 frozen datasets from CP3-Origins/Things-to-bench (MIT, commit 69a45dd) whose ground truth the
paper (arXiv:2406.15531) explicitly declares UNKNOWN: C3a-f (two-region backreaction quantities
Q_D / R_D from numerical simulation; C3g-h, the two members WITH closed forms, live in
cp3-cosmo) and C4a-e (light-propagation / single-ray statistics with intrinsic statistical
error). Black-box problems: ``gt_kind="none"``, EVAL-ONLY (nothing to hold out, by definition;
FVU-family metrics only, no recovery metrics).

Upstream quirks preserved faithfully (documented per entry): the C4c column literally named
``OomQ`` (upstream typo), C3e/C3f's constant H0 column (0.0715898516 everywhere, superfluous by
construction), and C4a-e's leading rows with target exactly 0.

Run from the repo root: ``PYTHONPATH=src python tools/build_cp3_blackbox.py``
"""
from __future__ import annotations

import gzip
import io
import os

import numpy as np

from symbolic_data import Problem, ProblemCatalog

HERE = os.path.dirname(os.path.abspath(__file__))
UPSTREAM = os.path.join(HERE, "..", "assets", "upstream", "cp3_bench")
OUT_NPZ = os.path.join(HERE, "..", "assets", "catalogs", "cp3-blackbox.npz")
OUT_REPORT = os.path.join(HERE, "..", "assets", "catalogs", "CP3_BLACKBOX_REPORT.md")

SOURCE = ("cp3-bench / Things-to-bench (Thing & Koksbang, arXiv:2406.15531; "
          "github.com/CP3-Origins/Things-to-bench, MIT, commit 69a45dd)")

ENTRIES = {
    "C3a": ["10^2 * Q_D (kinematical backreaction) vs mean z; two-region toy model f_o=0.2, "
            "numerically simulated; GT unknown per paper."],
    "C3b": ["10^8 * R_D (averaged spatial curvature) vs mean z; GT unknown per paper."],
    "C3c": ["10^2 * Q_D vs (f_o, z); GT unknown per paper."],
    "C3d": ["10^8 * R_D vs (f_o, z); GT unknown per paper."],
    "C3e": ["10^2 * Q_D vs 5 features; the H0 column is CONSTANT (0.0715898516) -- superfluous "
            "by construction; GT unknown per paper."],
    "C3f": ["10^8 * R_D, same features as C3e; GT unknown per paper."],
    "C4a": ["Light-propagation statistic vs z; intrinsic single-ray statistical error; GT "
            "unknown per paper."],
    "C4b": ["Light-propagation statistic vs (f, z); GT unknown per paper."],
    "C4c": ["Light-propagation statistic; upstream column 'OomQ' is a verbatim upstream typo "
            "(kept for fidelity); GT unknown per paper."],
    "C4d": ["Light-propagation statistic; Hav feature spans [0.0716, 0.134]; GT unknown per "
            "paper."],
    "C4e": ["Light-propagation statistic; GT unknown per paper."],
}


def main() -> None:
    problems, rows = [], []
    for eq_id, notes in ENTRIES.items():
        raw = np.genfromtxt(io.TextIOWrapper(gzip.open(os.path.join(UPSTREAM, f"{eq_id}.csv.gz"))),
                            delimiter=",", names=True)
        cols = [c for c in raw.dtype.names if c != "target"]
        X = np.stack([raw[c].astype(np.float64) for c in cols], axis=1)
        y = raw["target"].astype(np.float64)
        assert np.all(np.isfinite(X)) and np.all(np.isfinite(y)), eq_id
        assert np.all(np.isfinite(X.astype(np.float32))) and np.all(np.isfinite(y.astype(np.float32))), eq_id

        problem = Problem.from_data(
            X, y, eq_id=eq_id,
            meta={
                "upstream_csv": f"cosmo_data/{eq_id}.csv",
                "columns": cols,
                "law_source": SOURCE,
                "gt_status": "unknown per the paper (discovery dataset)",
                "license": "MIT (Things-to-bench)",
                "notes": notes,
            },
        )
        assert problem.gt_kind == "none" and problem.skeleton is None, eq_id
        problems.append(problem)
        rows.append(f"| {eq_id} | {len(y)} | {len(cols)} | {', '.join(cols)} |")
        print(f"{eq_id} n={len(y):6d} d={len(cols)} cols={cols}")

    catalog = ProblemCatalog.from_problems(
        problems, name="cp3-blackbox", version=1,
        meta={
            "description": "GT-free discovery subset of cp3-bench (Things-to-bench): two-region "
                           "backreaction (C3a-f) + light-propagation statistics (C4a-e). "
                           "Black-box: gt_kind='none', EVAL-ONLY (FVU-family metrics; no "
                           "recovery metrics, nothing to hold out).",
            "license": "MIT (CP3-Origins/Things-to-bench)",
            "builder": "tools/build_cp3_blackbox.py",
            "engine": "dev_7-3",
            "eval_tier": "extended",
            "split_policy": "all upstream points are support; no validation split",
            "companion": "the 17 known-GT siblings live in cp3-cosmo@1",
        })
    out = catalog.save(OUT_NPZ)
    with open(OUT_REPORT, "w", encoding="utf-8") as handle:
        handle.write("\n".join([
            "# cp3-blackbox build report", "",
            f"{SOURCE}. GT-free discovery sets; gt_kind='none', eval-only.", "",
            "| eq_id | n | d | columns |", "|---|---|---|---|",
        ] + rows + ["", f"{len(problems)} problems -> `{os.path.basename(str(out))}`."]) + "\n")
    print(f"\nSaved {out} + report.")


if __name__ == "__main__":
    main()
