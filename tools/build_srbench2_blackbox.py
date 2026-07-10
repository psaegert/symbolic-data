"""Build the ``srbench2-blackbox`` catalog: the SRBench 2.0 black-box 12-selection (P6 tier).

Source: "Call for Action: Towards the Next Generation of Symbolic Regression Benchmark" (Aldeia
et al., GECCO 2025 workshop = SRBench 2.0), paper Table 1 == datasets/download_data.py
['blackbox'] on the srbench_2025 branch (verbatim agreement, adversarially verified). Data:
PMLB (MIT), vendored at the pinned commit in PMLB_REVISION; frozen RAW arrays (SRBench's
75/25-split + StandardScaler(x,y) + 30-seed protocol is an EVALUATION protocol, recorded as
provenance, never baked into the data).

Friedman carve-out (resolves the holdout question): the selection contains 3 fri_* datasets
whose generator family (Friedman-1: 10*sin(pi*x1*x2) + 20*(x3-0.5)^2 + 10*x4 + 5*x5 + e) is
publicly documented on OpenML. For the c0 (independent-features) variants the law is directly
testable on the stored columns: where it fits (FVU consistent with the e~N(0,1) noise), the
entry imports as gt_kind="reference" with the Friedman-1 reference predictions -- so the
skeleton joins the v24 union holdout. The c2 variant's colinearity transform is unpublished
(the law is NOT reconstructible on the stored features): it stays gt_kind="none" with a
meta.gt_family flag; its skeleton is held out via the c0 siblings anyway.

Run from the repo root: ``PYTHONPATH=src python tools/build_srbench2_blackbox.py``
"""
from __future__ import annotations

import gzip
import io
import os

import numpy as np

from symbolic_data import Problem, ProblemCatalog
from symbolic_data._evaluation import compile_expression, load_engine

HERE = os.path.dirname(os.path.abspath(__file__))
UPSTREAM = os.path.join(HERE, "..", "assets", "upstream", "srbench2_blackbox")
OUT_NPZ = os.path.join(HERE, "..", "assets", "catalogs", "srbench2-blackbox.npz")
OUT_REPORT = os.path.join(HERE, "..", "assets", "catalogs", "SRBENCH2_BLACKBOX_REPORT.md")

SOURCE = ("SRBench 2.0 black-box selection (Aldeia et al., GECCO 2025 workshop; "
          "github.com/cavalab/srbench @ srbench_2025) — data from PMLB (MIT, "
          "github.com/EpistasisLab/pmlb, commit pinned in PMLB_REVISION)")

DATASETS = {
    "1028_SWD": "SWD ordinal-regression data (integer target, 4 levels); 1 binary + 9 "
                "categorical features stored numerically.",
    "1089_USCrime": "1960 FBI UCR crime statistics, 47 US states (Vandaele 1978, openml d/1089).",
    "1193_BNG_lowbwt": "OpenML BNG(lowbwt): Bayesian-network-generated from the low-birth-weight "
                       "data; synthetic but genuinely black-box (no symbolic GT).",
    "1199_BNG_echoMonths": "OpenML BNG(echoMonths): BN-generated from echocardiogram survival "
                           "months; synthetic, no symbolic GT.",
    "192_vineyard": "Lake Erie vineyard harvest (Simonoff 1996, openml d/192).",
    "210_cloud": "cloud-seeding rainfall experiment; 1 binary + 1 categorical feature.",
    "522_pm10": "Oslo road air pollution (target is log PM10 concentration upstream; paper "
                "Table 1's R+ codomain note is cosmetic).",
    "557_analcatdata_apnea1": "sleep-apnea scoring agreement (Simonoff); 2 of 3 features are "
                              "categorical IDs.",
    "579_fri_c0_250_5": "Friedman-1 family, c0 = independent features, all 5 relevant.",
    "606_fri_c2_1000_10": "Friedman-1 family, colinearity degree 2 (UNPUBLISHED transform), "
                          "5 relevant + 5 correlated/irrelevant of 10 features.",
    "650_fri_c0_500_50": "Friedman-1 family, c0, 5 relevant + 45 random padding features "
                         "(feature-selection stress case).",
    "678_visualizing_environmental": "ozone/radiation/temperature environmental set (Cleveland).",
}
FRIEDMAN_C0 = {"579_fri_c0_250_5", "650_fri_c0_500_50"}
# PMLB stores the fri_* data Z-SCORED (features and target; verified: means 0, sds 1, and the
# affine-wrapped law fits with residual sd == fitted scale, i.e. the documented e~N(0,1) noise).
# u_i = x_i/sqrt(12) + 0.5 maps the stored features back to the generator's U[0,1] draws.
_U = "(0.28867513459481287*v{i}+0.5)"
FRIEDMAN_CORE = ("10*sin(3.141592653589793*" + _U.format(i=1) + "*" + _U.format(i=2) + ")"
                 "+20*((" + _U.format(i=3) + "-0.5)**(2))"
                 "+10*" + _U.format(i=4) + "+5*" + _U.format(i=5))


def main() -> None:
    engine = load_engine("dev_7-3")
    problems, rows = [], []
    for eq_id, note in DATASETS.items():
        raw = np.genfromtxt(io.TextIOWrapper(gzip.open(os.path.join(UPSTREAM, f"{eq_id}.tsv.gz"))),
                            delimiter="\t", names=True)
        cols = [c for c in raw.dtype.names if c != "target"]
        X = np.stack([raw[c].astype(np.float64) for c in cols], axis=1)
        y = raw["target"].astype(np.float64)
        assert np.all(np.isfinite(X.astype(np.float32))) and np.all(np.isfinite(y.astype(np.float32))), eq_id

        gt_kind, extra_meta, expression, skeleton, constants, y_ref = "none", {}, None, None, None, None
        if eq_id in FRIEDMAN_C0:
            # fit the affine wrap (y is z-scored too): y = scale*friedman1(u(x)) + offset
            core_vars = {f"v{i}": {"name": cols[i - 1]} for i in range(1, 6)}
            core = compile_expression(engine, f"{eq_id}:core", FRIEDMAN_CORE, core_vars,
                                      name="srbench2-blackbox")
            f1 = np.broadcast_to(np.asarray(
                core["callable"](*[X[:, i] for i in range(5)]), dtype=np.float64), y.shape)
            A = np.stack([f1, np.ones_like(f1)], axis=1)
            (scale, offset), *_ = np.linalg.lstsq(A, y, rcond=None)
            infix = f"{float(scale)!r}*({FRIEDMAN_CORE})+{float(offset)!r}"
            compiled = compile_expression(engine, eq_id, infix, core_vars,
                                          name="srbench2-blackbox")
            y_law = np.broadcast_to(np.asarray(
                compiled["callable"](*[X[:, i] for i in range(5)]), dtype=np.float64), y.shape)
            fvu = float(np.mean((y - y_law) ** 2) / np.var(y))
            resid_sd = float(np.std(y - y_law))
            print(f"  {eq_id}: affine-Friedman-1 scale={scale:.4f} FVU={fvu:.4f} "
                  f"resid_sd={resid_sd:.4f} (scaled e~N(0,1) predicts ~{abs(scale):.4f})")
            assert fvu < 0.2 and 0.5 < resid_sd / abs(scale) < 2.0,                 (eq_id, "z-scored Friedman-1 no longer verifies -- data changed?")
            gt_kind = "reference"
            expression = list(compiled["expression"])
            skeleton = tuple(compiled["prefix"])
            constants = list(compiled["constants"])
            y_ref = y_law
            extra_meta = {
                "gt_family": "friedman1",
                "reference_law": infix,
                "prepared_infix": infix,
                "reference_fvu_build": fvu,
                "notes_friedman": "PMLB stores this dataset Z-SCORED (features + target); the "
                                  "reference law composes the documented Friedman-1 generator "
                                  "with the affine de-standardization (u = x/sqrt(12) + 0.5) and "
                                  "a fitted output scale/offset. Residual sd equals the scaled "
                                  "unit noise, verifying the identification. Joins the union "
                                  "holdout.",
            }
        elif eq_id == "606_fri_c2_1000_10":
            extra_meta = {"gt_family": "friedman1",
                          "notes_friedman": "colinearity-degree-2 variant: the feature transform "
                                            "is unpublished, the law is not reconstructible on "
                                            "the stored columns; the Friedman-1 skeleton is held "
                                            "out via the c0 siblings."}

        problem = Problem.from_data(
            X, y, eq_id=eq_id,
            expression=expression, skeleton=skeleton,
            constants=constants if constants else None,
            gt_kind=gt_kind, y_reference_support=y_ref,
            meta={
                "pmlb_dataset": eq_id,
                "columns": cols,
                "law_source": SOURCE,
                "license": "MIT (PMLB)",
                "protocol_provenance": "SRBench 2.0 evaluates with 75/25 splits, 30 fixed seeds, "
                                       "StandardScaler on X and y (function defaults; CLI flags "
                                       "differ), 3-fold CV tuning — protocol, not data; arrays "
                                       "here are RAW.",
                "notes": [note],
                **extra_meta,
            },
        )
        problems.append(problem)
        rows.append(f"| {eq_id} | {len(y)} | {len(cols)} | {problem.gt_kind} |")
        print(f"{eq_id:30s} n={len(y):6d} d={len(cols):2d} gt_kind={problem.gt_kind}")

    catalog = ProblemCatalog.from_problems(
        problems, name="srbench2-blackbox", version=1,
        meta={
            "description": "SRBench 2.0 black-box 12-selection (PMLB data, frozen raw arrays): "
                           "7 real-world + 2 Bayesian-network synthetic + 3 Friedman-family "
                           "(c0 variants imported as gt_kind='reference' where the documented "
                           "Friedman-1 law verifies; c2 stays black-box with a family flag).",
            "license": "MIT (PMLB; selection list from the SRBench 2.0 paper/repo — facts only, "
                       "no GPL code or data imported)",
            "builder": "tools/build_srbench2_blackbox.py",
            "engine": "dev_7-3",
            "eval_tier": "headline (the paper's curated selection)",
            "split_policy": "all upstream points are support; no validation split; SRBench's "
                            "split/scaling protocol recorded per entry as provenance",
        })
    out = catalog.save(OUT_NPZ)
    with open(OUT_REPORT, "w", encoding="utf-8") as handle:
        handle.write("\n".join([
            "# srbench2-blackbox build report", "", SOURCE, "",
            "| dataset | n | d | gt_kind |", "|---|---|---|---|",
        ] + rows + ["", f"{len(problems)} problems -> `{os.path.basename(str(out))}`."]) + "\n")
    print(f"\nSaved {out} + report.")


if __name__ == "__main__":
    main()
