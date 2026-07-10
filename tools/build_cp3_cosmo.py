"""Build the ``cp3-cosmo`` catalog: the known-ground-truth cosmology subset of cp3-bench.

17 frozen datasets from CP3-Origins/Things-to-bench (MIT, commit 69a45dd), the data companion of
"cp3-bench: A tool for benchmarking symbolic regression algorithms tested with cosmology"
(Thing & Koksbang, arXiv:2406.15531): LCDM H(z) (C1a-d), FLRW redshift drift (C2a-b), two-region
backreaction (C3g-h, the only C3 members WITH a closed form), NFW/Burkert halo profiles (C5a-f),
and GW-inspired h_plus toys (C6a-c). The GT-free discovery sets (C3a-f, C4a-e) are NOT imported
here (black-box tier, later); the F1-F8 GP-toy re-renderings dedup to our DSO-derived suites.

All constants are UPSTREAM-STATED (no refitting; H0 = 70 km/s/Mpc = 0.07158985 Gyr^-1 throughout)
and every law is verified against every CSV row at build time. Three upstream paper-vs-data
discrepancies are resolved empirically (data wins, forms verified at machine precision):
- C3h: paper prints Q_D = -6*(addot/a + H0^2*Om0/(2 a^3)); the shipped target matches +3*(...)
  (a -1/2 factor vs the printed form). The CSV's H column varies and is UNUSED by the GT.
- C6a: paper prints sin(-3t^2 + v0*t); the data uses the phase 2*Phi = -3t^2 + 2*v0*t.
- C6c: paper text suggests fixed m2 = 0.5; the data is the EQUAL-MASS binary m2 = m1.

Datasets are frozen by necessity, not convenience: C2a/C6b have interdependent features that no
per-variable sampling spec can express, C1b/C5b/C5c are unseeded frozen noise realizations, and
the rest keep the exact upstream grids so results remain comparable with the cp3-bench paper.

Run from the repo root: ``PYTHONPATH=src python tools/build_cp3_cosmo.py``
"""
from __future__ import annotations

import gzip
import io
import os
from dataclasses import dataclass, field

import numpy as np

from symbolic_data import Problem, ProblemCatalog
from symbolic_data._evaluation import compile_expression, load_engine

HERE = os.path.dirname(os.path.abspath(__file__))
UPSTREAM = os.path.join(HERE, "..", "assets", "upstream", "cp3_bench")
OUT_NPZ = os.path.join(HERE, "..", "assets", "catalogs", "cp3-cosmo.npz")
OUT_REPORT = os.path.join(HERE, "..", "assets", "catalogs", "CP3_COSMO_REPORT.md")

SOURCE = ("cp3-bench / Things-to-bench (Thing & Koksbang, arXiv:2406.15531; "
          "github.com/CP3-Origins/Things-to-bench, MIT, commit 69a45dd)")

H0 = 0.07158985                      # 70 km/s/Mpc in Gyr^-1 (upstream's constant, verified)
E1 = repr(-10.0 / 3.0)               # GW-toy exponents, full float64 precision
E2 = repr(2.0 / 3.0)
E3 = repr(-4.0 / 3.0)

NFW = "1/((v2/v1)*((1+v2/v1)**(2)))"                                   # rho_NFW(r=v2, R0=v1)
BURKERT = "((v1)**(3))/((v2+v1)*(((v2)**(2))+((v1)**(2))))"            # rho_core(r=v2, R0=v1)


@dataclass
class Entry:
    eq_id: str
    infix: str                        # v-infix over the CSV's non-target columns, in CSV order
    noise: str | None = None          # None = noise-free (exact-GT candidate)
    superfluous: list[str] = field(default_factory=list)
    alternates: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    max_rel_err: float = 1e-6         # build gate vs the verified upstream precision


def build_entries() -> list[Entry]:
    lcdm1 = f"{H0!r}*sqrt(0.3*((1+v1)**(3))+0.7)"
    gw_phase = "-3*((v1)**(2))+20*v1"
    return [
        Entry("C1a", lcdm1,
              notes=["LCDM H(z), Om0=0.3, H0=70 km/s/Mpc in Gyr^-1; z grid [0.1,2]."]),
        Entry("C1b", lcdm1, noise="10% multiplicative Gaussian (frozen unseeded realization)",
              notes=["Noise variant of C1a (the paper's noise-robustness probe)."]),
        Entry("C1c", f"{H0!r}*sqrt(v1*((1+v2)**(3))+1-v1)",
              notes=["H(z; Omega_m) family; grids z [0.1,2] x omega_m [0.1,0.5]."]),
        Entry("C1d", "v1*sqrt(v2*((1+v3)**(3))+1-v2)",
              notes=["Constant-free: H0 is a feature (v1, [20,100] km/s/Mpc in Gyr^-1); 50^3 grid."]),
        Entry("C2a", f"{H0!r}*(1+v2)-v1",
              notes=["Redshift drift dz/dt0 = H0*(1+z) - H(z); the H feature (v1) is "
                     "INTERDEPENDENT with z via a hidden equation-of-state parameter -- "
                     "per-variable sampling specs cannot express this; frozen by necessity."]),
        Entry("C2b", f"{H0!r}*((1+v2)-((1+v2)**(1.5*(1+v1))))",
              notes=["Same drift with independent features (eos=v1, z=v2). Upstream discrepancy: "
                     "paper Table 1 says eos max 1/3, the data's max is 0.0 (data wins)."]),
        Entry("C3g", "600*((v1)**(2))*v3*(1-v3)*((1-v2)**(2))/((1-v3+v2*v3)**(2))",
              notes=["10^2 * Q_D two-region kinematical backreaction, constant-free."]),
        Entry("C3h", f"300*(v4/v3+{0.5 * H0 ** 2!r}*v2/((v3)**(3)))",
              superfluous=["H"],
              notes=["UPSTREAM DISCREPANCY (data wins, machine-precision verified): the paper "
                     "prints Q_D = -6*(addot/a + H0^2*Om0/(2 a^3)); the shipped target equals "
                     "+3*(a_av_tt/a_av + (H0^2/2)*Omega_m0/a_av^3). The H column (v1) VARIES "
                     "and is unused by the GT (superfluous feature).",
                     "Target equals C3g's target row-by-row (same quantity, other variables)."],
              max_rel_err=1e-6),
        Entry("C5a", "1/(v1*((1+v1)**(2)))", superfluous=["x"],
              notes=["NFW profile, R0=1; x (v2) is constant 1 = the superfluous-feature probe."]),
        Entry("C5b", "1/(v1*((1+v1)**(2)))", superfluous=["x"],
              noise="1% multiplicative Gaussian (frozen unseeded realization)",
              notes=["1% noise variant of C5a."]),
        Entry("C5c", "1/(v1*((1+v1)**(2)))", superfluous=["x"],
              noise="10% multiplicative Gaussian (frozen unseeded realization)",
              notes=["10% noise variant of C5a."]),
        Entry("C5d", NFW, superfluous=["x"],
              alternates=["((v1)**(3))/(v2*((v2+v1)**(2)))"],
              notes=["NFW with varying R0 (v1); r=v2; x (v3) constant 1, superfluous."]),
        Entry("C5e", BURKERT, superfluous=["x"],
              notes=["Burkert core profile; x (v3) constant -1, superfluous."]),
        Entry("C5f", f"(1/2)*(({NFW})+({BURKERT}))+(v3/2)*(({NFW})-({BURKERT}))",
              notes=["Profile switch: x (v3) in {-1,+1} SELECTS NFW vs Burkert -- x is "
                     "load-bearing here (discrete-feature case), unlike C5a-e."]),
        Entry("C6a", f"-0.1*((-4*((10-3*v1)**({E1}))+2*((10-3*v1)**({E2})))*cos({gw_phase})"
                     f"+4*((10-3*v1)**({E3}))*sin({gw_phase}))",
              notes=["GW-inspired h_plus toy (M=1, eta=0.25, R=5, v0=10); the suite's only "
                     "strongly oscillating target.",
                     "UPSTREAM DISCREPANCY (data wins): the paper prints the phase "
                     "sin(-3t^2 + v0*t); the data uses 2*Phi = -3t^2 + 2*v0*t."]),
        Entry("C6b", "-0.1*((-((v2)**(2))+((v1)**(2))*((v4)**(2))+1/v1)*cos(2*v3)"
                     "+2*v1*v2*v4*sin(2*v3))",
              notes=["h_plus in physical variables (r, dotr, Phi, dotPhi); features are "
                     "INTERDEPENDENT (orbit trajectory) -- frozen by necessity."]),
        Entry("C6c", f"-(v2/5)*((-16*((v2)**(2))*((10-3*v1)**({E1}))+2*((10-3*v1)**({E2})))"
                     f"*cos(({gw_phase})/(2*v2))+8*v2*((10-3*v1)**({E3}))"
                     f"*sin(({gw_phase})/(2*v2)))",
              notes=["Equal-mass binary h_plus vs (t, m1). UPSTREAM DISCREPANCY (data wins): "
                     "the paper text suggests fixed m2=0.5, the data is m2=m1 (M=2*m1)."],
              max_rel_err=1e-6),
    ]


def main() -> None:
    engine = load_engine("dev_7-3")
    problems: list[Problem] = []
    report: list[str] = [
        "# cp3-cosmo build report",
        "",
        f"Known-GT cosmology subset of cp3-bench ({SOURCE}). Constants upstream-stated (no "
        "refitting); every law verified against every CSV row. Deterministic rebuild "
        "(`tools/build_cp3_cosmo.py`).",
        "",
        "| eq_id | n | d | max rel err | FVU | gt_kind | noise |",
        "|---|---|---|---|---|---|---|",
    ]

    for entry in build_entries():
        path = os.path.join(UPSTREAM, f"{entry.eq_id}.csv.gz")
        raw = np.genfromtxt(io.TextIOWrapper(gzip.open(path)), delimiter=",", names=True)
        cols = [c for c in raw.dtype.names if c != "target"]
        X = np.stack([raw[c].astype(np.float64) for c in cols], axis=0)
        y = raw["target"].astype(np.float64)

        vars_info = {f"v{i+1}": {"name": cols[i]} for i in range(len(cols))}
        compiled = compile_expression(engine, entry.eq_id, entry.infix, vars_info, name="cp3-cosmo")
        with np.errstate(all="ignore"):
            y_ref = compiled["callable"](*[X[i] for i in range(len(cols))])
        y_ref = np.broadcast_to(np.asarray(y_ref, dtype=np.float64), y.shape).copy()
        assert np.all(np.isfinite(y_ref)), f"{entry.eq_id}: reference law non-finite on support"

        rel = float(np.max(np.abs(y_ref - y) / np.maximum(np.abs(y), 1e-30)))
        fvu = float(np.mean((y - y_ref) ** 2) / np.var(y))
        if entry.noise is None:
            assert rel <= entry.max_rel_err, \
                f"{entry.eq_id}: law deviates from noise-free data (max rel {rel:.2e})"
        else:
            # multiplicative noise: FVU = sd^2 * E[y^2]/Var(y), e.g. C1b's 10% on its skewed y
            # gives ~0.11; the gate only needs to catch a WRONG law (FVU would approach/exceed 1)
            assert fvu < 0.5, f"{entry.eq_id}: noise variant FVU implausibly large ({fvu:.3e})"

        for alt in entry.alternates:
            alt_c = compile_expression(engine, f"{entry.eq_id}:alt", alt, vars_info, name="cp3-cosmo")
            with np.errstate(all="ignore"):
                y_alt = np.broadcast_to(np.asarray(
                    alt_c["callable"](*[X[i] for i in range(len(cols))]), dtype=np.float64), y.shape)
            finite = np.isfinite(y_alt)
            assert finite.mean() >= 0.5, f"{entry.eq_id}: alternate mostly non-finite: {alt}"
            rel_alt = np.max(np.abs(y_alt[finite] - y_ref[finite])
                             / np.maximum(np.abs(y_ref[finite]), 1e-30))
            assert rel_alt < 1e-6, f"{entry.eq_id}: alternate deviates ({rel_alt:.2e}): {alt}"

        expression = list(compiled["expression"])
        constants = list(compiled["constants"])
        skeleton = tuple(compiled["prefix"])
        assert "<constant>" not in expression, f"{entry.eq_id}: masked token in expression"

        gt_kind = "exact" if (entry.noise is None and fvu <= 1e-12) else "reference"
        problem = Problem.from_data(
            X.T, y,
            expression=expression,
            skeleton=skeleton,
            constants=constants,
            variables=list(compiled["variable_order"]),
            gt_kind=gt_kind,
            y_reference_support=y_ref,
            eq_id=entry.eq_id,
            meta={
                "upstream_csv": f"cosmo_data/{entry.eq_id}.csv",
                "columns": cols,
                "reference_law": entry.infix,
                "prepared_infix": entry.infix,
                "alternate_renderings": entry.alternates,
                "law_source": SOURCE,
                "constant_policy": "upstream-stated constants (verified numerically; no refit)",
                "noise": entry.noise,
                "superfluous_columns": entry.superfluous,
                "verified_max_rel_err": rel,
                "reference_fvu_build": fvu,
                "license": "MIT (Things-to-bench)",
                "notes": entry.notes,
            },
        )
        assert problem.skeleton is not None, f"{entry.eq_id}: skeleton derivation failed"
        problems.append(problem)
        report.append(f"| {entry.eq_id} | {len(y)} | {len(cols)} | {rel:.2e} | {fvu:.3e} | "
                      f"{gt_kind} | {entry.noise or '-'} |")
        print(f"{entry.eq_id:5s} n={len(y):6d} d={len(cols)} rel={rel:9.2e} FVU={fvu:10.3e} {gt_kind}")

    catalog = ProblemCatalog.from_problems(
        problems, name="cp3-cosmo", version=1,
        meta={
            "description": "Known-ground-truth cosmology subset of cp3-bench (Things-to-bench): "
                           "LCDM H(z), redshift drift, two-region backreaction, NFW/Burkert halo "
                           "profiles, GW-inspired h_plus toys. Frozen upstream CSVs.",
            "license": "MIT (CP3-Origins/Things-to-bench)",
            "builder": "tools/build_cp3_cosmo.py",
            "engine": "dev_7-3",
            "split_policy": "all upstream points are support; no validation split",
            "excluded_upstream": "C3a-f, C4a-e (GT-free discovery sets; black-box tier later); "
                                 "F1-F8 (verbatim GP-toy re-renderings, dedup to the DSO suites)",
            "oracle_warning": "C3h/C5a-e carry superfluous feature columns BY DESIGN; variable "
                              "masking must be OFF for every method on this catalog",
        })
    out = catalog.save(OUT_NPZ)
    report += ["", f"{len(problems)} problems -> `{os.path.basename(str(out))}`.",
               "", "Upstream paper-vs-data discrepancies resolved empirically (data wins): "
               "C3h sign/prefactor, C6a phase, C6c equal-mass; C2b eos range. Excluded: "
               "C3a-f/C4a-e (no GT), F1-F8 (dedup to DSO suites, incl. F8's mutated 6.78 "
               "constant vs Korns' 6.87)."]
    with open(OUT_REPORT, "w", encoding="utf-8") as handle:
        handle.write("\n".join(report) + "\n")
    print(f"\nSaved {out} + report.")


if __name__ == "__main__":
    main()
