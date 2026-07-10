"""Build the ``physo-streams`` catalog: PhySO's Milky-Way stellar-streams Class-SR demo.

29 frozen problems (one per stream, 100 points each) from streams.csv (PhySO, MIT): mock orbits
integrated in an NFW potential, progenitors approximating the 29 real thin streams of Ibata et
al. 2023. The Class-SR structure (one functional form, per-stream constant) is flattened to one
problem per stream: E_kin = E_t_s + A * log(B*r + 1) / r in the demo's normalized units
(r = |x|/20 kpc, v = |v|/200 km/s, E_kin = v^2/2), with the SHARED constants A = 3.77705922384934,
B = 1.0000075063141 and the 29 per-stream E_t values taken VERBATIM from the demo's target
expressions (MW_streams_results_analysis.py). No fitting: the builder matches each stream to its
upstream E_t bijectively (closed-form LS estimate -> nearest listed value, with an explicit
identifiability margin against the list's 1.77e-4 twin pair)
and verifies the law on every point.

gt_kind: the data comes from orbit INTEGRATION, so energy conservation holds to integrator
precision (FVU ~ 1e-9), not to float precision -> "reference" under the mechanical 1e-12 rule.

Run from the repo root: ``PYTHONPATH=src python tools/build_physo_streams.py``
"""
from __future__ import annotations

import gzip
import io
import os

import numpy as np

from symbolic_data import Problem, ProblemCatalog
from symbolic_data._evaluation import compile_expression, load_engine

HERE = os.path.dirname(os.path.abspath(__file__))
UPSTREAM = os.path.join(HERE, "..", "assets", "upstream", "physo")
OUT_NPZ = os.path.join(HERE, "..", "assets", "catalogs", "physo-streams.npz")
OUT_REPORT = os.path.join(HERE, "..", "assets", "catalogs", "PHYSO_STREAMS_REPORT.md")

SOURCE = ("PhySO Class-SR MW streams demo (Tenachi et al. 2024, arXiv:2312.01816; "
          "github.com/WassimTenachi/PhySO, MIT; mock streams after Ibata et al. 2023, "
          "arXiv:2311.17202)")

A = 3.77705922384934                 # shared NFW-potential constants (upstream, verbatim)
B = 1.0000075063141
R_NORM_KPC = 20.0                    # demo units: r/20 kpc, v/200 km/s
V_NORM_KMS = 200.0

# the 29 per-stream total-energy constants, VERBATIM from MW_streams_results_analysis.py
E_T = [-2.25433197869809, -2.41436719312905, -2.2447343815812, -1.5859020918779,
       -2.5645735964909, -1.62503655452927, -2.15162507988352, -1.35017082006533,
       -1.34999371983704, -0.886739103968145, -1.4852048039817, -1.93110067461651,
       -2.27868502863944, -1.90140528126145, -1.79236813095344, -2.0266623875962,
       -1.76631299530485, -1.62558963183134, -2.31576548893306, -1.41578671710615,
       -1.17092711945236, -1.48716147657053, -1.51377591474388, -1.9125453150739,
       -1.07822521728563, -1.29561794758918, -1.56992178190473, -2.05315490058432,
       -1.68884448806975]


def main() -> None:
    engine = load_engine("dev_7-3")
    raw = np.genfromtxt(io.TextIOWrapper(gzip.open(os.path.join(UPSTREAM, "streams.csv.gz"))),
                        delimiter=",", names=True)
    r = np.sqrt(raw["x"] ** 2 + raw["y"] ** 2 + raw["z"] ** 2) / R_NORM_KPC
    e_kin = 0.5 * ((raw["vx"] ** 2 + raw["vy"] ** 2 + raw["vz"] ** 2) / V_NORM_KMS ** 2)
    sid = raw["sID"].astype(int)
    stream_ids = sorted(set(sid.tolist()))
    assert len(stream_ids) == 29 and len(r) == 2900

    problems, rows, used = [], [], set()
    for s in stream_ids:
        mask = sid == s
        rs, ys = r[mask].astype(np.float64), e_kin[mask].astype(np.float64)
        assert len(rs) == 100, (s, len(rs))

        # bijective upstream-constant matching: closed-form LS E_t -> nearest listed value
        basis = A * np.log(B * rs + 1.0) / rs
        e_fit = float(np.mean(ys - basis))
        dists = sorted((abs(e_fit - e), i) for i, e in enumerate(E_T))
        (dist, idx), (runner, _) = dists[0], dists[1]
        # upstream E_t values are theoretical (orbit initial conditions), not per-point LS refits:
        # measured LS-to-match distances are <= 4.7e-5. The list contains a TWIN pair only
        # 1.77e-4 apart, so nearest-match alone cannot exclude a mutual swap -- require an
        # explicit identifiability margin (runner-up clearly farther) on top of closeness.
        assert dist < 6e-5, (s, e_fit, E_T[idx], dist)
        assert runner - dist > 6e-5, (s, "ambiguous E_t match", dist, runner)
        assert idx not in used, f"stream {s}: E_t {E_T[idx]} already consumed (non-bijective)"
        used.add(idx)
        e_t = E_T[idx]

        infix = f"{e_t!r}+{A!r}*log({B!r}*v1+1)/v1"
        compiled = compile_expression(engine, f"stream-{s:02d}", infix, {"v1": {"name": "r"}},
                                      name="physo-streams")
        y_ref = np.broadcast_to(np.asarray(compiled["callable"](rs), dtype=np.float64), ys.shape)
        fvu = float(np.mean((ys - y_ref) ** 2) / np.var(ys))
        assert fvu < 1e-6, (s, fvu)

        gt_kind = "exact" if fvu <= 1e-12 else "reference"
        problems.append(Problem.from_data(
            rs.reshape(-1, 1), ys,
            expression=list(compiled["expression"]),
            skeleton=tuple(compiled["prefix"]),
            constants=list(compiled["constants"]),
            variables=list(compiled["variable_order"]),
            gt_kind=gt_kind,
            y_reference_support=y_ref,
            eq_id=f"stream-{s:02d}",
            meta={
                "columns": ["r"],
                "target": "E_kin (normalized: v^2/2 with v in 200 km/s units; r in 20 kpc units)",
                "reference_law": "E_t + A*log(B*r+1)/r (shared A,B; per-stream E_t)",
                "prepared_infix": infix,
                "class_structure": {"shared": {"A": A, "B": B}, "per_stream": {"E_t": e_t},
                                    "upstream_sID": s},
                "law_source": SOURCE,
                "constant_policy": "upstream-stated constants, verbatim (bijective E_t matching "
                                   "verified; no fitting)",
                "reference_fvu_build": fvu,
                "license": "MIT (PhySO)",
                "notes": ["Mock orbit integration: energy conservation holds to integrator "
                          "precision (FVU ~1e-9), hence gt_kind=reference.",
                          "Raw phase-space columns (kpc, km/s) live in the vendored streams.csv; "
                          "this problem ships the demo's normalized 1-D task."],
            }))
        rows.append(f"| stream-{s:02d} | 100 | {e_t:.6f} | {fvu:.3e} | {gt_kind} |")
        print(f"stream-{s:02d} E_t={e_t:+.6f} FVU={fvu:.3e} {gt_kind}")

    assert len(used) == 29
    catalog = ProblemCatalog.from_problems(
        problems, name="physo-streams", version=1,
        meta={
            "description": "PhySO Class-SR Milky-Way stellar-streams demo, flattened: 29 frozen "
                           "per-stream problems sharing one NFW-potential law with a per-stream "
                           "total-energy constant.",
            "license": "MIT (PhySO)",
            "builder": "tools/build_physo_streams.py",
            "engine": "dev_7-3",
            "split_policy": "all 100 points per stream are support; no validation split",
            "class_protocol": "upstream evaluates CLASS recovery (one form, 29 realizations, "
                              "jointly); this flattening scores per-stream recovery -- joint-fit "
                              "class metrics need harness support (recorded as future work)",
        })
    out = catalog.save(OUT_NPZ)
    with open(OUT_REPORT, "w", encoding="utf-8") as handle:
        handle.write("\n".join([
            "# physo-streams build report", "",
            f"{SOURCE}. Shared A={A!r}, B={B!r}; per-stream E_t verbatim upstream.", "",
            "| eq_id | n | E_t | FVU | gt_kind |", "|---|---|---|---|---|",
        ] + rows + ["", f"{len(problems)} problems -> `{os.path.basename(str(out))}`."]) + "\n")
    print(f"\nSaved {out} + report.")


if __name__ == "__main__":
    main()
