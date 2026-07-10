"""Build the ``ai-descartes`` catalog: FSRD_noise (81) + the 6 measured real-world datasets.

Source: IBM/AI-Descartes (MIT), the benchmark data of Cornelio et al., Nat. Commun. 14, 1777
(2023). Two subsets in ONE source catalog (meta.subset tags them):

- ``fsrd_noise`` (81 entries): the paper's fixed low-data noisy Feynman condition. Support =
  the canonical frozen ``input.dat`` (10 points, additive Gaussian y-noise, sigma = 1% rms;
  UNSEEDED generator, so the frozen files ARE the benchmark). Validation = rows 10:20 of the
  frozen clean ``input_original.dat`` (10 UNSEEN clean points; rows 0:10 are the support x's,
  whose clean y is exactly ``y_reference_support``). Expressions are REUSED from the published
  ``feynman`` catalog (P2-verified aifeynman-original translations): upstream raw strings match
  byte-identically (2 id-spelling aliases normalized: I.15.10 -> I.15.1, I.48.20 -> I.48.2),
  and the per-entry numeric check (reference law vs clean parent y at <1e-6 rel) verifies both
  the expression and the name-based variable/column mapping. This does NOT count toward the
  Feynman generation-variant cap: it is a fixed_data source, not a generation recipe.

- ``real_world`` (6 entries): measured tables. kepler_{solar_system,exoplanets,binary_stars}
  (p^2 = 4 pi^2 d^3/(G(m1+m2)); raw columns kept, mass-unit conversions live INSIDE the law,
  leading constant refit); langmuir_{sun_et_al,table_IX} (q = a b p/(1+a p), constants refit --
  upstream states none); relativistic_time_dilation (constant-complete: c = 299792458, 1e15
  unit factor; textbook rendering stored, cancellation-stable form as a holdout alternate).

Run from the repo root: ``PYTHONPATH=src python tools/build_ai_descartes.py``
"""
from __future__ import annotations

import os
import warnings

import numpy as np
import yaml
from scipy.optimize import curve_fit

from symbolic_data import Problem, ProblemCatalog
from symbolic_data._evaluation import compile_expression, load_engine

HERE = os.path.dirname(os.path.abspath(__file__))
UPSTREAM = os.path.join(HERE, "..", "assets", "upstream", "ai_descartes")
FEYNMAN_YAML = os.path.join(HERE, "..", "assets", "catalogs", "feynman.yaml")
OUT_NPZ = os.path.join(HERE, "..", "assets", "catalogs", "ai-descartes.npz")
OUT_REPORT = os.path.join(HERE, "..", "assets", "catalogs", "AI_DESCARTES_REPORT.md")

SOURCE = ("AI-Descartes (Cornelio et al., Nat. Commun. 14, 1777 (2023); "
          "github.com/IBM/AI-Descartes, MIT)")
ID_ALIASES = {"I.15.10": "I.15.1", "I.48.20": "I.48.2"}   # upstream trailing-zero spellings

C_LIGHT = 299792458.0
M_EARTH_SOLAR = 5.972e24 / 1.9885e30      # README's mass conversions
M_JUP_SOLAR = 1.898e27 / 1.9885e30


def _fvu(y, y_ref):
    return float(np.mean((y - y_ref) ** 2) / np.var(y))


def _read_fsrd(path):
    """Read an FSRD_noise .dat: (eq_id, expr, input_names, target_name, rows)."""
    with open(path, encoding="utf-8") as handle:
        header = handle.readline().strip()
        names_line = handle.readline().strip()
        rows = np.loadtxt(handle)
    assert header.startswith("# Feynman "), header
    eq_id, expr = header[len("# Feynman "):].split(None, 1)
    names = names_line.lstrip("*").split()
    return eq_id, expr.strip(), names[:-1], names[-1], np.atleast_2d(rows)


def _build_fsrd_problems(engine, feynman):
    problems, report_rows = [], []
    base = os.path.join(UPSTREAM, "FSRD_noise")
    eq_dirs = sorted(d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d)))
    assert len(eq_dirs) == 81, f"expected 81 FSRD_noise dirs, found {len(eq_dirs)}"
    for eq_dir in eq_dirs:
        up_id, expr, in_names, target_name, noisy = _read_fsrd(os.path.join(base, eq_dir, "input.dat"))
        up_id2, expr2, in_names2, _, clean = _read_fsrd(os.path.join(base, eq_dir, "input_original.dat"))
        assert up_id == up_id2 == eq_dir and expr == expr2 and in_names == in_names2, eq_dir
        assert noisy.shape[0] == 10 and clean.shape[0] == 20, eq_dir

        eq_id = ID_ALIASES.get(up_id, up_id)
        entry = feynman[eq_id]
        assert entry["raw"] == expr, f"{eq_id}: upstream expr != feynman catalog raw\n{expr}\n{entry['raw']}"

        # name-based mapping: column of v_k = position of feynman's name(v_k) in the file header
        name_to_col = {n: i for i, n in enumerate(in_names)}
        order = []
        for k in range(1, entry["n_variables"] + 1):
            name = entry["vars"][f"v{k}"]["name"]
            assert name in name_to_col, f"{eq_id}: variable {name!r} not in file header {in_names}"
            order.append(name_to_col[name])
        assert sorted(order) == list(range(len(in_names))), f"{eq_id}: non-bijective mapping"

        X = noisy[:, order].astype(np.float64)          # v-order columns
        y = noisy[:, -1].astype(np.float64)
        X_val = clean[10:, order].astype(np.float64)
        y_val = clean[10:, -1].astype(np.float64)
        y_clean_support = clean[:10, -1].astype(np.float64)
        assert np.allclose(noisy[:, :-1], clean[:10, :-1]), f"{eq_id}: support x != clean parent rows"

        vars_info = {f"v{k}": {"name": entry["vars"][f"v{k}"]["name"]}
                     for k in range(1, entry["n_variables"] + 1)}
        compiled = compile_expression(engine, eq_id, entry["prepared"], vars_info, name="ai-descartes")
        with np.errstate(all="ignore"):
            y_ref = np.asarray(compiled["callable"](*[X[:, i] for i in range(X.shape[1])]), dtype=np.float64)
            y_ref_val = np.asarray(compiled["callable"](*[X_val[:, i] for i in range(X_val.shape[1])]), dtype=np.float64)
        y_ref = np.broadcast_to(y_ref, y.shape)
        y_ref_val = np.broadcast_to(y_ref_val, y_val.shape)

        # the law on the support x's must BE the clean parent y; the law on the val x's must BE y_val
        scale = np.maximum(np.abs(y_clean_support), 1e-12)
        assert np.max(np.abs(y_ref - y_clean_support) / scale) < 1e-6, eq_id
        scale_val = np.maximum(np.abs(y_val), 1e-12)
        assert np.max(np.abs(y_ref_val - y_val) / scale_val) < 1e-6, eq_id

        noise_sd = float(np.std(y - y_clean_support))
        fvu = _fvu(y, y_ref)
        problem = Problem.from_data(
            X, y, x_validation=X_val, y_validation=y_val,
            expression=list(compiled["expression"]),
            skeleton=tuple(compiled["prefix"]),
            constants=list(compiled["constants"]),
            variables=list(compiled["variable_order"]),
            gt_kind="reference",
            y_reference_support=y_ref, y_reference_validation=y_ref_val,
            eq_id=f"fsrd-noise:{eq_id}",
            meta={
                "subset": "fsrd_noise",
                "upstream_id": up_id,
                "base_catalog": "feynman@1", "base_eq_id": eq_id,
                "columns": in_names, "target": target_name,
                "reference_law": entry["raw"],
                "prepared_infix": entry["prepared"],
                "noise": "additive Gaussian on y, sigma = 0.01*rms(y_clean), frozen unseeded "
                         f"realization (measured sd {noise_sd:.3g})",
                "split_policy": "support = canonical 10-point noisy input.dat; validation = the "
                                "10 UNSEEN clean rows (10:20) of input_original.dat",
                "law_source": SOURCE,
                "constant_policy": "upstream FSRD expression (verified against the clean parent)",
                "reference_fvu_build": fvu,
                "license": "MIT (IBM/AI-Descartes)",
            },
        )
        problems.append(problem)
        report_rows.append(f"| fsrd-noise:{eq_id} | 10+10 | {X.shape[1]} | {fvu:.3e} | reference | 1% rms noise |")
    return problems, report_rows


def _fit_langmuir(p, q):
    """q = v*p/(1+u*p) (v = a*b, u = a): deterministic grid over u + exact linear v, then polish."""
    best = None
    for u in np.geomspace(1e-4, 1e3, 800):
        f = p / (1.0 + u * p)
        v = float(np.dot(f, q) / np.dot(f, f))
        fvu = _fvu(q, v * f)
        if best is None or fvu < best[0]:
            best = (fvu, u, v)
    _, u, v = best
    def model(_x, a, b):
        return a * b * p / (1.0 + a * p)
    try:
        theta, _ = curve_fit(model, np.zeros_like(p), q, p0=[u, v / u], maxfev=20000,
                             bounds=([1e-12, 1e-12], [np.inf, np.inf]))
        if _fvu(q, model(None, *theta)) <= best[0]:
            return float(theta[0]), float(theta[1])
    except Exception:
        pass
    return float(u), float(v / u)


def _build_real_problems(engine):
    problems, report_rows = [], []

    def add(eq_id, X, y, infix, columns, notes, alternates=(), fitted=None, fvu_gate=0.5,
            alt_rel_tol=1e-6):
        vars_info = {f"v{i+1}": {"name": columns[i]} for i in range(X.shape[1])}
        compiled = compile_expression(engine, eq_id, infix, vars_info, name="ai-descartes")
        with np.errstate(all="ignore"):
            y_ref = np.broadcast_to(np.asarray(
                compiled["callable"](*[X[:, i] for i in range(X.shape[1])]), dtype=np.float64), y.shape)
        assert np.all(np.isfinite(y_ref)), eq_id
        fvu = _fvu(y, y_ref)
        assert fvu < fvu_gate, f"{eq_id}: reference law FVU {fvu:.3f} exceeds gate {fvu_gate}"
        for alt in alternates:
            alt_c = compile_expression(engine, f"{eq_id}:alt", alt, vars_info, name="ai-descartes")
            with np.errstate(all="ignore"):
                y_alt = np.broadcast_to(np.asarray(
                    alt_c["callable"](*[X[:, i] for i in range(X.shape[1])]), dtype=np.float64), y.shape)
            finite = np.isfinite(y_alt)
            assert finite.mean() >= 0.5, (eq_id, alt)
            # scale-aware equivalence: an algebraically-identical alternate may be numerically
            # DEGENERATE on some rows (time dilation's textbook form returns exactly 0 at
            # v=0.55 m/s, where 1 - v^2/c^2 rounds to 1 in float64) -- judge the deviation
            # against the problem's scale, not row-by-row relative.
            scale = float(np.sqrt(np.mean(y_ref[finite] ** 2)))
            assert np.max(np.abs(y_alt[finite] - y_ref[finite])) < alt_rel_tol * max(scale, 1e-30), \
                (eq_id, alt)
        problem = Problem.from_data(
            X, y,
            expression=list(compiled["expression"]),
            skeleton=tuple(compiled["prefix"]),
            constants=list(compiled["constants"]),
            variables=list(compiled["variable_order"]),
            gt_kind="reference",
            y_reference_support=y_ref,
            eq_id=eq_id,
            meta={
                "subset": "real_world",
                "columns": columns,
                "reference_law": infix,
                "prepared_infix": infix,
                "alternate_renderings": list(alternates),
                "law_source": SOURCE,
                "constant_policy": ("least-squares refit (deterministic)" if fitted
                                    else "constant-complete (known physical constants)"),
                "fitted_constants": fitted or [],
                "reference_fvu_build": fvu,
                "license": "MIT (IBM/AI-Descartes)",
                "notes": notes,
            },
        )
        problems.append(problem)
        report_rows.append(f"| {eq_id} | {len(y)} | {X.shape[1]} | {fvu:.3e} | reference | measured |")

    # -- kepler (target p is the FIRST column; '# <planet>' row comments) ---------------------
    for fname, eq_id, k_m2, p_unit in [
        ("KEPLER_solar_system.dat", "kepler-solar-system", M_EARTH_SOLAR, "days/1000"),
        ("kepler_exoplanets.dat", "kepler-exoplanets", M_JUP_SOLAR, "days/1000"),
        ("kepler_binary_stars.dat", "kepler-binary-stars", 1.0, "years"),
    ]:
        raw = np.genfromtxt(os.path.join(UPSTREAM, "kepler", fname), names=True, comments="#")
        p, m1, m2, d = (raw[c].astype(np.float64) for c in ("p", "m1", "m2", "d"))
        X = np.stack([m1, m2, d], axis=1)
        feat = np.sqrt(d ** 3 / (m1 + k_m2 * m2))
        c1 = float(np.dot(feat, p) / np.dot(feat, feat))
        k_str = "" if k_m2 == 1.0 else f"{k_m2!r}*"
        infix = f"{c1!r}*sqrt(((v3)**(3))/(v1+{k_str}v2))"
        add(eq_id, X, p, infix, ["m1", "m2", "d"],
            notes=[f"Kepler's third law; p in {p_unit}, d in AU, m1 in solar masses"
                   + ("" if k_m2 == 1.0 else f", m2 converted inside the law (factor {k_m2:.4g})")
                   + "; leading constant refit on the data.",
                   "Real measured orbital parameters (row comments name the bodies upstream)."],
            fitted=[c1], fvu_gate=0.1)

    # -- langmuir (q = a*b*p/(1+a*p); constants refit -- upstream states none) -----------------
    for fname, eq_id, extra in [
        ("langmuir_sun_et_al.dat", "langmuir-sun-et-al",
         ["Data visibly exceeds a single-site Langmuir plateau at high pressure: the reference "
          "law is APPROXIMATE here (build FVU ~0.39) -- calibrate reference_fvu expectations "
          "accordingly."]),
        ("langmuir_table_IX.dat", "langmuir-table-ix",
         ["Langmuir 1918 Table IX; file ships with a UTF-8 BOM + CRLF endings (handled at "
          "import)."]),
    ]:
        text = open(os.path.join(UPSTREAM, "langmuir", fname), encoding="utf-8-sig", newline=None).read()
        rows = np.loadtxt([ln for ln in text.splitlines() if ln.strip() and not ln.lstrip().startswith("#")])
        p, q = rows[:, 0], rows[:, 1]
        a, b = _fit_langmuir(p, q)
        infix = f"{a!r}*{b!r}*v1/(1+{a!r}*v1)"
        add(eq_id, p.reshape(-1, 1), q, infix, ["p"],
            notes=["Adsorption isotherm; constants a (=k_ads/k_des) and b (=saturation loading) "
                   "refit -- upstream provides the FORM only."] + extra,
            fitted=[a, b], fvu_gate=0.6)

    # -- relativistic time dilation (constant-complete) ----------------------------------------
    raw = np.genfromtxt(os.path.join(UPSTREAM, "relativistic_time_dilation",
                                     "relativistic_time_dilation.dat"), names=True, comments="#")
    t, v = raw["t"].astype(np.float64), raw["v"].astype(np.float64)
    textbook = f"1e15*(sqrt(1-((v1/{C_LIGHT!r})**(2)))-1)"
    stable = f"-1e15*((v1/{C_LIGHT!r})**(2))/(sqrt(1-((v1/{C_LIGHT!r})**(2)))+1)"
    # the STABLE rendering is the stored expression: at v^2/c^2 ~ 1e-14 the textbook form loses
    # ~eps/x ~ 1e-2 RELATIVE accuracy to cancellation even in float64 (and rounds to exactly 0 in
    # float32), so the accurate reference predictions can only come from the stable form; the
    # textbook rendering registers as a holdout alternate with a tolerance matching its own
    # cancellation error.
    add("relativistic-time-dilation", v.reshape(-1, 1), t, stable, ["v"],
        notes=["Optical-clock fractional frequency shift in 1e-15 units; c = 299792458 m/s known "
               "-- constant-complete, NO refit.",
               "Stored expression is the cancellation-STABLE rendering of "
               "1e15*(sqrt(1-v^2/c^2)-1); the textbook form (alternate) loses ~1% relative "
               "accuracy in float64 at the smallest v and underflows to 0 in float32."],
        alternates=[textbook], fvu_gate=0.5, alt_rel_tol=0.05)

    return problems, report_rows


def main() -> None:
    engine = load_engine("dev_7-3")
    feynman = yaml.safe_load(open(FEYNMAN_YAML, encoding="utf-8"))["expressions"]

    fsrd_problems, fsrd_rows = _build_fsrd_problems(engine, feynman)
    real_problems, real_rows = _build_real_problems(engine)
    problems = fsrd_problems + real_problems
    print(f"fsrd_noise: {len(fsrd_problems)}  real_world: {len(real_problems)}")

    catalog = ProblemCatalog.from_problems(
        problems, name="ai-descartes", version=1,
        meta={
            "description": "AI-Descartes benchmark data: FSRD_noise (81 frozen 10-point noisy "
                           "Feynman renderings + clean validation rows) and 6 measured real-world "
                           "datasets (Kepler orbits, Langmuir isotherms, relativistic time "
                           "dilation).",
            "license": "MIT (IBM/AI-Descartes)",
            "builder": "tools/build_ai_descartes.py",
            "engine": "dev_7-3",
            "subsets": {"fsrd_noise": 81, "real_world": 6},
            "split_policy": "fsrd_noise: 10 noisy support + 10 unseen clean validation; "
                            "real_world: all measured points support, no validation",
            "feynman_overlap": "fsrd_noise expressions == feynman@1 (aifeynman-original) with 2 "
                               "id aliases (I.15.10->I.15.1, I.48.20->I.48.2); fixed_data source, "
                               "NOT a Feynman generation variant (cap untouched); "
                               "meta.base_eq_id links every entry",
        })
    out = catalog.save(OUT_NPZ)
    report = [
        "# ai-descartes build report", "",
        f"{SOURCE}. FSRD_noise expressions reused from the P2-verified feynman catalog "
        "(name-based variable mapping, per-entry numeric verification vs the clean parent).", "",
        "| eq_id | n | d | FVU(ref) | gt_kind | data |", "|---|---|---|---|---|---|",
    ] + fsrd_rows + real_rows + ["", f"{len(problems)} problems -> `{os.path.basename(str(out))}`."]
    with open(OUT_REPORT, "w", encoding="utf-8") as handle:
        handle.write("\n".join(report) + "\n")
    print(f"Saved {out} + report.")


if __name__ == "__main__":
    main()
