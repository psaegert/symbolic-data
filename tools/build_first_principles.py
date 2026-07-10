"""Build the ``first-principles`` catalog: the SRBench-2.0 phenomenological track (PMLB rendering).

13 measured/frozen datasets (PMLB ``first_principles_*``, MIT) + their accepted reference laws:
10 from EmpiricalBench (Cranmer 2023, arXiv:2305.01582), absorption's refit simplified form from the
MvSR notebook (Russeil et al. 2024, GECCO), and the Bazin et al. 2009 light-curve function for the
two supernova bands (the domain reference named in the MvSR notebook). This is the first FROZEN
(measured-data) catalog: Problems are built via ``Problem.from_data`` with ``gt_kind`` reference/
exact, all points as support (no held-out validation split -- n ranges 6..243), and the reference
law's predictions in ``y_reference_support`` (the ``reference_fvu`` baseline srbf re-derives).

Constant-fitting policy: constants are least-squares REFITS on the dataset itself (documented per
entry) -- exact linear solves where the law is linear in parameters, deterministic separable
grid + ``curve_fit`` polish otherwise. No RNG anywhere; rebuilds are bit-stable.

gt_kind rule (disclosure-first): metadata-declared synthetic provenance AND fitted FVU <= 1e-12
-> "exact" (the refit recovered the generating law); everything else -> "reference".
``numeric_recovery_relative`` is endpoint-identical either way.

Run from the repo root: ``PYTHONPATH=src python tools/build_first_principles.py``
"""
from __future__ import annotations

import gzip
import io
import os
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from scipy.optimize import curve_fit

from symbolic_data import Problem, ProblemCatalog
from symbolic_data._evaluation import compile_expression, load_engine


def _is_number(token: str) -> bool:
    try:
        float(token)
        return True
    except (TypeError, ValueError):
        return False

HERE = os.path.dirname(os.path.abspath(__file__))
UPSTREAM = os.path.join(HERE, "..", "assets", "upstream", "pmlb_first_principles")
OUT_NPZ = os.path.join(HERE, "..", "assets", "catalogs", "first-principles.npz")
OUT_REPORT = os.path.join(HERE, "..", "assets", "catalogs", "FIRST_PRINCIPLES_REPORT.md")

EMPIRICALBENCH = ("EmpiricalBench (Cranmer 2023, arXiv:2305.01582) via PMLB (MIT), "
                  "https://github.com/EpistasisLab/pmlb")
MVSR = ("MvSR (Russeil et al. 2024, GECCO, doi:10.1145/3638529.3654087) via PMLB (MIT), "
        "https://github.com/EpistasisLab/pmlb")


def _fvu(y: np.ndarray, y_ref: np.ndarray) -> float:
    return float(np.mean((y - y_ref) ** 2) / np.var(y))


@dataclass
class Law:
    eq_id: str
    dataset: str                      # PMLB dataset name suffix
    infix_template: str               # v-infix with {c1}.. placeholders
    model: Callable                   # model(X_cols, *theta) -> y (float64)
    fit: Callable                     # fit(X_cols, y) -> theta list
    source: str
    synthetic: bool                   # metadata-declared synthetic provenance
    notes: list[str] = field(default_factory=list)
    columns: list[str] = field(default_factory=list)
    target: str = ""
    # Algebraically-equivalent canonical renderings of the SAME law (v-infix templates over the
    # same {c}-placeholders, possibly derived constants): registered as additional holdout
    # structures so the textbook form of a stabilized/rewritten stored law cannot evade the
    # structure layer. Each is numerically verified against the fitted law on its finite domain.
    alternates: Callable | None = None    # alternates(theta) -> list[str] (concrete v-infix)


# ---------------------------------------------------------------- fitting helpers (deterministic)
def _lstsq(features: np.ndarray, y: np.ndarray) -> np.ndarray:
    theta, *_ = np.linalg.lstsq(features, y, rcond=None)
    return theta


def _polish(model: Callable, X, y, theta0, bounds=(-np.inf, np.inf)) -> np.ndarray:
    """curve_fit polish; falls back to the init if the polish does not improve the FVU."""
    def f(_x, *theta):
        return model(X, *theta)
    try:
        theta, _ = curve_fit(f, np.zeros(len(y)), y, p0=theta0, maxfev=20000, bounds=bounds)
    except Exception:
        return np.asarray(theta0, dtype=float)
    theta0 = np.asarray(theta0, dtype=float)
    return theta if _fvu(y, model(X, *theta)) <= _fvu(y, model(X, *theta0)) else theta0


# ---------------------------------------------------------------- the 13 laws
def build_laws() -> list[Law]:
    laws: list[Law] = []

    # -- hubble: v = H0 * D (through-origin linear) ------------------------------------------
    def m_hubble(X, c1):
        return c1 * X[0]
    laws.append(Law(
        "hubble", "hubble", "{c1}*v1", m_hubble,
        lambda X, y: [float(np.dot(X[0], y) / np.dot(X[0], X[0]))],
        EMPIRICALBENCH, synthetic=False,
        columns=["D"], target="velocity, km/s",
        notes=["Hubble's law; real 1929-style galaxy data (large intrinsic scatter expected)."]))

    # -- kepler: P = c1 * a^(3/2) (P^2 = k a^3) ----------------------------------------------
    def m_kepler(X, c1):
        return c1 * X[0] ** 1.5
    laws.append(Law(
        "kepler", "kepler", "{c1}*((v1)**(1.5))", m_kepler,
        lambda X, y: [float(np.dot(X[0] ** 1.5, y) / np.dot(X[0] ** 1.5, X[0] ** 1.5))],
        EMPIRICALBENCH, synthetic=False,
        columns=["a"], target="orbital period, days",
        notes=["Kepler's third law on the 6 classical planets (Kepler 1618 data)."]))

    # -- newton: y = ln(F) = ln(c1 m1 m2 / r^2) (cols r, m1, m2; c1 = G) -----------------------
    def m_newton(X, c1):
        return np.log(c1 * X[1] * X[2] / X[0] ** 2)
    def f_newton(X, y):
        return [float(np.exp(np.mean(y - np.log(X[1] * X[2] / X[0] ** 2))))]
    laws.append(Law(
        "newton", "newton", "log({c1}*v2*v3/((v1)**(2)))", m_newton, f_newton,
        EMPIRICALBENCH, synthetic=True,
        columns=["r", "m1", "m2"], target="ln(force)",
        notes=["Target is the NATURAL LOG of the force magnitude (SI); fitted c1 should recover "
               "G = 6.674e-11.", "Masses reach ~1e29, so the m1*m2 intermediate exceeds the "
               "float32 range: evaluate candidate expressions in float64.",
               "Synthetically generated per PMLB metadata."]))

    # -- ideal_gas: y = ln(P) = ln(c1 n T / V) (cols n, T, V; c1 = R) --------------------------
    def m_ideal(X, c1):
        return np.log(c1 * X[0] * X[1] / X[2])
    def f_ideal(X, y):
        return [float(np.exp(np.mean(y - np.log(X[0] * X[1] / X[2]))))]
    laws.append(Law(
        "ideal_gas", "ideal_gas", "log({c1}*v1*v2/v3)", m_ideal, f_ideal,
        EMPIRICALBENCH, synthetic=True,
        columns=["n", "T", "V"], target="ln(pressure)",
        notes=["Target is the NATURAL LOG of the pressure; fitted c1 should recover the gas "
               "constant R = 8.314.", "Synthetically generated per PMLB metadata."]))

    # -- planck: y = ln(B) = ln(c1 nu^3 / (exp(c2 nu / T) - 1)) --------------------------------
    def _logexpm1(x):
        # ln(exp(x) - 1), stable for large x: x + log1p(-exp(-x))
        x = np.asarray(x, dtype=np.float64)
        return np.where(x > 30.0, x + np.log1p(-np.exp(-np.minimum(x, 700.0))),
                        np.log(np.expm1(np.minimum(x, 30.0))))
    def m_planck(X, c1, c2):
        return np.log(c1) + 3.0 * np.log(X[0]) - _logexpm1(c2 * X[0] / X[1])
    def f_planck(X, y):
        best = None
        for c2 in np.geomspace(1e-14, 1e-7, 600):
            resid = y - 3.0 * np.log(X[0]) + _logexpm1(c2 * X[0] / X[1])
            if not np.all(np.isfinite(resid)):
                continue
            c1 = float(np.exp(np.mean(resid)))
            fvu = _fvu(y, m_planck(X, c1, c2))
            if best is None or fvu < best[0]:
                best = (fvu, [c1, float(c2)])
        return list(_polish(m_planck, X, y, best[1],
                            bounds=([1e-300, 1e-16], [np.inf, 1e-5])))
    laws.append(Law(
        "planck", "planck",
        "log({c1})+3*log(v1)-{c2}*v1/v2-log(1-exp(-{c2}*v1/v2))", m_planck, f_planck,
        EMPIRICALBENCH, synthetic=True,
        columns=["nu", "T"], target="ln(spectral radiance)",
        notes=["Target is the NATURAL LOG of the spectral radiance (deep Wien regime: y reaches "
               "-1379); the reference expression is the log-STABILIZED rendering of "
               "ln(c1 nu^3 / (exp(c2 nu/T) - 1)) -- the naive form overflows exp() at the "
               "support's max c2*nu/T = 1372 even in float64.",
               "Fitted constants recover the physics: c1 = 1.455e-50 vs 2h/c^2 = 1.47e-50, "
               "c2 vs h/k = 4.799e-11.", "Synthetically generated per PMLB metadata."],
        alternates=lambda t: [f"log({t[0]!r}*((v1)**(3))/(exp({t[1]!r}*v1/v2)-1))"]))

    # -- rydberg: y = ln(lambda) = -ln(c1 (1/n1^2 - 1/n2^2)) ---------------------------------
    def m_rydberg(X, c1):
        return -np.log(c1 * (1.0 / X[0] ** 2 - 1.0 / X[1] ** 2))
    def f_rydberg(X, y):
        term = 1.0 / X[0] ** 2 - 1.0 / X[1] ** 2
        return [float(np.exp(-np.mean(y + np.log(term))))]
    laws.append(Law(
        "rydberg", "rydberg", "-log({c1}*(1/((v1)**(2))-1/((v2)**(2))))", m_rydberg, f_rydberg,
        EMPIRICALBENCH, synthetic=True,
        columns=["n_1", "n_2"], target="ln(wavelength / m)",
        notes=["Target verified to be the NATURAL LOG of the wavelength in meters "
               "(n1=1, n2=2 row reproduces ln(121.6 nm) = -15.92, the Lyman-alpha line).",
               "Synthetically generated per PMLB metadata."],
        alternates=lambda t: [f"log(1/({t[0]!r}*(1/((v1)**(2))-1/((v2)**(2)))))"]))

    # -- leavitt: M = c1 logP + c2 ------------------------------------------------------------
    def m_leavitt(X, c1, c2):
        return c1 * X[0] + c2
    laws.append(Law(
        "leavitt", "leavitt", "{c1}*v1+{c2}", m_leavitt,
        lambda X, y: list(_lstsq(np.stack([X[0], np.ones_like(X[0])], axis=1), y)),
        EMPIRICALBENCH, synthetic=False,
        columns=["logP"], target="magnitude",
        notes=["Leavitt 1912 Cepheid data; x is already log10(period/days)."]))

    # -- schechter: ln(phi) = c1 + c2 ln(L) + c3 L (c3 = -1/L*, expected negative) -------------
    def m_schechter(X, c1, c2, c3):
        return c1 + c2 * np.log(X[0]) + c3 * X[0]
    laws.append(Law(
        "schechter", "schechter", "{c1}+{c2}*log(v1)+{c3}*v1", m_schechter,
        lambda X, y: list(_lstsq(np.stack([np.ones_like(X[0]), np.log(X[0]), X[0]], axis=1), y)),
        EMPIRICALBENCH, synthetic=True,
        columns=["L"], target="ln(number density)",
        notes=["Schechter function in log-density form: ln(phi* (L/L*)^alpha exp(-L/L*)) is linear "
               "in [1, ln L, L].", "Synthetically generated per PMLB metadata."],
        alternates=lambda t: [
            # product form with the derived (phi*, alpha, L*): alpha = c2, L* = -1/c3,
            # phi* = exp(c1 + c2 ln L*)
            f"log({float(np.exp(t[0] + t[1] * np.log(-1.0 / t[2])))!r}"
            f"*((v1/{-1.0 / t[2]!r})**({t[1]!r}))*exp(-v1/{-1.0 / t[2]!r}))"]))

    # -- bode: a = c1 + c2 exp(c3 n)  (canonical 0.4 + 0.3 * 2^n; c3 = ln 2) -------------------
    def m_bode(X, c1, c2, c3):
        return c1 + c2 * np.exp(c3 * X[0])
    def f_bode(X, y):
        best = None
        for c3 in np.linspace(0.05, 2.0, 400):
            feats = np.stack([np.ones_like(X[0]), np.exp(np.clip(c3 * X[0], -745, 700))], axis=1)
            theta = _lstsq(feats, y)
            fvu = _fvu(y, feats @ theta)
            if best is None or fvu < best[0]:
                best = (fvu, [float(theta[0]), float(theta[1]), float(c3)])
        return list(_polish(m_bode, X, y, best[1]))
    laws.append(Law(
        "bode", "bode", "{c1}+{c2}*exp({c3}*v1)", m_bode, f_bode,
        EMPIRICALBENCH, synthetic=False,
        columns=["n"], target="semi-major axis, AU",
        notes=["Bode's law 0.4 + 0.3*2^(n-1) rendered in exp-form (fitted c3 recovers ln 2; the "
               "dataset indexes Venus at n = 1, so the 2^(n-1) halving is absorbed into c2); "
               "Mercury ships as the PMLB sentinel n = -1000 (2^n underflows to exactly 0, as "
               "intended)."],
        alternates=lambda t: [f"{t[0]!r}+{t[1]!r}*(({float(np.exp(t[2]))!r})**(v1))"]))

    # -- tully_fisher: M = c1 ln(DV) + c2 (L ~ DV^2.5 in magnitudes) ---------------------------
    def m_tully(X, c1, c2):
        return c1 * np.log(X[0]) + c2
    laws.append(Law(
        "tully_fisher", "tully_fisher", "{c1}*log(v1)+{c2}", m_tully,
        lambda X, y: list(_lstsq(np.stack([np.log(X[0]), np.ones_like(X[0])], axis=1), y)),
        EMPIRICALBENCH, synthetic=False,
        columns=["DV"], target="absolute magnitude",
        notes=["Values -16..-21 are MAGNITUDES (PMLB metadata says 'luminosity'; Fig 5a of Tully & "
               "Fisher 1977 plots magnitude): L ~ DV^2.5 becomes M = c1 log(DV) + c2, theory slope "
               "c1 = -6.25/ln(10) = -2.71 in natural-log form."]))

    # -- absorption: A = log(1/(c1 + exp(-c2 x))) (MvSR refit simplified form) -----------------
    def m_absorption(X, c1, c2):
        return np.log(1.0 / (c1 + np.exp(-c2 * X[0])))
    def f_absorption(X, y):
        best = None
        for c2 in np.geomspace(1e-3, 10.0, 400):
            c1 = float(np.mean(np.exp(-y) - np.exp(-c2 * X[0])))
            if c1 <= 0:
                continue
            fvu = _fvu(y, m_absorption(X, c1, c2))
            if best is None or fvu < best[0]:
                best = (fvu, [c1, float(c2)])
        return list(_polish(m_absorption, X, y, best[1], bounds=([1e-12, 1e-12], [np.inf, np.inf])))
    laws.append(Law(
        "absorption", "absorption", "log(1/({c1}+exp(-{c2}*v1)))", m_absorption, f_absorption,
        MVSR, synthetic=False,
        columns=["concentration"], target="absorption",
        notes=["MvSR's refit simplified parametric form from the absorption notebook "
               "(log(1/(A + exp(-B*X))), guesses A=0.03, B=2)."]))

    # -- supernovae (2 bands): Bazin et al. 2009, f = c1 / (c2 exp(c3 t) + exp(-c4 t)) ---------
    def m_bazin(X, c1, c2, c3, c4):
        return c1 / (c2 * np.exp(np.clip(c3 * X[0], -700, 700)) + np.exp(np.clip(-c4 * X[0], -700, 700)))
    def f_bazin(X, y):
        mask = y > 1e-3
        t, ym = X[0][mask], y[mask]
        best = None
        for c3 in np.geomspace(1e-3, 1.0, 40):
            for c4 in np.geomspace(1e-2, 10.0, 40):
                feats = np.stack([np.exp(np.clip(c3 * t, -700, 700)),
                                  np.exp(np.clip(-c4 * t, -700, 700))], axis=1)
                w = ym ** 2                       # 1/y-space fit weighted back toward y-space
                theta = _lstsq(feats * w[:, None], (1.0 / ym) * w)
                a, b = float(theta[0]), float(theta[1])
                if a <= 0 or b <= 0:
                    continue
                cand = [1.0 / b, a / b, float(c3), float(c4)]
                fvu = _fvu(y, m_bazin(X, *cand))
                if best is None or fvu < best[0]:
                    best = (fvu, cand)
        return list(_polish(m_bazin, X, y, best[1],
                            bounds=([1e-12, 1e-12, 1e-6, 1e-6], [np.inf] * 4)))
    for band in ("zg", "zr"):
        laws.append(Law(
            f"supernovae_{band}", f"supernovae_{band}",
            "{c1}/({c2}*exp({c3}*v1)+exp(-{c4}*v1))", m_bazin, f_bazin,
            MVSR, synthetic=False,
            columns=["t"], target="normalized flux",
            notes=["Bazin et al. 2009 supernova light-curve function (the domain reference named "
                   f"in the MvSR notebook); ZTF DR17 event, {band} band, flux min-max normalized."]))

    return laws


# ---------------------------------------------------------------- build
def main() -> None:
    engine = load_engine("dev_7-3")
    problems: list[Problem] = []
    report: list[str] = [
        "# first-principles build report",
        "",
        "PMLB `first_principles_*` (MIT) + refit reference laws. Deterministic rebuild "
        "(`tools/build_first_principles.py`); no RNG.",
        "",
        "| eq_id | n | d | law | constants | FVU(ref) | R2 | gt_kind |",
        "|---|---|---|---|---|---|---|---|",
    ]

    for law in build_laws():
        path = os.path.join(UPSTREAM, f"first_principles_{law.dataset}.tsv.gz")
        raw = np.genfromtxt(io.TextIOWrapper(gzip.open(path)), delimiter="\t", names=True)
        cols = list(raw.dtype.names)
        assert cols[-1] == "target", f"{law.eq_id}: unexpected column layout {cols}"
        X = np.stack([raw[c].astype(np.float64) for c in cols[:-1]], axis=0)
        y = raw["target"].astype(np.float64)

        theta = [float(v) for v in law.fit(X, y)]
        y_ref = law.model(X, *theta)
        fvu = _fvu(y, y_ref)

        # render the law under dev_7-3 (constants inlined, full precision) + verify the rendering
        infix = law.infix_template.format(**{f"c{i+1}": repr(t) for i, t in enumerate(theta)})
        vars_info = {f"v{i+1}": {"name": law.columns[i]} for i in range(X.shape[0])}
        compiled = compile_expression(engine, law.eq_id, infix, vars_info, name="first-principles")
        y_engine = compiled["callable"](*[X[i] for i in range(X.shape[0])])
        y_engine = np.broadcast_to(np.asarray(y_engine, dtype=np.float64), y.shape)
        rel = np.max(np.abs(y_engine - y_ref) / np.maximum(np.abs(y_ref), 1e-30))
        assert rel < 1e-8, f"{law.eq_id}: engine rendering deviates from the fitted law (rel={rel:.2e})"

        # the CONCRETE ground truth: literal tokens + parse-order constants (the realize-path
        # convention; compiled["prefix"] is the masked SKELETON and must never land in expression)
        expression = list(compiled["expression"])
        constants = list(compiled["constants"])
        skeleton = tuple(compiled["prefix"])
        assert "<constant>" not in expression, f"{law.eq_id}: masked token leaked into expression"
        assert len(constants) == sum(1 for tok in expression if _is_number(tok)), \
            f"{law.eq_id}: constants/literals misaligned"

        # alternate canonical renderings: verify each is numerically THE SAME LAW on its finite
        # domain (a wrong alternate would silently hold out the wrong structure), then ship it
        # in meta for holdout registration.
        alternate_infixes: list[str] = []
        for alt in (law.alternates(theta) if law.alternates else []):
            alt_compiled = compile_expression(engine, f"{law.eq_id}:alt", alt, vars_info,
                                              name="first-principles")
            with np.errstate(all="ignore"):
                y_alt = alt_compiled["callable"](*[X[i] for i in range(X.shape[0])])
            y_alt = np.broadcast_to(np.asarray(y_alt, dtype=np.float64), y.shape)
            finite = np.isfinite(y_alt)
            assert finite.mean() >= 0.5, f"{law.eq_id}: alternate finite on <50% of support: {alt}"
            rel_alt = np.max(np.abs(y_alt[finite] - y_ref[finite])
                             / np.maximum(np.abs(y_ref[finite]), 1e-30))
            assert rel_alt < 1e-6, f"{law.eq_id}: alternate deviates (rel={rel_alt:.2e}): {alt}"
            assert tuple(alt_compiled["prefix"]) != skeleton, \
                f"{law.eq_id}: alternate is structurally identical to the stored form: {alt}"
            alternate_infixes.append(alt)

        gt_kind = "exact" if (law.synthetic and fvu <= 1e-12) else "reference"
        if law.synthetic and fvu > 1e-12:
            print(f"!! {law.eq_id}: metadata-synthetic but refit FVU={fvu:.3e} > 1e-12 -> kept 'reference'")

        problem = Problem.from_data(
            X.T, y,
            expression=expression,
            skeleton=skeleton,
            constants=constants,
            variables=list(compiled["variable_order"]),
            gt_kind=gt_kind,
            y_reference_support=y_ref,
            eq_id=law.eq_id,
            meta={
                "pmlb_dataset": f"first_principles_{law.dataset}",
                "columns": law.columns,
                "target": law.target,
                "reference_law": law.infix_template,
                "prepared_infix": infix,
                "alternate_renderings": alternate_infixes,
                "law_source": law.source,
                "constant_policy": "least-squares refit on the full dataset (deterministic)",
                "fitted_constants": theta,
                "reference_fvu_build": fvu,
                "license": "MIT (PMLB rendering)",
                "notes": law.notes,
            },
        )
        assert problem.skeleton is not None, f"{law.eq_id}: skeleton derivation failed"
        problems.append(problem)
        report.append(f"| {law.eq_id} | {len(y)} | {X.shape[0]} | `{law.infix_template}` | "
                      f"{['%.6g' % t for t in theta]} | {fvu:.3e} | {1 - fvu:.5f} | {gt_kind} |")
        print(f"{law.eq_id:16s} n={len(y):4d} d={X.shape[0]} FVU={fvu:10.3e} R2={1-fvu:8.5f} "
              f"{gt_kind:9s} theta={['%.6g' % t for t in theta]}")

    catalog = ProblemCatalog.from_problems(
        problems, name="first-principles", version=1,
        meta={
            "description": "SRBench-2.0 phenomenological track: PMLB first_principles_* measured/"
                           "frozen datasets + refit accepted reference laws (EmpiricalBench, MvSR, "
                           "Bazin 2009).",
            "license": "MIT (PMLB); laws: EmpiricalBench (Cranmer 2023) / MvSR (Russeil et al. 2024)",
            "builder": "tools/build_first_principles.py",
            "engine": "dev_7-3",
            "split_policy": "all measured points are support; no validation split (n = 6..243)",
        })
    out = catalog.save(OUT_NPZ)
    report += ["", f"{len(problems)} problems -> `{os.path.basename(str(out))}`.",
               "", "Split policy: all points support, empty validation. gt_kind rule: "
               "metadata-synthetic AND FVU<=1e-12 -> exact, else reference."]
    with open(OUT_REPORT, "w", encoding="utf-8") as handle:
        handle.write("\n".join(report) + "\n")
    print(f"\nSaved {out} + report.")


if __name__ == "__main__":
    main()
