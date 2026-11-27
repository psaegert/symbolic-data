"""The rows of Brum et al.'s dynamical-systems benchmark, regenerated: each ODE right-hand side at the authors'
solved states.

Brum, Lober, Previdelli & Rodrigues, "Discovering equations from data: symbolic regression in dynamical systems"
(arXiv:2508.20257; J. Phys. Complexity, doi:10.1088/2632-072X/ae3eed) benchmark symbolic regression on nine ODE
systems: the Lorenz system, the nonlinear pendulum, Lotka-Volterra predator-prey dynamics and six compartmental
epidemic models (SIS, SIR, SIRV, SIRS, SEIR, SEIRD). The authors publish no data files: their notebooks
(github.com/luizalober/review_symb_regression @ 094bd154) solve each system with scipy's solve_ivp (RK45, default
tolerances rtol 1e-3, atol 1e-6) on t_eval = np.arange(0, t_end, t_step), and fit one regression per state variable:
all states as inputs, that state's right-hand side evaluated at the solved states as the target. This module redoes
that from the settings the notebooks use (the PySR and ODEFormer notebooks'; for every system most of the other
notebooks share them), with its own right-hand sides written in the notebooks' order of operations.

Where the paper's parameter tables differ from the notebooks, the notebooks are followed, because they made the rows:
the Lorenz grid step is 0.02 (the paper's table prints 2E-3, which only the PySINDy notebook uses); the pendulum runs
on [0, 10) with step 0.01 from
theta0 = 45 degrees in radians (table: [0, 5], 2E-3); the Lotka-Volterra predator term is d*b*u*v = 0.1875*u*v (the
paper prints delta = 0.375); SIRV starts from I0 = 0.1, S0 = 0.85 with vaccination rate 0.2 (table: I0 = 0.01,
S0 = 0.94, epsilon = 0.5; the paper's results table prints the rate 0.2). Compartments are fractions (N = 1).

The integrator is scipy's RK45 re-written (Dormand-Prince 5(4): the same tableau, error estimate, step-size control,
initial step, dense output and t_eval handling as scipy.integrate.solve_ivp), with every sum spelled out in a fixed
order, the step-size arithmetic in Python floats and sin from the C library. scipy's own RK45 forms its sums with BLAS
(np.dot), whose kernels differ between CPUs in the last bit, and numpy dispatches sin to CPU-specific code; written out
this way the bytes are the same on any IEEE-754 machine with the same C library (pow, sin). The states agree with
scipy's solve_ivp to rounding (solve_scipy is the cross-check).

``generate()`` returns, per problem id, the inputs ``x`` (all state variables, one row per time point), the target
``y`` (one state's derivative) and the feature names. No split: the authors fit and score every derivative on all rows
of its trajectory. numpy only (scipy is used only by ``solve_scipy``, the cross-check); deterministic (no random
numbers; fixed grid, method and tolerances).
"""
from __future__ import annotations

import math

import numpy as np

RTOL, ATOL = 1e-3, 1e-6        # solve_ivp's defaults: the notebooks pass no tolerances


# The right-hand sides, each in the notebooks' order of operations. They work on one state vector (inside the
# integrator) and on the (states, rows) array of solved states.
def lorenz(t, s):
    x, y, z = s
    return [2.0 * (y - x), x * (1.0 - z) - y, x * y - 2.6 * z]


def _sin(theta):
    """sin from the C library, element by element (numpy may dispatch sin to CPU-specific vector code)."""
    if np.ndim(theta) == 0:
        return math.sin(theta)
    return np.array([math.sin(v) for v in np.asarray(theta, dtype=np.float64).ravel()]).reshape(np.shape(theta))


def pendulum(t, s):
    theta, omega = s
    return [omega, -(9.8 / 1.0) * _sin(theta)]


def lotka_volterra(t, s):
    u, v = s
    return [2 * u - 0.5 * u * v, -1 * v + 0.375 * 0.5 * u * v]


def sis(t, s):
    S, I = s  # noqa: E741  (the compartments as the paper names them)
    return [-0.3 * S * I + 0.1 * I, 0.3 * S * I - 0.1 * I]


def sir(t, s):
    S, I, R = s  # noqa: E741  (the compartments as the paper names them)
    return [-0.5 * S * I, 0.5 * S * I - 0.1 * I, 0.1 * I]


def sirv(t, s):
    S, I, R, V = s  # noqa: E741  (the compartments as the paper names them)
    return [-0.5 * S * I - 0.2 * S, 0.5 * S * I - 0.1 * I, 0.1 * I, 0.2 * S]


def sirs(t, s):
    S, I, R = s  # noqa: E741  (the compartments as the paper names them)
    return [-0.5 * S * I + 0.2 * R, 0.5 * S * I - 0.1 * I, 0.1 * I - 0.2 * R]


def seir(t, s):
    S, E, I, R = s  # noqa: E741  (the compartments as the paper names them)
    return [-0.5 * S * I, 0.5 * S * I - 0.5 * E, 0.5 * E - 0.1 * I, 0.1 * I]


def seird(t, s):
    S, E, I, R, D = s  # noqa: E741  (the compartments as the paper names them)
    return [-0.5 * S * I, 0.5 * S * I - 0.2 * E, 0.2 * E - (0.1 + 0.1) * I, 0.1 * I, 0.1 * I]


# Each system: subset (the paper's two tables), right-hand side, feature names (the paper's symbols, spelled for the
# compartments as the authors' GPLearn notebook names its columns, which also keeps sympy's S, E, I free), initial
# state (computed as the notebooks compute it), t_end, t_step, and each derivative as a formula over the features.
N = 1
SYSTEMS = {
    "lorenz": dict(
        subset="general-dynamics", rhs=lorenz, features=["x", "y", "z"], y0=[0.6, 2.0, 1.0], t_end=5.0, t_step=0.02,
        formulas=["2.0*(y - x)", "x*(1.0 - z) - y", "x*y - 2.6*z"]),
    "pendulum": dict(
        subset="general-dynamics", rhs=pendulum, features=["theta", "omega"], y0=[np.radians(45), 0],
        t_end=10, t_step=0.01, formulas=["omega", "-9.8*sin(theta)"]),
    "lotka_volterra": dict(
        subset="general-dynamics", rhs=lotka_volterra, features=["u", "v"], y0=[20, 5], t_end=7.5, t_step=0.1,
        formulas=["2*u - 0.5*u*v", "-v + 0.1875*u*v"]),
    "sis": dict(
        subset="epidemiological", rhs=sis, features=["susceptible", "infected"], y0=[N - 0.01, 0.01],
        t_end=100.0, t_step=0.1,
        formulas=["-0.3*susceptible*infected + 0.1*infected", "0.3*susceptible*infected - 0.1*infected"]),
    "sir": dict(
        subset="epidemiological", rhs=sir, features=["susceptible", "infected", "recovered"],
        y0=[N - 0.01 - 0.0, 0.01, 0.0], t_end=75.0, t_step=0.1,
        formulas=["-0.5*susceptible*infected", "0.5*susceptible*infected - 0.1*infected", "0.1*infected"]),
    "sirv": dict(
        subset="epidemiological", rhs=sirv, features=["susceptible", "infected", "recovered", "vaccinated"],
        y0=[N - 0.1 - 0.0 - 0.05, 0.1, 0.0, 0.05], t_end=35.0, t_step=0.1,
        formulas=["-0.5*susceptible*infected - 0.2*susceptible", "0.5*susceptible*infected - 0.1*infected",
                  "0.1*infected", "0.2*susceptible"]),
    "sirs": dict(
        subset="epidemiological", rhs=sirs, features=["susceptible", "infected", "recovered"],
        y0=[N - 0.01 - 0.0, 0.01, 0.0], t_end=60.0, t_step=0.1,
        formulas=["-0.5*susceptible*infected + 0.2*recovered", "0.5*susceptible*infected - 0.1*infected",
                  "0.1*infected - 0.2*recovered"]),
    "seir": dict(
        subset="epidemiological", rhs=seir, features=["susceptible", "exposed", "infected", "recovered"],
        y0=[N - 0.1 - 0.0 - 0.1, 0.1, 0.1, 0.0], t_end=80.0, t_step=0.1,
        formulas=["-0.5*susceptible*infected", "0.5*susceptible*infected - 0.5*exposed",
                  "0.5*exposed - 0.1*infected", "0.1*infected"]),
    "seird": dict(
        subset="epidemiological", rhs=seird, features=["susceptible", "exposed", "infected", "recovered", "deceased"],
        y0=[N - 0.01 - 0.0 - 0.0 - 0.0, 0.0, 0.01, 0.0, 0.0], t_end=120.0, t_step=0.1,
        formulas=["-0.5*susceptible*infected", "0.5*susceptible*infected - 0.2*exposed",
                  "0.2*exposed - 0.2*infected", "0.1*infected", "0.1*infected"]),
}

# ------------------------------------------------------------------------------------ Dormand-Prince 5(4) (RK45)
# scipy.integrate._ivp.rk.RK45's coefficients, as the same Python divisions
C = (0, 1 / 5, 3 / 10, 4 / 5, 8 / 9, 1)
A = ((), (1 / 5,), (3 / 40, 9 / 40), (44 / 45, -56 / 15, 32 / 9),
     (19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729),
     (9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656))
B = (35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84)
E = (-71 / 57600, 0, 71 / 16695, -71 / 1920, 17253 / 339200, -22 / 525, 1 / 40)
P = ((1, -8048581381 / 2820520608, 8663915743 / 2820520608, -12715105075 / 11282082432),
     (0, 0, 0, 0),
     (0, 131558114200 / 32700410799, -68118460800 / 10900136933, 87487479700 / 32700410799),
     (0, -1754552775 / 470086768, 14199869525 / 1410260304, -10690763975 / 1880347072),
     (0, 127303824393 / 49829197408, -318862633887 / 49829197408, 701980252875 / 199316789632),
     (0, -282668133 / 205662961, 2019193451 / 616988883, -1453857185 / 822651844),
     (0, 40617522 / 29380423, -110615467 / 29380423, 69997945 / 29380423))
SAFETY, MIN_FACTOR, MAX_FACTOR = 0.9, 0.2, 10
ERROR_EXPONENT = -1 / (4 + 1)


def _comb(vectors, coef):
    """sum_j vectors[j] * coef[j], left to right (scipy: np.dot, i.e. BLAS)."""
    acc = vectors[0] * float(coef[0])
    for j in range(1, len(coef)):
        acc = acc + vectors[j] * float(coef[j])
    return acc


def _rms(x) -> float:
    """scipy's norm, np.linalg.norm(x) / x.size ** 0.5, with the sum of squares left to right."""
    s = 0.0
    for v in np.asarray(x, dtype=np.float64).tolist():
        s = s + v * v
    return math.sqrt(s) / len(x) ** 0.5


def _initial_step(fun, t0, y0, f0, t_bound) -> float:
    """scipy.integrate._ivp.common.select_initial_step for RK45 (error estimator order 4, direction +1)."""
    interval_length = abs(t_bound - t0)
    scale = ATOL + np.abs(y0) * RTOL
    d0, d1 = _rms(y0 / scale), _rms(f0 / scale)
    h0 = 1e-6 if d0 < 1e-5 or d1 < 1e-5 else 0.01 * d0 / d1
    h0 = min(h0, interval_length)
    f1 = fun(t0 + h0, y0 + h0 * f0)
    d2 = _rms((f1 - f0) / scale) / h0
    if d1 <= 1e-15 and d2 <= 1e-15:
        h1 = max(1e-6, h0 * 1e-3)
    else:
        h1 = (0.01 / max(d1, d2)) ** (1 / (4 + 1))
    return min(100 * h0, h1, interval_length)


def integrate(rhs, t_span, y0, t_eval) -> np.ndarray:
    """The states at t_eval (rows x states), step for step as solve_ivp(rhs, t_span, y0, method='RK45',
    t_eval=t_eval) with rtol 1e-3 and atol 1e-6 computes them, the sums in a fixed order."""
    def fun(t, y):
        return np.asarray(rhs(t, y), dtype=np.float64)

    t, t_bound = map(float, t_span)
    y = np.asarray(y0).astype(np.float64)
    f = fun(t, y)
    h_abs = _initial_step(fun, t, y, f, t_bound)
    t_eval = np.asarray(t_eval, dtype=np.float64)
    out, i_eval = [], 0
    while True:
        min_step = 10 * abs(math.nextafter(t, math.inf) - t)
        h_abs = min_step if h_abs < min_step else h_abs
        rejected = False
        while True:                                                   # one accepted step (RK45._step_impl)
            if h_abs < min_step:
                raise RuntimeError("step size too small")
            t_new = t + h_abs
            if t_new - t_bound > 0:
                t_new = t_bound
            h = t_new - t
            h_abs = abs(h)
            K = [f]
            for s in range(1, 6):
                K.append(fun(t + C[s] * h, y + _comb(K, A[s]) * h))
            y_new = y + h * _comb(K, B)
            f_new = fun(t + h, y_new)
            K.append(f_new)
            scale = ATOL + np.maximum(np.abs(y), np.abs(y_new)) * RTOL
            error_norm = _rms(_comb(K, E) * h / scale)
            if error_norm < 1:
                factor = MAX_FACTOR if error_norm == 0 else min(MAX_FACTOR, SAFETY * error_norm ** ERROR_EXPONENT)
                if rejected:
                    factor = min(1, factor)
                h_abs *= factor
                break
            h_abs *= max(MIN_FACTOR, SAFETY * error_norm ** ERROR_EXPONENT)
            rejected = True
        t_old, y_old = t, y
        t, y, f = t_new, y_new, f_new
        i_new = int(np.searchsorted(t_eval, t, side="right"))         # t_eval points in (t_old, t]: dense output
        if i_new > i_eval:
            Q = [_comb(K, [P[s][j] for s in range(7)]) for j in range(4)]
            hh = t - t_old
            for te in t_eval[i_eval:i_new].tolist():
                x = (te - t_old) / hh
                p = [x]
                for _ in range(3):
                    p.append(p[-1] * x)
                out.append(_comb(Q, p) * hh + y_old)
            i_eval = i_new
        if t - t_bound >= 0:
            break
    return np.array(out, dtype=np.float64).reshape(len(t_eval), len(y))


def problem_id(system: str, feature: str) -> str:
    return f"brum-2025/{system}_{feature}"


def solve(system: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The time grid, the solved states (rows x states) and the right-hand side at them (rows x states)."""
    s = SYSTEMS[system]
    t_eval = np.arange(0.0, s["t_end"], s["t_step"])
    states = np.ascontiguousarray(integrate(s["rhs"], (0.0, s["t_end"]), s["y0"], t_eval))
    rates = np.ascontiguousarray(np.asarray(s["rhs"](0.0, states.T), dtype=np.float64).T)
    return t_eval, states, rates


def solve_scipy(system: str) -> np.ndarray:
    """The cross-check: the states as scipy's solve_ivp computes them on this machine (BLAS-dependent last bits)."""
    from scipy.integrate import solve_ivp
    s = SYSTEMS[system]
    t_eval = np.arange(0.0, s["t_end"], s["t_step"])
    sol = solve_ivp(s["rhs"], (0.0, s["t_end"]), s["y0"], method="RK45", t_eval=t_eval, rtol=RTOL, atol=ATOL)
    return np.ascontiguousarray(sol.y.T)


def generate() -> dict[str, dict]:
    """Per problem id: ``features``, ``x`` (float64, rows x states), ``y`` (float64, one derivative), ``system``,
    ``target`` and ``formula``. One problem per state variable of each system (28 in all); no test arrays (no split)."""
    out: dict[str, dict] = {}
    for system, s in SYSTEMS.items():
        _, states, rates = solve(system)
        for k, feat in enumerate(s["features"]):
            out[problem_id(system, feat)] = {
                "features": list(s["features"]), "x": states.copy(), "y": np.ascontiguousarray(rates[:, k]),
                "system": system, "target": f"d{feat}/dt", "formula": s["formulas"][k]}
    return out


if __name__ == "__main__":
    import hashlib
    for pid, p in generate().items():
        digest = {r: hashlib.sha256(np.ascontiguousarray(p[r]).tobytes()).hexdigest()[:16] for r in ("x", "y")}
        print(f"{pid:40s} {p['x'].shape} x {digest['x']} y {digest['y']}")
