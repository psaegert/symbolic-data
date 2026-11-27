"""srbf's regeneration of the Bayesian machine scientist's Rössler data (Guimerà et al. 2020, Sci. Adv. 6:eaav6971,
Fig. 4, Supplementary Text S4, Figs. S4-S5).

The authors published no data files for this experiment. Their generator is identified from the vector data of
Supplementary Figs. S4 and S5 (identify_generator.py): forward Euler with step h = 0.001 from the origin
(x, y, z) = (0, 0, 0) for the Rössler system with a = b = 0.2, c = 5.7; the first sample is taken after the first step
and then one every 100 steps, 1,000 samples (plotted at t = 0, 0.1, ..., 99.9). This trajectory reproduces the
authors' true-model curves of both figures to 1e-7 (ẏ) and 3e-7 (ż), the figures' coordinate precision, at every
plotted vertex over the whole run. The states are stored without error, as in the paper.

Targets: ẏ and ż at σ = 5, the authors' own noisy values, recovered from the figures' markers (bms_rossler_figures.py;
two independent recoveries of ż agree to 1.3e-7) and stored rounded to 1e-6. The authors' ẋ at σ = 5 and the three
series at σ = 1 were never published, so srbf does not run them.

The two files go into SYMBOLIC_DATA_CATALOG_RAW/bms-rossler, where the records (additions_2026-11.json) pin them by sha256:

    bms_rossler_figures.py <aav6971_SM.pdf> <fig_extract.json>
    bms_rossler_rows.py <fig_extract.json> <SYMBOLIC_DATA_CATALOG_RAW>/bms-rossler
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

A = B = 0.2
C = 5.7
H = 0.001          # Euler step
EVERY = 100        # steps between samples (0.1 time units)
N = 1000
FIGURE = {("ydot", 5.0): "Ros_N5_deviations_ydot.pdf", ("zdot", 5.0): "Ros_N5_deviations_zdot.pdf"}


def trajectory() -> np.ndarray:
    """The 1,000 sampled states (x, y, z): forward Euler, step H, from the origin; samples after steps 1, 101, ..."""
    s = np.zeros(3)
    out = np.empty((N, 3))
    for k in range(N):
        for _ in range(1 if k == 0 else EVERY):
            x, y, z = s
            s = s + H * np.array([-y - z, x + A * y, B + z * (x - C)])
        out[k] = s
    return out


def rhs(states: np.ndarray) -> dict[str, np.ndarray]:
    x, y, z = states.T
    return {"xdot": -y - z, "ydot": x + A * y, "zdot": B + z * (x - C)}


def targets(states: np.ndarray, fig: dict) -> tuple[dict, dict]:
    """The two noisy targets the authors published, by (derivative, σ), and where each comes from."""
    out, origin = {}, {}
    for key, name in FIGURE.items():
        out[key] = np.round(np.array(fig[name]["observed"]), 6)
        origin[key] = f"authors' values recovered from the vector data of {name} (Supplementary Figs. S4/S5)"
    return out, origin


def file_name(deriv: str, sigma: float) -> str:
    return f"rossler_{deriv}_sigma{sigma:g}.csv"


def write(out_dir: Path, fig: dict) -> dict:
    states = trajectory()
    ys, origin = targets(states, fig)
    out_dir.mkdir(parents=True, exist_ok=True)
    t = 0.1 * np.arange(N)
    info = {}
    for (deriv, sigma), y in sorted(ys.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        lines = [f"t,x,y,z,{deriv}"]
        for k in range(N):
            # shortest round-trip spelling of each float64 (repr of a Python float)
            row = [float(states[k, 0]), float(states[k, 1]), float(states[k, 2]), float(y[k])]
            lines.append(f"{t[k]:.1f}," + ",".join(repr(v) for v in row))
        data = ("\n".join(lines) + "\n").encode()
        path = out_dir / file_name(deriv, sigma)
        path.write_bytes(data)
        info[file_name(deriv, sigma)] = {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data),
                                         "target": deriv, "sigma": sigma, "origin": origin[(deriv, sigma)]}
    return info


def main() -> None:
    fig = json.load(open(sys.argv[1]))
    info = write(Path(sys.argv[2]), fig)
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
