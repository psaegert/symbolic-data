"""Recover the plotted data of the BMS Rössler figures (Supplementary Figs. S4 and S5 of Guimerà et al. 2020).

The supplementary PDF embeds each matplotlib figure verbatim as a form XObject (its PTEX.FileName names the source:
Ros_N5_deviations_ydot.pdf, Ros_N5_deviations_zdot.pdf, Ros_N5_deviations_minBIC_zdot.pdf). In those streams the
observed points are matplotlib markers placed by relative ``1 0 0 1 dx dy cm /M0 Do`` operators written with 10
decimals, so their positions are exact to ~1e-10 pt; the lines (observed polyline, true model, most plausible or
minimum-BIC model) are ``m``/``l`` paths with 6 decimals, thinned by matplotlib's path simplification (a subset of the
plotted vertices). The axes are calibrated from the tick marks and their labels.

Reads only the PDF bytes (stdlib zlib); runs nothing from the file.

    bms_rossler_figures.py <supplement.pdf> <out.json>
"""
from __future__ import annotations

import json
import re
import sys
import zlib

import numpy as np

FIGS = ("Ros_N5_deviations_ydot.pdf", "Ros_N5_deviations_zdot.pdf", "Ros_N5_deviations_minBIC_zdot.pdf")
# stroke colours (matplotlib may wrap the blue triple across two lines, so its key is the line's tail)
GREY, RED, BLUE = "0.7333333333 G", "1 0 0 RG", "0.7058823529 RG"


def form_streams(pdf: bytes) -> dict[str, bytes]:
    out = {}
    for m in re.finditer(rb"(\d+) (\d+) obj(.*?)stream\r?\n", pdf, re.S):
        head = m.group(3)
        name = re.search(rb"/PTEX\.FileName\(([^)]*)\)", head)
        if not name or b"Roessler" not in name.group(1):
            continue
        length = int(re.search(rb"/Length (\d+)", head).group(1))
        raw = pdf[m.end():m.end() + length]
        out[name.group(1).decode().rsplit("/", 1)[-1]] = zlib.decompress(raw)
    return out


def parse(stream: bytes) -> dict:
    lines = stream.decode("latin-1").splitlines()
    xticks, yticks = [], []            # (position, label)
    paths = []                         # (colour, [(x, y), ...]) of stroked paths inside the axes clip
    markers = []
    colour, cur = None, None
    pos = None
    pending = None                     # a tick waiting for its label: ("x"|"y", position)
    for i, ln in enumerate(lines):
        s = ln.strip()
        for c in (GREY, RED, BLUE):
            if c in s:
                colour = c
        m_cm = re.fullmatch(r"1 0 0 1 (-?[\d.]+) (-?[\d.]+) cm /M0 Do", s)
        if m_cm:
            dx, dy = float(m_cm.group(1)), float(m_cm.group(2))
            pos = (dx, dy) if pos is None else (pos[0] + dx, pos[1] + dy)
            markers.append(pos)
            continue
        m_m = re.fullmatch(r"(-?[\d.]+) (-?[\d.]+) m", s)
        m_l = re.fullmatch(r"(-?[\d.]+) (-?[\d.]+) l", s)
        if m_m:
            cur = [(float(m_m.group(1)), float(m_m.group(2)))]
            continue
        if m_l and cur is not None:
            cur.append((float(m_l.group(1)), float(m_l.group(2))))
            continue
        if s in ("S", "B") and cur is not None:
            if len(cur) == 2 and cur[0][0] == cur[1][0] and abs(cur[0][1] - cur[1][1]) == 6:   # an x tick
                pending = ("x", cur[0][0])
            elif len(cur) == 2 and cur[0][1] == cur[1][1] and abs(cur[0][0] - cur[1][0]) == 6:  # a y tick
                pending = ("y", cur[0][1])
            elif s == "S" and len(cur) > 20:
                paths.append((colour, cur))
            cur = None
            continue
        m_t = re.search(r"\((-?[\d.]+)\) Tj", s)
        if m_t and pending is not None:
            label = m_t.group(1)
            # the minus glyph is drawn after the digits (inside the same q..Q block): look ahead a few lines
            ahead = "\n".join(lines[i + 1:i + 8])
            neg = ("minus Do" in ahead.split(" m\n")[0]) if " m\n" in ahead else ("minus Do" in ahead)
            val = -float(label) if neg else float(label)
            (xticks if pending[0] == "x" else yticks).append((pending[1], val))
            pending = None
    return {"xticks": xticks, "yticks": yticks, "paths": paths, "markers": markers}


def affine(ticks):
    p = np.array([t[0] for t in ticks])
    v = np.array([t[1] for t in ticks])
    s, b = np.polyfit(v, p, 1)
    resid = p - (s * v + b)
    return s, b, float(np.max(np.abs(resid)))


def main() -> None:
    pdf = open(sys.argv[1], "rb").read()
    streams = form_streams(pdf)
    out = {}
    for name in FIGS:
        f = parse(streams[name])
        sx, bx, rx = affine(f["xticks"])
        sy, by, ry = affine(f["yticks"])
        mk = np.array(f["markers"])
        t = (mk[:, 0] - bx) / sx
        v = (mk[:, 1] - by) / sy
        series = {}
        for colour, pts in f["paths"]:
            key = {GREY: "observed_line", RED: "true_model", BLUE: "fitted_model"}.get(colour, colour)
            a = np.array(pts)
            series.setdefault(key, []).append({"t": ((a[:, 0] - bx) / sx).tolist(), "v": ((a[:, 1] - by) / sy).tolist()})
        out[name] = {"xticks": f["xticks"], "yticks": f["yticks"], "x_scale_pt": sx, "x_offset_pt": bx,
                     "y_scale_pt": sy, "y_offset_pt": by, "tick_fit_max_resid_pt": [rx, ry],
                     "n_markers": int(len(mk)), "t": t.tolist(), "observed": v.tolist(), "lines": series}
        print(name, "markers", len(mk), "xticks", f["xticks"], "yticks", f["yticks"], "scale", sx, sy,
              "tick resid", rx, ry, "lines", {k: [len(s["t"]) for s in vv] for k, vv in series.items()})
    json.dump(out, open(sys.argv[2], "w"))


if __name__ == "__main__":
    main()
