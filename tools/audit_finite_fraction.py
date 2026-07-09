"""Per-entry valid-domain disclosure: MC-estimate each entry's per-point finite fraction f.

Writes `finite_fraction` (and `low_validity: true` below the floor) into entry meta for
UNPUBLISHED catalogs passed with --write; published catalogs are report-only (forward-only
policy). The sampling law disclosure: accepted points follow the DECLARED distribution
CONDITIONED on the expression's valid domain (identical under whole-draw and per-point
rejection); f quantifies how far that conditional sits from the nominal box.

Run: python tools/audit_finite_fraction.py [--write] <catalog.yaml> [...]
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import yaml

from simplipy import SimpliPyEngine
from symbolic_data.catalog import ProblemCatalog
from symbolic_data.distributions import fastsrb_dist
from symbolic_data._evaluation import evaluate

N_MC = 20_000
FLOOR = 0.05


def audit_catalog(path: Path, engine, write: bool) -> dict:
    cfg = yaml.safe_load(path.read_text())
    catalog = ProblemCatalog.load(str(path))
    rng = np.random.default_rng(20260710)   # audit-only MC seed; never used for benchmark draws
    rows, flagged = {}, []
    for entry in catalog.entries.values():
        try:
            compiled = catalog._compiled(entry, engine)
            order = compiled["variable_order"]
            cols = [fastsrb_dist(entry.variables[k]["sample_range"][0],
                                 entry.variables[k]["sample_range"][1],
                                 base=entry.variables[k]["sample_type"][0],
                                 sign=entry.variables[k]["sample_type"][1],
                                 layout="random", size=N_MC, rng=rng) for k in order]
            x = np.column_stack(cols).astype(float)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                y = np.asarray(evaluate(compiled, {v: x[:, i] for i, v in enumerate(order)}),
                               dtype=float)
            y = np.broadcast_to(y, (N_MC,) if y.ndim == 0 else y.shape)
            f = float((np.isfinite(x).all(axis=1) & np.isfinite(np.atleast_2d(y).reshape(N_MC, -1)).all(axis=1)).mean())
        except Exception as exc:
            rows[entry.id] = {"finite_fraction": None, "error": str(exc)[:80]}
            flagged.append(entry.id)
            continue
        rows[entry.id] = {"finite_fraction": round(f, 4)}
        if f < FLOOR:
            rows[entry.id]["low_validity"] = True
            flagged.append(entry.id)
        if write:
            meta = cfg["expressions"][entry.id].setdefault("meta", {})
            meta["finite_fraction"] = round(f, 4)
            if f < FLOOR:
                meta["low_validity"] = True
    if write:
        conv = cfg["metadata"].setdefault("conventions", {})
        conv["validity"] = ("accepted points follow the declared per-variable distribution "
                            "CONDITIONED on the expression's valid domain; meta.finite_fraction "
                            f"is the MC-estimated per-point valid fraction (n={N_MC}); entries "
                            f"below {FLOOR} carry low_validity: true and need per-entry review.")
        path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return {"catalog": path.stem, "n": len(rows), "flagged": flagged,
            "min_f": min((r["finite_fraction"] for r in rows.values() if r["finite_fraction"] is not None), default=None),
            "rows": rows}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("catalogs", nargs="+")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--report", default=None)
    args = ap.parse_args()
    engine = SimpliPyEngine.load("dev_7-3", install=True)
    reports = [audit_catalog(Path(c), engine, args.write) for c in args.catalogs]
    for r in reports:
        fs = [v["finite_fraction"] for v in r["rows"].values() if v["finite_fraction"] is not None]
        n_partial = sum(1 for f in fs if f < 1.0)
        print(f"{r['catalog']:15s} n={r['n']:4d}  partial-domain={n_partial:4d}  "
              f"min_f={r['min_f']}  flagged(<{FLOOR})={len(r['flagged'])}: {r['flagged'][:6]}")
    if args.report:
        Path(args.report).write_text(json.dumps(reports, indent=1))


if __name__ == "__main__":
    main()
