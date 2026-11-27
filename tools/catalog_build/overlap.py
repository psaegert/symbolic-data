"""How much of each release catalog srbf also runs in another benchmark (owner 2026-11-27: every compilation is kept
whole as published and every copy is fitted on its own; the overlap is measured and shown).

Per catalog, over the problems the release runs (count_release.py's rules):

* ``same_law_elsewhere``: problems whose law is also posed by another benchmark: the same function of its inputs, up
  to renaming and reordering the inputs (unused inputs aside), its constants included;
* ``identical_elsewhere``: of those, the problems that are identical there: the same law on the same input
  distributions and number of inputs (formula problems), or the same fitting and test rows (data problems);
* ``identical_within``: problems identical to another problem of the same catalog (copies inside one benchmark).

A law is fingerprinted numerically: the formula evaluated on a fixed set of points, over every assignment of the
points' coordinates to its inputs (up to 5 inputs; more in their order of first use), the values rounded to 9
significant digits. A data problem's law is its stored ground truth (exact or reference); one without a formula has
none. Written to tags/overlap.json, for the site.

    overlap.py [--catalogs build/data-catalogs] [--out tools/catalog_build/tags/overlap.json]
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import re
import sys
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import count_release  # noqa: E402  the release's catalog rules
import paths  # noqa: E402

K_POINTS = 6
MAX_PERMUTED = 5
_VAR = re.compile(r"\bv(\d+)\b")
_POINTS = np.random.default_rng(20261128).uniform(0.15, 0.85, (K_POINTS, 2048))     # column j: the j-th coordinate


def _lambdified(prepared: str, used: list[str]):
    import sympy
    from sympy.parsing.sympy_parser import parse_expr
    local = {v: sympy.Symbol(v) for v in used}
    e = parse_expr(prepared.replace("^", "**"), local_dict={**local, "abs": sympy.Abs})
    return sympy.lambdify([local[v] for v in used], e, modules=["numpy", "scipy"])


def law_key(prepared: str) -> str | None:
    """The fingerprint of a formula over v1..vN (see the module text); None when it cannot be evaluated."""
    order = list(dict.fromkeys(f"v{m}" for m in _VAR.findall(prepared)))
    if not order:
        return None
    try:
        f = _lambdified(prepared, order)
    except Exception:
        return None
    d = len(order)
    perms = itertools.permutations(range(d)) if d <= MAX_PERMUTED else [tuple(range(d))]
    values = []
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        for perm in perms:
            cols = [_POINTS[:, perm[i]] for i in range(d)]
            try:
                y = np.broadcast_to(np.asarray(f(*cols), dtype=complex), (K_POINTS,))
            except Exception:
                return None
            values.extend(np.where((y.imag == 0) & np.isfinite(y.real), y.real, np.nan))
    text = ",".join(sorted("nan" if not np.isfinite(v) else f"{v:.9g}" for v in values))
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _spec(v: dict[str, Any]) -> str:
    keep = {k: v[k] for k in ("sample_range", "sample_type", "dist", "expr") if k in v}
    return json.dumps(keep, sort_keys=True)


def yaml_problems(path: Path, keep: set[str] | None) -> list[dict[str, Any]]:
    doc = paths.load_yaml(path)
    out = []
    for key, e in doc["expressions"].items():
        if keep is not None and str(key) not in keep:
            continue
        prepared = str(e.get("prepared") or "")
        used = list(dict.fromkeys(f"v{m}" for m in _VAR.findall(prepared)))
        inputs = e.get("inputs") or {}
        specs = sorted(_spec((e.get("vars") or {}).get(v) or {}) for v in used)
        tail = json.dumps({"inputs": inputs, "latent": e.get("latent"), "per_run": bool(e.get("per_run")),
                           "test": e.get("test")}, sort_keys=True, default=str)
        out.append({"id": str(key), "prepared": prepared,
                    "inputs": f"{len([k for k in (e.get('vars') or {}) if k != 'v0'])}|" + "|".join(specs) + tail})
    return out


def npz_problems(path: Path, engine) -> list[dict[str, Any]]:
    z = np.load(path)
    meta = json.loads(str(z["_meta"]))
    out = []
    for i, p in enumerate(meta["problems"]):
        prepared = None
        if p.get("expression") and p.get("gt_kind") in ("exact", "reference"):
            try:
                prepared = engine.prefix_to_infix(list(p["expression"]), power="**", realization=True)
            except Exception:
                prepared = None
            if prepared:
                prepared = re.sub(r"\bx(\d+)\b", lambda m: f"v{m.group(1)}", prepared)
        digest = hashlib.sha256()
        for part in ("x_support", "y_support", "x_validation", "y_validation"):     # the fitting and the test rows
            if f"p{i}__{part}" in z.files:
                a = np.round(np.asarray(z[f"p{i}__{part}"], dtype=np.float64), 10)
                digest.update(str(a.shape).encode() + a.tobytes())
        rows = digest.hexdigest()[:16]
        out.append({"id": str(p.get("eq_id")), "prepared": prepared, "data": rows})
    return out


def measure(symbolic_data: Path, data_build: Path) -> dict[str, Any]:
    from symbolic_data._evaluation import load_engine
    engine = load_engine(None)
    benchmark = {c: b["benchmark"] for b in json.loads((Path(__file__).resolve().parent / "tags/benchmarks.json").read_text())
                 for c in b["catalogs"]}
    frozen = {n: set(json.loads(paths.frozen_path(f).read_text())["ids"]) for n, f in count_release.FROZEN.items()}
    problems: dict[str, list[dict[str, Any]]] = {}
    from render_draws import release_catalogs
    cache: dict[str, str | None] = {}
    for path in release_catalogs(symbolic_data, data_build):
        name = path.stem.replace(".v2", "")
        rows = yaml_problems(path, frozen.get(name)) if path.suffix == ".yaml" else npz_problems(path, engine)
        for r in rows:
            if r["prepared"]:
                if r["prepared"] not in cache:
                    cache[r["prepared"]] = law_key(r["prepared"])
                r["law"] = cache[r["prepared"]]
            else:
                r["law"] = None
            r["identical"] = (f"{r['law']}|{r['inputs']}" if r.get("inputs") is not None and r["law"]
                              else f"data|{r['data']}" if r.get("data") else None)
        problems[name] = rows
    by_law: dict[str, set[str]] = defaultdict(set)
    by_identical: dict[str, set[str]] = defaultdict(set)
    for name, rows in problems.items():
        for r in rows:
            if r["law"]:
                by_law[r["law"]].add(benchmark.get(name, name))
            if r["identical"]:
                by_identical[r["identical"]].add(benchmark.get(name, name))
    out = []
    for name, rows in sorted(problems.items()):
        mine = benchmark.get(name, name)
        same = [r for r in rows if r["law"] and by_law[r["law"]] - {mine}]
        ident = [r for r in rows if r["identical"] and by_identical[r["identical"]] - {mine}]
        within = Counter(r["identical"] for r in rows if r["identical"])
        others: Counter[str] = Counter()
        for r in same:
            others.update(by_law[r["law"]] - {mine})
        out.append({"catalog": name, "benchmark": mine, "problems": len(rows),
                    "with_law": sum(1 for r in rows if r["law"]),
                    "same_law_elsewhere": len(same), "identical_elsewhere": len(ident),
                    "identical_within": sum(n for n in within.values() if n > 1),
                    "shares_laws_with": dict(others.most_common())})
    total = sum(r["problems"] for r in out)
    return {"rule": __doc__.split("\n\n")[1].strip(),
            "totals": {"catalogs": len(out), "problems": total,
                       "same_law_elsewhere": sum(r["same_law_elsewhere"] for r in out),
                       "identical_elsewhere": sum(r["identical_elsewhere"] for r in out),
                       "identical_within": sum(r["identical_within"] for r in out)},
            "catalogs": out}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbolic-data", default=str(paths.REPO))
    ap.add_argument("--catalogs", default=str(paths.DATA_BUILD))
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "tags/overlap.json"))
    a = ap.parse_args()
    report = measure(Path(a.symbolic_data), Path(a.catalogs))
    Path(a.out).write_text(json.dumps(report, indent=1) + "\n")
    t = report["totals"]
    print(f"{t['catalogs']} catalogs, {t['problems']:,} problems: same law in another benchmark {t['same_law_elsewhere']:,} "
          f"({t['same_law_elsewhere'] / t['problems']:.1%}), identical there {t['identical_elsewhere']:,} "
          f"({t['identical_elsewhere'] / t['problems']:.1%}), identical within their catalog {t['identical_within']:,} -> {a.out}")


if __name__ == "__main__":
    main()
