"""Build the ``feynman-bonus`` catalog: the 20 AI-Feynman BONUS equations, original rendering.

Modeled 1:1 on feynman@1 (aifeynman-original, main set): declarative entries with the original
uniform sampling boxes; `raw` keeps the symbolic variable names and arc-function spellings,
`prepared` is v-indexed with `asin`/`acos` and numeric pi (house conventions). Ids B1..B20 align
with the SRSD rendering already in the family (fastsrb@1 / srsd-dummy@1 B1-B20); per-entry
meta.base_catalog/base_eq_id links record the 1:1 correspondence for the dedup registry.

Errata applied + asserted (PhySO's documented fix): the '# variables' column undercounts
test_12/13/18 (4 -> 5) and test_19 (5 -> 6); the parser derives the true count from the filled
v{i}_name columns and asserts it equals the declared count after correction.

Run from the repo root: ``PYTHONPATH=src python tools/build_feynman_bonus.py``
"""
from __future__ import annotations

import csv
import os
import re
import warnings

import numpy as np
import yaml

from symbolic_data._evaluation import compile_expression, load_engine

HERE = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(HERE, "..", "assets", "upstream", "feynman_bonus", "BonusEquations.csv")
OUT = os.path.join(HERE, "..", "assets", "catalogs", "feynman-bonus.yaml")

SOURCES = [
    "Feynman Symbolic Regression Database BONUS set (Udrescu & Tegmark 2020, Science Advances "
    "6(16):eaay2631)",
    "BonusEquations.csv vendored from WassimTenachi/PhySO (MIT, commit bfbfa88) — the canonical "
    "surviving copy (the original space.mit.edu host is dead); see assets/upstream/feynman_bonus/NOTICE",
]
COUNT_FIX = {"test_12": 5, "test_13": 5, "test_18": 5, "test_19": 6}
ARC = {"arcsin": "asin", "arccos": "acos", "arctan": "atan"}


def main() -> None:
    engine = load_engine("dev_7-3")
    with open(CSV_PATH, encoding="utf-8-sig") as handle:
        rows = [r for r in csv.DictReader(handle) if r.get("Filename")]
    assert len(rows) == 20, len(rows)

    expressions = {}
    for row in rows:
        filename = row["Filename"]
        number = int(row["Number"])
        declared = int(row["# variables"])
        expected = COUNT_FIX.get(filename, declared)

        names, ranges = [], []
        for i in range(1, 11):
            name = (row.get(f"v{i}_name") or "").strip()
            if not name:
                break
            names.append(name)
            ranges.append((float(row[f"v{i}_low"]), float(row[f"v{i}_high"])))
        assert len(names) == expected, (filename, declared, expected, names)

        raw = row["Formula"].strip()
        # symbolic names -> v-tokens (longest-first, word-bounded: 'c' must not hit 'cos')
        prepared = raw
        for j, name in sorted(enumerate(names), key=lambda t: -len(t[1])):
            prepared = re.sub(rf"\b{re.escape(name)}\b", f"v{j + 1}", prepared)
        for arc, short in ARC.items():
            prepared = prepared.replace(arc, short)
        prepared = re.sub(r"\bpi\b", "3.141592653589793", prepared).replace(" ", "")
        leftover = set(re.findall(r"[A-Za-z_][A-Za-z_0-9]*", prepared)) - \
            {"sin", "cos", "tan", "asin", "acos", "atan", "exp", "log", "sqrt", "abs"} - \
            {f"v{k}" for k in range(1, expected + 1)}
        assert not leftover, (filename, leftover)

        # numeric sanity: compiles + finite on midpoint-of-box samples
        vars_info = {f"v{j + 1}": {"name": names[j]} for j in range(expected)}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            compiled = compile_expression(engine, filename, prepared, vars_info)
            rng = np.random.default_rng(0)
            cols = [rng.uniform(lo, hi, 64) for lo, hi in ranges]
            y = np.asarray(compiled["callable"](*cols), dtype=np.float64)
        assert np.isfinite(y).mean() > 0.95, (filename, float(np.isfinite(y).mean()))

        eq_id = f"B{number}"
        expressions[eq_id] = {
            "raw": raw,
            "prepared": prepared,
            "n_variables": expected,
            "sources": SOURCES,
            "vars": {
                **{f"v{j + 1}": {"name": names[j],
                                 "sample_range": [ranges[j][0], ranges[j][1]],
                                 "sample_type": ["uni", "pos"]} for j in range(expected)},
                "v0": {"name": row["Output"].strip() or "y"},
            },
            "meta": {
                "bonus_filename": filename,
                "bonus_name": row["Name"].strip(),
                "base_catalog": "fastsrb@1", "base_eq_id": eq_id,
                **({"variables_count_corrected": f"{declared}->{expected}"}
                   if filename in COUNT_FIX else {}),
            },
        }
        print(f"{eq_id:4s} {row['Name'].strip():40s} d={expected} finite-ok")

    doc = {
        "metadata": {
            "name": "feynman-bonus",
            "version": 1,
            "description": "AI-Feynman BONUS set (20 named physics results), aifeynman-original "
                           "rendering — the 3rd rendering of the bonus set alongside fastsrb "
                           "B1-B20 (SRSD) and srsd-dummy B1-B20, mirroring the main set's "
                           "variant structure.",
            "sources": SOURCES,
            "license": "specs regenerated from the equation database (formulas + ranges = cited "
                       "derived facts; original data files unlicensed and NOT redistributed); "
                       "csv copy via PhySO (MIT)",
            "sampling_defaults": {"n_points": 100, "method": "random", "noise": 0.0},
            "source_kind": "set",
            "conventions": {
                "sampling": "per-variable uniform in the original integer-endpoint boxes; "
                            "singularity-free (verified); per-point rejection conditioning as "
                            "everywhere.",
                "errata": "the csv's '# variables' column undercounts test_12/13/18/19; corrected "
                          "+ asserted (PhySO's documented fix).",
                "variants": "1:1 base_eq_id links to fastsrb B<N>; renderings join the union "
                            "holdout independently. ERBench's Feynman collection drops bonus_10 "
                            "(arccos) — this catalog keeps the arccos original.",
            },
        },
        "expressions": expressions,
    }
    with open(OUT, "w", encoding="utf-8") as handle:
        yaml.safe_dump(doc, handle, sort_keys=False, allow_unicode=True, width=110)
    print(f"20 entries -> {OUT}")


if __name__ == "__main__":
    main()
