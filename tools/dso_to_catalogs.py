"""Convert the vendored DSO benchmarks.csv into per-suite declarative catalogs (P1 of the
benchmark-import program).

One catalog per ORIGINAL suite (source separation); entries carry raw (upstream form),
prepared (v-normalized infix the realize path compiles), per-variable sampling from the DSO
train_spec, and provenance lines. Excluded rows (non-elementary functions) are logged, never
silently dropped. The published `nguyen@1` catalog is left untouched (forward-only); the
Nguyen rows here are recorded only in the dedup report.

Run from the repo root: python tools/dso_to_catalogs.py
"""
from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / "assets" / "upstream" / "dso_benchmarks.csv"
OUT = ROOT / "assets" / "catalogs"
REPORT = ROOT / "assets" / "upstream" / "dso_conversion_report.json"

DSO_CITE = "formulas + ranges from dso-org/deep-symbolic-optimization benchmarks.csv (BSD-3; Petersen et al. 2021, Mundhenk et al. 2021)"

# suite key -> (catalog name, description, origin citation)
SUITES = {
    "Keijzer": ("keijzer", "Keijzer symbolic-regression suite", "Keijzer 2003 (EuroGP), interval arithmetic + linear scaling"),
    "Korns": ("korns", "Korns accuracy suite (5-variable, constant-heavy)", "Korns 2011 (GPTP IX), Accuracy in Symbolic Regression"),
    "Vladislavleva": ("vladislavleva", "Vladislavleva order-of-nonlinearity suite", "Vladislavleva, Smits, den Hertog 2009 (IEEE TEC)"),
    "Jin": ("jin", "Jin Bayesian-SR suite", "Jin et al. 2019 (arXiv:1910.08892), Bayesian Symbolic Regression"),
    "Neat": ("neat", "neat-GP suite (aggregated classics)", "Trujillo et al. 2016 (Information Sciences), neat Genetic Programming"),
    "R": ("r-rationals", "Krawiec-Pawlak rational-function suite (R1-R3 + wide-range a variants)", "Krawiec & Pawlak 2013 (GECCO), semantic backpropagation"),
    "Livermore": ("livermore", "Livermore suite (LLNL/DSR)", "Petersen et al. 2021 (ICLR) + Mundhenk et al. 2021 (NeurIPS), LLNL"),
    "Livermore2": ("livermore2", "Livermore2 suite (Vars2-7 groups, 25 each)", "Mundhenk et al. 2021 (NeurIPS), LLNL neural-guided GP seeding"),
    "Pagie": ("pagie", "Pagie coevolution benchmark", "Pagie & Hogeweg 1997 (Evolutionary Computation)"),
    "Meier": ("meier", "Meier suite", "Meier et al. (via GP benchmark surveys)"),
    "Poly": ("poly", "Poly suite incl. Poly-10", "Poli 2003 lineage (via DSO aggregation)"),
    "Nonic": ("nonic", "Nonic polynomial benchmark", "GP benchmark surveys (McDermott et al. 2012 lineage)"),
    "Sine": ("sine", "Sine composite benchmark", "GP benchmark surveys (McDermott et al. 2012 lineage)"),
    "Constant": ("constant", "Constant-optimization suite (incl. Const-Test sanity checks)", "Petersen et al. 2021 (ICLR), Deep Symbolic Regression, Appendix"),
    "Const-Test": ("constant", None, None),          # merged into `constant`
    "Koza": ("koza", "Koza classic GP problems (quintic/sextic)", "Koza 1992/1994 (MIT Press), Genetic Programming I/II"),
    "GrammarVAE": ("grammarvae", "GrammarVAE equation benchmark", "Kusner et al. 2017 (ICML), Grammar Variational Autoencoder"),
    "Nguyen": (None, None, None),                    # published as nguyen@1 already; dedup-log only
}

PI = repr(math.pi)


def suite_of(name: str) -> str:
    if name.startswith("Livermore2"):
        return "Livermore2"
    if re.match(r"^R\d", name):
        return "R"
    if name.startswith("Const-Test"):
        return "Const-Test"
    return name.split("-")[0]


def translate(expr: str) -> str:
    """DSO python-ish infix -> prepared v-infix the catalog realize path compiles."""
    out = expr

    def rewrite_call(source: str, fname: str, template) -> str:
        # rewrite fname(a, b) / fname(a) with balanced-paren argument splitting
        while True:
            m = re.search(rf"\b{fname}\(", source)
            if not m:
                return source
            depth, i = 1, m.end()
            args, start = [], m.end()
            while depth:
                c = source[i]
                if c == "(":
                    depth += 1
                elif c == ")":
                    depth -= 1
                    if depth == 0:
                        args.append(source[start:i])
                elif c == "," and depth == 1:
                    args.append(source[start:i])
                    start = i + 1
                i += 1
            source = source[:m.start()] + template(*[a.strip() for a in args]) + source[i:]

    out = rewrite_call(out, "div", lambda a, b: f"(({a})/({b}))")
    out = rewrite_call(out, "pow", lambda a, b: f"(({a})**({b}))")
    out = re.sub(r"(?<![\w.])pi(?![\w(])", PI, out)
    out = re.sub(r"\bx(\d+)\b", r"v\1", out)
    return out


def spec_vars(row: dict) -> dict:
    spec = json.loads(row["train_spec"])
    n = int(row["variables"])
    out = {}
    for i in range(1, n + 1):
        var_spec = spec.get(f"x{i}", spec.get("all"))
        kind, params = next(iter(var_spec.items()))
        lo, hi, third = params
        base = "int" if (kind == "E" and float(third) == 1.0 and float(lo) == int(lo)) else "uni"
        out[f"v{i}"] = {
            "name": f"x{i}",
            "sample_range": [float(lo), float(hi)],
            "sample_type": [base, "pos"],
        }
    out["v0"] = {"name": "y"}
    return out


def main() -> None:
    rows = [r for r in csv.DictReader(CSV.open()) if r.get("name")]
    catalogs: dict[str, dict] = {}
    report = {"excluded": [], "nguyen_rows_logged": [], "counts": {}}

    for row in rows:
        suite = suite_of(row["name"])
        if suite not in SUITES:
            report["excluded"].append({"name": row["name"], "reason": f"unknown suite {suite}"})
            continue
        cat_name = SUITES[suite][0]
        if cat_name is None:                                    # Nguyen: published already
            report["nguyen_rows_logged"].append(row["name"])
            continue
        if "harmonic(" in row["expression"]:
            report["excluded"].append({"name": row["name"],
                                       "reason": "harmonic() sum is not an elementary closed form in the shared grammar"})
            continue
        cat = catalogs.setdefault(cat_name, {
            "metadata": {
                "name": cat_name,
                "version": 1,
                "description": None,        # filled below
                "sources": [],
                "sampling_defaults": {"n_points": 20, "method": "random", "noise": 0.0},
                "source_kind": "set",
                "conventions": {
                    "sampling": "vars use the `fastsrb` distribution (sample_range/sample_type); "
                                "DSO train_spec ranges; evenly-spaced (E) upstream grids are sampled "
                                "uniformly here (noted per entry as meta.dso_spec)."},
            },
            "expressions": {},
        })
        first_suite_row = not cat["expressions"]
        if first_suite_row and SUITES[suite][1]:
            cat["metadata"]["description"] = f"{SUITES[suite][1]}."
            cat["metadata"]["sources"] = [SUITES[suite][2], DSO_CITE]
        entry = {
            "raw": row["expression"],
            "prepared": translate(row["expression"]),
            "n_variables": int(row["variables"]),
            "sources": [SUITES[suite][2] or SUITES[suite_of(row['name'])][2], DSO_CITE],
            "vars": spec_vars(row),
            "meta": {"dso_spec": {"train": json.loads(row["train_spec"]),
                                  "test": (json.loads(row["test_spec"])
                                           if row["test_spec"].strip() not in ("", "None") else None),
                                  "function_set": row["function_set"]}},
        }
        cat["expressions"][row["name"]] = entry

    OUT.mkdir(parents=True, exist_ok=True)
    for cat_name, cat in sorted(catalogs.items()):
        path = OUT / f"{cat_name}.yaml"
        path.write_text(yaml.safe_dump(cat, sort_keys=False, allow_unicode=True), encoding="utf-8")
        report["counts"][cat_name] = len(cat["expressions"])
        print(f"wrote {path.name}: {len(cat['expressions'])} entries")
    REPORT.write_text(json.dumps(report, indent=1))
    print("report:", {k: v for k, v in report["counts"].items()},
          "| excluded:", len(report["excluded"]), "| nguyen logged:", len(report["nguyen_rows_logged"]))


if __name__ == "__main__":
    main()
