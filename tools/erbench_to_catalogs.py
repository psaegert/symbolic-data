"""Build the five NOVEL ERBench family catalogs (P5): syneq, oeis, phybench, densities, eponymous.

Source: HuggingFace dataset EquationDiscovery/Equation_Recovery_Benchmark (ERBench, Kahlmeyer et
al., arXiv:2606.09276) — 10,000 declarative rows (expression + per-variable support + dtype; the
dataset ships NO fixed arrays). Only the five novel families become catalogs; the re-rendering
families (Feynman/SRDS/Nguyen/Keijzer/Korns/Livermore/Vladislavleva/Koza/Pagie) dedup to our
existing imports and Strogatz is skipped (ODE-derived + GPL-3.0).

LICENSE FIREWALL (verified against the OEIS EULA + CC BY-SA 4.0 legal code): OEIS and Eponymous
are CC-BY-SA-4.0 (OEIS-/Wikipedia-derived). Their catalogs + minimized vendored rows live under
``assets_sa/`` with their own LICENSE and publish to a SEPARATE HF repo — never folded into any
permissive bundle. The vendored SA csv drops the ``description`` column entirely (upstream rows
embed whole OEIS entry texts with contributor comments — far more SA content than the formulas).

Canonicalization (documented per entry where applied): '^' -> '**'; sympy 'Abs(' -> 'abs(';
two-arg log(u, 10) -> 0.4342944819032518*log(u); x_k -> v{k+1}. Entries whose expression still
fails to compile under dev_7-3 are EXCLUDED and logged in the build report (never silently).
Intra-family exact duplicates (same normalized expression + support + dtype) are dropped with
their names merged into the kept entry's meta.

Run from the repo root: ``PYTHONPATH=src python tools/erbench_to_catalogs.py``
"""
from __future__ import annotations

import ast
import csv
import os
import re
import warnings

import yaml

from symbolic_data._evaluation import compile_expression, load_engine

HERE = os.path.dirname(os.path.abspath(__file__))
TRAIN_CSV = os.path.join(HERE, "..", "assets", "upstream", "erbench", "train_permissive.csv")
TRAIN_SA_CSV = os.path.join(HERE, "..", "assets_sa", "upstream", "erbench", "train_sa.csv")
OUT_DIR = os.path.join(HERE, "..", "assets", "catalogs")
OUT_SA_DIR = os.path.join(HERE, "..", "assets_sa", "catalogs")
REPORT = os.path.join(HERE, "..", "assets", "catalogs", "ERBENCH_REPORT.md")

PAPER = "ERBench (Kahlmeyer, Voigt, Habeck & Giesen, arXiv:2606.09276)"
HF = "HuggingFace dataset EquationDiscovery/Equation_Recovery_Benchmark"

def _rewrite_two_arg_logs(expr: str) -> str:
    """sympy two-arg log(u, base) -> engine-parseable natural-log forms (paren-balanced)."""
    out = []
    i = 0
    while True:
        j = expr.find("log(", i)
        if j < 0:
            out.append(expr[i:])
            break
        out.append(expr[i:j])
        depth, k, comma = 0, j + 3, -1
        while True:                       # balance parens from the 'log(' opener
            ch = expr[k]
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    break
            elif ch == "," and depth == 1:
                comma = k
            k += 1
        if comma < 0:
            out.append(expr[j:k + 1])
        else:
            arg = _rewrite_two_arg_logs(expr[j + 4:comma].strip())
            base = expr[comma + 1:k].strip()
            if base == "E":
                out.append(f"log({arg})")
            elif base == "10":
                out.append(f"0.4342944819032518*log({arg})")
            else:
                out.append(f"log({arg})/log({_rewrite_two_arg_logs(base)})")
        i = k + 1
    return "".join(out)


def normalize_expression(expr: str) -> str:
    expr = expr.replace("^", "**").replace("Abs(", "abs(")
    return _rewrite_two_arg_logs(expr)


def substitute_pi(prepared: str) -> str:
    # house convention (cf. feynman.yaml): `raw` keeps symbolic constants, `prepared` carries
    # numeric literals -- the engine's evaluation namespace defines neither `pi` nor `E`
    prepared = re.sub(r"\bpi\b", "3.141592653589793", prepared)
    return re.sub(r"\bE\b", "2.718281828459045", prepared)


def to_v_space(expr: str) -> tuple[str, list[int]]:
    used = sorted(set(int(m) for m in re.findall(r"x_(\d+)", expr)))
    prepared = re.sub(r"x_(\d+)", lambda m: f"v{int(m.group(1)) + 1}", expr)
    return prepared.replace(" ", ""), used


FAMILY = {
    # name, output dir, license line, eval tier, id_fn
    "SynEq": ("erbench-syneq", OUT_DIR, "MIT (ERBench author-generated)", "extended"),
    "PHYBench": ("erbench-phybench", OUT_DIR,
                 "MIT (ERBench paper table + upstream Eureka-Lab/PHYBench)", "headline"),
    "Densities": ("erbench-densities", OUT_DIR, "BSD-3-Clause (SciPy-derived, per ERBench README)",
                  "headline"),
    "OEIS": ("erbench-oeis", OUT_SA_DIR,
             "CC-BY-SA-4.0 (OEIS-derived; see assets_sa LICENSE + per-entry attribution)",
             "extended"),
    "Eponymous": ("erbench-eponymous", OUT_SA_DIR,
                  "CC-BY-SA-4.0 (Wikipedia-derived; see assets_sa LICENSE)", "headline"),
}


def entry_meta(fam: str, row: dict, aliases: list[str], degenerate: list[int]) -> dict:
    meta: dict = {"erbench_task_id": row["task_ID"]}
    if fam == "OEIS":
        meta.update({"oeis_id": row["name"], "source_url": f"https://oeis.org/{row['name']}",
                     "attribution": "The On-Line Encyclopedia of Integer Sequences (OEIS), "
                                    "The OEIS Foundation Inc., CC-BY-SA-4.0"})
    elif fam == "Eponymous":
        meta.update({"equation_name": row["name"],
                     "attribution": "Wikipedia, 'List of scientific equations named after people' "
                                    "(CC-BY-SA-4.0); intervals LLM-suggested (mistral-large), "
                                    "author cross-verified per the ERBench paper"})
    elif fam == "Densities":
        meta.update({"distribution": row["name"], "scipy_doc": row.get("description", "")})
    elif fam == "PHYBench":
        meta.update({"upstream_name": row["name"], "problem": (row.get("description") or "")[:400]})
    elif fam == "SynEq":
        meta.update({"complexity": row.get("description", "").replace(
            "Unbiased expression of complexity ", "")})
    if aliases:
        meta["duplicate_of_rows"] = aliases          # merged exact-duplicate names/task_ids
    if degenerate:
        meta["near_degenerate_intervals"] = degenerate   # v-indices with width <= 1e-3*scale
    return meta


def main() -> None:
    engine = load_engine("dev_7-3")
    rows = []
    for path in (TRAIN_CSV, TRAIN_SA_CSV):
        with open(path, encoding="utf-8") as handle:
            rows.extend(csv.DictReader(handle))

    report = ["# ERBench import report", "",
              f"{PAPER}; {HF}. Only the five NOVEL families become catalogs; re-renderings dedup "
              "to existing imports (see BENCHMARK_IMPORT_PROGRAM.md); Strogatz skipped (ODE + "
              "GPL-3.0).", ""]
    for fam, (cat_name, out_dir, license_line, tier) in FAMILY.items():
        frows = [r for r in rows if r["collection"] == fam]
        expressions: dict = {}
        seen: dict[tuple, str] = {}
        excluded, n_dups = [], 0

        for row in frows:
            norm = normalize_expression(row["expression"])
            prepared, used = to_v_space(norm)
            prepared = substitute_pi(prepared)
            support = ast.literal_eval(row["support"])
            assert len(support) == len(used) and used == list(range(len(used))), \
                (fam, row["task_ID"], used, len(support))

            key = (prepared, str(support), row["dtype"])
            if key in seen:
                n_dups += 1
                prev = expressions[seen[key]]
                prev.setdefault("meta", {}).setdefault("duplicate_of_rows", []).append(
                    f"{row['task_ID']}:{row['name']}")
                continue

            vars_info = {f"v{i + 1}": {} for i in range(len(used))}
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    compile_expression(engine, row["task_ID"], prepared, vars_info)
            except Exception as exc:  # noqa: BLE001 - excluded loudly, never silently
                excluded.append((row["task_ID"], row["expression"][:70], str(exc)[:60]))
                continue

            sample_type = ["int", "pos"] if row["dtype"] == "int" else ["uni", "pos"]
            degenerate = [i + 1 for i, (lo, hi) in enumerate(support)
                          if (float(hi) - float(lo)) <= 1e-3 * max(abs(float(lo)), abs(float(hi)), 1.0)]
            eq_id = {"OEIS": row["name"], "Densities": row["name"].replace(" density", ""),
                     }.get(fam, row["task_ID"] if fam != "Eponymous"
                           else f"{row['task_ID']}")
            if eq_id in expressions:       # e.g. duplicated A-number with DIFFERENT support
                eq_id = f"{eq_id}~{row['task_ID']}"
            entry = {
                "raw": row["expression"],
                "prepared": prepared,
                "n_variables": len(used),
                "sources": [PAPER, HF],
                "vars": {
                    **{f"v{i + 1}": {"name": f"x_{i}",
                                     "sample_range": [float(support[i][0]), float(support[i][1])],
                                     "sample_type": sample_type} for i in range(len(used))},
                    "v0": {"name": "y"},
                },
                "meta": entry_meta(fam, row, [], degenerate),
            }
            if not entry["meta"].get("near_degenerate_intervals"):
                entry["meta"].pop("near_degenerate_intervals", None)
            expressions[eq_id] = entry
            seen[key] = eq_id

        doc = {
            "metadata": {
                "name": cat_name,
                "version": 1,
                "description": f"ERBench {fam} family ({len(expressions)} entries after "
                               f"deduplication; {tier} eval tier).",
                "sources": [PAPER, HF],
                "license": license_line,
                "eval_tier": tier,
                "sampling_defaults": {"n_points": 100, "method": "random", "noise": 0.0},
                "source_kind": "set",
                "conventions": {
                    "sampling": "per-variable uniform (or uniform-integer for dtype=int rows) in "
                                "the upstream support box; per-point rejection conditioning + "
                                "meta.finite_fraction disclosure as everywhere. ERBench's own "
                                "REFERENCE sampler (100 pts/dim, 1-3-component mixtures in a "
                                "half-width train sub-box, extrapolative eval region, noise w.p. "
                                "0.2) is an EVALUATION PROTOCOL, recorded here as provenance -- "
                                "our harness samples the declared box directly.",
                    "canonicalization": "'^'->'**', 'Abs('->'abs(', log(u,10)->log10-rewrite, "
                                        "x_k->v{k+1}; exact duplicates (expression+support+dtype) "
                                        "merged with names in meta.duplicate_of_rows.",
                },
            },
            "expressions": expressions,
        }
        path = os.path.join(out_dir, f"{cat_name}.yaml")
        with open(path, "w", encoding="utf-8") as handle:
            yaml.safe_dump(doc, handle, sort_keys=False, allow_unicode=True, width=110)
        sa = " [SA]" if out_dir == OUT_SA_DIR else ""
        print(f"{cat_name:22s} rows={len(frows):5d} entries={len(expressions):5d} "
              f"dups={n_dups:3d} excluded={len(excluded):3d}{sa}")
        report.append(f"## {cat_name}{sa}")
        report.append(f"- rows {len(frows)} -> entries {len(expressions)} "
                      f"(dups merged {n_dups}, excluded {len(excluded)}); tier {tier}; {license_line}")
        for tid, expr, err in excluded:
            report.append(f"  - EXCLUDED {tid}: `{expr}` ({err})")
        report.append("")

    with open(REPORT, "w", encoding="utf-8") as handle:
        handle.write("\n".join(report) + "\n")
    print(f"report -> {REPORT}")


if __name__ == "__main__":
    main()
