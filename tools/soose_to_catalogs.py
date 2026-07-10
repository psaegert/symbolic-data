"""Build the soose-nc / soose-wc / soose-fc catalogs (SOOSE, NeSymReS ICML 2021, MIT).

SOOSE ("Strictly Out-Of-Sample Equations"): 200 skeletons sampled from the NeSymReS
pre-training distribution and then EXCLUDED from it, instantiated three ways — NC (no
constants), WC (0-3 constant slots sampled), FC (every slot filled). Three sibling
declarative catalogs because the constants are baked into the infix strings (each variant is
a complete standalone GT list) and the paper reports the variants separately.

Sources (assets/upstream/soose/, see NOTICE): nc.csv = test_set/nc.csv @ HEAD 09f5af4;
WC/FC recovered from the DELETED data/benchmark/old_test.csv @ 0cfff79 (rows 200-399 = FC,
rows 400-599 = WC; block identity inferred from constant density — FC by definition has no
constant-free rows, and the file has exactly the NC block byte-equal to nc.csv first).

Row i shares its skeleton across all three variants: WC/FC entries carry
meta.base_catalog/base_eq_id links to the NC row, and the dedup registry clusters the
constant-masked skeletons across the three catalogs (union holdout dedups; suites stay
citable separately).

Run from the repo root: ``PYTHONPATH=src python tools/soose_to_catalogs.py``
"""
from __future__ import annotations

import ast
import csv
import os
import re

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
UPSTREAM = os.path.join(HERE, "..", "assets", "upstream", "soose")
OUT_DIR = os.path.join(HERE, "..", "assets", "catalogs")

SOURCES = [
    "Biggio et al. 2021 (arXiv:2106.06427), Neural Symbolic Regression that Scales (NeSymReS) "
    "— SOOSE test set",
    "SymposiumOrganization/NeuralSymbolicRegressionThatScales (MIT); NC: test_set/nc.csv @ "
    "09f5af4; WC/FC: data/benchmark/old_test.csv @ 0cfff79 (recovered from deleted history, "
    "see assets/upstream/soose/NOTICE)",
]

VARIANT_NOTES = {
    "nc": "no constants: all multiplicative placeholders = 1, additive = 0",
    "wc": "with constants: upstream sampled 0-3 of the placeholder slots (repo config: additive "
          "U(-2,2), multiplicative U(0.1,5)), then sympy.simplify — 47/200 rows end "
          "constant-free (kept as upstream ships them); literals are the FROZEN historical "
          "instantiation (unseeded generator, not re-renderable)",
    "fc": "full constants: EVERY placeholder slot sampled ('one constant per term'), then "
          "sympy.simplify (which compounds literals beyond the sampling ranges); 2-6 literals "
          "per expression, no constant-free rows; frozen historical instantiation",
}


def read_rows(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def to_prepared(eq: str, var_names: list[str]) -> tuple[str, int]:
    """x_k -> v<j> with CONTIGUOUS v-indices in x_k order; returns (prepared, n_variables)."""
    mapping = {name: f"v{j + 1}" for j, name in enumerate(var_names)}
    prepared = re.sub(r"x_(\d+)", lambda m: mapping[f"x_{m.group(1)}"], eq)
    assert "x_" not in prepared, eq
    return prepared.replace(" ", ""), len(var_names)


def build_variant(variant: str, rows: list[dict], nc_ids: list[str] | None) -> tuple[str, list[str]]:
    """Emit one catalog yaml; returns (path, eq_ids)."""
    expressions = {}
    eq_ids = []
    for i, row in enumerate(rows):
        support = ast.literal_eval(row["support"])
        var_names = sorted(support.keys(), key=lambda k: int(k.split("_")[1]))
        # contiguity check: SOOSE supports are x_1..x_n without gaps (assert, don't assume)
        assert var_names == [f"x_{j + 1}" for j in range(len(var_names))], (variant, i, var_names)
        prepared, n_vars = to_prepared(row["eq"], var_names)
        assert int(row["num_points"]) == 500, (variant, i)
        for name in var_names:
            assert support[name] == {"max": 10, "min": -10}, (variant, i, support)

        eq_id = f"SOOSE-{variant.upper()}-{i:03d}"
        entry = {
            "raw": row["eq"],
            "prepared": prepared,
            "n_variables": n_vars,
            "sources": SOURCES,
            "vars": {
                **{f"v{j + 1}": {"name": var_names[j], "sample_range": [-10.0, 10.0],
                                 "sample_type": ["uni", "pos"]} for j in range(n_vars)},
                "v0": {"name": "y"},
            },
        }
        if nc_ids is not None:
            entry["meta"] = {"base_catalog": "soose-nc@1", "base_eq_id": nc_ids[i],
                             "upstream_row": int(row[""])}
        expressions[eq_id] = entry
        eq_ids.append(eq_id)

    doc = {
        "metadata": {
            "name": f"soose-{variant}",
            "version": 1,
            "description": f"SOOSE-{variant.upper()}: 200 out-of-training-sample NeSymReS "
                           f"skeletons, {VARIANT_NOTES[variant]}.",
            "sources": SOURCES,
            "license": "MIT (SymposiumOrganization/NeuralSymbolicRegressionThatScales)",
            "sampling_defaults": {"n_points": 500, "method": "random", "noise": 0.0},
            "source_kind": "set",
            "conventions": {
                "sampling": "all variables uniform on [-10, 10] (upstream support), 500 points; "
                            "domain-restricted ops (sqrt/log/asin over the full box) follow the "
                            "per-point rejection convention — accepted points are the declared "
                            "distribution conditioned on the valid domain; meta.finite_fraction "
                            "(audit tool) discloses the per-entry valid fraction.",
                "variants": "row i shares its skeleton across soose-nc/wc/fc; WC/FC carry "
                            "meta.base_eq_id links to the NC row; block identity of the "
                            "historical WC/FC source is inferred from constant density (NOTICE).",
            },
        },
        "expressions": expressions,
    }
    path = os.path.join(OUT_DIR, f"soose-{variant}.yaml")
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(doc, handle, sort_keys=False, allow_unicode=True, width=100)
    print(f"soose-{variant}: {len(expressions)} entries -> {path}")
    return path, eq_ids


def main() -> None:
    nc_rows = read_rows(os.path.join(UPSTREAM, "nc.csv"))
    old_rows = read_rows(os.path.join(UPSTREAM, "old_test.csv"))
    assert len(nc_rows) == 200 and len(old_rows) == 600
    # block layout of the recovered file: 0-199 NC (byte-equal to nc.csv), 200-399 FC, 400-599 WC
    assert [r["eq"] for r in old_rows[:200]] == [r["eq"] for r in nc_rows], "NC block mismatch"
    fc_rows, wc_rows = old_rows[200:400], old_rows[400:600]
    # sanity of the inferred block identity: FC has no constant-free rows, WC has some
    float_re = re.compile(r"\d+\.\d+")
    assert all(float_re.search(r["eq"]) for r in fc_rows), "FC block has constant-free rows?!"
    assert sum(1 for r in wc_rows if not float_re.search(r["eq"])) == 47, "WC constant-free count"

    _, nc_ids = build_variant("nc", nc_rows, None)
    build_variant("wc", wc_rows, nc_ids)
    build_variant("fc", fc_rows, nc_ids)


if __name__ == "__main__":
    main()
