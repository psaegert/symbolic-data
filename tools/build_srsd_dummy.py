"""P2: build the `srsd-dummy` catalog — the SRSD dummy-variable Feynman variant.

Base = the 120 SRSD-lineage problems as canonicalized in the published `fastsrb` catalog
(same skeletons + physically realistic ranges). Dummy recipe replicated from SRSD's
dummy_column_mixer.py (omron-sinicx/srsd-benchmark, MIT): per problem 1-3 dummy columns,
each sampled log-uniform in [10^(k-1), 10^(k+1)] with k ~ randint(-32, 32), sign-symmetric
with p=0.5, inserted at random positions among the original variables. The dummy
CONFIGURATION is fixed here with an authoring seed (the catalog is a deterministic
declaration); data sampling at eval time remains unseeded per house doctrine.

ORACLE WARNING (WP7 distractor-parity): adapters must NOT receive the used-variable oracle
on this catalog (e.g. PySR `padding: false` masks unused variables -> defeats the benchmark).
Eval configs for srsd-dummy must disable variable masking for every method.


NOTE (2026-07-10 audit): the COMMITTED catalog yamls are this builder's output PLUS a
subsequent tools/audit_finite_fraction.py pass (which writes per-entry meta.finite_fraction
and the conventions.validity block). Re-running this builder alone therefore does not
byte-reproduce the committed files -- run the audit tool afterwards to reproduce them.
"""
from __future__ import annotations

import random
import re
from pathlib import Path

import yaml

SEED = 20260710
SRC = Path("assets/catalogs/fastsrb.yaml")
OUT = Path("assets/catalogs/srsd-dummy.yaml")

SRSD_CITE = ("dummy recipe from omron-sinicx/srsd-benchmark dummy_column_mixer.py (MIT; "
             "Matsubara et al. 2024, 'Rethinking Symbolic Regression Datasets and Benchmarks "
             "for Scientific Discovery', DMLR)")


def main() -> None:
    rng = random.Random(SEED)
    base = yaml.safe_load(SRC.read_text())
    out_exprs = {}
    for eq_id, entry in base["expressions"].items():
        real_keys = [k for k in entry["vars"] if k != "v0"]
        n_org = len(real_keys)
        n_dummy = rng.choice([1, 2, 3])
        positions = sorted((rng.randint(0, n_org) for _ in range(n_dummy)), reverse=True)

        # ordered real variable specs (v1..vn), then insert dummies at sampled positions.
        # Sanitize inherited sample_range: fastsrb@1's III.21.20 carries a stray
        # 'uses_positive=False' string as a third element (upstream import leak; fastsrb@2 TODO).
        ordered = []
        for i in range(1, n_org + 1):
            spec = dict(entry["vars"][f"v{i}"])
            spec["sample_range"] = [float(spec["sample_range"][0]), float(spec["sample_range"][1])]
            ordered.append(spec)
        for j, pos in enumerate(positions):
            k = rng.randint(-32, 32)
            uses_negative = rng.random() < 0.5
            ordered.insert(pos, {
                "name": f"dummy{n_dummy - j}",
                "sample_range": [float(10.0 ** (k - 1)), float(10.0 ** (k + 1))],
                "sample_type": ["log", "pos_neg" if uses_negative else "pos"],
            })

        # index mapping old v_i -> new position; rewrite prepared accordingly
        mapping = {}
        new_vars = {}
        real_seen = 0
        for new_idx, spec in enumerate(ordered, start=1):
            new_vars[f"v{new_idx}"] = spec
            if not str(spec["name"]).startswith("dummy"):
                real_seen += 1
                mapping[f"v{real_seen}"] = f"v{new_idx}"
        new_vars["v0"] = dict(entry["vars"].get("v0", {"name": "y"}))
        prepared = re.sub(r"\bv(\d+)\b", lambda m: mapping.get(m.group(0), m.group(0)) + "\x00",
                          entry["prepared"]).replace("\x00", "")

        out_entry = {
            "raw": entry["raw"],
            "prepared": prepared,
            "n_variables": n_org + n_dummy,
            "sources": list(entry.get("sources", [])) + [SRSD_CITE],
            "vars": new_vars,
        }
        if "accept" in entry:
            out_entry["accept_note"] = ("acceptable forms tracked on the fastsrb rendering; "
                                        "variable indices differ here (dummy insertions)")
        out_entry["meta"] = {"dummy_seed": SEED, "n_dummy": n_dummy,
                             "base_catalog": "fastsrb@1", "base_eq_id": eq_id}
        out_exprs[eq_id] = out_entry

    catalog = {
        "metadata": {
            "name": "srsd-dummy",
            "version": 1,
            "description": ("SRSD Feynman dummy-variable variant (120): the fastsrb/SRSD-lineage "
                            "problems with 1-3 irrelevant variables inserted at random positions "
                            "(feature-selection axis)."),
            "sources": [
                "Feynman Symbolic Regression Database (Udrescu & Tegmark 2020, AI Feynman)",
                "SRSD realistic ranges (Matsubara et al. 2024, CC BY 4.0 datasets / MIT code)",
                SRSD_CITE,
                f"dummy configuration fixed with authoring seed {SEED} (deterministic catalog)",
            ],
            "sampling_defaults": {"n_points": 100, "method": "random", "noise": 0.0},
            "source_kind": "set",
            "conventions": {
                "sampling": "vars use the `fastsrb` distribution (sample_range/sample_type)",
                "oracle_warning": ("adapters must NOT use a used-variable oracle here: variable "
                                   "masking (e.g. PySR padding:false) defeats the benchmark; "
                                   "eval configs must disable masking for every method"),
            },
        },
        "expressions": out_exprs,
    }
    OUT.write_text(yaml.safe_dump(catalog, sort_keys=False, allow_unicode=True), encoding="utf-8")
    n_d = [e["meta"]["n_dummy"] for e in out_exprs.values()]
    print(f"wrote {OUT.name}: {len(out_exprs)} entries; dummy counts 1/2/3 = "
          f"{n_d.count(1)}/{n_d.count(2)}/{n_d.count(3)}")


if __name__ == "__main__":
    main()
