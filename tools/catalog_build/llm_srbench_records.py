"""LLM-SRBench's records for the data builder, made from the authors' release (owner 2026-11-09: "LLM-SRBench*").

srbf runs LLM-SRBench as its authors published it: the frozen arrays of the Hugging Face dataset nnheui/llm-srbench
(revision REVISION; lsr_bench_data.hdf5, one group per problem with ``train``, ``test`` and, for LSR-Synth,
``ood_test``; column 0 the target, then the inputs in the order of the problem's symbols), all 240 problems under
their names in the release, each LSR-Synth out-of-domain test a problem of its own. The ground truths come from the
release's parquet files, with the repairs srbf makes (marked in tags/deviations.json):

* the function notation of the time series is dropped (``A(t)`` is the input ``A``), and the exponent 1/3, which
  the authors' printing cut to 0.333333333333333 (31 problems), is written 1/3;
* the chemistry and physics formulas name constants without giving their values (80 problems: a chemistry rate
  constant ``k_z`` is printed as "<the value of k>_z", the physics constants omega0, beta, F0, gamma, mu, alpha are
  bare names). srbf fits them to each problem's training rows by least squares (fit()); every one is a coefficient,
  and the fitted formula reproduces the training, test and out-of-domain rows to float32 precision. omega0 enters
  only squared; srbf takes the positive root.

Every formula is checked against all of its problem's rows (FVU per array, kept in the record), and every array is
pinned by the sha256 of its bytes. Each problem states gt_kind 'exact' with its check over the training rows
(``gt_check``); the builder honours the stated kind only when that check puts every row within float32 input rounding. The script writes the set's record into data_sets.json.gz and its 240 data
problems into final_problems.json.gz (assets/upstream/catalog-prep); build_data_catalogs.py builds the catalog
``llm-srbench`` from them (reader ``llm_srbench`` in data_catalog_readers.py).

    llm_srbench_records.py [--write]     (the files are taken from SYMBOLIC_DATA_CATALOG_RAW/llm-srbench)
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths  # noqa: E402

TAG = "llm-srbench"
REVISION = "4f22d48100d94125d3af1e7bd40897fcc7f9597d"
CODE = "https://github.com/deep-symbolic-mathematics/llm-srbench/blob/6fdbe409b75ea749c0a93c69b6861637d804dec4"
HF = f"https://huggingface.co/datasets/nnheui/llm-srbench/resolve/{REVISION}"
HDF5 = ("lsr_bench_data.hdf5", "28280168829ab788ee730d6843595140388af6ef826bbccbd18bb3e8fae42f94", 239063948)
# the authors' domains: Hugging Face split, HDF5 group, srbf subset, sha256 of the split's parquet file
DOMAINS = [
    ("lsr_transform", "lsr_transform", "LSR-Transform",
     "2cf14d45d16c6fceb1109a93b0ffc6a342ab8cf65f7e91d83907422239fccc11"),
    ("lsr_synth_chem_react", "lsr_synth/chem_react", "LSR-Synth chemistry",
     "65631e562d5a52de33f80db87bf50d3144069bfe89aa63b3f93e879ea05ae28d"),
    ("lsr_synth_bio_pop_growth", "lsr_synth/bio_pop_growth", "LSR-Synth biology",
     "cad8399f889255fc4c2bc9c728e3d2c6064f988003dfd9a3bae9018d8a443883"),
    ("lsr_synth_phys_osc", "lsr_synth/phys_osc", "LSR-Synth physics",
     "b0618cc597dfd1e1f192ebbec15662e197bbbba1834c0f67d409863ff56b6655"),
    ("lsr_synth_matsci", "lsr_synth/matsci", "LSR-Synth materials science",
     "f97bbb0bf46d5a4bd1d3ede0304a159b5552498249f5292cfdc359b17b3ab7f2"),
]
FUNCTIONS = {"sin", "cos", "tan", "exp", "log", "sqrt", "Abs", "pi"}
F32_EPS = float(np.finfo(np.float32).eps)
STARTS = 40          # least-squares starts per problem: all ones, then normal(0, 2) draws from default_rng(0)
FLAT = 1e-3          # an out-of-domain target whose spread is below this share of the training spread is flat


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def array_sha256(a: np.ndarray) -> str:
    """The sha256 of an array's bytes in C order, as the reader checks them."""
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def spelled(expression: str, symbols: list[str]) -> tuple[str, list[str]]:
    """The formula over its inputs (``A(t)`` -> ``A``), with the exponent the authors' code computed as 1/3 written so
    (their printing cut it to 0.333333333333333), and each value-less constant as a name of its own (a chemistry
    "0.1899..._z" -> ``k_z``); and those names."""
    text = re.sub(r"\b([A-Za-z_]\w*)\(t\)", lambda m: m.group(1) if m.group(1) in symbols else m.group(0), expression)
    text = re.sub(r"(?<![\w.])0\.333333333333333(?![\d_])", "(1/3)", text)
    text = re.sub(r"\b\d+\.\d+_([A-Za-z]\w*)\b", lambda m: f"k_{m.group(1)}", text)
    unknown = sorted(set(re.findall(r"\b[A-Za-z_]\w*\b", text)) - FUNCTIONS - set(symbols))
    return text, unknown


def compiled(text: str, inputs: list[str], constants: list[str]):
    import sympy as sp
    names = {n: sp.Symbol(n) for n in [*inputs, *constants]}
    return sp.lambdify([names[n] for n in [*inputs, *constants]], sp.sympify(text, locals=names), modules="numpy")


def predict(f, a: np.ndarray, values) -> np.ndarray:
    with np.errstate(all="ignore"):
        return np.asarray(f(*[a[:, i] for i in range(1, a.shape[1])], *values), dtype=np.float64) * np.ones(len(a))


def fvu(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((y - p) ** 2) / np.var(y))


def unexplained(f, a: np.ndarray, values) -> int:
    """The rows whose target the formula misses by more than rounding explains. The authors computed each target
    before storing the row as float32, so a formula that generated the data may miss a stored row by what moving each
    input half a float32 ulp moves the formula, plus the target's own rounding (taken four times, for slack)."""
    y, p = a[:, 0], predict(f, a, values)
    bound = np.abs(y) * 2.0 ** -23
    for i in range(1, a.shape[1]):
        h = np.abs(a[:, i]) * 2.0 ** -24
        up, down = a.copy(), a.copy()
        up[:, i] += h
        down[:, i] -= h
        bound = bound + np.abs(predict(f, up, values) - predict(f, down, values)) / 2
    return int(np.sum(~(np.abs(p - y) <= 4 * bound)))


def fit(f, train: np.ndarray, n: int) -> np.ndarray:
    """The constants that fit the training rows best, by Levenberg-Marquardt from STARTS starts."""
    from scipy.optimize import least_squares
    y = train[:, 0]
    rng = np.random.default_rng(0)
    best = None
    for start in range(STARTS):
        c0 = np.ones(n) if start == 0 else rng.normal(0, 2, n)
        try:
            r = least_squares(lambda c: np.nan_to_num(predict(f, train, c) - y, nan=1e6, posinf=1e6, neginf=-1e6), c0,
                              method="lm", xtol=1e-15, ftol=1e-15, gtol=1e-15, max_nfev=20000)
        except (ValueError, FloatingPointError):
            continue
        if best is None or r.cost < best.cost:
            best = r
    return best.x


def with_values(text: str, values: dict[str, float]) -> str:
    return re.sub(r"\b(" + "|".join(map(re.escape, values)) + r")\b", lambda m: f"({values[m.group(1)]!r})", text) \
        if values else text


def problems(raw: Path):
    import h5py
    import pandas as pd
    h5 = h5py.File(raw / HDF5[0], "r")
    for split, group, subset, parquet_sha in DOMAINS:
        parquet = raw / f"{split}-00000-of-00001.parquet"
        if sha256_file(parquet) != parquet_sha:
            raise SystemExit(f"{parquet}: sha256 differs from the release")
        for _, row in pd.read_parquet(parquet).iterrows():
            symbols, props = list(row["symbols"]), list(row["symbol_properties"])
            if props[0] != "O" or any(p != "V" for p in props[1:]):
                raise SystemExit(f"{row['name']}: not one output followed by inputs")
            g = h5[f"{group}/{row['name']}"]
            arrays = {k: g[k][...] for k in ("train", "test", "ood_test") if k in g}
            yield split, group, subset, row, symbols, arrays


def record(raw: Path) -> tuple[dict, list[dict], dict]:
    datasets, finals, report = [], [], {"fitted": 0, "flat_ood": [], "exact_by_rounding": [], "worst_fvu": {}}
    members: dict[str, list[str]] = {d[2]: [] for d in DOMAINS}
    for split, group, subset, row, symbols, arrays in problems(raw):
        name, inputs = str(row["name"]), symbols[1:]
        text, unknown = spelled(row["expression"], symbols)
        f = compiled(text, inputs, unknown)
        values = {}
        if unknown:
            x = fit(f, arrays["train"].astype(np.float64), len(unknown))
            values = {n: float(abs(v)) if n == "omega0" else float(v) for n, v in zip(unknown, x)}
            report["fitted"] += 1
        checks = {}
        for k, a in arrays.items():
            a64 = a.astype(np.float64)
            checks[k] = {"fvu": fvu(a64[:, 0], predict(f, a64, list(values.values()))),
                         "rows_unexplained_by_rounding": unexplained(f, a64, list(values.values()))}
            report["worst_fvu"][k] = max(report["worst_fvu"].get(k, 0.0), checks[k]["fvu"])
        if checks["train"]["rows_unexplained_by_rounding"]:       # the builder honours 'exact' only with every row
            raise SystemExit(f"{name}: the formula misses its training rows ({checks['train']})")
        if checks["train"]["fvu"] > F32_EPS:
            report["exact_by_rounding"].append(name)
        dataset_id = f"{group}/{name}"
        members[subset].append(dataset_id)
        train = arrays["train"].astype(np.float64)
        flat = "ood_test" in arrays and np.std(arrays["ood_test"][:, 0]) < FLAT * np.std(train[:, 0])
        if flat:
            report["flat_ood"].append(name)
        datasets.append({
            "id": dataset_id, "name": name, "subset": subset, "split_name": split,
            "source": f"Hugging Face dataset nnheui/llm-srbench @ {REVISION} (gated, access granted on request; no "
                      f"license stated), {HDF5[0]}, group /{dataset_id}",
            "url": f"{HF}/{HDF5[0]}", "file": f"catalog-prep/raw/{TAG}/{HDF5[0]}", "sha256": HDF5[1], "bytes": HDF5[2],
            "group": f"/{dataset_id}",
            "arrays": {k: {"shape": list(a.shape), "dtype": str(a.dtype), "sha256": array_sha256(a)}
                       for k, a in arrays.items()},
            "target": symbols[0], "target_description": str(row["symbol_descs"][0]),
            "feature_names": inputs, "feature_descriptions": [str(d) for d in row["symbol_descs"][1:]],
            "split": ("given: the published arrays, train to fit, test the in-domain test"
                      + (", ood_test the out-of-domain test (each a problem of its own)" if "ood_test" in arrays else "")
                      + f" (the authors' harness: {CODE}/bench/dataclasses.py)"),
            "formula_as_published": str(row["expression"]),
            "formula": with_values(text, values),
            "constants_fitted": values,
            "formula_check": checks,
            **({"ood_target_flat": True} if flat else {}),
        })
        finals.append({
            "unit": f"{TAG}:{subset}", "set": TAG, "id": dataset_id, "kind": "D", "formula": with_values(text, values),
            "variables": [{"name": n, "description": str(d), "low": float(train[:, i + 1].min()),
                           "high": float(train[:, i + 1].max())}
                          for i, (n, d) in enumerate(zip(inputs, row["symbol_descs"][1:]))],
            "constants": values, "data_key": ["sha", HDF5[1], f"/{dataset_id}"], "formula_of": None,
            "gt_kind": "exact",
            "gt_check": {"rows": len(train), "fit_fvu": checks["train"]["fvu"],
                         "rows_within_input_rounding": len(train) - checks["train"]["rows_unexplained_by_rounding"]}})
    rec = {
        "tag": TAG, "name": "LLM-SRBench", "kind": "data", "status": "complete", "blocked_reason": None,
        "sources": ["https://arxiv.org/abs/2504.10415", f"https://huggingface.co/datasets/nnheui/llm-srbench/tree/{REVISION}",
                    f"{HF}/{HDF5[0]}", *[f"{HF}/data/{d[0]}-00000-of-00001.parquet" for d in DOMAINS],
                    f"{CODE}/bench/datamodules.py", f"{CODE}/bench/dataclasses.py", f"{CODE}/bench/pipelines.py"],
        "subsets": [{"name": s, "n": len(m), "defined_by": "the authors' domain (Hugging Face split)", "member_ids": m}
                    for s, m in members.items()],
        "problems": [], "datasets": datasets,
        "files": {"hdf5": {"path": f"catalog-prep/raw/{TAG}/{HDF5[0]}", "url": f"{HF}/{HDF5[0]}", "sha256": HDF5[1],
                           "bytes": HDF5[2]},
                  **{d[0]: {"path": f"catalog-prep/raw/{TAG}/{d[0]}-00000-of-00001.parquet",
                            "url": f"{HF}/data/{d[0]}-00000-of-00001.parquet", "sha256": d[3]} for d in DOMAINS}},
        "notes": (
            "Made by tools/catalog_build/llm_srbench_records.py from the authors' release. The release holds 240 "
            "problems (LSR-Transform 111; LSR-Synth chemistry 36, biology 24, physics 44, materials science 25); the "
            "paper's text counts 239 with 43 physics problems, its Table 4 lists 44. Names are the release's, numbered "
            "from 0 where Table 4 numbers from 1 (CRK0 is its CRK1); the materials science names skip MatSci15, 17, 26 "
            "and 27 (MatSci28 is its MatSci25). The authors publish no generation "
            f"code, input ranges or seeds. Chemistry and physics formulas name constants without values "
            f"({report['fitted']} problems); srbf fitted them to the training rows ({STARTS} Levenberg-Marquardt "
            "starts). Every formula reproduces its rows to float32 precision (formula_check): within FVU 1.2e-7, or, "
            f"for {len(report['exact_by_rounding'])} LSR-Transform problems whose formula is very steep somewhere in "
            f"their input range ({', '.join(sorted(report['exact_by_rounding']))}), on every row within the error "
            "that storing the inputs as float32 explains (the authors computed each target before rounding its row). "
            "LSR-Synth chemistry, biology and physics rows lie on one simulated trajectory per problem, the "
            "out-of-domain test is its last stretch of time; the out-of-domain targets of "
            f"{len(report['flat_ood'])} problems are nearly constant (spread below {FLAT:g} of the training spread: "
            f"{', '.join(sorted(report['flat_ood']))}). Materials science strain and temperature move together on "
            "one line (T = 273 + 500 strain), the out-of-domain test is its upper end."),
    }
    return paths.public(rec), [paths.public(p) for p in finals], report


def write_gz(path: Path, obj) -> None:
    raw = json.dumps(obj, indent=None, separators=(",", ":"), ensure_ascii=False).encode()
    with open(path, "wb") as fh, gzip.GzipFile(fileobj=fh, mode="wb", mtime=0, filename="") as gz:
        gz.write(raw)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="write the records into assets/upstream/catalog-prep")
    a = ap.parse_args()
    raw = paths.RAW / TAG
    if sha256_file(raw / HDF5[0]) != HDF5[1]:
        raise SystemExit(f"{raw / HDF5[0]}: sha256 differs from the release")
    rec, finals, report = record(raw)
    print(json.dumps({"problems": len(finals), **report}, indent=1))
    if not a.write:
        return
    sets = json.load(gzip.open(paths.PREP / "data_sets.json.gz", "rt"))
    sets[TAG] = rec
    write_gz(paths.PREP / "data_sets.json.gz", dict(sorted(sets.items())))
    final = json.load(gzip.open(paths.PREP / "final_problems.json.gz", "rt"))
    final["problems"] = [p for p in final["problems"] if p["set"] != TAG] + finals
    write_gz(paths.PREP / "final_problems.json.gz", final)
    print(f"wrote {TAG}: {len(finals)} data problems")


if __name__ == "__main__":
    main()
