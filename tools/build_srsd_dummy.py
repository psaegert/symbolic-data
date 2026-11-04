"""srsd-dummy: FastSRB's laws (fastsrb.v2.yaml) with SRSD's PUBLISHED dummy columns.

Rule (owner, 2026-11-04): dummies come from the benchmark. SRSD publishes its dummy configuration only as
data: the Hugging Face datasets yoshitomo-matsubara/srsd-feynman_{easy,medium,hard}_dummy (CC BY 4.0). No
seed exists anywhere in omron-sinicx/srsd-benchmark (dummy_column_mixer.py uses the unseeded `random`
module), so re-running the recipe cannot reproduce the published layout; the files are the record.

Stages
  1. derive  For every problem, read supp_info.json's `dummy_vars` (which input columns are dummies) and
             recover each dummy's law from the published rows. dummy_column_mixer.py (srsd-benchmark @
             7d00b45, L79-87) draws every dummy as DefaultSampling(10^(k-1), 10^(k+1), uses_negative=u),
             k ~ randint(-32, 32), u ~ Bernoulli(1/2): base-10 log-uniform over two decades, and (u) an
             independent fair sign per row. Both parameters are identified exactly by the 10,000 published
             rows: log10|x| fills (k-1, k+1) to within ~1e-3 at both ends, and the negative share is 0 or
             ~1/2. The stage asserts this per split and pooled.             -> srsd_dummy_spec.json
  2. build   For every fastsrb.v2 entry: SRSD's column order, i.e. the real variables v1..vn in their
             fastsrb order with the dummies inserted at SRSD's published positions; `prepared`, `accept`
             and `constraints` renumbered; every other field carried verbatim.  -> srsd-dummy.v2.yaml
  3. diff    Against the committed srsd-dummy.yaml (srsd-dummy@1).          -> srsd-dummy.v2.diff.md
  4. verify  (--verify) End to end: the candidate's `prepared`, evaluated on SRSD's published dummy files
             with column j bound to v(j+1), must reproduce the published target; its declared dummy laws
             must contain the published dummy columns. Optionally (--validate-loader) the candidate is
             linted and realized through symbolic_data's strict loader.

The spec (stage 1's output) is vendored as assets/upstream/srsd/srsd_dummy_spec.json, so a build needs no network;
--rederive (and --verify) read the pinned Hugging Face files into --hf-dir.

Run from the repo root:
    python tools/build_srsd_dummy.py [--report-dir DIR] [--rederive] [--verify] [--validate-loader]
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import urllib.request
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
FASTSRB = REPO / "assets/catalogs/fastsrb.v2.yaml"
CURRENT = REPO / "assets/catalogs/srsd-dummy.yaml"

HF_REPO = "yoshitomo-matsubara/srsd-feynman_{tier}_dummy"
HF_REVISIONS = {  # HF dataset commits (all lastModified 2024-03-05); pin these
    "easy": "67a1cac8420adf421fd0a18d91fbee5f1c0bca2d",
    "medium": "e373ffb9b2ee1f46f1eb8204fe35c6abeb589e65",
    "hard": "223c23c15837aa998e43010f125b3da5fd8cb7a8",
}
SRSD_CODE = "omron-sinicx/srsd-benchmark @ 7d00b45d56250717ac08a256dc9f79b836d61027 (MIT)"
SPLITS = ("train", "val", "test")
SIGN_TOL = 0.03       # |neg share - 1/2| for a signed dummy; 10,000 rows give sd 0.005
EDGE_TOL = 0.01       # decades: log10|x| must reach within this of k-1 and k+1
FLOAT32_TOL = 1e-6    # the published dummy files are float32-rounded throughout


# ----------------------------------------------------------------------------------------------- naming
def srsd_key_to_fastsrb(key: str) -> str:
    """'feynman-ii.38.14' -> 'II.38.14', 'feynman-bonus.6' -> 'B6'.

    SRSD's file names keep AI Feynman's trailing zero ('I.6.20' is AI Feynman's I.6.2), and SRSD calls the
    angular-momentum law m r v sin(theta) 'I.18.16' where FastSRB calls it 'I.18.14'."""
    s = key[len("feynman-"):]
    if s.startswith("bonus."):
        return "B" + s.split(".")[1]
    chapter, *rest = s.split(".")
    fid = chapter.upper() + "." + ".".join(rest)
    return {"I.6.20": "I.6.2", "I.6.20a": "I.6.2a", "I.6.20b": "I.6.2b", "I.15.10": "I.15.1",
            "I.34.10": "I.34.1", "I.39.10": "I.39.1", "I.18.16": "I.18.14"}.get(fid, fid)


# ------------------------------------------------------------------------------------------- 1. derive
def hf_file(hf_dir: Path, tier: str, rel: str) -> Path:
    """A published file, cached under hf_dir/<tier>/<rel>; fetched at the pinned revision if absent."""
    path = hf_dir / tier / rel
    if not path.exists():
        url = f"https://huggingface.co/datasets/{HF_REPO.format(tier=tier)}/resolve/{HF_REVISIONS[tier]}/{rel}"
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url) as resp:  # noqa: S310 (pinned https URL)
            path.write_bytes(resp.read())
    return path


def recover_law(x: np.ndarray) -> dict:
    lg = np.log10(np.abs(x.astype(np.float64)))
    lo, hi = float(lg.min()), float(lg.max())
    k = int(round((lo + hi) / 2))
    neg = float(np.mean(x < 0))
    return {"k": k, "uses_negative": neg > 0, "log10_abs_min": round(lo, 5), "log10_abs_max": round(hi, 5),
            "neg_share": round(neg, 4), "n": int(x.size)}


def check_law(law: dict, where: str) -> None:
    k = law["k"]
    assert -32 <= k <= 32, f"{where}: k={k} outside randint(-32, 32)"
    assert k - 1 - 1e-6 <= law["log10_abs_min"] <= k - 1 + EDGE_TOL, f"{where}: lower edge {law}"
    assert k + 1 - EDGE_TOL <= law["log10_abs_max"] <= k + 1 + 1e-6, f"{where}: upper edge {law}"
    if law["uses_negative"]:
        assert abs(law["neg_share"] - 0.5) <= SIGN_TOL * (1 if law["n"] >= 5000 else 3), f"{where}: sign {law}"


def derive(hf_dir: Path) -> dict:
    problems = {}
    for tier in HF_REVISIONS:
        supp = json.loads(hf_file(hf_dir, tier, "supp_info.json").read_text())
        for key, info in supp.items():
            fid = srsd_key_to_fastsrb(key)
            parts = {s: np.loadtxt(hf_file(hf_dir, tier, f"{s}/{key}.txt")) for s in SPLITS}
            n_in = parts["train"].shape[1] - 1
            assert all(p.shape[1] - 1 == n_in for p in parts.values()), fid
            dummy_cols = sorted(int(v[1:]) for v in info["dummy_vars"])
            used = sorted({int(m) for m in re.findall(r"\bx(\d+)\b", info["sympy_eq_str"])})
            assert used == [c for c in range(n_in) if c not in dummy_cols], \
                f"{fid}: supp_info dummy_vars {dummy_cols} do not complement the ground truth's {used}"
            dummies = {}
            for c in dummy_cols:
                per_split = {s: recover_law(p[:, c]) for s, p in parts.items()}
                pooled = recover_law(np.concatenate([p[:, c] for p in parts.values()]))
                check_law(pooled, f"{fid} col {c}")
                for s, law in per_split.items():  # the mixer draws one law per problem for all splits
                    assert (law["k"], law["uses_negative"]) == (pooled["k"], pooled["uses_negative"]), (fid, c, s)
                dummies[str(c)] = pooled
            problems[fid] = {"srsd_key": key, "tier": tier, "n_inputs": n_in, "dummy_columns": dummy_cols,
                             "dummies": dummies, "sympy_eq_str": info["sympy_eq_str"],
                             "rows": {s: int(p.shape[0]) for s, p in parts.items()}}
    return {
        "source": {
            "datasets": {t: f"{HF_REPO.format(tier=t)}@{sha}" for t, sha in HF_REVISIONS.items()},
            "license": "CC BY 4.0 (datasets); recipe MIT",
            "recipe": f"{SRSD_CODE}: dummy_column_mixer.py L62-87 (positions randint(0, n_org) per dummy, "
                      "inserted in descending order; law DefaultSampling(10^(k-1), 10^(k+1), uses_negative=u), "
                      "k ~ randint(-32, 32), u ~ Bernoulli(1/2))",
            "positions": "supp_info.json `dummy_vars` (0-based column indices x<i> of the published files)",
            "laws": "recovered from the published rows (train+val+test, 10,000 per problem); k and u are "
                    "identified exactly, see check_law()",
        },
        "problems": problems,
    }


# -------------------------------------------------------------------------------------------- 2. build
SRSD_CITE = ("dummy columns as published by SRSD (Matsubara et al. 2024, 'Rethinking Symbolic Regression "
             "Datasets and Benchmarks for Scientific Discovery', DMLR): positions from the Hugging Face "
             "datasets' supp_info.json, laws (k, sign) read off the published rows; recipe "
             "omron-sinicx/srsd-benchmark dummy_column_mixer.py (MIT), data CC BY 4.0")


def renumber(text: str, mapping: dict[str, str]) -> str:
    """One-pass v-index substitution (re.sub never re-reads its own output, so v1->v2, v2->v4 is safe)."""
    return re.sub(r"\bv(\d+)\b", lambda m: mapping.get(m.group(0), m.group(0)), text)


def decade(e: int) -> float:
    return float(f"1e{e}")  # the correctly rounded literal 10^e (SRSD's np.power(10.0, e) may be 1 ulp off)


def dummy_var(j: int, law: dict) -> dict:
    return {"name": f"dummy{j}",
            "sample_range": [decade(law["k"] - 1), decade(law["k"] + 1)],
            "sample_type": ["log", "pos_neg" if law["uses_negative"] else "pos"]}


def norm_var(spec: dict) -> dict:
    spec = dict(spec)
    if "sample_range" in spec:  # YAML-1.1 string literals ('10e24') compare by value; leaked tokens stay
        def num(v):
            try:
                return float(v)
            except (TypeError, ValueError):
                return v
        spec["sample_range"] = [num(v) for v in spec["sample_range"]]
    return spec


def real_vars_of(entry: dict) -> list[dict]:
    keys = sorted((k for k in entry["vars"] if k != "v0"), key=lambda s: int(s[1:]))
    return [entry["vars"][k] for k in keys]


def build(spec: dict, fastsrb_path: Path, current_path: Path) -> tuple[dict, list[str]]:
    fs_doc = yaml.safe_load(fastsrb_path.read_text())
    cur = yaml.safe_load(current_path.read_text())["expressions"]
    base_version = fs_doc["metadata"]["version"]
    base_sha = hashlib.sha256(fastsrb_path.read_bytes()).hexdigest()
    notes: list[str] = []
    out = {}
    for fid, entry in fs_doc["expressions"].items():
        s = spec["problems"][fid]
        keys = sorted((k for k in entry["vars"] if k != "v0"), key=lambda k: int(k[1:]))
        assert keys == [f"v{i}" for i in range(1, len(keys) + 1)], fid
        dcols = s["dummy_columns"]
        real_cols = [c for c in range(s["n_inputs"]) if c not in dcols]
        assert len(real_cols) == len(keys), f"{fid}: SRSD has {len(real_cols)} real inputs, fastsrb {len(keys)}"
        if entry.get("difficulty") and entry["difficulty"] != s["tier"]:
            notes.append(f"{fid}: fastsrb difficulty {entry['difficulty']!r} != SRSD tier {s['tier']!r}")
        mapping = {f"v{i + 1}": f"v{c + 1}" for i, c in enumerate(real_cols)}

        new_vars, j = {}, 0
        for c in range(s["n_inputs"]):
            if c in dcols:
                j += 1
                new_vars[f"v{c + 1}"] = dummy_var(j, s["dummies"][str(c)])
            else:
                new_vars[f"v{c + 1}"] = copy.deepcopy(entry["vars"][f"v{real_cols.index(c) + 1}"])
        new_vars["v0"] = copy.deepcopy(entry["vars"]["v0"])

        # finite_fraction: dummies never enter the expression, so an entry's valid fraction is its base
        # law's. fastsrb.v2 discloses it where its law changed; elsewhere srsd-dummy@1's estimate carries
        # over iff the base law (raw + every real variable's spec) is unchanged.
        meta_in = entry.get("meta") or {}
        ce = cur.get(fid, {})
        cur_real = [v for v in real_vars_of(ce) if not str(v.get("name", "")).startswith("dummy")] if ce else []
        same_law = bool(ce) and ce.get("raw") == entry["raw"] and \
            [norm_var(v) for v in cur_real] == [norm_var(v) for v in real_vars_of(entry)]
        if "finite_fraction" in meta_in:
            ff, ff_src = meta_in["finite_fraction"], "fastsrb.v2"
        elif same_law and "finite_fraction" in (ce.get("meta") or {}):
            ff, ff_src = ce["meta"]["finite_fraction"], "srsd-dummy@1 (same base law)"
        else:
            ff, ff_src = None, "RECOMPUTE"
            notes.append(f"{fid}: no finite_fraction to carry; run tools/audit_finite_fraction.py")

        new = {}
        for key, val in entry.items():  # fastsrb's field order; every field carried unless renumbered
            if key == "prepared":
                new[key] = renumber(val, mapping)
            elif key == "accept":
                new[key] = [renumber(a, mapping) for a in val]
            elif key == "constraints":
                new[key] = [renumber(c, mapping) for c in val] if val else val
            elif key == "row_constraints":
                new[key] = [renumber(c, mapping) for c in val]
            elif key == "vars":
                new[key] = new_vars
            elif key == "n_variables":
                new[key] = len(real_cols) + len(dcols)
            elif key == "sources":
                new[key] = list(val) + [SRSD_CITE]
            elif key == "meta":
                continue
            else:
                new[key] = copy.deepcopy(val)
        new.setdefault("n_variables", len(real_cols) + len(dcols))
        meta = {
            "base_catalog": f"fastsrb@{base_version}", "base_eq_id": fid,
            "srsd_dataset": f"{HF_REPO.format(tier=s['tier'])}@{HF_REVISIONS[s['tier']][:7]}",
            "srsd_file": s["srsd_key"],
            "n_dummy": len(dcols),
            "dummy_vars": [f"v{c + 1}" for c in dcols],
        }
        if ff is not None:
            meta["finite_fraction"] = ff
            if ff < 0.05:
                meta["low_validity"] = True
        meta["finite_fraction_source"] = ff_src
        new["meta"] = meta
        out[fid] = new

    counts = Counter(len(spec["problems"][f]["dummy_columns"]) for f in out)
    catalog = {
        "metadata": {
            "name": "srsd-dummy",
            "version": 2,
            "license": ("MIT (base laws: viktmar/FastSRB; dummy recipe: omron-sinicx/srsd-benchmark) + "
                        "CC BY 4.0 (dummy configuration: SRSD's Hugging Face datasets); see THIRD_PARTY_LICENSES"),
            "description": (f"SRSD-Feynman with dummy variables (120): FastSRB's laws (fastsrb@{base_version}) "
                            "with SRSD's published dummy columns -- the same number, the same column positions "
                            "and the same per-column law as SRSD's srsd-feynman_{easy,medium,hard}_dummy "
                            f"(1/2/3 dummies on {counts[1]}/{counts[2]}/{counts[3]} problems)."),
            "sources": [
                "Feynman Symbolic Regression Database (Udrescu & Tegmark 2020, AI Feynman)",
                "SRSD realistic ranges (Matsubara et al. 2024, CC BY 4.0 datasets / MIT code)",
                f"FastSRB (Martinek, arXiv:2508.14481, MIT): base laws from fastsrb.v2.yaml "
                f"(metadata.version {base_version}, sha256 {base_sha[:16]})",
                *[f"dummy configuration: huggingface.co/datasets/{HF_REPO.format(tier=t)} @ {sha}"
                  for t, sha in HF_REVISIONS.items()],
                f"dummy recipe: {SRSD_CODE}, dummy_column_mixer.py",
            ],
            "sampling_defaults": {"n_points": 100, "method": "random", "noise": 0.0},
            "source_kind": "set",
            "conventions": {
                "sampling": "vars use the `fastsrb` distribution (sample_range/sample_type) or `dist`",
                "dummies": ("a variable named dummy<j> is irrelevant (absent from `prepared`). Its column is "
                            "SRSD's published one (supp_info.json dummy_vars), and its law is SRSD's "
                            "DefaultSampling(10^(k-1), 10^(k+1), uses_negative=u) with k and u read off the "
                            "published rows: sample_type [log, pos_neg] when u, else [log, pos]. Only the "
                            "configuration is SRSD's; the rows are drawn fresh at evaluation time."),
                "oracle_warning": ("adapters must NOT use a used-variable oracle here: variable masking "
                                   "(e.g. PySR padding:false) defeats the benchmark; eval configs must "
                                   "disable masking for every method"),
                "validity": ("accepted points follow the declared per-variable distribution CONDITIONED on the "
                             "expression's valid domain; meta.finite_fraction is the MC-estimated per-point "
                             "valid fraction (n=20000); entries below 0.05 carry low_validity: true and need "
                             "per-entry review."),
            },
            "changelog": {
                "2": ("rebuilt on fastsrb.v2 (I.29.16 +cos, B6 epsilon unsquared, B7 curvature {-1,0,1}, "
                      "III.15.27 alpha in +-{1..99}, III.21.20 rho/q negative, I.44.4 n on [1e24, 1e26], "
                      "integer inputs on {lo..hi-1} as SRSD's IntegerSampling) "
                      "with SRSD's published dummy columns in place of srsd-dummy@1's re-draw with seed "
                      "20260710 (which matched SRSD's dummy count on 41/120, positions on 9/120, full "
                      "configuration on 0/120)."),
            },
        },
        "expressions": out,
    }
    return catalog, notes


# --------------------------------------------------------------------------------------------- 3. diff
def layout(entry: dict) -> list[tuple[int, int | None, str | None]]:
    """[(0-based column, k, sign)] of an entry's dummies."""
    rows = []
    for i, spec in enumerate(real_vars_of(entry)):
        if str(spec.get("name", "")).startswith("dummy"):
            lo, hi = spec["sample_range"]
            rows.append((i, int(round(np.log10(float(lo)))) + 1, spec["sample_type"][1]))
    return rows


def fmt(lay: list) -> str:
    return " ".join(f"x{c}:1e{k - 1}..1e{k + 1}{'±' if s == 'pos_neg' else '+'}" for c, k, s in lay)


def diff(candidate: dict, current_path: Path, fastsrb1_path: Path) -> tuple[str, dict]:
    cur = yaml.safe_load(current_path.read_text())["expressions"]
    fs1 = yaml.safe_load(fastsrb1_path.read_text())["expressions"]
    new = candidate["expressions"]
    stats = Counter()
    lines = ["| id | tier | real | srsd-dummy@1 dummies | candidate (= SRSD) dummies | count | pos | full | base law |",
             "|---|---|---|---|---|---|---|---|---|"]
    for fid, ne in new.items():
        ce = cur[fid]
        a, b = layout(ce), layout(ne)
        same_count = len(a) == len(b)
        same_pos = [c for c, _, _ in a] == [c for c, _, _ in b]
        same_full = a == b
        stats["count"] += same_count
        stats["positions"] += same_pos
        stats["full"] += same_full
        law = "changed" if (fs1[fid]["raw"] != ne["raw"] or
                            [norm_var(v) for v in real_vars_of(fs1[fid])] !=
                            [norm_var(v) for v in real_vars_of(ne) if not str(v.get("name")).startswith("dummy")]) else ""
        stats["base_law_changed"] += bool(law)
        tier = ne["meta"]["srsd_dataset"].split("_")[1]
        n_real = ne["n_variables"] - ne["meta"]["n_dummy"]
        lines.append(f"| {fid} | {tier} | {n_real} | {fmt(a)} | {fmt(b)} | {'=' if same_count else 'x'} | "
                     f"{'=' if same_pos else 'x'} | {'=' if same_full else 'x'} | {law} |")
    cnt_a = Counter(len(layout(e)) for e in cur.values())
    cnt_b = Counter(len(layout(e)) for e in new.values())
    head = [
        "# srsd-dummy@1 (committed) vs candidate srsd-dummy v2", "",
        f"- dummy count equal: {stats['count']}/120; positions equal: {stats['positions']}/120; "
        f"positions + k + sign equal: {stats['full']}/120",
        f"- dummies per problem (1/2/3): srsd-dummy@1 {cnt_a[1]}/{cnt_a[2]}/{cnt_a[3]} "
        f"({sum(k * v for k, v in cnt_a.items())} columns); candidate {cnt_b[1]}/{cnt_b[2]}/{cnt_b[3]} "
        f"({sum(k * v for k, v in cnt_b.items())} columns)",
        f"- base law differs from fastsrb@1 (srsd-dummy@1's base) on {stats['base_law_changed']} entries",
        "- notation: x<c> = 0-based input column; 1eA..1eB = log-uniform |x|; ± = random sign, + = positive", "",
    ]
    return "\n".join(head + lines) + "\n", dict(stats)


# ------------------------------------------------------------------------------------------- 4. verify
FUNCS = {"sqrt": np.sqrt, "cos": np.cos, "sin": np.sin, "exp": np.exp, "log": np.log, "tanh": np.tanh,
         "cosh": np.cosh, "neg": np.negative, "abs": np.abs}
# Where SRSD's data generator (eq_func) is not its own ground truth (sympy_eq) -- the data follows eq_func.
EQ_FUNC_FORMS = {
    "III.9.52": "(v1 * v2 * v3 / (6.626e-34 / (2 * 3.1415926535897))) ^ 2 * sin((v4 - v5) * v3 / 2) ^ 2 "
                "/ ((v4 - v5) * v3 / 2) ^ 2",                                    # prefactor squared
    "I.34.14": "(1 + v1 / v2) / sqrt(1 - v1 ^ 2 / v2 ^ 2) * v2",               # omega_0 in place of c
    "B13": "1 / (4 * 3.1415926535897 * 8.854e-12) * v2 / sqrt(v3 ^ 2 + v4 ^ 2 - 2 * v3 * v4 * cos(v5))",  # eps0
}


def evaluate(expr: str, cols: dict) -> np.ndarray:
    with np.errstate(all="ignore"):
        return np.asarray(eval(expr.replace("^", "**"), {"__builtins__": {}}, {**FUNCS, **cols}), dtype=float)  # noqa: S307


def verify(candidate: dict, spec: dict, hf_dir: Path) -> list[str]:
    report = []
    for fid, e in candidate["expressions"].items():
        s = spec["problems"][fid]
        data = np.vstack([np.loadtxt(hf_file(hf_dir, s["tier"], f"{sp}/{s['srsd_key']}.txt")) for sp in SPLITS])
        X, y = data[:, :-1], data[:, -1]
        assert X.shape[1] == e["n_variables"], fid
        # (a) the dummy laws contain the published dummy columns
        for vk in e["meta"]["dummy_vars"]:
            spec_v = e["vars"][vk]
            x = X[:, int(vk[1:]) - 1]
            lo, hi = spec_v["sample_range"]
            a = np.abs(x)
            assert a.min() >= lo * (1 - FLOAT32_TOL) and a.max() <= hi * (1 + FLOAT32_TOL), (fid, vk)
            assert (spec_v["sample_type"][1] == "pos") == bool(np.all(x > 0)), (fid, vk)
        # (b) the published target from the candidate's own formula over the file's columns in order
        cols = {f"v{i + 1}": X[:, i] for i in range(X.shape[1])}
        expr = EQ_FUNC_FORMS.get(fid)
        if expr is not None:  # renumber the eq_func form like the candidate's prepared
            real = [i for i in range(X.shape[1]) if f"v{i + 1}" not in e["meta"]["dummy_vars"]]
            expr = renumber(expr, {f"v{j + 1}": f"v{c + 1}" for j, c in enumerate(real)})
        for tag, ex in (("prepared", e["prepared"]), ("eq_func", expr)):
            if ex is None:
                continue
            pred = evaluate(ex, cols)
            with np.errstate(all="ignore"):
                r = np.abs(pred - y) / np.maximum(np.abs(y), 1e-300)
            ok = float(np.mean(r < 1e-4))
            tiny = float(np.mean(np.abs(y) < 1e-37))  # float32 underflow of the published target
            if ok < 0.999 and ok + tiny < 0.999:
                report.append(f"{fid}: {tag} reproduces the published target on {ok:.4f} of rows "
                              f"(median rel err {np.nanmedian(r):.3g})")
            elif ok < 0.999:
                report.append(f"{fid}: {tag} reproduces {ok:.4f} of rows; the rest are float32-underflowed "
                              f"targets (|y| < 1e-37: {tiny:.4f})")
    return report


def validate_loader(path: Path, engine_name: str) -> list[str]:
    """Lint + realize every entry through symbolic_data's strict loader (needs the worktree's src on sys.path)."""
    import sys
    sys.path.insert(0, str(REPO / "src"))
    sys.path.insert(0, str(REPO / "tools"))
    from lint_catalogs import lint_paths  # type: ignore
    from simplipy import SimpliPyEngine
    from symbolic_data.catalog import ProblemCatalog

    out = [f"lint clean: {lint_paths([str(path)])}"]
    engine = SimpliPyEngine.load(engine_name, install=False)
    cat = ProblemCatalog.load(str(path))
    rng = np.random.default_rng(0)
    bad = []
    for entry in cat.entries.values():
        try:
            rex = cat.realize(entry, 64, rng, engine=engine)
            n_in = len([k for k in entry.variables if k != "v0"])
            if rex.x.shape != (64, n_in) or list(rex.variables) != [f"v{i}" for i in range(1, n_in + 1)]:
                bad.append(f"{entry.id}: realized x {rex.x.shape}, variables {rex.variables}")
            elif not np.all(np.isfinite(rex.y)):
                bad.append(f"{entry.id}: non-finite y")
            else:
                for i, vk in enumerate(rex.variables):  # every column inside its declared law
                    spec_v = entry.variables[vk]
                    if "sample_range" in spec_v:
                        lo, hi = map(float, spec_v["sample_range"])
                        a = np.abs(rex.x[:, i]) if spec_v["sample_type"][1] != "pos" or lo >= 0 else rex.x[:, i]
                        if a.min() < lo - 1e-12 * abs(lo) or a.max() > hi + 1e-12 * abs(hi):
                            bad.append(f"{entry.id}.{vk}: drawn outside [{lo}, {hi}]")
        except Exception as exc:  # noqa: BLE001
            bad.append(f"{entry.id}: {type(exc).__name__}: {str(exc)[:120]}")
    out.append(f"realized {len(cat.entries) - len(bad)}/{len(cat.entries)} entries")
    return out + bad


# ---------------------------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--fastsrb", type=Path, default=FASTSRB)
    ap.add_argument("--fastsrb1", type=Path, default=REPO / "assets/catalogs/fastsrb.yaml")
    ap.add_argument("--current", type=Path, default=CURRENT)
    ap.add_argument("--hf-dir", type=Path, default=Path.home() / ".cache/symbolic-data/srsd-hf")
    ap.add_argument("--spec", type=Path, default=REPO / "assets/upstream/srsd/srsd_dummy_spec.json")
    ap.add_argument("--out", type=Path, default=CURRENT)
    ap.add_argument("--report-dir", type=Path, default=None, help="where the diff and verify reports go")
    ap.add_argument("--rederive", action="store_true", help="re-read the HF files even if the spec exists")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--validate-loader", action="store_true")
    ap.add_argument("--engine", default="acj-4")
    args = ap.parse_args()

    spec_path = args.spec
    if args.rederive or not spec_path.exists():
        spec_path.parent.mkdir(parents=True, exist_ok=True)
        spec = derive(args.hf_dir)
        spec_path.write_text(json.dumps(spec, indent=1))
    spec = json.loads(spec_path.read_text())
    n_cols = sum(len(p["dummy_columns"]) for p in spec["problems"].values())
    print(f"spec: {len(spec['problems'])} problems, {n_cols} dummy columns -> {spec_path.name}")

    catalog, notes = build(spec, args.fastsrb, args.current)
    md, stats = diff(catalog, args.current, args.fastsrb1)   # against the catalog as it was, before writing
    if args.report_dir:
        args.report_dir.mkdir(parents=True, exist_ok=True)
        (args.report_dir / "srsd-dummy.diff.md").write_text(md, encoding="utf-8")
    print("diff vs the previous catalog:", stats)
    out_path = args.out
    out_path.write_text(yaml.safe_dump(catalog, sort_keys=False, allow_unicode=True, width=120), encoding="utf-8")
    print(f"wrote {out_path} ({len(catalog['expressions'])} entries)")
    for n in notes:
        print("  note:", n)

    if args.verify:
        rep = verify(catalog, spec, args.hf_dir)
        if args.report_dir:
            (args.report_dir / "verify.txt").write_text("\n".join(rep) + "\n")
        print("verify: candidate prepared vs published target --", len(rep), "findings")
        for r in rep:
            print("  ", r)
    if args.validate_loader:
        for line in validate_loader(out_path, args.engine):
            print("  loader:", line)


if __name__ == "__main__":
    main()
