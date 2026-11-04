"""How each survey set's published files become data-problem rows (build_data_catalogs.py).

One reader per set: ``read(record, problem, set_record) -> Rows | list[Rows]``. A reader takes the rows exactly as the
record defines the problem (columns, filters, the authors' own transforms that define the target) and states the
split it reads from the record's ``split`` text:

* ``GIVEN``: the authors pin the test rows (files, indices, a deterministic rule, one seeded split with known rows);
* ``random(f)``: the authors redraw a random split per run (holdout ``f``; k-fold CV is ``1/k``);
* ``NONE``: no split and no procedure.

Every file is checked against the record's sha256 before it is read. A record names a file by its place in the survey's
working copy (``catalog-prep/raw/...``); paths.input_file finds it under SYMBOLIC_DATA_CATALOG_RAW.
"""
from __future__ import annotations

import hashlib
import io
import json
import urllib.request
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

import paths

CACHE = paths.CACHE / "data"

GIVEN = "given"
NONE = "none"


def random(f: float) -> dict:
    return {"random": float(f)}


def rule_of(split) -> str:
    return split if isinstance(split, str) else "random"


@dataclass
class Rows:
    """One data problem's rows: the fitting pool, the authors' test rows (``GIVEN`` only) and how the split was read."""
    x: np.ndarray
    y: np.ndarray
    features: list[str]
    split: str | dict
    split_note: str
    x_test: np.ndarray | None = None
    y_test: np.ndarray | None = None
    subset: str | None = None          # set for a problem the record splits into several (e.g. a second test set)
    eq_id: str | None = None
    formula_names: list[str] | None = None   # the formula's variable names, aligned with ``features``
    transforms: list[str] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    def finish(self) -> None:
        """float64 arrays; rows with a non-finite value dropped (counted in meta); the split's shape checked."""
        self.x = _matrix(self.x)
        self.y = np.asarray(self.y, dtype=np.float64).reshape(-1)
        dropped = 0
        self.x, self.y, n = _finite(self.x, self.y)
        dropped += n
        if rule_of(self.split) == GIVEN:
            if self.x_test is None or len(self.x_test) == 0:
                raise ValueError(f"{self.eq_id}: a given split without test rows")
            self.x_test = _matrix(self.x_test)
            self.y_test = np.asarray(self.y_test, dtype=np.float64).reshape(-1)
            self.x_test, self.y_test, n = _finite(self.x_test, self.y_test)
            dropped += n
        elif self.x_test is not None:
            raise ValueError(f"{self.eq_id}: test rows stored for split {self.split}")
        if dropped:
            self.meta["dropped_nonfinite_rows"] = int(dropped)
        if self.x.shape[1] != len(self.features):
            raise ValueError(f"{self.eq_id}: {self.x.shape[1]} columns for {len(self.features)} features")
        if len(self.y) < 2:
            raise ValueError(f"{self.eq_id}: {len(self.y)} fitting rows")

    def summary(self) -> str:
        test = 0 if self.y_test is None else len(self.y_test)
        return f"{self.eq_id or ''} fit {len(self.y)} test {test} split {json.dumps(self.split)} x{self.x.shape[1]}"


def _matrix(a) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    return a.reshape(len(a), -1)


def _finite(x: np.ndarray, y: np.ndarray):
    keep = np.isfinite(x).all(axis=1) & np.isfinite(y)
    return x[keep], y[keep], int((~keep).sum())


class SkipDataset(Exception):
    """A dataset the builder leaves out, with the reason (reported)."""


@dataclass
class Spec:
    slug: str
    display: str
    read: Callable


SETS: dict[str, Spec] = {}


def reader(tag: str, slug: str, display: str):
    def deco(fn):
        SETS[tag] = Spec(slug, display, fn)
        return fn
    return deco


def as_list(rows) -> list[Rows]:
    return rows if isinstance(rows, list) else [rows]


# --------------------------------------------------------------------------------------------------- files and checks
def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def checked(path, want: str | None = None) -> bytes:
    """The file's bytes, checked against the recorded sha256 when there is one."""
    data = paths.input_file(path).read_bytes()
    if isinstance(want, str) and sha256(data) != want:
        raise ValueError(f"{path}: sha256 {sha256(data)} != recorded {want}")
    return data


def fetch(url: str, want: str, suffix: str = "") -> bytes:
    """A recorded URL's bytes, cached under data_cache/<sha256><suffix> and checked against the recorded sha256."""
    cached = CACHE / f"{want}{suffix}"
    if cached.exists():
        return checked(cached, want)
    req = urllib.request.Request(url, headers={"User-Agent": "srbf-catalog-build"})
    with urllib.request.urlopen(req, timeout=300) as resp:
        data = resp.read()
    if sha256(data) != want:
        raise ValueError(f"{url}: sha256 {sha256(data)} != recorded {want}")
    CACHE.mkdir(parents=True, exist_ok=True)
    cached.write_bytes(data)
    return data


def csv(data: bytes, **kw) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(data), **kw)


def columns(df: pd.DataFrame, features: list[str], target: str):
    missing = [c for c in [*features, target] if c not in df.columns]
    if missing:
        raise KeyError(f"columns {missing} not in {list(df.columns)[:12]}")
    return df[features].to_numpy(dtype=np.float64), df[target].to_numpy(dtype=np.float64)


# ------------------------------------------------------------------------------------------------------------ readers
@reader("fresh-60", "mvsr-2024", "Multi-view SR (2024) real data")
def mvsr_2024(rec, prob, _set):
    # every record: Xaxis0 -> yaxis (finance files also carry the raw bin 'count', not an input)
    df = csv(checked(rec["file"], rec["sha256"]))
    x, y = columns(df, ["Xaxis0"], "yaxis")
    return Rows(x, y, [rec.get("feature") or "Xaxis0"], NONE,
                "none: the authors refit and score each form on all rows of the view")


@reader("fresh-61", "mvsr-2025", "Multi-view SR (2025) real-world data")
def mvsr_2025(rec, prob, _set):
    df = csv(checked(rec["file"], rec["sha256"]))
    feat, target = rec["feature"], rec["target"]
    x, y = columns(df, [feat], target)
    return Rows(x, y, [feat], random(0.2),
                "random 0.2: each view is split anew per seed, 80% train / 20% test (the authors draw the test rows "
                "with replacement; srbf draws without)")


@reader("fresh-94", "sldbench", "SLDBench (scaling-law discovery)")
def sldbench(rec, prob, _set):
    tr = pd.read_parquet(io.BytesIO(checked(rec["file"], rec["sha256"])))
    te = pd.read_parquet(io.BytesIO(checked(rec["test_file"], rec["test_sha256"])))
    for key, val in (rec.get("row_filter") or {}).items():
        tr, te = tr[tr[key].astype(str) == str(val)], te[te[key].astype(str) == str(val)]
    feats = list(rec["features"])
    x, y = columns(tr, feats, rec["target"])
    xt, yt = columns(te, feats, rec["target"])
    if (len(y), len(yt)) != (rec["n_rows_train"], rec["n_rows_test"]):
        raise ValueError(f"{rec['id']}: {len(y)}/{len(yt)} rows, record {rec['n_rows_train']}/{rec['n_rows_test']}")
    return Rows(x, y, feats, GIVEN, "given: the published train (seen) / test (unseen) files of the group",
                x_test=xt, y_test=yt)


@reader("fresh-40", "gp-challenge-protein", "GP Challenge (protein energy function)")
def gp_challenge(rec, prob, _set):
    f = rec["files"]
    x = np.loadtxt(io.BytesIO(checked(f["energy_terms"]["path"], f["energy_terms"]["sha256"])))
    d = np.loadtxt(io.BytesIO(checked(f["distances"]["path"], f["distances"]["sha256"])))
    if len(x) != len(d) or len(x) != rec["n_rows"]:
        raise ValueError(f"{rec['id']}: {len(x)} energy rows, {len(d)} distance rows, record {rec['n_rows']}")
    return Rows(x, d[:, 0], [n.split(" ")[0] for n in rec["feature_names"]], NONE,
                "none: the authors train on all decoys of all proteins and define no test set",
                meta={"target": "rank (the authors' RMSD ranking R2)"})


def _role_files(rec) -> dict:
    """{'train': entry, 'test': entry, ...} from a record's files (a dict by role, or a list with 'role' texts)."""
    f = rec.get("files")
    if isinstance(f, dict):
        return f
    out = {}
    for e in f or []:
        role = str(e.get("role", "")).split(" ")[0].lower()
        out.setdefault(role, e)
    return out


def _gecco_2022(rec, prob, _set, note):
    f = _role_files(rec)
    tr = csv(checked(f["train"]["path"], f["train"].get("sha256")))
    te = csv(checked(f["test"]["path"], f["test"].get("sha256")))
    feats = [c for c in tr.columns if c != "y"]
    x, y = columns(tr, feats, "y")
    xt, yt = columns(te, feats, "y")
    return Rows(x, y, feats, GIVEN, note, x_test=xt, y_test=yt)


@reader("done-11", "gecco-2022-exact-rediscovery", "GECCO 2022 synthetic: exact rediscovery")
def gecco_exact(rec, prob, s):
    return _gecco_2022(rec, prob, s, "given: the published _train / _test files (rows 0-999 / 1000-1999)")


@reader("done-12", "gecco-2022-feature-selection", "GECCO 2022 synthetic: feature selection")
def gecco_features(rec, prob, s):
    return _gecco_2022(rec, prob, s, "given: the published _train / _test files (rows 0-499 / 500-999)")


@reader("done-13", "gecco-2022-local-optima", "GECCO 2022 synthetic: escaping local optima")
def gecco_localopt(rec, prob, s):
    return _gecco_2022(rec, prob, s, "given: the published _train / _test files (rows 0-99 / 100-999)")


@reader("done-14", "gecco-2022-extrapolation", "GECCO 2022 synthetic: extrapolation")
def gecco_extrapolation(rec, prob, s):
    return _gecco_2022(rec, prob, s, "given: the published _train (x <= 15) / _test (x > 15) files")


@reader("done-15", "gecco-2022-noise", "GECCO 2022 synthetic: noise sensitivity")
def gecco_noise(rec, prob, s):
    return _gecco_2022(rec, prob, s, "given: the published _train (grid) / _test (uniform draws) files")


@reader("fresh-95", "scilaws-real", "SciLaws-Bench SciLaws-Real (Type I)")
def scilaws(rec, prob, _set):
    f = _role_files(rec)
    tr = csv(fetch(f["train"]["url"], f["train"]["sha256"], ".csv"))
    te = csv(fetch(f["test"]["url"], f["test"]["sha256"], ".csv"))
    target = next(c["name"] for c in rec["columns"] if c["role"] == "target")
    feats = [c["name"] for c in rec["columns"] if c["role"] == "input"]
    if feats != list(rec["inputs"]):
        raise ValueError(f"{rec['id']}: inputs {feats} != record {rec['inputs']}")
    x, y = columns(tr, feats, target)
    xt, yt = columns(te, feats, target)
    return Rows(x, y, feats, GIVEN, "given: the published train.csv / test.csv", x_test=xt, y_test=yt)


@reader("fresh-170", "implied-volatility-slices", "Implied-volatility slices")
def implied_volatility(rec, prob, _set):
    arr = np.load(io.BytesIO(checked(rec["file"], rec["sha256"])))   # (slices, [x, y], points), 125 points
    row = int(rec["id"].rsplit("-", 1)[1])
    return Rows(arr[row, 0], arr[row, 1], ["normalized strike"], NONE,
                "none: the authors fit every slice in-sample",
                meta={"target": "total implied variance w = sigma^2 T", "points": "the 125-point array"})


@reader("fresh-58", "physo-milky-way-streams", "PhySO Milky Way stellar streams")
def physo_streams(rec, prob, _set):
    df = csv(checked(rec["file"], rec["sha256"]))
    sid = int(rec["source"].split("sID == ")[1].split(",")[0].split(" ")[0])
    s = df[df["sID"] == sid]
    if len(s) != rec["n_rows"]:
        raise ValueError(f"{rec['id']}: {len(s)} rows for sID {sid}, record {rec['n_rows']}")
    r = np.sqrt(s["x"] ** 2 + s["y"] ** 2 + s["z"] ** 2).to_numpy() / 20.0
    v = np.sqrt(s["vx"] ** 2 + s["vy"] ** 2 + s["vz"] ** 2).to_numpy() / 200.0
    return Rows(r, 0.5 * v ** 2, ["r"], NONE, "none: the authors fit and score all 100 points of a stream",
                transforms=["r = sqrt(x^2+y^2+z^2)/20 (kpc -> 20 kpc)", "y = 0.5 (|v|/200)^2 (km/s -> 200 km/s)"
                            " (the authors' normalization, MW_streams_run.py)"])


_CP3_SPLIT = None


@reader("fresh-54", "cp3-bench", "CP3-bench cosmology")
def cp3_bench(rec, prob, _set):
    global _CP3_SPLIT
    if _CP3_SPLIT is None:
        _CP3_SPLIT = json.loads(paths.input_file("catalog-prep/raw/fresh-54/split_train_test_random_state42.json").read_text())
    df = csv(checked(rec["file"], rec["sha256"]))
    feats = list(rec["feature_columns"])
    x, y = columns(df, feats, "target")
    sp = _CP3_SPLIT[rec["id"]]
    tr = np.asarray(sp["train"], dtype=int)
    te = np.asarray(sp["test"], dtype=int) if "test" in sp else np.setdiff1d(np.arange(len(y)), tr)
    if (len(tr), len(te)) != (sp["n_train"], sp["n_test"]) or len(np.intersect1d(tr, te)):
        raise ValueError(f"{rec['id']}: split {len(tr)}/{len(te)} vs {sp['n_train']}/{sp['n_test']}")
    return Rows(x[tr], y[tr], feats, GIVEN,
                "given: the one seeded split cp3-bench evaluates on (train_test_split test_size 0.2, random_state 42)",
                x_test=x[te], y_test=y[te])


# Oliveira et al. 2018 (GECCO 2013-2017 compilation): the compilers' protocol is one shuffled 5-fold CV per dataset,
# so a member without a split of its own data authors is read as random 0.2 (as SRBench's 75/25 is for SRBench).
_OLIVEIRA_CV = "random 0.2: the compilers' shuffled 5-fold CV (Oliveira et al. 2018); the data authors give no split"
_OLIVEIRA_PARTS = None


def _xls(data: bytes, **kw) -> pd.DataFrame:
    return pd.read_excel(io.BytesIO(data), **kw)


@reader("fresh-33", "gecco-2013-2017-real", "GECCO 2013-2017 GP regression compilation (real data)")
def oliveira_2018(rec, prob, _set):
    global _OLIVEIRA_PARTS
    k = rec["id"]
    data = checked(rec["file"], rec["sha256"]) if rec.get("file") else None
    names = [str(n) for n in (rec.get("feature_names") or [])]
    if k == "ABA":   # the compilation's abalone: sex one-hot (F, I, M first), then its 500-row sample
        if _OLIVEIRA_PARTS is None:
            _OLIVEIRA_PARTS = json.loads(paths.input_file("catalog-prep/raw/fresh-33/oliveira2018_real_partitions_reconstructed.json").read_text())
        df = csv(data, header=None)
        sex = pd.get_dummies(df[0]).astype(float)[["F", "I", "M"]]
        x = np.column_stack([sex.to_numpy(), df.iloc[:, 1:8].to_numpy(dtype=float)])
        rows = np.asarray(_OLIVEIRA_PARTS["abalone_sample_rows"], dtype=int)
        return Rows(x[rows], df.iloc[:, 8].to_numpy(dtype=float)[rows], ["sex=F", "sex=I", "sex=M", *names[1:]],
                    random(0.2), _OLIVEIRA_CV,
                    transforms=["sex one-hot encoded", "the compilation's 500-row sample (RandomState 4567)"])
    if k == "AFN":
        df = csv(data, sep="\t", header=None)
        return Rows(df.iloc[:, :5], df.iloc[:, 5], names, random(0.2), _OLIVEIRA_CV)
    if k == "BOH":   # StatLib text: 22 header lines, each record wrapped over two lines
        vals = np.array(" ".join(data.decode("latin-1").splitlines()[22:]).split(), dtype=float).reshape(-1, 14)
        return Rows(vals[:, :13], vals[:, 13], names, random(0.2), _OLIVEIRA_CV)
    if k == "CCP":   # the compilation reads Sheet1 of Folds5x2_pp.xlsx
        import zipfile
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            member = next(n for n in z.namelist() if n.endswith("Folds5x2_pp.xlsx"))
            df = _xls(z.read(member))
        return Rows(df[["AT", "V", "AP", "RH"]], df["PE"], names, random(0.2), _OLIVEIRA_CV)
    if k == "CPU":   # vendor and model dropped; PRP is an input, ERP the target
        df = csv(data, header=None)
        return Rows(df.iloc[:, 2:9], df.iloc[:, 9], ["MYCT", "MMIN", "MMAX", "CACH", "CHMIN", "CHMAX", "PRP"],
                    random(0.2), _OLIVEIRA_CV)
    if k == "CST":
        df = _xls(data)
        return Rows(df.iloc[:, :8], df.iloc[:, 8], names, random(0.2), _OLIVEIRA_CV)
    if k in ("ENC", "ENH"):
        df = _xls(data).dropna(how="all")
        return Rows(df.iloc[:, :8], df["Y2" if k == "ENC" else "Y1"], names[:8], random(0.2), _OLIVEIRA_CV)
    if k == "FFR":
        df = csv(data)
        feats = [c for c in df.columns if c not in ("month", "day", "area")]
        return Rows(df[feats], df["area"], feats, random(0.2), _OLIVEIRA_CV)
    if k == "OZO":   # the compilation's mlbench file, NaN -> column mean, LAST column (V13) as run by the compilers
        df = csv(data, header=None, na_values=["NA"])
        df = df.fillna(df.mean())
        return Rows(df.iloc[:, :12], df.iloc[:, 12], [f"V{i}" for i in range(1, 13)], random(0.2), _OLIVEIRA_CV,
                    meta={"target": "V13 (visibility), the column the compilers' script fits"},
                    transforms=["missing values replaced by the column mean (the compilers' preprocessing)"])
    if k in ("WIR", "WIW"):
        df = csv(data, sep=";")
        feats = [c for c in df.columns if c != "quality"]
        return Rows(df[feats], df["quality"], feats, random(0.2), _OLIVEIRA_CV)
    if k == "YAC":
        vals = np.loadtxt(io.BytesIO(data))
        return Rows(vals[:, :6], vals[:, 6], names[:6], random(0.2), _OLIVEIRA_CV)
    if k == "DOW":
        tr, te = _xls(data, sheet_name="TrainingData"), _xls(data, sheet_name="TestData")
        feats = [c for c in tr.columns if str(c).lower() != "y"]
        return Rows(tr[feats], tr.iloc[:, -1], [str(c) for c in feats], GIVEN,
                    "given: the organiser's TrainingData / TestData worksheets", x_test=te[feats], y_test=te.iloc[:, -1])
    if k == "TOW":   # Vladislavleva's box rule is not reproducible on the published file: no split
        df = csv(data, sep="\t")
        feats = [c for c in df.columns if c != "towerResponse"]
        return Rows(df[feats], df["towerResponse"], feats, NONE,
                    "none: the authors' extrapolation rule cannot be reproduced on the published file")
    if k in ("PPB", "LD50", "BIOAV"):   # the dataset authors' protocol: a fresh random 70/30 split per run
        df = csv(data)
        return Rows(df.iloc[:, :-1], df.iloc[:, -1], [f"x{i + 1}" for i in range(df.shape[1] - 1)], random(0.3),
                    "random 0.3: the dataset authors draw a fresh 70/30 split per run")
    if k == "MSD":
        raise SkipDataset("MSD: 515,345 rows x 90 inputs (375 MB as float64) left out for its size (srbf 2026-11-04)")
    raise SkipDataset(f"{k}: no public data")


def _hl_partition(split: str, n: int):
    import re
    m = re.search(r"training rows \[(\d+), (\d+)\).*test rows \[(\d+), (\d+)\)", split)
    a, b, c, d = map(int, m.groups())
    if (a, d) != (0, n) or b != c:
        raise ValueError(f"HL partition {m.groups()} does not cover {n} rows")
    return b


def _one_hot(df: pd.DataFrame, cols: list[str]):
    """Inputs in order, a non-numeric (factor) column as one indicator column per level, as HL's factor variables."""
    parts, names = [], []
    for c in cols:
        s = df[c]
        if pd.api.types.is_numeric_dtype(s):
            parts.append(s.to_numpy(dtype=float)[:, None])
            names.append(c)
        else:
            for level in sorted(s.astype(str).unique()):
                parts.append((s.astype(str) == level).to_numpy(dtype=float)[:, None])
                names.append(f"{c}={level}")
    return np.hstack(parts), names


@reader("fresh-46", "heuristiclab-real-world", "HeuristicLab regression instances (real-world data)")
def heuristiclab(rec, prob, _set):
    import re
    k = rec["id"]
    df = pd.read_csv(io.BytesIO(checked(rec["file"], rec["sha256"])), sep=None, engine="python", encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    if k.startswith(("CF", "Nvh", "Temp", "Wear")):   # MibaFriction: HL's allowed inputs, its single target
        allowed = [c.strip() for c in re.search(r"allowed inputs \(\d+\): ([^.]*)", rec["notes"]).group(1).split(",")]
        target = df.columns[-1]
        x, feats = _one_hot(df, allowed)
        transforms = ["factor inputs (letter codes) as one indicator column per level, as HL's factor variables"]
    else:
        target = {"Chemical-I": "y", "Housing": "MEDV", "Powermeter": "Power [W]", "SARCOS": "y1",
                  "Tower": "towerResponse"}[k]
        feats = [c for c in df.columns if c != target and c != "Date" and not (k == "SARCOS" and c.startswith("y"))]
        x, transforms = df[feats].to_numpy(dtype=float), []
    y = df[target].to_numpy(dtype=float)
    cut = _hl_partition(rec["split"], len(y))
    return Rows(x[:cut], y[:cut], feats, GIVEN, "given: HL's fixed contiguous training / test partition",
                x_test=x[cut:], y_test=y[cut:], transforms=transforms)


_TARS: dict = {}


def _member(entry: dict) -> bytes:
    """A member file's bytes: its extracted path, else its tar or zip member; checked against its sha256."""
    if entry.get("path") and paths.input_file(entry["path"]).exists():
        return checked(entry["path"], entry.get("sha256"))
    if entry.get("tar"):
        import tarfile
        if entry["tar"] not in _TARS:
            _TARS[entry["tar"]] = tarfile.open(paths.input_file(entry["tar"]))
        data = _TARS[entry["tar"]].extractfile(entry["tar_member"]).read()
    else:
        import zipfile
        with zipfile.ZipFile(paths.input_file(entry["zip"])) as z:
            data = z.read(entry["zip_member"])
    if entry.get("sha256") and sha256(data) != entry["sha256"]:
        raise ValueError(f"{entry}: sha256 mismatch")
    return data


def _ffx(rec):
    """FFX's transposed whitespace files: *_in (one line per input) and *_out (one line), for train and test."""
    m = rec["member_files"]
    pick = {(r, kind): e for name, e in m.items() for r in ("train", "test") for kind in ("in", "out")
            if name.startswith(r + "_") and name.rsplit(".", 1)[0].endswith("_" + kind)}
    load = lambda r, kind: np.loadtxt(io.BytesIO(_member(pick[(r, kind)])), ndmin=2)  # noqa: E731
    x, y = load("train", "in").T, load("train", "out").reshape(-1)
    xt, yt = load("test", "in").T, load("test", "out").reshape(-1)
    names = next((e for n, e in m.items() if n.endswith("varnames.csv") or n.endswith("varnames.txt")), None)
    feats = _member(names).decode().split() if names else list(rec["features"])
    if len(feats) == x.shape[1] + 1:   # some varnames files end with the target's name
        feats = feats[:-1]
    if (len(y), len(yt)) != (rec["n_rows_train"], rec["n_rows_test"]):
        raise ValueError(f"{rec['id']}: {len(y)}/{len(yt)} rows, record {rec['n_rows_train']}/{rec['n_rows_test']}")
    return x, y, xt, yt, feats


@reader("fresh-143", "ffx-high-dimensional-circuits", "FFX high-dimensional circuit problems")
def ffx_high(rec, prob, _set):
    x, y, xt, yt, feats = _ffx(rec)
    return Rows(x, y, feats, GIVEN, "given: the published train / test files (every 4th point by sorted y is test)",
                x_test=xt, y_test=yt)


@reader("fresh-142", "ffx-cmos-ota", "FFX CMOS OTA circuit targets")
def ffx_ota(rec, prob, _set):
    x, y, xt, yt, feats = _ffx(rec)
    return Rows(x, y, feats, GIVEN, "given: the published train / test files (two separate DOEs)",
                x_test=xt, y_test=yt)


@reader("fresh-62", "falling-balls", "Free-falling balls with air resistance")
def falling_balls(rec, prob, _set):
    f = rec["files"]
    tr = csv(checked(f["train"]["path"], f["train"]["sha256"]), header=None)
    te = csv(checked(f["test"]["path"], f["test"]["sha256"]), header=None)
    return Rows(tr.iloc[:, 0], tr.iloc[:, 1], ["t"], GIVEN,
                "given: the authors' time split of drop 1 (t <= 2 s fits, t > 2 s tests)",
                x_test=te.iloc[:, 0], y_test=te.iloc[:, 1])


@reader("fresh-35", "gp-gomea-real-world", "GP-GOMEA real-world set")
def gp_gomea(rec, prob, _set):
    vals = np.loadtxt(io.BytesIO(checked(rec["file"], rec["sha256"])), ndmin=2)
    names = list(rec.get("feature_names") or [f"x{i + 1}" for i in range(vals.shape[1] - 1)])
    return Rows(vals[:, :-1], vals[:, -1], names, random(0.25),
                "random 0.25: the authors redraw a 50 % train / 25 % validation / 25 % test split per repetition "
                "(the validation quarter belongs to the method's fitting rows)")


def _slim(data: bytes):
    df = csv(data)   # header row of integer indices, last column = target
    return df.iloc[:, :-1], df.iloc[:, -1], [f"d{i + 1}" for i in range(df.shape[1] - 1)]


_GSGP_SPLIT = "random 0.3: the set's defining paper draws a fresh 70/30 split per run"


@reader("critic-3", "gsgp-real-world", "GSGP real-world regression set")
def gsgp(rec, prob, _set):
    k, data = rec["id"], checked(rec["file"], rec["sha256"])
    names = list(rec.get("feature_names") or [])
    if k == "airfoil":
        df = csv(data, sep="\t", header=None)
        return Rows(df.iloc[:, :5], df.iloc[:, 5], names, random(0.3), _GSGP_SPLIT)
    if k in ("bioav", "ppb", "ld50"):
        x, y, feats = _slim(data)
        return Rows(x, y, feats, random(0.3), _GSGP_SPLIT)
    if k == "concrete":
        df = _xls(data)
        return Rows(df.iloc[:, :8], df.iloc[:, 8], names[:8] or [str(c) for c in df.columns[:8]], random(0.3), _GSGP_SPLIT)
    if k == "parkinson":
        df = csv(data)
        feats = [c for c in df.columns if c not in ("subject#", "motor_UPDRS", "total_UPDRS")]
        return Rows(df[feats], df["total_UPDRS"], feats, random(0.3), _GSGP_SPLIT)
    if k == "slump":
        df = csv(data)
        feats = list(df.columns[1:8])
        return Rows(df[feats], df["SLUMP(cm)"], feats, random(0.3), _GSGP_SPLIT)
    raise SkipDataset(f"{k}: no reader")


def _lags(s: np.ndarray, lags: list[int]) -> np.ndarray:
    """Columns s[t - l] for each lag l, zero before the series starts."""
    out = np.zeros((len(s), len(lags)))
    for j, lag in enumerate(lags):
        out[lag:, j] = s[:-lag]
    return out


@reader("fresh-48", "mcdermott-predictive-modelling", "McDermott et al. predictive-modelling sets")
def mcdermott(rec, prob, _set):
    k = rec["id"]
    if k in ("drug-bioavailability", "median-oral-lethal-dose", "plasma-protein-binding"):
        x, y, feats = _slim(checked(rec["file"], rec["sha256"]))
        return Rows(x, y, feats, random(0.3), "random 0.3: the dataset authors draw a fresh 70/30 split per run")
    if k == "mackey-glass":   # x(t) from x(t-1), x(t-2), ..., x(t-128); zero before the start (Langdon & Banzhaf)
        s = np.loadtxt(io.BytesIO(checked(rec["file"], rec["sha256"])))[:, 1]
        lags = [1, 2, 4, 8, 16, 32, 64, 128]
        return Rows(_lags(s, lags), s, [f"x(t-{lag})" for lag in lags], NONE,
                    "none: the fitness is the error over all 1201 time points",
                    transforms=["inputs = the series at lags 1, 2, 4, ..., 128 (zero before t = 0)"])
    if k == "sunspots":       # s(n+1) from s(n), ..., s(n-11); 1700-1920 fits, 1921-1955 and 1956-1979 test
        d = np.loadtxt(io.BytesIO(checked(rec["file"], rec["sha256"])))
        d = d[(d[:, 0] >= 1700) & (d[:, 0] < 1980)]
        years, s = d[:, 0].astype(int), d[:, 1]
        idx = np.arange(12, len(s))                       # target index n+1 with 12 earlier values available
        x = np.stack([s[idx - 1 - j] for j in range(12)], axis=1)
        y, ty = s[idx], years[idx]
        fit = ty <= 1920
        out = []
        for lo, hi in ((1921, 1955), (1956, 1979)):
            te = (ty >= lo) & (ty <= hi)
            out.append(Rows(x[fit], y[fit], [f"s(n-{j})" for j in range(12)], GIVEN,
                            f"given: the authors' years split, 1700-1920 fits, {lo}-{hi} tests",
                            x_test=x[te], y_test=y[te], eq_id=f"sunspots/test-{lo}-{hi}",
                            subset=f"sunspots test {lo}-{hi}",
                            transforms=["inputs = the 12 previous yearly values (a tapped-delay line)"]))
        return out
    raise SkipDataset(f"{k}: not recordable as a regression dataset (no published data)")


@reader("fresh-125", "treloar-rubber", "Treloar rubber hyperelasticity")
def treloar(rec, prob, _set):
    # one problem per temperature and loading mode; the authors' split is ACROSS modes (UT+ET calibrate, PS
    # validates), which a one-input stretch-stress problem cannot hold, so each mode's curve is fitted whole
    df = csv(checked(rec["file"], rec["sha256"]))
    temp, mode = rec["id"].split("-")[1:3]
    s = df[(df["temperature_C"] == int(temp.rstrip("C"))) & (df["mode"] == mode)]
    if len(s) != rec["n_rows"]:
        raise ValueError(f"{rec['id']}: {len(s)} rows, record {rec['n_rows']}")
    return Rows(s["lambda"], s["P11_nominal_MPa"], ["lambda"], NONE,
                "none within a mode (the authors split across modes: UT+ET calibrate, PS validates)")


@reader("fresh-132", "ideasearch-pdf", "IdeaSearchFitter parton distributions (CT18NNLO)")
def ideasearch_pdf(rec, prob, _set):
    g = rec["files"]["regenerated_grid"]
    df = csv(checked(g["path"], g["sha256"]))
    flav = rec["id"].replace("ideasearch_pdf_", "")
    s = df[df["flavour"] == flav]
    tr, te = s[s["split"] == "train"], s[s["split"] != "train"]
    if (len(tr), len(te)) != (400, 20):
        raise ValueError(f"{rec['id']}: {len(tr)}/{len(te)} grid points")
    return Rows(tr[["x", "Q_GeV"]], tr["xf"], ["x", "Q"], GIVEN,
                "given: the authors' grids, Q in [2, 5] GeV fits, Q = 100 GeV tests (extrapolation)",
                x_test=te[["x", "Q_GeV"]], y_test=te["xf"],
                meta={"target": "x f(x, Q), CT18NNLO central member, regenerated on the authors' grid"})


def _whitespace_table(data: bytes, ncols: int) -> np.ndarray:
    """Numeric rows of a whitespace table with '#' comments, a header line and a BOM tolerated."""
    rows = []
    for line in data.decode("utf-8-sig").splitlines():
        tok = line.split("#", 1)[0].split()
        if len(tok) == ncols:
            try:
                rows.append([float(t) for t in tok])
            except ValueError:
                continue
    return np.array(rows)


@reader("fresh-3", "ai-descartes-real-data", "AI Descartes real-data laws")
def ai_descartes(rec, prob, _set):
    cols = list(rec["columns_in_file"])
    t = _whitespace_table(checked(rec["file"], rec["sha256"]), len(cols))
    if len(t) != rec["n_rows"]:
        raise ValueError(f"{rec['id']}: {len(t)} rows, record {rec['n_rows']}")
    feats = list(rec["features"])
    return Rows(t[:, [cols.index(f) for f in feats]], t[:, cols.index(rec["target"])], feats, NONE,
                "none: the authors fit every measured point and judge generalization against the axioms")


_DIMLESS_INPUTS = {"rayleigh_benard": ["h", "delta_T", "lambda", "g", "alpha", "v", "k"],
                   "keyhole": ["etaP", "Vs", "r0", "alpha", "rho", "cp", "Tl-T0"]}


@reader("critic-6", "dimensionless-learning", "Dimensionless-learning scaling laws (Xie et al. 2022)")
def dimensionless_learning(rec, prob, _set):
    df = pd.read_csv(io.BytesIO(checked(rec["file"], rec["sha256"])), encoding="utf-8-sig")
    feats = rec["features"]
    if isinstance(feats, str):   # 'same 7 input columns as <sibling> (...)'
        feats = _DIMLESS_INPUTS["rayleigh_benard" if rec["id"].startswith("rayleigh") else "keyhole"]
    else:
        feats = [f["name"] if isinstance(f, dict) else f for f in feats]
    x, y = columns(df, feats, rec["target_column"])
    return Rows(x, y, feats, random(0.2), "random 0.2: the authors' random 80/20 train/test split (no seed)")


_MAGIC_N = np.array([2, 8, 20, 28, 50, 82, 126, 184])
_MAGIC_Z = np.array([2, 8, 20, 28, 50, 82, 126])


def _valence(n: np.ndarray, magic: np.ndarray) -> np.ndarray:
    """n minus the closest magic number (signed), as the authors' code computes Nn and Np."""
    return n - magic[np.abs(n[:, None] - magic[None, :]).argmin(axis=1)]


@reader("critic-5", "nuclear-misr", "Nuclear binding energies and charge radii (MISR)")
def nuclear_misr(rec, prob, _set):
    if not rec["id"].startswith("misr"):
        raise SkipDataset(f"{rec['id']}: no published data or membership (rebuilt from AME2020 in the record only)")
    f = _role_files(rec)
    target = rec["target"]["name"]

    def load(role):
        df = csv(checked(f[role]["path"], f[role]["sha256"]))
        df = df[(df["Z"] >= 12) & (df["Z"] <= 50)]          # the paper's window 12 <= Z <= 50
        n, z = df["N"].to_numpy(dtype=float), df["Z"].to_numpy(dtype=float)
        x = np.column_stack([n, z, df["A"], df["S"], df["P"], _valence(n, _MAGIC_N), _valence(z, _MAGIC_Z)])
        return x, df[target].to_numpy(dtype=float)
    x, y = load("train")
    xt, yt = load("test")
    return Rows(x, y, ["N", "Z", "A", "I", "P", "Nn", "Np"], GIVEN,
                "given: the published train / test files, inside the paper's window 12 <= Z <= 50",
                x_test=xt, y_test=yt,
                transforms=["rows with 12 <= Z <= 50 (the paper's window)",
                            "Nn, Np = N, Z minus the closest magic number (signed, as the authors' code)"])


@reader("fresh-166", "symbolfit-lhc-spectra", "SymbolFit LHC background spectra")
def symbolfit(rec, prob, _set):
    import ast
    src = checked(rec["file"], rec["sha256"]).decode()
    lists = {}
    for line in src.splitlines():
        name, _, rest = line.partition("=")
        if name.strip() in ("x", "y") and rest.strip().startswith("["):
            lists[name.strip()] = np.array(ast.literal_eval(rest.strip()), dtype=float)
    x, y = lists["x"], lists["y"]
    fit, held = np.array(rec["split"]["fit_idx"]), np.array(rec["split"]["held_out_idx"])
    return Rows(x[fit], y[fit], ["x"], GIVEN, "given: the authors' blinded signal window is the test (deterministic)",
                x_test=x[held], y_test=y[held], meta={"target": "differential cross-section (bin centres x)"})


@reader("fresh-120", "langlaw-perovskites", "LangLaw perovskite datasets")
def langlaw(rec, prob, _set):
    df = csv(checked(rec["file"], rec["sha256"]))
    feats = [f["name"] if isinstance(f, dict) else f for f in rec["features"]]
    target = rec["target"].split(" ")[0]
    df = df.dropna(subset=[*feats, target])               # the authors' loader drops rows with a missing value
    x, y = columns(df, feats, target)
    if rec["id"].startswith("hierbosss"):
        return Rows(x, y, feats, random(0.1), "random 0.1: the authors' 5 independent random 90/10 splits")
    return Rows(x, y, feats, random(0.2), "random 0.2: the authors' shuffled 5-fold CV",
                transforms=["rows with a missing input or target dropped (the authors' loader)"])


_PENDULUM_TARGETS = {"eql-dpk-y1": lambda a, b: np.cos(a), "eql-dpk-y2": lambda a, b: np.cos(a) + np.cos(a + b),
                     "eql-dpk-y3": lambda a, b: np.sin(a), "eql-dpk-y4": lambda a, b: np.sin(a) + np.sin(a + b)}


@reader("fresh-152", "eql-double-pendulum", "EQL double-pendulum kinematics")
def eql_pendulum(rec, prob, _set):
    t = np.loadtxt(io.BytesIO(checked(rec["file"], rec["sha256"])), comments="%")
    traj, o = t[:, 0], t[:, 2:4]
    f = _PENDULUM_TARGETS[rec["id"]]
    tr, te = o[traj == 0], np.angle(np.exp(1j * o[traj == 1]))   # test angles wrapped to [-pi, pi] (the authors)
    return Rows(tr, f(tr[:, 0], tr[:, 1]), ["x1", "x2"], GIVEN,
                "given: trajectory 0 fits, trajectory 1 tests (the authors' extrapolation set)",
                x_test=te, y_test=f(te[:, 0], te[:, 1]), formula_names=["x1", "x2"],
                transforms=["target computed from the measured angles by the authors' formula",
                            "test angles wrapped to [-pi, pi]"])


@reader("fresh-2", "pysr-empiricalbench", "PySR EmpiricalBench")
def pysr_empiricalbench(rec, prob, _set):
    k = rec["id"]
    note = "none: each algorithm is fed the entire dataset; the test is the output expression"
    if k in ("kepler", "bode"):
        checked(rec["file"].split(" (")[0], rec["sha256"])
        d = rec["data_inline"]
        if k == "kepler":
            return Rows(d["a"], d["target_P"], ["a"], NONE, note)
        return Rows(d["n"], np.log(d["target_a"]), ["n"], NONE, note,
                    transforms=["target = ln(a) as the authors evaluate it; n = -1000 stands for -inf (Mercury)"])
    df = csv(checked(rec["file"], rec["sha256"]))
    if k == "leavitt":
        return Rows(10 ** df["logP"].to_numpy(), df["M"], ["P"], NONE, note, transforms=["P = 10^logP (the authors' input)"])
    return Rows(df["D"], df["v"], ["D"], NONE, note)


@reader("fresh-50", "shape-constrained-real-world", "Shape-constrained SR real-world set")
def shape_constrained(rec, prob, _set):
    f = rec["files"]
    cols = list(rec["columns"])
    tr = csv(checked(f["train"]["path"], f["train"]["sha256"]), header=None, names=cols)
    te = csv(checked(f["test"]["path"], f["test"]["sha256"]), header=None, names=cols)
    feats, target = list(rec["features"]), cols[-1]
    x, y = columns(tr, feats, target)
    xt, yt = columns(te, feats, target)
    return Rows(x, y, feats, GIVEN, "given: the authors' published train / test files", x_test=xt, y_test=yt)


@reader("done-16", "gecco-2022-covid-ny", "GECCO 2022 real-world: COVID-19 New York")
def gecco_covid(rec, prob, _set):
    f = rec["files"]
    target = {"cases": "value_cases", "deaths": "value_deaths", "hosp": "value_hosp"}[rec["id"].split("_")[3]]
    tr = csv(checked(f["train"]["path"], f["train"]["sha256"]), index_col="date")
    te = csv(checked(f["test"]["path"], f["test"]["sha256"]), index_col="date")
    feats = list(rec["feature_names"])
    x, y = columns(tr, feats, target)
    xt, yt = columns(te, feats, target)
    return Rows(x, y, feats, GIVEN, "given: the published _train / _test file pair", x_test=xt, y_test=yt)


@reader("done-17", "gecco-2023-performance", "GECCO 2023 competition Track 1 (performance)")
def gecco_2023(rec, prob, _set):
    f = _role_files(rec)
    tr = csv(checked(f["train"]["path"], f["train"]["sha256"]))
    te = csv(checked(f["test"]["path"], f["test"]["sha256"]))
    feats = [c for c in tr.columns if c != "y"]
    x, y = columns(tr, feats, "y")
    xt, yt = columns(te, feats, "y")
    return Rows(x, y, feats, GIVEN, "given: the published dataset_k.csv / dataset_k_test.csv", x_test=xt, y_test=yt)


@reader("fresh-122", "materials-laws-multi-agent", "Multi-agent materials law datasets")
def materials_multi_agent(rec, prob, _set):
    if rec["id"] != "vickers_hardness_tantardini61":
        why = "no published data" if not rec.get("file") else "inputs are not published (derived by the prep)"
        raise SkipDataset(f"{rec['id']}: {why}")
    df = pd.read_csv(io.BytesIO(checked(rec["file"], rec["sha256"])), sep=r"\s+")
    feats = list(rec["feature_names"])
    x, y = columns(df, feats, "property")
    return Rows(x, y, feats, NONE, "none: SISSO is fitted on all 61 compounds (CV only selects the descriptor dimension)")


@reader("fresh-140", "evogp-uci", "EvoGP regression datasets")
def evogp(rec, prob, _set):
    k = rec["id"]
    note = "none: EvoGP fits and scores on all rows"
    if k == "daily_demand_forecasting_orders":
        df = csv(checked(rec["file"], rec["sha256"]))
        return Rows(df.iloc[:, :-1], df.iloc[:, -1], list(df.columns[:-1]), NONE, note)
    if k == "auto_mpg":
        df = csv(checked(rec["file"], rec["sha256"])).dropna(subset=["horsepower"])
        feats = [c for c in df.columns if c not in ("car_name", "mpg")]
        return Rows(df[feats], df["mpg"], feats, NONE, note, transforms=["the 6 rows without horsepower dropped (EvoGP)"])
    import tarfile   # california housing as scikit-learn renders it (EvoGP loads it from scikit-learn)
    with tarfile.open(fileobj=io.BytesIO(checked(rec["file"], rec["sha256"]))) as t:
        raw = np.loadtxt(io.BytesIO(t.extractfile(next(m for m in t.getmembers() if m.name.endswith("cal_housing.data"))).read()),
                         delimiter=",")
    raw = raw[:, [8, 7, 2, 3, 4, 5, 6, 1, 0]]
    y, x = raw[:, 0] / 100000.0, raw[:, 1:].copy()
    x[:, 2] /= x[:, 5]
    x[:, 3] /= x[:, 5]
    x[:, 5] = x[:, 4] / x[:, 5]
    return Rows(x, y, ["MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population", "AveOccup", "Latitude", "Longitude"],
                NONE, note, transforms=["scikit-learn's fetch_california_housing rendering (per-household averages, "
                                        "target in units of 100,000)"])


@reader("fresh-149", "tisr-air-density", "TiSR air-mixture density data")
def tisr_air(rec, prob, _set):
    df = csv(checked(rec["file"], rec["sha256"]))
    cols = rec["columns_used"]
    feats = list(cols["features"])
    x, y = columns(df, feats, cols["target"])
    return Rows(x, y, feats, NONE, "none: the authors fit on the whole simulation grid / all experimental points")


@reader("fresh-39", "shadow-economies", "Shadow Economies")
def shadow_economies(rec, prob, _set):
    if rec["id"] == "PER_CAPITA_GDP":
        raise SkipDataset("PER_CAPITA_GDP: the target is not a column of the published file")
    df = _xls(checked(rec["file"], rec["sha256"]), sheet_name="Sheet1").iloc[1:]   # row 1 holds descriptions
    df.columns = [str(c).replace(" ", "") for c in df.columns]
    df = df[pd.to_numeric(df["SHADOW_PER"], errors="coerce").notna()]               # the 260 labelled rows
    feats = list(rec["feature_names"])
    x, y = columns(df.apply(pd.to_numeric, errors="coerce"), feats, rec["id"])
    return Rows(x, y, feats, NONE, "none: the chapter fits on all labelled rows; no split is defined")


@reader("fresh-63", "bms-real-data", "Bayesian machine scientist real datasets")
def bms_real(rec, prob, _set):
    df = csv(checked(rec["file"], rec["sha256"]))
    feats = list(rec["feature_names"])
    if rec["id"] == "LogYe":
        x, y = df[feats].to_numpy(dtype=float), np.log(df["rec"].to_numpy(dtype=float))
        transforms = ["target = ln(rec), as the authors' read_data"]
    else:
        target = {"RDFunding": "FinancialSuccessRate", "Trepat": "Sxx"}[rec["id"]]
        x, y = columns(df, feats, target)
        transforms = []
    return Rows(x, y, feats, NONE,
                "none: the authors' protocol is leave-one-out, which leaves one test row per run (no test error "
                "with a variance); every row is fitted", transforms=transforms)


@reader("critic-4", "superconductor-tc", "Superconducting Tc (Allen-Dynes data)")
def superconductor_tc(rec, prob, _set):
    if rec["id"] == "AllenDynes_Tc":   # data.csv rows 1-29 fit, rows 30-38 are the authors' test materials
        df = csv(checked(rec["file"], rec["sha256"]))
        x, y = df[["w", "l", "m"]].to_numpy(dtype=float), df["t_c"].to_numpy(dtype=float)
        return Rows(x[:29], y[:29], ["wlog", "lambda", "mu"], GIVEN,
                    "given: the authors' fixed split by material (29 fit, 9 test)", x_test=x[29:38], y_test=y[29:38])
    df = pd.read_csv(io.BytesIO(checked(rec["file"], rec["sha256"])), sep=r"\s+")
    feats = ["wlog", "w1", "w2", "wph", "eta", "lambd", "mus"]
    x, y = columns(df, feats, "Tc")
    return Rows(x, y, feats, NONE, "none: leave-one-out over the 29 rows (one test row per run) and no test set")


@reader("critic-7", "oxygen-ion-conductors", "Oxygen-ion conductor database")
def oxygen_ion(rec, prob, _set):
    df = pd.read_csv(io.BytesIO(checked(rec["file"], rec["sha256"])), sep="\t")
    target = rec["target"]["column"]
    feats = list(rec["feature_names"])
    x, y = columns(df, feats, target)
    test = df["Fold1"].to_numpy() == 1
    return Rows(x[~test], y[~test], feats, GIVEN,
                "given: the authors' fixed outer fold 1 (a contiguous block of the database) tests, folds 2-5 fit",
                x_test=x[test], y_test=y[test])


@reader("fresh-137", "drsr-debcat", "DRSR DEBCat mass-luminosity")
def drsr_debcat(rec, prob, _set):
    df = csv(checked(rec["derived_file"], rec["derived_sha256"]))
    if rec["id"].endswith("log"):
        x, y = (df["logM"] + 0.9682) / 2.4039, (df["logL"] + 2.313) / 7.5
        transforms = ["x = (log M + 0.9682)/2.4039, y = (log L + 2.313)/7.5 (the authors' min-max scaling)"]
    else:
        x, y, transforms = df["M_Msun"], df["L_Lsun"], ["M = 10^logM, L = 10^logL (solar units)"]
    return Rows(x, y, ["M"], NONE, "none: the search fitness is computed in-sample on all 324 stars",
                transforms=["both components of each catalogued system with a luminosity (324 stars)", *transforms])


@reader("fresh-32", "white-2013-proposed", "White et al. 2013 proposed SR set")
def white_2013(rec, prob, _set):
    if rec["id"] == "GP-Challenge":
        raise SkipDataset("GP-Challenge: a learning-to-rank task over 54 proteins (pooled 82,965 rows); "
                          "the per-protein problems are in gp-challenge-protein")
    data = checked(rec["files"]["data_workbook"]["path"], rec["files"]["data_workbook"]["sha256"])
    tr, te = _xls(data, sheet_name="TrainingData"), _xls(data, sheet_name="TestData")
    feats = [c for c in tr.columns if str(c).lower() != "y"]
    return Rows(tr[feats], tr.iloc[:, -1], [str(c) for c in feats], GIVEN,
                "given: the organiser's TrainingData / TestData worksheets", x_test=te[feats], y_test=te.iloc[:, -1])


@reader("fresh-38", "dow-emulator", "Emulator problem (Dow)")
def dow_emulator(rec, prob, _set):
    if not rec.get("file"):
        raise SkipDataset(f"{rec['id']}: the values were never recovered from any source")
    df = pd.read_csv(io.BytesIO(checked(rec["file"], rec["sha256"])), sep=";", decimal=",")
    feats = [f"x{i}" for i in range(1, 11)]
    x, y = columns(df, feats, "y2")
    return Rows(x[:320], y[:320], feats, GIVEN,
                "given: the authors' DOE block (rows 1-320) fits, the random block (rows 321-595) tests",
                x_test=x[320:], y_test=y[320:])


@reader("fresh-4", "ai-hilbert-real-data", "AI Hilbert real data")
def ai_hilbert(rec, prob, _set):
    df = csv(checked(rec["file"].split(" (")[0], rec["sha256"]))
    note = "none: all rows are fitted; success is judged symbolically"
    if rec["id"] == "kepler_binary_stars":
        return Rows(df[["m1_solar", "m2_solar", "d_AU"]], df["p_years"], ["m1", "m2", "d"], NONE, note)
    return Rows(df["v_m_per_s"], df["frac_shift_1e-15"], ["v"], NONE, note)


@reader("fresh-51", "nikuradse", "Nikuradse rough-pipe friction")
def nikuradse(rec, prob, _set):
    import gzip
    df = csv(gzip.decompress(fetch(rec["url"], rec["sha256"], ".tsv.gz")), sep="\t")
    feats = [c for c in df.columns if c != "target"]
    x, y = columns(df, feats, "target")
    return Rows(x, y, feats, NONE, "none: PMLB ships all rows without a split")


@reader("fresh-53", "esr-cosmology", "ESR cosmology datasets")
def esr_cosmology(rec, prob, _set):
    if rec["id"] == "cc-hubble":
        t = np.loadtxt(io.BytesIO(checked(rec["file"], rec["sha256"])))
        return Rows(t[:, 0], t[:, 1], ["z"], NONE, "none: every function is fitted and scored on all 32 points")
    raise SkipDataset("pantheon-plus: not y = f(x) (ESR's H(z)^2 enters through the luminosity-distance integral)")


@reader("done-31", "llm-sr-stress-strain", "LLM-SR stress-strain (Al 6061-T651)")
def llm_sr_stressstrain(rec, prob, _set):
    f = rec["files"]
    load = lambda role: csv(checked(f[role]["path"], f[role]["sha256"]))  # noqa: E731
    tr = load("train")
    x, y = columns(tr, ["strain", "temp"], "stress")
    out = []
    for role, label in (("test_id", "in-domain"), ("test_ood", "out-of-domain")):
        xt, yt = columns(load(role), ["strain", "temp"], "stress")
        out.append(Rows(x, y, ["strain", "temp"], GIVEN, f"given: the published train.csv / {role}.csv ({label} test)",
                        x_test=xt, y_test=yt, eq_id=f"stressstrain/{role}", subset=f"stress-strain {label}"))
    return out


@reader("interrupted-0", "llm-sr-bactgrow", "LLM-SR E. coli growth rate")
def llm_sr_bactgrow(rec, prob, _set):
    f = rec["files"]
    load = lambda role: csv(checked(f[role]["path"], f[role]["sha256"]))  # noqa: E731
    feats = list(rec["features"])
    x, y = columns(load("train"), feats, rec["target"])
    out = []
    for role, label in (("test_id", "in-domain"), ("test_ood", "out-of-domain")):
        xt, yt = columns(load(role), feats, rec["target"])
        out.append(Rows(x, y, feats, GIVEN, f"given: the published train.csv / {role}.csv ({label} test)",
                        x_test=xt, y_test=yt, eq_id=f"bactgrow/{role}", subset=f"bactgrow {label}"))
    return out


@reader("fresh-116", "torchsisso-specific-energy", "Specific energy of organic compounds (TorchSISSO)")
def specific_energy(rec, prob, _set):
    f = rec["files"]
    tr = pd.read_csv(io.BytesIO(checked(f["train"]["path"], f["train"]["sha256"])), sep="\t")
    te = csv(checked(f["test"]["path"], f["test"]["sha256"]))
    feats = [c for c in tr.columns if c not in ("Index", "Target")]
    x, y = columns(tr, feats, "Target")
    xt, yt = columns(te, feats, "Target")          # the test file restricted to the training columns, in their order
    return Rows(x, y, feats, GIVEN, "given: train.dat (115 paraquinones) fits, redox_test_pade.csv (1000 quinones) tests",
                x_test=xt, y_test=yt)


@reader("fresh-117", "sisso-octet-binaries", "SISSO octet binaries (rock-salt vs zinc-blende)")
def sisso_octet(rec, prob, _set):
    df = csv(checked(rec["file"], rec["sha256"]))
    feats = list(rec["feature_names"])
    x, y = columns(df, feats, "dE_eV")
    return Rows(x, y, feats, random(0.1), "random 0.1: the definers' repeated leave-10%-out CV")


@reader("fresh-123", "hierbosss-single-atom", "HierBOSSS single-atom catalysis")
def hierbosss_single_atom(rec, prob, _set):
    df = csv(checked(rec["derived_file"], rec["derived_file_sha256"]))
    feats = ["dH_sub_eV", "dH_f_ox_bulk_eV", "dE_vac_eV"]
    x, y = columns(df, feats, "BE_eV")
    return Rows(x, y, feats, random(0.1), "random 0.1: HierBOSSS's 25 random 90/10 train/test splits")


@reader("done-18", "gecco-2023-montreal-bikes", "GECCO 2023 Track 2 (Montreal bike lanes)")
def montreal_bikes(rec, prob, _set):
    raise SkipDataset("MontrealBikeLane: the organisers fix no target layout and no split (judged on write-ups)")


@reader("fresh-126", "llm-pysr-battery", "LLM-PySR battery cycle life")
def battery(rec, prob, _set):
    f = rec["files"]
    load = lambda role: csv(checked(f[role]["path"], f[role]["sha256"]))  # noqa: E731
    feats, target = list(rec["feature_names"]), "log10_cycle_life"
    fit = pd.concat([load("train"), load("validation")])   # Severson's 41 training cells (the validation is the method's)
    x, y = columns(fit, feats, target)
    out = []
    for role, label in (("test_id", "in-distribution"), ("test_ood", "out-of-distribution")):
        xt, yt = columns(load(role), feats, target)
        out.append(Rows(x, y, feats, GIVEN, f"given: train + validation fit (Severson's 41 cells), {role}.csv tests",
                        x_test=xt, y_test=yt, eq_id=f"battery/{role}", subset=f"battery {label}",
                        transforms=["inputs z-scored by the authors (files as published)"]))
    return out


@reader("fresh-146", "gplearn-diabetes", "gplearn documentation example (diabetes)")
def gplearn_diabetes(rec, prob, _set):
    import gzip
    entries = {e["path"].rsplit("/", 1)[1]: e for e in rec["files"]}
    x = np.loadtxt(io.BytesIO(gzip.decompress(checked(rec["file"], rec["sha256"]))))
    t = entries["diabetes_target.csv.gz"]
    y = np.loadtxt(io.BytesIO(gzip.decompress(checked(t["path"], t["sha256"]))))
    x = (x - x.mean(axis=0)) / (x.std(axis=0) * np.sqrt(len(x)))   # load_diabetes(scaled=True), as gplearn uses it
    s = entries["diabetes_gplearn_split.json"]
    sp = json.loads(checked(s["path"], s["sha256"]))
    tr, te = np.array(sp["train"]), np.array(sp["test"])
    return Rows(x[tr], y[tr], list(rec["features"]), GIVEN,
                "given: gplearn's example split (RandomState(0) permutation, first 300 fit, 142 test)",
                x_test=x[te], y_test=y[te], transforms=["features scaled as load_diabetes(scaled=True)"])


@reader("fresh-156", "eql-xray-kalpha2", "EQL X-ray transition energies")
def eql_xray(rec, prob, _set):
    df = csv(checked(rec["file"], rec["sha256"]))
    tr, te = df[df["role"] == "trainval"], df[df["role"] == "extrapolation"]
    return Rows(tr["x"], tr["y"], ["Z/100"], GIVEN,
                "given: Z in [10, 91] fits (the authors' train + validation), Z in [92, 100] tests (extrapolation)",
                x_test=te["x"], y_test=te["y"], transforms=["x = Z/100, y = E/1e5 eV (the authors' scaling)"])


@reader("fresh-164", "black-hole-spiral-plane", "Black-hole mass scaling in spiral galaxies")
def black_hole_plane(rec, prob, _set):
    t = rec["files"]["table1_canonical_csv"]
    df = csv(checked(t["path"], t["sha256"]))
    df = df[df["in_preferred_fit"].astype(str) == "True"]       # the authors' 41 galaxies (without the Milky Way)
    feats = list(rec["feature_names"])
    x, y = columns(df, feats, "M_bullet_dex")
    return Rows(x, y, feats, NONE, "none: the authors fit all 41 galaxies in-sample")


@reader("fresh-36", "dow-chemical-evostar-2010", "Dow Chemical (EvoStar 2010 competition)")
def dow_chemical(rec, prob, _set):
    data = checked(rec["file"], rec["sha256"])
    tr, te = _xls(data, sheet_name="TrainingData"), _xls(data, sheet_name="TestData")
    feats = list(rec["feature_names"])
    x, y = columns(tr, feats, "y")
    xt, yt = columns(te, feats, "y")
    return Rows(x, y, feats, GIVEN, "given: the organiser's TrainingData / TestData worksheets", x_test=xt, y_test=yt)


@reader("fresh-37", "dow-distillation-tower", "Tower (Dow distillation tower)")
def dow_tower(rec, prob, _set):
    df = csv(checked(rec["file"], rec["sha256"]), sep="\t")
    feats = list(rec["feature_names"])
    x, y = columns(df, feats, "towerResponse")
    return Rows(x, y, feats, NONE, "none: the website prescribes no split and the paper's box rule cannot be "
                                   "reproduced on the published file")


@reader("fresh-52", "sparc-radial-acceleration", "Radial acceleration relation (SPARC)")
def sparc_rar(rec, prob, _set):
    f = rec["files"]
    gal = checked(f["SPARC_Lelli2016c.mrt"]["path"], f["SPARC_Lelli2016c.mrt"]["sha256"]).decode().splitlines()
    keep = set()
    for line in gal[98:]:
        tok = line.split()
        if len(tok) >= 18 and float(tok[5]) >= 30 and int(tok[17]) != 3:    # Inc >= 30 deg and quality Q != 3
            keep.add(tok[0])
    rows = []
    for line in checked(rec["file"], rec["sha256"]).decode().splitlines()[25:]:
        tok = line.split()
        if len(tok) == 10 and tok[0] in keep:
            r, vobs, evobs, vgas, vdisk, vbul = map(float, tok[2:8])
            if evobs / vobs < 0.10:
                rows.append((r, vobs, vgas, vdisk, vbul))
    r, vobs, vgas, vdisk, vbul = np.array(rows).T
    unit = 1e6 / 3.0856775814913673e19 / 1e-10          # (km/s)^2 / kpc -> 1e-10 m s^-2
    g_obs = vobs ** 2 / r * unit
    g_bar = (vgas * np.abs(vgas) + 0.5 * vdisk * np.abs(vdisk) + 0.7 * vbul * np.abs(vbul)) / r * unit
    if len(g_obs) != rec["n_rows"]:
        raise ValueError(f"sparc: {len(g_obs)} points, record {rec['n_rows']}")
    return Rows(g_bar, g_obs, ["g_bar"], NONE, "none: all 2696 points are used for fitting and model selection",
                transforms=["the authors' selection (Q != 3, Inc >= 30 deg, e_Vobs/Vobs < 0.1)",
                            "g_obs = Vobs^2/R, g_bar = (Vgas|Vgas| + 0.5 Vdisk|Vdisk| + 0.7 Vbul|Vbul|)/R in 1e-10 m/s^2"])


@reader("fresh-57", "physo-pantheon", "PhySO astrophysical case studies")
def physo_pantheon(rec, prob, _set):
    raise SkipDataset("pantheon-hubble-diagram: not y = f(x) (the searched H(z)^2 enters through the luminosity-"
                      "distance integral)")


@reader("fresh-93", "lasr-bigbench", "LaSR BIG-bench scaling-law task")
def lasr_bigbench(rec, prob, _set):
    df = csv(checked(rec["file"], rec.get("sha256")))
    df = df.replace([np.inf, -np.inf], np.nan).dropna()           # the authors' cleaning
    feats = list(rec["feature_names"])
    x, y = columns(df, feats, rec["target"])
    return Rows(x, y, feats, random(0.2), "random 0.2: the authors' random 80/20 row split",
                transforms=["rows with a missing value dropped (the authors' notebook)"])


@reader("fresh-103", "phye2e-space-physics", "PhyE2E space-physics applications")
def phye2e(rec, prob, _set):
    k = rec["id"]
    if k == "lunar-tide":
        raise SkipDataset("lunar-tide: 13.3M raw samples (532 MB); the paper's 50^3 gridding is not published")
    data = fetch(rec["url"], rec["sha256"], "." + rec["figshare_name"].rsplit(".", 1)[-1])
    if k == "ssn":   # 1855-09..1976-06 fits (rows 1280:2730), cycles 21-24 (1976-07..2019-12, rows 2730:3252) test
        t = np.loadtxt(io.BytesIO(data), usecols=(2, 3))
        s = lambda a: (a - 1855.707) / (1976.456 - 1855.707) * 8   # noqa: E731  the authors' time scaling
        tr, te = t[1280:2730], t[2730:3252]
        return Rows(s(tr[:, 0]), tr[:, 1], ["t"], GIVEN,
                    "given: 1855-1976 fits, the next four solar cycles (1976-07..2019-12, SILSO cycle minima) test",
                    x_test=s(te[:, 0]), y_test=te[:, 1], transforms=["t scaled to [0, 8] over the training span (the authors)"])
    if k.startswith("plasma"):   # near side (r < 28 RE) fits, far side (r > 28 RE) tests (the paper's Fig. 4c score)
        t = np.loadtxt(io.BytesIO(data), skiprows=1, usecols=(0, 1, 6))
        t = t[(t[:, 2] >= 1e-10) & (t[:, 0] <= 0)]
        x, y = -t[:, 0], t[:, 1]
        r, theta = np.hypot(x, y), np.arctan2(y, x) + np.pi / 2
        xs = np.column_stack([r / 8, theta])
        near, far = r < 28, r > 28
        return Rows(xs[near], t[near, 2], ["r/8", "theta"], GIVEN,
                    "given: the near side r < 28 RE fits, the far side r > 28 RE tests (the paper's split)",
                    x_test=xs[far], y_test=t[far, 2],
                    transforms=["rows with Pt,norm < 1e-10 or X_GSM > 0 dropped; X -> -X; inputs r/8 and "
                                "theta = atan2(Y, -X) + pi/2 (the authors' notebook)"])
    if k == "diffrot":           # the notebook's one seeded split (RandomState(0): 7 test, 27 fit)
        df = csv(data)
        x = df["Latitude"].to_numpy() / 180 * np.pi + np.pi / 2
        y = df["Two-day_Lag_mean"].to_numpy()
        perm = np.random.RandomState(0).permutation(len(y))
        return Rows(x[perm[7:]], y[perm[7:]], ["latitude (rad) + pi/2"], GIVEN,
                    "given: the authors' one seeded split (train_test_split test_size 0.2, random_state 0)",
                    x_test=x[perm[:7]], y_test=y[perm[:7]], transforms=["x = latitude/180 pi + pi/2 (the authors)"])
    if k.startswith("fex"):      # T < 2.8e6 K fits, T >= 2.8e6 K tests (the paper)
        df = csv(data)
        x = np.column_stack([df["temperature"] / 1e6, df["eDensity"] / 1e10])
        y = df["gofnt"].to_numpy() / 1e-25
        lo = df["temperature"].to_numpy() < 2.8e6
        return Rows(x[lo], y[lo], ["T/1e6", "n_e/1e10"], GIVEN,
                    "given: T < 2.8e6 K fits, T >= 2.8e6 K tests (the paper's temperature cutoff)",
                    x_test=x[~lo], y_test=y[~lo], transforms=["T/1e6, n_e/1e10, G(T, n)/1e-25 (the authors' scaling)"])
    raise SkipDataset(f"{k}: no reader")


@reader("fresh-169", "flowsr-mobility", "Human mobility flows (FlowSR)")
def flowsr(rec, prob, _set):
    raise SkipDataset(f"{rec['id']}: an allocation weight normalised per origin (grouped loss), not y = f(x)")


@reader("fresh-163", "camels-subhalo-mass", "Subhalo total-mass relation (CAMELS)")
def camels(rec, prob, _set):
    raise SkipDataset(f"{rec['id']}: built from 1000 CAMELS snapshot files (15.8 GB download); left out for its size (srbf 2026-11-04)")


@reader("fresh-167", "syren-power-spectrum", "syren matter power spectrum emulators")
def syren(rec, prob, _set):
    raise SkipDataset(f"{rec['id']}: no data published (the authors' Latin hypercubes would have to be regenerated "
                      "with CAMB)")
