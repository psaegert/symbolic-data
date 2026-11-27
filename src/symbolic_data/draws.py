"""Frozen draws: each problem's data rendered once, so that every reader sees the same points.

A catalog is a set of problem templates; a :class:`~symbolic_data.source.ProblemSource` draws its points afresh on
every pass. Frozen draws fix that draw ahead of time: for each problem of a catalog the renderer draws, once,
everything that a run would draw, and writes it to disk. An evaluation then reads the draw and never samples. A draw
carries a repeat index ``r``: one draw per problem (``r = 1``) is the default, and a render may add further repeats
without a change of format.

What one draw holds:

* a formula problem (a declarative entry): its fitting rows and held-out rows (``n_support`` and ``n_validation``,
  drawn together or from the entry's ``test`` block exactly as a ``ProblemSource`` draws them), the values its
  ``per_run`` block draws (``meta["per_run_values"]``, with the ground truth written with them), and one standard
  normal vector per side, ``noise_support`` and ``noise_validation``. A noise level ``s`` makes the targets
  ``y + s * std(y) * noise`` per side (the scalar ``noise`` of a ``ProblemSource``), so every noise level scales the
  same vector;
* a data problem (pools, :mod:`symbolic_data.pools`): the rows of one draw from its pools, its split rule applied
  (a random split or a fold picked, at most ``n_support`` fitting and ``n_validation`` test rows taken);
* a realized problem (a frozen problem without a split rule): the problem as stored.

**Attempts.** A problem's draw is chosen among seeded attempts ``k = 1, 2, ...``: attempt ``k`` draws from
``SeedSequence(sha256(seed_root, name, eq_id, r, k))``, spawned into one stream for the points (and per-run values)
and one for the noise vectors. Only what the benchmark leaves to chance differs between attempts (input points,
random constants, per-run values); the formula and its fixed constants never change. Which attempt a problem keeps
is decided outside this module and handed to :func:`render` as an *attempts document*
(``symbolic-data/draw-attempts``): srbf keeps the first attempt on which the problem's own formula, refitted, is
recognized, and attempt 1 with the flag ``formula_inherent`` when no attempt within ``max_trials`` is (srbf's
``srbf.draw_acceptance``). A problem the document does not list keeps attempt 1. :class:`CatalogDraws` is the
interface both sides share: the catalog's problems and attempt ``k`` of each, drawn exactly as the render draws it.
A rerun with the same software and the same attempts writes the same bytes, and a problem's draw does not depend on
the other problems of its catalog, on their order or on which ones are rendered.

Layout of a rendered root::

    <root>/manifest.json
    <root>/<catalog name>/repeat-<r>.npz

Each ``.npz`` holds ``_meta`` (UTF-8 JSON as a ``uint8`` array: the format, the catalog's identity, the repeat, the
settings, and one record per problem with its ground truth, ``meta``, the attempt it keeps, whether the acceptance
rule was applied to it (``checked``), ``formula_inherent`` and ``failures``, the kind of each failed attempt in order
as the attempts document names it) and the arrays of problem ``i`` as ``p<i>__<name>``:
``x_support``, ``y_support``, ``x_validation``, ``y_validation`` (float64), the two noise vectors for formula
problems, and, where present, ``y_support_noisy`` / ``y_validation_noisy`` (stored targets that differ from the clean
ones), ``y_reference_*`` and ``outlier_mask_*``. A placeholder (an entry with no valid draw) has no arrays. The
manifest records, per catalog, its source file's sha256, each repeat file's path, size and sha256, and what the
acceptance changed (problems checked, problems on a later attempt, the ids flagged formula-inherent, failed attempts
by kind); at the top, the attempts document's sha256 and rule.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from symbolic_data.numeric import STORAGE_DTYPE

__all__ = [
    "ATTEMPTS_FORMAT", "DEFAULT_ENGINE", "DEFAULT_REPEATS", "DEFAULT_SEED_ROOT", "FORMAT", "FORMAT_VERSION", "MANIFEST",
    "CatalogDraws", "Draw", "DrawsSettings", "load_attempts", "needs_check", "read_meta", "render", "render_catalog",
    "seed_sequence", "sha256_file", "verify",
]

FORMAT = "symbolic-data/frozen-draws"
FORMAT_VERSION = 1
#: The document that names each problem's accepted attempt (written by the acceptance check, read by :func:`render`).
ATTEMPTS_FORMAT = "symbolic-data/draw-attempts"
MANIFEST = "manifest.json"
DEFAULT_SEED_ROOT = "symbolic-data/frozen-draws/1"
#: One draw per problem; the repeat index stays in the layout, so that a later render can add repeats.
DEFAULT_REPEATS = (1,)
#: The engine the ground truth is compiled with (srbf's suite engine).
DEFAULT_ENGINE = "acj-5-4-llm"

_CORE = ("x_support", "y_support", "x_validation", "y_validation")
_NOISE = ("noise_support", "noise_validation")
_OPTIONAL = ("y_reference_support", "y_reference_validation", "outlier_mask_support", "outlier_mask_validation")
_NOISY = (("y_support_noisy", "y_support"), ("y_validation_noisy", "y_validation"))


@dataclass(frozen=True)
class DrawsSettings:
    """What every catalog of one rendered root shares; recorded in the manifest and in every file."""

    n_support: int = 512
    n_validation: int = 512
    repeats: tuple[int, ...] = DEFAULT_REPEATS
    seed_root: str = DEFAULT_SEED_ROOT
    simplipy_engine: str = DEFAULT_ENGINE
    layout: str = "random"
    max_trials: int = 100

    def __post_init__(self) -> None:
        repeats = tuple(int(r) for r in self.repeats)
        if not repeats or len(set(repeats)) != len(repeats) or min(repeats) < 1:
            raise ValueError(f"repeats must be distinct integers >= 1, got {self.repeats!r}")
        if self.n_support < 1 or self.n_validation < 0:
            raise ValueError(f"n_support must be >= 1 and n_validation >= 0, got {self.n_support}, {self.n_validation}")
        object.__setattr__(self, "repeats", repeats)

    def sampling(self) -> dict[str, Any]:
        """The ``ProblemSource`` sampling block the draws follow (noise off: the noise is the stored vector)."""
        return {"n_support": self.n_support, "n_validation": self.n_validation, "noise": 0.0, "method": "iterate",
                "problems_per_expression": 1, "layout": self.layout, "max_trials": self.max_trials}

    def record(self) -> dict[str, Any]:
        """The settings as the manifest and the files record them."""
        return {"n_support": self.n_support, "n_validation": self.n_validation, "layout": self.layout,
                "max_trials": self.max_trials, "seed_root": self.seed_root, "simplipy_engine": self.simplipy_engine}


def seed_sequence(seed_root: str, catalog: str, eq_id: str, repeat: int, attempt: int = 1) -> np.random.SeedSequence:
    """The seed of attempt ``attempt`` of problem ``eq_id`` of catalog ``catalog`` in repeat ``repeat``."""
    if int(attempt) < 1:
        raise ValueError(f"attempts count from 1, got {attempt}")
    key = "\x1f".join((str(seed_root), str(catalog), str(eq_id), str(int(repeat)), str(int(attempt)))).encode("utf-8")
    return np.random.SeedSequence(int.from_bytes(hashlib.sha256(key).digest(), "big"))


def sha256_file(path: str | os.PathLike[str]) -> str:
    """The sha256 of a file's bytes."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_meta(path: str | os.PathLike[str]) -> dict[str, Any]:
    """The ``_meta`` record of one rendered file."""
    with np.load(path, allow_pickle=False) as data:
        return json.loads(data["_meta"].tobytes().decode("utf-8"))


# ------------------------------------------------------------------------------------------------ attempts
def needs_check(kind: str, problem: Any) -> bool:
    """Whether the acceptance rule applies to a problem (judged on its attempt 1): a formula problem whose formula is not
    a constant by construction (skeleton ``<constant>``). Data and stored problems hold no values left to chance."""
    return kind == "formula" and tuple(problem.skeleton or ()) != ("<constant>",)


@dataclass(frozen=True)
class Draw:
    """One attempt of one problem: the problem as the render stores it and, for a formula problem, its noise vectors."""

    problem: Any
    kind: str
    noise_support: np.ndarray | None = None
    noise_validation: np.ndarray | None = None


def _load_engine(engine: Any) -> Any:
    from symbolic_data._evaluation import load_engine

    return load_engine(engine)


class CatalogDraws:
    """One catalog's problems and their seeded attempts, drawn exactly as :func:`render` draws them.

    The interface between the renderer and an acceptance check: the check draws attempt ``k`` of problem ``row`` with
    :meth:`draw`, judges it, and hands the attempt it keeps back to :func:`render` in an attempts document.
    """

    def __init__(self, ref: str, settings: DrawsSettings = DrawsSettings(), *, simplipy_engine: Any = None) -> None:
        from symbolic_data.generative import GenerativeCatalog
        from symbolic_data.pools import has_pools
        from symbolic_data.resolver import resolve
        from symbolic_data.source import ProblemSource

        self.ref = ref
        self.settings = settings
        artifact = resolve(ref)
        self.source = ProblemSource({"catalog": artifact.path, "sampling": settings.sampling()},
                                    simplipy_engine=_load_engine(simplipy_engine or settings.simplipy_engine))
        catalog = self.source.catalog
        if isinstance(catalog, GenerativeCatalog):
            raise ValueError(f"{ref}: a generative catalog has no fixed problems to render")
        if getattr(catalog, "frozen", False):
            problems = list(catalog.problems or [])
            self.ids = [p.eq_id for p in problems]
            self.kinds = ["data" if has_pools(p) else "stored" for p in problems]
        else:
            self.ids = list(catalog.entries)
            self.kinds = ["formula"] * len(self.ids)
        if any(i is None for i in self.ids) or len(set(map(str, self.ids))) != len(self.ids):
            raise ValueError(f"{ref}: every problem needs a distinct eq_id (it names the problem's seed)")
        self.ids = [str(i) for i in self.ids]
        self.name = str(catalog.name)
        self.identity = {"name": self.name, "version": catalog.version, "source": os.path.basename(artifact.path),
                         "source_sha256": sha256_file(artifact.path)}

    def __len__(self) -> int:
        return len(self.ids)

    def rows(self, ids: Sequence[str] | None = None) -> list[int]:
        """The rows to render: all, or those of ``ids`` (in the catalog's order)."""
        if ids is None:
            return list(range(len(self.ids)))
        position = {i: row for row, i in enumerate(self.ids)}
        unknown = [i for i in ids if str(i) not in position]
        if unknown:
            raise ValueError(f"{self.ref}: ids not in the catalog: {unknown[:5]}{' ...' if len(unknown) > 5 else ''}")
        keep = {position[str(i)] for i in ids}
        return [row for row in range(len(self.ids)) if row in keep]

    def draw(self, row: int, repeat: int = 1, attempt: int = 1) -> Draw:
        """Attempt ``attempt`` of problem ``row`` in repeat ``repeat``."""
        points, noise = (np.random.default_rng(s) for s in seed_sequence(
            self.settings.seed_root, self.name, self.ids[row], repeat, attempt).spawn(2))
        problem = self.source.draw(row, points)
        kind = self.kinds[row]
        if kind != "formula" or problem.is_placeholder:
            return Draw(problem, kind)
        return Draw(problem, kind, noise.standard_normal(size=np.shape(problem.y_support)),
                    noise.standard_normal(size=np.shape(problem.y_validation)))


def load_attempts(attempts: str | os.PathLike[str] | Mapping[str, Any], settings: DrawsSettings) -> tuple[dict[str, Any], str | None]:
    """An attempts document (a path or the parsed mapping), checked against ``settings``; returns it and the file's sha256.

    The document: ``{format: symbolic-data/draw-attempts, format_version: 1, settings, repeats, rule, catalogs: {name:
    {source_sha256, attempts: {"<repeat>": {eq_id: {attempt, formula_inherent, ...}}}}}}``. A problem it lists was
    checked; the others keep attempt 1.
    """
    sha = None
    if not isinstance(attempts, Mapping):
        sha = sha256_file(attempts)
        attempts = json.loads(Path(attempts).read_text(encoding="utf-8"))
    doc = dict(attempts)
    if doc.get("format") != ATTEMPTS_FORMAT or doc.get("format_version") != 1:
        raise ValueError(f"not a {ATTEMPTS_FORMAT} document of version 1")
    if doc.get("settings") != settings.record():
        raise ValueError(f"the attempts were chosen under other settings: {doc.get('settings')}")
    missing = sorted(set(settings.repeats) - {int(r) for r in doc.get("repeats", [])})
    if missing:
        raise ValueError(f"the attempts document holds no repeat {missing}")
    return doc, sha


def _problem_record(problem: Any, row: int, kind: str, attempt: int, checked: bool, inherent: bool,
                    failures: Sequence[str] = ()) -> dict[str, Any]:
    return {
        "row": int(row), "eq_id": problem.eq_id, "kind": kind,
        "skeleton": [str(t) for t in problem.skeleton] if problem.skeleton is not None else None,
        "expression": [str(t) for t in problem.expression] if problem.expression is not None else None,
        "constants": [float(c) for c in problem.constants], "variables": [str(v) for v in problem.variables],
        "complexity": problem.complexity, "gt_kind": problem.gt_kind, "noise": problem.noise, "meta": problem.meta,
        "is_placeholder": bool(problem.is_placeholder), "placeholder_reason": problem.placeholder_reason,
        "attempt": int(attempt), "checked": bool(checked), "formula_inherent": bool(inherent),
        "failures": [str(f) for f in failures],
    }


def _draw_arrays(draw: Draw) -> dict[str, np.ndarray]:
    problem = draw.problem
    if problem.is_placeholder:
        return {}
    arrays = {name: np.asarray(getattr(problem, name), dtype=STORAGE_DTYPE) for name in _CORE}
    if draw.kind == "formula":
        arrays["noise_support"] = np.asarray(draw.noise_support)
        arrays["noise_validation"] = np.asarray(draw.noise_validation)
    else:
        for noisy, clean in _NOISY:
            value = getattr(problem, noisy)
            if value is not None and not np.array_equal(np.asarray(value), arrays[clean], equal_nan=True):
                arrays[noisy] = np.asarray(value)
    for name in _OPTIONAL:
        value = getattr(problem, name)
        if value is not None:
            arrays[name] = np.asarray(value)
    return arrays


# ------------------------------------------------------------------------------------------------ rendering
def _write_npz(path: Path, members: Iterable[tuple[str, np.ndarray]]) -> None:
    """An ``.npz`` written member by member (the bytes ``np.savez`` writes: stored, zip64, entries dated 1980-01-01),
    so that a large catalog never sits in memory whole. Atomic: a temporary file is renamed into place."""
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    try:
        with zipfile.ZipFile(tmp, mode="w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
            for name, value in members:
                with archive.open(f"{name}.npy", "w", force_zip64=True) as handle:
                    np.lib.format.write_array(handle, np.asanyarray(value), allow_pickle=False)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def _versions() -> dict[str, str]:
    from importlib.metadata import PackageNotFoundError, version

    out = {"python": sys.version.split()[0], "numpy": np.__version__}
    for package in ("symbolic-data", "simplipy"):
        try:
            out[package] = version(package)
        except PackageNotFoundError:  # pragma: no cover - a source tree without installed metadata
            out[package] = "unknown"
    return out


def render_catalog(ref: str, root: str | os.PathLike[str], settings: DrawsSettings = DrawsSettings(), *,
                   ids: Sequence[str] | None = None, attempts: Mapping[str, Any] | None = None,
                   simplipy_engine: Any = None) -> tuple[str, dict[str, Any]]:
    """Render every repeat of one catalog into ``root/<name>/repeat-<r>.npz``; returns its name and manifest entry.

    ``ref`` is anything :func:`~symbolic_data.resolver.resolve` takes (a local path, a ``name[@version]``). ``ids``
    renders only those problems (in the catalog's order), for a benchmark evaluated on a frozen subset. ``attempts`` is
    this catalog's entry of an attempts document (:func:`load_attempts`; its ``source_sha256`` must be this catalog's);
    without it every problem keeps attempt 1 and no acceptance is recorded. ``simplipy_engine`` passes a loaded engine;
    by default ``settings.simplipy_engine`` is loaded.
    """
    draws = CatalogDraws(ref, settings, simplipy_engine=simplipy_engine)
    name, identity = draws.name, draws.identity
    if attempts is not None and attempts.get("source_sha256") != identity["source_sha256"]:
        raise ValueError(f"{name}: the attempts were chosen on another version of the catalog "
                         f"({str(attempts.get('source_sha256'))[:12]}, not {identity['source_sha256'][:12]})")
    rows = draws.rows(ids)
    folder = Path(root) / name
    folder.mkdir(parents=True, exist_ok=True)
    files: dict[str, dict[str, Any]] = {}
    placeholders: dict[str, int] = {}
    acceptance: dict[str, Any] = {}
    for repeat in settings.repeats:
        records: list[dict[str, Any]] = []
        chosen = dict(((attempts or {}).get("attempts") or {}).get(str(repeat)) or {})

        def members(repeat: int = repeat, records: list[dict[str, Any]] = records,
                    chosen: dict[str, Any] = chosen) -> Iterable[tuple[str, np.ndarray]]:
            for index, row in enumerate(rows):
                choice = chosen.get(draws.ids[row])
                attempt = int(choice["attempt"]) if choice is not None else 1
                draw = draws.draw(row, repeat, attempt)
                records.append(_problem_record(draw.problem, row, draw.kind, attempt, choice is not None,
                                               bool(choice is not None and choice.get("formula_inherent")),
                                               (choice or {}).get("failures") or ()))
                for key, value in _draw_arrays(draw).items():
                    yield f"p{index}__{key}", value
            meta = {"format": FORMAT, "format_version": FORMAT_VERSION, "catalog": identity, "repeat": repeat,
                    "settings": settings.record(), "problems": records}
            yield "_meta", np.frombuffer(json.dumps(meta).encode("utf-8"), dtype=np.uint8)

        path = folder / f"repeat-{repeat}.npz"
        _write_npz(path, members())
        placeholders[str(repeat)] = sum(r["is_placeholder"] for r in records)
        files[str(repeat)] = {"path": f"{name}/{path.name}", "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        failed = Counter(f for r in records for f in r["failures"])
        acceptance[str(repeat)] = {"checked": sum(r["checked"] for r in records),
                                   "later_attempt": sum(r["attempt"] > 1 for r in records),
                                   "formula_inherent": [r["eq_id"] for r in records if r["formula_inherent"]],
                                   "failures": dict(sorted(failed.items()))}
    kept = [draws.kinds[r] for r in rows]
    entry = {**identity, "problems": len(rows), "kinds": {k: kept.count(k) for k in sorted(set(kept))},
             "ids": None if ids is None else {"count": len(rows), "sha256": hashlib.sha256(
                 json.dumps([draws.ids[r] for r in rows]).encode("utf-8")).hexdigest()},
             "placeholders": placeholders, "acceptance": acceptance if attempts is not None else None, "files": files}
    return name, entry


def _render_task(task: tuple[str, str, DrawsSettings, Sequence[str] | None, Any]) -> tuple[str, str, dict[str, Any], float]:
    import time

    ref, root, settings, ids, attempts = task
    start = time.perf_counter()
    name, entry = render_catalog(ref, root, settings, ids=ids, attempts=attempts, simplipy_engine=_ENGINE)
    return ref, name, entry, time.perf_counter() - start


_ENGINE: Any = None   # the engine a render worker loads once and every catalog it renders shares


def _load_worker_engine(engine: str) -> None:
    global _ENGINE
    _ENGINE = _load_engine(engine)


def _manifest(settings: DrawsSettings, catalogs: Mapping[str, Any], acceptance: Any) -> dict[str, Any]:
    return {"format": FORMAT, "format_version": FORMAT_VERSION, "settings": settings.record(),
            "repeats": list(settings.repeats), "rendered_with": _versions(), "acceptance": acceptance,
            "catalogs": dict(sorted(catalogs.items()))}


def _write_manifest(root: Path, manifest: Mapping[str, Any]) -> Path:
    path = root / MANIFEST
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    return path


def _catalog_name(ref: str) -> str:
    """The name a catalog file declares, without drawing anything (to pick its entry of an attempts document)."""
    from symbolic_data.catalog import ProblemCatalog
    from symbolic_data.resolver import resolve

    path = resolve(ref).path
    if path.endswith(".npz"):
        with np.load(path, allow_pickle=False) as data:
            return str(json.loads(str(data["_meta"].item()))["catalog"]["name"])
    return ProblemCatalog.from_yaml(path).name


def render(refs: Sequence[str], root: str | os.PathLike[str], settings: DrawsSettings = DrawsSettings(), *,
           ids: Mapping[str, Sequence[str]] | None = None, attempts: str | os.PathLike[str] | Mapping[str, Any] | None = None,
           jobs: int = 1, update: bool = False, log: Any = print) -> dict[str, Any]:
    """Render ``refs`` into ``root`` and write its manifest; returns the manifest.

    ``ids`` maps a catalog (its ref or its name) to the problems to render. ``attempts`` is an attempts document (a
    path or a mapping, :func:`load_attempts`): every catalog rendered must be in it, and each problem it lists keeps
    the attempt it names. ``jobs`` renders that many catalogs at once (worker processes, each loading the engine once).
    A root that holds a manifest is refused unless ``update`` adds to it, which needs the same settings and the same
    attempts document; a catalog rendered again replaces its earlier entry.
    """
    global _ENGINE
    root = Path(root)
    doc, doc_sha = load_attempts(attempts, settings) if attempts is not None else (None, None)
    acceptance = None if doc is None else {"attempts_sha256": doc_sha, "rule": doc.get("rule")}
    existing: dict[str, Any] = {}
    if (root / MANIFEST).exists():
        if not update:
            raise FileExistsError(f"{root} already holds frozen draws; pass update=True (--update) to add catalogs")
        previous = json.loads((root / MANIFEST).read_text(encoding="utf-8"))
        if previous.get("settings") != settings.record() or previous.get("repeats") != list(settings.repeats):
            raise ValueError(f"{root}: an update needs the settings it was rendered with: {previous.get('settings')}, "
                             f"repeats {previous.get('repeats')}")
        if previous.get("acceptance") != acceptance:
            raise ValueError(f"{root}: an update needs the attempts it was rendered with: {previous.get('acceptance')}")
        existing = dict(previous.get("catalogs") or {})
    root.mkdir(parents=True, exist_ok=True)
    ids = dict(ids or {})
    tasks = []
    for ref in refs:
        chosen = None
        if doc is not None:
            name = _catalog_name(ref)
            if name not in (doc.get("catalogs") or {}):
                raise ValueError(f"{ref}: catalog {name!r} is not in the attempts document")
            chosen = doc["catalogs"][name]
        tasks.append((ref, str(root), settings, ids.get(ref, ids.get(Path(ref).stem)), chosen))
    # the largest local catalogs first, so that parallel workers finish together
    tasks.sort(key=lambda task: -(os.path.getsize(task[0]) if os.path.isfile(task[0]) else 0))
    rendered: dict[str, Any] = {}
    sources: dict[str, str] = {}
    if jobs > 1 and len(tasks) > 1:
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor, as_completed

        with ProcessPoolExecutor(max_workers=min(jobs, len(tasks)), mp_context=multiprocessing.get_context("spawn"),
                                 initializer=_load_worker_engine, initargs=(settings.simplipy_engine,)) as pool:
            futures = [pool.submit(_render_task, task) for task in tasks]
            results = [future.result() for future in as_completed(futures)]
    else:
        _load_worker_engine(settings.simplipy_engine)
        try:
            results = [_render_task(task) for task in tasks]
        finally:
            _ENGINE = None
    for ref, name, entry, seconds in sorted(results, key=lambda item: item[1]):
        if name in sources:
            raise ValueError(f"two catalogs are named {name!r} ({sources[name]}, {ref}); render them into separate roots")
        sources[name] = ref
        rendered[name] = entry
        log(f"{name}: {entry['problems']} problems x {len(settings.repeats)} repeats, "
            f"{sum(f['bytes'] for f in entry['files'].values()) / 1e6:.1f} MB in {seconds:.1f} s")
    manifest = _manifest(settings, {**existing, **rendered}, acceptance)
    _write_manifest(root, manifest)
    return manifest


# ------------------------------------------------------------------------------------------------ verification
def _recorded_attempts(root: Path, manifest: Mapping[str, Any]) -> dict[str, Any] | None:
    """The attempts a rendered root keeps, as an attempts document (from its files), to render it again."""
    if manifest.get("acceptance") is None:
        return None
    catalogs: dict[str, Any] = {}
    for name, entry in manifest["catalogs"].items():
        per_repeat: dict[str, Any] = {}
        for repeat, info in entry["files"].items():
            path = root / info["path"]
            if path.is_file():
                per_repeat[repeat] = {r["eq_id"]: {"attempt": r["attempt"], "formula_inherent": r["formula_inherent"],
                                                   "failures": list(r.get("failures") or [])}
                                      for r in read_meta(path)["problems"] if r.get("checked")}
        catalogs[name] = {"source_sha256": entry["source_sha256"], "attempts": per_repeat}
    return {"format": ATTEMPTS_FORMAT, "format_version": 1, "settings": manifest["settings"],
            "repeats": manifest["repeats"], "rule": manifest["acceptance"].get("rule"), "catalogs": catalogs}


def verify(root: str | os.PathLike[str], refs: Sequence[str] = (), *, ids: Mapping[str, Sequence[str]] | None = None,
           jobs: int = 1) -> list[str]:
    """Check a rendered root; returns what is wrong (empty when it is sound).

    Every file listed in the manifest must exist with its recorded sha256 and name its catalog and repeat. ``refs``
    are rendered again, with the attempts the files keep, into a scratch folder inside ``root`` and must reproduce the
    recorded bytes, which is how a render is shown to be reproducible.
    """
    import shutil

    root = Path(root)
    manifest = json.loads((root / MANIFEST).read_text(encoding="utf-8"))
    problems: list[str] = []
    if manifest.get("format") != FORMAT or manifest.get("format_version") != FORMAT_VERSION:
        return [f"{root}: not {FORMAT} version {FORMAT_VERSION}"]
    for name, entry in manifest["catalogs"].items():
        for repeat, info in entry["files"].items():
            path = root / info["path"]
            if not path.is_file():
                problems.append(f"{name} repeat {repeat}: {info['path']} is missing")
                continue
            if sha256_file(path) != info["sha256"]:
                problems.append(f"{name} repeat {repeat}: {info['path']} does not match its sha256")
                continue
            meta = read_meta(path)
            if meta["catalog"]["source_sha256"] != entry["source_sha256"] or str(meta["repeat"]) != str(repeat):
                problems.append(f"{name} repeat {repeat}: the file names another catalog or repeat")
    if refs:
        settings = DrawsSettings(n_support=manifest["settings"]["n_support"],
                                 n_validation=manifest["settings"]["n_validation"],
                                 repeats=tuple(manifest["repeats"]), seed_root=manifest["settings"]["seed_root"],
                                 simplipy_engine=manifest["settings"]["simplipy_engine"],
                                 layout=manifest["settings"]["layout"], max_trials=manifest["settings"]["max_trials"])
        scratch = root / f".verify-{os.getpid()}"
        try:
            again = render(refs, scratch, settings, ids=ids, attempts=_recorded_attempts(root, manifest), jobs=jobs,
                           log=lambda _: None)
            for name, entry in again["catalogs"].items():
                recorded = manifest["catalogs"].get(name)
                if recorded is None:
                    problems.append(f"{name}: rendered again but not in the manifest")
                    continue
                if entry["source_sha256"] != recorded["source_sha256"]:
                    problems.append(f"{name}: the catalog's source differs from the rendered one")
                for repeat, info in entry["files"].items():
                    if info["sha256"] != recorded["files"].get(repeat, {}).get("sha256"):
                        problems.append(f"{name} repeat {repeat}: a second render writes other bytes")
        except ValueError as exc:
            problems.append(f"rendering again failed: {exc}")
        finally:
            shutil.rmtree(scratch, ignore_errors=True)
    return problems
