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

Layout of a rendered root::

    <root>/manifest.json
    <root>/<catalog name>/repeat-<r>.npz

Each ``.npz`` holds ``_meta`` (UTF-8 JSON as a ``uint8`` array: the format, the catalog's identity, the repeat, the
settings, and one record per problem with its ground truth and ``meta``) and the arrays of problem ``i`` as
``p<i>__<name>``: ``x_support``, ``y_support``, ``x_validation``, ``y_validation`` (float64), the two noise vectors for
formula problems, and, where present, ``y_support_noisy`` / ``y_validation_noisy`` (stored targets that differ from the
clean ones), ``y_reference_*`` and ``outlier_mask_*``. A placeholder (an entry with no valid draw) has no arrays. The
manifest records, per catalog, its source file's sha256 and each repeat file's path, size and sha256.

Seeds are derived, not chosen: problem ``eq_id`` of catalog ``name`` in repeat ``r`` draws from
``SeedSequence(sha256(seed_root, name, eq_id, r))``, spawned into one stream for the points (and per-run values) and
one for the noise vectors. A rerun with the same software writes the same bytes, and a problem's draw does not
depend on the other problems of its catalog, on their order or on which ones are rendered.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from symbolic_data.numeric import STORAGE_DTYPE

__all__ = [
    "DEFAULT_ENGINE", "DEFAULT_REPEATS", "DEFAULT_SEED_ROOT", "FORMAT", "FORMAT_VERSION", "MANIFEST",
    "DrawsSettings", "read_meta", "render", "render_catalog", "seed_sequence", "sha256_file", "verify",
]

FORMAT = "symbolic-data/frozen-draws"
FORMAT_VERSION = 1
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


def seed_sequence(seed_root: str, catalog: str, eq_id: str, repeat: int) -> np.random.SeedSequence:
    """The seed of problem ``eq_id`` of catalog ``catalog`` in repeat ``repeat``."""
    key = "\x1f".join((str(seed_root), str(catalog), str(eq_id), str(int(repeat)))).encode("utf-8")
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


# ------------------------------------------------------------------------------------------------ rendering
def _problem_record(problem: Any, row: int, kind: str) -> dict[str, Any]:
    return {
        "row": int(row), "eq_id": problem.eq_id, "kind": kind,
        "skeleton": [str(t) for t in problem.skeleton] if problem.skeleton is not None else None,
        "expression": [str(t) for t in problem.expression] if problem.expression is not None else None,
        "constants": [float(c) for c in problem.constants], "variables": [str(v) for v in problem.variables],
        "complexity": problem.complexity, "gt_kind": problem.gt_kind, "noise": problem.noise, "meta": problem.meta,
        "is_placeholder": bool(problem.is_placeholder), "placeholder_reason": problem.placeholder_reason,
    }


def _problem_arrays(problem: Any, kind: str, noise_rng: np.random.Generator) -> dict[str, np.ndarray]:
    if problem.is_placeholder:
        return {}
    arrays = {name: np.asarray(getattr(problem, name), dtype=STORAGE_DTYPE) for name in _CORE}
    if kind == "formula":
        arrays["noise_support"] = noise_rng.standard_normal(size=arrays["y_support"].shape)
        arrays["noise_validation"] = noise_rng.standard_normal(size=arrays["y_validation"].shape)
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


def _load_engine(engine: Any) -> Any:
    from symbolic_data._evaluation import load_engine

    return load_engine(engine)


def render_catalog(ref: str, root: str | os.PathLike[str], settings: DrawsSettings = DrawsSettings(), *,
                   ids: Sequence[str] | None = None, simplipy_engine: Any = None) -> tuple[str, dict[str, Any]]:
    """Render every repeat of one catalog into ``root/<name>/repeat-<r>.npz``; returns its name and manifest entry.

    ``ref`` is anything :func:`~symbolic_data.resolver.resolve` takes (a local path, a ``name[@version]``). ``ids``
    renders only those problems (in the catalog's order), for a benchmark evaluated on a frozen subset.
    ``simplipy_engine`` passes a loaded engine; by default ``settings.simplipy_engine`` is loaded.
    """
    from symbolic_data.generative import GenerativeCatalog
    from symbolic_data.resolver import resolve
    from symbolic_data.source import ProblemSource

    artifact = resolve(ref)
    source = ProblemSource({"catalog": artifact.path, "sampling": settings.sampling()},
                           simplipy_engine=_load_engine(simplipy_engine or settings.simplipy_engine))
    catalog = source.catalog
    if isinstance(catalog, GenerativeCatalog):
        raise ValueError(f"{ref}: a generative catalog has no fixed problems to render")
    frozen = bool(getattr(catalog, "frozen", False))
    if frozen:
        from symbolic_data.pools import has_pools

        problems = list(catalog.problems or [])
        all_ids = [p.eq_id for p in problems]
        kinds = ["data" if has_pools(p) else "stored" for p in problems]
    else:
        all_ids = list(catalog.entries)
        kinds = ["formula"] * len(all_ids)
    if any(i is None for i in all_ids) or len(set(map(str, all_ids))) != len(all_ids):
        raise ValueError(f"{ref}: every problem needs a distinct eq_id (it names the problem's seed)")
    rows = list(range(len(all_ids)))
    if ids is not None:
        position = {str(i): row for row, i in enumerate(all_ids)}
        unknown = [i for i in ids if str(i) not in position]
        if unknown:
            raise ValueError(f"{ref}: ids not in the catalog: {unknown[:5]}{' ...' if len(unknown) > 5 else ''}")
        keep = {position[str(i)] for i in ids}
        rows = [row for row in rows if row in keep]
    name = str(catalog.name)
    identity = {"name": name, "version": catalog.version, "source": os.path.basename(artifact.path),
                "source_sha256": sha256_file(artifact.path)}
    folder = Path(root) / name
    folder.mkdir(parents=True, exist_ok=True)
    files: dict[str, dict[str, Any]] = {}
    placeholders: dict[str, int] = {}
    for repeat in settings.repeats:
        records: list[dict[str, Any]] = []

        def members(repeat: int = repeat, records: list[dict[str, Any]] = records) -> Iterable[tuple[str, np.ndarray]]:
            for index, row in enumerate(rows):
                points, noise = (np.random.default_rng(s) for s in
                                 seed_sequence(settings.seed_root, name, str(all_ids[row]), repeat).spawn(2))
                problem = source.draw(row, points)
                records.append(_problem_record(problem, row, kinds[row]))
                for key, value in _problem_arrays(problem, kinds[row], noise).items():
                    yield f"p{index}__{key}", value
            meta = {"format": FORMAT, "format_version": FORMAT_VERSION, "catalog": identity, "repeat": repeat,
                    "settings": settings.record(), "problems": records}
            yield "_meta", np.frombuffer(json.dumps(meta).encode("utf-8"), dtype=np.uint8)

        path = folder / f"repeat-{repeat}.npz"
        _write_npz(path, members())
        placeholders[str(repeat)] = sum(r["is_placeholder"] for r in records)
        files[str(repeat)] = {"path": f"{name}/{path.name}", "bytes": path.stat().st_size, "sha256": sha256_file(path)}
    kept = [kinds[r] for r in rows]
    entry = {**identity, "problems": len(rows), "kinds": {k: kept.count(k) for k in sorted(set(kept))},
             "ids": None if ids is None else {"count": len(rows), "sha256": hashlib.sha256(
                 json.dumps([str(all_ids[r]) for r in rows]).encode("utf-8")).hexdigest()},
             "placeholders": placeholders, "files": files}
    return name, entry


def _render_task(task: tuple[str, str, DrawsSettings, Sequence[str] | None]) -> tuple[str, str, dict[str, Any], float]:
    import time

    ref, root, settings, ids = task
    start = time.perf_counter()
    name, entry = render_catalog(ref, root, settings, ids=ids, simplipy_engine=_ENGINE)
    return ref, name, entry, time.perf_counter() - start


_ENGINE: Any = None   # the engine a render worker loads once and every catalog it renders shares


def _load_worker_engine(engine: str) -> None:
    global _ENGINE
    _ENGINE = _load_engine(engine)


def _manifest(settings: DrawsSettings, catalogs: Mapping[str, Any]) -> dict[str, Any]:
    return {"format": FORMAT, "format_version": FORMAT_VERSION, "settings": settings.record(),
            "repeats": list(settings.repeats), "rendered_with": _versions(), "catalogs": dict(sorted(catalogs.items()))}


def _write_manifest(root: Path, manifest: Mapping[str, Any]) -> Path:
    path = root / MANIFEST
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    return path


def render(refs: Sequence[str], root: str | os.PathLike[str], settings: DrawsSettings = DrawsSettings(), *,
           ids: Mapping[str, Sequence[str]] | None = None, jobs: int = 1, update: bool = False,
           log: Any = print) -> dict[str, Any]:
    """Render ``refs`` into ``root`` and write its manifest; returns the manifest.

    ``ids`` maps a catalog (its ref or its name) to the problems to render. ``jobs`` renders that many catalogs at
    once (worker processes, each loading the engine once). A root that holds a manifest is refused unless ``update`` adds
    to it, which needs the same settings; a catalog rendered again replaces its earlier entry.
    """
    global _ENGINE
    root = Path(root)
    existing: dict[str, Any] = {}
    if (root / MANIFEST).exists():
        if not update:
            raise FileExistsError(f"{root} already holds frozen draws; pass update=True (--update) to add catalogs")
        previous = json.loads((root / MANIFEST).read_text(encoding="utf-8"))
        if previous.get("settings") != settings.record() or previous.get("repeats") != list(settings.repeats):
            raise ValueError(f"{root}: an update needs the settings it was rendered with: {previous.get('settings')}, "
                             f"repeats {previous.get('repeats')}")
        existing = dict(previous.get("catalogs") or {})
    root.mkdir(parents=True, exist_ok=True)
    ids = dict(ids or {})
    tasks = [(ref, str(root), settings, ids.get(ref, ids.get(Path(ref).stem))) for ref in refs]
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
    manifest = _manifest(settings, {**existing, **rendered})
    _write_manifest(root, manifest)
    return manifest


# ------------------------------------------------------------------------------------------------ verification
def verify(root: str | os.PathLike[str], refs: Sequence[str] = (), *, ids: Mapping[str, Sequence[str]] | None = None,
           jobs: int = 1) -> list[str]:
    """Check a rendered root; returns what is wrong (empty when it is sound).

    Every file listed in the manifest must exist with its recorded sha256 and name its catalog and repeat. ``refs``
    are rendered again into a scratch folder inside ``root`` and must reproduce the recorded bytes, which is how a
    render is shown to be reproducible.
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
            again = render(refs, scratch, settings, ids=ids, jobs=jobs, log=lambda _: None)
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
        finally:
            shutil.rmtree(scratch, ignore_errors=True)
    return problems
