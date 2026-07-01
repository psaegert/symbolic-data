"""The level-1 data artifact: a :class:`ProblemCatalog`.

A catalog is a declarative, versioned, reusable collection of symbolic-regression problems --
expressions plus their *intrinsic* per-variable sampling -- with no usage policy (holdouts,
draw method, noise are the :class:`~symbolic_data.source.ProblemSource`'s job, level 2).

On-disk a catalog is a single yaml with two top-level blocks::

    metadata: {name, version, description, sources, sampling_defaults, conventions, ...}
    expressions:
      <eq_id>: {raw, prepared, n_variables, vars: {vN: {name, sample_range, sample_type, ...}}, ...}

Catalogs are loaded through :func:`load_catalog`, which resolves a reference (a local path, a
``name[@version]`` against the official Hugging Face manifest, or ``repo_id:name[@version]`` against
a third-party one) via :mod:`symbolic_data.resolver`. Curated catalogs are HF artifacts, not bundled
in the wheel, so a bare name needs network on first use; pass a local path for offline operation.
"""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import yaml

from symbolic_data._evaluation import broadcast_target, compile_expression, evaluate
from symbolic_data.distributions import fastsrb_dist
from symbolic_data.errors import CatalogEntryError, NoValidSampleFoundError
from symbolic_data.problem import Problem
from symbolic_data.resolver import resolve

# Known curated catalog names (for discovery; all resolved from the HF assets repo, not bundled).
CATALOGS: tuple[str, ...] = ("fastsrb", "feynman", "nguyen")

# Keys consumed as structured fields on a CatalogEntry; everything else on an entry -> entry.meta.
_ENTRY_FIELDS = {"raw", "prepared", "n_variables", "vars"}

# The per-Problem array fields a FROZEN catalog stores in its .npz sidecar.
_PROBLEM_ARRAY_FIELDS = ("x_support", "y_support", "y_support_noisy", "x_validation", "y_validation", "y_validation_noisy")


@dataclass
class CatalogEntry:
    """One catalog problem template: an expression + its intrinsic per-variable sampling."""

    id: str
    raw: str | None = None
    prepared: str | None = None
    n_variables: int | None = None
    variables: dict[str, dict[str, Any]] = field(default_factory=dict)   # the `vars` block (vN -> spec)
    meta: dict[str, Any] = field(default_factory=dict)                   # sources, moniker, accept, constraints, ...

    @classmethod
    def from_mapping(cls, eq_id: str, mapping: dict[str, Any]) -> "CatalogEntry":
        meta = {k: v for k, v in mapping.items() if k not in _ENTRY_FIELDS}
        return cls(
            id=eq_id,
            raw=mapping.get("raw"),
            prepared=mapping.get("prepared"),
            n_variables=mapping.get("n_variables"),
            variables=dict(mapping.get("vars", {})),
            meta=meta,
        )

    def to_mapping(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        if self.raw is not None:
            out["raw"] = self.raw
        if self.prepared is not None:
            out["prepared"] = self.prepared
        if self.n_variables is not None:
            out["n_variables"] = self.n_variables
        out.update(self.meta)
        if self.variables:
            out["vars"] = self.variables
        return out


@dataclass
class RealizedExpression:
    """One expression realized into raw (X, y) data plus its ground truth.

    The :class:`Catalog`'s *intrinsic* output (``n_points`` rows, before any usage policy): the
    :class:`~symbolic_data.source.ProblemSource` splits it into support/validation, injects noise,
    and wraps it into a :class:`~symbolic_data.problem.Problem`.
    """

    x: np.ndarray                         # (n_points, n_variables)
    y: np.ndarray                         # (n_points, 1)
    skeleton: tuple[str, ...] | None
    expression: list[str] | None          # constants substituted + normalized
    constants: list[float]
    variables: list[str]                  # column order of x
    complexity: int | None
    eq_id: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)


class Catalog(ABC):
    """The level-1 abstraction a :class:`~symbolic_data.source.ProblemSource` samples from.

    A catalog supplies *expressions* and knows how to realize each into raw (X, y) data via its
    *intrinsic* sampling (per-variable ranges for a declarative :class:`ProblemCatalog`; a learned
    generator for a :class:`~symbolic_data.generative.GenerativeCatalog`). It owns NO usage policy
    (draw counts, noise, holdouts) -- that is the ``ProblemSource``'s job.
    """

    name: str

    @abstractmethod
    def iter_entries(self, rng: np.random.Generator, *, method: str = "iterate", size: int | None = None) -> Iterator[Any]:
        """Yield the entries to realize (a finite catalog honors ``method``; a generative one
        produces fresh entries, ``size`` of them or unbounded when ``size`` is ``None``)."""

    @abstractmethod
    def realize(self, entry: Any, n_points: int, rng: np.random.Generator, *, engine: Any = None, layout: str = "random") -> RealizedExpression:
        """Sample ``n_points`` rows of (X, y) for one ``entry`` using the catalog's intrinsic
        sampling. Raise :class:`~symbolic_data.errors.NoValidSampleFoundError` on a transient
        failure (retryable) or :class:`~symbolic_data.errors.CatalogEntryError` on a permanent one.
        Subclasses ignore kwargs they do not need."""

    def is_finite(self) -> bool:
        return True


@dataclass
class ProblemCatalog(Catalog):
    """A declarative, versioned collection of problem templates (level 1)."""

    name: str
    version: int | None
    entries: dict[str, CatalogEntry]
    meta: dict[str, Any] = field(default_factory=dict)
    frozen: bool = False
    source: str | None = None
    problems: list[Problem] | None = None      # present iff frozen (materialized (X, y) data)

    # --- construction -------------------------------------------------------------------------
    @classmethod
    def from_problems(cls, problems: list[Problem], *, name: str = "materialized", version: int | None = 1, meta: dict[str, Any] | None = None) -> "ProblemCatalog":
        """Build a FROZEN catalog from realized Problems (the output of ``ProblemSource.materialize``)."""
        merged = dict(meta or {})
        merged.update({"name": name, "version": version, "frozen": True})
        return cls(name=name, version=version, entries={}, meta=merged, frozen=True, problems=list(problems))

    @classmethod
    def from_yaml(
        cls,
        path_or_mapping: str | Path | dict[str, Any],
        *,
        name: str | None = None,
        version: int | None = None,
        source: str | None = None,
    ) -> "ProblemCatalog":
        if isinstance(path_or_mapping, (str, Path)):
            mapping = yaml.safe_load(Path(path_or_mapping).read_text(encoding="utf-8"))
        else:
            mapping = path_or_mapping
        if not isinstance(mapping, dict):
            raise ValueError("catalog yaml must be a mapping")

        # Structured form: a top-level `metadata` and/or `expressions` block. `metadata` and
        # `expressions` are RESERVED top-level keys -- a flat catalog may not use them as eq_ids.
        # (Detecting on `expressions` alone would mis-eat a metadata-only catalog as a phantom
        # entry named "metadata".)
        if "expressions" in mapping or "metadata" in mapping:
            meta = dict(mapping.get("metadata", {}))
            raw_entries = mapping.get("expressions", {})
        else:
            # flat form: every top-level key is an expression (no metadata block)
            meta = {}
            raw_entries = mapping
        resolved_name = name or meta.get("name", "catalog")
        resolved_version = version if version is not None else meta.get("version")
        # Make the metadata block consistent with the resolved identity so the in-memory object
        # (cat.meta["version"] == cat.version) and any to_yaml round-trip preserve the loaded
        # name/version -- the arg/resolver wins over a possibly-stale embedded block.
        meta["name"] = resolved_name
        if resolved_version is not None:
            meta["version"] = resolved_version
        entries = {eq_id: CatalogEntry.from_mapping(eq_id, body) for eq_id, body in raw_entries.items()}
        return cls(
            name=resolved_name,
            version=resolved_version,
            entries=entries,
            meta=meta,
            source=source,
        )

    @classmethod
    def load(cls, ref: str = "fastsrb", *, install: bool = True, repo_id: str | None = None) -> "ProblemCatalog":
        artifact = resolve(ref, install=install, repo_id=repo_id)
        if str(artifact.path).endswith(".npz"):
            return cls.from_npz(artifact.path)
        return cls.from_yaml(artifact.path, version=artifact.version, source=artifact.source)

    @classmethod
    def from_npz(cls, path: str | Path) -> "ProblemCatalog":
        """Load a FROZEN catalog (a self-contained ``.npz`` written by :meth:`save`)."""
        data = np.load(path, allow_pickle=False)
        blob = json.loads(str(data["_meta"].item()))
        cat = blob["catalog"]
        problems: list[Problem] = []
        for i, scalar in enumerate(blob["problems"]):
            kwargs = dict(scalar)
            kwargs["skeleton"] = tuple(kwargs["skeleton"]) if kwargs.get("skeleton") is not None else None
            for fld in _PROBLEM_ARRAY_FIELDS:
                kwargs[fld] = data[f"p{i}__{fld}"]
            problems.append(Problem.from_dict(kwargs))
        return cls(name=cat["name"], version=cat.get("version"), entries={}, meta=cat.get("meta", {}), frozen=True, problems=problems, source="local")

    # --- persistence --------------------------------------------------------------------------
    def to_mapping(self) -> dict[str, Any]:
        meta = dict(self.meta)
        # Assignment (not setdefault): self.name/self.version are authoritative over a possibly
        # stale embedded block, so a loaded-then-saved catalog preserves its resolved identity
        # rather than silently reverting/downgrading the version.
        meta["name"] = self.name
        if self.version is not None:
            meta["version"] = self.version
        return {"metadata": meta, "expressions": {e.id: e.to_mapping() for e in self.entries.values()}}

    def to_yaml(self, path: str | Path) -> None:
        Path(path).write_text(yaml.safe_dump(self.to_mapping(), sort_keys=False, allow_unicode=True), encoding="utf-8")

    def save(self, path: str | Path) -> Path:
        """Persist the catalog. FROZEN catalogs -> a self-contained ``.npz``; declarative -> ``.yaml``."""
        path = Path(path)
        if self.frozen:
            if path.suffix != ".npz":
                path = path.with_suffix(".npz")
            arrays: dict[str, np.ndarray] = {}
            scalars: list[dict[str, Any]] = []
            for i, p in enumerate(self.problems or []):
                for fld in _PROBLEM_ARRAY_FIELDS:
                    arrays[f"p{i}__{fld}"] = np.asarray(getattr(p, fld))
                scalars.append({
                    "skeleton": [str(t) for t in p.skeleton] if p.skeleton is not None else None,
                    "expression": [str(t) for t in p.expression] if p.expression is not None else None,
                    "constants": [float(c) for c in p.constants],
                    "variables": [str(v) for v in p.variables],
                    "complexity": p.complexity,
                    "noise": p.noise,
                    "eq_id": p.eq_id,
                    "meta": p.meta,
                    "is_placeholder": p.is_placeholder,
                    "placeholder_reason": p.placeholder_reason,
                })
            blob = json.dumps({"catalog": {"name": self.name, "version": self.version, "meta": self.meta}, "problems": scalars})
            np.savez(path, _meta=np.array(blob), **arrays)
        else:
            if path.suffix not in (".yaml", ".yml"):
                path = path.with_suffix(".yaml")
            self.to_yaml(path)
        return path

    # --- access -------------------------------------------------------------------------------
    def iter_expressions(self) -> Iterator[CatalogEntry]:
        if self.frozen:
            raise TypeError(
                "a frozen ProblemCatalog holds realized Problems, not declarative expression entries; "
                "iterate its `.problems` (or a ProblemSource over it) instead of `iter_expressions()`."
            )
        return iter(self.entries.values())

    def __iter__(self) -> Iterator[CatalogEntry]:
        return self.iter_expressions()

    def __len__(self) -> int:
        return len(self.entries)

    def __contains__(self, eq_id: str) -> bool:
        return eq_id in self.entries

    def __getitem__(self, eq_id: str) -> CatalogEntry:
        return self.entries[eq_id]

    def is_finite(self) -> bool:
        return True

    # --- Catalog interface (sampled by ProblemSource) -----------------------------------------
    def iter_entries(self, rng: np.random.Generator, *, method: str = "iterate", size: int | None = None) -> Iterator[CatalogEntry]:
        """Yield catalog entries in the order dictated by the draw ``method`` (``size`` ignored:
        a declarative catalog has a fixed entry set)."""
        eq_ids = list(self.entries.keys())
        if method == "iterate":
            order = eq_ids
        elif method == "random_without_replacement":
            order = [eq_ids[i] for i in rng.permutation(len(eq_ids))]
        elif method == "random_with_replacement":
            order = [eq_ids[i] for i in rng.integers(0, len(eq_ids), size=len(eq_ids))]
        else:
            raise ValueError(f"Unknown set draw method {method!r}; expected iterate | random_without_replacement | random_with_replacement")
        for eq_id in order:
            yield self.entries[eq_id]

    def realize(self, entry: CatalogEntry, n_points: int, rng: np.random.Generator, *, engine: Any = None, layout: str = "random") -> RealizedExpression:
        """Realize one entry: sample each variable from its intrinsic ``fastsrb`` range and evaluate."""
        if engine is None:
            raise ValueError("declarative ProblemCatalog.realize requires a simplipy engine (pass engine=...)")
        compiled = self._compiled(entry, engine)
        variable_order = compiled["variable_order"]
        columns = []
        try:
            for key in variable_order:
                spec = entry.variables[key]
                base, sign = spec["sample_type"]
                low, high = spec["sample_range"]
                columns.append(fastsrb_dist(low, high, base=base, sign=sign, layout=layout, size=n_points, rng=rng))
        except (KeyError, ValueError, TypeError) as exc:
            # A missing/malformed per-variable spec is a permanent entry defect (not a transient draw
            # failure): raise CatalogEntryError so the source yields a placeholder instead of crashing
            # the whole iteration -- mirroring _compiled()'s compile_failed handling.
            raise CatalogEntryError(f"malformed variable spec for {entry.id!r}: {exc}") from exc
        x_all = np.column_stack(columns).astype(float)
        value_map = {var: x_all[:, i] for i, var in enumerate(variable_order)}
        try:
            y_all = broadcast_target(evaluate(compiled, value_map), n_points, entry.id).reshape(-1, 1)
        except Exception as exc:
            raise NoValidSampleFoundError(f"evaluation failed for {entry.id!r}: {exc}") from exc
        if not (np.all(np.isfinite(x_all)) and np.all(np.isfinite(y_all))):
            raise NoValidSampleFoundError(f"non-finite support/target for {entry.id!r}")
        return RealizedExpression(
            x=x_all, y=y_all,
            skeleton=tuple(compiled["prefix"]), expression=list(compiled["prefix"]),
            constants=[], variables=list(variable_order), complexity=len(compiled["prefix"]),
            eq_id=entry.id,
            meta={"prepared_normalized": compiled["normalized_infix"], **{k: v for k, v in entry.meta.items() if k != "sources"}},
        )

    def _compiled(self, entry: CatalogEntry, engine: Any) -> dict[str, Any]:
        cache = self.__dict__.setdefault("_compiled_cache", {})
        if entry.id not in cache:
            try:
                cache[entry.id] = compile_expression(engine, entry.id, entry.prepared, entry.variables, name=self.name)
            except Exception as exc:  # malformed entry -> permanent failure (placeholder, no retry)
                raise CatalogEntryError(f"compile_failed: {exc}") from exc
        return cache[entry.id]


def load_catalog(ref: str = "fastsrb", *, install: bool = True, repo_id: str | None = None) -> ProblemCatalog:
    """Load a :class:`ProblemCatalog` by reference.

    ``ref`` is a local path, a ``name[@version]`` (resolved from the official Hugging Face manifest;
    network on first use, then cached), or ``repo_id:name[@version]`` for a third-party catalog.
    Curated names: ``fastsrb`` (120), ``feynman`` (100), ``nguyen`` (12).
    """
    return ProblemCatalog.load(ref, install=install, repo_id=repo_id)
