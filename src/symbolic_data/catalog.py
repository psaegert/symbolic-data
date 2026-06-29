"""The level-1 data artifact: a :class:`ProblemCatalog`.

A catalog is a declarative, versioned, reusable collection of symbolic-regression problems --
expressions plus their *intrinsic* per-variable sampling -- with no usage policy (holdouts,
draw method, noise are the :class:`~symbolic_data.source.ProblemSource`'s job, level 2).

On-disk a catalog is a single yaml with two top-level blocks::

    metadata: {name, version, description, sources, sampling_defaults, conventions, ...}
    expressions:
      <eq_id>: {raw, prepared, n_variables, vars: {vN: {name, sample_range, sample_type, ...}}, ...}

Catalogs are loaded through :func:`load_catalog`, which resolves a reference (a local path, a
``name[@version]`` against the official manifest, or ``repo_id:name[@version]`` against a
third-party one) via :mod:`symbolic_data.resolver`, with vendored package-data as the offline
fallback for the curated sets.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import yaml

from symbolic_data.resolver import resolve

# Known curated catalog names (for discovery; all ship as vendored package data).
CATALOGS: tuple[str, ...] = ("fastsrb", "feynman", "nguyen")

# Keys consumed as structured fields on a CatalogEntry; everything else on an entry -> entry.meta.
_ENTRY_FIELDS = {"raw", "prepared", "n_variables", "vars"}


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
class ProblemCatalog:
    """A declarative, versioned collection of problem templates (level 1)."""

    name: str
    version: int | None
    entries: dict[str, CatalogEntry]
    meta: dict[str, Any] = field(default_factory=dict)
    frozen: bool = False
    source: str | None = None

    # --- construction -------------------------------------------------------------------------
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

        if "expressions" in mapping:
            meta = dict(mapping.get("metadata", {}))
            raw_entries = mapping["expressions"]
        else:
            # flat form: every top-level key is an expression (no metadata block)
            meta = {}
            raw_entries = mapping
        entries = {eq_id: CatalogEntry.from_mapping(eq_id, body) for eq_id, body in raw_entries.items()}
        return cls(
            name=name or meta.get("name", "catalog"),
            version=version if version is not None else meta.get("version"),
            entries=entries,
            meta=meta,
            source=source,
        )

    @classmethod
    def load(cls, ref: str = "fastsrb", *, install: bool = True, repo_id: str | None = None) -> "ProblemCatalog":
        artifact = resolve(ref, install=install, repo_id=repo_id)
        return cls.from_yaml(artifact.path, version=artifact.version, source=artifact.source)

    # --- persistence --------------------------------------------------------------------------
    def to_mapping(self) -> dict[str, Any]:
        meta = dict(self.meta)
        meta.setdefault("name", self.name)
        if self.version is not None:
            meta.setdefault("version", self.version)
        return {"metadata": meta, "expressions": {e.id: e.to_mapping() for e in self.entries.values()}}

    def to_yaml(self, path: str | Path) -> None:
        Path(path).write_text(yaml.safe_dump(self.to_mapping(), sort_keys=False, allow_unicode=True), encoding="utf-8")

    # --- access -------------------------------------------------------------------------------
    def iter_expressions(self) -> Iterator[CatalogEntry]:
        return iter(self.entries.values())

    def __iter__(self) -> Iterator[CatalogEntry]:
        return self.iter_expressions()

    def __len__(self) -> int:
        return len(self.entries)

    def __contains__(self, eq_id: str) -> bool:
        return eq_id in self.entries

    def __getitem__(self, eq_id: str) -> CatalogEntry:
        return self.entries[eq_id]


def load_catalog(ref: str = "fastsrb", *, install: bool = True, repo_id: str | None = None) -> ProblemCatalog:
    """Load a :class:`ProblemCatalog` by reference.

    ``ref`` is a local path, a ``name[@version]`` (resolved against the official manifest, with
    the vendored curated copy as the offline fallback), or ``repo_id:name[@version]`` for a
    third-party catalog. Curated names: ``fastsrb`` (120), ``feynman`` (100), ``nguyen`` (12).
    """
    return ProblemCatalog.load(ref, install=install, repo_id=repo_id)
