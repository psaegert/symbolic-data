"""Where the catalog builders read their inputs and write their outputs.

The inputs are the curated records of srbf's benchmark survey (catalog-prep, 2026-10..11): one record per problem
with its formula, constants, input distributions and source (final_problems.json), each benchmark set's name and
sources (sets.json), the authors' test specifications (testspecs.json) and the frozen 1,024-problem subsamples
(frozen/), and for the data sets each dataset's files, sha256, columns and split as published (data_sets.json). They
are vendored, compressed and with local paths removed, in assets/upstream/catalog-prep; SYMBOLIC_DATA_CATALOG_PREP
points the builders at another copy (a directory holding the same files, plain or .gz, or the survey's working tree
with sets/<tag>.json). Downloads are cached in SYMBOLIC_DATA_BUILD_CACHE (default
~/.cache/symbolic-data/catalog-build).

The data sets' files (1.1 GB, most of them under their authors' licenses) are not vendored: a record names each file
by its place in the survey's working copy (``catalog-prep/raw/<set>/...``), its source URL and its sha256, and
SYMBOLIC_DATA_CATALOG_RAW points at a folder holding them (default: the cache's raw/). The data catalogs are built
into DATA_BUILD (git-ignored) and published to Hugging Face only.
"""
from __future__ import annotations

import gzip
import json
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
PREP = Path(os.environ.get("SYMBOLIC_DATA_CATALOG_PREP") or REPO / "assets/upstream/catalog-prep")
CACHE = Path(os.environ.get("SYMBOLIC_DATA_BUILD_CACHE") or Path.home() / ".cache/symbolic-data/catalog-build")
CATALOGS = REPO / "assets/catalogs"
RAW = Path(os.environ.get("SYMBOLIC_DATA_CATALOG_RAW") or CACHE / "raw")
DATA_BUILD = REPO / "build/data-catalogs"


def _read(name: str) -> Any:
    for path, op in ((PREP / f"{name}.gz", gzip.open), (PREP / name, open)):
        if path.exists():
            with op(path, "rt", encoding="utf-8") as fh:
                return json.load(fh)
    raise FileNotFoundError(f"{name}(.gz) not in {PREP}")


@lru_cache(maxsize=1)
def final_problems() -> list[dict[str, Any]]:
    """Every problem record (``set``, ``id``, ``kind`` F/D, ``unit``, ``formula``, ``variables``, ...)."""
    return _read("final_problems.json")["problems"]


@lru_cache(maxsize=None)
def set_record(tag: str) -> dict[str, Any]:
    """A benchmark set's record (``name``, ``kind``, ``sources``, ...)."""
    if (PREP / "sets").is_dir():
        with open(PREP / "sets" / f"{tag}.json", encoding="utf-8") as fh:
            return json.load(fh)
    return _sets()[tag]


@lru_cache(maxsize=1)
def _sets() -> dict[str, dict[str, Any]]:
    return _read("sets.json")


@lru_cache(maxsize=None)
def data_set_record(tag: str) -> dict[str, Any]:
    """A data set's full record: its ``datasets`` (files, sha256, columns, split as published) and notes."""
    if (PREP / "sets").is_dir():
        return set_record(tag)
    return _data_sets()[tag]


@lru_cache(maxsize=1)
def _data_sets() -> dict[str, dict[str, Any]]:
    return _read("data_sets.json")


def input_file(name: str | os.PathLike) -> Path:
    """The local copy of a file a record names: ``catalog-prep/raw/...`` under RAW, ``symbolic-data/...`` in this
    repository, anything else as given (the survey's working tree names its files by absolute path)."""
    text = str(name)
    if text.startswith("catalog-prep/raw/"):
        return RAW / text.removeprefix("catalog-prep/raw/")
    if text.startswith("symbolic-data/"):
        return REPO / text.removeprefix("symbolic-data/")
    return Path(text).expanduser()


@lru_cache(maxsize=1)
def testspecs() -> list[dict[str, Any]]:
    """The authors' test specifications, one record per problem with a test."""
    return _read("testspecs.json")["problems"]


def frozen_path(name: str) -> Path:
    """A frozen subsample's list (``frozen/<name>``)."""
    return PREP / "frozen" / name


def release_subsets() -> dict[str, Any]:
    return _read("release_subsets.json")


# text copied from the records into a catalog is public: no local paths, no internal wording
_LOCAL = [(re.compile(r"/home/[^/\s]+/srbf_eval_local/workflow_archive/"), ""),
          (re.compile(r"/home/[^/\s]+/srbf_eval_local/"), ""),
          (re.compile(r"/home/[^/\s]+/Projects/symbolic-data(-wt-[\w-]+)?/"), "symbolic-data/"),
          (re.compile(r"/home/[^/\s]+/"), "~/"),
          (re.compile(r"\bOWNER DECISION\b"), "DECISION"),
          (re.compile(r"\bowner call\b"), "srbf's call"),
          # the survey's notes name srbf's maintainer as "the owner"; a data owner and owner-occupied homes stay
          (re.compile(r"(?<![Dd]ata )\b(?:[Tt]he )?[Oo]wner\b(?!-occupied)"), "srbf")]


def public(value: Any) -> Any:
    """``value`` with every string's local paths and internal wording replaced (recursively through dicts and lists)."""
    if isinstance(value, str):
        for pattern, repl in _LOCAL:
            value = pattern.sub(repl, value)
        return value
    if isinstance(value, dict):
        return {k: public(v) for k, v in value.items()}
    if isinstance(value, list):
        return [public(v) for v in value]
    return value
