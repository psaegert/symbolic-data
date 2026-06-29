"""Repo-agnostic resolver for versioned symbolic-data artifacts (catalogs, grids).

A *reference* resolves to a local filesystem path. Three forms:

* a **local path** -- something that exists on disk, or is explicitly path-like (``./x.yaml``,
  ``/abs/x.yaml``, or contains a separator). Returned as-is (``source="local"``).
* ``name`` or ``name@version`` -- looked up in a manifest on a Hugging Face *dataset* repo (ours
  by default). The version pins a ``revision`` (git sha) and per-file ``sha256``; files are
  fetched via ``hf_hub_download`` (which caches them) and integrity-checked
  (``source="huggingface"``).
* ``repo_id:name`` / ``repo_id:name@version`` -- the same, against a THIRD-PARTY repo's manifest,
  so anyone can publish and load their own catalogs through this mechanism.

If the manifest cannot be fetched or the name is unknown, a vendored package-data copy is used
when available (``source="vendored"``) so the curated catalogs work offline.

Design notes (avoiding the pitfalls of the simplipy ``asset_manager`` this generalizes):
* we rely on ``hf_hub_download``'s own content-addressed cache -- no custom install/uninstall and
  therefore no ``shutil.rmtree`` of a cache root on a partial fetch;
* a bare manifest *name* never silently resolves to a same-named local file -- a reference is
  treated as local only when it is explicitly path-like (see :func:`_looks_like_path`).
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from importlib import resources
from typing import Any

# Default manifest location (our official catalogs/grids). Third-party refs override the repo.
HF_MANIFEST_REPO = "psaegert/symbolic-data-assets"
HF_MANIFEST_FILENAME = "manifest.json"

# Where vendored copies of the curated artifacts ship as package data (offline fallback).
VENDORED_PACKAGE = "symbolic_data.catalogs"
VENDORED_SUBDIR = "data"

_PATH_LIKE_SUFFIXES = (".yaml", ".yml", ".json", ".npz")


class ResolverError(RuntimeError):
    """Raised when a reference cannot be resolved (and no vendored fallback applies)."""


@dataclass
class ResolvedArtifact:
    """The outcome of resolving a reference."""

    path: str                       # absolute path to the entrypoint file
    name: str                       # logical name ("feynman"); the path stem for local refs
    source: str                     # "local" | "huggingface" | "vendored"
    version: int | None = None
    repo_id: str | None = None
    revision: str | None = None
    files: list[str] = field(default_factory=list)
    paths: dict[str, str] = field(default_factory=dict)   # file -> local path (all fetched files)


def _looks_like_path(ref: str) -> bool:
    """True iff ``ref`` should be treated as a filesystem path rather than a manifest name."""
    if os.sep in ref or (os.altsep and os.altsep in ref):
        return True
    if ref.startswith((".", "~")):
        return True
    if ref.lower().endswith(_PATH_LIKE_SUFFIXES):
        return True
    return os.path.isfile(ref)


def _parse_ref(ref: str) -> tuple[str | None, str, int | None]:
    """Split ``[repo_id:]name[@version]`` into (repo_id | None, name, version | None).

    ``repo_id`` itself contains exactly one ``/`` (``user/repo``); we only treat a leading
    ``a/b:`` as a repo override, never a path (paths are handled before this is called).
    """
    repo_id: str | None = None
    body = ref
    if ":" in ref:
        head, _, tail = ref.partition(":")
        if "/" in head:
            repo_id, body = head, tail
    name, _, version_str = body.partition("@")
    version = int(version_str) if version_str else None
    return repo_id, name, version


def fetch_manifest(repo_id: str | None = None, manifest_filename: str | None = None) -> dict[str, Any]:
    """Download + parse the manifest from an HF dataset repo. Returns ``{}`` on failure."""
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import HfHubHTTPError

    try:
        manifest_path = hf_hub_download(
            repo_id=repo_id or HF_MANIFEST_REPO,
            filename=manifest_filename or HF_MANIFEST_FILENAME,
            repo_type="dataset",
        )
    except (HfHubHTTPError, OSError):
        return {}
    with open(manifest_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _vendored_path(name: str) -> str | None:
    """Path to a vendored package-data copy of ``<name>`` (offline fallback), or None."""
    for suffix in (".yaml", ".yml"):
        ref = resources.files(VENDORED_PACKAGE).joinpath(VENDORED_SUBDIR, f"{name}{suffix}")
        try:
            if ref.is_file():
                return str(ref)
        except (FileNotFoundError, OSError, ModuleNotFoundError):
            continue
    return None


def resolve(
    ref: str,
    *,
    install: bool = True,
    repo_id: str | None = None,
    manifest_filename: str | None = None,
    vendored_fallback: bool = True,
    verify: bool = True,
) -> ResolvedArtifact:
    """Resolve ``ref`` (local path / ``name[@version]`` / ``repo:name[@version]``) to a local file."""
    if not ref or not isinstance(ref, str):
        raise ValueError("ref must be a non-empty string")

    # A "repo_id:name[@version]" override (one '/' in the part before ':') is a manifest ref,
    # never a local path -- check this first so the repo id's '/' does not trip path detection.
    is_repo_ref = ":" in ref and "/" in ref.split(":", 1)[0]

    # 1. explicit local path
    if not is_repo_ref and _looks_like_path(ref):
        if not os.path.isfile(ref):
            raise ResolverError(f"Local artifact path does not exist: {ref!r}")
        return ResolvedArtifact(path=os.path.abspath(ref), name=os.path.splitext(os.path.basename(ref))[0], source="local")

    repo_override, name, version = _parse_ref(ref)
    effective_repo = repo_override or repo_id

    # 2. manifest lookup (+ integrity)
    manifest = fetch_manifest(repo_id=effective_repo, manifest_filename=manifest_filename)
    entry = manifest.get(name) if manifest else None
    if entry is not None:
        try:
            return _resolve_from_manifest(name, entry, version, install=install, verify=verify)
        except ResolverError:
            if not vendored_fallback:
                raise

    # 3. vendored package-data fallback (offline / unknown name)
    if vendored_fallback:
        vendored = _vendored_path(name)
        if vendored is not None:
            return ResolvedArtifact(path=vendored, name=name, source="vendored", version=version)

    raise ResolverError(
        f"Could not resolve {ref!r}: not a local path, not in the manifest"
        f"{' (repo ' + effective_repo + ')' if effective_repo else ''}, and no vendored copy."
    )


def _resolve_from_manifest(
    name: str,
    entry: dict[str, Any],
    version: int | None,
    *,
    install: bool,
    verify: bool,
) -> ResolvedArtifact:
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import HfHubHTTPError

    versions = entry.get("versions", {})
    resolved_version = version if version is not None else entry.get("default_version")
    if resolved_version is None or str(resolved_version) not in versions:
        raise ResolverError(f"Version {resolved_version!r} not found for {name!r}; have {sorted(versions)}")
    ventry = versions[str(resolved_version)]

    if not install:
        raise ResolverError(f"{name}@{resolved_version} not installed and install=False")

    repo_id = ventry.get("repo_id", entry.get("repo_id"))
    directory = ventry.get("directory", "")
    files = list(ventry.get("files", []))
    revision = ventry.get("revision")
    sha = ventry.get("sha256", {})
    if not files:
        raise ResolverError(f"Manifest entry for {name}@{resolved_version} lists no files")

    paths: dict[str, str] = {}
    for filename in files:
        remote = f"{directory}/{filename}" if directory else filename
        try:
            local = hf_hub_download(repo_id=repo_id, filename=remote, repo_type="dataset", revision=revision)
        except (HfHubHTTPError, OSError) as exc:
            raise ResolverError(f"Failed to fetch {remote} from {repo_id}: {exc}") from exc
        if verify and filename in sha:
            actual = _sha256(local)
            if actual != sha[filename]:
                raise ResolverError(f"sha256 mismatch for {name}@{resolved_version}/{filename}: expected {sha[filename]}, got {actual}")
        paths[filename] = local

    return ResolvedArtifact(
        path=paths[files[0]], name=name, source="huggingface", version=int(resolved_version),
        repo_id=repo_id, revision=revision, files=files, paths=paths,
    )
