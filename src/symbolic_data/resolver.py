"""Repo-agnostic resolver for versioned symbolic-data artifacts (catalogs, grids).

Catalogs are distributed as artifacts on Hugging Face (the artifact store), NOT bundled in the wheel.
A *reference* resolves to a local filesystem path. Three forms:

* a **local path** -- something that exists on disk, or is explicitly path-like (``./x.yaml``,
  ``/abs/x.yaml``, or contains a separator). Returned as-is (``source="local"``).
* ``name`` or ``name@version`` -- looked up in a manifest on a Hugging Face *dataset* repo (ours
  by default). The version pins a ``revision`` (git sha) and per-file ``sha256``; files are
  fetched via ``hf_hub_download`` (which caches them) and integrity-checked
  (``source="huggingface"``).
* ``repo_id:name`` / ``repo_id:name@version`` -- the same, against a THIRD-PARTY repo's manifest,
  so anyone can publish and load their own catalogs through this mechanism.

There is NO vendored package-data fallback: a bare ``name`` resolves only via the HF manifest, so a
fresh install needs network on first use (``hf_hub_download`` then caches). Use an explicit local
path for fully offline operation.

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
import warnings
from dataclasses import dataclass, field
from typing import Any

# Default manifest location (our official catalogs/grids). Third-party refs override the repo.
HF_MANIFEST_REPO = "psaegert/symbolic-data-assets"
HF_MANIFEST_FILENAME = "manifest.json"

_PATH_LIKE_SUFFIXES = (".yaml", ".yml", ".json", ".npz")


class ResolverError(RuntimeError):
    """Raised when a reference cannot be resolved (not a local path and not in the HF manifest)."""


class IntegrityError(ResolverError):
    """Raised when a fetched artifact fails sha256 verification. Never falls back silently."""


@dataclass
class ResolvedArtifact:
    """The outcome of resolving a reference."""

    path: str                       # absolute path to the entrypoint file
    name: str                       # logical name ("feynman"); the path stem for local refs
    source: str                     # "local" | "huggingface"
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
    # A bare name (no separator / leading dot/tilde / known suffix) is NEVER a path -- even if a
    # same-named file happens to exist in the CWD. Otherwise a stray file could silently shadow a
    # manifest name and bypass the versioned + sha256-checked path. Callers wanting a literal local
    # file pass an explicit path (``./name`` or an absolute path).
    return False


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


def resolve(
    ref: str,
    *,
    install: bool = True,
    repo_id: str | None = None,
    manifest_filename: str | None = None,
    verify: bool = True,
) -> ResolvedArtifact:
    """Resolve ``ref`` (local path / ``name[@version]`` / ``repo:name[@version]``) to a local file."""
    if not ref or not isinstance(ref, str):
        raise ValueError("ref must be a non-empty string")

    # A "repo_id:name[@version]" override is a manifest ref, never a local path -- check first so
    # the repo id's '/' does not trip path detection. The repo_id is a "user/repo" slug (exactly
    # one '/', not path-like), so a local path that merely contains a colon (e.g. a colon in a
    # directory name, or an absolute path) is NOT misclassified as a repo ref.
    _head = ref.split(":", 1)[0] if ":" in ref else ""
    is_repo_ref = bool(_head) and _head.count("/") == 1 and not _head.startswith((".", "~", os.sep))

    # 1. explicit local path
    if not is_repo_ref and _looks_like_path(ref):
        if not os.path.isfile(ref):
            raise ResolverError(f"Local artifact path does not exist: {ref!r}")
        return ResolvedArtifact(path=os.path.abspath(ref), name=os.path.splitext(os.path.basename(ref))[0], source="local")

    repo_override, name, version = _parse_ref(ref)
    effective_repo = repo_override or repo_id

    # 2. HF manifest lookup (+ integrity). No vendored fallback: a name resolves ONLY via the
    #    manifest, so an unknown name / offline first-use is an error (use a local path for offline).
    manifest = fetch_manifest(repo_id=effective_repo, manifest_filename=manifest_filename)
    entry = manifest.get(name) if manifest else None
    if entry is None:
        raise ResolverError(
            f"Could not resolve {ref!r}: not a local path and not in the manifest"
            f"{' (repo ' + effective_repo + ')' if effective_repo else ''}. "
            "Bare names resolve from Hugging Face (network needed on first use); pass an explicit "
            "local path for offline operation."
        )
    return _resolve_from_manifest(name, entry, version, install=install, verify=verify)


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
        if verify:
            if filename in sha:
                actual = _sha256(local)
                if actual != sha[filename]:
                    raise IntegrityError(f"sha256 mismatch for {name}@{resolved_version}/{filename}: expected {sha[filename]}, got {actual}")
            else:
                warnings.warn(
                    f"No sha256 in manifest for {name}@{resolved_version}/{filename}; integrity NOT verified for this file.",
                    RuntimeWarning, stacklevel=2,
                )
        paths[filename] = local

    return ResolvedArtifact(
        path=paths[files[0]], name=name, source="huggingface", version=int(resolved_version),
        repo_id=repo_id, revision=revision, files=files, paths=paths,
    )
