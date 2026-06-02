"""Git-provenance helper for de-identified DICOM outputs.

Embeds the build identity (``git_sha`` + ``git_dirty``) into every export so a
de-identified DICOM can be traced back to the exact commit that produced it.

Resolution chain (first hit wins):
  1. Baked ``_version`` module — generated at build time by CMake
     ``configure_file()``. Required for installed extensions where no ``.git``
     directory is present.
  2. Runtime ``git rev-parse HEAD`` + ``git status --porcelain`` against the
     repository containing this module. For developer / dev-clone workflows.
  3. Sentinel ``{'git_sha': 'unknown', 'git_dirty': False}`` — export is
     never blocked.

Result is memoized at module scope: at most one filesystem walk and one
subprocess invocation per Slicer session.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Dict, Optional, Union

# Public type alias for callers.
Provenance = Dict[str, Union[str, bool]]

# Sentinel for the "never resolved" cache state. We use ``None`` rather than a
# ``functools.lru_cache`` decorator so tests can reset it deterministically.
_PROVENANCE_CACHE: Optional[Provenance] = None

_UNKNOWN_PROVENANCE: Provenance = {"git_sha": "unknown", "git_dirty": False}

# DICOM private group + creator string scoping the provenance tag. Group
# 0x0099 is unassigned in DICOM PS3.6; creator string ``99BWHLUS`` (Brigham
# and Women's Hospital, Lung Ultrasound) uniquely scopes our sub-tag so we
# don't collide with other private blocks at the same group.
_PRIVATE_GROUP = 0x0099
_PRIVATE_CREATOR = "99BWHLUS"

_GIT_TIMEOUT_SECONDS = 5


def get_provenance() -> Provenance:
    """Return memoized ``{'git_sha': str, 'git_dirty': bool}``.

    Always succeeds — falls through to the unknown sentinel on any failure.
    """
    global _PROVENANCE_CACHE
    if _PROVENANCE_CACHE is None:
        _PROVENANCE_CACHE = _fetch_provenance()
    # Defensive copy so callers can't mutate the cache by accident.
    return dict(_PROVENANCE_CACHE)


def apply_provenance_tag(dataset) -> None:
    """Attach git provenance as a DICOM private tag on ``dataset``.

    Writes the following private block (creates if absent):

      (0x0099, 0x0010)   "99BWHLUS"                                       VR=LO
      (0x0099, 0x1000)   '{"git_sha": "...", "git_dirty": ...}'           VR=LT

    LT (Long Text, 10240 chars) is used instead of LO (64 chars) because the
    JSON payload (~85 chars) exceeds LO's limit.
    """
    provenance = get_provenance()
    block = dataset.private_block(_PRIVATE_GROUP, _PRIVATE_CREATOR, create=True)
    block.add_new(0x00, "LT", json.dumps(provenance))


# ---------- Internal resolution chain ----------------------------------------

def _fetch_provenance() -> Provenance:
    """Run the resolution chain once. No caching."""
    baked = _read_baked_version()
    if baked is not None:
        return baked

    runtime = _read_runtime_git()
    if runtime is not None:
        return runtime

    return dict(_UNKNOWN_PROVENANCE)


def _read_baked_version() -> Optional[Provenance]:
    """Try to import the build-time ``_version`` module. None on any failure."""
    try:
        import _version  # type: ignore[import-not-found]
    except ImportError:
        return None
    try:
        return {
            "git_sha": str(_version.GIT_SHA),
            "git_dirty": bool(_version.GIT_DIRTY),
        }
    except AttributeError:
        # Malformed baked file — fall through to git path.
        return None


def _read_runtime_git() -> Optional[Provenance]:
    """Spawn ``git`` against the module's repo root. None on any failure."""
    module_dir = Path(__file__).resolve().parent
    repo_root = _find_git_root(module_dir)
    if repo_root is None:
        return None

    try:
        sha_result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            capture_output=True,
            check=False,
            timeout=_GIT_TIMEOUT_SECONDS,
            text=True,
        )
        if sha_result.returncode != 0:
            return None

        status_result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(repo_root),
            capture_output=True,
            check=False,
            timeout=_GIT_TIMEOUT_SECONDS,
            text=True,
        )
        return {
            "git_sha": sha_result.stdout.strip(),
            "git_dirty": bool(status_result.stdout.strip()),
        }
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None


def _find_git_root(start: Path) -> Optional[Path]:
    """Walk up from ``start`` looking for a ``.git`` entry. None if not found."""
    for candidate in [start, *start.parents]:
        if (candidate / ".git").exists():
            return candidate
    return None


def _reset_cache_for_tests() -> None:
    """Reset memoization. Intended for tests only; do not call from production."""
    global _PROVENANCE_CACHE
    _PROVENANCE_CACHE = None
