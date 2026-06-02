"""Unit tests for the git-provenance helper at common/version_info.py.

The helper resolves provenance via a fallback chain:
  1. baked _version module produced by CMake configure_file()
  2. runtime ``git rev-parse HEAD`` + ``git status --porcelain``
  3. sentinel ``{'git_sha': 'unknown', 'git_dirty': False}``

Result is memoized at module scope so we only touch the filesystem/subprocess
once per Slicer session.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest


# ---------- Fixtures ----------------------------------------------------------

@pytest.fixture
def version_info(monkeypatch):
    """Provide common.version_info with a clean cache, restored after the test."""
    # Ensure no stale _version module from a previous test polluted sys.modules.
    monkeypatch.delitem(sys.modules, "_version", raising=False)

    from .. import version_info as mod

    mod._reset_cache_for_tests()
    yield mod
    mod._reset_cache_for_tests()


@pytest.fixture
def baked_version_module(monkeypatch):
    """Factory: inject a stub `_version` module into sys.modules."""

    def _make(sha: str = "a" * 40, dirty: bool = False, *, omit_sha: bool = False):
        stub = types.ModuleType("_version")
        if not omit_sha:
            stub.GIT_SHA = sha
        stub.GIT_DIRTY = dirty
        monkeypatch.setitem(sys.modules, "_version", stub)
        return stub

    return _make


@pytest.fixture
def no_baked_version(monkeypatch):
    """Force ``import _version`` to raise ImportError inside the helper."""
    monkeypatch.setitem(sys.modules, "_version", None)


# ---------- Helpers -----------------------------------------------------------

def _completed_proc(stdout: str = "", returncode: int = 0):
    proc = MagicMock()
    proc.stdout = stdout
    proc.returncode = returncode
    proc.stderr = ""
    return proc


def _fake_git_run(sha: str, dirty: bool):
    """Mock subprocess.run that responds to git rev-parse and git status."""

    def run(cmd, *args, **kwargs):
        if "rev-parse" in cmd:
            return _completed_proc(stdout=sha + "\n", returncode=0)
        if "status" in cmd:
            return _completed_proc(
                stdout=" M file.py\n" if dirty else "",
                returncode=0,
            )
        return _completed_proc()

    return run


# ---------- Tests -------------------------------------------------------------

def test_returns_baked_values_when_version_module_present(
    version_info, baked_version_module
):
    """Baked _version is the source of truth when present."""
    baked_version_module(sha="b" * 40, dirty=True)

    result = version_info.get_provenance()

    assert result == {"git_sha": "b" * 40, "git_dirty": True}


def test_falls_back_to_git_when_baked_file_missing(
    version_info, no_baked_version, monkeypatch
):
    """No _version → spawn `git rev-parse HEAD` and use its output."""
    monkeypatch.setattr("subprocess.run", _fake_git_run(sha="f" * 40, dirty=False))
    monkeypatch.setattr(version_info, "_find_git_root", lambda _: Path("/repo"))

    result = version_info.get_provenance()

    assert result == {"git_sha": "f" * 40, "git_dirty": False}


def test_detects_dirty_when_git_status_porcelain_nonempty(
    version_info, no_baked_version, monkeypatch
):
    """`git status --porcelain` with output flips git_dirty to True."""
    monkeypatch.setattr("subprocess.run", _fake_git_run(sha="1" * 40, dirty=True))
    monkeypatch.setattr(version_info, "_find_git_root", lambda _: Path("/repo"))

    result = version_info.get_provenance()

    assert result["git_sha"] == "1" * 40
    assert result["git_dirty"] is True


def test_returns_unknown_when_git_unavailable(
    version_info, no_baked_version, monkeypatch
):
    """Missing `git` binary surfaces as the unknown sentinel — never blocks."""

    def raise_fnf(*_a, **_kw):
        raise FileNotFoundError("git not installed")

    monkeypatch.setattr("subprocess.run", raise_fnf)
    monkeypatch.setattr(version_info, "_find_git_root", lambda _: Path("/repo"))

    result = version_info.get_provenance()

    assert result == {"git_sha": "unknown", "git_dirty": False}


def test_returns_unknown_when_not_in_git_repo(
    version_info, no_baked_version, monkeypatch
):
    """No `.git/` found by the walk → unknown WITHOUT spawning git."""
    monkeypatch.setattr(version_info, "_find_git_root", lambda _: None)

    calls: list = []

    def tracking_run(*args, **kwargs):
        calls.append((args, kwargs))
        return _completed_proc()

    monkeypatch.setattr("subprocess.run", tracking_run)

    result = version_info.get_provenance()

    assert result == {"git_sha": "unknown", "git_dirty": False}
    assert calls == [], "Helper must skip subprocess when no .git is present"


def test_handles_malformed_baked_file(version_info, baked_version_module, monkeypatch):
    """_version missing GIT_SHA attr → fall back to the git path, not crash."""
    baked_version_module(omit_sha=True, dirty=False)

    monkeypatch.setattr("subprocess.run", _fake_git_run(sha="c" * 40, dirty=False))
    monkeypatch.setattr(version_info, "_find_git_root", lambda _: Path("/repo"))

    result = version_info.get_provenance()

    assert result["git_sha"] == "c" * 40
    assert result["git_dirty"] is False


def test_result_is_memoized_per_process(version_info, no_baked_version, monkeypatch):
    """get_provenance() runs the resolution chain at most once."""
    monkeypatch.setattr(version_info, "_find_git_root", lambda _: Path("/repo"))

    counter = MagicMock(side_effect=_fake_git_run(sha="d" * 40, dirty=False))
    monkeypatch.setattr("subprocess.run", counter)

    first = version_info.get_provenance()
    calls_after_first = counter.call_count
    second = version_info.get_provenance()

    assert first == second
    assert counter.call_count == calls_after_first, (
        "Second call must not spawn additional git subprocesses"
    )
