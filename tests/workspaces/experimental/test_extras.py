"""The sandbox extra is checked when the workspace is built, not on open.

These run with or without the extra installed: a `None` entry in
`sys.modules` makes the import fail the way a missing package does.
"""

import sys
import types

import pytest

import ai
from ai.workspaces import experimental as workspaces


def test_vercel_sandbox_needs_its_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "vercel.sandbox", None)
    with pytest.raises(ai.InstallationError, match=r"ai\[sandbox\]"):
        workspaces.VercelSandbox()


def test_vercel_sandbox_needs_interactive_pty_support(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # vercel-sandbox 0.7.0 on PyPI: the module is there, the feature is not.
    released = types.ModuleType("vercel.sandbox")
    released.__dict__["Sandbox"] = type("Sandbox", (), {})
    monkeypatch.setitem(sys.modules, "vercel.sandbox", released)
    with pytest.raises(ai.InstallationError, match="open_interactive"):
        workspaces.VercelSandbox()


def test_local_needs_no_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "vercel.sandbox", None)
    assert workspaces.Local(".").kind == "local"
