"""The claude-code extra is checked when the harness is built, not later.

These run with or without the extra installed: a `None` entry in
`sys.modules` makes the import fail the way a missing package does.
"""

import subprocess
import sys

import pytest

import ai
from ai.harnesses import experimental as harnesses
from ai.workspaces import experimental as workspaces


def test_claude_code_needs_its_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "claude_agent_sdk", None)
    with pytest.raises(ai.InstallationError, match=r"ai\[claude-code\]"):
        harnesses.claude_code(workspace=workspaces.Local("."))


def test_codex_needs_no_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "claude_agent_sdk", None)
    harness = harnesses.codex(workspace=workspaces.Local("."))
    assert harness.name == "codex"


def test_import_ai_loads_no_optional_harness_package() -> None:
    code = (
        "import sys, ai\n"
        "from ai.harnesses.experimental import *\n"
        "from ai.workspaces.experimental import *\n"
        "loaded = [m for m in sys.modules\n"
        "          if m.split('.')[0] == 'claude_agent_sdk'\n"
        "          or m.startswith('vercel.sandbox')]\n"
        "assert not loaded, loaded\n"
        "assert 'termios' not in sys.modules or sys.platform != 'win32'\n"
        "assert 'ai.workspaces.experimental.tty' not in sys.modules\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
