"""Which CLI a sandbox gets: the latest, without pulling it out from under
an agent that is still running there."""

from __future__ import annotations

from typing import Any

import pytest

from ai.harnesses.experimental._adapters import base
from ai.workspaces.experimental._base import ExecResult

from . import fakes


class Machine(fakes.FakeWorkspace):
    def __init__(
        self,
        *,
        owner: str,
        installed: str | None,
        latest: str | None,
        running: list[str] | None = None,
    ) -> None:
        super().__init__()
        self.owner = owner
        self.installed = installed
        self.latest = latest
        self.running = running or []
        self.commands: list[list[str]] = []
        self.files: dict[str, str] = {}

    async def exec(
        self, argv: Any, *, env: Any = None, timeout: float | None = None
    ) -> ExecResult:
        argv = list(argv)
        self.commands.append(argv)
        if argv[1:] == ["--version"]:
            if self.installed is None:
                return ExecResult(exit_code=127)
            return ExecResult(exit_code=0, stdout=f"{self.installed} (Tool)\n")
        if argv[:2] == ["npm", "view"]:
            if self.latest is None:
                return ExecResult(exit_code=1)
            return ExecResult(exit_code=0, stdout=self.latest + "\n")
        if argv[:2] == ["npm", "install"]:
            self.installed = self.latest or "9.9.9"
            return ExecResult(exit_code=0)
        if argv[0] == "ps":
            lines = [
                f"{100 + i} Fri Oct  2 16:20:16 2026 {cmd}"
                for i, cmd in enumerate(self.running)
            ]
            return ExecResult(exit_code=0, stdout="\n".join(lines))
        raise AssertionError(argv)

    async def home(self) -> str:
        return "/home/u"

    async def read_text(self, path: str) -> str:
        if path not in self.files:
            raise FileNotFoundError(path)
        return self.files[path]

    async def write_text(self, path: str, data: str) -> None:
        self.files[path] = data

    @property
    def asked_npm(self) -> int:
        return sum(1 for c in self.commands if c[:2] == ["npm", "view"])

    @property
    def installs(self) -> list[list[str]]:
        return [c for c in self.commands if c[:2] == ["npm", "install"]]


async def _provision(machine: Machine) -> ExecResult:
    return await base.provision(machine, "claude", "@anthropic-ai/claude-code")


async def test_your_own_machine_is_only_probed() -> None:
    machine = Machine(owner="user", installed=None, latest="2.0.0")
    assert (await _provision(machine)).exit_code == 127
    assert machine.commands == [["claude", "--version"]]


@pytest.mark.parametrize("installed", [None, "1.9.0"])
async def test_a_sandbox_gets_the_latest(installed: str | None) -> None:
    machine = Machine(owner="provider", installed=installed, latest="2.0.0")
    probe = await _provision(machine)
    assert machine.installs == [
        ["npm", "install", "-g", "@anthropic-ai/claude-code@latest"]
    ]
    assert "2.0.0" in probe.stdout


async def test_the_latest_already_is_left_alone() -> None:
    machine = Machine(owner="provider", installed="2.0.0", latest="2.0.0")
    await _provision(machine)
    assert machine.installs == []


async def test_a_running_cli_is_not_upgraded_under_it() -> None:
    machine = Machine(
        owner="provider",
        installed="1.9.0",
        latest="2.0.0",
        running=["/usr/bin/node /usr/local/bin/claude --resume abc"],
    )
    await _provision(machine)
    assert machine.installs == []


async def test_no_registry_answer_keeps_what_is_there() -> None:
    machine = Machine(owner="provider", installed="1.9.0", latest=None)
    await _provision(machine)
    assert machine.installs == []


async def test_a_recent_check_is_not_asked_again() -> None:
    """Every reconnect asked npm for the latest release: measured ~0.6-1s of
    each, and a listing reconnects to every running sandbox."""
    machine = Machine(owner="provider", installed="2.0.0", latest="2.0.0")
    await _provision(machine)
    await _provision(machine)
    assert machine.asked_npm == 1


async def test_a_stale_check_is_asked_again(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    machine = Machine(owner="provider", installed="2.0.0", latest="2.0.0")
    await _provision(machine)
    monkeypatch.setattr(base, "CHECK_EVERY", 0.0)
    await _provision(machine)
    assert machine.asked_npm == 2


async def test_a_skipped_upgrade_is_not_remembered_as_checked() -> None:
    machine = Machine(
        owner="provider",
        installed="1.9.0",
        latest="2.0.0",
        running=["/usr/local/bin/claude --resume abc"],
    )
    await _provision(machine)
    machine.running = []
    await _provision(machine)
    assert machine.installs, "upgraded once nothing was running"
