"""Day-to-day options reach each CLI in its own spelling."""

from __future__ import annotations

import inspect
from typing import Any

import pytest

from ai.harnesses.experimental import Handle, _factories
from ai.harnesses.experimental._adapters import codex
from ai.workspaces.experimental._base import WorkspaceCoords

from . import fakes


def test_mcp_servers_become_codex_config() -> None:
    assert codex.mcp_config(
        {
            "files": {
                "command": "python3",
                "args": ["server.py"],
                "env": {"TOKEN": "x"},
            },
            "web": {"type": "http", "url": "https://mcp.example/mcp"},
        }
    ) == {
        "mcp_servers.files.command": "python3",
        "mcp_servers.files.args": ["server.py"],
        "mcp_servers.files.env.TOKEN": "x",
        "mcp_servers.web.url": "https://mcp.example/mcp",
    }


def test_codex_instructions_go_on_every_thread() -> None:
    adapter = codex.CodexAdapter(instructions="be brief")
    adapter._sandbox = "danger-full-access"
    assert adapter._thread_params()["developerInstructions"] == "be brief"
    assert (
        codex.CodexAdapter()._thread_params().get("developerInstructions")
        is None
    )


def test_claude_tui_carries_the_options() -> None:
    claude = pytest.importorskip("ai.harnesses.experimental._adapters.claude")
    adapter = claude.ClaudeAdapter(
        instructions="be brief",
        mcp_servers={"files": {"command": "python3", "args": ["s.py"]}},
        tools=["Read"],
        disallowed_tools=["Bash"],
    )
    adapter._workspace = fakes.FakeWorkspace()
    argv, _, _ = adapter.tui_launch(None)
    assert argv[argv.index("--append-system-prompt") + 1] == "be brief"
    assert "mcpServers" in argv[argv.index("--mcp-config") + 1]
    assert argv[argv.index("--tools") + 1] == "Read"
    assert argv[argv.index("--disallowedTools") + 1] == "Bash"


def test_a_handle_records_options_but_never_mcp_servers() -> None:
    pytest.importorskip("claude_agent_sdk")
    harness = _factories.claude_code(
        workspace=fakes.FakeWorkspace(),
        instructions="be brief",
        tools=["Read"],
        max_turns=3,
        mcp_servers={"s": {"command": "x", "env": {"TOKEN": "secret"}}},
    )
    assert harness.options["instructions"] == "be brief"
    assert harness.options["max_turns"] == 3
    assert "secret" not in repr(harness.options)


async def test_harness_from_rebuilds_with_the_recorded_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    built: dict[str, object] = {}

    def fake_builder(
        *,
        workspace: object,
        approve: object = None,
        approval_timeout: float = 1,
        instructions: str | None = None,
        model: str | None = None,
        mcp_servers: object = None,
    ) -> object:
        built.update(instructions=instructions, model=model, mcp=mcp_servers)
        raise RuntimeError("stop here")

    monkeypatch.setitem(_factories.BUILDERS, "fake", fake_builder)
    handle = Handle(
        kind="fake",
        session_id="s1",
        workspace=WorkspaceCoords(provider="local", location="/tmp"),
        options={"instructions": "be brief", "model": None, "unknown": 1},
    )
    with pytest.raises(RuntimeError, match="stop here"):
        async with _factories.harness_from(
            handle, mcp_servers={"s": {"command": "x"}}
        ):
            pass
    assert built == {
        "instructions": "be brief",
        "model": None,
        "mcp": {"s": {"command": "x"}},
    }


@pytest.mark.parametrize(
    "option", [{"tools": ["Read"]}, {"disallowed_tools": []}, {"max_turns": 3}]
)
def test_codex_takes_claude_only_options_but_refuses_them(
    option: dict[str, Any],
) -> None:
    """Every `claude_code()` argument is a `codex()` argument, so swapping
    the two is one word and a dict of both factories keeps its type; codex
    has no equivalent, so a value is refused, never ignored."""
    from ai.harnesses.experimental import codex as make_codex
    from ai.harnesses.experimental.errors import UnsupportedError

    with pytest.raises(UnsupportedError, match=next(iter(option))):
        make_codex(workspace=fakes.FakeWorkspace(), **option)


def test_a_dict_of_both_factories_keeps_a_callable_type() -> None:
    """Measured: adding claude-only options to claude alone made
    `{"claude-code": claude_code, "codex": codex}` a dict of bare
    `function` to mypy, and every call through it an error (afk's
    HARNESSES, examples/harnesses/tui.py)."""
    from ai.harnesses.experimental import claude_code, codex

    assert set(inspect.signature(claude_code).parameters) <= set(
        inspect.signature(codex).parameters
    )
