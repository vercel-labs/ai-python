from __future__ import annotations

import asyncio
import os
import pathlib
import sys
from typing import Any, ClassVar

import pytest

import ai
from ai import harness
from ai.harness import __main__ as main


@pytest.fixture(autouse=True)
def _cwd(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # keep the real AGENTS.md / skills out of the system prompt
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))


@pytest.fixture
def typist(monkeypatch: pytest.MonkeyPatch) -> ai.AgentTool:
    """A tool that types lines into the REPL's stdin, like a user mid-run.

    stdin starts with the prompt ``go``. The tool returns once the REPL
    has queued the lines as steering; with ``last`` it closes stdin, so
    the REPL exits after the run.
    """
    r, w = os.pipe()
    monkeypatch.setattr(sys, "stdin", os.fdopen(r))
    os.write(w, b"go\n")

    queued: list[list[str]] = [[]]
    run = main._run

    async def spy(*args: Any) -> None:
        queued[0] = args[3]  # the REPL's steering list
        await run(*args)

    monkeypatch.setattr(main, "_run", spy)

    @ai.tool
    async def type_(lines: list[str], last: bool = False) -> str:
        """Type lines into the REPL."""
        os.write(w, "".join(f"{line}\n" for line in lines).encode())
        if last:
            os.close(w)
        async with asyncio.timeout(5):
            while queued[0] != lines:
                await asyncio.sleep(0.01)
        return "typed"

    return type_


async def _repl(model: ai.Model, tool: ai.AgentTool) -> None:
    class A(harness.Agent):
        TOOLS: ClassVar[list[ai.AgentTool]] = [tool]

    await asyncio.wait_for(main._repl(A(), model), 10)


async def test_steering_in_two_turns_prints_two_markers(
    typist: ai.AgentTool, capsys: pytest.CaptureFixture[str]
) -> None:
    model = ai.testing.FakeModel(
        [
            ai.user_message("go"),
            ai.assistant_message(ai.testing.tool_call(typist, lines=["s1"])),
            ai.user_message("s1"),
            ai.assistant_message(
                ai.testing.tool_call(typist, lines=["s2"], last=True)
            ),
            ai.user_message("s2"),
            ai.assistant_message("ok"),
        ]
    )

    await _repl(model, typist)

    assert capsys.readouterr().out.count("[steered]") == 2
    assert [m.text for m in model.calls[-1] if m.role == "user"] == [
        "go",
        "s1",
        "s2",
    ]
    assert not model.unused


async def test_steering_twice_in_one_turn_is_one_batch(
    typist: ai.AgentTool, capsys: pytest.CaptureFixture[str]
) -> None:
    # both lines land, but in one Steered event: one marker for two lines
    model = ai.testing.FakeModel(
        [
            ai.user_message("go"),
            ai.assistant_message(
                ai.testing.tool_call(typist, lines=["s1", "s2"], last=True)
            ),
            ai.user_message("s1"),
            ai.user_message("s2"),
            ai.assistant_message("ok"),
        ]
    )

    await _repl(model, typist)

    assert capsys.readouterr().out.count("[steered]") == 1
    assert [m.text for m in model.calls[-1] if m.role == "user"] == [
        "go",
        "s1",
        "s2",
    ]
    assert not model.unused
