"""The pi adapter against a scripted `pi --mode rpc`, offline.

The records here are the shapes a real pi 1.3.0 sent when measured.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import pytest

from ai import workers
from ai.types import events
from ai.types import messages as m
from ai.workers.pi import adapter as adapter_

from .. import fakes

SID = "s1"


def _update(kind: str, index: int = 0, **fields: Any) -> dict[str, Any]:
    return {
        "type": "message_update",
        "assistantMessageEvent": {
            "type": kind,
            "contentIndex": index,
            **fields,
        },
    }


def _assistant(*content: dict[str, Any], stop: str = "stop") -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": list(content),
        "usage": {
            "input": 10,
            "output": 5,
            "cacheRead": 100,
            "cacheWrite": 0,
            "cost": {"total": 0.01},
        },
        "stopReason": stop,
    }


CALL = {
    "type": "toolCall",
    "id": "toolu_1",
    "name": "bash",
    "arguments": {"command": "echo hi"},
}

# text -> tool call -> tool result -> text: what pi sent for
# "run `echo hi`, then say done".
TOOL_TURN: list[dict[str, Any]] = [
    {"type": "agent_start"},
    {"type": "message_start", "message": {"role": "assistant"}},
    _update("thinking_start"),
    _update("thinking_delta", delta="plan"),
    _update("thinking_end", content="plan"),
    _update("toolcall_start", 1, id="toolu_1", toolName="bash"),
    _update("toolcall_delta", 1, delta='{"command": "ec'),
    _update("toolcall_end", 1, toolCall=CALL),
    {
        "type": "message_end",
        "message": _assistant(
            {"type": "thinking", "thinking": "plan"}, CALL, stop="toolUse"
        ),
    },
    {
        "type": "tool_execution_end",
        "toolCallId": "toolu_1",
        "toolName": "bash",
        "result": {"content": [{"type": "text", "text": "hi\n"}]},
        "isError": False,
    },
    {"type": "message_start", "message": {"role": "assistant"}},
    _update("text_start"),
    _update("text_delta", delta="done"),
    _update("text_end", content="done"),
    {
        "type": "message_end",
        "message": _assistant({"type": "text", "text": "done"}),
    },
    {"type": "agent_end", "messages": []},
    {"type": "agent_settled", "aborted": False},
]


class Pi:
    """Answers like `pi --mode rpc`: a prompt is accepted, then `turn`
    plays out; `abort` settles the turn as aborted."""

    def __init__(
        self,
        turn: list[dict[str, Any]] | None = None,
        *,
        prompt_error: str | None = None,
    ) -> None:
        self.process = fakes.FakeProcess(self.answer)
        self.turn = turn if turn is not None else TOOL_TURN
        self.prompt_error = prompt_error
        self.on_prompt: Callable[[], None] | None = None

    def says(self, record: dict[str, Any]) -> None:
        self.process.server_says(record)

    def respond(self, message: dict[str, Any], **fields: Any) -> None:
        self.says(
            {
                "type": "response",
                "id": message["id"],
                "command": message["type"],
                **fields,
            }
        )

    def answer(self, message: dict[str, Any]) -> None:
        kind = message.get("type")
        if kind == "prompt":
            if self.prompt_error is not None:
                self.respond(message, success=False, error=self.prompt_error)
                return
            self.respond(message, success=True, data={"disposition": "started"})
            if self.on_prompt is not None:
                self.on_prompt()
                return
            for record in self.turn:
                self.says(record)
        elif kind == "abort":
            self.says({"type": "agent_settled", "aborted": True})
            self.respond(message, success=True)
        elif kind == "get_state":
            self.respond(message, success=True, data={"sessionId": SID})
        elif "id" in message:
            self.respond(message, success=True)

    def sent(self, kind: str) -> list[dict[str, Any]]:
        return [w for w in self.process.written if w.get("type") == kind]


def _adapter(
    server: Pi | None = None, approve: Any = None, cwd: str = "/w", **env: str
) -> adapter_.PiAdapter:
    adapter = adapter_.PiAdapter(
        cwd=cwd,
        env=env,
        executable="pi",
        model=None,
        effort=None,
        mcp_servers={},
        approve=approve,
        allowed_tools=None,
        disallowed_tools=None,
    )
    if server is not None:
        agent = adapter_._Agent(
            cast("Any", server.process), on_dialog=adapter._on_dialog
        )
        agent.start()
        adapter._agent = agent
    return adapter


async def _turn(adapter: adapter_.PiAdapter, *prompt: Any) -> list[Any]:
    parts: list[m.TextPart | m.FilePart] = list(prompt) or [
        m.TextPart(text="go")
    ]
    async with asyncio.timeout(5):
        return [e async for e in adapter.turn(parts)]


async def test_a_turn_streams_in_the_model_view() -> None:
    server = Pi()
    seen = await _turn(_adapter(server), m.TextPart(text="say done"))
    assert [type(e).__name__ for e in seen] == [
        "StreamStart",
        "ReasoningStart",
        "ReasoningDelta",
        "ReasoningEnd",
        "BuiltinToolStart",
        "BuiltinToolDelta",
        "BuiltinToolEnd",
        "BuiltinToolResult",
        "TextStart",
        "TextDelta",
        "TextEnd",
        "StreamEnd",
    ]
    # The arguments travel once, whole: pi's fragments are not what it
    # settles on.
    delta = next(e for e in seen if isinstance(e, events.BuiltinToolDelta))
    assert json.loads(delta.chunk) == {"command": "echo hi"}
    result = next(e for e in seen if isinstance(e, events.BuiltinToolResult))
    assert result.result.result == "hi\n"
    end = seen[-1]
    assert end.finish_reason == "stop"
    # Two model calls: cached input counts as input, cost rides in raw.
    assert end.usage.input_tokens == 220
    assert end.usage.output_tokens == 10
    assert end.usage.cache_read_tokens == 200
    assert end.usage.raw == {"cost_usd": 0.02}
    assert server.sent("prompt")[0]["message"] == "say done"
    # Blocks of different model calls never share an id.
    starts = [e.block_id for e in seen if isinstance(e, events.TextStart)]
    assert starts == ["2.0"]


async def test_images_travel_as_base64() -> None:
    server = Pi()
    png = m.FilePart(data=b"\x89PNG", media_type="image/png")
    await _turn(_adapter(server), m.TextPart(text="look"), png)
    assert server.sent("prompt")[0]["images"] == [
        {"type": "image", "data": "iVBORw==", "mimeType": "image/png"}
    ]


async def test_a_rejected_prompt_without_a_key_is_not_authenticated() -> None:
    """Measured: pi refuses the prompt itself when it has no credentials."""
    server = Pi(prompt_error="No API key found for openai.\n\nUse /login")
    with pytest.raises(workers.NotAuthenticatedError):
        await _turn(_adapter(server))


async def test_a_failed_model_call_fails_the_turn() -> None:
    server = Pi(
        [
            {
                "type": "message_end",
                "message": {
                    **_assistant(stop="error"),
                    "errorMessage": "529 overloaded",
                },
            },
            {"type": "agent_settled", "aborted": False},
        ]
    )
    with pytest.raises(workers.WorkerError, match="overloaded"):
        await _turn(_adapter(server))


def _ask(server: Pi) -> None:
    server.says(
        {
            "type": "extension_ui_request",
            "id": "ui-1",
            "method": "input",
            "title": adapter_.APPROVE,
            "placeholder": json.dumps(
                {
                    "toolCallId": "toolu_1",
                    "toolName": "bash",
                    "input": {"command": "echo hi"},
                }
            ),
        }
    )


@pytest.mark.parametrize("granted", [True, False])
async def test_approvals_ask_and_answer(granted: bool) -> None:
    server = Pi()
    asked: list[Any] = []

    async def approve(
        call_id: str, name: str, tool_input: dict[str, Any]
    ) -> tuple[bool, str]:
        asked.append((call_id, name, tool_input))
        return granted, "not that"

    def answer(message: dict[str, Any]) -> None:
        if message.get("type") == "extension_ui_response":
            for record in TOOL_TURN:
                server.says(record)
        else:
            Pi.answer(server, message)

    server.on_prompt = lambda: _ask(server)
    server.process.answer = answer
    seen = await _turn(_adapter(server, approve))
    assert asked == [("toolu_1", "bash", {"command": "echo hi"})]
    reply = server.sent("extension_ui_response")[0]
    assert reply["id"] == "ui-1"
    assert json.loads(reply["value"]) == (
        {"decision": "allow"}
        if granted
        else {"decision": "deny", "reason": "not that"}
    )
    assert isinstance(seen[-1], events.StreamEnd)


async def test_another_extensions_dialog_is_dismissed() -> None:
    """Nobody is at a terminal to answer it; the agent must not wait."""
    server = Pi()
    adapter = _adapter(server)
    assert adapter._agent is not None
    await adapter._on_dialog(
        adapter._agent, {"id": "x", "method": "confirm", "title": "Clear?"}
    )
    await adapter._on_dialog(
        adapter._agent, {"id": "y", "method": "notify", "message": "hello"}
    )
    assert server.sent("extension_ui_response") == [
        {"type": "extension_ui_response", "id": "x", "cancelled": True}
    ]


async def test_an_interrupted_turn_ends_as_cancelled() -> None:
    server = Pi([])
    adapter = _adapter(server)
    turn = adapter.turn([m.TextPart(text="go")])
    assert isinstance(await anext(turn), events.StreamStart)
    reader = asyncio.ensure_future(_rest(turn))
    await adapter.interrupt()
    seen = await asyncio.wait_for(reader, 5)
    assert seen[-1].finish_reason == "cancelled"


async def _rest(turn: Any) -> list[Any]:
    return [e async for e in turn]


async def test_a_dead_pi_crashes_the_turn() -> None:
    server = Pi([])
    server.on_prompt = lambda: server.process._lines.put_nowait("")
    with pytest.raises(workers.WorkerError, match="crashed"):
        await _turn(_adapter(server))


async def test_a_run_an_extension_command_starts_is_followed() -> None:
    """Measured: `/ask` answered `handled`, and the run the command started
    followed, with pi already streaming. Its text was never streamed."""
    server = Pi(
        [
            {"type": "agent_start"},
            {"type": "message_start", "message": {"role": "assistant"}},
            {
                "type": "message_end",
                "message": _assistant({"type": "text", "text": "PONG"}),
            },
            {"type": "agent_settled", "aborted": False},
        ]
    )

    def answer(message: dict[str, Any]) -> None:
        if message.get("type") == "prompt":
            server.respond(
                message, success=True, data={"disposition": "handled"}
            )
            for record in server.turn:
                server.says(record)
        elif message.get("type") == "get_state":
            server.respond(message, success=True, data={"isStreaming": True})
        else:
            Pi.answer(server, message)

    server.process.answer = answer
    seen = await _turn(_adapter(server), m.TextPart(text="/ask PONG"))
    assert [e.chunk for e in seen if isinstance(e, events.TextDelta)] == [
        "PONG"
    ]


async def test_a_command_that_starts_nothing_ends_the_turn_empty() -> None:
    server = Pi()

    def answer(message: dict[str, Any]) -> None:
        if message.get("type") == "prompt":
            server.respond(
                message, success=True, data={"disposition": "handled"}
            )
        elif message.get("type") == "get_state":
            server.respond(message, success=True, data={"isStreaming": False})
        else:
            Pi.answer(server, message)

    server.process.answer = answer
    seen = await _turn(_adapter(server), m.TextPart(text="/noop"))
    assert [type(e).__name__ for e in seen] == ["StreamStart", "StreamEnd"]


def test_history_follows_the_active_branch_only() -> None:
    lines = [
        {"type": "session", "id": "S", "cwd": "/w"},
        {
            "type": "message",
            "id": "a",
            "parentId": None,
            "message": {"role": "user", "content": "first"},
        },
        {
            "type": "message",
            "id": "b",
            "parentId": "a",
            "message": _assistant({"type": "text", "text": "abandoned"}),
        },
        {
            "type": "message",
            "id": "c",
            "parentId": "a",
            "message": _assistant(CALL, stop="toolUse"),
        },
        {
            "type": "message",
            "id": "d",
            "parentId": "c",
            "message": {
                "role": "toolResult",
                "toolCallId": "toolu_1",
                "toolName": "bash",
                "content": [{"type": "text", "text": "hi"}],
                "isError": True,
            },
        },
        {"type": "label", "id": "e", "parentId": "d"},
    ]
    text = "\n".join(json.dumps(line) for line in lines)
    stored = adapter_._read(adapter_._active_branch(text))
    assert [msg.role for msg in stored] == ["user", "assistant"]
    call, result = stored[1].parts  # the result joins its call
    assert isinstance(call, m.BuiltinToolCallPart)
    assert isinstance(result, m.BuiltinToolReturnPart)
    assert (result.result, result.is_error) == ("hi", True)


async def test_a_seed_is_listed_and_read_back(tmp_path: Path) -> None:
    """The store, for real: write where pi looks, find it there."""
    agent_dir = tmp_path / "agent"
    project = tmp_path / "project"
    project.mkdir()
    adapter = _adapter(cwd=str(project), PI_CODING_AGENT_DIR=str(agent_dir))
    seed = [
        m.Message(role="user", parts=[m.TextPart(text="the word is HERON")]),
        m.Message(
            role="assistant",
            parts=[
                m.ReasoningPart(text="hmm"),
                m.TextPart(text="Reading."),
                m.BuiltinToolCallPart(
                    tool_call_id="c1",
                    tool_name="read",
                    tool_args='{"path": "a.txt"}',
                ),
                m.BuiltinToolReturnPart(
                    tool_call_id="c1", tool_name="read", result="milk"
                ),
            ],
        ),
        m.Message(role="assistant", parts=[m.TextPart(text="It says milk.")]),
    ]
    with pytest.warns(UserWarning, match="reasoning"):
        session_id = await adapter._stage(seed)
    infos = await adapter.list_sessions()
    assert [i.session_id for i in infos] == [session_id]
    assert infos[0].title == "the word is HERON"
    files = list(agent_dir.glob("sessions/*/*.jsonl"))
    assert len(files) == 1
    assert files[0].parent.name.startswith("--")
    assert files[0].name.endswith(f"_{session_id}.jsonl")
    back = await adapter.history(session_id)
    assert [msg.role for msg in back] == ["user", "assistant", "assistant"]
    text, call, result = back[1].parts  # the reasoning was not carried
    assert isinstance(text, m.TextPart)
    assert isinstance(call, m.BuiltinToolCallPart)
    assert json.loads(call.tool_args) == {"path": "a.txt"}
    assert isinstance(result, m.BuiltinToolReturnPart)
    assert result.result == "milk"
    assert back[2].text == "It says milk."
    with pytest.raises(workers.WorkerError):
        await adapter.history("not-a-session")


async def test_a_seeded_pi_conversation_stays_on_its_model(
    tmp_path: Path,
) -> None:
    """Measured: a seeded pi conversation ran on pi's default model, not the
    one it was on. History says which model wrote each reply, and seeding
    writes it back the way pi records a switch."""
    agent_dir = tmp_path / "agent"
    project = tmp_path / "project"
    project.mkdir()
    adapter = _adapter(cwd=str(project), PI_CODING_AGENT_DIR=str(agent_dir))
    stored = {
        "type": "message",
        "message": {
            **_assistant({"type": "text", "text": "hi"}),
            "api": "anthropic-messages",
            "provider": "vercel-ai-gateway",
            "model": "anthropic/claude-haiku-4.5",
            "thinkingLevel": "high",
        },
    }
    seed = [
        m.Message(role="user", parts=[m.TextPart(text="hello")]),
        *adapter_._stored_to_messages(stored),
    ]
    session_id = await adapter._stage(seed)
    (written,) = agent_dir.glob(f"sessions/*/*_{session_id}.jsonl")
    entries = [json.loads(line) for line in written.read_text().splitlines()]
    assert [e["type"] for e in entries] == [
        "session",
        "model_change",
        "thinking_level_change",
        "message",
        "message",
    ]
    assert (entries[1]["provider"], entries[1]["modelId"]) == (
        "vercel-ai-gateway",
        "anthropic/claude-haiku-4.5",
    )
    # A chain, root first: pi reads it as one branch.
    assert [e.get("parentId") for e in entries[1:]] == [
        None,
        *[e["id"] for e in entries[1:-1]],
    ]


def test_options_reach_pi_in_its_own_spelling() -> None:
    adapter = adapter_.PiAdapter(
        cwd="/w",
        env={"K": "v"},
        executable="pi",
        model="anthropic/claude-sonnet-4.6",
        effort="high",
        mcp_servers={
            "ai_sdk": {"type": "http", "url": "http://127.0.0.1:1/mcp"},
            "docs": {"command": "docs-mcp", "ignored": True},
        },
        approve=None,
        allowed_tools=[],
        disallowed_tools=["bash"],
    )
    argv, env, minted = adapter.tui_command(None)
    assert argv[:3] == ["pi", "--session-id", minted]
    assert argv[3:5] == ["-e", adapter_.EXTENSION]
    assert argv[argv.index("--model") + 1] == "anthropic/claude-sonnet-4.6"
    assert argv[argv.index("--thinking") + 1] == "high"
    assert "--no-tools" in argv
    assert argv[argv.index("--exclude-tools") + 1] == "bash"
    assert env["K"] == "v"
    # The person at the TUI is the gate; the bridge is not there.
    assert json.loads(env[adapter_.CONFIG_VAR]) == {
        "approve": False,
        "mcp": {"docs": {"command": "docs-mcp"}},
    }
