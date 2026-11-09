"""The Pi worker against a scripted `pi --mode rpc`, offline.

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
from ai.workers.pi import worker as worker_

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
        self.streaming = False

    def says(self, record: dict[str, Any]) -> None:
        if record.get("type") == "agent_start":
            self.streaming = True
        elif record.get("type") == "agent_settled":
            self.streaming = False
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
            self.streaming = True
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
            self.respond(
                message,
                success=True,
                data={"sessionId": SID, "isStreaming": self.streaming},
            )
        elif kind == "get_commands":
            self.respond(
                message,
                success=True,
                data={"commands": [{"name": worker_.READY}]},
            )
        elif "id" in message:
            self.respond(message, success=True)

    def sent(self, kind: str) -> list[dict[str, Any]]:
        return [w for w in self.process.written if w.get("type") == kind]


def _worker(
    server: Pi | None = None, approve: Any = None, cwd: str = "/w", **env: str
) -> worker_.Pi:
    worker = worker_.Pi(cwd=cwd, env=env, approve=approve is not None)
    worker._approval_callback = approve
    if server is not None:
        agent = worker_._Agent(
            cast("Any", server.process), on_dialog=worker._on_dialog
        )
        agent.start()
        worker._agent = agent
    return worker


async def _turn(worker: worker_.Pi, *prompt: Any) -> list[Any]:
    parts: list[m.TextPart | m.FilePart] = list(prompt) or [
        m.TextPart(text="go")
    ]
    async with asyncio.timeout(5):
        return [e async for e in worker._cli_turn(parts)]


async def test_a_turn_streams_in_the_model_view() -> None:
    server = Pi()
    seen = await _turn(_worker(server), m.TextPart(text="say done"))
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
    await _turn(_worker(server), m.TextPart(text="look"), png)
    assert server.sent("prompt")[0]["images"] == [
        {"type": "image", "data": "iVBORw==", "mimeType": "image/png"}
    ]


async def test_a_rejected_prompt_without_a_key_is_not_authenticated() -> None:
    """Measured: pi refuses the prompt itself when it has no credentials."""
    server = Pi(prompt_error="No API key found for openai.\n\nUse /login")
    with pytest.raises(workers.NotAuthenticatedError):
        await _turn(_worker(server))


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
        await _turn(_worker(server))


def _ask(server: Pi) -> None:
    server.says(
        {
            "type": "extension_ui_request",
            "id": "ui-1",
            "method": "input",
            "title": worker_.APPROVE,
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
    seen = await _turn(_worker(server, approve))
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
    worker = _worker(server)
    assert worker._agent is not None
    await worker._on_dialog(
        worker._agent, {"id": "x", "method": "confirm", "title": "Clear?"}
    )
    await worker._on_dialog(
        worker._agent, {"id": "y", "method": "notify", "message": "hello"}
    )
    assert server.sent("extension_ui_response") == [
        {"type": "extension_ui_response", "id": "x", "cancelled": True}
    ]


async def test_an_interrupted_turn_ends_as_cancelled() -> None:
    server = Pi([])
    worker = _worker(server)
    turn = worker._cli_turn([m.TextPart(text="go")])
    assert isinstance(await anext(turn), events.StreamStart)
    reader = asyncio.ensure_future(_rest(turn))
    await worker._cli_interrupt()
    seen = await asyncio.wait_for(reader, 5)
    assert seen[-1].finish_reason == "cancelled"


async def _rest(turn: Any) -> list[Any]:
    return [e async for e in turn]


async def test_a_dead_pi_crashes_the_turn() -> None:
    server = Pi([])
    server.on_prompt = lambda: server.process._lines.put_nowait("")
    with pytest.raises(workers.WorkerError, match="crashed"):
        await _turn(_worker(server))


@pytest.mark.parametrize("still_streaming", [True, False])
async def test_a_run_an_extension_command_starts_is_followed(
    still_streaming: bool,
) -> None:
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
            server.respond(
                message, success=True, data={"isStreaming": still_streaming}
            )
        else:
            Pi.answer(server, message)

    server.process.answer = answer
    seen = await _turn(_worker(server), m.TextPart(text="/ask PONG"))
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
    seen = await _turn(_worker(server), m.TextPart(text="/noop"))
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
    stored = worker_._read(worker_._active_branch(text))
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
    worker = _worker(cwd=str(project), PI_CODING_AGENT_DIR=str(agent_dir))
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
        session_id = await worker._stage(seed)
    infos = await worker._cli_list_sessions()
    assert [i.session_id for i in infos] == [session_id]
    assert infos[0].title == "the word is HERON"
    files = list(agent_dir.glob("sessions/*/*.jsonl"))
    assert len(files) == 1
    assert files[0].parent.name.startswith("--")
    assert files[0].name.endswith(f"_{session_id}.jsonl")
    back = await worker._cli_history(session_id)
    assert [msg.role for msg in back] == ["user", "assistant", "assistant"]
    text, call, result = back[1].parts  # the reasoning was not carried
    assert isinstance(text, m.TextPart)
    assert isinstance(call, m.BuiltinToolCallPart)
    assert json.loads(call.tool_args) == {"path": "a.txt"}
    assert isinstance(result, m.BuiltinToolReturnPart)
    assert result.result == "milk"
    assert back[2].text == "It says milk."
    with pytest.raises(workers.WorkerError):
        await worker._cli_history("not-a-session")


async def test_a_seeded_pi_conversation_stays_on_its_model(
    tmp_path: Path,
) -> None:
    """Measured: a seeded pi conversation ran on pi's default model, not the
    one it was on. History says which model wrote each reply, and seeding
    writes it back the way pi records a switch."""
    agent_dir = tmp_path / "agent"
    project = tmp_path / "project"
    project.mkdir()
    worker = _worker(cwd=str(project), PI_CODING_AGENT_DIR=str(agent_dir))
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
        *worker_._stored_to_messages(stored),
    ]
    session_id = await worker._stage(seed)
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
    worker = worker_.Pi(
        cwd="/w",
        env={"K": "v"},
        executable="pi",
        model_name="anthropic/claude-sonnet-4.6",
        effort="high",
        mcp_servers={
            "ai_sdk": {"type": "http", "url": "http://127.0.0.1:1/mcp"},
            "docs": {"command": "docs-mcp", "ignored": True},
        },
        approve=False,
        allowed_tools=[],
        disallowed_tools=["bash"],
    )
    argv, env, minted = worker._cli_tui_command(None)
    assert argv[:3] == ["pi", "--session-id", minted]
    assert argv[3:5] == ["-e", worker_.EXTENSION]
    assert argv[argv.index("--model") + 1] == "anthropic/claude-sonnet-4.6"
    assert argv[argv.index("--thinking") + 1] == "high"
    assert "--no-tools" in argv
    assert argv[argv.index("--exclude-tools") + 1] == "bash"
    assert env["K"] == "v"
    # The person at the TUI is the gate; the bridge is not there.
    assert json.loads(env[worker_.CONFIG_VAR]) == {
        "approve": False,
        "mcp": {"docs": {"command": "docs-mcp"}},
    }
    # The base's public TUI method must work without open()/a running process.
    public_argv, public_env = worker.tui_command()
    assert public_argv[0] == "pi"
    assert public_env == env
    assert worker._agent is None


async def test_stream_start_waits_for_prompt_acceptance() -> None:
    server = Pi([])
    submitted = asyncio.Event()

    def answer(message: dict[str, Any]) -> None:
        if message.get("type") == "prompt":
            submitted.set()
        else:
            Pi.answer(server, message)

    server.process.answer = answer
    worker = _worker(server)
    turn = worker._cli_turn([m.TextPart(text="go")])
    first = asyncio.create_task(anext(turn))
    try:
        await asyncio.wait_for(submitted.wait(), 5)
        assert not first.done()
        server.respond(
            server.sent("prompt")[0],
            success=True,
            data={"disposition": "started"},
        )
        assert isinstance(await asyncio.wait_for(first, 5), events.StreamStart)
    finally:
        await turn.aclose()
        await worker._cli_close()


async def test_running_input_keeps_images_and_text_files() -> None:
    server = Pi([])
    worker = _worker(server)
    turn = worker._cli_turn([m.TextPart(text="go")])
    assert isinstance(await anext(turn), events.StreamStart)

    def answer(message: dict[str, Any]) -> None:
        if message.get("streamingBehavior") == "steer":
            server.respond(
                message, success=True, data={"disposition": "queued"}
            )
        else:
            Pi.answer(server, message)

    server.process.answer = answer
    try:
        assert await worker._cli_send_running(
            [
                m.TextPart(text="look"),
                m.FilePart(data=b"notes", media_type="text/plain"),
                m.FilePart(data=b"\x89PNG", media_type="image/png"),
            ]
        )
        command = server.sent("prompt")[1]
        assert "look" in command["message"]
        assert "notes" in command["message"]
        assert command["images"] == [
            {"type": "image", "data": "iVBORw==", "mimeType": "image/png"}
        ]
        assert not server.sent("steer")
        assert not server.sent("clear_queue")
    finally:
        await turn.aclose()
        await worker._cli_close()


async def test_confirmed_idle_does_not_submit_running_input() -> None:
    server = Pi([])
    worker = _worker(server)
    turn = worker._cli_turn([m.TextPart(text="go")])
    assert isinstance(await anext(turn), events.StreamStart)
    # Completion has arrived, but the old generator has not consumed it.
    server.says({"type": "agent_settled", "aborted": False})
    try:
        assert worker._running
        assert not await worker._cli_send_running([m.TextPart(text="next")])
        assert len(server.sent("prompt")) == 1
        assert isinstance((await _rest(turn))[-1], events.StreamEnd)
    finally:
        await turn.aclose()
        await worker._cli_close()


@pytest.mark.parametrize("completion_before_ack", [True, False])
@pytest.mark.parametrize("disposition", ["started", "handled"])
async def test_turn_end_race_keeps_the_accepted_new_runs_events(
    completion_before_ack: bool,
    disposition: str,
) -> None:
    server = Pi([])
    worker = _worker(server)
    turn = worker._cli_turn([m.TextPart(text="go")])
    assert isinstance(await anext(turn), events.StreamStart)
    submitted = asyncio.Event()

    def answer(message: dict[str, Any]) -> None:
        if message.get("streamingBehavior") == "steer":
            # get_state said streaming, but the run ended before acceptance.
            server.says({"type": "agent_settled", "aborted": False})
            submitted.set()
        else:
            Pi.answer(server, message)

    server.process.answer = answer
    reader = asyncio.create_task(_rest(turn))
    send = asyncio.create_task(
        worker._cli_send_running([m.TextPart(text="next")])
    )
    try:
        await asyncio.wait_for(submitted.wait(), 5)
        assert not reader.done()
        assert not send.done()
        server.says(TOOL_TURN[0])  # The accepted input starts a new run.
        if completion_before_ack:
            for record in TOOL_TURN[1:]:
                server.says(record)
        server.respond(
            server.sent("prompt")[1],
            success=True,
            data={"disposition": disposition},
        )
        assert await asyncio.wait_for(send, 5)
        if not completion_before_ack:
            assert not reader.done()
            for record in TOOL_TURN[1:]:
                server.says(record)
        seen = await asyncio.wait_for(reader, 5)
        assert [e.chunk for e in seen if isinstance(e, events.TextDelta)] == [
            "done"
        ]
        assert sum(isinstance(e, events.StreamEnd) for e in seen) == 1
        assert not worker._running
        assert worker._continued_runs == 0
        assert len(server.sent("prompt")) == 2
    finally:
        send.cancel()
        reader.cancel()
        await asyncio.gather(send, reader, return_exceptions=True)
        await turn.aclose()
        await worker._cli_close()


@pytest.mark.parametrize("disposition", [None, "unexpected"])
async def test_ambiguous_acceptance_is_not_retried(disposition: Any) -> None:
    server = Pi([])
    worker = _worker(server)
    turn = worker._cli_turn([m.TextPart(text="go")])
    assert isinstance(await anext(turn), events.StreamStart)

    def answer(message: dict[str, Any]) -> None:
        if message.get("streamingBehavior") == "steer":
            server.respond(
                message, success=True, data={"disposition": disposition}
            )
        else:
            Pi.answer(server, message)

    server.process.answer = answer
    try:
        with pytest.raises(workers.WorkerError, match="unknown prompt"):
            await worker._cli_send_running([m.TextPart(text="next")])
        assert len(server.sent("prompt")) == 2
    finally:
        await turn.aclose()
        await worker._cli_close()


async def test_rejected_running_input_is_not_reported_as_idle() -> None:
    server = Pi([])
    worker = _worker(server)
    turn = worker._cli_turn([m.TextPart(text="go")])
    assert isinstance(await anext(turn), events.StreamStart)
    server.prompt_error = (
        "Cannot submit a prompt while compaction is in progress"
    )
    try:
        with pytest.raises(workers.WorkerError, match="compaction"):
            await worker._cli_send_running([m.TextPart(text="next")])
        assert len(server.sent("prompt")) == 2
    finally:
        await turn.aclose()
        await worker._cli_close()


@pytest.mark.parametrize("failure", ["timeout", "crash"])
async def test_uncertain_running_input_is_never_retried(
    failure: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server = Pi([])
    worker = _worker(server)
    turn = worker._cli_turn([m.TextPart(text="go")])
    assert isinstance(await anext(turn), events.StreamStart)
    assert worker._agent is not None
    call = worker._agent.call

    async def short_call(command: dict[str, Any], **kwargs: Any) -> Any:
        return await call(command, timeout=0.01)

    def answer(message: dict[str, Any]) -> None:
        if message.get("streamingBehavior") == "steer":
            if failure == "crash":
                server.process._lines.put_nowait("")
            # No response: input may already have been accepted.
        else:
            Pi.answer(server, message)

    server.process.answer = answer
    monkeypatch.setattr(worker._agent, "call", short_call)
    try:
        error = TimeoutError if failure == "timeout" else workers.WorkerError
        with pytest.raises(error):
            await worker._cli_send_running([m.TextPart(text="next")])
        assert len(server.sent("prompt")) == 2
    finally:
        await turn.aclose()
        await worker._cli_close()


async def test_start_wires_approval_and_private_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def probe(*args: Any, **kwargs: Any) -> str:
        return "pi test"

    from ai.workers import _adapter

    monkeypatch.setattr(_adapter, "probe", probe)
    servers = {"docs": {"command": "docs-mcp"}}
    worker = worker_.Pi(approve=True)
    await worker._cli_start(servers)
    assert worker._approval_callback == worker._approve
    assert worker._cli_version == "pi test"
    assert worker._mcp_servers == servers
    assert worker._mcp_servers is not servers
    dumped = worker.model_dump()
    assert dumped["worker_class_id"] == "pi"
    assert "model_name" in dumped
    assert "kind" not in dumped
    assert "model" not in dumped
    assert not any(key.startswith("_") for key in dumped)
    restored = worker_.Pi.model_validate(dumped)
    assert restored._agent is None
    assert restored._approval_callback is None
    await worker._cli_close()
    assert worker._approval_callback is None
    worker.approve = False
    await worker._cli_start({})
    assert worker._approval_callback is None


async def test_open_and_close_own_the_rpc_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server = Pi()
    started: list[Any] = []

    async def spawn(argv: Any, **kwargs: Any) -> Any:
        started.append((argv, kwargs))
        return server.process

    def answer(message: dict[str, Any]) -> None:
        if message.get("type") == "get_state":
            server.respond(
                message, success=True, data={"sessionId": started[0][0][4]}
            )
        else:
            Pi.answer(server, message)

    server.process.answer = answer
    monkeypatch.setattr(worker_, "spawn", spawn)
    worker = worker_.Pi(cwd="/w", model_name="test-model")
    try:
        session_id = await worker._cli_open(
            session_id=None, fork=False, seed=None, instructions="be brief"
        )
        argv, options = started[0]
        assert argv[:5] == ["pi", "--mode", "rpc", "--session-id", session_id]
        assert argv[argv.index("--append-system-prompt") + 1] == "be brief"
        assert options["cwd"] == "/w"
        assert worker._agent is not None
    finally:
        await worker._cli_close()
    assert worker._agent is None
    assert await asyncio.wait_for(server.process.readline(), 5) == ""
