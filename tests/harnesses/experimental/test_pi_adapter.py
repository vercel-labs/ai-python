"""The pi adapter against a scripted `pi --mode rpc`, offline.

The records here are the shapes a real pi 1.3.0 sent when measured.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from ai.harnesses.experimental import Allow, ApprovalContext, Deny
from ai.harnesses.experimental import errors as harness_errors
from ai.harnesses.experimental._adapters import pi
from ai.types import events
from ai.types import messages as m
from ai.workspaces.experimental import Local
from ai.workspaces.experimental import errors as workspace_errors

from . import fakes

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
        self.pending_abort: dict[str, Any] | None = None
        self.extension_loaded = True
        self.terminated = False

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
        elif kind == "get_commands":
            commands = [{"name": pi.READY}] if self.extension_loaded else []
            self.respond(message, success=True, data={"commands": commands})
        elif "id" in message:
            self.respond(message, success=True)

    def sent(self, kind: str) -> list[dict[str, Any]]:
        return [w for w in self.process.written if w.get("type") == kind]


def _adapter(server: Pi, approve: Any = None) -> pi.PiAdapter:
    adapter = pi.PiAdapter()
    adapter.approve = approve
    adapter._workspace = fakes.FakeWorkspace()
    agent = pi._Agent(server.process, on_dialog=adapter._dialog_handler(SID))
    agent.start()
    adapter._agents[SID] = agent
    return adapter


async def _turn(adapter: pi.PiAdapter, text: str = "go") -> list[Any]:
    async with asyncio.timeout(5):
        return [e async for e in adapter.turn(SID, [m.TextPart(text=text)])]


async def test_a_turn_streams_and_settles_on_agent_settled() -> None:
    server = Pi()
    adapter = _adapter(server)
    seen = await _turn(adapter, "say done")
    kinds = [type(e).__name__ for e in seen]
    assert kinds == [
        "StreamStart",
        "ReasoningStart",
        "ReasoningDelta",
        "ReasoningEnd",
        "ToolStart",
        "ToolDelta",
        "ToolEnd",
        "ToolCallResult",
        "TextStart",
        "TextDelta",
        "TextEnd",
        "StreamEnd",
    ]
    # The arguments travel once, whole: the hydrator builds calls from
    # deltas, and pi's fragments are not what it settles on.
    delta = next(e for e in seen if isinstance(e, events.ToolDelta))
    assert json.loads(delta.chunk) == {"command": "echo hi"}
    result = next(e for e in seen if isinstance(e, events.ToolCallResult))
    assert result.results[0].result == "hi\n"
    end = seen[-1]
    assert isinstance(end, events.StreamEnd)
    assert end.finish_reason == "stop"
    assert [type(p).__name__ for p in end.message.parts] == [
        "ReasoningPart",
        "ToolCallPart",
        "TextPart",
    ]
    # Two model calls: cached input counts as input, cost rides in raw.
    usage = end.usage
    assert usage is not None
    assert usage.input_tokens == 220
    assert usage.output_tokens == 10
    assert usage.cache_read_tokens == 200
    assert usage.raw == {"cost_usd": 0.02}
    assert server.sent("prompt")[0]["message"] == "say done"
    # Blocks of different model calls never share an id.
    starts = [e.block_id for e in seen if isinstance(e, events.TextStart)]
    assert starts == ["2.0"]


async def test_images_travel_as_base64() -> None:
    server = Pi()
    adapter = _adapter(server)
    png = m.FilePart(data=b"\x89PNG", media_type="image/png")
    async with asyncio.timeout(5):
        async for _ in adapter.turn(SID, [m.TextPart(text="look"), png]):
            pass
    prompt = server.sent("prompt")[0]
    assert prompt["images"] == [
        {"type": "image", "data": "iVBORw==", "mimeType": "image/png"}
    ]


async def test_a_rejected_prompt_without_a_key_is_not_authenticated() -> None:
    """Measured: pi refuses the prompt itself when it has no credentials."""
    server = Pi(prompt_error="No API key found for openai.\n\nUse /login")
    adapter = _adapter(server)
    with pytest.raises(workspace_errors.NotAuthenticatedError):
        await _turn(adapter)


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
    adapter = _adapter(server)
    with pytest.raises(harness_errors.TurnFailedError, match="overloaded"):
        await _turn(adapter)


async def test_the_hook_sees_the_call_and_can_rewrite_it() -> None:
    server = Pi()
    seen: list[ApprovalContext] = []

    async def approve(ctx: ApprovalContext) -> Allow:
        seen.append(ctx)
        return Allow(input={"command": "echo rewritten"})

    adapter = _adapter(server, approve)
    answered = asyncio.Event()

    def ask() -> None:
        server.says(
            {
                "type": "extension_ui_request",
                "id": "ui-1",
                "method": "input",
                "title": pi.APPROVE,
                "placeholder": json.dumps(
                    {
                        "toolCallId": "toolu_1",
                        "toolName": "bash",
                        "input": {"command": "echo hi"},
                    }
                ),
            }
        )

    def answer(message: dict[str, Any]) -> None:
        if message.get("type") == "extension_ui_response":
            answered.set()
            for record in TOOL_TURN:
                server.says(record)
        else:
            Pi.answer(server, message)

    server.on_prompt = ask
    server.process.answer = answer
    seen_events = await _turn(adapter)
    assert answered.is_set()
    assert seen[0].call.name == "bash"
    assert seen[0].call.input == {"command": "echo hi"}
    assert seen[0].harness == "pi"
    reply = server.sent("extension_ui_response")[0]
    assert reply["id"] == "ui-1"
    assert json.loads(reply["value"]) == {
        "decision": "allow",
        "input": {"command": "echo rewritten"},
    }
    assert isinstance(seen_events[-1], events.StreamEnd)


async def test_deny_and_stop_answers_then_aborts() -> None:
    server = Pi()

    async def deny(ctx: ApprovalContext) -> Deny:
        return Deny("not that", stop=True)

    adapter = _adapter(server, deny)
    server.on_prompt = lambda: server.says(
        {
            "type": "extension_ui_request",
            "id": "ui-1",
            "method": "input",
            "title": pi.APPROVE,
            "placeholder": json.dumps({"toolCallId": "t", "toolName": "bash"}),
        }
    )
    seen = await _turn(adapter)
    reply = json.loads(server.sent("extension_ui_response")[0]["value"])
    assert reply == {"decision": "deny", "reason": "not that", "stop": True}
    # The answer first, then the abort: pi must hear the denial.
    order = [w["type"] for w in server.process.written]
    assert order.index("extension_ui_response") < order.index("abort")
    end = seen[-1]
    assert isinstance(end, events.StreamEnd)
    assert end.finish_reason == "cancelled"


async def test_a_failing_hook_denies_and_is_reported() -> None:
    server = Pi()
    reported: list[str] = []

    async def broken(ctx: ApprovalContext) -> Allow:
        raise ValueError("boom")

    adapter = _adapter(server, broken)
    adapter.on_approval_error = lambda sid, detail: reported.append(detail)
    request = {"method": "input", "title": pi.APPROVE, "placeholder": "{}"}
    decision = await adapter._judge(SID, request)
    assert decision["decision"] == "deny"
    assert reported == ["ValueError: boom"]


async def test_another_extensions_dialog_is_dismissed() -> None:
    """Nobody is at a terminal to answer it; the agent must not wait."""
    server = Pi()
    adapter = _adapter(server)
    agent = adapter._agents[SID]
    await adapter._dialog_handler(SID)(
        agent,
        {"id": "x", "method": "confirm", "title": "Clear session?"},
    )
    await adapter._dialog_handler(SID)(
        agent, {"id": "y", "method": "notify", "message": "hello"}
    )
    assert server.sent("extension_ui_response") == [
        {"type": "extension_ui_response", "id": "x", "cancelled": True}
    ]


async def test_stop_settles_the_turn_as_cancelled() -> None:
    server = Pi([])
    adapter = _adapter(server)
    turn = adapter.turn(SID, [m.TextPart(text="go")])
    assert isinstance(await anext(turn), events.StreamStart)
    reader = asyncio.ensure_future(_rest(turn))
    await adapter.stop(SID)
    seen = await asyncio.wait_for(reader, 5)
    assert isinstance(seen[-1], events.StreamEnd)
    assert seen[-1].finish_reason == "cancelled"


async def _rest(turn: Any) -> list[Any]:
    return [e async for e in turn]


async def test_a_dead_pi_crashes_the_turn() -> None:
    server = Pi([])
    adapter = _adapter(server)
    server.on_prompt = lambda: server.process._lines.put_nowait("")
    with pytest.raises(harness_errors.AgentCrashedError):
        await _turn(adapter)


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
    messages = [
        msg
        for entry in pi._active_branch(text)
        for msg in pi._stored_to_messages(entry, replay=True)
    ]
    assert [msg.role for msg in messages] == ["user", "assistant", "tool"]
    result = messages[2].parts[0]
    assert isinstance(result, m.ToolResultPart)
    assert (result.result, result.result_kind) == ("hi", "error")
    assert all(msg.replay for msg in messages)


@pytest.fixture
def agent_dir(tmp_path: Path) -> Path:
    return tmp_path / "agent"


async def test_a_staged_history_is_listed_and_read_back(
    tmp_path: Path, agent_dir: Path
) -> None:
    """The store helper, for real: write where pi looks, find it there."""
    project = tmp_path / "project"
    project.mkdir()
    workspace = Local(project, env={"PI_CODING_AGENT_DIR": str(agent_dir)})
    adapter = pi.PiAdapter()
    adapter._workspace = workspace
    history = [
        m.Message(role="user", parts=[m.TextPart(text="the word is HERON")]),
        m.Message(
            role="assistant",
            parts=[
                m.ReasoningPart(text="hmm"),
                m.TextPart(text="Reading."),
                m.ToolCallPart(
                    tool_call_id="c1",
                    tool_name="read",
                    tool_args='{"path": "a.txt"}',
                ),
            ],
        ),
        m.Message(
            role="tool",
            parts=[
                m.ToolResultPart(
                    tool_call_id="c1", tool_name="read", result="milk"
                )
            ],
        ),
        m.Message(role="assistant", parts=[m.TextPart(text="It says milk.")]),
    ]
    with pytest.warns(UserWarning, match="reasoning"):
        session_id = await adapter.stage_history(history)
    infos = await adapter.list_sessions()
    assert [i.session_id for i in infos] == [session_id]
    assert infos[0].title == "the word is HERON"
    assert infos[0].kind == "pi"
    files = list(agent_dir.glob("sessions/*/*.jsonl"))
    assert len(files) == 1
    assert files[0].parent.name.startswith("--") and files[0].name.endswith(
        f"_{session_id}.jsonl"
    )
    back = await adapter.history(session_id)
    assert [msg.role for msg in back] == [
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    call = back[1].parts[-1]
    assert isinstance(call, m.ToolCallPart)
    assert json.loads(call.tool_args) == {"path": "a.txt"}
    with pytest.raises(harness_errors.ResumeFailedError):
        await adapter.history("not-a-session")


def test_handoff_merges_assistant_messages_and_keeps_tool_results() -> None:
    from ai.harnesses.experimental import _handoff

    rendered = _handoff.to_pi_messages(
        [
            m.Message(role="user", parts=[m.TextPart(text="hi")]),
            m.Message(role="assistant", parts=[m.TextPart(text="one")]),
            m.Message(
                role="assistant",
                parts=[
                    m.ToolCallPart(
                        tool_call_id="c1", tool_name="Bash", tool_args="{}"
                    )
                ],
            ),
            m.Message(
                role="tool",
                parts=[
                    m.ToolResultPart(
                        tool_call_id="c1",
                        tool_name="Bash",
                        result={"ok": True},
                        result_kind="error",
                    )
                ],
            ),
        ],
        timestamp=1,
    )
    assert [r["role"] for r in rendered] == ["user", "assistant", "toolResult"]
    assert [b["type"] for b in rendered[1]["content"]] == ["text", "toolCall"]
    assert rendered[1]["stopReason"] == "toolUse"
    assert rendered[2]["content"] == [{"type": "text", "text": '{"ok": true}'}]
    assert rendered[2]["isError"] is True


def test_options_reach_pi_in_its_own_spelling() -> None:
    from ai.workspaces.experimental import vercel_ai_gateway

    adapter = pi.PiAdapter(
        model="anthropic/claude-haiku-4.5",
        gateway=vercel_ai_gateway(api_key="k"),
        effort="high",
        instructions="be brief",
        mcp_servers={"web": {"type": "http", "url": "https://mcp.example"}},
        tools=[],
        disallowed_tools=["bash"],
    )
    adapter._workspace = fakes.FakeWorkspace()
    adapter._extension = "/home/.ai-python/pi/extension.js"
    argv, env, running_as = adapter.tui_launch(None)
    assert argv[argv.index("--model") + 1] == (
        "vercel-ai-gateway/anthropic/claude-haiku-4.5"
    )
    assert argv[argv.index("--thinking") + 1] == "high"
    assert argv[argv.index("--append-system-prompt") + 1] == "be brief"
    assert "--no-tools" in argv
    assert argv[argv.index("--exclude-tools") + 1] == "bash"
    assert argv[argv.index("--session-id") + 1] == running_as
    assert argv[argv.index("-e") + 1] == adapter._extension
    assert env["AI_GATEWAY_API_KEY"] == "k"
    config = json.loads(env[pi.CONFIG_VAR])
    # The person at the TUI is the gate; the type key is claude's, not pi's.
    assert config == {
        "approve": False,
        "mcp": {"web": {"url": "https://mcp.example"}},
        "gateway": {},
    }


def test_pi_refuses_max_turns() -> None:
    from ai.harnesses.experimental import pi as make_pi

    with pytest.raises(harness_errors.UnsupportedError, match="max_turns"):
        make_pi(workspace=fakes.FakeWorkspace(), max_turns=3)


class _LaunchFails(fakes.FakeWorkspace):
    """A workspace whose launches fail with what `exc` names."""

    def __init__(self, exc: BaseException) -> None:
        super().__init__()
        self.exc = exc

    async def spawn(self, argv: Any, *, env: Any = None) -> Any:
        raise self.exc


class _Lock:
    def __init__(self) -> None:
        self.held: set[str] = set()

    async def acquire(self, session_id: str, **kwargs: Any) -> None:
        self.held.add(session_id)

    async def release(self, session_id: str) -> None:
        self.held.discard(session_id)


@pytest.mark.parametrize(
    "exc", [TimeoutError(), asyncio.CancelledError(), OSError("no pi")]
)
@pytest.mark.parametrize("verb", ["resume", "fork"])
async def test_a_failed_launch_releases_its_claim(
    exc: BaseException, verb: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Codex review: only a HarnessError released the lock, so a timeout,
    a cancellation or a workspace error left the conversation claimed and
    every other client got SessionBusyError until the claim went stale."""
    adapter = pi.PiAdapter()
    adapter._workspace = _LaunchFails(exc)
    adapter._extension = "/ext.js"
    lock = _Lock()
    monkeypatch.setattr(adapter, "_lock", lock)

    async def found(session_id: str) -> dict[str, Any]:
        return {"id": session_id, "path": f"/s/{session_id}.jsonl"}

    monkeypatch.setattr(adapter, "_find", found)
    # Not a harness failure: it propagates as itself, never renamed.
    with pytest.raises(type(exc)):
        if verb == "resume":
            await adapter.resume_session("S")
        else:
            await adapter.fork("S")
    assert lock.held == set()
    assert adapter._held == set()


async def test_stop_withdraws_an_approval_nobody_has_answered() -> None:
    """Codex review, then measured on pi 1.3.0: pi answers `abort` only once
    it is idle, and a tool call waiting on the hook keeps it busy — stop()
    timed out after 60s and the next run waited behind it. The question is
    withdrawn first now: pi hears `cancelled`, blocks the call, goes idle."""
    server = Pi([])
    parked = asyncio.Event()

    async def never(ctx: ApprovalContext) -> Allow:
        parked.set()
        await asyncio.Event().wait()  # a person who never answers
        return Allow()

    adapter = _adapter(server, never)
    asking = True

    def answer(message: dict[str, Any]) -> None:
        nonlocal asking
        kind = message.get("type")
        if kind == "extension_ui_response":
            asking = False
            if server.pending_abort is not None:
                Pi.answer(server, server.pending_abort)
        elif kind == "abort" and asking:
            server.pending_abort = message  # like pi: busy, no answer yet
        else:
            Pi.answer(server, message)

    server.process.answer = answer
    server.on_prompt = lambda: server.says(
        {
            "type": "extension_ui_request",
            "id": "ui-1",
            "method": "input",
            "title": pi.APPROVE,
            "placeholder": json.dumps({"toolCallId": "t", "toolName": "bash"}),
        }
    )
    turn = adapter.turn(SID, [m.TextPart(text="go")])
    assert isinstance(await anext(turn), events.StreamStart)
    reader = asyncio.ensure_future(_rest(turn))
    await asyncio.wait_for(parked.wait(), 2)
    async with asyncio.timeout(2):
        await adapter.stop(SID)
    seen = await asyncio.wait_for(reader, 2)
    assert isinstance(seen[-1], events.StreamEnd)
    assert seen[-1].finish_reason == "cancelled"
    assert server.sent("extension_ui_response") == [
        {"type": "extension_ui_response", "id": "ui-1", "cancelled": True}
    ]
    assert adapter._agents[SID].pending == {}


def test_without_a_model_pi_chooses_as_it_always_does() -> None:
    """No model means pi's own choice, gateway or not — measured: a fresh pi
    with only a gateway key picks its own default gateway model, and with
    no credentials refuses the prompt (NotAuthenticatedError). The SDK
    passes nothing that would change either."""
    from ai.workspaces.experimental import vercel_ai_gateway

    for gateway in (None, vercel_ai_gateway(api_key="k")):
        adapter = pi.PiAdapter(gateway=gateway)
        adapter._workspace = fakes.FakeWorkspace()
        adapter._extension = "/ext.js"
        argv, _, _ = adapter.tui_launch(None)
        assert "--model" not in argv and "--provider" not in argv


class _Spawns(fakes.FakeWorkspace):
    """A workspace whose every launch is the scripted pi."""

    def __init__(self, server: Pi) -> None:
        super().__init__()
        self.server = server

    async def spawn(self, argv: Any, *, env: Any = None) -> Any:
        return _Terminable(self.server)


class _Terminable(fakes.FakeProcess):
    """The scripted pi's process, noting whether it was terminated."""

    def __init__(self, server: Pi) -> None:
        super().__init__(server.process.answer)
        self._lines = server.process._lines
        self.written = server.process.written
        self.server = server

    async def terminate(self) -> None:
        self.server.terminated = True
        await super().terminate()


def _launcher(server: Pi) -> pi.PiAdapter:
    adapter = pi.PiAdapter()
    adapter._workspace = _Spawns(server)
    adapter._extension = "/ext.js"
    return adapter


async def test_a_session_without_the_extension_is_refused() -> None:
    """Codex review: a gate that did not load is a gate that fails open.
    The extension's marker command must be listed, or nothing starts."""
    server = Pi()
    server.extension_loaded = False
    adapter = _launcher(server)
    with pytest.raises(harness_errors.AgentCrashedError, match="extension"):
        await adapter._spawn(SID, ["--session-id", SID])
    assert server.terminated
    assert SID not in adapter._agents


async def test_a_process_whose_claim_fails_does_not_outlive_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Codex review: a failed pid claim left pi registered and running
    while the caller released the lock — a writer nobody accounted for."""
    server = Pi()
    adapter = _launcher(server)

    class Lock:
        async def claim_pid(self, *args: Any, **kwargs: Any) -> None:
            raise OSError("the workspace went away")

    monkeypatch.setattr(adapter, "_lock", Lock())
    with pytest.raises(OSError):
        await adapter._spawn(SID, ["--session-id", SID])
    assert server.terminated
    assert SID not in adapter._agents


async def test_closing_a_session_ends_the_turn_reading_it() -> None:
    """Codex review: close() stopped the event producer and nobody told the
    turn waiting on it, so it waited forever."""
    server = Pi([])
    adapter = _adapter(server)
    turn = adapter.turn(SID, [m.TextPart(text="go")])
    assert isinstance(await anext(turn), events.StreamStart)
    reader = asyncio.ensure_future(_rest(turn))
    await asyncio.sleep(0)
    await adapter.close_session(SID)
    with pytest.raises(harness_errors.HarnessClosedError):
        await asyncio.wait_for(reader, 2)


async def test_reopening_a_session_hands_it_over(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Codex review: the lock lets its own holder through, so a second
    resume spawned a second pi on the same transcript and orphaned the
    first. Now the first is ended before the second starts — and while a
    turn runs on it, the resume is refused."""
    server = Pi()
    adapter = _adapter(server)
    old = adapter._agents[SID]
    spawned: list[Any] = []

    async def found(session_id: str) -> dict[str, Any]:
        return {"id": session_id, "path": "/s.jsonl"}

    async def spawn(session_id: str, args: list[str]) -> None:
        assert session_id not in adapter._agents, "one pi at a time"
        spawned.append(session_id)

    monkeypatch.setattr(adapter, "_find", found)
    monkeypatch.setattr(adapter, "_spawn", spawn)
    adapter._running.add(SID)
    with pytest.raises(harness_errors.SessionBusyError):
        await adapter.resume_session(SID)
    assert adapter._agents[SID] is old and spawned == []
    adapter._running.discard(SID)
    await adapter.resume_session(SID)
    assert spawned == [SID]
    assert old.dead is not None, "the first pi was closed"


async def test_a_question_asked_while_stopping_is_withdrawn_at_once() -> None:
    """Codex review: an approval that arrived after stop() withdrew the
    pending ones still ran the hook, and `abort` waited on it."""
    server = Pi()
    asked: list[Any] = []

    async def approve(ctx: ApprovalContext) -> Allow:
        asked.append(ctx)
        return Allow()

    adapter = _adapter(server, approve)
    adapter._interrupted.add(SID)
    await adapter._dialog_handler(SID)(
        adapter._agents[SID],
        {"id": "late", "method": "input", "title": pi.APPROVE},
    )
    assert asked == []
    assert server.sent("extension_ui_response") == [
        {"type": "extension_ui_response", "id": "late", "cancelled": True}
    ]


def test_history_keeps_what_pi_showed_the_model() -> None:
    """Codex review: extension-injected context was dropped from history,
    so a resumed or handed-off conversation lost what the agent was told."""
    entries: list[dict[str, Any]] = [
        {"type": "custom_message", "content": "ticket #42 is P1"},
        {
            "type": "message",
            "message": {
                "role": "bashExecution",
                "command": "ls",
                "output": "a.txt",
            },
        },
        {
            "type": "message",
            "message": {
                "role": "bashExecution",
                "command": "env",
                "output": "SECRET=1",
                "excludeFromContext": True,
            },
        },
        {"type": "branch_summary", "summary": "tried approach A first"},
        {"type": "compaction", "summary": "everything above"},
        {"type": "model_change", "provider": "x"},
    ]
    messages = [msg for e in entries for msg in pi._stored_to_messages(e)]
    assert [(msg.role, msg.text) for msg in messages] == [
        ("user", "ticket #42 is P1"),
        ("user", "Ran `ls`\n```\na.txt\n```"),
        ("user", "tried approach A first"),
    ]


def test_handoff_never_makes_instructions_the_agents_own_words() -> None:
    """Codex review: a system message fell through to the assistant branch
    and merged into the model's reply. Measured: as a pi system message it
    replaces pi's own prompt and tools, so it travels as a user note."""
    from ai.harnesses.experimental import _handoff

    rendered = _handoff.to_pi_messages(
        [
            m.Message(role="assistant", parts=[m.TextPart(text="hello")]),
            m.Message(role="system", parts=[m.TextPart(text="be terse")]),
            m.Message(role="assistant", parts=[m.TextPart(text="ok")]),
        ],
        timestamp=1,
    )
    assert [r["role"] for r in rendered] == ["assistant", "user", "assistant"]
    assert "be terse" in rendered[1]["content"][0]["text"]
    assert rendered[0]["content"] == [{"type": "text", "text": "hello"}]


async def test_a_turn_stopped_before_its_prompt_is_sent_sends_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Codex review: stop() while the prompt's files were being read found
    pi idle, and the prompt went out anyway — a cancelled turn that still
    called the model and ran tools."""
    server = Pi()
    adapter = _adapter(server)

    async def slow(prompt: Any) -> dict[str, Any]:
        await adapter.stop(SID)  # the Stop button, mid-download
        return {"type": "prompt", "message": "go"}

    monkeypatch.setattr(pi, "_prompt_command", slow)
    seen = await _turn(adapter)
    assert server.sent("prompt") == []
    end = seen[-1]
    assert isinstance(end, events.StreamEnd)
    assert end.finish_reason == "cancelled"


async def test_a_failed_attempt_is_not_part_of_the_retried_answer() -> None:
    """Codex review: pi retries a failed model call, and both attempts end
    in `message_end` — the half-written first one was glued to the reply."""
    failed = {
        **_assistant({"type": "text", "text": "Half an ans"}, stop="error"),
        "errorMessage": "529 overloaded",
    }
    server = Pi(
        [
            {"type": "message_end", "message": failed},
            {"type": "auto_retry_start", "attempt": 1},
            {
                "type": "message_end",
                "message": _assistant({"type": "text", "text": "The answer"}),
            },
            {"type": "auto_retry_end", "success": True},
            {"type": "agent_settled", "aborted": False},
        ]
    )
    seen = await _turn(_adapter(server))
    end = seen[-1]
    assert isinstance(end, events.StreamEnd)
    assert end.finish_reason == "stop"
    assert end.message.text == "The answer"


async def test_two_resumes_at_once_start_one_pi_at_a_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Codex review: two resumes racing through the hand-over each started
    a pi, one of them untracked."""
    adapter = _adapter(Pi())
    live: list[str] = []
    most = 0

    async def found(session_id: str) -> dict[str, Any]:
        return {"id": session_id, "path": "/s.jsonl"}

    async def spawn(session_id: str, args: list[str]) -> None:
        nonlocal most
        live.append(session_id)
        most = max(most, len(live))
        await asyncio.sleep(0.01)
        adapter._agents[session_id] = pi._Agent(
            fakes.FakeProcess(), on_dialog=None
        )
        live.remove(session_id)

    monkeypatch.setattr(adapter, "_find", found)
    monkeypatch.setattr(adapter, "_spawn", spawn)
    await asyncio.gather(
        adapter.resume_session(SID), adapter.resume_session(SID)
    )
    assert most == 1


async def test_a_handed_off_pi_conversation_stays_on_its_model(
    tmp_path: Path, agent_dir: Path
) -> None:
    """Measured with afk: a pushed pi conversation ran on pi's default model
    in the sandbox (refused by the gateway), not the one it was on. History
    now says which model wrote each reply, and staging writes it back the
    way pi records a switch, so pi restores it on open."""
    project = tmp_path / "project"
    project.mkdir()
    workspace = Local(project, env={"PI_CODING_AGENT_DIR": str(agent_dir)})
    adapter = pi.PiAdapter()
    adapter._workspace = workspace
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
    history = [
        m.Message(role="user", parts=[m.TextPart(text="hello")]),
        *pi._stored_to_messages(stored),
    ]
    assert history[1].provider_metadata == {
        "pi": {
            "api": "anthropic-messages",
            "provider": "vercel-ai-gateway",
            "model": "anthropic/claude-haiku-4.5",
            "thinkingLevel": "high",
        }
    }
    session_id = await adapter.stage_history(history)
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
    # pi restores the model from the last reply's author: it is the real one.
    reply = entries[-1]["message"]
    assert (reply["api"], reply["provider"], reply["model"]) == (
        "anthropic-messages",
        "vercel-ai-gateway",
        "anthropic/claude-haiku-4.5",
    )
    # A chain, root first: pi reads it as one branch.
    assert [e.get("parentId") for e in entries[1:]] == [
        None,
        *[e["id"] for e in entries[1:-1]],
    ]


async def test_without_pi_history_no_model_is_carried(
    tmp_path: Path, agent_dir: Path
) -> None:
    """From another harness there is no pi model to name: pi chooses."""
    project = tmp_path / "project"
    project.mkdir()
    adapter = pi.PiAdapter()
    adapter._workspace = Local(
        project, env={"PI_CODING_AGENT_DIR": str(agent_dir)}
    )
    session_id = await adapter.stage_history(
        [
            m.Message(role="user", parts=[m.TextPart(text="hello")]),
            m.Message(role="assistant", parts=[m.TextPart(text="hi")]),
        ]
    )
    (written,) = agent_dir.glob(f"sessions/*/*_{session_id}.jsonl")
    assert "model_change" not in written.read_text()
