"""The Pi worker against a scripted `pi --mode rpc`, offline.

The records are the shapes a real pi 1.5.0 sent when measured. The fake
plays a run's records BEFORE the `prompt` response unless a test says
otherwise: the worker must still put the user message first.
"""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import pytest

import ai
from ai import workers
from ai.types import messages as m
from ai.workers import _process
from ai.workers.pi import Pi
from ai.workers.pi import wire as wire_

from .. import fakes


def update(kind: str, index: int = 0, **fields: Any) -> dict[str, Any]:
    return {
        "type": "message_update",
        "assistantMessageEvent": {
            "type": kind,
            "contentIndex": index,
            **fields,
        },
    }


def assistant(*content: dict[str, Any], stop: str = "stop") -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": list(content),
        "api": "anthropic-messages",
        "provider": "vercel-ai-gateway",
        "model": "anthropic/claude-opus-5.5-fast",
        "usage": {
            "input": 10,
            "output": 5,
            "cacheRead": 100,
            "cacheWrite": 0,
            "cost": {"total": 0.01},
        },
        "stopReason": stop,
    }


def user(text: str) -> list[dict[str, Any]]:
    message = {"role": "user", "content": [{"type": "text", "text": text}]}
    return [
        {"type": "message_start", "message": message},
        {"type": "message_end", "message": message},
    ]


def begin(text: str) -> list[dict[str, Any]]:
    """The start of a run: pi's own system message, then the prompt."""
    system = {"role": "system", "content": "", "sections": {}}
    return [
        {"type": "agent_start"},
        {"type": "turn_start"},
        {"type": "message_start", "message": system},
        {"type": "message_end", "message": system},
        *user(text),
    ]


def said(text: str, index: int = 0) -> list[dict[str, Any]]:
    """One assistant message that streams `text`."""
    return [
        {"type": "message_start", "message": assistant()},
        update("text_start", index),
        update("text_delta", index, delta=text),
        update("text_end", index, content=text),
        {
            "type": "message_end",
            "message": assistant({"type": "text", "text": text}),
        },
    ]


SETTLED: list[dict[str, Any]] = [
    {"type": "agent_end", "messages": []},
    {"type": "agent_settled", "aborted": False},
]

CALL = {
    "type": "toolCall",
    "id": "toolu_1",
    "name": "bash",
    "arguments": {"command": "echo hi"},
}

# thinking -> tool call -> tool result -> text: what pi sent for
# "run `echo hi`, then say done".
TOOL_TURN: list[dict[str, Any]] = [
    *begin("say done"),
    {"type": "message_start", "message": assistant()},
    update("thinking_start"),
    update("thinking_delta", delta="plan"),
    update("thinking_end", content="plan"),
    update("toolcall_start", 1, id="toolu_1", toolName="bash"),
    update("toolcall_delta", 1, delta='{"command": "ec'),
    update("toolcall_end", 1, toolCall=CALL),
    {
        "type": "message_end",
        "message": assistant(
            {"type": "thinking", "thinking": "plan"}, CALL, stop="toolUse"
        ),
    },
    {
        "type": "tool_execution_start",
        "toolCallId": "toolu_1",
        "toolName": "bash",
        "args": {"command": "echo hi"},
    },
    {
        "type": "tool_execution_end",
        "toolCallId": "toolu_1",
        "toolName": "bash",
        "result": {"content": [{"type": "text", "text": "hi\n"}]},
        "isError": False,
    },
    *said("done"),
    *SETTLED,
]


class PiServer:
    """Answers like `pi --mode rpc`.

    `turns[i]` plays out on the i-th prompt that starts a run, then the
    response; `on_prompt`, if set, does it all instead. A prompt
    while a run streams is queued, as a steer. `abort` settles the run as
    aborted, then answers.
    """

    def __init__(self) -> None:
        self.process = fakes.FakeProcess(self.answer)
        self.turns: list[list[dict[str, Any]]] = []
        self.on_prompt: Callable[[dict[str, Any]], None] | None = None
        self.started = 0
        self.streaming = False
        self.loaded = True
        self.prompt_error: str | None = None
        self.spawned: list[tuple[list[str], dict[str, Any]]] = []

    @property
    def session_id(self) -> str:
        argv = self.spawned[-1][0]
        return argv[argv.index("--session-id") + 1]

    def says(self, *records: dict[str, Any]) -> None:
        for record in records:
            if record.get("type") == "agent_start":
                self.streaming = True
            elif record.get("type") == "agent_settled":
                self.streaming = False
            self.process.server_says(record)

    def respond(self, command: dict[str, Any], **fields: Any) -> None:
        self.says(
            {
                "type": "response",
                "id": command["id"],
                "command": command["type"],
                "success": True,
                **fields,
            }
        )

    def answer(self, command: dict[str, Any]) -> None:
        kind = command.get("type")
        if kind == "prompt":
            if self.prompt_error is not None:
                self.respond(command, success=False, error=self.prompt_error)
            elif self.streaming:
                self.says(
                    {
                        "type": "queue_update",
                        "steering": [command["message"]],
                        "followUp": [],
                    }
                )
                self.respond(command, data={"disposition": "queued"})
            elif self.on_prompt is not None:
                self.on_prompt(command)
            else:
                turn = self.turns[self.started] if self.turns else []
                self.started += 1
                self.says(*turn)
                self.respond(command, data={"disposition": "started"})
        elif kind == "abort":
            self.says(
                {
                    "type": "message_end",
                    "message": assistant(stop="aborted"),
                },
                {"type": "agent_end", "messages": []},
                {"type": "agent_settled", "aborted": True},
            )
            self.respond(command)
        elif kind == "get_state":
            self.respond(
                command,
                data={
                    "sessionId": self.session_id,
                    "isStreaming": self.streaming,
                },
            )
        elif kind == "get_commands":
            names = [wire_.READY] if self.loaded else []
            self.respond(
                command, data={"commands": [{"name": n} for n in names]}
            )
        elif "id" in command:
            self.respond(command)

    def sent(self, kind: str) -> list[dict[str, Any]]:
        return [w for w in self.process.written if w.get("type") == kind]


@pytest.fixture
def cwd(tmp_path: Path) -> str:
    project = tmp_path / "project"
    project.mkdir()
    return os.path.realpath(project)


@pytest.fixture
def agent_dir(tmp_path: Path) -> Path:
    return tmp_path / "agent"


@pytest.fixture
def server(monkeypatch: pytest.MonkeyPatch) -> PiServer:
    server = PiServer()

    async def probe(kind: str, executable: str, **kwargs: Any) -> str:
        return "1.5.0"

    async def spawn(argv: list[str], **kwargs: Any) -> _process.Process:
        server.spawned.append((argv, kwargs))
        if server.process.terminated:
            server.process = fakes.FakeProcess(server.process.answer)
        return cast("_process.Process", server.process)

    monkeypatch.setattr(_process, "probe", probe)
    monkeypatch.setattr(_process, "spawn", spawn)
    return server


def make(cwd: str, agent_dir: Path, **fields: Any) -> Pi:
    provider = workers.PiProvider(
        executable=fields.pop("executable", None),
        env={"PI_CODING_AGENT_DIR": str(agent_dir), **fields.pop("env", {})},
    )
    return Pi(cwd=cwd, provider=provider, **fields)


async def run(worker: Pi, messages: list[m.Message]) -> ai.Stream[str]:
    async with asyncio.timeout(5):
        async with ai.stream(worker, messages) as s:
            async for _ in s:
                pass
    return s


def session_file(agent_dir: Path, session_id: str) -> Path:
    (path,) = agent_dir.glob(f"sessions/*/*_{session_id}.jsonl")
    return path


async def test_a_turn_streams_in_the_model_view(
    server: PiServer, cwd: str, agent_dir: Path
) -> None:
    server.turns = [TOOL_TURN]
    prompt = ai.user_message("say done")
    worker = make(cwd, agent_dir)
    async with worker.session() as session:
        s = await run(worker, [prompt])
    message = s.message
    assert message.reasoning == "plan"
    assert message.text == "done"
    (call,) = message.builtin_tool_calls
    assert call.tool_name == "bash"
    # The arguments travel once, whole: pi's fragments are not what it
    # settles on.
    assert json.loads(call.tool_args) == {"command": "echo hi"}
    (result,) = message.builtin_tool_returns
    assert result.result == "hi\n"
    # Two model calls: cached input counts as input, cost rides in raw.
    usage = message.usage
    assert usage is not None
    assert (usage.input_tokens, usage.output_tokens) == (220, 10)
    assert usage.cache_read_tokens == 200
    assert usage.raw == {"cost_usd": 0.02}
    # Acceptance came first, although the run came before the response.
    assert [msg.role for msg in session.messages] == ["user", "assistant"]
    assert session.messages[0].id == prompt.id
    assert session.id == server.session_id
    (sent,) = server.sent("prompt")
    assert sent["message"] == "say done"
    assert sent["streamingBehavior"] == "steer"
    # Who wrote the reply, so a seeded pi continues on the same model.
    assert message.provider_metadata == {
        "pi": {
            "api": "anthropic-messages",
            "provider": "vercel-ai-gateway",
            "model": "anthropic/claude-opus-5.5-fast",
        }
    }


async def test_start_runs_pi_with_the_extension_and_the_bridge(
    server: PiServer, cwd: str, agent_dir: Path
) -> None:
    worker = make(
        cwd,
        agent_dir,
        executable="/custom/pi",
        env={"ANTHROPIC_API_KEY": "k"},
        model_name="claude-test",
        model_provider="anthropic",
        thinking="high",
        instructions="Be brief.",
        allowed_tools=[],
        disallowed_tools=["write"],
        args=["--no-skills"],
    )
    async with worker.session() as session:
        await session.start()
        (argv, kwargs) = server.spawned[0]
        assert argv[:3] == ["/custom/pi", "--mode", "rpc"]
        assert argv[argv.index("-e") + 1] == wire_.EXTENSION
        assert Path(wire_.EXTENSION).is_file()
        assert argv[argv.index("--session-id") + 1] == session.id
        for flag, value in [
            ("--model", "claude-test"),
            ("--provider", "anthropic"),
            ("--thinking", "high"),
            ("--append-system-prompt", "Be brief."),
            ("--exclude-tools", "write"),
        ]:
            assert argv[argv.index(flag) + 1] == value
        # No pi tools, but the bridge's (extension) tools stay.
        assert "--no-builtin-tools" in argv
        assert "--no-tools" not in argv
        assert argv[-1] == "--no-skills"
        assert kwargs["cwd"] == cwd
        env = kwargs["env"]
        assert env["ANTHROPIC_API_KEY"] == "k"
        assert session._run is not None
        assert json.loads(env[wire_.CONFIG_VAR]) == {
            "mcp": {
                "ai_sdk": {
                    "url": session._run.bridge.url,
                    "exposure": "direct",
                    "timeout": 86400,
                }
            }
        }
    assert server.process.terminated


async def test_a_pi_without_the_extension_is_refused(
    server: PiServer, cwd: str, agent_dir: Path
) -> None:
    server.loaded = False
    worker = make(cwd, agent_dir)
    with pytest.raises(workers.WorkerError, match="did not load"):
        async with worker.session() as session:
            await session.start()
    assert server.process.terminated


async def test_close_is_idempotent(
    server: PiServer, cwd: str, agent_dir: Path
) -> None:
    worker = make(cwd, agent_dir)
    session = worker.session()
    async with session:
        await session.start()
    await session.close()
    assert server.process.terminated
    wire = worker.provider._make_wire(worker)
    await wire.close()
    await wire.close()


async def test_bridge_calls_are_ai_tool_calls_not_builtin(
    server: PiServer, cwd: str, agent_dir: Path
) -> None:
    @ai.tool
    async def lookup(word: str) -> str:
        """Look a word up."""
        return word.upper()

    name = "mcp__ai_sdk__lookup"
    bridge_call = {
        "type": "toolCall",
        "id": "toolu_b",
        "name": name,
        "arguments": {"word": "heron"},
    }
    answers: list[dict[str, Any]] = []

    async def call_bridge(command: dict[str, Any]) -> None:
        # pi calls the MCP server the extension registered.
        env = server.spawned[-1][1]["env"]
        url = json.loads(env[wire_.CONFIG_VAR])["mcp"]["ai_sdk"]["url"]
        reply = await fakes.mcp_call(
            url,
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {"name": "lookup", "arguments": {"word": "heron"}},
            },
        )
        answers.append(reply)
        text = reply["result"]["content"][0]["text"]
        server.says(
            {
                "type": "tool_execution_end",
                "toolCallId": "toolu_b",
                "toolName": name,
                "result": {"content": [{"type": "text", "text": text}]},
                "isError": False,
            },
            *said(f"got {text}"),
            *SETTLED,
        )

    tasks: list[asyncio.Task[None]] = []

    def turn(command: dict[str, Any]) -> None:
        server.says(
            *begin("go"),
            {"type": "message_start", "message": assistant()},
            update("toolcall_start", 0, id="toolu_b", toolName=name),
            update("toolcall_end", 0, toolCall=bridge_call),
            {
                "type": "message_end",
                "message": assistant(bridge_call, stop="toolUse"),
            },
            {
                "type": "tool_execution_start",
                "toolCallId": "toolu_b",
                "toolName": name,
                "args": {"word": "heron"},
            },
        )
        server.respond(command, data={"disposition": "started"})
        tasks.append(asyncio.create_task(call_bridge(command)))

    server.on_prompt = turn
    worker = make(cwd, agent_dir)
    try:
        async with worker.session() as session:
            async with asyncio.timeout(5):
                async with ai.Agent(tools=[lookup]).run(
                    worker, [ai.user_message("go")]
                ) as agent_run:
                    async for _ in agent_run:
                        pass
            roles = [msg.role for msg in session.messages]
            assert roles == ["user", "assistant", "tool", "assistant"]
            (call,) = session.messages[1].tool_calls
            assert call.tool_name == "lookup"
            for message in session.messages:
                assert message.builtin_tool_calls == []
                assert message.builtin_tool_returns == []
            assert session.messages[3].text == "got HERON"
            assert answers[0]["result"]["isError"] is False
    finally:
        for task in tasks:
            await task


async def test_a_send_during_a_run_steers_it(
    server: PiServer, cwd: str, agent_dir: Path
) -> None:
    server.turns = [
        [
            *begin("fix"),
            {"type": "message_start", "message": assistant()},
            update("text_start"),
            update("text_delta", delta="Look"),
        ]
    ]
    worker = make(cwd, agent_dir)
    async with worker.session() as session:
        feed = session.events()
        await session.send(ai.user_message("fix"))
        await session.send(ai.user_message("skip slow ones"))
        assert len(server.sent("prompt")) == 2
        steer = server.sent("prompt")[-1]
        assert steer["message"] == "skip slow ones"
        assert steer["streamingBehavior"] == "steer"
        # Queued, not delivered: pi did not put it in the conversation yet.
        assert "skip slow ones" not in [msg.text for msg in session.messages]
        server.says(
            update("text_end", content="Look"),
            {
                "type": "message_end",
                "message": assistant({"type": "text", "text": "Look"}),
            },
            {"type": "turn_end"},
            {"type": "turn_start"},
            {"type": "queue_update", "steering": [], "followUp": []},
            *user("skip slow ones"),
            *said("OK"),
            *SETTLED,
        )
        async with asyncio.timeout(5):
            while len(session.messages) < 4:
                await anext(feed)
        assert [(msg.role, msg.text) for msg in session.messages] == [
            ("user", "fix"),
            ("assistant", "Look"),
            ("user", "skip slow ones"),
            ("assistant", "OK"),
        ]


async def test_interrupt_ends_the_run_as_interrupted(
    server: PiServer, cwd: str, agent_dir: Path
) -> None:
    server.turns = [
        [
            *begin("long job"),
            {"type": "message_start", "message": assistant()},
            update("text_start"),
            update("text_delta", delta="work"),
        ]
    ]
    worker = make(cwd, agent_dir)
    async with worker.session() as session:
        feed = session.events()
        await session.send(ai.user_message("long job"))
        async with asyncio.timeout(5):
            async for event in feed:
                if isinstance(event, ai.events.TextDelta):
                    break
            await session.interrupt()
        assert len(server.sent("abort")) == 1
        assert (session.messages[-1].provider_metadata or {}).get("worker") == {
            "stop_reason": "interrupted"
        }
        assert session.messages[-1].text == "work"


async def test_a_rejected_prompt_without_a_key_is_not_authenticated(
    server: PiServer, cwd: str, agent_dir: Path
) -> None:
    """Measured: pi refuses the prompt itself when it has no credentials."""
    server.prompt_error = "No API key found for anthropic.\n\nUse /login"
    worker = make(cwd, agent_dir)
    async with worker.session():
        with pytest.raises(workers.NotAuthenticatedError):
            await run(worker, [ai.user_message("go")])


async def test_a_failed_run_without_a_key_is_not_authenticated(
    server: PiServer, cwd: str, agent_dir: Path
) -> None:
    server.turns = [
        [
            *begin("go"),
            {"type": "message_start", "message": assistant()},
            {
                "type": "message_end",
                "message": {
                    **assistant(stop="error"),
                    "errorMessage": "401 Unauthorized: invalid api key",
                },
            },
            *SETTLED,
        ]
    ]
    worker = make(cwd, agent_dir)
    async with worker.session() as session:
        with pytest.raises(workers.NotAuthenticatedError):
            await run(worker, [ai.user_message("go")])
        assert "error" in (session.messages[-1].provider_metadata or {}).get(
            "worker", {}
        )


async def test_a_new_session_is_seeded_with_the_past(
    server: PiServer, cwd: str, agent_dir: Path
) -> None:
    server.turns = [[*begin("two"), *said("fine"), *SETTLED]]
    history = [
        ai.user_message("one"),
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
        ai.user_message("two"),
    ]
    worker = make(cwd, agent_dir)
    with pytest.warns(UserWarning, match="reasoning"):
        async with (
            worker.session() as session
        ):  # the first call starts pi, seeded
            s = await run(worker, history)
    assert s.text == "fine"
    session_id = session.id
    assert session_id is not None
    assert session_id == server.session_id
    path = session_file(agent_dir, session_id)
    assert path.parent.name.startswith("--")
    entries = [json.loads(line) for line in path.read_text().splitlines()]
    header = entries[0]
    assert (header["type"], header["id"], header["cwd"]) == (
        "session",
        session_id,
        cwd,
    )
    stored = [e["message"] for e in entries[1:]]
    assert [msg["role"] for msg in stored] == [
        "user",
        "assistant",
        "toolResult",
    ]
    assert stored[0]["content"] == [{"type": "text", "text": "one"}]
    # The reasoning was not carried; the tool call was, natively.
    assert [b["type"] for b in stored[1]["content"]] == ["text", "toolCall"]
    assert stored[2]["content"][0]["text"] == "milk"
    # A chain, root first: pi reads it as one branch.
    assert [e["parentId"] for e in entries[1:]] == [
        None,
        *[e["id"] for e in entries[1:-1]],
    ]
    (prompt,) = server.sent("prompt")
    assert prompt["message"] == "two"


async def test_an_existing_session_opens_here_only(
    server: PiServer, cwd: str, agent_dir: Path, tmp_path: Path
) -> None:
    server.turns = [[*begin("one"), *said("ok"), *SETTLED]]
    first = make(cwd, agent_dir)
    past = [ai.user_message("zero"), ai.assistant_message("fine")]
    async with first.session() as opened:
        await run(first, [*past, ai.user_message("one")])
    session_id = opened.id
    assert session_id is not None
    assert server.process.terminated
    # The seed made the session file (a real pi also writes it itself).
    path = session_file(agent_dir, session_id)

    again = make(cwd, agent_dir)

    async with again.session(session_id) as session:
        await session.start()
        assert server.session_id == session_id
        assert session.id == session_id

    with pytest.raises(workers.WorkerError, match="no session"):
        unknown = make(cwd, agent_dir)
        async with unknown.session("0000-unknown") as session:
            await session.start()
    with pytest.raises(workers.WorkerError, match="not a pi session id"):
        escape = make(cwd, agent_dir)
        async with escape.session("../escape") as session:
            await session.start()
    # A session that names another directory is not this one's.
    entries = path.read_text().splitlines()
    header = json.loads(entries[0])
    other = path.with_name(f"x_{session_id}-moved.jsonl")
    other.write_text(
        json.dumps(
            {**header, "id": f"{session_id}-moved", "cwd": str(tmp_path)}
        )
        + "\n"
    )
    with pytest.raises(workers.WorkerError, match="belongs to"):
        moved = make(cwd, agent_dir)
        async with moved.session(f"{session_id}-moved") as session:
            await session.start()
    assert len(server.spawned) == 2


async def test_a_dialog_is_answered_without_a_person(
    server: PiServer, cwd: str, agent_dir: Path
) -> None:
    async with make(cwd, agent_dir).session() as session:
        await session.start()
        server.says(
            {
                "type": "extension_ui_request",
                "id": "ui-1",
                "method": "confirm",
                "title": "Allow?",
            },
            {
                "type": "extension_ui_request",
                "id": "ui-2",
                "method": "notify",
                "message": "hello",
            },
        )
        async with asyncio.timeout(5):
            while not server.sent("extension_ui_response"):
                await asyncio.sleep(0.01)
        await asyncio.sleep(0.05)
    assert server.sent("extension_ui_response") == [
        {"type": "extension_ui_response", "id": "ui-1", "cancelled": True}
    ]


async def test_images_travel_as_base64(
    server: PiServer, cwd: str, agent_dir: Path
) -> None:
    server.turns = [[*begin("look"), *said("a dot"), *SETTLED]]
    png = m.FilePart(data=b"\x89PNG", media_type="image/png")
    message = m.Message(role="user", parts=[m.TextPart(text="look"), png])
    worker = make(cwd, agent_dir)
    async with worker.session() as session:
        await run(worker, [message])
        with pytest.raises(workers.UnsupportedError):
            pdf = m.FilePart(data=b"%PDF", media_type="application/pdf")
            await session.send(m.Message(role="user", parts=[pdf]))
    (prompt,) = server.sent("prompt")
    assert prompt["images"] == [
        {"type": "image", "data": "iVBORw==", "mimeType": "image/png"}
    ]


async def test_a_command_that_starts_nothing_ends_the_turn_empty(
    server: PiServer, cwd: str, agent_dir: Path
) -> None:
    def turn(command: dict[str, Any]) -> None:
        server.respond(command, data={"disposition": "handled"})

    server.on_prompt = turn
    worker = make(cwd, agent_dir)
    async with worker.session() as session:
        s = await run(worker, [ai.user_message("/noop")])
    assert s.message.parts == []
    assert [msg.role for msg in session.messages] == ["user", "assistant"]


async def test_a_crash_reaches_the_stream(
    server: PiServer, cwd: str, agent_dir: Path
) -> None:
    server.turns = [
        [
            *begin("go"),
            {"type": "message_start", "message": assistant()},
            update("text_start"),
            update("text_delta", delta="half"),
        ]
    ]

    async def die() -> None:
        await asyncio.sleep(0.05)
        await server.process.terminate()

    worker = make(cwd, agent_dir)

    async with worker.session():
        task = asyncio.create_task(die())
        with pytest.raises(workers.WorkerError, match=r"crashed|stopped"):
            await run(worker, [ai.user_message("go")])
        await task
