"""The Claude Code worker against a scripted claude-agent-sdk client, offline.

The fake replaces `ClaudeSDKClient` only: the records it plays are the real
SDK message types, in the order a real `claude` 2.1.319 sends them.
"""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import AsyncIterable, AsyncIterator, Callable
from pathlib import Path
from typing import Any

import pytest

import ai
from ai import workers
from ai.types import messages as m
from ai.workers import _process

from .. import fakes

sdk = pytest.importorskip("claude_agent_sdk")

from ai.workers.claude_code import ClaudeCode  # noqa: E402

SESSION = "11111111-1111-1111-1111-111111111111"

OnQuery = Callable[[dict[str, Any]], list[Any]]


class FakeCLI:
    """What the test scripts: the records each written message causes.

    `on_query(record)` returns what the CLI says when it reads a message
    (default: echo it). `say()` adds records at any time. `on_interrupt`
    is what an interrupt makes the CLI say.
    """

    def __init__(self) -> None:
        self.clients: list[FakeClient] = []
        self.on_query: OnQuery = lambda record: [echo(record)]
        self.on_interrupt: list[Any] = [
            sdk.UserMessage(
                content=[
                    sdk.TextBlock(
                        text="[Request interrupted by user for tool use]"
                    )
                ],
                uuid="cli-own",
            ),
            result(subtype="error_during_execution", is_error=True),
        ]
        self.sessions = {SESSION}

    @property
    def client(self) -> FakeClient:
        return self.clients[-1]

    def say(self, *records: Any) -> None:
        for record in records:
            self.client.queue.put_nowait(record)


class FakeClient:
    def __init__(self, cli: FakeCLI, options: Any) -> None:
        self.cli = cli
        self.options = options
        self.queue: asyncio.Queue[Any] = asyncio.Queue()
        self.written: list[dict[str, Any]] = []
        self.interrupts = 0
        self.connected = False
        self.disconnects = 0
        cli.clients.append(self)

    async def connect(self) -> None:
        self.connected = True

    async def query(self, prompt: AsyncIterable[dict[str, Any]]) -> None:
        async for record in prompt:
            self.written.append(record)
            # Said at once, before query() returns: the worker must still
            # put the user message first.
            for item in self.cli.on_query(record):
                self.queue.put_nowait(item)

    async def receive_messages(self) -> AsyncIterator[Any]:
        while True:
            item = await self.queue.get()
            if item is None:
                return
            if isinstance(item, BaseException):
                raise item
            yield item

    async def interrupt(self) -> None:
        self.interrupts += 1
        for item in self.cli.on_interrupt:
            self.queue.put_nowait(item)

    async def disconnect(self) -> None:
        self.disconnects += 1
        self.queue.put_nowait(None)


# -- records, as the CLI sends them -------------------------------------------


def echo(record: dict[str, Any]) -> Any:
    """The `--replay-user-messages` echo of a written message."""
    return sdk.UserMessage(
        content=record["message"]["content"], uuid=record["uuid"]
    )


def stream(msg_id: str, *events: dict[str, Any]) -> list[Any]:
    return [
        sdk.StreamEvent(uuid="e", session_id=SESSION, event=event)
        for event in (
            {"type": "message_start", "message": {"id": msg_id}},
            *events,
        )
    ]


def block(index: int, kind: str, chunk: str) -> list[dict[str, Any]]:
    """Start, delta and stop of one text or thinking block."""
    key = "text" if kind == "text" else "thinking"
    return [
        {
            "type": "content_block_start",
            "index": index,
            "content_block": {"type": kind, key: ""},
        },
        {
            "type": "content_block_delta",
            "index": index,
            "delta": {"type": f"{key}_delta", key: chunk},
        },
        {"type": "content_block_stop", "index": index},
    ]


def tool_use(
    msg_id: str, index: int, call_id: str, name: str, args: dict[str, Any]
) -> list[Any]:
    """A tool call: its stream start, the whole message, its stream stop."""
    return [
        sdk.StreamEvent(
            uuid="e",
            session_id=SESSION,
            event={
                "type": "content_block_start",
                "index": index,
                "content_block": {
                    "type": "tool_use",
                    "id": call_id,
                    "name": name,
                },
            },
        ),
        sdk.AssistantMessage(
            content=[sdk.ToolUseBlock(id=call_id, name=name, input=args)],
            model="claude",
            message_id=msg_id,
        ),
        sdk.StreamEvent(
            uuid="e",
            session_id=SESSION,
            event={"type": "content_block_stop", "index": index},
        ),
    ]


def tool_result(call_id: str, content: Any, *, is_error: bool = False) -> Any:
    return sdk.UserMessage(
        content=[
            sdk.ToolResultBlock(
                tool_use_id=call_id, content=content, is_error=is_error
            )
        ],
        uuid=f"r-{call_id}",
    )


def result(**fields: Any) -> Any:
    values: dict[str, Any] = {
        "subtype": "success",
        "duration_ms": 1,
        "duration_api_ms": 1,
        "is_error": False,
        "num_turns": 1,
        "session_id": SESSION,
        "usage": {
            "input_tokens": 3,
            "cache_read_input_tokens": 7,
            "cache_creation_input_tokens": 5,
            "output_tokens": 4,
            "output_tokens_details": {"thinking_tokens": 2},
        },
        "total_cost_usd": 0.01,
        **fields,
    }
    return sdk.ResultMessage(**values)


# -- fixtures -----------------------------------------------------------------


@pytest.fixture
def cwd(tmp_path: Path) -> str:
    path = tmp_path / "project"
    path.mkdir()
    return os.path.realpath(path)


@pytest.fixture
def cli(monkeypatch: pytest.MonkeyPatch) -> FakeCLI:
    cli = FakeCLI()

    async def probe(kind: str, executable: str, **kwargs: Any) -> str:
        return "2.1.319 (Claude Code)"

    def session_info(session_id: str, directory: str | None = None) -> Any:
        return object() if session_id in cli.sessions else None

    monkeypatch.setattr(_process, "probe", probe)
    monkeypatch.setattr(
        sdk, "ClaudeSDKClient", lambda options: FakeClient(cli, options)
    )
    monkeypatch.setattr(sdk, "get_session_info", session_info)
    return cli


async def run(
    worker: ClaudeCode, messages: list[m.Message]
) -> tuple[ai.Stream[str], ai.events.StreamEnd]:
    ends: list[ai.events.StreamEnd] = []
    async with asyncio.timeout(5):
        async with ai.stream(worker, messages) as s:
            async for event in s:
                if isinstance(event, ai.events.StreamEnd):
                    ends.append(event)
    (end,) = ends
    return s, end


# -- tests --------------------------------------------------------------------


async def test_a_turn_streams_in_the_model_view(cli: FakeCLI, cwd: str) -> None:
    cli.on_query = lambda record: [
        echo(record),
        *stream("m1", *block(0, "thinking", "plan")),
        *tool_use("m1", 1, "toolu_1", "Read", {"file_path": "secret.txt"}),
        tool_result("toolu_1", "the word is HERON"),
        *stream("m2", *block(0, "text", "HERON")),
        sdk.AssistantMessage(
            content=[sdk.TextBlock(text="HERON")],
            model="claude",
            message_id="m2",
        ),
        result(),
    ]
    user = ai.user_message("go")
    worker = ClaudeCode(cwd=cwd)
    async with worker.session():
        s, end = await run(worker, [user])
    message = s.message
    assert message.reasoning == "plan"
    assert message.text == "HERON"  # the whole message does not repeat it
    (call,) = message.builtin_tool_calls
    assert call.tool_name == "Read"
    assert json.loads(call.tool_args) == {"file_path": "secret.txt"}
    (returned,) = message.builtin_tool_returns
    assert returned.result == "the word is HERON"
    assert message.usage is not None
    assert message.usage.input_tokens == 10
    assert message.usage.output_tokens == 4
    assert message.usage.reasoning_tokens == 2
    assert message.usage.cache_write_tokens == 5
    assert end.finish_reason == "stop"
    # The echo came at once, but acceptance still comes first.
    assert [msg.role for msg in worker.messages] == ["user", "assistant"]
    assert worker.messages[0].id == user.id
    # The uuid the echo matched is the one written.
    (written,) = cli.client.written
    assert written["message"]["content"] == [{"type": "text", "text": "go"}]
    assert cli.client.disconnects == 1


async def test_start_connects_with_the_bridge_and_no_approvals(
    cli: FakeCLI, cwd: str
) -> None:
    worker = ClaudeCode(
        cwd=cwd,
        provider=workers.ClaudeCodeProvider(
            executable="/custom/claude", env={"ANTHROPIC_API_KEY": "k"}
        ),
        model_name="haiku",
        instructions="Be brief.",
        allowed_tools=["Read"],
        disallowed_tools=["Bash"],
        max_turns=3,
    )
    async with worker.session() as session:
        await session.start()
        options = cli.client.options
        assert session._run is not None
        assert options.mcp_servers == {
            "ai_sdk": {"type": "http", "url": session._run.bridge.url}
        }
        assert options.permission_mode == "bypassPermissions"
        assert options.can_use_tool is None
        assert options.allowed_tools == ["mcp__ai_sdk"]
        assert options.extra_args == {"replay-user-messages": None}
        assert options.include_partial_messages
        assert options.system_prompt == {
            "type": "preset",
            "preset": "claude_code",
            "append": "Be brief.",
        }
        assert options.cwd == cwd
        assert str(options.cli_path) == "/custom/claude"
        assert options.env == {
            "MCP_TOOL_TIMEOUT": "86400000",
            "ANTHROPIC_API_KEY": "k",
        }
        assert options.model == "haiku"
        assert options.tools == ["Read"]
        assert options.disallowed_tools == ["Bash"]
        assert options.max_turns == 3
        # A new session: the id we chose is the CLI's.
        assert options.resume is None
        assert options.session_id == worker.session_id


async def test_a_resumed_session_must_exist(cli: FakeCLI, cwd: str) -> None:
    worker = ClaudeCode(cwd=cwd, session_id=SESSION)
    async with worker.session() as session:
        await session.start()
        assert cli.client.options.resume == SESSION
        assert cli.client.options.session_id is None
        assert worker.session_id == SESSION
    with pytest.raises(workers.WorkerError, match="no session"):
        async with ClaudeCode(cwd=cwd, session_id="ghost").session() as session:
            await session.start()


async def test_a_new_session_is_seeded_from_the_past(
    cli: FakeCLI, cwd: str, tmp_path: Path
) -> None:
    config = tmp_path / "config"
    cli.on_query = lambda record: [
        echo(record),
        *stream("m1", *block(0, "text", "ok")),
        result(),
    ]
    history = [
        ai.user_message("one"),
        m.Message(
            role="assistant",
            parts=[
                m.ReasoningPart(text="secret thoughts"),
                m.TextPart(text="done"),
                m.BuiltinToolCallPart(
                    tool_call_id="t1", tool_name="Bash", tool_args='{"a": 1}'
                ),
                m.BuiltinToolReturnPart(
                    tool_call_id="t1", tool_name="Bash", result="out"
                ),
            ],
        ),
        ai.user_message("two"),
    ]
    worker = ClaudeCode(
        cwd=cwd,
        provider=workers.ClaudeCodeProvider(
            env={"CLAUDE_CONFIG_DIR": str(config)}
        ),
    )
    with pytest.warns(UserWarning, match="1 reasoning part"):
        async with worker.session():
            await run(worker, history)
    options = cli.client.options
    assert options.resume == worker.session_id
    key = sdk.project_key_for_directory(cwd)
    path = config / "projects" / key / f"{worker.session_id}.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert [(r["type"], r["message"]["content"]) for r in records] == [
        ("user", [{"type": "text", "text": "one"}]),
        (
            "assistant",
            [
                {"type": "text", "text": "done"},
                {
                    "type": "tool_use",
                    "id": "t1",
                    "name": "Bash",
                    "input": {"a": 1},
                },
            ],
        ),
        (
            "user",
            [{"type": "tool_result", "tool_use_id": "t1", "content": "out"}],
        ),
    ]
    assert records[1]["parentUuid"] == records[0]["uuid"]
    assert {r["sessionId"] for r in records} == {worker.session_id}
    # Only the new message went to the CLI.
    (written,) = cli.client.written
    assert written["message"]["content"] == [{"type": "text", "text": "two"}]


async def test_bridge_calls_are_ai_tool_calls_not_builtin(
    cli: FakeCLI, cwd: str
) -> None:
    @ai.tool
    async def lookup(word: str) -> str:
        """Look a word up."""
        return word.upper()

    answers: list[dict[str, Any]] = []

    async def call_bridge() -> None:
        # The CLI calls the bridge over MCP, as Claude Code does.
        url = cli.client.options.mcp_servers["ai_sdk"]["url"]
        reply = await fakes.mcp_call(
            url,
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {"name": "lookup", "arguments": {"word": "heron"}},
            },
        )
        answers.append(reply["result"])
        text = reply["result"]["content"][0]["text"]
        cli.say(
            tool_result("toolu_b", text),
            *stream("m2", *block(0, "text", f"got {text}")),
            result(),
        )

    tasks: list[asyncio.Task[None]] = []

    def on_query(record: dict[str, Any]) -> list[Any]:
        tasks.append(asyncio.ensure_future(call_bridge()))
        return [
            echo(record),
            *stream("m1"),
            *tool_use(
                "m1", 0, "toolu_b", "mcp__ai_sdk__lookup", {"word": "heron"}
            ),
        ]

    cli.on_query = on_query
    worker = ClaudeCode(cwd=cwd)
    async with worker.session():
        async with asyncio.timeout(5):
            async with ai.Agent(tools=[lookup]).run(
                worker, [ai.user_message("go")]
            ) as agent_run:
                async for _ in agent_run:
                    pass
        await asyncio.gather(*tasks)
    assert [msg.role for msg in worker.messages] == [
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    (call,) = worker.messages[1].tool_calls
    assert call.tool_name == "lookup"
    assert all(not msg.builtin_tool_calls for msg in worker.messages)
    assert all(not msg.builtin_tool_returns for msg in worker.messages)
    assert worker.messages[3].text == "got HERON"
    assert answers[0]["content"][0]["text"] == "HERON"


async def test_a_send_during_a_turn_steers_it(cli: FakeCLI, cwd: str) -> None:
    first = [True]

    def on_query(record: dict[str, Any]) -> list[Any]:
        if first:
            first.clear()
            return [
                echo(record),
                *stream("m1", *block(0, "text", "Look")),
                *tool_use("m1", 1, "toolu_1", "Bash", {"command": "sleep 4"}),
            ]
        return []  # the CLI holds a steer until the next step

    cli.on_query = on_query
    worker = ClaudeCode(cwd=cwd)
    async with worker.session() as session:
        feed = session.events()
        await session.send(ai.user_message("fix"))
        await session.send(ai.user_message("skip slow ones"))
        steer = cli.client.written[1]
        # The step ends, then the CLI echoes the steer where it injects it.
        cli.say(
            tool_result("toolu_1", ""),
            echo(steer),
            *stream("m2", *block(0, "text", "OK")),
            result(),
        )
        async with asyncio.timeout(5):
            while len(worker.messages) < 4:
                await anext(feed)
        await feed.aclose()
    assert [(msg.role, msg.text) for msg in worker.messages] == [
        ("user", "fix"),
        ("assistant", "Look"),
        ("user", "skip slow ones"),
        ("assistant", "OK"),
    ]
    assert len(worker.messages[1].builtin_tool_returns) == 1


async def test_interrupt_ends_the_turn_as_interrupted(
    cli: FakeCLI, cwd: str
) -> None:
    cli.on_query = lambda record: [
        echo(record),
        *stream(
            "m1",
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "work"},
            },
        ),
    ]
    worker = ClaudeCode(cwd=cwd)
    async with worker.session() as session:
        feed = session.events()
        await session.send(ai.user_message("long job"))
        async with asyncio.timeout(5):
            async for event in feed:
                if isinstance(event, ai.events.TextDelta):
                    break
            await session.interrupt()
        assert cli.client.interrupts == 1
        assert workers.is_interrupted(worker.messages[-1])
        assert worker.messages[-1].text == "work"
        await feed.aclose()
        # The next turn is a normal one: the interrupt was used up.
        cli.on_query = lambda record: [
            echo(record),
            *stream("m2", *block(0, "text", "OK")),
            result(),
        ]
        s, end = await run(worker, [*worker.messages, ai.user_message("again")])
    assert s.text == "OK"
    assert end.finish_reason == "stop"


async def test_a_failed_turn_without_a_key_is_not_authenticated(
    cli: FakeCLI, cwd: str
) -> None:
    detail = "Invalid API key · Fix external API key"
    cli.on_query = lambda record: [
        echo(record),
        # Measured: not streamed, from model "<synthetic>".
        sdk.AssistantMessage(
            content=[sdk.TextBlock(text=detail)],
            model="<synthetic>",
            error="authentication_failed",
        ),
        result(is_error=True, result=detail),
    ]
    worker = ClaudeCode(cwd=cwd)
    async with worker.session():
        with pytest.raises(workers.NotAuthenticatedError):
            await run(worker, [ai.user_message("go")])
    assert worker.messages[-1].text == detail


async def test_max_turns_ends_with_length(cli: FakeCLI, cwd: str) -> None:
    cli.on_query = lambda record: [
        echo(record),
        result(subtype="error_max_turns", is_error=True),
    ]
    worker = ClaudeCode(cwd=cwd, max_turns=1)
    async with worker.session():
        _, end = await run(worker, [ai.user_message("go")])
    assert end.finish_reason == "length"


async def test_a_crash_reaches_the_stream(cli: FakeCLI, cwd: str) -> None:
    cli.on_query = lambda record: [
        echo(record),
        sdk.ProcessError("boom", exit_code=1),
    ]
    worker = ClaudeCode(cwd=cwd)
    async with worker.session():
        with pytest.raises(workers.WorkerError, match="crashed"):
            await run(worker, [ai.user_message("go")])


async def test_unsupported_parts_are_refused_before_writing(
    cli: FakeCLI, cwd: str
) -> None:
    worker = ClaudeCode(cwd=cwd)
    async with worker.session() as session:
        with pytest.raises(workers.UnsupportedError):
            await session.send(
                ai.user_message(
                    m.FilePart(data=b"x", media_type="audio/wav"),
                )
            )
        assert cli.client.written == []


async def test_images_go_as_base64_blocks(cli: FakeCLI, cwd: str) -> None:
    cli.on_query = lambda record: [echo(record), result()]
    worker = ClaudeCode(cwd=cwd)
    async with worker.session():
        await run(
            worker,
            [
                ai.user_message(
                    "what is this",
                    m.FilePart(data=b"\x89PNG", media_type="image/png"),
                )
            ],
        )
    (written,) = cli.client.written
    assert written["message"]["content"][1] == {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/png",
            "data": "iVBORw==",
        },
    }


async def test_close_is_idempotent(cli: FakeCLI, cwd: str) -> None:
    session = ClaudeCode(cwd=cwd).session()
    async with session:
        await session.start()
        assert session._run is not None
        wire = session._run.wire
    await session.close()
    await wire.close()  # the wire itself too
    assert cli.client.disconnects == 1


async def test_the_state_round_trips(cli: FakeCLI, cwd: str) -> None:
    worker = ClaudeCode(cwd=cwd, session_id=SESSION, max_turns=2)
    again = workers.Worker.model_validate(worker.model_dump())
    assert isinstance(again, ClaudeCode)
    assert again.max_turns == 2
    assert again.worker_class_id == "claude-code"
