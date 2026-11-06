"""`ai.workers.Worker` over a scripted adapter: no real CLI runs here."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterator
from typing import Any

import pydantic
import pytest

import ai
from ai import workers
from ai.types import events
from ai.types import messages as m
from ai.workers import _bridge

from . import fakes


@pytest.fixture
def worker() -> fakes.FakeWorker:
    return fakes.FakeWorker()


async def post(url: str, payload: Any) -> Any:
    """POST JSON the way an MCP client does, over a raw socket."""
    host, rest = url.removeprefix("http://").split(":", 1)
    port, path = rest.split("/", 1)
    reader, writer = await asyncio.open_connection(host, int(port))
    body = json.dumps(payload).encode()
    writer.write(
        (
            f"POST /{path} HTTP/1.1\r\nHost: {host}\r\n"
            f"Content-Type: application/json\r\n"
            f"Content-Length: {len(body)}\r\n\r\n"
        ).encode()
        + body
    )
    await writer.drain()
    head = await reader.readuntil(b"\r\n\r\n")
    length = 0
    for line in head.decode().split("\r\n"):
        if line.lower().startswith("content-length:"):
            length = int(line.split(":")[1])
    data = await reader.readexactly(length) if length else b""
    writer.close()
    return json.loads(data) if data else None


async def collect(stream: Any) -> list[Any]:
    return [event async for event in stream]


# -- the model protocol -------------------------------------------------------


async def test_a_worker_is_model_like(worker: fakes.FakeWorker) -> None:
    assert isinstance(worker, ai.ModelLike)
    assert worker.provider_name == "fake"
    assert worker.id == "default"
    async with worker:
        assert worker.version == "fake 1.0"
        async with ai.stream(worker, [ai.user_message("hello")]) as s:
            seen = await collect(s)
    assert isinstance(seen[0], events.StreamStart)
    assert isinstance(seen[-1], events.StreamEnd)
    assert seen[-1].finish_reason == "stop"
    assert s.text == "hello"
    assert s.message.usage is not None and s.message.usage.input_tokens == 3
    meta = seen[-1].provider_metadata
    assert meta is not None and meta["worker"]["session_id"] == "s1"


async def test_generate_drains_the_stream(worker: fakes.FakeWorker) -> None:
    async with worker:
        message = await ai.experimental_generate(
            worker, [ai.user_message("hi")]
        )
    assert message.text == "hi"


async def test_cli_tools_are_builtin(worker: fakes.FakeWorker) -> None:
    worker.script(
        lambda prompt: [
            *fakes.tool("c1", "Bash", '{"command": "ls"}', "a.py"),
            *fakes.text("done"),
        ]
    )
    async with worker:
        async with ai.stream(worker, [ai.user_message("go")]) as s:
            kinds = [type(e).__name__ for e in await collect(s)]
    assert "BuiltinToolStart" in kinds
    assert "BuiltinToolResult" in kinds
    assert "ToolStart" not in kinds
    assert s.tool_calls == []  # nothing for an ai.Agent to run
    call, result = s.message.parts[0], s.message.parts[1]
    assert isinstance(call, m.BuiltinToolCallPart)
    assert call.tool_name == "Bash"
    assert call.tool_args == '{"command": "ls"}'
    assert isinstance(result, m.BuiltinToolReturnPart)
    assert result.result == "a.py"


async def test_a_cancelled_turn_ends_as_other(worker: fakes.FakeWorker) -> None:
    worker.script(lambda prompt: [*fakes.text("a"), fakes.Pause()])
    async with worker:
        stream = worker.stream([ai.user_message("go")])
        seen = []
        async for event in stream:
            seen.append(event)
            if isinstance(event, events.TextEnd):
                await worker.fake.interrupt()
    end = seen[-1]
    assert isinstance(end, events.StreamEnd)
    assert end.finish_reason == "other"
    assert end.provider_metadata is not None
    assert end.provider_metadata["worker"]["finish_reason"] == "cancelled"


# -- the conversation ---------------------------------------------------------


async def test_only_new_turns_reach_the_cli(worker: fakes.FakeWorker) -> None:
    async with worker:
        first = await ai.experimental_generate(worker, [ai.user_message("one")])
        assert worker.messages[-1].id == first.id
        await ai.experimental_generate(
            worker, [*worker.messages, ai.user_message("two")]
        )
        assert worker.fake.turns == ["one", "two"]
        assert len(worker.fake.opens) == 1
    assert [x.role for x in worker.messages] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]


async def test_a_new_worker_is_seeded_with_the_past(
    worker: fakes.FakeWorker,
) -> None:
    past = [
        ai.system_message("Be terse."),
        ai.user_message("earlier"),
        m.Message(role="assistant", parts=[m.TextPart(text="answer")]),
    ]
    async with worker:
        await ai.experimental_generate(worker, [*past, ai.user_message("now")])
        opened = worker.fake.opens[0]
        assert worker.fake.turns == ["now"]
    # The system message is the CLI's instructions, not part of the seed.
    assert opened["instructions"] == "Be terse."
    assert [x.id for x in opened["seed"]] == [x.id for x in past[1:]]
    assert opened["session_id"] is None


async def test_a_later_system_message_joins_the_prompt(
    worker: fakes.FakeWorker,
) -> None:
    async with worker:
        await ai.experimental_generate(worker, [ai.user_message("one")])
        await ai.experimental_generate(
            worker,
            [
                *worker.messages,
                ai.system_message("Now in French."),
                ai.user_message("two"),
            ],
        )
        assert worker.fake.turns[1] == "Now in French.\n\n\n\ntwo"


async def test_a_rewritten_past_is_refused(worker: fakes.FakeWorker) -> None:
    async with worker:
        await ai.experimental_generate(worker, [ai.user_message("one")])
        with pytest.raises(workers.TranscriptMismatchError):
            await ai.experimental_generate(worker, [ai.user_message("other")])


async def test_a_failed_call_can_be_sent_again(
    worker: fakes.FakeWorker,
) -> None:
    def boom(prompt: str) -> list[fakes.Step]:
        raise RuntimeError("boom")

    worker.script(boom)
    messages = [ai.user_message("one")]
    async with worker:
        with pytest.raises(RuntimeError, match="boom"):
            await ai.experimental_generate(worker, messages)
        assert worker.messages == []
        worker.script(fakes.reply)
        again = await ai.experimental_generate(worker, messages)
    assert again.text == "one"


async def test_walking_away_interrupts_the_turn(
    worker: fakes.FakeWorker,
) -> None:
    worker.script(lambda prompt: [*fakes.text("a"), fakes.Pause()])
    async with worker:
        async with ai.stream(worker, [ai.user_message("go")]) as s:
            async for event in s:
                if isinstance(event, events.TextDelta):
                    break
        assert worker.fake.interrupts == 1


async def test_structured_output(worker: fakes.FakeWorker) -> None:
    class Finding(pydantic.BaseModel):
        file: str
        line: int

    worker.script(
        lambda prompt: [
            *fakes.text("Let me look."),
            *fakes.tool("c1", "Read"),
            *fakes.text('```json\n{"file": "a.py", "line": 3}\n```', block="x"),
        ]
    )
    async with worker:
        async with ai.stream(
            worker, [ai.user_message("scan")], output_type=Finding
        ) as s:
            await collect(s)
        assert "JSON value matching this schema" in worker.fake.turns[0]
    assert s.output == Finding(file="a.py", line=3)


async def test_structured_output_asks_again(worker: fakes.FakeWorker) -> None:
    class Count(pydantic.BaseModel):
        n: int

    worker.script(
        lambda prompt: fakes.text(
            '{"n": 2}' if "failed validation" in prompt else "two"
        )
    )
    async with worker:
        async with ai.stream(
            worker, [ai.user_message("how many?")], output_type=Count
        ) as s:
            await collect(s)
        assert len(worker.fake.turns) == 2
    assert s.output == Count(n=2)


# -- a worker is data ---------------------------------------------------------


async def test_a_dumped_worker_continues_its_conversation(
    worker: fakes.FakeWorker,
) -> None:
    async with worker:
        await ai.experimental_generate(worker, [ai.user_message("one")])
    data = json.loads(worker.model_dump_json())
    assert data["session_id"] == "s1"
    again = workers.Worker.model_validate(data)
    assert isinstance(again, fakes.FakeWorker)
    assert [x.id for x in again.messages] == [x.id for x in worker.messages]
    async with again:
        await ai.experimental_generate(
            again, [*again.messages, ai.user_message("two")]
        )
        assert again.fake.opens[0]["session_id"] == "s1"  # resumed
        assert again.fake.turns == ["two"]


async def test_an_unknown_kind_is_refused() -> None:
    with pytest.raises(pydantic.ValidationError):
        workers.Worker.model_validate({"kind": "nope"})


async def test_resume_by_session_id() -> None:
    worker = fakes.FakeWorker(session_id="old")
    async with worker:
        # The CLI holds the past: what is given is all new.
        await ai.experimental_generate(worker, [ai.user_message("go on")])
        assert worker.fake.opens[0]["session_id"] == "old"
        assert worker.fake.opens[0]["seed"] is None
        assert worker.fake.turns == ["go on"]
    assert worker.session_id == "old"


async def test_load_reads_the_stored_past() -> None:
    worker = fakes.FakeWorker(session_id="old")
    stored = [
        ai.user_message("earlier"),
        m.Message(role="assistant", parts=[m.TextPart(text="answer")]),
    ]
    async with worker:
        worker.fake.stored["old"] = stored
        await worker.load()
        assert worker.messages == stored
        await ai.experimental_generate(
            worker, [*worker.messages, ai.user_message("now")]
        )
        assert worker.fake.turns == ["now"]


async def test_a_copy_with_fork_branches_natively(
    worker: fakes.FakeWorker,
) -> None:
    async with worker:
        await ai.experimental_generate(worker, [ai.user_message("one")])
        branch = worker.model_copy(update={"fork": True})
        async with branch:
            assert branch.fake is not worker.fake  # a CLI of its own
            await ai.experimental_generate(
                branch, [*branch.messages, ai.user_message("two")]
            )
            assert branch.fake.opens[0] == {
                "session_id": "s1",
                "fork": True,
                "seed": None,
                "instructions": None,
            }
    assert branch.session_id == "s1-fork"
    assert branch.fork is False
    assert worker.session_id == "s1"
    assert len(worker.messages) == 2
    assert len(branch.messages) == 4


async def test_a_worker_goes_in_a_context(worker: fakes.FakeWorker) -> None:
    context = ai.Context(model=worker, messages=[], tools=[])
    data = json.loads(context.model_dump_json())
    assert data["model"]["kind"] == "fake"
    restored = ai.Context.model_validate(data)
    assert isinstance(restored.model, fakes.FakeWorker)


async def test_close_keeps_the_fields(worker: fakes.FakeWorker) -> None:
    async with worker:
        await ai.experimental_generate(worker, [ai.user_message("one")])
        fake = worker.fake
    assert fake.closed
    assert worker.session_id == "s1"
    async with worker:  # opens again, and resumes
        await ai.experimental_generate(
            worker, [*worker.messages, ai.user_message("two")]
        )
        assert worker.fake.opens[0]["session_id"] == "s1"


# -- ai.Agent tools, through the bridge ---------------------------------------


async def test_agent_tools_run_in_the_agent(worker: fakes.FakeWorker) -> None:
    ran: list[tuple[int, int]] = []

    @ai.tool
    async def add(a: int, b: int) -> int:
        """Add two numbers."""
        ran.append((a, b))
        return a + b

    answers: list[Any] = []

    def script(prompt: str) -> Iterator[fakes.Step]:
        url = worker.servers[_bridge.SERVER_NAME]["url"]

        async def call() -> None:
            listed = await post(
                url, {"jsonrpc": "2.0", "id": 1, "method": "tools/list"}
            )
            assert [t["name"] for t in listed["result"]["tools"]] == ["add"]
            answers.append(
                await post(
                    url,
                    {
                        "jsonrpc": "2.0",
                        "id": 2,
                        "method": "tools/call",
                        "params": {
                            "name": "add",
                            "arguments": {"a": 1, "b": 2},
                        },
                    },
                )
            )

        # What the CLI reports about its own MCP call: hidden from the
        # stream, because the agent reports the call itself.
        yield events.BuiltinToolStart(
            tool_call_id="x1", tool_name="mcp__ai_sdk__add"
        )
        yield fakes.Run(call)
        yield from fakes.tool("x1", "mcp__ai_sdk__add")[1:]
        text = answers[0]["result"]["content"][0]["text"]
        yield from fakes.text(f"the sum is {text}")

    worker.script(script)
    agent = ai.Agent(tools=[add])
    async with worker:
        async with agent.run(worker, [ai.user_message("add 1 and 2")]) as run:
            seen = await collect(run)
    assert ran == [(1, 2)]
    assert run.messages[-1].text == "the sum is 3"
    names = [
        e.tool_name for e in seen if isinstance(e, events.BuiltinToolStart)
    ]
    assert names == []  # the CLI's own report of the MCP call is hidden
    assert [x.role for x in run.messages] == [
        "user",
        "assistant",
        "tool",
        "assistant",
    ]


async def test_bridge_tool_names(worker: fakes.FakeWorker) -> None:
    @ai.tool
    async def lookup(q: str) -> str:
        """Look up."""
        return q

    async with worker:
        assert worker._runtime is not None
        worker._runtime.bridge.set_tools([lookup.tool])
        assert worker._is_bridge_tool("lookup")  # codex, with a tool name
        assert worker._is_bridge_tool("ai_sdk")  # codex, without one
        assert worker._is_bridge_tool("mcp__ai_sdk__lookup")  # claude code
        assert not worker._is_bridge_tool("Bash")


# -- approvals, through ai.hook -----------------------------------------------


@pytest.mark.parametrize("granted", [True, False])
async def test_approvals_are_tool_approval_hooks(granted: bool) -> None:
    worker = fakes.FakeWorker(approve=True)
    worker.script(
        lambda prompt: [
            fakes.Ask("c1", "Bash", {"command": "rm -rf build"}),
            *fakes.text("done"),
        ]
    )
    pending: list[m.HookPart[Any]] = []
    async with worker:
        async with ai.Agent().run(worker, [ai.user_message("clean")]) as run:
            async for event in run:
                if isinstance(event, events.PendingHookEvent):
                    pending.append(event.hook)
                    ai.resolve_hook(
                        event.hook,
                        {"granted": granted, "reason": "not today"},
                    )
        answers = worker.fake.answers
    assert [h.hook_type for h in pending] == ["ToolApproval"]
    assert pending[0].tool_call_id == "c1"
    assert pending[0].metadata == {
        "tool": "Bash",
        "kwargs": {"command": "rm -rf build"},
    }
    assert [granted for granted, _ in answers] == [granted]
    if not granted:
        assert answers[0][1] == "not today"  # the reason the agent reads


async def test_approve_needs_a_run() -> None:
    worker = fakes.FakeWorker(approve=True)
    async with worker:
        with pytest.raises(workers.WorkerError, match=r"ai\.Agent\.run"):
            await ai.experimental_generate(worker, [ai.user_message("go")])


# -- steer, the TUI, the store ------------------------------------------------


async def test_steer_reaches_the_running_turn(
    worker: fakes.FakeWorker,
) -> None:
    worker.script(lambda prompt: [*fakes.text("working"), fakes.Pause()])
    async with worker:
        with pytest.raises(workers.WorkerError):
            await worker.steer("too early")
        async with ai.stream(worker, [ai.user_message("go")]) as s:
            async for event in s:
                if isinstance(event, events.TextEnd):
                    await worker.steer("use tabs")
                    await worker.fake.interrupt()
        assert worker.fake.steers == ["use tabs"]


async def test_tui_command_mints_a_session(worker: fakes.FakeWorker) -> None:
    argv, env = worker.tui_command()
    assert env == {"FAKE_TUI": "1"}
    assert argv[0]
    assert worker.session_id == "minted"


async def test_tui_hands_over_and_reads_back() -> None:
    worker = fakes.FakeWorker(session_id="s1")
    stored = [
        ai.user_message("typed by a person"),
        m.Message(role="assistant", parts=[m.TextPart(text="ok")]),
    ]
    async with worker:
        assert await worker.tui() == 0
        # tui() closed the worker's CLI, ran the TUI, and opened again.
        worker.fake.stored["s1"] = stored
        await worker.load()
    assert worker.messages == stored


async def test_list_sessions() -> None:
    sessions = await fakes.FakeWorker.list_sessions(cwd="/tmp")
    assert [s.session_id for s in sessions] == ["s1"]


# -- the bridge alone ---------------------------------------------------------


async def test_bridge_speaks_mcp() -> None:
    async def on_call(name: str, arguments: dict[str, Any]) -> tuple[str, bool]:
        return (f"{name}:{arguments['q']}", False)

    @ai.tool
    async def search(q: str) -> str:
        """Search."""
        return q

    bridge = _bridge.Bridge(on_call)
    await bridge.start()
    try:
        bridge.set_tools([search.tool])
        init = await post(
            bridge.url,
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {"protocolVersion": "2025-03-26"},
            },
        )
        assert init["result"]["protocolVersion"] == "2025-03-26"
        assert (
            await post(
                bridge.url,
                {"jsonrpc": "2.0", "method": "notifications/initialized"},
            )
            is None
        )
        listed = await post(
            bridge.url, {"jsonrpc": "2.0", "id": 2, "method": "tools/list"}
        )
        tool = listed["result"]["tools"][0]
        assert tool["name"] == "search"
        assert tool["inputSchema"]["properties"]["q"]["type"] == "string"
        called = await post(
            bridge.url,
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {"name": "search", "arguments": {"q": "x"}},
            },
        )
        assert called["result"]["content"][0]["text"] == "search:x"
        unknown = await post(
            bridge.url, {"jsonrpc": "2.0", "id": 4, "method": "nope"}
        )
        assert unknown["error"]["code"] == -32601
    finally:
        await bridge.close()
