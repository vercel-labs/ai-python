"""`ai.workers.Worker` over a scripted adapter: no real CLI runs here."""

from __future__ import annotations

import asyncio
import contextlib
import json
from collections.abc import AsyncIterator, Iterator
from typing import Any

import pydantic
import pytest

import ai
from ai import workers
from ai.types import events
from ai.types import messages as m
from ai.workers import _adapter, _bridge

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
                await worker.interrupt()
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


async def test_failure_after_acceptance_keeps_the_accepted_user(
    worker: fakes.FakeWorker,
) -> None:
    def boom(prompt: str) -> list[fakes.Step]:
        raise RuntimeError("boom")

    worker.script(boom)
    message = ai.user_message("one")
    async with worker:
        with pytest.raises(RuntimeError, match="boom"):
            await ai.experimental_generate(worker, [message])
        assert worker.messages[0] == message
        assert [x.role for x in worker.messages] == ["user", "assistant"]
        assert worker.messages[1].parts == []
        worker.script(fakes.reply)
        with pytest.raises(ValueError, match="no new user message"):
            await ai.experimental_generate(worker, list(worker.messages))
        again = await ai.experimental_generate(
            worker, [*worker.messages, ai.user_message("two")]
        )
        assert worker.fake.turns == ["one", "two"]
    assert again.text == "two"
    assert sum(x.id == message.id for x in worker.messages) == 1


async def test_failure_before_acceptance_can_be_retried(
    worker: fakes.FakeWorker, monkeypatch: pytest.MonkeyPatch
) -> None:
    turn = fakes.FakeAdapter.turn
    reject = True

    async def sometimes_reject(
        adapter: fakes.FakeAdapter, prompt: _adapter.Prompt
    ) -> AsyncIterator[events.Event]:
        if reject:
            raise RuntimeError("not accepted")
        async for event in turn(adapter, prompt):
            yield event

    monkeypatch.setattr(fakes.FakeAdapter, "turn", sometimes_reject)
    message = ai.user_message("one")
    async with worker:
        with pytest.raises(RuntimeError, match="not accepted"):
            await ai.experimental_generate(worker, [message])
        assert worker.messages == []
        assert worker._runtime is not None
        assert worker._runtime.stream_owner is None
        assert worker._runtime.submission is None
        reject = False
        again = await ai.experimental_generate(worker, [message])
        assert worker.fake.turns == ["one"]
    assert again.text == "one"
    assert worker.messages[0] == message


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


async def test_an_unknown_worker_class_is_refused() -> None:
    with pytest.raises(pydantic.ValidationError, match="unknown worker class"):
        workers.Worker.model_validate({"worker_class_id": "nope"})


def test_worker_identity_and_removed_fields() -> None:
    worker = fakes.FakeWorker(model_name="test-model")
    assert worker.id == "test-model"
    assert worker.worker_class_id == worker.provider_name == "fake"
    assert worker.version is None
    data = worker.model_dump()
    assert data["model_name"] == "test-model"
    assert data["worker_class_id"] == "fake"
    assert "fork" not in data
    assert callable(worker.fork)
    for field, value in (("kind", "fake"), ("model", "old"), ("fork", True)):
        with pytest.raises(pydantic.ValidationError, match="Extra inputs"):
            fakes.FakeWorker.model_validate({field: value})


async def test_resume_by_session_id() -> None:
    worker = fakes.FakeWorker(session_id="old")
    async with worker:
        # The CLI holds the past: what is given is all new.
        await ai.experimental_generate(worker, [ai.user_message("go on")])
        assert worker.fake.opens[0]["session_id"] == "old"
        assert worker.fake.opens[0]["seed"] is None
        assert worker.fake.turns == ["go on"]
    assert worker.session_id == "old"


async def test_sync_reads_the_stored_past() -> None:
    worker = fakes.FakeWorker(session_id="old")
    stored = [
        ai.user_message("earlier"),
        m.Message(role="assistant", parts=[m.TextPart(text="answer")]),
    ]
    async with worker:
        worker.fake.stored["old"] = stored
        await worker.sync()
        assert worker.messages == stored
        assert worker.messages is not stored
        assert worker.messages[0] is not stored[0]
        await ai.experimental_generate(
            worker, [*worker.messages, ai.user_message("now")]
        )
        assert worker.fake.turns == ["now"]


async def test_fork_branches_natively(worker: fakes.FakeWorker) -> None:
    async with worker:
        await ai.experimental_generate(worker, [ai.user_message("one")])
        branch = await worker.fork()
        async with branch:
            assert branch.fake is not worker.fake
            assert branch.session_id == "s1-fork"
            assert branch.messages == worker.messages
            assert branch.messages is not worker.messages
            assert branch.messages[0] is not worker.messages[0]
            assert (
                branch.messages[0].parts[0] is not worker.messages[0].parts[0]
            )
            stored = branch.fake.stored["s1-fork"]
            assert branch.messages == stored
            assert branch.messages[0] is not stored[0]
            await ai.experimental_generate(
                branch, [*branch.messages, ai.user_message("two")]
            )
            assert branch.fake.opens == [
                {
                    "session_id": "s1",
                    "fork": True,
                    "seed": None,
                    "instructions": None,
                }
            ]
            assert branch.fake.turns == ["two"]
    assert worker.session_id == "s1"
    assert len(worker.messages) == 2
    assert len(branch.messages) == 4


async def test_a_worker_goes_in_a_context(worker: fakes.FakeWorker) -> None:
    context = ai.Context(model=worker, messages=[], tools=[])
    data = json.loads(context.model_dump_json())
    assert data["model"]["worker_class_id"] == "fake"
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


@pytest.mark.parametrize("mode", ["resume", "copy", "deepcopy"])
async def test_resume_and_copy_reset_all_private_state(
    worker: fakes.FakeWorker, mode: str
) -> None:
    async with worker:
        await ai.experimental_generate(worker, [ai.user_message("one")])
        worker.script(lambda prompt: fakes.text("private script"))
        if mode == "resume":
            restored = workers.Worker.model_validate_json(
                worker.model_dump_json()
            )
        else:
            restored = worker.model_copy(deep=mode == "deepcopy")
        assert isinstance(restored, fakes.FakeWorker)
        assert restored.model_dump() == worker.model_dump()
        assert restored.messages is not worker.messages
        assert restored.messages[0].parts[0] is not worker.messages[0].parts[0]
        assert restored._runtime is None
        assert restored._fake is None
        assert restored._servers == {}
        assert restored._script is fakes.reply
        assert restored.version is None
        assert restored._open_lock is not worker._open_lock
        async with restored:
            assert restored._runtime is not worker._runtime
            assert restored.fake is not worker.fake
            assert restored.hook_registry is not worker.hook_registry
            answer = await ai.experimental_generate(
                restored, [*restored.messages, ai.user_message("two")]
            )
            assert answer.text == "two"
            assert restored.fake.opens == [
                {
                    "session_id": "s1",
                    "fork": False,
                    "seed": None,
                    "instructions": None,
                }
            ]
        assert not worker.fake.closed
        assert len(worker.messages) == 2


async def test_concurrent_open_starts_one_transport(
    worker: fakes.FakeWorker, monkeypatch: pytest.MonkeyPatch
) -> None:
    entered = asyncio.Event()
    release = asyncio.Event()
    second_entered = asyncio.Event()
    starts: list[fakes.FakeAdapter] = []
    start = fakes.FakeAdapter.start

    async def slow_start(adapter: fakes.FakeAdapter) -> None:
        starts.append(adapter)
        entered.set()
        await release.wait()
        await start(adapter)

    async def second_open() -> None:
        second_entered.set()
        await worker.open()

    monkeypatch.setattr(fakes.FakeAdapter, "start", slow_start)
    try:
        async with asyncio.timeout(5), asyncio.TaskGroup() as tasks:
            first = tasks.create_task(worker.open())
            await entered.wait()
            second = tasks.create_task(second_open())
            await second_entered.wait()
            assert not first.done()
            assert not second.done()
            assert len(starts) == 1
            release.set()
        assert starts == [worker.fake]
        assert worker.version == "fake 1.0"
        runtime = worker._runtime
        await worker.open()
        assert worker._runtime is runtime
        assert len(starts) == 1
        message = await ai.experimental_generate(
            worker, [ai.user_message("one")]
        )
        assert message.text == "one"
        assert len(worker.fake.opens) == 1
    finally:
        release.set()
        await worker.close()
    assert starts[0].closed


async def test_initial_stream_and_send_open_once_and_preserve_input(
    worker: fakes.FakeWorker, monkeypatch: pytest.MonkeyPatch
) -> None:
    opening = asyncio.Event()
    release_open = asyncio.Event()
    sending = asyncio.Event()
    reading = asyncio.Event()
    finish = asyncio.Event()
    native_open = fakes.FakeAdapter.open
    initial = ai.user_message("one")

    async def slow_open(
        adapter: fakes.FakeAdapter,
        *,
        session_id: str | None,
        fork: bool,
        seed: list[m.Message] | None,
        instructions: str | None,
    ) -> str:
        opened = await native_open(
            adapter,
            session_id=session_id,
            fork=fork,
            seed=seed,
            instructions=instructions,
        )
        opening.set()
        await release_open.wait()
        return opened

    async def read() -> list[events.Event]:
        seen = []
        async with contextlib.aclosing(worker.stream([initial])) as feed:
            async for event in feed:
                seen.append(event)
                if isinstance(event, events.StreamStart):
                    reading.set()
        return seen

    async def send_guidance() -> None:
        sending.set()
        await worker.send("use tabs")

    async def wait_for_finish() -> None:
        await finish.wait()

    monkeypatch.setattr(fakes.FakeAdapter, "open", slow_open)
    worker.script(
        lambda prompt: [fakes.Run(wait_for_finish), *fakes.text("done")]
    )
    async with worker, asyncio.timeout(5):
        reader = asyncio.create_task(read())
        sender: asyncio.Task[None] | None = None
        try:
            await opening.wait()
            assert worker._runtime is not None
            assert worker._runtime.stream_owner is reader
            # The stream has reserved the reader, but its native session and
            # initial input are not ready. A public send must wait, then steer.
            sender = asyncio.create_task(send_guidance())
            await sending.wait()
            assert not sender.done()
            assert len(worker.fake.opens) == 1
            assert worker.fake.turns == []
            release_open.set()
            await sender
            await reading.wait()
            finish.set()
            seen = await reader
            assert len(worker.fake.opens) == 1
            assert worker.fake.turns == ["one"]
            assert worker.fake.steers == ["use tabs"]
            users = [
                message for message in worker.messages if message.role == "user"
            ]
            assert [message.text for message in users] == ["one", "use tabs"]
            assert users[0].id == initial.id
            assert users[1].id != initial.id
            assert worker.messages[-1].text == "done"
            assert isinstance(seen[-1], events.StreamEnd)
            assert seen[-1].finish_reason == "stop"
        finally:
            release_open.set()
            finish.set()
            reader.cancel()
            tasks: list[asyncio.Task[Any]] = [reader]
            if sender is not None:
                sender.cancel()
                tasks.append(sender)
            await asyncio.gather(*tasks, return_exceptions=True)


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


@pytest.mark.parametrize("granted", [True, False])
async def test_agent_resolves_approval_from_a_preexisting_task(
    granted: bool,
) -> None:
    worker = fakes.FakeWorker(approve=True)
    requested = asyncio.Event()
    answered = asyncio.Event()

    async def external_callback() -> tuple[bool, str]:
        await requested.wait()
        approve = worker.fake.approve
        assert approve is not None
        answer = await approve("c1", "Bash", {"command": "ls"})
        answered.set()
        return answer

    async def request_approval() -> None:
        requested.set()
        await answered.wait()

    worker.script(
        lambda prompt: [fakes.Run(request_approval), *fakes.text("done")]
    )
    async with worker, asyncio.timeout(5):
        # CLI readers are often created before Agent.run establishes its
        # context. The callback must use the stream's registry, not its own.
        callback = asyncio.create_task(external_callback())
        seen: list[events.HookEvent] = []
        try:
            async with ai.Agent().run(worker, [ai.user_message("go")]) as run:
                async for event in run:
                    if isinstance(event, events.HookEvent):
                        seen.append(event)
                    if isinstance(event, events.PendingHookEvent):
                        assert event.hook.tool_call_id == "c1"
                        assert event.hook.metadata == {
                            "tool": "Bash",
                            "kwargs": {"command": "ls"},
                        }
                        # Deliberately use the Agent context, without registry=.
                        ai.resolve_hook(
                            event.hook,
                            {"granted": granted, "reason": "user decision"},
                        )
                        done, _ = await asyncio.wait({callback}, timeout=1)
                        assert callback in done, (
                            "Agent resolution did not reach the approval "
                            "callback created before Agent.run"
                        )
            assert callback.result() == (granted, "user decision")
            assert [type(event) for event in seen] == [
                events.PendingHookEvent,
                events.ResolvedHookEvent,
            ]
            assert run.messages[-1].text == "done"
            assert worker.hook_registry._live_hooks == {}
            assert worker._runtime is not None
            assert worker._runtime.approvals == set()
        finally:
            callback.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await callback


@pytest.mark.parametrize("granted", [True, False])
async def test_direct_approvals_use_the_worker_registry(granted: bool) -> None:
    worker = fakes.FakeWorker(approve=True)
    worker.script(
        lambda prompt: [
            fakes.Ask("c1", "Bash", {"command": "ls"}),
            *fakes.text("done"),
        ]
    )
    seen: list[workers.SessionEvent] = []
    async with worker, contextlib.aclosing(worker.events()) as feed:
        async with asyncio.timeout(5):
            await worker.send("go")
            async for event in feed:
                seen.append(event)
                if isinstance(event, events.PendingHookEvent):
                    assert worker.fake.answers == []
                    assert event.hook.tool_call_id == "c1"
                    assert event.hook.hook_type == "ToolApproval"
                    assert event.hook.metadata == {
                        "tool": "Bash",
                        "kwargs": {"command": "ls"},
                    }
                    ai.resolve_hook(
                        event.hook,
                        {"granted": granted, "reason": "user decision"},
                        registry=worker.hook_registry,
                    )
                if isinstance(event, events.StreamEnd):
                    break
        assert worker.fake.answers == [(granted, "user decision")]
        assert [
            type(event) for event in seen if isinstance(event, events.HookEvent)
        ] == [events.PendingHookEvent, events.ResolvedHookEvent]
        assert worker.hook_registry._live_hooks == {}
        assert worker._runtime is not None
        assert worker._runtime.approvals == set()
        assert worker.messages[-1].text == "done"


@pytest.mark.parametrize("action", ["interrupt", "cancel"])
async def test_stopping_a_pending_approval_cleans_up(action: str) -> None:
    worker = fakes.FakeWorker(approve=True)
    worker.script(lambda prompt: [fakes.Ask("c1", "Bash"), fakes.Pause()])
    pending = asyncio.Event()
    seen: list[workers.SessionEvent] = []

    async def read() -> None:
        async with contextlib.aclosing(worker.events()) as feed:
            async for event in feed:
                seen.append(event)
                if isinstance(event, events.PendingHookEvent):
                    pending.set()
                if isinstance(event, events.StreamEnd):
                    return

    async with worker, asyncio.timeout(5):
        await worker.send("one")
        reader = asyncio.create_task(read())
        try:
            await pending.wait()
            registry = worker.hook_registry
            assert len(registry._live_hooks) == 1
            if action == "interrupt":
                await worker.interrupt()
                await reader
                assert isinstance(seen[-1], events.StreamEnd)
                assert seen[-1].finish_reason == "other"
            else:
                reader.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await reader
        finally:
            if not reader.done():
                reader.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await reader
        assert worker.fake.answers == [(False, "the approval was cancelled")]
        assert registry._live_hooks == {}
        assert not worker.fake.running
        assert worker._runtime is not None
        assert not worker._runtime.reader
        assert worker._runtime.pump is None
        assert worker._runtime.approvals == set()
        worker.script(fakes.reply)
        answer = await ai.experimental_generate(
            worker, [*worker.messages, ai.user_message("two")]
        )
        assert answer.text == "two"
        assert worker.messages[-2].text == "two"
        assert worker.fake.turns == ["one", "two"]


async def test_approve_without_a_hook_does_not_require_an_agent() -> None:
    async with fakes.FakeWorker(approve=True) as worker:
        message = await ai.experimental_generate(
            worker, [ai.user_message("go")]
        )
    assert message.text == "go"


# -- direct control, the TUI, the store ---------------------------------------


async def test_send_reaches_the_running_turn(worker: fakes.FakeWorker) -> None:
    worker.script(lambda prompt: [*fakes.text("working"), fakes.Pause()])
    async with worker, contextlib.aclosing(worker.events()) as feed:
        async with asyncio.timeout(5):
            await worker.send("go")
            accepted = []
            async for event in feed:
                if isinstance(event, workers.UserMessageAccepted):
                    accepted.append(event.message.text)
                    if event.message.text == "use tabs":
                        await worker.interrupt()
                if isinstance(event, events.TextEnd):
                    await worker.send("use tabs")
                    assert [
                        x.text for x in worker.messages if x.role == "user"
                    ] == ["go"]
                if isinstance(event, events.StreamEnd):
                    assert event.finish_reason == "other"
                    break
        assert accepted == ["go", "use tabs"]
        assert worker.fake.turns == ["go"]
        assert worker.fake.steers == ["use tabs"]


async def test_one_persistent_feed_mirrors_multiple_sends(
    worker: fakes.FakeWorker,
) -> None:
    async with worker, contextlib.aclosing(worker.events()) as feed:
        async with asyncio.timeout(5):
            for index, prompt in enumerate(("one", "two")):
                await worker.send(prompt)
                # Acceptance is queued, but send() must not mutate the mirror.
                assert len(worker.messages) == index * 2
                accepted = await anext(feed)
                assert isinstance(accepted, workers.UserMessageAccepted)
                assert accepted.message.text == prompt
                assert worker.messages[-1] is accepted.message
                assert len(worker.messages) == index * 2 + 1
                seen = []
                async for event in feed:
                    seen.append(event)
                    if isinstance(event, events.StreamEnd):
                        break
                assert [type(event) for event in seen] == [
                    events.StreamStart,
                    events.TextStart,
                    events.TextDelta,
                    events.TextEnd,
                    events.StreamEnd,
                ]
                assert len(worker.messages) == (index + 1) * 2
                assert worker.messages[-1].text == prompt
                assert all(
                    event.message is worker.messages[-1] for event in seen
                )
            assert worker.fake.turns == ["one", "two"]
            assert len(worker.fake.opens) == 1
            await worker.close()
            with pytest.raises(StopAsyncIteration):
                await anext(feed)
    assert [message.role for message in worker.messages] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    assert len({message.id for message in worker.messages}) == 4


@pytest.mark.parametrize("owner", ["direct", "stream"])
@pytest.mark.parametrize("challenger", ["direct", "stream"])
async def test_one_reader_guard_across_direct_and_model_apis(
    worker: fakes.FakeWorker, owner: str, challenger: str
) -> None:
    worker.script(lambda prompt: [*fakes.text("working"), fakes.Pause()])
    async with worker:
        if owner == "direct":
            await worker.send("one")
            feed = worker.events()
        else:
            feed = worker.stream([ai.user_message("one")])
        async with contextlib.aclosing(feed):
            await anext(feed)
            other = (
                worker.events()
                if challenger == "direct"
                else worker.stream([*worker.messages, ai.user_message("two")])
            )
            async with contextlib.aclosing(other):
                with pytest.raises(
                    workers.WorkerError, match="already has a reader"
                ):
                    await anext(other)
            assert worker.fake.turns == ["one"]
            assert worker.fake.interrupts == 0
            async with asyncio.timeout(5):
                await worker.interrupt()
                async for event in feed:
                    if isinstance(event, events.StreamEnd):
                        assert event.finish_reason == "other"
                        break
        assert worker._runtime is not None
        assert not worker._runtime.reader
        assert worker._runtime.stream_owner is None


async def test_stream_reserves_the_reader_before_send_acceptance(
    worker: fakes.FakeWorker, monkeypatch: pytest.MonkeyPatch
) -> None:
    entered = asyncio.Event()
    release = asyncio.Event()
    turn = fakes.FakeAdapter.turn

    async def slow_turn(
        adapter: fakes.FakeAdapter, prompt: _adapter.Prompt
    ) -> AsyncIterator[events.Event]:
        entered.set()
        await release.wait()
        async for event in turn(adapter, prompt):
            yield event

    monkeypatch.setattr(fakes.FakeAdapter, "turn", slow_turn)
    async with worker, asyncio.timeout(5), asyncio.TaskGroup() as tasks:
        first = tasks.create_task(
            collect(worker.stream([ai.user_message("one")]))
        )
        await entered.wait()
        try:
            for other in (
                worker.events(),
                worker.stream([ai.user_message("two")]),
            ):
                async with contextlib.aclosing(other):
                    with pytest.raises(
                        workers.WorkerError, match="already has a reader"
                    ):
                        await anext(other)
        finally:
            release.set()
    assert isinstance(first.result()[-1], events.StreamEnd)
    assert [message.text for message in worker.messages] == ["one", "one"]


async def test_cancellation_before_acceptance_cleans_up(
    worker: fakes.FakeWorker, monkeypatch: pytest.MonkeyPatch
) -> None:
    entered = asyncio.Event()
    release = asyncio.Event()
    exited = asyncio.Event()
    turn = fakes.FakeAdapter.turn

    async def slow_turn(
        adapter: fakes.FakeAdapter, prompt: _adapter.Prompt
    ) -> AsyncIterator[events.Event]:
        entered.set()
        try:
            await release.wait()
            async for event in turn(adapter, prompt):
                yield event
        finally:
            exited.set()

    monkeypatch.setattr(fakes.FakeAdapter, "turn", slow_turn)
    async with worker, asyncio.timeout(5):
        reader = asyncio.create_task(
            collect(worker.stream([ai.user_message("one")]))
        )
        try:
            await entered.wait()
        finally:
            reader.cancel()
            with pytest.raises(asyncio.CancelledError):
                await reader
        assert exited.is_set()
        assert worker.messages == []
        assert worker.fake.interrupts == 1
        assert worker._runtime is not None
        assert worker._runtime.pump is None
        assert not worker._runtime.reader
        assert worker._runtime.stream_owner is None
        assert worker._runtime.submission is None
        release.set()
        answer = await ai.experimental_generate(
            worker, [ai.user_message("one")]
        )
        assert answer.text == "one"
        assert worker.fake.turns == ["one"]


async def test_interrupt_ends_a_turn_but_not_the_session_feed(
    worker: fakes.FakeWorker,
) -> None:
    worker.script(lambda prompt: [*fakes.text("partial"), fakes.Pause()])
    async with worker, contextlib.aclosing(worker.events()) as feed:
        async with asyncio.timeout(5):
            await worker.send("one")
            async for event in feed:
                if isinstance(event, events.TextEnd):
                    await worker.interrupt()
                if isinstance(event, events.StreamEnd):
                    assert event.finish_reason == "other"
                    assert event.provider_metadata is not None
                    assert (
                        event.provider_metadata["worker"]["finish_reason"]
                        == "cancelled"
                    )
                    break
            assert worker.fake.interrupts == 1
            assert worker.messages[-1].text == "partial"
            worker.script(fakes.reply)
            await worker.send("two")
            accepted = await anext(feed)
            assert isinstance(accepted, workers.UserMessageAccepted)
            assert accepted.message.text == "two"
            async for event in feed:
                if isinstance(event, events.StreamEnd):
                    assert event.finish_reason == "stop"
                    break
            assert worker.messages[-1].text == "two"
            assert worker.fake.turns == ["one", "two"]


@pytest.mark.parametrize("api", ["direct", "stream"])
async def test_cancelling_a_reader_cleans_up_and_allows_another_turn(
    worker: fakes.FakeWorker, api: str
) -> None:
    waiting = asyncio.Event()
    worker.script(lambda prompt: [*fakes.text("partial"), fakes.Pause()])

    async def read() -> None:
        if api == "direct":
            await worker.send("one")
            feed = worker.events()
        else:
            feed = worker.stream([ai.user_message("one")])
        async with contextlib.aclosing(feed):
            async for event in feed:
                if isinstance(event, events.TextEnd):
                    waiting.set()

    async with worker, asyncio.timeout(5):
        reader = asyncio.create_task(read())
        try:
            await waiting.wait()
        finally:
            reader.cancel()
            with pytest.raises(asyncio.CancelledError):
                await reader
        assert worker.fake.interrupts == 1
        assert not worker.fake.running
        assert worker._runtime is not None
        assert worker._runtime.pump is None
        assert not worker._runtime.reader
        assert worker._runtime.stream_owner is None
        assert worker._runtime.pending == {}
        assert worker._runtime.approvals == set()
        assert [message.text for message in worker.messages] == [
            "one",
            "partial",
        ]
        worker.script(fakes.reply)
        answer = await ai.experimental_generate(
            worker, [*worker.messages, ai.user_message("two")]
        )
        assert answer.text == "two"
        assert worker.fake.turns == ["one", "two"]


async def test_sync_and_fork_require_an_idle_session(
    worker: fakes.FakeWorker,
) -> None:
    async with worker:
        for action in (worker.sync, worker.fork):
            with pytest.raises(workers.WorkerError, match="no session_id"):
                await action()
        worker.script(lambda prompt: [fakes.Pause()])
        await worker.send("one")
        for action in (worker.sync, worker.fork):
            with pytest.raises(
                workers.WorkerError, match="stop the turn and reader"
            ):
                await action()
        async with (
            contextlib.aclosing(worker.events()) as feed,
            asyncio.timeout(5),
        ):
            await worker.interrupt()
            async for event in feed:
                if isinstance(event, events.StreamEnd):
                    break
            # Even between turns a live reader owns the mirror.
            for action in (worker.sync, worker.fork):
                with pytest.raises(
                    workers.WorkerError, match="stop the turn and reader"
                ):
                    await action()


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
        await worker.sync()
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
