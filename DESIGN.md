# `ai.workers` design

Status: implemented for Codex, Claude Code and Pi (see section 10). Written in ASD-STE100 style.

## 1. Goal

A worker is one session of a coding-agent CLI (Claude Code, Codex, Pi).

1. A worker obeys the model protocol. You make a worker, then you give it to `ai.stream` or `ai.Agent.run`, as you give a model.
2. A worker has a small direct API: `start`, `send`, `events`, `close`. The model protocol uses this API. It does not use a second path.
3. The worker state is plain data: settings, `session_id`, `messages`. The messages use `ai.types.messages` only. Other models and workers can read them.

## 2. Rules

- Use Python protocols before new methods: `async with`, `async for`, task cancellation, `model_dump` / `model_validate`, `model_copy`.
- One worker class per CLI. The constructor is the only factory.
- The CLI's own tools are builtin tools: `BuiltinToolCallPart`, `BuiltinToolReturnPart`, `BuiltinTool*` events.
- `ai` Python tools go to the CLI through a local MCP bridge. They are normal `ToolCallPart`s.
- The worker does not run `ai` tools. `ai.Agent` runs them, or you run them.
- No capability object. An operation that a CLI cannot do raises `UnsupportedError`.

## 3. Layers

```
ai.stream / ai.Agent.run         framework (needs one core change, see 8)
        |  model protocol: worker.stream(messages, tools=, output_type=, params=)
L2  model adapter                 Worker.stream(), written one time in the base class
        |  start / send / events
L1  Worker (public)               state + 4 operations; event pump, fan-out,
        |                         message state, MCP bridge, telemetry
        |  private wire protocol
L0  wire (private, one per CLI)   process, native protocol, native -> ai events
```

Compare with models and providers:

| Models | Workers | Why |
|---|---|---|
| `Model` (id, data) | `Worker` (settings, state) | The thing that you give to `ai.stream`. |
| `Provider` (config, shared client) | none | Sessions do not share a client. Config goes on the worker. |
| `ProviderProtocol` (wire, can change) | private wire class | One CLI has one wire. You cannot change it. |
| `provider_class_id` registry | `worker_class_id` registry | `Worker.model_validate(data)` makes the correct subclass. |

### L0: wire (private)

Each CLI module has one wire class. Only the L1 core uses it.

```python
class _Wire(Protocol):
    async def start(self, worker: Worker, bridge_url: str) -> str: ...  # returns session_id
    async def send(self, message: Message) -> None: ...
    def events(self) -> AsyncIterator[WorkerEvent]: ...  # one reader: the L1 pump
    async def interrupt(self) -> None: ...                # used by L2 on cancel
    async def close(self) -> None: ...
```

The wire translates native records to `ai` events. It does not keep state. It does not know about subscribers, telemetry, or `ai.Agent`.

### L1: `Worker` (public)

```python
class Worker(pydantic.BaseModel):
    worker_class_id: str

    # settings
    model_name: str | None = None        # None: the CLI default
    cwd: str = Field(default_factory=os.getcwd)
    env: dict[str, str] = {}
    executable: str | None = None        # None: find the CLI on PATH
    tools: list[ai.tools.Tool] = []      # ai tool specs for the MCP bridge

    # state
    session_id: str | None = None
    messages: list[ai.Message] = []

    async def start(self) -> None: ...   # also __aenter__
    async def send(self, message: ai.Message | str) -> None: ...
    def events(self) -> AsyncIterator[WorkerEvent]: ...
    async def interrupt(self) -> None: ...  # see decision D2
    async def close(self) -> None: ...   # also __aexit__


class ClaudeCode(Worker): worker_class_id: Literal["claude-code"] = "claude-code"
class Codex(Worker):      worker_class_id: Literal["codex"] = "codex"
class Pi(Worker):         worker_class_id: Literal["pi"] = "pi"

WorkerEvent = ai.events.Event | MessageEvent

class MessageEvent(ai.events.BaseEvent):
    """A complete message went into `worker.messages`."""
    message: ai.Message
```

Operations:

- `start()`: Start the CLI and the MCP bridge. If `session_id` is set, resume that session. If not, and `messages` is not empty, seed a new session with `messages` (lossy). Start a second time: nothing happens.
- `send(message)`: Give one message to the CLI. A `str` becomes a user message. It returns when the wire writes the message. The CLI decides: a new turn or a steer. Role `user`: prompt or steer. Role `tool`: results for `ai` tool calls.
- `events()`: Make a new subscriber. The subscription starts when you call `events()`, not at the first `__anext__`. Each subscriber gets all events after that point. Many subscribers are permitted. Cancel the reader to stop it. The iterator stops at `close()`.
- `interrupt()`: Stop the current turn. The CLI keeps the session. It returns when the CLI ends the turn, or after a time limit. No turn: nothing happens.
- `close()`: Stop the CLI. The state stays. Close a second time: nothing happens.

The worker updates `messages` itself, from the pump. It does not need a subscriber.

### Message rules

These rules make `worker.messages` valid input for any model.

1. One assistant reply is one assistant message, from `StreamStart` to `StreamEnd`.
2. CLI tool calls and results are builtin parts in that assistant message.
3. An `ai` tool call ends the assistant message with `finish_reason="tool_call"`. The CLI waits. You send a `tool` message. The CLI continues in a new assistant message.
4. A steer ends the current assistant message. Then a user message follows. Then a new assistant message starts.
5. A message goes into `messages` when the CLI accepts it. Then `MessageEvent` comes.

Rule 4 removes the steer race. Old designs had this race: you send at the end of a turn, and you do not know the result. Here, a steer and a new turn give the same messages: a user message, then an assistant message.

### L2: model adapter

`Worker.stream(messages, *, tools, output_type, params)`. The base class has it. Workers do not change it.

1. `await self.start()`.
2. `self.messages` must be a prefix of `messages` (by message id). If not, raise `TranscriptMismatchError`. Exception: a worker that has no session and did not start. Then all of `messages` is the seed. The check ignores assistant messages that an interrupt ended (see D2).
3. Put the function tools of `tools=` in `self.tools`.
4. Call `events()`. Send each new message. Yield events until the next `StreamEnd`.
5. If the caller cancels or stops reading, call `interrupt()` (see D2).
6. One `stream()` at a time on one worker.
7. `params` has no effect. Use worker settings.

Thus `ai.Agent` runs `ai` tools as for a model: the stream ends with `tool_call`, the agent runs the tool, then it calls `stream()` again with the tool message.

## 4. Use

Model protocol:

```python
worker = workers.ClaudeCode(cwd=".")

async with ai.stream(worker, [ai.user_message("Fix the tests")]) as s:
    async for event in s: ...

async with ai.Agent(tools=[search_docs]).run(worker, messages) as run: ...
```

Direct control:

```python
async with workers.Codex(cwd=".") as w:
    feed = w.events()
    await w.send("Fix the tests")
    async for event in feed:
        if needs_steer(event):
            await w.send("Skip the slow tests")
        if isinstance(event, ai.events.StreamEnd) and event.finish_reason == "stop":
            break
```

State:

```python
data = w.model_dump()                       # save
w = workers.Worker.model_validate(data)     # continue the same session
branch = w.model_copy(update={"session_id": None})  # new session, seeded from messages

async with ai.stream(ai.get_model("openai:gpt-5"), [*w.messages, ai.user_message("Review this")]) as s: ...
```

## 5. Telemetry

- L1 makes one span for each session (`start` to `close`).
- L1 makes one span for each assistant message: usage, finish reason, model name.
- L1 makes one child span for each CLI tool call.
- On the model path, `ai.stream` makes its normal span. The L1 message span goes under it.

## 6. Decisions

### D1. No tool approval now

Decision: workers do not ask for approval of CLI tool calls. The wire never waits for a person. Each CLI uses its own non-interactive policy (sandbox, allowed tools, permission mode). The worker settings and the CLI's own config control that policy.

Reasons:

1. Approval is a framework-wide problem, not a worker problem. Provider-executed tools have the same gap today: the OpenAI protocol reads `mcp_approval_request` into a `BuiltinToolReturnPart`, and there is no way to send an answer.
2. Prior art agrees on one shape. The AI SDK v4 (TS) puts a `tool-approval-request` part in the assistant message and takes a `tool-approval-response` part in a `tool` message. `@ai-sdk/harness` uses the same parts for CLIs. Our `ai.ui.ai_sdk.approvals` already maps these parts to `HookPart(ToolApproval)`.
3. If workers take an approval shape now, the framework must later keep it or break it. Old attempts show the risk: #318 had its own `Allow` / `Deny` types, and #343 needed a hook registry outside `ai.Agent`.

When the framework has a design for provider-executed approvals, workers use it with no change to `send()`. It is probably a pause, like an `ai` tool call: the assistant message ends, and the answer comes in a `tool` message.

### D2. Interrupt: cancellation on the model path, `interrupt()` on the direct path

Decision: L1 has a public `interrupt()`. L2 calls it when the caller cancels `stream()` or stops reading it.

Reasons:

1. Cancellation needs one task that owns the turn. On the model path, the `ai.stream` caller owns it. Cancel and stop-reading then mean "stop the turn", as they do for a model.
2. On the direct path, no task owns the turn. `send()` returns at once, and many readers share `events()`. To cancel one reader cannot mean "stop the turn". To give ownership to `send()` or to a turn handle brings back the `Turn` concept.
3. All three CLIs have a native interrupt: Claude Code `interrupt` control request, Codex `turn/interrupt`, Pi `abort`. The wire needs it for L2 anyway, so the public method costs one line.

Rules for L2:

1. Catch `CancelledError` and `GeneratorExit` in `stream()`. Call `interrupt()` under a shield, with a time limit. Then raise again. (Checked: cancel, break, `asyncio.timeout` and a second cancel. An unshielded cleanup does not survive a second cancel.)
2. If the CLI does not end the turn in the time limit, close the CLI. `session_id` and `messages` stay, so the session can continue.
3. A cancel can come when no `stream()` runs, for example while `ai.Agent` runs an `ai` tool and the CLI waits on the MCP call. Then the next `send()` of a non-tool message, `interrupt()` or `close()` fails the waiting calls.
4. After an interrupt, the CLI and `worker.messages` keep the partial assistant message. The worker marks it as interrupted. `ai.Agent` does not keep a cancelled message, so the L2 prefix check ignores marked messages.

### D3. Tool changes and a late past restart an idle CLI

Decision: when `stream()` gets tools that differ from the tools that the CLI saw, and no turn runs, the worker closes the CLI and starts it again on the same session. When a started session is new and got no message yet, the restart opens a new session. That new session also takes the past of the first `stream()` call (seed).

Reasons:

1. A CLI reads the MCP tool list one time, when it connects to the bridge. Found in a live Codex test: `async with worker:` and then `ai.Agent(tools=...).run(worker, ...)` gave Codex no tools.
2. MCP `tools/list_changed` needs a server-to-client stream. The bridge has no stream, and CLIs do not all obey the notification.
3. `ai.Agent` gives the same tools on each call. Thus the restart occurs one time, not on each step.
4. A restart that resumes an unused session can fail: a CLI may store a session only after its first turn.

If the tools change while a turn runs, the worker warns and keeps the old tools until the next restart.

### D4. Rules found during implementation

1. A worker takes a past (seed) only when its session starts. With `session_id`, the CLI has the past, and `stream()` sends only the new trailing user messages.
2. System messages before the first user message, and `instructions=`, become the CLI's instructions. A system message after the start is refused.
3. `output_type` raises `UnsupportedError` for now (open question 2).
4. A steer during an L2 `stream()` ends that stream at the split (`finish_reason="other"`). `ai.Agent` then ends its run. The next run must use `worker.messages` as its history.
5. The wire must yield `Accepted` before the content that the message causes. Each CLI tells it in its own way:
   - Codex can send a turn's notifications before the `turn/start` response. Its wire keeps them until the response comes.
   - Claude Code echoes each user message when it takes it into the conversation (`--replay-user-messages`). The wire matches the echo by a `uuid` that it sets. A steer lands at the next step of the turn, not when it is written (measured).
   - Pi sends `message_start` with role `user` when a message goes into the conversation. The command response cannot be used: for a steer it says `queued` long before the message lands (measured).

## 7. Open questions

1. `output_type`. Native JSON schema flag (Codex `outputSchema`, Claude Code `--json-schema`), or instructions plus validation.
2. Steering during `ai.Agent.run` (D4.4).
3. `ai.durable.DurableAgent` replay with a stateful worker.
4. A slow subscriber. Unbounded queue, or drop with an error.
5. Native fork, sync from the native store, list sessions, TUI. Proposal: later, as plain functions in each CLI module (for example `workers.codex.list_sessions(cwd)`), not as `Worker` methods.
6. Tool approval (D1).

## 8. Core change

`ai.stream`, `ai.experimental_generate`, `ai.Agent.run` and `Context.model` accept a model-like object: `id`, `provider_name`, `stream(...)`. `Context` serializes a worker through `worker_class_id`.

## 9. Feature table

Old designs:

- #318 (`harnesses-experimental`, with #334 for Pi): `Harness` -> `Session` -> `Turn` -> `Adapter`, plus `ai.workspaces`.
- #343 stack (#342 `ModelLike`, #343, #344 head): one `Worker` with `send` / `events` / `interrupt` / `sync` / `fork` / `tui`.

| Capability | #318 | #343 stack | New |
|---|---|---|---|
| Use in `ai.stream` / `ai.Agent.run` | No | Yes | Yes |
| CLI tools as builtin events | No (host tool events) | Yes | Yes |
| `ai` Python tools through MCP | No | Yes | Yes |
| Telemetry | No | `ai.stream` span only | Session, message, tool spans |
| Start a turn directly | `session.run` / `stream` | `send` | `send` |
| Steer | `steer` (only in a turn) | `send` | `send` |
| Event feed | One iterator per turn | One session feed, one reader | One session feed, many readers |
| Steer race has one result in state | No | No | Yes (message rule 4) |
| Interrupt a turn | `stop` | `interrupt` | `interrupt`, and cancel on the model path (D2) |
| Approve CLI tool calls | Callable, `Allow` / `Deny` | `ai.hook` `ToolApproval` | No (D1) |
| Change tool input on approval, remember, deny and stop | Yes | No | No |
| Portable messages in state | Yes (not serializable) | Yes | Yes |
| Save and restore | `Handle`, `harness_from` | `model_dump` / `model_validate` | `model_dump` / `model_validate` |
| Resume by session id | Yes | Yes | Yes |
| Seed a new session from history | Yes | Yes | Yes |
| Native fork | Yes | Yes | No (later) |
| Read the native store again | `history` | `sync` | No (later) |
| List sessions | Yes | Yes | No (later) |
| Open the CLI TUI | Yes (PTY, detach, attach) | `tui_command`, `tui` | No (later) |
| Workspaces: local, Vercel Sandbox, remote | Yes | No | No |
| AI Gateway routing | Yes | No | No |
| Detach and attach to a live process | Yes | No | No |
| Session locks across processes | Yes | No | No |
| Capability flags | `Capabilities` | `UnsupportedError` | `UnsupportedError` |
| Structured output | Yes | Yes | No, not yet (open question 1) |
| Image and file input | Yes | Yes | Yes (message parts) |
| CLIs | Claude Code, Codex, Pi | Claude Code, Codex, Pi | Claude Code, Codex, Pi |

What the new design cannot express now:

1. Tool approval of any kind (D1).
2. Native fork, native store sync, list sessions, TUI.
3. Remote execution: sandbox, remote CLI, detach and attach, cross-process locks.
4. Gateway routing of the CLI model.

Items 1 and 2 can come back later without a change to the core API. Items 3 and 4 are out of scope.

## 10. Status

| Part | Where | State |
|---|---|---|
| L0 contract | `src/ai/workers/_wire.py` | Done |
| L1 + L2 core | `src/ai/workers/_worker.py` | Done |
| MCP bridge | `src/ai/workers/_bridge.py` | Done (from #343) |
| Codex wire | `src/ai/workers/codex/` | Done. Live tests pass with codex-cli 0.167.0. |
| Claude Code wire | `src/ai/workers/claude_code/` | Done, through `claude-agent-sdk` (extra `claude-code`). Live tests pass with Claude Code 2.1.319. |
| Pi wire | `src/ai/workers/pi/` | Done, through `pi --mode rpc` and a small extension that registers the bridge. Live tests pass with pi 1.5.0. |

Tests: `tests/workers` (offline). Live tests: `uv run pytest -m live tests/workers` (opt-in, they spend tokens).
