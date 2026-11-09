# Workers: one object, one CLI session

`ai.workers` is experimental. `ClaudeCode`, `Codex`, and `Pi` share one session
API. They also work with `ai.stream`, `ai.experimental_generate`, and
`ai.Agent.run`. The CLI store owns the transcript; `worker.messages` is its
Python mirror.

## Direct control

```python
import contextlib

import ai
from ai import workers

async def main():
    async with workers.ClaudeCode(model_name="sonnet", approve=True) as worker:
        await worker.send("Inspect this directory and summarize it.")
        async with contextlib.aclosing(worker.events()) as feed:
            async for event in feed:
                if isinstance(event, ai.events.PendingHookEvent):
                    ai.resolve_hook(
                        event.hook,
                        {"granted": False, "reason": "No tool access today."},
                        registry=worker.hook_registry,
                    )
                elif isinstance(event, ai.events.TextDelta):
                    print(event.chunk, end="")
                elif isinstance(event, ai.events.StreamEnd):
                    break
```

- `open()` starts the transport once. Session creation is lazy until input.
  The context manager and input/output methods call `open()` automatically.
- `send()` takes a string or a list of `TextPart | FilePart`. It waits for
  transport acceptance, not an answer. During a turn, it sends guidance.
- `events()` is one feed across turns. Only one reader is permitted. A second
  reader, including an overlapping model stream, raises `WorkerError`.
- `interrupt()` requests a clean turn end. Keep reading to receive the end event.
- `close()` stops the CLI and unblocks the reader. The saved fields remain.
  Use `contextlib.aclosing` if you leave a reader early; it stops active work.

The feed type is `workers.SessionEvent`:

| Event | Meaning |
| --- | --- |
| `StreamStart`, `StreamEnd` | Assistant response boundaries |
| `UserMessageAccepted` | Transport accepted input; reading this event updates the mirror |
| `Text*`, `Reasoning*`, `BuiltinTool*` | Output and CLI-owned tool activity |
| `HookEvent` | An approval is pending, resolved, or cancelled |

Direct approvals work without Agent. Pass `registry=worker.hook_registry` to
`ai.resolve_hook`. Inside Agent, its current registry is used, including when
the CLI transport was opened before the run. Approval coverage depends on the
native policy; `approve=True` is not a sandbox. Bare model streaming outside
Agent cannot deliver hook events: use the direct feed or Agent for approvals.

## Model use

```python
async def model_use():
    async with workers.Codex() as worker:
        answer = await ai.experimental_generate(
            worker, [ai.user_message("Explain the tests.")]
        )
        follow_up = await ai.experimental_generate(
            worker, [*worker.messages, ai.user_message("Give a short summary.")]
        )
        return answer, follow_up
```

`stream()` sends only new input, then reads `events()` through one assistant
message. To add Python tools, use `ai.Agent(tools=[...]).run(worker, messages)`.
The MCP bridge passes these calls to Agent. CLI tools remain built-in events
and are not executed again by Agent. Direct `send()` does not take Python tools.

The request must preserve the known history. Changed ids, roles, or parts raise
`TranscriptMismatchError`; a mismatch never silently creates a new session.
Structured output keeps native parts in the mirror and saves the model-facing
parts in message metadata. Validation retries also remain in native history,
but are hidden from the model-facing view. Both complete views can continue the
session, including after dump/validate. After `sync()`, use the new native mirror.

Read pending direct events before switching to model use. Alternatively, once
the turn is idle, `sync()` replaces the mirror and discards superseded events.

## State, fork, and TUI

- `model_dump()` / `Worker.model_validate()` save and restore settings,
  `session_id`, and `messages`. They do not save processes or pending hooks.
- `sync()` reads authoritative native history. It requires a session id and
  no active turn or reader. Store normalization can change message ids/parts.
- `fork()` creates an already-open native branch with a different session id
  and loaded history. It requires an idle session with no reader. Close the
  returned worker, or manage it with `async with`.
- `model_copy()` resets private transport state but retains the session id.
  It is **not a fork**. Do not use two copies as concurrent writers.
- `tui_command()` returns `(argv, env)`. Close the worker before running that
  command in `worker.cwd`; call `sync()` when the person finishes. `tui()`
  handles those steps. `ConcreteWorker.list_sessions(cwd=...)` lists sessions.
- Unsupported operations raise `UnsupportedError`. There is no capabilities
  object or adapter instance. Concrete workers own private CLI state and
  `_cli_*` methods; common `send()` / `events()` logic stays in the base.

## Migration

| Old API | New API |
| --- | --- |
| `kind` | `worker_class_id` |
| `model` | `model_name` |
| `fork=True` | `await worker.fork()` |
| `load()` | `await worker.sync()` |
| `steer()` | `await worker.send(prompt)` |

Old field names have no aliases. Rename saved keys and remove the old fork flag
before validation. If it was true, explicitly create a native fork.
`executable` remains a string: a PATH command or filesystem path.

A fresh worker can import history on its first model call. This is **lossy
seeding**, not a native fork. Unsupported reasoning/files are omitted or rendered
as notes with warnings. Native ids and arbitrary metadata are not guaranteed.

## CLI limits and verification

Acceptance is transport-level. Claude confirms an SDK input write; Codex confirms
a start/steer RPC; Pi confirms a prompt that can be queued or extension-handled.
This does not prove native storage or expose universal input-to-turn correlation.
Only `sync()` reconciles the mirror with the CLI store. Claude's response counting
assumes one terminal result per accepted query; native coalescing and background
responses need live verification. Pi has no public follow-up selector.

Offline verification includes workers, Agent hooks, and model integration:

```sh
uv run --frozen pytest -q tests/workers tests/agents tests/models --timeout=15 --tb=short
uv run --frozen ruff format --check src tests examples
uv run --frozen ruff check src tests examples
uv run --frozen mypy
uv run --frozen ty check
```

Live tests were updated but not run. They use real CLI logins and paid model
calls. Historical `VERIFIED_VERSION` comments do not verify this rework.
