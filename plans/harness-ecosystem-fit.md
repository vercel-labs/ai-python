# Harness SDK: fit it into the `ai` ecosystem

## 0. Framing

A harness (Claude Code, Codex) is an **agent**, not a model. It owns its
loop, its tools, its context window and its transcript. So it sits at the
level of `Agent.run`, or above it (a harness can run subagents of its own).

```
ai.stream            one model call        we see every token, we own nothing else
Agent.run            loop of model calls   we own the loop, tools, hooks, steps
harness turn         someone else's loop   we see what the CLI reports, we own very little
```

What follows from that:

- Compare `Session`/`Turn` with `Agent`/`AgentStream`, **not** with
  `ai.stream`/`Stream`.
- Events are **coarser and uneven**. Each CLI reports different detail
  (claude streams tokens and whole messages; codex reports "items"). Never
  make up detail the CLI did not report: no fake step boundaries, no fake
  timings, no fake per-step usage. The harness code already lives by this
  ("no invented certainty").
- Control is **different, not smaller**. The harness lacks the agent's
  fine-grained levers (custom `loop`, `ToolRunner`, model/tool middleware,
  per-step params, replay, serverless hook deferral). It has levers the
  agent lacks: `steer`, `stop` from any task, `resume`/`fork` by id, TUI,
  `detach`. Neither should be forced into the other's shape.
- So the shared ground is the **run-level contract**: an `AgentEvent` stream
  with a few guarantees, the resulting `list[Message]`, the output, and the
  "blocked on approval" state. Consumers (UI adapter, aggregators,
  persistence, telemetry) should depend only on that contract, and must
  tolerate coarse streams.

## 1. Levers each level gives

| Lever | `Agent.run` | Harness turn |
|---|---|---|
| Change the loop | override `loop` | no |
| Run / gate each tool | `ToolRunner`, `require_approval`, tool middleware | `approve=` only (and only what the CLI offers to the hook: claude all, codex policy) |
| Rewrite tool input | middleware | `Allow(input=)` (claude only) |
| Per-step model/params | yes | no (model/effort per harness) |
| Approval while nobody is connected | `HookDeferredException` + replay | no: the CLI waits (or runs without a gate) |
| Interrupt | close the stream / cancel | `stop()` from any task, `break`, `timeout=` |
| Inject guidance mid-run | no | `steer()` |
| Continue later | pass `messages` again (stateless) | `resume(id)`, `fork(id)`, `Handle`, `history=` |
| Human drives it | no | `tui()` |

## 2. Event detail each source gives

| Signal | `Agent.run` | claude | codex |
|---|---|---|---|
| Text deltas | yes | yes | yes |
| Reasoning | provider-dependent | yes | summaries/deltas |
| Tool call args streamed | yes | no (whole block, one delta) | no |
| Tool call → result pairs | yes | yes | yes |
| Model-call (step) boundaries | yes | probably (raw `message_start`/`message_stop`; adapter drops them today — check) | no |
| Per-step usage | yes | no (turn total) | no (turn total) |
| Turn usage | sum of steps | yes, plus cost | yes |
| Subagent activity | via `PartialToolCallResult` | `agent_id` on tool calls only | no |

The contract (section 4) only promises the rows every source has.

## 3. What was checked (probe with `FakeAdapter`, `FakeModel`)

Script: text → tool call → tool result → text.

1. Harness turn: `StreamStart, Text*, Tool*, ToolCallResult, Text*,
   StreamEnd` — one `StreamStart/StreamEnd` around the whole turn. With the
   framing above this is **honest**, not a bug: it is a run, not one model
   call.
2. `ai.ui.ai_sdk.to_stream(turn)`: one UI step for the whole turn. Text,
   tool input and tool output all render. Fine.
3. Harness as a `SubAgentTool` inside `Agent.run`: **works today**. Events
   arrive as `PartialToolCallResult`, the result is a `MessageBundle`, the
   parent model sees the final text.
4. `util.merge` treats a `Turn` as restartable (`aiter(turn) is not turn`).
   Harmless now; differs from `AgentStream`.

Found by reading code (reproduce before fixing):

5. `RunStateTracker` can never report "blocked" on a harness stream: it
   needs a `StreamEnd` before the hook, and the harness's only `StreamEnd` is
   the last event. Inferring is wrong here anyway: the harness *knows* when
   it is blocked (an approval is pending and the CLI is waiting on us).
6. **The approval hook runs in the wrong context.** Codex answers server
   requests in tasks spawned from the jsonrpc pump, made at `harness.open()`.
   Claude's hooks run in tasks spawned from claude-agent-sdk's read loop,
   made at `client.connect()` — during the **first** turn. So contextvars in
   `approve` are those of `open()` or of turn 1: `ai.hook()`,
   `get_runtime()`, `current_tool_call()`, the current telemetry span and any
   user contextvar are missing or **stale** (turn 2 sees turn 1's).

## 4. The shared run-level contract

What any agent-level stream (`AgentStream`, `Turn`, a future harness)
promises. Write it down in `ai.types.events` docs; make consumers rely on
nothing more.

Always:

- `AgentEvent`s; every `ToolCallResult` follows the `ToolEnd` of its call.
- At least one `StreamStart … StreamEnd`; the last `StreamEnd` carries the
  finish reason. `StreamStart/StreamEnd` mark *a model stream as the source
  reports it* — one per model call for `Agent.run`, possibly one per run
  for a harness.
- Every `ModelEvent.message` is the assistant message so far; message ids
  are stable.
- At the end: `messages` (the conversation in `ai` shape), `output`
  (final text by default), usage if known, `blocked`.

Sometimes (consumers must not require them): step boundaries, reasoning,
streamed tool args, per-step usage, `HookEvent`/`RunBlocked`,
`PartialToolCallResult`.

Never assumed: that a `ToolEnd` is ours to execute.

## 5. Plan

Order: G1 → G2 → G3 → G4 → G5, then the rest. Offline tests in
`tests/harnesses/experimental/` (`FakeAdapter`), CI checks from `AGENTS.md`.

### G1. Fix the approval context (P0 bug)

At turn start, `_drive` stores `contextvars.copy_context()` on the session.
`Harness` wraps `self.approve` once (it already assigns `adapter.approve`):
the wrapper runs the user hook with `asyncio.create_task(hook(ctx),
context=<active turn's context>)`. One place, no adapter changes.

- Test: a `FakeAdapter` that calls `approve` from a task spawned at
  `start()`; the hook sees the consumer's contextvar, and turn 2 sees turn
  2's value.

### G2. `Turn` shaped like `AgentStream`, where the meaning matches (P1)

Additive:

- `Turn.__aiter__` returns `self`; `__anext__` drives `_drive` (also drops
  a generator layer, which helps close propagation).
- `async with session.stream(prompt) as turn:`; exit = `aclose()`.
- Live `turn.messages`, `turn.output` (final text by default; `Result.output`
  too), `turn.usage`, `turn.blocked`, `turn.deferred_hooks`.
- Not copied: `context` (there is no `Context`: no model, no tools of ours),
  `asend/athrow`.
- Optional: a structural `Protocol` both `AgentStream` and `Turn` satisfy
  (`AsyncIterator[AgentEvent]`, `messages`, `output`, `blocked`, `aclose`),
  so app code (an SSE endpoint, persistence) is written once. Only if a
  real caller needs the type; duck typing may be enough.

### G3. The harness reports what it knows; consumers stop requiring more (P1)

Do **not** add fake step boundaries. Instead:

- Harness emits `RunBlocked` itself when an approval is pending (it knows),
  and `HookEvent`s for approvals (see G4). `Turn.blocked` reads that.
- Audit core consumers for step-level assumptions and make them tolerant:
  - `RunStateTracker`: document that it infers from step events and is for
    loops that emit them; harness streams carry their own `RunBlocked`.
  - `ui.ai_sdk.outbound_stream`: already fine with one step (probe 2);
    add a test that pins it.
  - `MessageAggregator` / `MessageHydrator`: already fine with id changes
    mid-stream (probe 3); add a test.
- Claude only, if verified: the raw stream has `message_start/message_stop`
  per model call. If so, the adapter may emit real step boundaries there
  (with no per-step usage). Gate it on a capability (`steps: bool`), so the
  difference is declared, not hidden. Codex stays one stream per turn.

### G4. Approvals can go through `ai.hook` (P1)

`approve=` stays the plain-Python primitive. Add one ready-made hook, e.g.
`harnesses.hook_approval`:

```python
approval = await ai.hook(
    f"approve_{call.id}",
    payload=ai.ToolApproval,
    metadata={"tool": call.name, "kwargs": call.input, "providerExecuted": True},
    tool_call_id=call.id,
)
return Allow() if approval.granted else Deny(approval.reason or "")
```

Same label and metadata as `GatedToolCall`, so `ai.resolve_hook`, the AI SDK
UI approval events and `approvals.py` work unchanged.

- The turn hosts the sink, like `Agent.run`: wrap `_drive` in
  `runtime.run(...)` and `use_hook_registry(current or new)`; expose
  `turn.hook_registry`. Consider pulling the "host a run" part of
  `Agent._run` (runtime, registry, replay filter) into something both share
  — share the hosting, not the shape.
- Needs G1.
- Honest limits, in docs: this is control at the one point the CLI offers
  (a tool call it asks about: all on claude, policy on codex). Deferral
  (`HookDeferredException`, serverless) is impossible — the CLI is blocked
  waiting — so it denies and is listed in `approval_errors`.
- Nested in an `Agent` tool, approval events reach the parent as
  `PartialToolCallResult`, same as a nested `ai.Agent` today. Not fixed here.

### G5. Telemetry at the run level (P1)

- New span kind `harness_turn` (`HarnessTurnSpanData`: harness, version,
  session_id, prompt, messages, finish_reason, usage incl. cost), a peer of
  `run`. Opened in `_drive`. Inside an `Agent` tool it nests under
  `tool_execution` on its own.
- Events on that span, not child spans, for what the CLI reports but we do
  not time: first output, tool call seen, tool result seen.
- Approval `hook` spans come free with G4.
- No per-model-call spans (we do not see model calls), except claude steps
  if G3's capability lands.
- Update `otel.py` and the console/viewer.

### G6. Harness inside an `Agent`, supported (P1)

Works today (probe 3). The primitive is a plain `@ai.tool` that yields
`session.stream(...)` events; no helper.

- Example `examples/harnesses/as_tool.py`.
- Offline test with `FakeModel` + `FakeAdapter`: result, model input, and
  closing the parent run stops the CLI turn (`adapter.stops`).
- Docs: when to use a harness as a tool (delegate a whole coding task) vs.
  an `ai.Agent` (you need the levers in section 1).

### G7. Mark tools the CLI ran (P2)

Harness tool calls are not ours to execute (contract: "never assume a
`ToolEnd` is ours"). Keep `ToolCallPart`/`ToolResultPart` (history stays
portable for handoff and for other models), and mark them, e.g.
`provider_metadata={"harness": {"executed_by": "claude-code"}}`. The UI
adapter can map that to `providerExecuted`. Do not switch to `BuiltinTool*`:
it moves results into the assistant message and breaks handoff.

### G8. Messages in, like the rest of `ai` (P2)

- `Prompt` accepts a user `Message` (`ai.user_message(...)`).
- `harness.run(messages=[...])`: last user message is the prompt, the rest
  is `history`. The stateless mental model, on top of the stateful session.

### G9. Docs (P2)

- Name the harness `harness` in docs/examples, not `agent`.
- Add sections 1, 2 and 4 of this file to the docs: what each level lets
  you see and control.

### G10. Async hygiene (P3)

- jsonrpc request tasks and pump: own them in a `util.TaskGroup`, or check
  close cancels and awaits all of them.
- `Turn._closer` via `ensure_future`: keep, document why.
- Codex per-turn queue is unbounded: note it.
- Recheck `util.merge`/`decouple` with `Turn` after G2.

## 6. Not doing

- Fake step boundaries or per-step spans (dropped from the first draft:
  makes up detail the CLI does not report).
- A harness as an `ai.Model` provider: wrong level (it is an agent).
- A harness as an `ai.Agent` subclass: `Agent.run` needs a model, tools and
  a stateless `Context`; a harness has none of them. Share internals instead
  (G4).
- Middleware for harness turns: no hook points to wrap below the turn.
- Exposing `ai.tool`s to the CLI (in-process MCP for claude, stdio for
  codex): good follow-up, separate plan. It is the other direction of
  nesting (an `ai` tool inside a harness).

## Progress

- [x] G1 approval context
- [x] G2 Turn shape (see decisions)
- [x] G3 honest events, tolerant consumers (claude `message_start` check
  not done)
- [ ] G4 ai.hook bridge
- [ ] G5 run-level telemetry
- [x] G6 harness as tool
- [ ] G7 mark CLI-run tools
- [x] G8 messages in
- [x] G9 docs
- [x] G10 async hygiene

## Decisions made while building

- `Turn` is NOT its own iterator (`__aiter__` still returns a fresh
  generator). A self-iterating turn stays referenced by the caller, so a
  bare `break` would no longer close it and the CLI would keep working.
  The `ai` convention is served by `async with` instead.
- `_drive` races the adapter's next event against an approval-notice queue,
  each in its own task, one adapter step in flight at a time (still paced
  by the consumer). Needed because the CLI's stream is silent while it
  waits on the hook. Safe for both adapters: claude reads an anyio memory
  stream, codex an asyncio queue; no adapter holds a timeout or a cancel
  scope across a yield.
- Approval events use `hook_type="harness_approval"`
  (`APPROVAL_HOOK_TYPE`), not `ToolApproval`: the `approve=` hook answers
  them, so the UI must not offer to. G4 would add `ToolApproval` hooks on
  top.
- `Result.output` is the text when there is no `output_type`.
- `Harness.run(list[Message])`; a session turn refuses a conversation.
- jsonrpc already cancelled and awaited its request tasks on close (and a
  test covers it); no change there.

## Verified

- Offline: harness + workspace suites, the whole repo suite, afk offline
  tests, ruff, mypy (`src tests`, as CI), ty, check-examples.
- Live, local: the whole e2e suite on claude and codex (215 passed, 6
  sandbox-only skips). Sandbox axis not run.

## Session 2: patterns documented and run

- Docs "Combine and customize": merge turns, fan out with `yield_from`, a
  turn as a step of `Agent.loop`, a loop around turns, and what a loop
  cannot change (events report what already happened; control is
  `approve=`, steer/stop, and the loop between turns). Each pattern has an
  offline test in `test_interop.py`.
- Examples `as_tool.py`, `combine.py`, `custom_loop.py`, `until_done.py`,
  run for real against claude (all PASS through `run-examples.py`), plus
  basic, streaming, steering, tool_approval, take_over, resume (PASS).
- `custom_loop.py` first gave the reviewer only the harness's claim, and it
  answered NO; it now reviews evidence read from the workspace.
- Not run: `handoff.py`'s sandbox step, `sandbox.py`, `copy_files.py`,
  `reconnect.py` (`VERCEL_OIDC_TOKEN` expired; `vercel env pull` needs an
  ask first).
- The shell's exported `AI_GATEWAY_API_KEY`/`ANTHROPIC_AUTH_TOKEN` are stale
  (401 from claude); examples were run with `.env.local` applied on top, as
  the e2e fixture does.
