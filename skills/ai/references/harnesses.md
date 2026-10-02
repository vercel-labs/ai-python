# Harnesses and workspaces: traps

Read `https://ai-python.dev/docs/basics/harnesses.md` and
`https://ai-python.dev/docs/basics/workspaces.md` first. Both APIs are
experimental. Import them as module aliases:

```python
from ai.harnesses import experimental as harnesses
from ai.workspaces import experimental as workspaces
```

A harness drives a CLI (`claude`, `codex`). It is not `ai.Agent`: there is no
`ai.get_model`, and Python `@ai.tool` functions do not reach it. The CLI
brings its own tools, login, and session store.

## Setup

- `claude_code()` needs `ai[claude-code]`; `VercelSandbox` needs
  `ai[sandbox]` with a `vercel-sandbox` build that has interactive PTY
  support (vercel/vercel-py#415, unreleased). Both raise
  `ai.errors.InstallationError` at construction. `codex()` needs only the
  `codex` binary.
- On `Local`, a missing CLI raises `ExecutableMissingError`. The SDK installs
  CLIs only into a sandbox.
- Locally codex defaults to `sandbox="read-only"`, and in that mode it
  declines writes without trying. Pass `sandbox="workspace-write"` when it
  must edit files. In a sandbox leave it unset: other modes raise
  `UnsupportedError`, because codex's own sandbox cannot start in the VM.
- Open one harness per workspace and reuse it for many sessions. The CLI
  processes live as long as the `async with`.
- A harness opens its workspace but never closes it. Use `async with` on the
  workspace too, or a created sandbox runs until its time limit.

## Sessions and turns

- `agent.session()` has `session_id == ""` until its first turn. Save
  `session.handle` after a turn, not before.
- A `VercelSandbox` has no `coords` until it is open, so neither does a
  handle from it.
- `result.text` is only the text after the last tool call. Narration stays in
  `result.messages`. Use `output_type=` when you need to parse the answer.
- `run(..., timeout=)` and `session.stop()` return a `Result` with
  `finish_reason == "cancelled"`. They do not raise, except that with
  `output_type=` a cancelled reply that does not validate raises
  `ValidationError`.
- To end a stream early, `await session.stop()` then `break`. A bare `break`
  leaves the turn running; the next `session.run()` settles it, but a new
  `session.stream()` waits until you call `await session.reclaim()`.
- `steer` and `stop` live on the session, never the turn. Whether a steer
  interrupts work already in flight depends on the CLI; only rely on it being
  obeyed before the turn ends.
- `ToolCallPart.tool_args` is a JSON string. Act on `ToolEnd` for complete
  calls. Codex tool calls are named by item kind (`commandExecution`,
  `fileChange`), never by command line; the command is in `tool_args`.

## Approvals

- No `approve=` hook means every tool call is allowed, not "ask the user".
  Pass a hook to restrict.
- The hook is a gate, not an audit log (`capabilities.approval ==
  "policy"`). Read-only tools never reach it. Read the event stream to see
  every call.
- `ctx.call.name` and `ctx.call.input` are native. Claude uses `Bash`,
  `Write`, `Edit` with `file_path`. Codex uses `commandExecution` (the
  command is a string, often wrapped in `/bin/bash -lc`) and `fileChange`.
  In a sandbox codex often writes files through the shell. A hook for both
  harnesses must handle both dialects. Do not parse shell to judge
  containment; use `ctx.workspace.contains(path)` on absolute paths.
- `harnesses.Deny("reason")` and `Deny(reason="reason")` are the same.
- On codex (`deny_reason=False`) the reason never reaches the agent.
  `Allow(input=...)` works only on Claude. `Allow(remember=True)` works on
  neither. Unsupported options raise `UnsupportedError` from the turn.
- A hook that raises or exceeds `approval_timeout` (300s) denies the call and
  is listed in `result.approval_errors`.
- Never call session methods from inside the hook; the agent is blocked
  waiting on it.
- A detached agent with a hook stops at its next tool call until a client
  reconnects. For unattended work, run without a hook.

## Resume, fork, handoff

- One writer per conversation. `resume()` or `harness_from(handle)` on a
  conversation another SDK client holds (including one you `detach()`ed)
  raises `SessionBusyError`. Catch it and retry with `fork=True`, or
  `agent.fork(session_id)`.
- A person's open terminal TUI holds no claim. Fork conversations you did not
  start.
- A `Handle` never holds the hook or the gateway. Pass `approve=` and
  `gateway=` to `harness_from` again. The SDK keeps no registry; store handles
  yourself.
- `history=` handoff keeps paths from the source machine. `copy` the files and
  tell the agent where they are. Reasoning is dropped going into Claude, with
  a warning.

## Sandbox

- A CLI authenticates where it runs; your terminal login is not in the VM.
  Give the workspace a gateway: `VercelSandbox(gateway=vercel_ai_gateway())`.
  The VM only holds a placeholder key.
- `vercel_ai_gateway()` reads `AI_GATEWAY_API_KEY` from the process
  environment only. `VercelSandbox` reads only Vercel credentials
  (`VERCEL_OIDC_TOKEN`, `VERCEL_TOKEN`, `VERCEL_TEAM_ID`,
  `VERCEL_PROJECT_ID`) from `.env.local`. Other variables stay local unless
  passed in `env=` or `forward_project_env=True`.
- `VERCEL_OIDC_TOKEN` is short-lived; expiry raises `NotAuthenticatedError`.
  Run `vercel env pull` again.
- With a gateway, outbound traffic is an allowlist fixed at creation. Add
  hosts (package indexes, APIs) with `allow_hosts=[...]` up front.
  Reconnecting with `gateway=` to a VM made without one raises
  `WorkspaceError`.
- Every `exec`, `read_text`, `write_text`, and `exists` is a network round
  trip. Do not check `exists` before `read_text` (it raises
  `FileNotFoundError`). Move trees with `copy()`, which batches.
- Never use `pathlib` for workspace files. Use `read_text`, `write_text`,
  `exec`, and `copy`.
- The default VM lifetime is short. Set `execution_time_limit=` in seconds.
- `keep=True` keeps a VM you created. A VM reopened by `name=` stays running
  unless `keep=False`. To stop one:
  `async with workspaces.VercelSandbox(name=name, keep=False): pass`.
- Forget a saved sandbox only on `WorkspaceGoneError`. Other
  `WorkspaceError`s may mean it is still running.

## TUI

- `from ai.workspaces.experimental import tty` is explicit and POSIX only.
  `tty.bridge` needs a real terminal on stdin.
- Give `tui()` a `name` to survive detach (Ctrl-]). Reattach with
  `workspace.attach(name)`.
- A bare `tui()` on codex has `session_id is None` (and no `handle`): codex
  picks its own id. Find it afterwards with `agent.sessions()`.
