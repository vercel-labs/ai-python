# Harness SDK fixes

Issues found reviewing `src/ai/harnesses/experimental` and
`src/ai/workspaces/experimental` on `harnesses-experimental`. One section per
issue, so each can be picked up and fixed on its own.

## How to use this file (read first, every session)

1. Read **Progress** and **Session log** below to see where we stopped.
2. Pick the first unchecked issue in **Progress** (or the one agreed with the
   user). Mark it `[~]` in Progress when starting.
3. Work through that issue's **Tasks** checklist in order. Tick each box
   (`- [x]`) as soon as it is done, not at the end.
4. When every task of an issue is ticked, mark it `[x]` in Progress.
5. Before ending a session, add a line to **Session log**: date, issues
   touched, what's left, anything surprising.
6. Every code change ends with the CI checks (see **Checks**).

Markers: `[ ]` open · `[~]` in progress · `[x]` done · `[-]` dropped (say why).

Offline doubles for every fix: `tests/harnesses/experimental/fakes.py`
(`FakeAdapter`, `FakeWorkspace`, `FakeProcess`, `LockWorkspace`).

## Progress

Suggested order: H0 → H1 → H2 → H3/H4 → H5 → H6 → H7 → H8 → rest.

- [x] **H0** Fake adapter test fixture (P3, prerequisite)
- [x] **H1** Messages come out in the wrong order (P0)
- [x] **H2** `session.stop()` from another task raises (P0)
- [x] **H3** Codex `Deny(stop=True)` freezes for 30s (P0)
- [x] **H4** Approval hook blocks all codex traffic (P0)
- [x] **H5** `break` leaves the CLI turn running (P1)
- [x] **H6** Session lock can let two clients in (P1)
- [x] **H7** Error state shared across sessions (P1)
- [x] **H8** Claude steer may desync turns, verify first (P1)
- [x] **H9** Different default permissions per CLI (P2)
- [x] **H10** Approval hook is a weak gate (P2)
- [x] **H11** Depends on undocumented CLI internals (P2)
- [x] **H12** Too many network round trips in a sandbox (P3)
- [x] **H13** `history()` has no incremental read (P3)
- [x] **H14** Text-only input; tools/MCP not configurable (P3)
- [x] **H15** Docs fixes (P2, ongoing alongside the others)

Severity:

- **P0**: breaks normal usage (a UI, a service, a handoff).
- **P1**: wrong under concurrency or races; data or state corruption.
- **P2**: unsafe default, misleading docs, or fragility.
- **P3**: performance, test gaps, missing features.

## Checks

Run after every change (from `AGENTS.md`):

```bash
uv run ruff format --check src tests examples
uv run ruff check src tests examples
uv run mypy
uv run ty check
uv run pytest tests/harnesses/experimental tests/workspaces/experimental \
  --ignore=tests/harnesses/experimental/e2e \
  --ignore=tests/workspaces/experimental/e2e -q
```

Plus the relevant e2e tests when `claude` / `codex` (and Vercel creds for
sandbox tests) are available. afk offline tests:
`cd examples/apps/afk && uv run pytest -m "not live" -q`.

## Session log

Newest last. One entry per session.

- 2026-10-09 (review): found H0–H15, wrote this plan and
  `plans/harness_repro.py`. No code changed yet. Next: H0.
- Session 2: H0, H1, H2, H5, H3, H4 done.
  - `_session.py`: history built per segment (call → result → answer), live
    and mid-turn; `stop()` safe from any task (waits on `Turn._done`, closes
    only a suspended generator, `STOP_TIMEOUT`); a turn nobody reads (break,
    deadline, cancel) stops the agent in `_drive`'s finally and closes the
    adapter's generator; `stream()` reclaims an abandoned turn itself.
  - `jsonrpc.py`: server requests answered in their own tasks (cancelled on
    close, error reply if a handler fails); codex `_decide` takes the turn id
    from the request. Real codex `Deny(stop=True)`: 33s → 3.3s.
  - New offline tests: `test_session.py`, `test_jsonrpc.py`,
    `test_codex_adapter.py` (fake app-server). New e2e: stop from another
    task, break-then-run, tool-turn handoff, hook calling the harness.
  - Full local live suite passed (171) after H1/H2/H5; codex + approval
    suites passed after H3/H4. Sandbox e2e not run this session.
  - `plans/harness_repro.py` deleted (superseded by `fakes.py`).
  - Next: H6.
- Session 3: H6, H7, H8 done; H10, H11, H12 mostly done (open items need a
  decision or a sandbox).
  - H6: `_session_lock.py` rewritten around one stdlib helper run with one
    `exec`: flock'd check-and-set, lock built in a temp dir and renamed into
    place, per-holder token for claim/release, PermissionError = alive,
    recorded `ps` start time catches pid reuse. Forced interleaving test
    reproduced the double win on the old code.
  - H7: `_pending_error` per session in both adapters;
    `on_approval_error(session_id, detail)`; `Session.close()` forgets the
    session.
  - H8: measured claude: mid-turn steer is fine; a steer after the turn
    ended desynced every later turn. Session refuses it after `StreamEnd`;
    claude adapter refuses unless its CLI turn is running.
  - H10: claude hook now a PreToolUse hook (sees every call; measured) →
    `approval="all"`. `workspace.resolve()`. Example + docs rewritten.
  - H11: `VERIFIED_VERSION` per adapter, warning when older, version note on
    every turn error, `prepare_tui` fail-soft.
  - H12: claude `_cli_sessions` one command.
  - Final: full local live suite 198 passed / 6 skipped (sandbox-only);
    offline suite 1101 passed; afk offline 94 passed; all checks pass.
  - Blocked: sandbox e2e (`VERCEL_OIDC_TOKEN` expired; needs
    `vercel env pull`, ask first).
  - Waiting on the user: H9 option name/defaults; H10 policy helpers;
    H11 version pinning + scheduled CI; H13 follow API; H14 which options.
- Session 4: user decisions applied (H9 yolo default, H11 latest CLI, H13 no
  new API, H14 inputs + basic options, H10 no helpers). User requires e2e on
  multiple workspaces before merge; `vercel env pull` run (all local keys
  kept, backup was `/tmp/hx/env.local.bak`).
  - H11: found the sandbox image ships claude 2.1.283, so "install when
    missing" never gave latest → `base.provision()`.
  - H12: claude store listing now ≤2 round trips (was 1 + N).
  - Sandbox bug found by the matrix: `spawn()` returned before the
    supervisor's `mkfifo`, so the stdin pump could open a FIFO that did not
    exist yet (intermittent "input conduit exited before it was ready").
    `_PUMP` now waits for the FIFO; offline test with a late FIFO.
  - Runs: local live matrix 216 passed / 8 skipped. Sandbox matrix (before
    the pump fix): 215 passed, 4 failed = httpx ConnectError to the Vercel
    API at 6-way parallelism; rerun of all claude-sandbox + those: 90 passed,
    1 failed (claude-sandbox lock test). After the pump fix: 218 passed, the
    same test failed again — so the pump race was real but not the cause.
  - Real cause, found by making the failure report the program's own exit
    and stderr (`SandboxProcess._ended_note`): `claude --resume <fork>`
    exited with "No conversation found". `WorkspaceSessionStore.append`
    wrote the fork in place (sandbox file API stamps it 1970) and touched it
    after; a claude CLI already running in the VM sweeps "ancient"
    transcripts, and under load the sweep landed in between (then `touch`
    made an empty file). Now written to `<path>.part` (the CLI only
    enumerates `*.jsonl`, checked in the 2.1.289 bundle), touched, and
    renamed into place in one command. Repro under load: 1 of 12 failed
    before, 18 of 18 passed after.
  - FINAL (merge gate): sandbox matrix 219 passed / 2 skipped (claude-only
    options on codex); local live matrix 216 passed / 8 skipped; offline
    suite 1125 passed; afk offline 94 passed; ruff, mypy, ty, examples
    check all pass. Every H item done; remaining work is the afk bucket
    (not in this plan).
- Session 5: committed (6e8b734 workspaces, a14bc04 harnesses). Before
  committing, check-examples showed a regression the earlier runs missed
  (only the last summary line was read): claude-only options on claude
  alone made `{"claude-code": claude_code, "codex": codex}` a dict of bare
  `function` to mypy, failing afk and examples/harnesses/tui.py. Fixed by
  making codex() accept `tools`/`disallowed_tools`/`max_turns` and raise
  UnsupportedError when set (every claude_code() argument is a codex()
  argument again); offline + e2e (both × both workspaces) tests.
  Lesson: read check-examples' exit code, not its last line.
  - Pushed 6e8b734..71300f7 (SSH key not SAML-authorized for vercel-labs;
    pushed over HTTPS with the gh login, for that command only).
  - Then afk and deepysec were run for real (not only their tests), which
    found four SDK bugs, all older than this plan, and two app bugs:
    - codex 0.160 puts command output in `aggregatedOutput`; the adapter
      read `output`, so every codex command result was None in
      `result.messages`, `history()`, handoffs and afk peek (the agent
      itself saw the output). `_item_result`; offline + e2e tests.
    - `Local` spawned with asyncio's 64 KiB line limit; one codex
      `thread/list` reply on a machine with 3159 threads overflowed it
      ("codex crashed: Separator is found, but chunk is longer than
      limit"), so afk listed no codex conversations. `LINE_LIMIT`; test
      with a 5 MiB line.
    - codex `list_sessions()` fetched ONE unfiltered page of the machine's
      threads and filtered it here: conversations in this directory past
      that page were missing. Now `cwd`-filtered and paged.
    - `copy()` dropped the executable bit both ways (upload had no mode;
      `copy_tree` used copyfile): a pushed script could not run in the VM,
      one the agent made came home unrunnable. Modes now travel; e2e test.
    - afk `pull --files` also used copyfile; now copymode (not copy2: the
      sandbox's 1970 file times are not this tree's).
    - deepysec's sandboxes had the platform's 5-minute lifetime, shorter
      than one `all` run: the VM stopped under a revalidation turn
      (finish=cancelled, empty reply → "no valid reply after 3 attempts").
      `SANDBOX_LIFETIME` = 45 min. Measured: the CLI upgrade on a fresh
      sandbox costs 5.5s (0.7s when current), not the cause.
    - Not bugs: codex finds 0 deepysec findings on the fixture, same on
      the pre-plan code (model judgement); afk's live test 401 was this
      shell's stale credentials, same on the pre-plan code.
  - afk flows exercised on a pty: list (both CLIs), push --bg + peek,
    attach + instruct + detach, pull --files, codex TUI push + detach +
    peek + pull --files, stop. deepysec: all stages with claude locally,
    codex locally, claude in a sandbox (8.5 min run, after the lifetime
    fix).
  - Parallel sandbox runs use `uv run --with pytest-xdist ... -n 4` (no
    project dependency added).

---

## H0. Fake adapter test fixture (prerequisite) · P3

**Why first:** every e2e test drives a real CLI
(`tests/harnesses/experimental/conftest.py`: "no fakes"). That's why H1–H5
slipped through. We need fast, offline tests for the session state machine
before fixing it.

**Design:**

- `FakeAdapter`: implements the `base.Adapter` protocol. Its `turn()` is
  driven by a script (a list of events, plus optional pause points controlled
  by `asyncio.Event`s). It records `stop()`, `steer()`, `close_session()`
  calls, and tracks whether its turn generator's `finally` ran.
- `FakeWorkspace`: minimal `open()`, `coords`, `kind="local"`, in-memory
  `exec` / `read_text` / `write_text` / `exists` / `home` (enough for
  `SessionLock`), with a hook to inject interleavings.
- `FakeProcess`: an in-memory `readline` / `write` pair for `JsonRpcPeer`,
  with a hook to answer calls.

**Tasks:**

- [x] Create `tests/harnesses/experimental/fakes.py` with `FakeAdapter`
      (start from `plans/harness_repro.py`)
- [x] Add `FakeWorkspace` to `fakes.py`
- [x] Add `FakeProcess` to `fakes.py`
- [x] Add `tests/harnesses/experimental/test_session.py` with one smoke test
      (plain text turn → `Result.text`, `finish_reason="stop"`)
- [x] Add `tests/harnesses/experimental/test_jsonrpc.py` with one smoke test
      (call → response) — moved to H3, where it is first needed
- [-] Mark H1, H2, H3 repro tests as `xfail(strict=True)` — dropped: each
      test was written failing and fixed in the same session
- [x] Offline suite runs in < 5s with no CLI installed
- [x] Checks pass

---

## H1. Messages come out in the wrong order · P0

**Where:** `src/ai/harnesses/experimental/_session.py`, `_drive()` (~L295)
and `_settle()` (~L371).

**Problem:** tool results are appended to `self._messages` as they arrive
(~L360). The assistant message, which holds the tool calls, is appended only
at settle (~L394), using the adapter's `StreamEnd.message`, which carries
every part of the turn in one message. A tool-using turn produces:

```
[user, tool, assistant(tool_call + text)]
```

The tool result comes before the call that made it.

**Impact:**

- `result.messages` / `session.messages` are malformed for any tool turn.
  Docs say they work "anywhere `ai` messages do".
- `run(..., history=result.messages)` (documented handoff) feeds this to
  `_handoff.to_claude_records` / `to_codex_items`, writing a `tool_result`
  before its `tool_use`.
- The approval hook's `ctx.history` sees the same broken order.
- Live order differs from stored `history()`, which is correct.

**Fix approach:**

- Build history per *segment*, like the AI SDK agent loop: on
  `ToolCallResult`, first append the assistant segment so far (hydrated
  message for `segment.id`), then the tool message, then start a new
  segment. At settle, append only the final segment.
- At settle, ignore `StreamEnd.message.parts`; use hydrated segments and copy
  only metadata from `StreamEnd.message` (option b: keeps adapters simple).
- `Result.text` = text of the last segment (same "after the last tool call"
  behaviour, now for free; `_text_of` may simplify).
- The cancelled path in `_settle` (no `StreamEnd`) joins all segments into
  one today; split it the same way.

**Tasks:**

- [x] Fake test: tool turn → roles `[user, assistant, tool, assistant]`,
      call before result (fails first)
- [x] Fake test: two tool rounds → alternating segments, unique message ids
- [x] Fake test: cancelled mid-tool → no tool result before its call
- [x] Fake test: approval hook `ctx.history` has the right order mid-turn
- [x] Implement segment-based history in `_drive` / `_settle`
- [x] Keep provider metadata from `StreamEnd.message`
- [x] Recheck `Result.text` / `_text_of` (narration still excluded)
- [x] Recheck `steer()` message placement relative to segments
- [x] e2e: live `session.messages` and stored `history()` have the same role
      sequence, per harness
- [x] e2e (`test_handoff`): `history=result.messages` from a *tool-using*
      turn into the other harness succeeds
- [x] Docs (H15): note the order and that tool-turn handoff works
- [x] Checks pass

---

## H2. `session.stop()` from another task raises · P0

**Where:** `_session.py` `stop()` (~L431) → `_reclaim()` (~L216) →
`Turn.aclose()` (~L90).

**Problem:** `stop()` calls `_reclaim()`, which calls `aclose()` on the
turn's async generator. If the consumer task is awaiting the next event (the
usual case for a Stop button, watchdog, or HTTP cancel endpoint), Python
raises `RuntimeError: aclose(): asynchronous generator is already running`.
Same when `_once()` reclaims while another task's turn runs on the session.

**Fix approach:**

- `stop()` = set `_stopped`, call `adapter.stop()`, then *wait* for the
  active turn to settle (an `asyncio.Event` set in `_drive`'s `finally`),
  with a timeout. Never `aclose()` from a foreign task.
- The adapter's `turn()` ends on its own after interrupt (Claude: result
  after interrupt; codex: `turn/completed`).
- Fallback if the adapter doesn't end within N seconds: cancel the
  consumer's iteration through a stored task / cancel scope.
- Only `aclose()` when the generator is suspended (`ag_running` false), i.e.
  the "consumer walked away" case.
- `stop()` returns once settled, so `turn.result` is readable right after.

**Tasks:**

- [x] Fake test: consumer blocked mid-turn, `stop()` from another task →
      no exception, result `cancelled`, `adapter.stop` called once (fails
      first)
- [x] Fake test: `stop()` inside the `async for` body still works
- [x] Fake test: `stop()` with no active turn is a no-op
- [x] Fake test: adapter never ends → `stop()` returns after timeout, turn
      settled as cancelled
- [x] Fake test: concurrent `run()` on the same session doesn't raise
      (waits, or raises a clear error; decide which)
- [x] Add a "settled" event to `Turn` / `_drive`
- [x] Rewrite `stop()` and `_reclaim()` per the approach
- [x] Decide and pick the stop timeout constant
- [x] e2e: `stop()` from a separate task, both harnesses
- [x] Docs (H15): `stop()` safe from any task, returns once settled
- [x] Checks pass

---

## H3. Codex `Deny(stop=True)` freezes for 30s · P0

**Where:** `_adapters/jsonrpc.py` `_read_loop()` (~L123–150);
`_adapters/codex.py` `_on_request()` (~L725–745).

**Problem:** `_read_loop` does `await on_request(...)` inline. After
answering the approval, `_on_request` does
`await self._rpc().call("turn/interrupt", ..., timeout=30)`. That call waits
for a response only `_read_loop` can resolve, and `_read_loop` is stuck in
this very call. It always waits the full 30s, then `suppress` hides the
timeout. The e2e test only asserts `took < 60`.

**Fix approach:** H4's change (requests dispatched off the read loop) fixes
the deadlock. Also send `turn/interrupt` as its own task after `respond()`.

**Tasks:**

- [x] Fake test (`test_jsonrpc.py`): `call()` from inside `on_request`
      returns as soon as the server answers (fails first)
- [x] Fake codex test: `Deny(stop=True)` settles the turn in < 1s
- [x] Do H4's read-loop change (or land H3+H4 together)
- [-] Send `turn/interrupt` as a separate task after `respond()` — not needed:
      the whole request handler now runs in its own task. Also: `_decide` now
      takes the turn id from the request itself (`params.turnId`)
- [x] Tighten e2e bound in `e2e/test_approval.py` to `< 15s`
- [x] Checks pass

---

## H4. Approval hook blocks all codex traffic · P0

**Where:** `_adapters/jsonrpc.py` `_read_loop()`.

**Problem:** same root cause as H3. While the hook runs (up to
`approval_timeout`, default 300s, e.g. a human in a UI), nothing is read:

- no notifications → no streamed events for *any* session on that
  app-server;
- no RPC responses → `steer()`, `stop()`, `history()`, `sessions()` hang;
- a hook calling back into the harness (e.g. `agent.history()`) deadlocks
  until timeout.

**Fix approach:**

- Dispatch server requests with `asyncio.create_task(...)`; keep tasks in a
  set (no GC), cancel them on `close()`.
- Keep notifications inline to preserve order.
- Errors in request tasks: log and answer `decline`, never lose them.

**Tasks:**

- [x] Fake test: hook blocked on an event → notifications still reach the
      turn's queue, and a `call()` from another task resolves (fails first)
- [x] Fake test: two sessions, one blocked in approval, other streams
- [x] Fake test: hook that calls `agent.history()` doesn't deadlock
- [x] Dispatch requests as tasks in `_read_loop`
- [x] Track and cancel request tasks on `close()`
- [x] Error handling in request tasks (decline + report)
- [x] Recheck `_on_request`: interrupt still after `respond()`
- [x] Checks pass

---

## H5. `break` leaves the CLI turn running · P1

**Where:** `_session.py` `_reclaim()` / `Turn.aclose()`; Claude adapter
`turn()` (~L620+) and `_drain_interrupted()` (~L812); codex `turn()`.

**Problem:** `async for e in turn: break` (documented as fine) leaves the
CLI's turn running. `_reclaim()` closes the Python generator only:

- the inner `adapter.turn()` generator is abandoned, not closed; no
  `adapter.stop()` is sent;
- **Claude:** `_drain_interrupted` only runs for sessions in
  `_interrupted`. The next `query()` + `receive_response()` reads the *old*
  turn's tail and stops at the *old* `ResultMessage`. Turns are off by one
  from then on;
- **Codex:** the old turn keeps running server-side; the next `turn/start`
  errors or races.

**Fix approach (a):** on reclaim of an unsettled turn: `adapter.stop()`,
mark interrupted, explicitly `aclose()` the inner adapter generator (keep a
reference in `_drive`). Claude's `_drain_interrupted` then eats the leftover
result. `_drive`'s `finally` closes the adapter generator when suspended.

**Tasks:**

- [x] Fake test: break after first event, then `run()` → `adapter.stop`
      called, adapter generator `finally` ran, second result has the second
      prompt's text (fails first)
- [x] Keep a reference to the adapter generator in `_drive`, close it in
      `finally`
- [x] Reclaim of an unsettled turn calls `adapter.stop()` and marks it
      interrupted
- [x] Claude: confirm `_drain_interrupted` handles this path
- [x] Codex: confirm the next `turn/start` waits for the old
      `turn/completed` (or the interrupt ack)
- [x] e2e both harnesses: stream long answer, `break`, `run("say OK")` →
      text contains `OK`
- [x] Docs (H15): what `break` does now
- [x] Checks pass

---

## H6. Session lock can let two clients in · P1

**Where:** `_session_lock.py` `acquire()` (~L75–95), `_read()`
(~L147–160), `_live()` (~L169).

**Problems:**

1. **Double reclaim:** A and B see a stale lock. A `rm -rf` + `mkdir` (wins)
   + writes holder. B `rm -rf` (deletes A's *fresh* lock) + `mkdir` (wins).
   Both hold it. The docstring says this can't happen.
2. **Write window:** between `mkdir` and writing `holder.json` (a network
   round trip in a sandbox), a reader sees no record → `Holder(since=0)` →
   dead → steals it.
3. **Liveness:** `os.kill(pid, 0)` raising `PermissionError` exits non-zero
   → treated as dead. PID reuse → false busy.

**Fix approach:**

- Create the lock atomically with its record: write `tmpdir/holder.json`,
  then rename `tmpdir` → `lockdir` (atomic; fails if it exists).
- Reclaim by renaming the stale lock aside
  (`mv lockdir lockdir.stale.<rand>`); only the client whose rename succeeds
  goes on to create. Delete the aside dir after.
- Random token in the holder; `release()` only removes a lock whose token
  matches.
- Liveness: `PermissionError` = alive; store process start time to catch
  pid reuse.
- Do each operation in one `exec` of a small Python helper (ties into H12).

**Tasks:**

- [x] Fake test: two acquirers on a stale lock, interleaved → exactly one
      wins (fails first)
- [x] Fake test: a read during create never sees a record-less lock
- [x] Fake test: `release()` by a non-holder leaves the lock
- [x] Fake test: `PermissionError` pid counts as alive
- [x] Atomic create via temp dir + rename
- [x] Reclaim via rename-aside — done differently: the whole check-and-set
      runs under an `flock` mutex in one helper command, so reclaim is a
      plain remove + create with nobody else in between
- [x] Holder token, checked on release
- [x] Liveness fixes (PermissionError, start time)
- [x] Update the module docstring to match
- [x] Existing `test_session_lock.py` + `e2e/test_session_lock.py` pass
- [x] Checks pass

---

- [x] Verify in a Vercel Sandbox: `vercel env pull` done (user approved);
      lock e2e passes in the VM. Its one intermittent failure was NOT the
      lock: a sandbox spawn race (see session 4 log)

## H7. Error state shared across sessions · P1

**Where:**

- `_adapters/claude.py`: `self._pending_error` (~L94), reset at turn start
  (~L637), set in `_decide` (~L597), raised at ~L716.
- `_adapters/codex.py`: `self._pending_error` (~L123), reset ~L827, set
  ~L791/798, raised ~L898.
- `_harness.py`: `self._approval_errors` (~L41), `_record_approval_error`
  (~L332), `_take_approval_errors` (~L335).
- `_harness.py`: `self._sessions` (~L44) never shrinks.

**Problem:** one slot per adapter / one list per harness. With two
sessions: B's turn start clears A's pending error; A raises B's error; a
turn's `approval_errors` includes another session's hook failures.

**Tasks:**

- [x] Fake test: two sessions, hook fails only for A → only A's result has
      `approval_errors` (fails first)
- [x] Fake test: B's turn start doesn't clear A's pending error
- [x] Claude: `_pending_error` → `dict[session_id, BaseException]`
- [x] Codex: same
- [x] `on_approval_error(session_id, detail)`; per-session error list
- [x] Remove session from `Harness._sessions` on `Session.close()`
- [x] Checks pass

---

## H8. Claude steer may desync turns (verify first) · P1

**Where:** `_adapters/claude.py` `steer()` (~L826): `client.query(text)`
mid-turn.

**Problem (suspected):** if the CLI treats a mid-turn message as a queued
turn with its own `ResultMessage`, `receive_response()` stops at the first
result and the steer's result is read by the next turn (like H5).

**Tasks:**

- [x] Verify with real `claude`: steer mid-turn, count `ResultMessage`s,
      then `run("say OK")` and check its text. Findings (claude 2.1.289):
      a steer DURING the turn is folded into it (one `ResultMessage`, later
      turns in step). A steer sent once the turn has ended (e.g. on
      `StreamEnd`, while the session still counted the turn as active) was
      taken as a new turn: the next two answers were `LATE`, `TWO` instead of
      `TWO`, `THREE`. Codex already refused it (`no active turn`)
- [-] If no desync: add the e2e test as a guard, mark `[-]` for the fix —
      there was a desync, see below
- [x] Fix: the session refuses `steer()` once the adapter's `StreamEnd`
      arrived (`Session._ended`); the claude adapter refuses a steer unless
      its CLI turn is running (`_running`: prompted, no `ResultMessage` yet)
- [-] Count outstanding results per session — not needed for the measured
      cases. Known limit: a steer sent in the instant the CLI is finishing
      (before its `ResultMessage` reaches us) can still start a turn of its
      own; not reproduced
- [x] Offline test: steer on `StreamEnd` raises, adapter never called
- [x] e2e: late steer refused and turns stay in step; steered turn leaves
      later turns in step (both harnesses)
- [x] Checks pass

## H9. Different default permissions per CLI · P2

**Where:** `_adapters/claude.py` ~L498 (`bypassPermissions` with no hook);
`_adapters/codex.py` ~L200 (`read-only` on Local), ~L316
(`approvalPolicy="never"` with no hook).

**Problem:** with no hook on `Local`, Claude can do anything on your
machine; codex can't write at all. Docs claim swapping is "a one-word
edit". Allow-everything on the user's machine is a risky library default.

**Fix approach:** one shared `permissions=` option on both factories:
`"read-only" | "workspace-write" | "full"`, mapped to Claude
`permission_mode` / tool rules and codex `sandbox`. Default
`"workspace-write"` on `Local`, `"full"` on provider-owned workspaces. Keep
codex `sandbox=` as an advanced override.

**Decision (user):** full access ("yolo") by default on every harness; no
new `permissions=` option.

**Tasks:**

- [x] Codex: `sandbox` defaults to `"danger-full-access"` everywhere (was
      `read-only` on `Local`); explicit `sandbox=` still restricts on Local
- [x] Codex: resume and fork now apply the harness's sandbox / approval
      policy / model / instructions too (`_thread_params`); before, a resumed
      thread kept the policy it was created with
- [x] Claude: unchanged (already bypassPermissions without a hook)
- [x] Test helpers: `writable=True` no longer translates to
      `sandbox="workspace-write"`; `examples/harnesses/handoff.py` drops it
- [x] e2e: `test_every_harness_can_write_by_default` (any harness × any
      workspace)
- [x] Docs (H15): full-access paragraph, codex `sandbox` default, skill
- [x] Verified in a sandbox: full matrix 219 passed / 2 skipped (session 4)
- [x] Checks pass


## H10. Approval hook is a weak gate · P2

**Where:** `_approval.py`, `_capabilities.py` (`approval="policy"`),
`workspaces/experimental/_base.py` `contains()` (~L191),
`examples/harnesses/tool_approval.py`, `docs/.../basics/harnesses.mdx`.

**Problems:**

- Read-only tools never reach the hook, so reads outside the workspace
  can't be blocked.
- Doc/example hook only checks `file_path` → any `Bash` command is allowed.
- `contains()` is string-based; symlinks bypass it.
- `Allow(input=...)` is Claude-only; codex drops `Deny` reasons.

**Tasks:**

- [x] Rewrite the doc + example hook to handle shell tools (`Bash`,
      `commandExecution`) explicitly; say plainly what reaches the hook
- [x] Investigate Claude `PreToolUse` hooks. Findings (claude 2.1.289,
      claude-agent-sdk 0.2.160): a PreToolUse hook sees EVERY call (Read, a
      read-only Bash, Write) where `can_use_tool` saw only Write. Decisions
      map 1:1: `permissionDecision` allow / deny + reason (agent reads it) /
      allow + `updatedInput`; `continue_: False` ends the turn (3.4s). A hook
      "allow" is the permission: `can_use_tool` is not asked again. The CLI
      times hooks out at 60s by default (`HookMatcher.timeout`)
- [x] Investigate codex. Findings (codex 0.160.1, `approvalPolicy=untrusted`,
      `workspace-write`): every shell command (`cat`, `ls` included) and every
      `fileChange` reached the hook. MCP tools and web search not verified
- [x] Wire it: claude's hook is now a PreToolUse hook (timeout =
      `approval_timeout` + 30s so ours denies first; `can_use_tool` kept as a
      safety net with the same judgment, `_judge`); claude claims
      `approval="all"`. Codex keeps `"policy"` until MCP/web search are
      verified
- [x] Add async `workspace.resolve(path)` (base: one `exec`; `Local`:
      direct); document `contains()`'s symlink limit
- [-] Optional: small built-in policy helpers — not wanted for now (user:
      keep it simple)
- [x] Tests: `tests/workspaces/experimental/test_resolve.py` (Local and the
      exec path); e2e `test_what_the_hook_sees_matches_what_is_claimed`,
      `test_a_harness_claiming_every_call_shows_its_hook_every_call`
- [x] Claude local live suite passed (85) after the switch
- [x] Checks pass

## H11. Depends on undocumented CLI internals · P2

**Where:**

- Claude: `prepare_tui()` (~L330: onboarding / trust / API-key approvals in
  `~/.claude.json`), `_cli_sessions()` (~L425:
  `<config>/sessions/<pid>.json`, `procStart`), `process_ids` and
  `_publish_pid` (private `client._transport._process`).
- Codex: `prepare_tui()` (`config.toml` trust, `npm view` + reinstall),
  `_held_elsewhere()` (`~/.codex/thread-writer-locks/<id>.lock`),
  `_cli_sessions()` (argv parsing), `stage_history()` (stops the server to
  release the lock).
- Sandbox installs the latest CLI on every new VM.

**Problem:** any CLI release can break these quietly: TUIs stuck on a
dialog, wrong `running` flags, hangs.

**Decision (user):** install the latest CLI by default and document it; no
pinning. (Scheduled CI against latest CLIs: not requested.)

**Tasks:**

- [x] Add per-adapter `VERIFIED_VERSION` (claude 2.1.289, codex 0.160.1,
      exposed as `adapter.verified_version`); `Harness.open()` warns when
      the CLI is OLDER
- [x] Include the CLI version in errors: every `HarnessError` out of a turn
      gets a PEP 678 note `"<harness> version: <version>"` (`_drive`)
- [x] Latest by default: `base.provision()` — on a provider-owned workspace
      install when missing AND upgrade when npm has a newer release (found:
      the sandbox image ships claude 2.1.283 preinstalled, so "install only
      when missing" never gave the latest). Never upgrades while a copy of
      the CLI is running there (a detached agent). Never touches your own
      machine. Codex `prepare_tui`'s own upgrade removed (start does it)
- [-] Pinning — not wanted (decision above)
- [-] Scheduled CI job — not requested
- [x] Fail soft on internal-format reads: claude `prepare_tui` leaves an
      unexpected `~/.claude.json` alone and warns; `_cli_sessions` returns
      nothing on failure. Known limit, accepted: codex `_held_elsewhere`
      cannot tell a moved lock file from "not held" (nothing to detect)
- [x] Offline tests: `test_provision.py` (own machine probed only; missing
      or older → latest; latest → no-op; running → no-op; no registry → no-op)
- [x] Docs (H15): verified versions, warning, error note, "latest in a
      sandbox, never on your machine, not under a running CLI"
- [x] Verified in a sandbox: full matrix 219 passed / 2 skipped (session 4)
- [x] Checks pass


## H12. Too many network round trips in a sandbox · P3

**Where:** `_session_lock.py` `holders()` (~L119): `ls`, then per lock
`read_text` + maybe `exists` + `python3` liveness exec. Claude
`_cli_sessions()`: `ls` + per-file `read_text` + process list.
`Harness.sessions()` calls both.

**Problem:** ~3N+ remote calls to list N sessions; polling apps (afk) feel
slow.

**Tasks:**

- [x] Fake test: count workspace calls; finding running sessions costs a
      fixed number regardless of N (`test_listing_holders_is_one_command`,
      `test_running_cli_sessions_are_found_in_one_command`)
- [x] One Python helper per question, run with a single `exec`: lock
      holders (`_session_lock.HELPER`), claude's CLI session records +
      liveness + start time (`claude._CLI_SESSIONS`). `running_sessions()`
      is now 2 commands total on claude, 2 on codex (holders + `ps`)
- [-] Write the helper once per workspace open — dropped: the scripts are a
      few hundred bytes passed with `python3 -c`; writing them first would
      cost a round trip of its own
- [x] Reuse the helper for H6's atomic lock ops
- [x] Check `list_sessions()` through `WorkspaceSessionStore`: measured
      1 exec + 1 read PER transcript (the SDK loads each to summarize it).
      Now `list_session_summaries()` folds summaries from ONE batch read
      (`_SCAN` helper) with the SDK's own `fold_session_summary`, and
      `list_sessions()` uses the same helper (also drops GNU-only
      `find -printf`, so their mtimes agree): listing is ≤2 round trips,
      same results as the slow path (`test_listing_stored_sessions_is_a_
      fixed_number_of_round_trips`). `history()` was already 2 (list + load)
- [-] Measure wall-clock before/after on a sandbox — replaced by counted
      round trips in offline tests (deterministic); the sandbox e2e matrix
      exercises the new paths
- [x] Checks pass

## H13. `history()` has no incremental read · P3

**Where:** `Harness.history(session_id, limit, offset)`; Claude store reads;
codex `thread/read` (always full, then sliced in Python).

**Problem:** followers re-read the full transcript every poll.

**Decision (user):** no new API. Share the existing surface: load a
conversation (`resume()`), then run/stream on it like a new session;
`history(limit=, offset=)` stays the way to page a stored one.

**Tasks:**

- [-] `follow()` / `history(after=)` — not wanted (decision above)
- [x] Verified `limit`/`offset` count the same messages on both CLIs, tool
      turns included (suspected claude counted store records; it does not):
      e2e `test_pages_count_messages_when_tools_were_used`
- [x] e2e `test_a_loaded_conversation_streams_like_a_new_one`: `resume()`
      → `session.messages` equals the stored history; a streamed turn lands
      after it, in order (both harnesses × both workspaces)
- [x] Docs (H15): resume/fork return a session with its past; paging
- [-] afk `peek` could use `offset=` to draw only new messages — moved to
      the afk bucket (not part of this plan)
- [x] Checks pass


## H14. Text-only input; tools/MCP not configurable · P3

**Where:** `Session.run/stream(prompt: str)`; adapters send
`{"type": "text"}` only; `capabilities.images=True` but unused.

**Decision (user):** accept every input kind the harness supports; expose
basic day-to-day options (not comprehensive).

**Tasks:**

- [x] Prompt is `str | Sequence[str | TextPart | FilePart]` on `run()`,
      `stream()`, `Harness.run()` (`_input.py`: normalize, check, bytes)
- [x] Measured what each CLI takes: claude base64 image ✓, base64 PDF ✓,
      image by URL silently dropped ✗ → URLs are downloaded on the host
      (SDK's SSRF-safe downloader) and sent as data; codex `UserInput`:
      text, image (`url`, data URL used), audio (declared, not verified →
      not claimed)
- [x] Claude: images + PDFs as base64 content blocks (`documents=True`);
      codex: images as data URLs; text files inlined on both
- [x] Unsupported media → `UnsupportedError` in `stream()`, before anything
      runs or is recorded
- [x] History: claude image/document blocks and codex image inputs come
      back as `FilePart`s; handoff carries images (both) and PDFs (claude),
      warns about anything left out (`warn_unsupported_files`)
- [x] Structured output on a file prompt: schema goes onto the last text
      part (or a new one)
- [x] Options, shared: `instructions` (claude `--append-system-prompt` /
      preset append; codex `developerInstructions` on start/resume/fork and
      `-c developer_instructions` for the TUI), `mcp_servers` (claude's
      shape; codex `-c mcp_servers.*`). Claude only: `tools`,
      `disallowed_tools`, `max_turns`. TUI launches carry them too
- [x] Handle stores instructions/tools/disallowed_tools/max_turns, never
      `mcp_servers` (env may hold tokens); `harness_from(mcp_servers=...)`
- [x] Offline tests: `test_input.py`, `test_options.py`
- [x] e2e `test_input.py` (image incl. history, PDF / refusal, text file)
      and `test_options.py` (instructions, MCP server, `tools`, `max_turns`):
      12 passed locally on both CLIs
- [x] Docs (H15): "Send files", options tables, capabilities, reference,
      skill
- [x] Verified in a sandbox: full matrix 219 passed / 2 skipped (session 4)
- [x] Checks pass


## H15. Docs fixes · P2 (ongoing)

Files: `docs/ai-python/content/docs/basics/harnesses.mdx`,
`docs/ai-python/content/docs/reference/harnesses.mdx`,
`docs/ai-python/content/docs/basics/workspaces.mdx`,
`skills/ai/references/harnesses.md`, `CHANGELOG.md`.

**Tasks** (tick as each fix lands):

- [x] H1: `result.messages` order; tool-turn handoff works
- [x] H2: `stop()` is safe from any task, returns once settled
- [x] H5: what `break` does now
- [x] H9: full access by default on every harness
- [x] H10: approval example handles shell tools; what reaches the hook
      (claude: all); symlinks via `resolve()`
- [x] H11: verified versions; latest CLI in a sandbox, documented
- [x] H13: resume-then-stream; paging counts messages
- [x] H14: "Send files", options
- [-] `CHANGELOG.md` entry — dropped: the file only points to GitHub
      releases; write the release notes from the session log instead
