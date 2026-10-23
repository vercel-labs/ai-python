# afk improvements

What makes `afk` (examples/apps/afk) work for real "step away" use. One
section per item, decisions from the user, a checklist per item. Same rules
as `plans/harness-sdk-fixes.md`: tick boxes as they land, log each session,
verify everything for real (live sandboxes, real CLIs, a pty for TUIs).

Markers: `[ ]` open · `[~]` in progress · `[x]` done · `[-]` dropped (say why).

## Progress

- [x] **A0** SDK: stop leaking sandboxes; stopped means resumable
- [x] **A1** `--bg` takes an optional prompt instead of "continue"
- [x] **A2** Snapshots: an expired or stopped sandbox is listed and resumable
- [x] **A3** Know what happened without a notification service
- [x] **A4** Network open by default; `--no-network` locks it
- [x] **A5** Work on a branch; pull through git; `--git-history`
- [x] **A6** codex: honest status, no recaps
- [x] **A7** Quicker and cheaper everyday use
- [x] **A8** Live tests load credentials; cost of each sandbox, accurately

## Checks

```bash
uv run ruff format --check src tests examples
uv run ruff check src tests examples
uv run mypy src tests
uv run ty check
uv run pytest -q
uv run examples/.test_scripts/check-examples.py   # read its EXIT CODE
cd examples/apps/afk && uv run --frozen --with-editable ../../.. pytest -q
```

Live: load `.env.local` first (`set -a; . ./.env.local; set +a`); the
`VERCEL_OIDC_TOKEN` there lasts ~12h (`vercel env pull` refreshes it).

## Session log

- Session 1 (planning): measured what the platform does before designing.
  - Sandboxes are `persistent` by default: stopping one (or its time limit)
    snapshots it (≈0.8 GB) and any later command resumes it, files intact,
    in ~2s. The SDK said a stopped sandbox's "files are gone" and afk then
    forgot it: that is where "lost work" came from.
  - The SDK's `close()` STOPS a sandbox it created; with persistence every
    such close leaves a stopped sandbox and a 30-day snapshot. The user's
    project had 1,320 stopped sandboxes / 1,324 snapshots, 1,309 GB,
    ≈$105/month at Pro snapshot rates — mostly e2e test runs.
  - A stopped session reports `duration`, `activeCpuDurationMs`,
    `networkTransfer {ingress, egress}`, `vcpus`, `memory`; a running one
    only `startedAt`. The vercel-sandbox models drop these fields; the raw
    API response has them.
  - Pricing (docs, 2026-10): Active CPU per hour of CPU actually used,
    memory per GB-hour provisioned (1-minute minimum), network = data SENT
    (downloads free), creations $0.60/1M, snapshot storage $0.08/GB-month
    flat. CPU/memory/network rates vary by region (19 regions; iad1:
    $0.128, $0.0212, $0.15).
  - A custom network policy `{gateway: inject, "*": ()}` allows every host
    AND keeps the gateway key injection (pypi, github, gateway all 200).
- Session 2 (build + verify): A0–A8 done.
  - SDK: `close()` destroys; `stop()`, `info()`, `destroy()`; reconnecting
    resumes a stopped sandbox; 404 counts as gone (destroyed) and says what
    to do instead; `provision()` asks npm at most every 12h per VM and
    retries an install whose answer was lost (a cut-off response stream,
    seen under load).
  - afk live, on a scratch project with a pty driver: push / list with
    cost / stop (snapshot kept, cost) / list stopped / peek resumes and
    stops again / attach resumes into the TUI on the same conversation /
    rm (platform: gone, 0 snapshots) / expiry → `expired`, cost by hand =
    afk's ($0.01087) / `--prompt` acted on / open network (pypi, github
    200) vs `--no-network` (refused, model still reached) / git branch home,
    checkout untouched / `--git-history` (agent read `git log`) / `wait`
    rings when done / list marks `← new, was running` once / codex TUI with
    no turn reads as waiting, not "not running".
  - Found while verifying: the sessions API caps `limit` at 50; a timeout
    test assumed a model could not type 2000 numbers in 10s (now one can;
    codex declines 20000), so the deadline is 4s; a full matrix crawled
    while a 1,320-sandbox listing ran against the same account.
  - Final: sandbox matrix 225 passed; local live 221 passed / 6 skipped;
    offline 1145; afk 116 (live included); deepysec 67 (live, fixture);
    all checks pass.
- Session 3 (cleanup, user approved):
  - Test project: destroyed 1,322 stopped sandboxes (0 failed, 111s) and
    the one orphan snapshot left by an experiment: 0 sandboxes, 0
    snapshots (was 1,309 GB, ~$105/month).
  - afk's own project: 52 stopped sandboxes, 52 snapshots, 45.2 GB
    (~$3.60/month) — none created by this work (scratch runs used the test
    project via `.env.local`); 2 back the user's real afk conversations.
    Left untouched; the user's to decide.
  - Local: removed 699 claude project dirs and 693 codex rollouts for temp
    directories that no longer exist (since 2026-10-09); a stale SDK lock;
    `/tmp/hx` (incl. the two `.env.local` backups).
  - Found an SDK leak while cleaning: 8 pty holders still running codex
    TUIs from local e2e runs, alive since Oct 9. Cause: runs under `nohup`
    ignore SIGHUP, the child inherits that across exec, and the holder
    ended programs with SIGHUP only. The holder now restores default
    signal handling before exec, and follows a hangup with SIGTERM, then
    SIGKILL. It also removes its private socket directory (239 empty
    `harness-pty-*` dirs were left). Verified: the same nohup run leaves 0
    holders, 0 dirs.

---

## A0. SDK: stop leaking sandboxes; stopped means resumable

**Problem:** `VercelSandbox.close()` stops a sandbox it created, which on a
persistent sandbox keeps it and a snapshot for 30 days, billed. Reconnecting
to a stopped sandbox raises `WorkspaceGoneError` ("its files are gone"),
which is false: it resumes from its snapshot.

**Design:**

- `close()` on a sandbox not kept: **destroy** it, its snapshots with it.
- New `await sandbox.stop()`: stop the VM now and keep its files (a
  snapshot); `VercelSandbox(name=...)` later resumes it.
- `VercelSandbox(name=...).open()` on a stopped sandbox: resume it (files
  back, processes not). `WorkspaceGoneError` only when it no longer exists
  or cannot be resumed.
- New `await VercelSandbox(name=...).info()`: status, region, size,
  snapshot, and usage per session, WITHOUT resuming it (a listing must not
  boot VMs).

**Tasks:**

- [x] Offline tests with a fake vercel API: close destroys; stop stops;
      open resumes a stopped one; info does not resume
- [x] Implement `close()` → destroy (orphan snapshots too)
- [x] Implement `stop()`
- [x] Implement resume-on-open for a stopped sandbox
- [x] Implement `info()` → `SandboxInfo` (status, region, vcpus, memory,
      snapshot id/size/created/expires, sessions with usage)
- [x] e2e: close leaves nothing (404, no snapshot); stop + reopen has the
      files; info on a stopped sandbox leaves it stopped (6 passed live;
      found the sessions API caps `limit` at 50)
- [x] Update tests that expected "gone" after a stop
- [x] Docs: workspaces guide + reference (lifecycle, cost note)
- [x] Tell the user how to clean up what already leaked (do not delete
      without asking): `/tmp/hx/cleanup_leaked_sandboxes.py` lists by
      default, deletes only with `--delete`, skips running ones and any
      afk still records. Dry run: 1,320 stopped sandboxes, 1,324 snapshots,
      1,309 GB (~$105/month) — unchanged after two full matrices with the
      fix, so nothing new leaks
- [x] Checks pass

## A1. `--bg` takes an optional prompt

**Decision:** optional; not a task, a way to override the "continue" prompt.

**Tasks:**

- [x] `afk push --bg [--prompt TEXT]`; default stays the continue prompt
- [x] Record the prompt sent, so status/peek find where the turn began
      (today they look for the continue text)
- [x] Offline test: the prompt reaches the session; finished/running
      detection works with a custom prompt
- [x] Live: push --bg --prompt "create x" → the agent does it; peek and the
      list show finished
- [x] README
- [x] Checks pass

## A2. Snapshots: listed, with status, resumable

**Decision:** use snapshots; no auto pull. Snapshots show up in the list
with their status, and a session can start again from one in a sandbox.

**Design:** a pushed sandbox that stopped (expired, or `afk stop`) is listed
as `stopped · snapshot 0.8 GB · kept until <date>`. `attach` resumes it (a
new VM from the snapshot) and reopens the TUI on the same conversation;
`pull` resumes it to bring the conversation/branch home. `afk stop` stops
and keeps the snapshot; new `afk rm` destroys the sandbox and its snapshots
(stops storage cost). Only a sandbox the platform no longer has is "gone".

**Tasks:**

- [x] List uses `info()`: running / stopped (snapshot, expiry) / gone,
      never resuming to list
- [x] `attach` on a stopped sandbox resumes it and reopens the TUI
- [x] `pull` on a stopped sandbox resumes it
- [x] `peek` on a stopped sandbox: resumes it (the transcript lives there)
      and says so
- [x] `afk stop` stops and keeps the snapshot; `afk rm` destroys
- [x] Offline tests (rows from info, verbs pick the right path)
- [x] Live: push, stop, list shows stopped+snapshot, attach resumes with the
      conversation and files intact; expire (short `--hours`) behaves the
      same; rm leaves nothing (platform: sandbox gone, 0 snapshots)
- [x] README
- [x] Checks pass

## A3. Know what happened, no infrastructure

**Decision:** no webhooks; check when the CLI runs, or poll.

**Tasks:**

- [x] `afk` (list) says what changed since you last looked: finished,
      stopped/expired, waiting for input
- [x] `afk wait [id]`: poll until a pushed conversation (or any) finishes,
      stops, or needs input; exit code says which; rings the terminal bell
- [x] Offline tests; live run
- [x] README
- [x] Checks pass

## A4. Network open by default

**Decision:** networking allowed by default unless a flag disables it.

**Tasks:**

- [x] Push creates the sandbox with every host allowed (gateway key still
      injected, never in the VM)
- [x] `--no-network`: only the gateway (and npm, to install the CLI)
- [x] Live: open → pypi/github 200 in the sandbox; `--no-network` → 000;
      both agents still reach the model
- [x] README (what open means, the flag)
- [x] Checks pass

## A5. A branch, git for the round trip, `--git-history`

**Decision:** work on a branch; pull and resolve through git; a flag to
include git history.

**Design:** in a git repo, push records your working tree (uncommitted
changes included) as a commit on top of HEAD, without touching your
branch or index, and the sandbox starts from that tree on branch
`afk/<label>`. Without `--git-history` the sandbox repo has that one commit;
with it, the full history (bundle). `pull` commits what is left
uncommitted there, brings the commits back as a local branch `afk/<label>`
based on the recorded commit, and you merge it with git (conflicts are
git's). Not a git repo: today's hash-based `pull --files` stays.

**Tasks:**

- [x] Push: base commit (temp index, `commit-tree`), sandbox repo + branch;
      `--git-history` ships history
- [x] Pull: commit leftovers there, patches/bundle back, local branch
      `afk/<label>`; never touches your checkout
- [x] Offline tests with real git in temp dirs (both modes; your
      uncommitted edits survive; conflict shows as a git conflict)
- [x] Live: push, agent edits, pull → branch has the edit; merge; with
      `--git-history` the agent can `git log`
- [x] README
- [x] Checks pass

## A6. codex: honest status, no recaps

**Decision:** no recaps; if codex does not load a history natively, do not
fake it.

**Tasks:**

- [x] A pushed codex TUI with no turn yet reads as "no turn yet", not "not
      running"
- [x] Live check
- [x] Checks pass

## A7. Quicker and cheaper

**Measured** (9 pushed sandboxes, iad1): list 3.5-3.7s; one peek poll
(`history()`) 0.35s, `sessions()` 0.75s; a push: boot 0.9s, copy 0.2s, CLI
upgrade 5.7s, harness open 0.7s, first turn 4.5s (the model).

**Tasks:**

- [x] Reconnects stop asking npm every time: `provision()` remembers in the
      VM when it last checked (12h). List 3.6s → 3.0s (remote rows 2.83s →
      2.43s); offline tests
- [x] List: one `info()` per sandbox, in parallel, no VM resumed (A2)
- [-] `peek` reads only new messages — measured, not worth it: the store
      reads the whole transcript file in one round trip either way (0.35s)
- [-] Prepared snapshot with the CLIs for new pushes — measured: boot from
      snapshot 2.4-4.4s + provision 0.4s vs fresh 0.9s + upgrade 5.7s;
      saves 2-4s of a ~12s push, costs a 1.4 GB snapshot ($0.11/month),
      a rebuild per CLI release (24s), and a lifecycle to manage. The 5.7s
      exists only because the image ships an old claude; a current image
      makes a fresh push ~1s there. Not built
- [x] Stopped sandboxes are no longer resumed to be listed, and peek/pull
      stop what they resumed: nothing runs (and bills) by accident
- [x] Checks pass

## A8. Credentials in tests; cost of each sandbox

**Decision:** fix the test credentials; show each sandbox's cost so far,
accurately.

**Tasks:**

- [x] afk's live tests load `.env.local` like the SDK suite
- [x] Cost per sandbox from its sessions' usage and its snapshot, at the
      region's rates: exact for stopped sessions; for a running one,
      memory so far (CPU and data sent are reported when it stops)
- [x] Rates table in afk with its source and date; unknown region → say so
- [x] Shown in the list and on `stop`/`rm`; first push says sandboxes
      bill per use and claude via the gateway bills per token
- [x] Offline tests of the arithmetic against the docs' worked examples
- [x] Live: a stopped sandbox's cost matches the formula on its numbers
      (expired one: $0.01087 by hand, $0.01087 by afk)
- [x] README
- [x] Checks pass
