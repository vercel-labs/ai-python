# deepysec — deepsec, rebuilt on ai.harnesses

[vercel-labs/deepsec](https://github.com/vercel-labs/deepsec) is an
agent-powered vulnerability scanner: cheap regexes decide *what to look at*,
expensive coding agents decide *whether it is real*. Of its 36k lines, about
a quarter exists only because there was no library to drive the agents —
three vendor adapters, an OS sandbox around the CLI, a work queue, JSON
repair loops, a tarball-upload-bootstrap-snapshot-download pipeline for
running in microVMs.

This example rebuilds the same funnel, stage for stage, and hands that
quarter to `ai.harnesses`. It is an MVP: the flow is deepsec's, the data on
disk is deepsec's, the feature surface is the minimum that shows the shape.

```
scan ──> process ──> revalidate ──> enrich ──> export
(regex)  (agent)     (agent)        (git)      (json/md)
```

Five idempotent stages over one append-only store. Each reads the store,
adds to it, writes it back; none overwrites another's output. Re-running a
stage merges, so a run survives Ctrl-C, a crash, or an empty balance, and
picks up where it stopped.

## Run it

```bash
cd examples/apps/deepysec
uv run python -m deepysec all --root path/to/repo
```

That is scan → process → revalidate → enrich → export against a repository,
with your own `claude` login, and a report in
`path/to/repo/.deepsec/data/<repo>/reports/`. A small deliberately vulnerable
app to try it on is not part of this repository; the harness-sdk repository
keeps one in `examples/deepysec/fixtures/vulnerable-app`. Each stage
is also its own command, with deepsec's flags where they still mean
something:

```bash
uv run python -m deepysec scan       --root path/to/repo
uv run python -m deepysec process    --root path/to/repo --agent codex --batch-size 3 --concurrency 4 --limit 12
uv run python -m deepysec revalidate --root path/to/repo --min-severity HIGH
uv run python -m deepysec enrich     --root path/to/repo
uv run python -m deepysec export     --root path/to/repo
uv run python -m deepysec status     --root path/to/repo
```

`--agent claude` (default) or `--agent codex`; `--model` picks the model.
Both drive your own installation and its login; `--gateway` routes either
through the Vercel AI Gateway instead, with `AI_GATEWAY_API_KEY`. Records are per file, so a file processed
by claude and then by codex carries both analyses and one merged findings
list.

Run the agents in Vercel Sandboxes instead of on your machine:

```bash
uv run python -m deepysec process --root path/to/repo --sandboxes 3 --concurrency 6
```

Needs `AI_GATEWAY_API_KEY` (read from your environment or a `.env.local`
above the current directory) and the Vercel credentials `vercel env pull`
writes. Nothing else changes: the store stays on this machine, and so does
the program.

## What the SDK absorbed

The spec this was built from tags every part of deepsec **[PRESERVE]**,
**[REPLACE]** or **[JUDGMENT]**. Everything **[PRESERVE]** is here in
`deepsec/`. Everything **[REPLACE]** became one of these calls:

| deepsec                                                   | here                                                                                     |
|-----------------------------------------------------------|------------------------------------------------------------------------------------------|
| three SDK adapters behind `AgentPlugin` (3,141 lines)      | `claude_code(...)` / `codex(...)`; `--agent` picks the factory, nothing else changes      |
| read-only tool permissions (`Read`, `Glob`, `Grep`, `Bash`)| an `approve=` hook: `Deny` for writes and network, `Allow` for the rest (`agents.py`). The hook sees each harness's own tool names — Claude's `Bash`, codex's `commandExecution` — and judges the shell command either way |
| JSON output contract + three repair loops, `MAX_ATTEMPTS=3`| `session.run(prompt, output_type=list[FileReport], retries=2)` — the schema is the contract, the SDK re-asks |
| `accepted-risk` must never be agent-emitted               | it is not in `Verdict`'s type, so the schema forbids it and a reply carrying it is re-asked |
| per-call cost and token usage                             | `result.usage`; claude's dollar figure rides in `usage.raw["cost_usd"]`                  |
| turn budget (150 turns)                                   | `timeout=` per batch (`--max-duration`)                                                  |
| `--thinking-level` → per-vendor effort mapping             | `effort=` on both factories, one scale                                                   |
| quota exhaustion aborts every in-flight batch             | `NotAuthenticatedError` / `TurnFailedError` classified once; `session.stop()` on the others |
| OS sandbox + env allowlist around the CLI                 | run in a `VercelSandbox` instead — the microVM is the boundary                            |
| credential brokering: placeholder in the VM, header injected at egress (§9.2) | `VercelSandbox(gateway=gw)`; the harness inside is handed the placeholder automatically |
| tarball upload, bootstrap snapshot, in-VM worker CLI, result download + merge (3,049 lines) | `copy(root, sandbox / "project")` and the same `process()` loop; the harness runs in the VM, the program and the store do not |
| scan honours `.gitignore`                                 | `walk_uploadable` — the SDK's own (private) walk, the one `copy` uses                    |

The single biggest simplification is the last row. deepsec had to ship
*itself* into the VM, run a copy of its CLI there against a manifest, and
pull records back through an allowlist, because the agent SDKs only worked
in-process. Here the agent process is in the VM and the Python calling it
is not, so partitioning is a round-robin over a list of workers and merging
is the same `store.save()` as the local path.

## What is preserved

- **Data model** (`models.py`): `FileRecord`, `Finding`, `AnalysisEntry`,
  `Revalidation`, `RunMeta`, written as camelCase JSON under
  `data/<projectId>/files/<path>.json` exactly as deepsec writes them.
  `finding_id` is the same `sha256(projectId, path, title)` prefix, so the
  same issue in the same file has the same id across runs, models and machines.
- **Merge rules** (`store.py`): re-scan unions candidates, re-process
  appends an `AnalysisEntry` and unions findings by (slug, title),
  revalidate tags findings and never creates them, enrich sets `gitInfo`.
- **Leases** (`store.py`): a file is claimable when `pending`, or when
  `processing` under a lock older than an hour whose run is done, errored or
  gone — or, on the same host, whose pid is dead, so a SIGKILLed run is
  reclaimed immediately rather than in an hour. Locks are released on
  completion, on failure, and on quota abort.
- **Cost attribution**: one agent call covers a batch; cost, tokens and
  duration are divided over the files that produced valid results and
  stamped per file. The sum of per-file `costUsd` is the run's `totalCostUsd`.
- **Prompts** (`prompt.py`): deepsec's `CORE_PROMPT`, slug notes, investigation
  instructions and revalidation prompt verbatim, composed in `assemble.ts`'s
  order; `--thinking-level` with its vocabulary and default. Framework
  highlights are left out of the MVP.
- **Matchers** (`matchers.py`): fifteen of deepsec's rules ported with their
  `examples`, and the discovery test that asserts every example fires.

## What was left out, and what to know

- **Credential brokering is the SDK's.** deepsec's strongest security
  property, the real API key never entering the VM, holds here too:
  `VercelSandbox(gateway=gw)` locks the VM's egress to the gateway host, the
  sandbox firewall writes the key into requests on the way out, and the
  harness inside holds a placeholder. The SDK's own tests read the VM's
  process table for the key and try to leave for other hosts.
- **Git history in sandboxes.** `copy` leaves `.git` behind, so a
  revalidation run with `--sandboxes` cannot check whether an issue was
  already fixed. `enrich` is host-only either way, as in deepsec.
- **Batching.** Kept, because cost division and partial results are part of
  what the spec preserves. `--batch-size 1` deletes that machinery, and with
  prompt caching it may be the right default; measure before deciding.
- Not ported: `triage`, `init`, `--diff`, plugins, notifiers, ownership, the
  refusal follow-up, framework highlights, 184 of the 199 matchers. All of
  them fit the same seams.

## Layout

```
deepysec/            (the package; on-disk state stays under .deepsec/)
  models.py      data model + finding ids           [PRESERVE]
  store.py       append-only store, merges, leases  [PRESERVE]
  matchers.py    matcher contract + 15 rules        [PRESERVE]
  scan.py        stage 1
  prompt.py      CORE_PROMPT + batch assembly       [PRESERVE]
  agents.py      the ai.harnesses seam              [was REPLACE]
  process.py     stage 2
  revalidate.py  stage 3
  enrich.py      stage 4
  export.py      stage 5 + status
  cli.py
tests/                     matcher examples, store invariants, and a live pipeline run
```

```bash
cd examples/apps/deepysec
uv run pytest -m "not live"   # offline invariants only
uv run pytest                 # everything, including the live run
```

The scan tests and the live run need the vulnerable app: point
`DEEPYSEC_FIXTURE` at it, or they skip.
