## Harnesses and workspaces

`ai.harnesses.experimental` drives coding-agent CLIs. Adapters live in
`src/ai/harnesses/experimental/_adapters/` (`claude.py` on
`claude-agent-sdk`, `codex.py` on `codex app-server`). Workspaces live in
`src/ai/workspaces/experimental/` (`_local.py`, `_sandbox.py`).

Rules that apply to both:

- **Layering.** Caller, then `Harness`, then adapter, then workspace. Each
  layer knows only the one below. A workspace never imports harness names
  and never learns a CLI's dialect. A new CLI touches one adapter; a new
  place to run touches one workspace.
- **Capability honesty.** Every `Capabilities` field is a claim you watched
  be true. When a capability is absent, raise `UnsupportedError`. Never a
  silent no-op.
- **Translate errors at the boundary.** Callers never catch a CLI library's
  or provider's exceptions. Map them to `ai.harnesses.experimental.errors`
  or `ai.workspaces.experimental.errors`.

### Adding a harness

1. Pick the CLI protocol first. It must offer, in one mode: a long-lived
   process or resume by id, input while a turn runs (steer), an interrupt,
   and a way to ask the client about tool calls. Codex had to move from
   `codex exec --json` to `app-server` for this.
2. Implement the `Adapter` protocol in `_adapters/base.py`. `new_session()`
   returns the CLI's own id, never one you mint.
3. Never spawn a process yourself. Call `workspace.spawn(argv)`. If the CLI
   library insists on spawning, give it a transport backed by the workspace
   (see `_adapters/remote_cli.py`).
4. Yield `ai.types.events` only, ending with `StreamEnd`. Text, reasoning,
   and tool calls need Start / Delta / End triples. Emit tool arguments as a
   `ToolDelta`, even when the call arrives complete. Tool results are
   `ToolCallResult` events.
5. `approve is None` means approve everything. Answer the CLI's approval
   requests yourself; never fall back to asking a terminal that is not there.
   Exceptions inside a permission callback do not reach the caller: park them
   on the adapter and re-raise from `turn()`.
6. Hold the workspace session lock (`_session_lock.SessionLock`) for every
   conversation you open, so a second writer gets `SessionBusyError` on every
   CLI.
7. Store `history=` natively with the rules in `_handoff.py`: text verbatim,
   tool calls with their original names, reasoning as a note or dropped with
   a warning. Never drop context silently.
8. Spell the gateway in the adapter (`gateway_env`, `gateway_config`). When
   `workspace.injects_credentials_for(gateway.host)`, hand the CLI
   `gateway.brokered()`.
9. Add a factory in `_factories.py` (and to `BUILDERS`), then add the kind to
   `SPECS` and `HARNESS_CLI` in `tests/harnesses/experimental/conftest.py`.
   That runs the whole e2e suite against the new adapter.

### Adding a workspace

1. Subclass `Workspace` from `_base.py`. Set `kind`, `owner` (`"provider"`
   only for a box the SDK may install software into), and `duplex_spawn`.
   `env` belongs to the work, not the agent.
2. Do not override `contains()`. It is pure-path on purpose: a remote path
   does not exist on this machine.
3. Move files through `read_text`, `write_text`, `_copy_in`, and `_copy_out`.
   Filter both directions with `walk_uploadable` and `copy_tree`. Batch
   uploads; round trips are the cost on a remote box.
4. Map failures: a program that cannot run is `exit_code=127` from `exec`
   and `WorkspaceError` from `spawn`; a missing file is `FileNotFoundError`;
   a workspace that no longer exists is `WorkspaceGoneError`.
5. `Process.terminate()` returns only when the program is gone, and raises
   when it could not make that true.
6. If the workspace can outlive the client, support reconnect by `name=`,
   carry that identity in `coords`, and add a `keep` flag. Live state is
   found by asking a harness inside the workspace, never by the workspace.
7. For ptys, reuse `_pty_holder.py` and `_pty.FramedPty` over your byte
   channel. Never name a module `pty.py`.
8. Export it from `src/ai/workspaces/experimental/__init__.py`, and add it to
   the `any_workspace` fixtures in `tests/workspaces/experimental/conftest.py`
   and `tests/harnesses/experimental/conftest.py` as
   `pytest.param("yours", marks=pytest.mark.yours)`. It must present the same
   `project` the local branch does.

### Tests

Live tests drive real CLIs and real sandboxes. No fakes, no recorded doubles.
A missing CLI skips; it never passes silently. Markers:

- `live`: drives a real local `claude` or `codex`.
- `sandbox`: needs Vercel Sandbox credentials.
- `slow`: minutes, not seconds.

```bash
uv run pytest                                               # offline (default)
uv run pytest tests/harnesses tests/workspaces -m "not sandbox"  # plus local CLIs
uv run pytest tests/harnesses tests/workspaces -m ""        # everything
```

Plain `uv run pytest` deselects `live` and `sandbox` (see `addopts` in
`pyproject.toml`); passing `-m` replaces that filter.

The live suite needs `claude` and `codex` CLIs that can reach a model: a
login, or the gateway variables in `.env.local`. The sandbox suite
also needs Vercel credentials and `AI_GATEWAY_API_KEY` in a `.env.local` at
the repo root (`vercel link`, then `vercel env pull`). Every test is capped at
300 seconds.

When writing tests, never branch on native tool names, wait on events instead
of sleeping, assert exact answers, and put anything on the control surface
on `any_harness` so it runs on both harnesses and both workspaces.

Run the harness examples with:

```bash
uv run examples/.test_scripts/run-examples.py --harnesses
```

## Releasing

Every PR needs to be labeled as `breaking`, `feature`, `fix`, or
`internal`. Release notes are generated from those labels.

1. Draft the release:

   ```bash
   gh release create vX.Y.Z --generate-notes --draft
   ```

2. Review the draft on GitHub and add migration notes under **Breaking Changes**.
3. Publish the release. This creates the tag, which triggers
   `.github/workflows/publish.yml` and uploads the package to PyPI.

GitHub release notes are the changelog of record; `CHANGELOG.md` only holds
a pointer.
