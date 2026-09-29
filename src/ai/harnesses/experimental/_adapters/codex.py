"""Codex, driven through `codex app-server`.

Chosen over `codex exec --json` because exec cannot be controlled: one
process per turn, stdin unused, no approval channel. app-server has
`turn/steer`, `turn/interrupt` and `item/*/requestApproval`, which is the
whole point of this SDK.

The protocol shapes here were learned by running the server and reading
what it sent, not from the schema alone.
"""

from __future__ import annotations

import asyncio
import contextlib
import importlib.metadata
import json
from typing import TYPE_CHECKING, Any

from ....types import events as events_
from ....types import messages as messages_
from ....types import usage as usage_
from ....workspaces.experimental import _base, _gateway
from ....workspaces.experimental import errors as workspace_errors
from .. import (
    _approval,
    _capabilities,
    _handoff,
    _session,
    _session_lock,
    errors,
)
from . import base, jsonrpc

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

KIND = "codex"
# Internal marker delivered on a turn queue when the process dies.
CRASHED = "__crashed__"
INSTALL = "npm install -g @openai/codex"

# Codex asks only when its own policy escalates — sandbox escapes, blocked
# network, MCP prompts — so coverage is "policy", like Claude's, for a
# different reason. `acceptForSession` is a real persist-this-decision.
CAPABILITIES = _capabilities.Capabilities(
    steer=True,
    stop=True,
    resume=True,
    fork=True,
    history=True,
    rewrite_tool_input=False,  # accept / acceptForSession / decline only
    # Measured: answering `acceptForSession` did NOT stop the next identical
    # fileChange being asked about. The protocol offers the verb; the
    # behavior did not follow, so the capability says false.
    remember_decisions=False,
    images=True,
    approval="policy",
)

APPROVAL_METHODS = (
    "item/commandExecution/requestApproval",
    "item/fileChange/requestApproval",
    "item/permissions/requestApproval",
    "execCommandApproval",
    "applyPatchApproval",
)

# Item types that are the agent doing something, rather than speaking.
TOOL_ITEMS = (
    "fileChange",
    "commandExecution",
    "mcpToolCall",
    "webSearch",
    "toolCall",
)


class CodexAdapter:
    kind = KIND
    capabilities = CAPABILITIES

    def __init__(
        self,
        executable: str = "codex",
        model: str | None = None,
        sandbox: str | None = None,
        config: dict[str, Any] | None = None,
        gateway: _gateway.Gateway | None = None,
    ) -> None:
        self._executable = executable
        self._model = model
        self._sandbox = sandbox
        self._sandbox_explicit = sandbox is not None
        self._gateway = gateway
        # Passed to app-server as `-c key=value`, the same overrides the
        # codex CLI takes. This is how you point it at a model provider
        # without writing ~/.codex/config.toml on the far side. A gateway
        # supplies the provider table; an explicit config= wins over it.
        self._explicit_config = dict(config or {})
        self._config = {**gateway_config(gateway), **self._explicit_config}
        self._gateway_env: dict[str, str] = gateway_env(gateway)
        self._version: str | None = None
        self._workspace: _base.Workspace | None = None
        self._process: _base.Process | None = None
        self._lock: _session_lock.SessionLock | None = None
        self._held: set[str] = set()
        self._peer: jsonrpc.JsonRpcPeer | None = None
        self._threads: set[str] = set()
        self._active_turn: dict[str, str] = {}
        self._queues: dict[str, asyncio.Queue[Any]] = {}
        self._usage: dict[str, usage_.Usage] = {}
        self._interrupted: set[str] = set()
        self._pending_interrupt: dict[str, dict[str, str]] = {}
        self._closed = False
        # An error raised while answering an approval cannot propagate out
        # of the JSON-RPC pump — nobody is awaiting it there. It is parked
        # and re-raised on the turn that caused it.
        self._pending_error: BaseException | None = None
        self.approve: base.ApprovalHook | None = None
        self.approval_timeout: float = 300.0
        self.on_approval_error: Any = None
        self.live_history: Any = lambda session_id: []

    @property
    def version(self) -> str | None:
        return self._version

    @property
    def process_ids(self) -> list[int]:
        pid = getattr(self._process, "pid", None)
        return [pid] if isinstance(pid, int) else []

    # -- lifecycle ------------------------------------------------------------

    async def start(
        self, workspace: _base.Workspace, options: dict[str, Any]
    ) -> None:
        if not workspace.duplex_spawn:
            # app-server is a stdio protocol. A workspace that cannot give a
            # process a writable stdin cannot host it, and pretending
            # otherwise would fail later and less clearly.
            raise errors.UnsupportedError(
                KIND,
                f"{workspace.kind} workspaces",
                "app-server speaks JSON-RPC over stdin, which this workspace "
                + "cannot provide",
            )
        # An explicit gateway wins; otherwise the workspace's. Where the
        # workspace injects the credential at egress, codex is handed the
        # placeholder — the provider table names an env var, and that var
        # carries a value the firewall overwrites on the way out.
        gateway = self._gateway or workspace.gateway
        if gateway is not None and workspace.injects_credentials_for(
            gateway.host
        ):
            gateway = gateway.brokered()
        self._gateway = gateway
        self._config = {**gateway_config(gateway), **self._explicit_config}
        self._gateway_env = gateway_env(gateway)
        if (
            self._sandbox_explicit
            and workspace.owner == "provider"
            and self._sandbox != "danger-full-access"
        ):
            # Measured in a Vercel Sandbox: codex's own sandbox runs every
            # command through bubblewrap, which cannot start inside the
            # microVM (bwrap reports unexpected capabilities). So
            # read-only and workspace-write fail every command, reads
            # included, and the turn just narrates the failure. Refuse
            # before anything runs; the VM is already the boundary.
            raise errors.UnsupportedError(
                KIND,
                f"sandbox={self._sandbox!r} on a {workspace.kind} workspace",
                "codex's own sandbox cannot start inside the VM, so every "
                "command would fail; leave `sandbox` unset there (the VM "
                "is the isolation boundary)",
            )
        probe = await self._probe(workspace)
        if probe.exit_code != 0 and workspace.owner == "provider":
            # Ours to provision. Never on a user's own machine.
            await workspace.exec(["sh", "-c", INSTALL], timeout=600)
            probe = await self._probe(workspace)
        if probe.exit_code != 0:
            raise errors.ExecutableMissingError(KIND, self._executable, INSTALL)
        self._version = probe.stdout.strip().splitlines()[0] or None
        self._workspace = workspace
        if not self._sandbox_explicit:
            # On a provider-owned microVM the VM *is* the isolation
            # boundary, and codex's own sandbox cannot start nested inside
            # it (see above). On the user's own machine, read-only stays
            # the safe default.
            self._sandbox = (
                "danger-full-access"
                if workspace.owner == "provider"
                else "read-only"
            )
        self._lock = _session_lock.SessionLock(workspace, KIND)
        await self._launch(workspace)

    def _auth_hint(self) -> str:
        if self._gateway is not None and self._gateway.is_brokered:
            return BROKERED_HINT.format(host=self._gateway.host)
        return AUTH_HINT

    async def _launch(self, workspace: _base.Workspace) -> None:
        """Spawn app-server and complete the handshake.

        Called at start, and again on demand after a crash: one app-server
        serves every session, so when it dies the harness would otherwise
        be dead for good while claude — which spawns a CLI per session —
        simply carries on. Same SDK, same behaviour: the next piece of work
        gets a fresh server. Sessions that lived on the dead one are gone
        from memory, and a turn on one of them raises AgentCrashedError.
        """
        # The workspace applies its own environment to everything it spawns
        # — that is where the TASK's configuration belongs. A gateway's is
        # this harness's credentials, narrower, so it goes on top.
        self._process = await workspace.spawn(
            [self._executable, "app-server", *_config_args(self._config)],
            env=self._gateway_env or None,
        )
        self._peer = jsonrpc.JsonRpcPeer(self._process, harness=KIND)
        self._peer.start(
            self._on_notify, self._on_request, on_crash=self._on_crash
        )
        try:
            await self._peer.call(
                "initialize",
                {
                    "clientInfo": {
                        "name": "ai-python",
                        "title": "AI SDK for Python",
                        "version": importlib.metadata.version("ai"),
                    }
                },
                timeout=60,
            )
            await self._peer.notify("initialized")
        except BaseException:
            # A failed handshake must not leave the server running.
            await self.close()
            raise

    async def _ensure_running(self) -> None:
        """Relaunch after a crash, before beginning new work."""
        if (
            self._peer is None
            and self._workspace is not None
            and not self._closed
        ):
            await self._launch(self._workspace)

    async def _probe(self, workspace: _base.Workspace) -> _base.ExecResult:
        return await workspace.exec([self._executable, "--version"], timeout=60)

    async def close(self) -> None:
        self._closed = True
        if self._peer is not None:
            await self._peer.close()
            self._peer = None
        if self._process is not None:
            with contextlib.suppress(Exception):
                await self._process.terminate()
            self._process = None
        for session_id in list(self._held):
            await self._release(session_id)
        self._threads.clear()

    async def detach(self) -> None:
        """Let go without terminating app-server.

        Stop reading (close the peer) and drop the input conduit, leaving
        the process running — its FIFO stdin is held open, so it sees no EOF.
        Unlike close(), no terminate is attempted; whether the harness lives
        on is the workspace's `keep` decision.
        """
        self._closed = True
        if self._peer is not None:
            with contextlib.suppress(Exception):
                await self._peer.close()
            self._peer = None
        if self._process is not None:
            with contextlib.suppress(Exception):
                await self._process.detach()
            self._process = None
        self._threads.clear()
        self._queues.clear()

    def _rpc(self) -> jsonrpc.JsonRpcPeer:
        if self._peer is None:
            raise errors.AgentCrashedError(KIND, "app-server is not running")
        return self._peer

    # -- sessions -------------------------------------------------------------

    async def new_session(
        self, history: list[messages_.Message] | None = None
    ) -> str:
        assert self._workspace is not None
        await self._ensure_running()
        result = await self._rpc().call(
            "thread/start",
            {
                "cwd": self._workspace.path,
                "sandbox": self._sandbox,
                # With a hook: ask about everything the policy can ask
                # about, and the hook decides. Without one there is no
                # policy to consult, so the server never asks — a request
                # that must round-trip to this process would freeze an
                # agent whose controller is gone (measured).
                "approvalPolicy": "untrusted" if self.approve else "never",
                **({"model": self._model} if self._model else {}),
            },
        )
        thread_id = str((result or {}).get("thread", {}).get("id") or "")
        if not thread_id:
            raise errors.AgentCrashedError(
                KIND, f"thread/start returned no id: {result}"
            )
        self._threads.add(thread_id)
        await self._acquire(thread_id)
        if history:
            # The thread's model-visible history, before the first turn.
            await self._rpc().call(
                "thread/inject_items",
                {
                    "threadId": thread_id,
                    "items": _handoff.to_codex_items(history),
                },
            )
        return thread_id

    async def resume_session(self, session_id: str) -> None:
        await self._ensure_running()
        await self._require_here(session_id)
        await self._acquire(session_id)
        try:
            await self._rpc().call("thread/resume", {"threadId": session_id})
        except jsonrpc.JsonRpcError as exc:
            await self._release(session_id)
            if "active writer" in str(exc):
                raise errors.SessionBusyError(
                    session_id, None, str(exc)
                ) from exc
            raise errors.ResumeFailedError(session_id, str(exc)) from exc
        self._threads.add(session_id)

    async def _require_here(self, session_id: str) -> dict[str, Any]:
        """Refuse a thread that belongs to a different workspace.

        Codex's thread store is global, not per-directory, so resuming a
        conversation from another project SUCCEEDS natively — and the agent
        would then carry a transcript about files that are not here. The
        SDK scopes conversations to their workspace on every harness, so
        this is enforced rather than inherited.
        """
        assert self._workspace is not None
        try:
            result = await self._rpc().call(
                "thread/read", {"threadId": session_id}
            )
        except jsonrpc.JsonRpcError as exc:
            raise errors.ResumeFailedError(session_id, str(exc)) from exc
        thread = (result or {}).get("thread") or {}
        if not thread:
            raise errors.ResumeFailedError(session_id, "no such thread")
        cwd = thread.get("cwd")
        if cwd is not None and cwd != self._workspace.path:
            raise errors.ResumeFailedError(
                session_id,
                f"that conversation belongs to {cwd}, not "
                f"{self._workspace.path}",
            )
        return thread

    async def _acquire(self, session_id: str) -> None:
        """One writer per thread.

        app-server enforces it too; the lock makes the refusal ours, uniform
        across harnesses, and visible in listings.
        """
        if self._lock is None or session_id in self._held:
            return
        pid = (
            await self._process.os_pid() if self._process is not None else None
        )
        await self._lock.acquire(session_id, pid=pid, client=KIND)
        self._held.add(session_id)

    async def _release(self, session_id: str) -> None:
        self._held.discard(session_id)
        if self._lock is not None:
            with contextlib.suppress(Exception):
                await self._lock.release(session_id)

    async def prepare_tui(self) -> None:
        """On a sandbox that is ours, make sure codex is current.

        The TUI opens on an "Update now / Skip" dialog when a newer release
        exists, its default runs an installer, and the first keystrokes meant
        for the prompt pick it (measured — once on a user's own machine,
        which is exactly why this never touches one). The provider itself
        needs no seeding: it comes from the same -c overrides headless uses.
        """
        if self._workspace is None or self._workspace.owner != "provider":
            return
        # codex >= 0.158 opens a per-folder trust dialog ("Trust this folder?
        # 1. Trust and continue 2. Quit") the first time its TUI runs in a
        # directory. Record the trust in its config so the TUI opens on the
        # prompt — on a VM that is ours, never in a user's own config.
        home = await self._workspace.home()
        config_path = f"{home}/.codex/config.toml"
        try:
            config = await self._workspace.read_text(config_path)
        except FileNotFoundError:
            config = ""
        workdir = self._workspace.path
        if f'[projects."{workdir}"]' not in config:
            trust = f'\n[projects."{workdir}"]\ntrust_level = "trusted"\n'
            await self._workspace.write_text(config_path, config + trust)
        have = (
            await self._workspace.exec(
                [self._executable, "--version"], timeout=60
            )
        ).stdout.split()
        want = (
            await self._workspace.exec(
                ["npm", "view", "@openai/codex", "version"], timeout=120
            )
        ).stdout.strip()
        if want and have and have[-1] != want:
            await self._workspace.exec(["sh", "-c", INSTALL], timeout=600)

    def tui_launch(
        self, session_id: str | None
    ) -> tuple[list[str], dict[str, str], str | None]:
        assert self._workspace is not None
        # --disable in_app_updates: measured, the TUI otherwise opens on an
        # "Update now / Skip" dialog whose default runs an npm install inside
        # the VM, and the first keystrokes meant for the prompt pick it.
        argv = [
            self._executable,
            *(["resume", session_id] if session_id else []),
            "--disable",
            "in_app_updates",
            *_config_args(self._config),
        ]
        if self._model:
            argv += ["--model", self._model]
        env = {
            "TERM": "xterm-256color",
            **dict(self._workspace.env),
            **self._gateway_env,
        }
        # None: codex mints its own thread id, so a NEW TUI conversation is
        # not locked up front — its id is learned from the store afterwards.
        # A resumed one is locked via resume_session_lock. The provider comes
        # from the same -c overrides headless uses, so no config seeding.
        return argv, env, None

    async def claim_new_session(self, session_id: str) -> None:
        await self._acquire(session_id)

    async def record_tui_pid(self, session_id: str, pid: int | None) -> None:
        if self._lock is not None and pid is not None:
            await self._lock.claim_pid(session_id, pid, client=KIND)

    async def release_tui_lock(self, session_id: str) -> None:
        await self._release(session_id)

    async def resume_session_lock(self, session_id: str) -> None:
        await self._require_here(session_id)
        # If THIS adapter holds the thread headless, the TUI is about to be
        # the writer: hand it over, or codex itself will block the TUI.
        await self._hand_over(session_id)
        await self._acquire(session_id)

    async def running_sessions(self) -> set[str]:
        return await self._lock.running() if self._lock is not None else set()

    async def close_session(self, session_id: str) -> None:
        self._threads.discard(session_id)
        self._queues.pop(session_id, None)
        await self._release(session_id)

    async def stage_history(self, history: list[messages_.Message]) -> str:
        """Create a thread with this past and start nothing on it.

        Codex has no store-level import: the thread is created and its
        history injected through app-server, which then HOLDS it (codex's
        cross-process writer lock is a file, ~/.codex/thread-writer-locks/
        <id>.lock, and only ending the holder releases it). So this ends
        this client's app-server afterwards — it relaunches lazily — and the
        thread's rollout stays on disk (measured: 45 KB with the injected
        text, resumable by `codex resume <id>`). Two codex facts a caller
        should know: app-server does not list a thread until it has had a
        turn, so `sessions()` shows it after the first one; and the TUI
        treats injected history as the model's context, not as transcript
        it redraws — the agent knows the past, the screen does not show it.
        """
        assert self._workspace is not None
        await self._ensure_running()
        result = await self._rpc().call(
            "thread/start",
            {
                "cwd": self._workspace.path,
                "sandbox": self._sandbox,
                "approvalPolicy": "untrusted" if self.approve else "never",
                **({"model": self._model} if self._model else {}),
            },
        )
        thread_id = str((result or {}).get("thread", {}).get("id") or "")
        if not thread_id:
            raise errors.AgentCrashedError(
                KIND, f"thread/start returned no id: {result}"
            )
        await self._rpc().call(
            "thread/inject_items",
            {"threadId": thread_id, "items": _handoff.to_codex_items(history)},
        )
        if self._threads:
            # Other conversations are live on this server: do not end it.
            # The thread stays held until close()/detach(); say so.
            raise errors.SessionBusyError(
                thread_id,
                None,
                "this client holds other conversations headless; close() them "
                "first",
            )
        await self._stop_server()
        return thread_id

    async def _stop_server(self) -> None:
        if self._peer is not None:
            await self._peer.close()
            self._peer = None
        if self._process is not None:
            with contextlib.suppress(Exception):
                await self._process.terminate()
            self._process = None
        self._threads.clear()
        self._queues.clear()

    async def _held_elsewhere(self, session_id: str) -> bool:
        """Return whether codex's own writer lock for this thread exists.

        Held by any process, this one included.
        """
        assert self._workspace is not None
        return await self._workspace.exists(
            f"{await self._workspace.home()}/.codex/thread-writer-locks/"
            f"{session_id}.lock"
        )

    async def _hand_over(self, session_id: str) -> None:
        """Release a thread this app-server holds.

        Another process can then drive it.

        Codex enforces one writer ACROSS processes: a TUI resuming a thread
        that a live app-server has loaded shows "this conversation is open
        in another app" and waits. Measured: `thread/unsubscribe` answers
        `unsubscribed` but the thread stays in `thread/loaded/list` and the
        TUI stays blocked — only ending the app-server releases it, and
        there is no unload call. So: if this thread is the only one loaded,
        stop the server (it relaunches lazily for the next headless call);
        if others are loaded too, refuse rather than drop their work.
        """
        if not await self._held_elsewhere(session_id):
            return
        if session_id not in self._threads:
            raise errors.SessionBusyError(
                session_id, None, "another codex process holds this thread"
            )
        if self._threads - {session_id}:
            raise errors.SessionBusyError(
                session_id,
                None,
                "this client holds it headless alongside other conversations; "
                + "close() or detach() it first",
            )
        await self._stop_server()

    async def list_sessions(self) -> list[_session.SessionInfo]:
        await self._ensure_running()
        assert self._workspace is not None
        result = await self._rpc().call("thread/list", {})
        threads = (result or {}).get("data") or []
        here = self._workspace.path
        return [
            _session.SessionInfo(
                kind=KIND,
                session_id=t.get("id", ""),
                title=t.get("name") or t.get("preview") or None,
                cwd=t.get("cwd"),
                updated_at=t.get("updatedAt"),
                created_at=t.get("createdAt"),
            )
            for t in threads
            # Scoped to this workspace, like the Claude adapter: "the
            # conversations that happened here" is the useful question.
            if t.get("id") and (t.get("cwd") in (None, here))
        ]

    async def history(
        self, session_id: str, *, limit: int | None = None, offset: int = 0
    ) -> list[messages_.Message]:
        """Return the thread's stored record, in full.

        `itemsView: "full"` is "every ThreadItem available from persisted
        history". The summary view exists and is never used.
        """
        await self._ensure_running()
        await self._require_here(session_id)
        try:
            result = await self._rpc().call(
                "thread/read",
                {
                    "threadId": session_id,
                    "includeTurns": True,
                    "itemsView": "full",
                },
            )
        except jsonrpc.JsonRpcError as exc:
            raise errors.ResumeFailedError(session_id, str(exc)) from exc
        messages: list[messages_.Message] = []
        thread = (result or {}).get("thread") or {}
        if not thread:
            raise errors.ResumeFailedError(session_id, "no such thread")
        for turn in thread.get("turns") or []:
            for item in turn.get("items") or []:
                messages.extend(_item_to_messages(item, replay=True))
        window = messages[offset:]
        return window[:limit] if limit is not None else window

    async def fork(self, session_id: str) -> str:
        await self._ensure_running()
        await self._require_here(session_id)
        try:
            result = await self._rpc().call(
                "thread/fork", {"threadId": session_id}
            )
        except jsonrpc.JsonRpcError as exc:
            raise errors.ResumeFailedError(
                session_id, f"fork failed: {exc}"
            ) from exc
        forked = str((result or {}).get("thread", {}).get("id") or "")
        if not forked:
            raise errors.ResumeFailedError(
                session_id, f"fork returned no thread id: {result}"
            )
        self._threads.add(forked)
        await self._acquire(forked)
        return forked

    # -- notifications & approvals --------------------------------------------

    async def _on_notify(self, method: str, params: dict[str, Any]) -> None:
        thread_id = params.get("threadId") or (params.get("thread") or {}).get(
            "id"
        )
        if method == "turn/started":
            turn = params.get("turn") or {}
            if thread_id and turn.get("id"):
                self._active_turn[thread_id] = turn["id"]
        elif method == "thread/tokenUsage/updated" and thread_id:
            self._usage[thread_id] = _usage(params.get("tokenUsage") or {})
        queue = self._queues.get(thread_id or "")
        if queue is not None:
            queue.put_nowait((method, params))

    def _on_crash(self, exc: BaseException) -> None:
        """Handle the process being gone.

        Every turn still listening hears it now, as a message on its own queue,
        so the wait ends the same way any other notification would — and turn()
        turns it into AgentCrashedError. The dead peer is dropped so the next
        new session relaunches.
        """
        # The peer already raised an AgentCrashedError; carry its detail, not
        # its str(), or the caller reads "codex crashed: codex crashed: …".
        detail = getattr(exc, "detail", None) or str(exc)
        for queue in list(self._queues.values()):
            queue.put_nowait((CRASHED, {"detail": detail}))
        # Drop the peer so the next new work relaunches. KEEP the process
        # handle: close() still owns terminating it, and terminate() on a
        # pid that is already gone is harmless.
        self._peer = None
        self._threads.clear()

    async def _on_request(
        self, request_id: Any, method: str, params: dict[str, Any]
    ) -> None:
        if method not in APPROVAL_METHODS:
            # Unknown server request: decline rather than hang the agent
            # waiting for an answer we do not know how to give.
            await self._rpc().respond(request_id, {"decision": "decline"})
            return
        decision = await self._decide(method, params)
        await self._rpc().respond(request_id, {"decision": decision})
        # Only NOW interrupt. codex will not process turn/interrupt while
        # this approval is unanswered, so awaiting it inside _decide stalled
        # for the full 120s RPC timeout before the decline even went out —
        # measured 123s to settle a Deny(stop=True). The outcome was right
        # and two minutes late, and a test asserting only the outcome passed.
        pending = self._pending_interrupt.pop(
            params.get("threadId") or "", None
        )
        if pending:
            with contextlib.suppress(Exception):
                await self._rpc().call("turn/interrupt", pending, timeout=30)

    async def _decide(self, method: str, params: dict[str, Any]) -> str:
        thread_id = params.get("threadId") or ""
        if self.approve is None:
            # Unreachable by policy ("never" asks nothing); answered anyway
            # so an unexpected request cannot hang the agent.
            return "accept"
        try:
            assert self._workspace is not None
            call = _approval.ToolCall(
                id=str(params.get("itemId") or params.get("callId") or ""),
                name=_approval_tool_name(method, params),
                input=_approval_input(params),
                kind=method.rsplit("/", 2)[-2] if "/" in method else None,
                hints={
                    "reason": params.get("reason"),
                    "turnId": params.get("turnId"),
                },
                raw=dict(params),
            )
            request = _approval.ApprovalContext(
                call=call,
                history=list(self.live_history(thread_id)),
                workspace=self._workspace,
                harness=KIND,
                session_id=thread_id,
            )
            async with asyncio.timeout(self.approval_timeout):
                decision = await self.approve(request)
        except Exception as exc:
            self._record_error(f"{type(exc).__name__}: {exc}")
            return "decline"
        if isinstance(decision, _approval.Deny):
            if decision.stop and thread_id:
                self._interrupted.add(thread_id)
                turn_id = self._active_turn.get(thread_id)
                if turn_id:
                    # Deferred to _on_request, after the decline is answered.
                    self._pending_interrupt[thread_id] = {
                        "threadId": thread_id,
                        "turnId": turn_id,
                    }
            return "decline"
        if isinstance(decision, _approval.Allow):
            if decision.input is not None:
                self._pending_error = errors.UnsupportedError(
                    KIND,
                    "rewrite_tool_input",
                    "app-server approvals are accept/acceptForSession/decline",
                )
                return "decline"
            if decision.remember:
                self._pending_error = errors.UnsupportedError(
                    KIND,
                    "remember",
                    "acceptForSession did not suppress the next identical "
                    + "request when measured against a real app-server",
                )
                return "decline"
            return "accept"
        return "decline"

    def _record_error(self, detail: str) -> None:
        if callable(self.on_approval_error):
            self.on_approval_error(detail)

    # -- turns ----------------------------------------------------------------

    async def turn(
        self, session_id: str, prompt: str
    ) -> AsyncIterator[events_.AgentEvent]:
        queue: asyncio.Queue[Any] = asyncio.Queue()
        self._queues[session_id] = queue
        self._interrupted.discard(session_id)
        self._usage.pop(session_id, None)
        parts: list[Any] = []
        # Narration the model emits before acting; streamed, but kept out of
        # the settled answer.
        preamble: list[Any] = []
        open_text: set[str] = set()
        names: dict[str, str] = {}
        self._pending_error = None
        # The last transport error codex reported while retrying. If the
        # turn then fails without a message of its own, this is the only
        # explanation anyone gets.
        last_error = ""
        try:
            started = await self._rpc().call(
                "turn/start",
                {
                    "threadId": session_id,
                    "input": [{"type": "text", "text": prompt}],
                },
            )
        except jsonrpc.JsonRpcError as exc:
            raise errors.TurnFailedError(KIND, str(exc)) from exc
        # The turn id comes from the RESPONSE. There is a `turn/started`
        # notification in the protocol but this server does not send one,
        # and both steer and interrupt require the id — without it, stop()
        # silently did nothing and the turn ran on.
        turn_id = ((started or {}).get("turn") or {}).get("id")
        if turn_id:
            self._active_turn[session_id] = turn_id
        # StreamStart only AFTER the turn exists. Emitting it first let a
        # consumer act on the stream's first event — steer, for instance —
        # before there was any turn to act on.
        yield events_.StreamStart()
        while True:
            method, params = await queue.get()
            if method == CRASHED:
                self._queues.pop(session_id, None)
                raise errors.AgentCrashedError(
                    KIND, params.get("detail") or "process exited mid-turn"
                )
            # An interrupted turn's notifications can still be in flight when
            # the next one starts. Without this the new turn settles on the
            # OLD turn's completion and comes back empty.
            other = params.get("turnId") or (params.get("turn") or {}).get("id")
            if turn_id and other and other != turn_id:
                continue
            if method == "item/agentMessage/delta":
                item_id = params.get("itemId", "")
                if item_id not in open_text:
                    open_text.add(item_id)
                    yield events_.TextStart(block_id=item_id)
                yield events_.TextDelta(
                    chunk=params.get("delta", ""), block_id=item_id
                )
            elif method in (
                "item/reasoning/textDelta",
                "item/reasoning/summaryTextDelta",
            ):
                yield events_.ReasoningDelta(
                    chunk=params.get("delta", ""),
                    block_id=params.get("itemId", ""),
                )
            elif method == "item/started":
                item = params.get("item") or {}
                if item.get("type") in TOOL_ITEMS:
                    names[item.get("id", "")] = _tool_name(item)
                    yield events_.ToolStart(
                        tool_call_id=item.get("id", ""),
                        tool_name=_tool_name(item),
                    )
            elif method == "item/completed":
                item = params.get("item") or {}
                for event in _completed_events(
                    item, open_text, parts, names, preamble
                ):
                    yield event
            elif method == "turn/completed":
                self._active_turn.pop(session_id, None)
                if self._pending_error is not None:
                    error, self._pending_error = self._pending_error, None
                    self._queues.pop(session_id, None)
                    raise error
                usage = self._usage.get(session_id, usage_.Usage())
                interrupted = session_id in self._interrupted
                self._interrupted.discard(session_id)
                self._queues.pop(session_id, None)
                # A failed turn says so on the way out. Reporting it as a
                # normal stop with empty text turns "401 Unauthorized" into
                # silence, which is what it did.
                completed = params.get("turn") or {}
                if completed.get("status") == "failed" and not interrupted:
                    raise _turn_failure(
                        _failure_detail(completed) or last_error,
                        hint=self._auth_hint(),
                    )
                yield events_.StreamEnd(
                    # Fall back to the preamble only if the model never gave
                    # a final answer — better a preamble than nothing.
                    message=messages_.Message(
                        role="assistant", parts=parts or preamble
                    ),
                    usage=usage,
                    finish_reason="cancelled" if interrupted else "stop",
                )
                return
            elif method == "turn/failed":
                self._queues.pop(session_id, None)
                raise _turn_failure(
                    _failure_detail(params), hint=self._auth_hint()
                )
            elif method == "error":
                # Codex reports transport trouble as it retries. Keep the
                # last one: if the turn then fails without a message of its
                # own, this is the only explanation anyone gets.
                last_error = _error_message(params) or last_error

    async def steer(self, session_id: str, text: str) -> None:
        turn_id = self._active_turn.get(session_id)
        if not turn_id:
            raise RuntimeError("no active turn to steer")
        try:
            await self._rpc().call(
                "turn/steer",
                {
                    "threadId": session_id,
                    # A precondition, not decoration: steering the wrong turn
                    # is worse than failing to steer.
                    "expectedTurnId": turn_id,
                    "input": [{"type": "text", "text": text}],
                },
            )
        except jsonrpc.JsonRpcError as exc:
            raise errors.TurnFailedError(
                KIND, f"steer rejected: {exc}"
            ) from exc

    async def stop(self, session_id: str) -> None:
        turn_id = self._active_turn.get(session_id)
        self._interrupted.add(session_id)
        if not turn_id:
            return
        with contextlib.suppress(Exception):
            await self._rpc().call(
                "turn/interrupt", {"threadId": session_id, "turnId": turn_id}
            )


def _tool_name(item: dict[str, Any]) -> str:
    # Never the command line: that made every shell call a differently named
    # tool. The command is in the call's arguments (`_tool_args`), and the
    # approval hook sees the same native name (`_approval_tool_name`).
    return (
        item.get("toolName") or item.get("server") or item.get("type") or "tool"
    )


def _approval_tool_name(method: str, params: dict[str, Any]) -> str:
    """Return the NATIVE tool identifier: the item kind, never the args.

    This once returned the command line itself for command executions, so
    `ctx.call.name` was a different string per command and a hook could not
    key a policy on it. Found only in a sandbox: there codex runs with full
    access and writes files through the shell, so the name became
    `/bin/bash -lc "printf 'OK' > notes.txt"`; locally the same write went
    through fileChange and nothing looked wrong. The command is in `input`.
    """
    if "commandExecution" in method or "execCommand" in method:
        return "commandExecution"
    if "fileChange" in method or "applyPatch" in method:
        return "fileChange"
    return method.rsplit("/", 1)[-1]


def _approval_input(params: dict[str, Any]) -> dict[str, Any]:
    return {
        k: v
        for k, v in params.items()
        if k not in ("threadId", "turnId", "itemId") and v is not None
    }


def _completed_events(
    item: dict[str, Any],
    open_text: set[str],
    parts: list[Any],
    names: dict[str, str],
    preamble: list[Any],
) -> list[Any]:
    out: list[Any] = []
    item_id = item.get("id", "")
    kind = item.get("type")
    if kind == "agentMessage":
        if item_id in open_text:
            open_text.discard(item_id)
            out.append(events_.TextEnd(block_id=item_id))
        if item.get("text"):
            # Codex labels its messages: "commentary" is the preamble it
            # narrates before acting, "final_answer" is the reply. Both are
            # streamed, but only the answer belongs in the settled message —
            # otherwise result.text reads "I'll read util.py.add".
            part = messages_.TextPart(text=item["text"])
            if item.get("phase") == "commentary":
                preamble.append(part)
            else:
                parts.append(part)
    elif kind == "reasoning":
        text = _reasoning_text(item)
        if text:
            parts.append(messages_.ReasoningPart(text=text))
    elif kind in TOOL_ITEMS:
        call = messages_.ToolCallPart(
            tool_call_id=item_id,
            tool_name=names.get(item_id) or _tool_name(item),
            tool_args=json.dumps(_tool_args(item)),
        )
        parts.append(call)
        # The arguments must ride a ToolDelta: the hydrator assembles a
        # call only from deltas, and overwrites ToolEnd.tool_call with the
        # result. See the claude adapter for the same rule.
        out.append(
            events_.ToolDelta(tool_call_id=item_id, chunk=call.tool_args)
        )
        out.append(events_.ToolEnd(tool_call_id=item_id, tool_call=call))
        result = messages_.ToolResultPart(
            tool_call_id=item_id,
            tool_name=call.tool_name,
            result=item.get("output")
            or item.get("changes")
            or item.get("result"),
            result_kind="error" if item.get("status") == "failed" else "json",
        )
        out.append(
            events_.ToolCallResult(
                message=messages_.Message(role="tool", parts=[result]),
                results=[result],
            )
        )
    return out


def _reasoning_text(item: dict[str, Any]) -> str:
    """Codex reports reasoning as either text or a LIST of summary parts."""
    text = item.get("text")
    if isinstance(text, str) and text:
        return text
    summary = item.get("summary")
    if isinstance(summary, str):
        return summary
    if isinstance(summary, list):
        return "".join(part for part in summary if isinstance(part, str))
    return ""


def _tool_args(item: dict[str, Any]) -> dict[str, Any]:
    return {
        k: v
        for k, v in item.items()
        if k in ("command", "changes", "arguments", "query", "path", "cwd")
    }


def _item_to_messages(
    item: dict[str, Any], *, replay: bool = False
) -> list[messages_.Message]:
    kind = item.get("type")
    if kind == "userMessage":
        # A user item carries `content` blocks, not a flat `text`.
        text = item.get("text")
        if not isinstance(text, str):
            text = "".join(
                block.get("text", "")
                for block in item.get("content") or []
                if isinstance(block, dict)
            )
        return [
            messages_.Message(
                role="user",
                parts=[messages_.TextPart(text=text)],
                replay=replay,
            )
        ]
    if kind == "agentMessage" and item.get("text"):
        return [
            messages_.Message(
                role="assistant",
                parts=[messages_.TextPart(text=item["text"])],
                replay=replay,
            )
        ]
    if kind == "reasoning" and _reasoning_text(item):
        return [
            messages_.Message(
                role="assistant",
                parts=[messages_.ReasoningPart(text=_reasoning_text(item))],
                replay=replay,
            )
        ]
    if kind in TOOL_ITEMS:
        call = messages_.ToolCallPart(
            tool_call_id=item.get("id", ""),
            tool_name=_tool_name(item),
            tool_args=json.dumps(_tool_args(item)),
        )
        result = messages_.ToolResultPart(
            tool_call_id=item.get("id", ""),
            tool_name=call.tool_name,
            result=item.get("output") or item.get("changes"),
            result_kind="error" if item.get("status") == "failed" else "json",
        )
        return [
            messages_.Message(role="assistant", parts=[call], replay=replay),
            messages_.Message(role="tool", parts=[result], replay=replay),
        ]
    return []


def _usage(raw: dict[str, Any]) -> usage_.Usage:
    totals = raw.get("total") or raw.get("lastTurn") or raw

    def count(*names: str) -> int:
        for name in names:
            value = totals.get(name)
            if isinstance(value, int):
                return value
        return 0

    return usage_.Usage(
        input_tokens=count("inputTokens", "input_tokens"),
        output_tokens=count("outputTokens", "output_tokens"),
        cache_read_tokens=count("cachedInputTokens", "cached_input_tokens"),
        reasoning_tokens=count(
            "reasoningOutputTokens", "reasoning_output_tokens"
        ),
        raw={"codex": raw},
    )


def _error_message(payload: dict[str, Any]) -> str:
    error = payload.get("error")
    if isinstance(error, dict):
        detail = error.get("additionalDetails") or ""
        message = str(error.get("message") or "")
        return f"{message} {detail}".strip()
    return str(error or "")


def _failure_detail(payload: dict[str, Any]) -> str:
    """Whatever codex can tell us about why a turn failed."""
    return _error_message(payload)[:500]


AUTH_HINT = (
    "codex authenticates where it RUNS, and `codex login` on your machine "
    "does not travel to a sandbox. Give it a way to reach a model: "
    "VercelSandbox(gateway=vercel_ai_gateway()) injects a gateway key at "
    "egress, Local(path, env={'OPENAI_API_KEY': ...}) or "
    "VercelSandbox(env={...}) "
    "puts a key of its own in the environment, or point codex at another "
    "provider with codex(config={'model_provider': ..., "
    "'model_providers.<name>.base_url': ...})."
)
BROKERED_HINT = (
    "This workspace injects the credential into requests to {host} at egress, "
    "so codex never held it and no login in the VM is involved. A 401 here "
    "means the gateway rejected the injected key — check AI_GATEWAY_API_KEY on "
    "the machine that opened the sandbox — or the egress policy is not "
    "rewriting requests to that host."
)


def gateway_config(gateway: _gateway.Gateway | None) -> dict[str, str]:
    """Tell codex where its model is: a provider table, as `-c` overrides.

    The table names the env var that carries the key; it never carries the key
    itself. This is the one place that spelling lives.
    """
    if gateway is None:
        return {}
    return {
        "model_provider": "vercel",
        "model_providers.vercel.name": "Vercel AI Gateway",
        "model_providers.vercel.base_url": f"{gateway.base_url}/codex/v1",
        "model_providers.vercel.env_key": _gateway.KEY_VAR,
        # Required: codex no longer speaks Chat Completions.
        "model_providers.vercel.wire_api": "responses",
    }


def gateway_env(gateway: _gateway.Gateway | None) -> dict[str, str]:
    """Return the other half: the variable the provider table names."""
    return {} if gateway is None else {_gateway.KEY_VAR: gateway.credential}


def _turn_failure(detail: str, *, hint: str = AUTH_HINT) -> Exception:
    """Rename the failure for what the caller has to fix."""
    if errors._looks_unauthenticated(detail):
        return workspace_errors.NotAuthenticatedError(KIND, detail, hint)
    return errors.TurnFailedError(KIND, detail)


def _config_args(config: dict[str, Any]) -> list[str]:
    """`-c key=value` pairs, with values rendered as TOML.

    JSON string/number/bool literals are valid TOML, so json.dumps is the
    right renderer for the scalars anyone actually passes here.
    """
    args: list[str] = []
    for key, value in config.items():
        args += ["-c", f"{key}={json.dumps(value)}"]
    return args
