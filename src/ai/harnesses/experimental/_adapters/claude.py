"""Claude Code, driven through claude-agent-sdk.

Identity rule kept from v1 because it earned it: the session id we hand the
CLI IS its own session id, so `claude --resume <id>` works on it — in this
SDK or in the CLI's own UI — with no adapter-side registry.

Everything here is verified against a real `claude` CLI by the live tests; the
comments record what was MEASURED, not what the API suggested.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import uuid
import warnings
from typing import TYPE_CHECKING, Any, cast

import claude_agent_sdk

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
from . import base, claude_store, remote_cli

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

KIND = "claude-code"
INSTALL = "npm install -g @anthropic-ai/claude-code"

# Claude routes permission-worthy tool calls through can_use_tool. Read-only
# tools never reach it — verified by running with setting_sources=[] so no
# user config was loaded, and Read still did not appear. That is the CLI's
# own policy, so coverage is "policy", not "all".
CAPABILITIES = _capabilities.Capabilities(
    steer=True,
    stop=True,
    resume=True,
    fork=True,
    history=True,
    rewrite_tool_input=True,
    deny_reason=True,
    images=True,
    approval="policy",
)


class ClaudeAdapter:
    kind = KIND
    capabilities = CAPABILITIES

    def __init__(
        self,
        executable: str = "claude",
        model: str | None = None,
        gateway: _gateway.Gateway | None = None,
        effort: str | None = None,
    ) -> None:
        self._executable = executable
        self._model = model
        self._gateway = gateway
        self._gateway_env: dict[str, str] = gateway_env(gateway)
        self._effort = effort
        # The CLI's session store, read through the workspace when the CLI
        # runs somewhere other than this machine. None on Local: there the
        # SDK's own disk readers are exactly right.
        self._store: claude_store.WorkspaceSessionStore | None = None
        self._lock: _session_lock.SessionLock | None = None
        self._held: set[str] = set()
        self._version: str | None = None
        self._workspace: _base.Workspace | None = None
        self._clients: dict[str, Any] = {}
        self._contexts: dict[str, dict[str, Any]] = {}
        # Sessions whose turn WE interrupted. The CLI reports an error
        # result after an interrupt, and a stop the caller asked for must
        # settle as cancelled rather than raise as a failure.
        self._interrupted: set[str] = set()
        # See codex: an error raised inside the permission callback cannot
        # reach the caller, so it is parked and re-raised on the turn.
        self._pending_error: BaseException | None = None
        self.approve: base.ApprovalHook | None = None
        self.approval_timeout: float = 300.0
        self.on_approval_error: Any = None
        # Set by the harness: the session's LIVE transcript, so the approval
        # hook reads the same history the caller does. Distinct from
        # history(), which reads the harness's STORED record.
        self.live_history: Any = lambda session_id: []

    @property
    def version(self) -> str | None:
        return self._version

    @property
    def process_ids(self) -> list[int]:
        """Return the CLI's pids, best effort.

        claude-agent-sdk owns the CLI process and exposes no public pid, so this
        reads the transport's private handle. It returns [] rather than guessing
        if that ever changes shape — callers must treat an empty list as "not
        observable", not as "no processes".
        """
        pids: list[int] = []
        for client in self._clients.values():
            proc = getattr(
                getattr(client, "_transport", None), "_process", None
            )
            pid = getattr(proc, "pid", None)
            if isinstance(pid, int):
                pids.append(pid)
        return pids

    async def start(
        self, workspace: _base.Workspace, options: dict[str, Any]
    ) -> None:
        if workspace.kind != "local" and not workspace.duplex_spawn:
            # claude-agent-sdk spawns the CLI itself and only accepts a cwd.
            # Off this machine we supply our own transport instead — but
            # that needs a process we can write to.
            raise errors.UnsupportedError(
                KIND,
                f"{workspace.kind} workspaces",
                "the CLI speaks newline-delimited JSON on stdin, which this "
                + "workspace cannot provide",
            )
        # An explicit gateway wins; otherwise the workspace's. Either way,
        # where the workspace injects the credential at egress the CLI is
        # handed the placeholder: the key stays on the host that built the
        # firewall rule. The env it produces is this adapter's own dialect.
        gateway = self._gateway or workspace.gateway
        if gateway is not None and workspace.injects_credentials_for(
            gateway.host
        ):
            gateway = gateway.brokered()
        self._gateway = gateway
        self._gateway_env = gateway_env(gateway)
        probe = await self._probe(workspace)
        if probe.exit_code != 0 and workspace.owner == "provider":
            # Ours to provision. Never on a user's own machine.
            await workspace.exec(["sh", "-c", INSTALL], timeout=600)
            probe = await self._probe(workspace)
        if workspace.kind != "local":
            self._store = await claude_store.WorkspaceSessionStore.discover(
                workspace
            )
        if probe.exit_code != 0:
            raise errors.ExecutableMissingError(KIND, self._executable, INSTALL)
        self._version = probe.stdout.strip().splitlines()[0] or None
        self._workspace = workspace
        self._lock = _session_lock.SessionLock(workspace, KIND)

    async def _probe(self, workspace: _base.Workspace) -> _base.ExecResult:
        return await workspace.exec([self._executable, "--version"], timeout=60)

    async def close(self) -> None:
        for session_id in list(self._clients):
            await self.close_session(session_id)

    async def detach(self) -> None:
        """Let go of every session without terminating the CLIs.

        `client.disconnect()` terminates the transport's process; detach must
        not. It drops the conduit through the transport and forgets the
        client, leaving the remote CLI running for a later reconnect.
        """
        for session_id, client in list(self._clients.items()):
            transport = getattr(client, "_transport", None)
            if transport is not None and hasattr(transport, "release_on_close"):
                # Through the client's own disconnect(): that is what stops
                # its reader task. Dropping the client without it left a
                # reader iterating the process's stdout forever, and the
                # workspace's close() then hung on that stream (measured:
                # a 300s teardown timeout). The transport now releases
                # instead of terminating when disconnect() closes it.
                transport.release_on_close()
                with contextlib.suppress(Exception):
                    async with asyncio.timeout(30):
                        await client.disconnect()
            self._clients.pop(session_id, None)
            self._contexts.pop(session_id, None)

    # -- sessions -------------------------------------------------------------

    async def stage_history(self, history: list[messages_.Message]) -> str:
        """Write the past into the CLI's own store, and start nothing.

        Wherever the CLI runs — the store class reads and writes through the
        workspace — under a fresh id. A resume of that id, headless or in the
        TUI, then begins with this context.
        """
        assert self._workspace is not None
        session_id = str(uuid.uuid4())
        dropped = _handoff.reasoning_count(history)
        if dropped:
            warnings.warn(
                f"{dropped} reasoning part(s) in the history were not carried "
                "into "
                + "claude-code: Anthropic's safeguards refuse replayed model "
                "reasoning "
                + "([reasoning_extraction]). Text and tool calls were carried "
                "in full.",
                stacklevel=2,
            )
        store = (
            self._store
            or await claude_store.WorkspaceSessionStore.discover(
                self._workspace
            )
        )
        await store.append(
            {
                "project_key": claude_agent_sdk.project_key_for_directory(
                    self._workspace.path
                ),
                "session_id": session_id,
            },
            cast(
                "list[claude_agent_sdk.SessionStoreEntry]",
                _handoff.to_claude_records(
                    history, session_id, self._workspace.path
                ),
            ),
        )
        return session_id

    async def new_session(
        self, history: list[messages_.Message] | None = None
    ) -> str:
        if history:
            session_id = await self.stage_history(history)
            await self._acquire(session_id)
            await self._connect(session_id, resume=session_id)
            await self._publish_pid(session_id)
            return session_id
        session_id = str(uuid.uuid4())
        await self._acquire(session_id)
        await self._connect(session_id, resume=None)
        await self._publish_pid(session_id)
        return session_id

    async def resume_session(self, session_id: str) -> None:
        # Check the store first: connecting to a ghost surfaces as
        # claude-agent-sdk's own ResultError somewhere downstream, which is
        # not an error a caller of THIS library can be asked to catch.
        await self._require_known(session_id)
        await self._acquire(session_id)
        try:
            await self._connect(session_id, resume=session_id)
        except Exception as exc:
            await self._release(session_id)
            raise errors.ResumeFailedError(session_id, str(exc)) from exc
        await self._publish_pid(session_id)

    async def _require_known(self, session_id: str) -> None:
        assert self._workspace is not None
        try:
            if self._store is not None:
                key = claude_agent_sdk.project_key_for_directory(
                    self._workspace.path
                )
                known = {
                    e["session_id"]
                    for e in await self._store.list_sessions(key)
                }
                info: Any = session_id if session_id in known else None
            else:
                info = await asyncio.to_thread(
                    claude_agent_sdk.get_session_info,
                    session_id,
                    self._workspace.path,
                )
        except Exception as exc:
            raise errors.ResumeFailedError(session_id, str(exc)) from exc
        if info is None:
            raise errors.ResumeFailedError(
                session_id,
                f"no such session in workspace {self._workspace.path}",
            )

    async def _acquire(self, session_id: str) -> None:
        """Refuse before launching a second CLI: one writer per conversation."""
        if self._lock is None or session_id in self._held:
            return
        await self._lock.acquire(session_id, client=KIND)
        self._held.add(session_id)

    async def _publish_pid(self, session_id: str) -> None:
        client = self._clients.get(session_id)
        process = getattr(getattr(client, "_transport", None), "_process", None)
        if self._lock is None or process is None:
            return
        pid = (
            await process.os_pid()
            if hasattr(process, "os_pid")
            else getattr(process, "pid", None)
        )
        await self._lock.claim_pid(session_id, pid, client=KIND)

    async def _release(self, session_id: str) -> None:
        self._held.discard(session_id)
        if self._lock is not None:
            with contextlib.suppress(Exception):
                await self._lock.release(session_id)

    async def prepare_tui(self) -> None:
        # The claude TUI's first run walks a theme picker and a security
        # notice and never reaches a prompt until onboarding is marked done.
        # On a sandbox that is ours, mark it done so tui() opens on a prompt;
        # never write into a user's own ~/.claude.json.
        if self._workspace is None or self._workspace.owner != "provider":
            return
        path = f"{await self._workspace.home()}/.claude.json"
        try:
            config = json.loads(await self._workspace.read_text(path))
        except (FileNotFoundError, ValueError):
            config = {}
        # Two gates: the global onboarding walk, and a PER-PROJECT trust
        # dialog ("do you trust the files in this folder?") that appears the
        # first time the TUI opens in a directory. Both must be marked done,
        # or the TUI opens on a dialog and the first keystrokes answer it.
        workdir = self._workspace.path
        project = config.setdefault("projects", {}).setdefault(workdir, {})
        # A third gate: a real ANTHROPIC_API_KEY in the TUI's environment
        # brings up "use this API key? — No (recommended)", and a keystroke
        # meant for the prompt picks No. Claude records approvals by the
        # key's last 20 characters; pre-approve the key we are launching with.
        env = {**dict(self._workspace.env), **self._gateway_env}
        key = env.get("ANTHROPIC_API_KEY") or ""
        approvals = config.setdefault(
            "customApiKeyResponses", {"approved": [], "rejected": []}
        )
        key_ok = not key or key[-20:] in approvals.get("approved", [])
        already = (
            config.get("hasCompletedOnboarding")
            and config.get("bypassPermissionsModeAccepted")
            and project.get("hasTrustDialogAccepted")
            and key_ok
        )
        if already:
            return
        if key and key[-20:] not in approvals.setdefault("approved", []):
            approvals["approved"].append(key[-20:])
        config.update(
            hasCompletedOnboarding=True,
            bypassPermissionsModeAccepted=True,
            theme="dark",
        )
        project.update(
            hasTrustDialogAccepted=True,
            hasCompletedProjectOnboarding=True,
        )
        await self._workspace.write_text(path, json.dumps(config))

    def tui_launch(
        self, session_id: str | None
    ) -> tuple[list[str], dict[str, str], str | None]:
        assert self._workspace is not None
        running_as = session_id or str(uuid.uuid4())
        argv = [self._executable]
        argv += (
            ["--resume", session_id]
            if session_id
            else ["--session-id", running_as]
        )
        if self._model:
            argv += ["--model", self._model]
        # DISABLE_AUTOUPDATER: measured, the TUI otherwise updates itself in
        # the background and announces "Restart to apply" mid-session.
        env = {
            "TERM": "xterm-256color",
            "DISABLE_AUTOUPDATER": "1",
            **dict(self._workspace.env),
            **self._gateway_env,
        }
        return argv, env, running_as

    def _auth_hint(self) -> str:
        if self._gateway is not None and self._gateway.is_brokered:
            return BROKERED_HINT.format(host=self._gateway.host)
        return AUTH_HINT

    async def claim_new_session(self, session_id: str) -> None:
        # A minted --session-id is not in the store yet; lock it directly so
        # the open TUI shows as running and refuses a second writer.
        await self._acquire(session_id)

    async def record_tui_pid(self, session_id: str, pid: int | None) -> None:
        if self._lock is not None and pid is not None:
            await self._lock.claim_pid(session_id, pid, client=KIND)

    async def release_tui_lock(self, session_id: str) -> None:
        await self._release(session_id)

    async def resume_session_lock(self, session_id: str) -> None:
        await self._require_known(session_id)
        await self._acquire(session_id)

    async def running_sessions(self) -> set[str]:
        return await self._lock.running() if self._lock is not None else set()

    async def close_session(self, session_id: str) -> None:
        client = self._clients.pop(session_id, None)
        self._contexts.pop(session_id, None)
        if client is not None:
            with contextlib.suppress(Exception):
                await client.disconnect()
        await self._release(session_id)

    async def _connect(self, session_id: str, *, resume: str | None) -> None:
        assert self._workspace is not None
        options: dict[str, Any] = {
            "cwd": self._workspace.path,
            # Raw Anthropic stream events, which map 1:1 onto the AI SDK's
            # Start/Delta/End triples. Without this the SDK only yields whole
            # messages, so a turn interrupted mid-answer would settle with
            # nothing — a deadline has to be a settlement, not a loss.
            "include_partial_messages": True,
            "session_id": session_id if resume is None else None,
            "resume": resume,
        }
        # claude-agent-sdk spawns the CLI itself, so the workspace never
        # gets to apply its own environment the way it does for a process
        # we spawn through it. Pass it explicitly instead. A gateway's env
        # is this harness's credentials — narrower than the workspace's
        # task configuration — so it goes on top.
        env = {**dict(self._workspace.env), **self._gateway_env}
        if env:
            options["env"] = env
        if self._model:
            options["model"] = self._model
        if self._effort:
            # Reasoning effort, with adaptive thinking so the budget follows
            # it. The CLI's own default applies when none is asked for.
            options["effort"] = self._effort
            options["thinking"] = {"type": "adaptive"}
        # A hook is a gate answered by THIS process — a round-trip to the
        # controller for every gated call. No hook means there is no policy
        # to consult, so no gate is installed and the CLI applies its own
        # allow-everything mode instead. Measured: with an always-allow gate
        # here, an agent whose controller died froze at its next tool call
        # waiting for an answer; with the mode set in the CLI it finished
        # 40/40 files unattended. Same behaviour attached, and it keeps
        # working when nobody is. `approve=` is how you restrict it — and it
        # means the controller must stay alive to answer.
        if self.approve is not None:
            options["can_use_tool"] = self._make_can_use_tool(session_id)
        else:
            options["permission_mode"] = "bypassPermissions"
        agent_options = claude_agent_sdk.ClaudeAgentOptions(**options)
        transport = None
        if self._workspace is not None and self._workspace.kind != "local":
            transport = remote_cli.RemoteCliTransport(
                self._workspace,
                agent_options,
                executable=self._executable,
            )
        client = claude_agent_sdk.ClaudeSDKClient(
            options=agent_options, transport=transport
        )
        await client.connect()
        self._clients[session_id] = client

    def _client(self, session_id: str) -> Any:
        client = self._clients.get(session_id)
        if client is None:
            raise RuntimeError(f"unknown session {session_id}")
        return client

    # -- approval -------------------------------------------------------------

    def _make_can_use_tool(
        self, session_id: str
    ) -> claude_agent_sdk.CanUseTool:
        async def can_use_tool(
            name: str,
            tool_input: dict[str, Any],
            ctx: claude_agent_sdk.ToolPermissionContext,
        ) -> claude_agent_sdk.PermissionResult:
            # EVERYTHING here is guarded, not just the hook call: a bug of
            # ours must deny and be reported, never vanish into the agent as
            # an opaque internal error.
            try:
                return await self._decide(session_id, name, tool_input, ctx)
            except Exception as exc:
                self._record_error(f"{type(exc).__name__}: {exc}")
                return claude_agent_sdk.PermissionResultDeny(
                    message="approval failed; denied"
                )

        return can_use_tool

    async def _decide(
        self,
        session_id: str,
        name: str,
        tool_input: dict[str, Any],
        ctx: claude_agent_sdk.ToolPermissionContext,
    ) -> claude_agent_sdk.PermissionResult:
        assert (
            self.approve is not None
        ), "the gate is only installed with a hook"
        assert self._workspace is not None
        call = _approval.ToolCall(
            id=getattr(ctx, "tool_use_id", None)
            or f"call-{uuid.uuid4().hex[:12]}",
            # The VERIFIED native name, from the SDK's own callback —
            # never display text an agent can influence.
            name=name,
            input=dict(tool_input),
            agent_id=getattr(ctx, "agent_id", None),
            hints={"blocked_path": getattr(ctx, "blocked_path", None)},
            raw={"tool_name": name, "input": tool_input},
        )
        request = _approval.ApprovalContext(
            call=call,
            history=list(self.live_history(session_id)),
            workspace=self._workspace,
            harness=KIND,
            session_id=session_id,
            prompt=self._contexts.get(session_id, {}).get("prompt"),
        )
        assert self.approve is not None
        try:
            async with asyncio.timeout(self.approval_timeout):
                decision = await self.approve(request)
        except (Exception, TimeoutError) as exc:
            # An approval gate that fails open is not a gate.
            self._record_error(f"{type(exc).__name__}: {exc}")
            return claude_agent_sdk.PermissionResultDeny(
                message="approval hook failed; denied"
            )
        if isinstance(decision, _approval.Deny):
            if decision.stop:
                # We are ending this turn on purpose. The CLI reports an
                # error result after an interrupt, so remember that we asked
                # for it and settle as cancelled instead of raising.
                self._interrupted.add(session_id)
            return claude_agent_sdk.PermissionResultDeny(
                message=decision.reason or "denied by policy",
                interrupt=decision.stop,
            )
        if isinstance(decision, _approval.Allow):
            if decision.remember:
                # Parked and re-raised on the turn: an unsupported request
                # must reach the CALLER, and an exception thrown inside the
                # harness's permission callback does not.
                self._pending_error = errors.UnsupportedError(
                    KIND,
                    "remember",
                    "session-scoped permission rules do not suppress the next "
                    + "identical request, and the alternative would silently "
                    + "widen authority to all edits",
                )
                return claude_agent_sdk.PermissionResultDeny(
                    message="remember is unsupported here"
                )
            return claude_agent_sdk.PermissionResultAllow(
                updated_input=decision.input
            )
        return claude_agent_sdk.PermissionResultDeny(
            message="approval hook returned no decision"
        )

    def _record_error(self, detail: str) -> None:
        if callable(self.on_approval_error):
            self.on_approval_error(detail)

    # -- turns ----------------------------------------------------------------

    async def turn(
        self, session_id: str, prompt: str
    ) -> AsyncIterator[events_.AgentEvent]:
        client = self._client(session_id)
        turn_session_id = session_id
        # An interrupted turn leaves its trailing result unread on the
        # client: the CLI still emits one (as an error) after the interrupt.
        # Consume it before prompting, or THIS turn would settle on the
        # previous turn's ending.
        await self._drain_interrupted(session_id)
        state = self._contexts.setdefault(session_id, {"prompt": None})
        state["prompt"] = prompt

        parts: list[Any] = []
        names: dict[str, str] = {}
        open_blocks: dict[int, str] = {}
        text_so_far: list[str] = []
        self._pending_error = None
        try:
            await client.query(prompt)
        except claude_agent_sdk.ClaudeSDKError as exc:
            raise _translate(exc) from exc
        # StreamStart only once the CLI has answered for this turn — its
        # first message is proof it READ the prompt. "Written to the
        # transport" is not: in a sandbox the prompt travels through an
        # interactive session into the CLI's stdin, and a detach right after
        # the write closed that session with the prompt still in transit
        # (measured: an unattended push whose agent never saw its prompt).
        # A consumer may act on the first event; it must mean the turn is live.
        started = False
        async for message in _guard(client.receive_response()):
            if not started:
                started = True
                yield events_.StreamStart()
            if isinstance(message, claude_agent_sdk.StreamEvent):
                # Text and reasoning stream here, token by token. The
                # complete AssistantMessage that follows repeats them, so
                # only tool calls are taken from it.
                for event in _stream_events(message.event, open_blocks):
                    if isinstance(event, events_.TextDelta):
                        text_so_far.append(event.chunk)
                    yield event
            elif isinstance(message, claude_agent_sdk.AssistantMessage):
                for block in message.content:
                    if (
                        isinstance(block, claude_agent_sdk.TextBlock)
                        and block.text
                    ):
                        parts.append(messages_.TextPart(text=block.text))
                    elif isinstance(block, claude_agent_sdk.ToolUseBlock):
                        # tool_args is a JSON STRING in the AI SDK's model,
                        # not a mapping.
                        call = messages_.ToolCallPart(
                            tool_call_id=block.id,
                            tool_name=block.name,
                            tool_args=json.dumps(block.input),
                        )
                        names[block.id] = block.name
                        parts.append(call)
                        yield events_.ToolStart(
                            tool_call_id=block.id, tool_name=block.name
                        )
                        # claude-agent-sdk hands over a complete block, so the
                        # call's arguments are final the moment it arrives —
                        # but they still travel as a delta. The hydrator
                        # builds a call's arguments ONLY from ToolDeltas and
                        # then replaces ToolEnd.tool_call with its own copy,
                        # so without this one chunk every consumer, and the
                        # session's own history, saw tool_args="".
                        yield events_.ToolDelta(
                            tool_call_id=block.id, chunk=call.tool_args
                        )
                        yield events_.ToolEnd(
                            tool_call_id=block.id, tool_call=call
                        )
            elif isinstance(message, claude_agent_sdk.UserMessage):
                for block in (
                    message.content if isinstance(message.content, list) else []
                ):
                    if isinstance(block, claude_agent_sdk.ToolResultBlock):
                        result = messages_.ToolResultPart(
                            tool_call_id=block.tool_use_id,
                            tool_name=names.get(block.tool_use_id, ""),
                            result=block.content,
                            result_kind="error" if block.is_error else "json",
                        )
                        # Tool results are their own message in the AI SDK's
                        # model (role "tool"), not parts of the assistant's.
                        yield events_.ToolCallResult(
                            message=messages_.Message(
                                role="tool", parts=[result]
                            ),
                            results=[result],
                        )
            elif isinstance(message, claude_agent_sdk.ResultMessage):
                assistant = messages_.Message(role="assistant", parts=parts)
                if self._pending_error is not None:
                    error, self._pending_error = self._pending_error, None
                    raise error
                interrupted = turn_session_id in self._interrupted
                self._interrupted.discard(turn_session_id)
                yield events_.StreamEnd(
                    message=assistant,
                    usage=_usage(message),
                    finish_reason=_finish_reason(
                        message, interrupted=interrupted, hint=self._auth_hint()
                    ),
                )

    # -- stored conversations -------------------------------------------------

    async def list_sessions(self) -> list[_session.SessionInfo]:
        assert self._workspace is not None
        if self._store is not None:
            infos = await claude_agent_sdk.list_sessions_from_store(
                self._store, self._workspace.path
            )
        else:
            infos = await asyncio.to_thread(
                claude_agent_sdk.list_sessions, self._workspace.path
            )
        return [
            _session.SessionInfo(
                kind=KIND,
                session_id=i.session_id,
                title=i.custom_title or i.summary or i.first_prompt,
                cwd=i.cwd,
                updated_at=i.last_modified,
                created_at=i.created_at,
            )
            for i in infos
        ]

    async def history(
        self, session_id: str, *, limit: int | None = None, offset: int = 0
    ) -> list[messages_.Message]:
        """Return the CLI's own transcript, in full.

        `SessionMessage.message` is the raw Anthropic message, so every
        content block survives — text, thinking, tool_use and tool_result —
        and is mapped to the same AI SDK parts a live turn produces.
        """
        assert self._workspace is not None
        await self._require_known(session_id)
        try:
            if self._store is not None:
                stored = await claude_agent_sdk.get_session_messages_from_store(
                    self._store, session_id, self._workspace.path, limit, offset
                )
            else:
                stored = await asyncio.to_thread(
                    claude_agent_sdk.get_session_messages,
                    session_id,
                    self._workspace.path,
                    limit,
                    offset,
                )
        except Exception as exc:
            raise errors.ResumeFailedError(session_id, str(exc)) from exc
        messages: list[messages_.Message] = []
        for record in stored:
            messages.extend(_stored_to_messages(record))
        return messages

    async def fork(self, session_id: str) -> str:
        assert self._workspace is not None
        await self._require_known(session_id)
        try:
            if self._store is not None:
                # The SDK remaps every UUID and rewrites the chain; this
                # store only carries the bytes into the VM.
                result = await claude_agent_sdk.fork_session_via_store(
                    self._store, session_id, self._workspace.path
                )
            else:
                result = await asyncio.to_thread(
                    claude_agent_sdk.fork_session,
                    session_id,
                    self._workspace.path,
                )
            forked = result.session_id
            await self._acquire(forked)
            await self._connect(forked, resume=forked)
            await self._publish_pid(forked)
        except errors.ResumeFailedError:
            raise
        except Exception as exc:
            raise errors.ResumeFailedError(
                session_id, f"fork failed: {exc}"
            ) from exc
        return forked

    async def _drain_interrupted(self, session_id: str) -> None:
        if session_id not in self._interrupted:
            return

        client = self._clients.get(session_id)
        self._interrupted.discard(session_id)
        if client is None:
            return
        with contextlib.suppress(Exception):
            async with asyncio.timeout(30):
                async for message in client.receive_response():
                    if isinstance(message, claude_agent_sdk.ResultMessage):
                        break

    async def steer(self, session_id: str, text: str) -> None:
        """Send a message to the running turn.

        `query()` writes straight to the CLI's stdin without waiting, so a
        message can land while the turn is still running.
        """
        if not self.capabilities.steer:
            raise errors.UnsupportedError(KIND, "steer")
        await self._client(session_id).query(text)

    async def stop(self, session_id: str) -> None:
        self._interrupted.add(session_id)
        with contextlib.suppress(Exception):
            await self._client(session_id).interrupt()


def _usage(message: Any) -> Any:
    raw = message.usage or {}
    return usage_.Usage(
        input_tokens=int(raw.get("input_tokens", 0)),
        output_tokens=int(raw.get("output_tokens", 0)),
        cache_read_tokens=int(raw.get("cache_read_input_tokens", 0)),
        cache_write_tokens=int(raw.get("cache_creation_input_tokens", 0)),
        # Cost has no home in the AI SDK's Usage model; the harness's own
        # numbers ride in `raw` rather than being dropped or invented.
        raw={"cost_usd": message.total_cost_usd, "claude": raw},
    )


AUTH_HINT = (
    "claude authenticates where it RUNS. On a Local workspace that is your "
    "own login — run `claude` once and sign in. A sandbox is another "
    "machine that has never seen it, so give it a way to reach a model: "
    "VercelSandbox(gateway=vercel_ai_gateway()) injects a gateway key at "
    "egress, or VercelSandbox(env={'ANTHROPIC_API_KEY': ...}) puts a key of "
    "its own in the VM."
)
BROKERED_HINT = (
    "This workspace injects the credential into requests to {host} at egress, "
    "so the CLI never held it and no login in the VM is involved. A 401 here "
    "means the gateway rejected the injected key — check AI_GATEWAY_API_KEY on "
    "the machine that opened the sandbox — or the egress policy is not "
    "rewriting requests to that host."
)


def gateway_env(gateway: _gateway.Gateway | None) -> dict[str, str]:
    """Tell Claude Code where its model is, through its environment.

    This is the one place that spelling lives.
    """
    if gateway is None:
        return {}
    return {
        "ANTHROPIC_BASE_URL": f"{gateway.base_url}/claude-code",
        "ANTHROPIC_AUTH_TOKEN": gateway.credential,
        # Required, and required to be EMPTY: Claude Code checks this first,
        # so any value here would send the traffic straight past the gateway
        # while everything still appeared to work.
        "ANTHROPIC_API_KEY": "",
        # Puts every gateway model in the CLI's own picker.
        "CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY": "1",
    }


def _finish_reason(
    message: Any, *, interrupted: bool = False, hint: str = AUTH_HINT
) -> str:
    subtype = message.subtype or ""
    if interrupted:
        # We asked for this. The CLI reports an error result after an
        # interrupt; surfacing that as a failure would turn every
        # deliberate stop into an exception.
        return "cancelled"
    if subtype == "error_max_turns":
        return "length"
    if getattr(message, "is_error", False):
        # An errored turn is a failure, not the model declining. "refusal"
        # is reserved for the model saying no, and an infrastructure error
        # is not that.
        detail = str(
            getattr(message, "result", None) or subtype or "unknown error"
        )[:500]
        if errors._looks_unauthenticated(detail):
            raise workspace_errors.NotAuthenticatedError(KIND, detail, hint)
        raise errors.TurnFailedError(KIND, detail)
    return "stop"


def _translate(exc: BaseException) -> BaseException:
    """One harness library exception -> this SDK's taxonomy."""
    if isinstance(exc, claude_agent_sdk.CLINotFoundError):
        return errors.ExecutableMissingError(KIND, "claude", INSTALL)
    if isinstance(exc, claude_agent_sdk.ProcessError):
        return errors.AgentCrashedError(
            KIND, str(exc), getattr(exc, "exit_code", None)
        )
    if isinstance(exc, claude_agent_sdk.CLIConnectionError):
        return errors.AgentCrashedError(KIND, f"lost the CLI connection: {exc}")
    if isinstance(exc, claude_agent_sdk.CLIJSONDecodeError):
        return errors.AgentCrashedError(
            KIND, f"unparseable output from the CLI: {exc}"
        )
    return exc


async def _guard(stream: Any) -> Any:
    """Re-raise harness-library failures as ours, mid-stream included.

    A turn that dies half way through must not surface as claude-agent-sdk's
    internal exception type — the caller cannot be expected to import it.
    """
    try:
        async for item in stream:
            yield item
    except claude_agent_sdk.ClaudeSDKError as exc:
        raise _translate(exc) from exc


def _stream_events(
    raw: dict[str, Any], open_blocks: dict[int, str]
) -> list[Any]:
    """One raw Anthropic stream event -> AI SDK events.

    `open_blocks` tracks which content-block index is text vs thinking, so
    the closing event emits the matching End.
    """
    kind = raw.get("type")
    index = raw.get("index", 0)
    out: list[Any] = []
    block_id = f"b{index}"
    if kind == "content_block_start":
        block = raw.get("content_block") or {}
        block_type = block.get("type")
        if block_type == "text":
            open_blocks[index] = "text"
            out.append(events_.TextStart(block_id=block_id))
        elif block_type == "thinking":
            open_blocks[index] = "thinking"
            out.append(events_.ReasoningStart(block_id=block_id))
    elif kind == "content_block_delta":
        delta = raw.get("delta") or {}
        delta_type = delta.get("type")
        if delta_type == "text_delta" and delta.get("text"):
            out.append(
                events_.TextDelta(chunk=delta["text"], block_id=block_id)
            )
        elif delta_type == "thinking_delta" and delta.get("thinking"):
            out.append(
                events_.ReasoningDelta(
                    chunk=delta["thinking"], block_id=block_id
                )
            )
    elif kind == "content_block_stop":
        opened = open_blocks.pop(index, None)
        if opened == "text":
            out.append(events_.TextEnd(block_id=block_id))
        elif opened == "thinking":
            out.append(events_.ReasoningEnd(block_id=block_id))
    return out


def _stored_to_messages(record: Any) -> list[messages_.Message]:
    """One stored transcript record -> AI SDK messages.

    Tool results become their own role="tool" message, exactly as a live
    turn produces them, so a replayed conversation and a watched one are
    indistinguishable downstream.
    """
    raw = record.message if isinstance(record.message, dict) else {}
    role = raw.get("role") or record.type
    content = raw.get("content")
    blocks = (
        content
        if isinstance(content, list)
        else (
            [{"type": "text", "text": content}]
            if isinstance(content, str)
            else []
        )
    )
    parts: list[Any] = []
    tool_results: list[Any] = []
    for block in blocks:
        if not isinstance(block, dict):
            continue
        kind = block.get("type")
        if kind == "text" and block.get("text"):
            parts.append(messages_.TextPart(text=block["text"]))
        elif kind == "thinking" and block.get("thinking"):
            parts.append(messages_.ReasoningPart(text=block["thinking"]))
        elif kind == "tool_use":
            parts.append(
                messages_.ToolCallPart(
                    tool_call_id=block.get("id", ""),
                    tool_name=block.get("name", ""),
                    tool_args=json.dumps(block.get("input") or {}),
                )
            )
        elif kind == "tool_result":
            tool_results.append(
                messages_.ToolResultPart(
                    tool_call_id=block.get("tool_use_id", ""),
                    tool_name="",
                    result=block.get("content"),
                    result_kind="error" if block.get("is_error") else "json",
                )
            )
    out: list[messages_.Message] = []
    if parts and role in ("user", "assistant"):
        out.append(messages_.Message(role=role, parts=parts, replay=True))
    if tool_results:
        out.append(
            messages_.Message(role="tool", parts=tool_results, replay=True)
        )
    return out
