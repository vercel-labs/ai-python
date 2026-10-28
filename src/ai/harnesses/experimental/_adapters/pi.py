"""Pi, driven through `pi --mode rpc`.

One pi process per conversation, like claude: an RPC process holds one
active session. The id we hand pi (`--session-id`) IS its session id, so
`pi --session <id>` opens the same conversation in pi's own TUI.

pi has no permission prompt. Approval is an extension this SDK loads into
every pi it launches (`pi_extension.js`): its `tool_call` handler asks
through an RPC dialog, and this adapter answers it. The protocol shapes
here were measured against a real pi (see VERIFIED_VERSION).
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import datetime
import json
import pathlib
import re
import uuid
import warnings
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
    _input,
    _session,
    _session_lock,
    errors,
)
from . import base

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

KIND = "pi"
PACKAGE = "@earendil-works/pi-coding-agent"
INSTALL = f"npm install -g {PACKAGE}"
VERIFIED_VERSION = "1.3.0"
"""The CLI the live suite last passed against. Older ones get a warning."""
APPROVE = "ai-python:approve"
"""The dialog title the extension asks approvals under."""
SESSION_ID = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9._-]{0,198}[A-Za-z0-9])?\Z")
"""pi's rule for a session id (`--session-id`): letters, digits, `.`, `_`
and `-`, starting and ending with a letter or digit. Never a path."""
READY = "ai-python-ready"
"""The command the extension registers last: proof that it loaded."""
CONFIG_VAR = "AI_PYTHON_PI"
"""The environment variable that configures the extension."""
# Internal markers delivered on a session's event queue: pi died, or this
# client closed the connection under a running turn.
CRASHED = "__crashed__"
CLOSED = "__closed__"
DIALOGS = ("select", "confirm", "input", "editor")
"""Extension UI methods that wait for an answer; the rest are notices."""
PROVIDER = "vercel-ai-gateway"
"""pi's built-in provider for the Vercel AI Gateway."""

# The hook is pi's `tool_call` event, which fires for every tool call —
# built-in, extension, MCP, and the nested calls a codemode script makes —
# and can rewrite the input in place or block with a reason the agent reads.
CAPABILITIES = _capabilities.Capabilities(
    steer=True,
    stop=True,
    resume=True,
    fork=True,
    history=True,
    rewrite_tool_input=True,
    deny_reason=True,
    images=True,
    approval="all",
)


class PiAdapter:
    kind = KIND
    capabilities = CAPABILITIES
    verified_version: str | None = VERIFIED_VERSION

    def __init__(
        self,
        executable: str = "pi",
        model: str | None = None,
        gateway: _gateway.Gateway | None = None,
        effort: str | None = None,
        instructions: str | None = None,
        mcp_servers: dict[str, Any] | None = None,
        tools: list[str] | None = None,
        disallowed_tools: list[str] | None = None,
    ) -> None:
        self._executable = executable
        self._model = model
        self._gateway = gateway
        self._effort = effort
        self._instructions = instructions
        self._mcp_servers = dict(mcp_servers or {})
        self._tools = tools
        self._disallowed_tools = list(disallowed_tools or [])
        self._version: str | None = None
        self._workspace: _base.Workspace | None = None
        self._extension: str | None = None
        self._lock: _session_lock.SessionLock | None = None
        self._held: set[str] = set()
        self._opening: dict[str, asyncio.Lock] = {}
        self._agents: dict[str, _Agent] = {}
        # Sessions whose turn WE ended: they settle as cancelled.
        self._interrupted: set[str] = set()
        # Sessions with a prompt accepted and no `agent_settled` yet.
        self._running: set[str] = set()
        # An error raised while answering an approval cannot reach anyone
        # where it happens; it is parked and raised on that session's turn.
        self._pending_error: dict[str, BaseException] = {}
        self.approve: base.ApprovalHook | None = None
        self.approval_timeout: float = 300.0
        self.on_approval_error: Any = None
        self.live_history: Any = lambda session_id: []

    @property
    def version(self) -> str | None:
        return self._version

    @property
    def process_ids(self) -> list[int]:
        return [
            pid
            for agent in self._agents.values()
            if isinstance(pid := getattr(agent.process, "pid", None), int)
        ]

    # -- lifecycle ------------------------------------------------------------

    async def start(
        self, workspace: _base.Workspace, options: dict[str, Any]
    ) -> None:
        if not workspace.duplex_spawn:
            raise errors.UnsupportedError(
                KIND,
                f"{workspace.kind} workspaces",
                "pi's RPC mode speaks JSON lines on stdin, which this "
                + "workspace cannot provide",
            )
        # An explicit gateway wins; otherwise the workspace's. Where the
        # workspace injects the credential at egress, pi only ever holds the
        # placeholder: the extension sends it as `Authorization: Bearer`,
        # the header the firewall rewrites.
        gateway = self._gateway or workspace.gateway
        if gateway is not None and workspace.injects_credentials_for(
            gateway.host
        ):
            gateway = gateway.brokered()
        self._gateway = gateway
        probe = await base.provision(workspace, self._executable, PACKAGE)
        if probe.exit_code != 0:
            raise errors.ExecutableMissingError(KIND, self._executable, INSTALL)
        self._version = probe.stdout.strip().splitlines()[0] or None
        self._workspace = workspace
        self._lock = _session_lock.SessionLock(workspace, KIND)
        # The extension travels to wherever pi runs, as a file: pi loads
        # extensions by path.
        self._extension = f"{await workspace.home()}/.ai-python/pi/extension.js"
        await workspace.write_text(
            self._extension,
            pathlib.Path(__file__).with_name("pi_extension.js").read_text(),
        )

    async def close(self) -> None:
        for session_id in list(self._agents):
            await self.close_session(session_id)

    async def detach(self) -> None:
        """Let go of every session without terminating pi.

        Stop reading and drop the input conduit; where the workspace outlives
        the client (a kept sandbox) the process runs on. The session locks
        stay held: the process is still the writer.
        """
        for session_id, agent in list(self._agents.items()):
            await agent.close()
            with contextlib.suppress(Exception):
                await agent.process.detach()
            self._agents.pop(session_id, None)

    def _args(self) -> list[str]:
        """How this harness runs pi: headless and in the TUI alike."""
        assert self._extension is not None
        args = ["-e", self._extension]
        model = self._model
        if (
            model
            and self._gateway is not None
            and not model.startswith(f"{PROVIDER}/")
        ):
            # With a gateway, the model is the gateway's: `anthropic/...`
            # names it there, not at Anthropic.
            model = f"{PROVIDER}/{model}"
        if model:
            args += ["--model", model]
        if self._effort:
            args += ["--thinking", self._effort]
        if self._instructions:
            args += ["--append-system-prompt", self._instructions]
        if self._tools is not None:
            args += (
                ["--tools", ",".join(self._tools)]
                if self._tools
                else ["--no-tools"]
            )
        if self._disallowed_tools:
            args += ["--exclude-tools", ",".join(self._disallowed_tools)]
        return args

    def _env(self, *, approve: bool) -> dict[str, str]:
        """Return the extension's configuration, and the gateway's key."""
        config: dict[str, Any] = {"approve": approve}
        if self._mcp_servers:
            config["mcp"] = mcp_config(self._mcp_servers)
        env: dict[str, str] = {}
        if self._gateway is not None:
            base_url = self._gateway.base_url
            config["gateway"] = (
                {"baseUrl": base_url}
                if base_url != _gateway.DEFAULT_BASE_URL
                else {}
            )
            env[_gateway.KEY_VAR] = self._gateway.credential
        env[CONFIG_VAR] = json.dumps(config)
        return env

    def _auth_hint(self) -> str:
        if self._gateway is not None and self._gateway.is_brokered:
            return BROKERED_HINT.format(host=self._gateway.host)
        return AUTH_HINT

    # -- sessions -------------------------------------------------------------

    async def new_session(
        self, history: list[messages_.Message] | None = None
    ) -> str:
        session_id = (
            await self.stage_history(history) if history else str(uuid.uuid4())
        )
        await self._acquire(session_id)
        try:
            await self._spawn(session_id, ["--session-id", session_id])
        except BaseException:
            await self._release(session_id)
            raise
        return session_id

    async def resume_session(self, session_id: str) -> None:
        # Checked first: `--session-id` CREATES a session it cannot find,
        # so a typo would silently start an empty conversation.
        await self._find(session_id)
        # One launch per conversation at a time: two resumes racing through
        # the hand-over would each start a pi (two writers, one untracked).
        async with self._opening.setdefault(session_id, asyncio.Lock()):
            await self._hand_over(session_id)
            await self._acquire(session_id)
            try:
                await self._spawn(session_id, ["--session-id", session_id])
            except BaseException as exc:
                # Whatever ended the launch — a crash, a timeout, a cancelled
                # task — nothing holds the conversation now: free the claim,
                # or every other client is refused until it goes stale.
                await self._release(session_id)
                if isinstance(exc, errors.HarnessError):
                    raise errors.ResumeFailedError(
                        session_id, str(exc)
                    ) from exc
                raise

    async def fork(self, session_id: str) -> str:
        found = await self._find(session_id)
        forked = str(uuid.uuid4())
        await self._acquire(forked)
        try:
            # By path, not id: pi matches a partial id, a path is exact.
            await self._spawn(
                forked, ["--fork", found["path"], "--session-id", forked]
            )
        except BaseException as exc:
            await self._release(forked)  # see resume_session
            if isinstance(exc, errors.HarnessError):
                raise errors.ResumeFailedError(
                    session_id, f"fork failed: {exc}"
                ) from exc
            raise
        return forked

    async def close_session(self, session_id: str) -> None:
        agent = self._agents.pop(session_id, None)
        self._running.discard(session_id)
        if agent is not None:
            await agent.close()
            with contextlib.suppress(Exception):
                await agent.process.terminate()
        await self._release(session_id)

    async def _spawn(self, session_id: str, args: list[str]) -> None:
        """Launch pi on one conversation and wait until it answers.

        A pi that cannot start — an unknown model, a bad flag — exits at
        once, and the handshake reports its last words.
        """
        assert self._workspace is not None
        process = await self._workspace.spawn(
            [self._executable, "--mode", "rpc", *args, *self._args()],
            env=self._env(approve=self.approve is not None),
        )
        agent = _Agent(process, on_dialog=self._dialog_handler(session_id))
        agent.start()
        try:
            state = await agent.call({"type": "get_state"}, timeout=60)
            reported = (state or {}).get("sessionId")
            if reported and reported != session_id:
                raise errors.AgentCrashedError(
                    KIND, f"pi opened session {reported}, not {session_id}"
                )
            # The approval gate must be there before any work is accepted:
            # a gate that failed to load is a gate that fails open.
            listed = await agent.call({"type": "get_commands"}, timeout=60)
            if READY not in {
                c.get("name") for c in (listed or {}).get("commands") or []
            }:
                raise errors.AgentCrashedError(
                    KIND,
                    "pi did not load the SDK's extension "
                    + f"({self._extension}); refusing to run without it",
                )
            # Published before the session is registered: whatever fails
            # here, the process goes with it, never outliving its claim.
            if self._lock is not None:
                pid = await process.os_pid()
                await self._lock.claim_pid(session_id, pid, client=KIND)
        except BaseException:
            await agent.close()
            with contextlib.suppress(Exception):
                await process.terminate()
            raise
        self._agents[session_id] = agent

    async def _hand_over(self, session_id: str) -> None:
        """One writer per conversation, this client included.

        The lock is per client, so it lets this client through. A second pi
        on a conversation this client already has open would be a second
        writer on one transcript: the one already open is ended first (the
        claim stays ours), unless a turn is running on it.
        """
        if session_id in self._running:
            raise errors.SessionBusyError(
                session_id,
                None,
                "a turn is running on it in this client; stop() it first",
            )
        agent = self._agents.pop(session_id, None)
        if agent is not None:
            await agent.close()
            with contextlib.suppress(Exception):
                await agent.process.terminate()

    async def _acquire(self, session_id: str) -> None:
        """Refuse before launching a second pi: one writer per conversation."""
        if self._lock is None or session_id in self._held:
            return
        await self._lock.acquire(session_id, client=KIND)
        self._held.add(session_id)

    async def _release(self, session_id: str) -> None:
        self._held.discard(session_id)
        if self._lock is not None:
            with contextlib.suppress(Exception):
                await self._lock.release(session_id)

    # -- the session store ----------------------------------------------------

    async def _store(self, op: str, session_id: str = "") -> Any:
        """Ask the workspace about pi's session files: one command, one trip."""
        assert self._workspace is not None
        result = await self._workspace.exec(
            ["python3", "-c", _STORE, op, self._workspace.path, session_id],
            timeout=60,
        )
        if result.exit_code != 0:
            raise errors.HarnessError(
                f"could not read pi's sessions in the workspace (exit "
                f"{result.exit_code}): "
                + (result.stderr.strip()[-300:] or "no output")
                + ". Harnesses need python3 where the CLI runs."
            )
        return json.loads(result.stdout.strip().splitlines()[-1])

    async def _find(self, session_id: str) -> dict[str, Any]:
        """Return a conversation of THIS workspace, or raise ResumeFailed."""
        assert self._workspace is not None
        if not SESSION_ID.match(session_id):
            raise errors.ResumeFailedError(
                session_id, "not a pi session id (see SESSION_ID)"
            )
        found = await self._store("find", session_id)
        if not found:
            raise errors.ResumeFailedError(
                session_id,
                f"no such session in workspace {self._workspace.path}",
            )
        return dict(found)

    async def list_sessions(self) -> list[_session.SessionInfo]:
        return [
            _session.SessionInfo(
                kind=KIND,
                session_id=s["id"],
                title=s.get("title"),
                cwd=s.get("cwd"),
                updated_at=s.get("updated"),
                created_at=s.get("created"),
            )
            for s in await self._store("list")
        ]

    async def running_sessions(self) -> dict[str, int | None]:
        """Conversations a client of this SDK, or its `tui()`, has open.

        A pi started in a terminal is not seen: it renames its process to
        `pi`, so `ps` shows no session in its arguments, and it records no
        process per conversation.
        """
        return await self._lock.holders() if self._lock is not None else {}

    async def history(
        self, session_id: str, *, limit: int | None = None, offset: int = 0
    ) -> list[messages_.Message]:
        """Return the active branch of the session file, in full.

        Every message on it, from the root to the latest entry — including
        what a compaction summarized away, which pi keeps on disk.
        """
        assert self._workspace is not None
        found = await self._find(session_id)
        try:
            text = await self._workspace.read_text(found["path"])
        except Exception as exc:
            raise errors.ResumeFailedError(session_id, str(exc)) from exc
        messages: list[messages_.Message] = []
        for record in _active_branch(text):
            messages.extend(_stored_to_messages(record, replay=True))
        window = messages[offset:]
        return window[:limit] if limit is not None else window

    async def stage_history(self, history: list[messages_.Message]) -> str:
        """Write a session file with this past, and start nothing on it.

        Where pi will look for it: the session directory of this workspace,
        under a fresh id that `--session-id` then opens.
        """
        assert self._workspace is not None
        session_id = str(uuid.uuid4())
        _handoff.warn_unsupported_files(history, KIND)
        dropped = _handoff.reasoning_count(history)
        if dropped:
            # pi itself drops thinking that another model produced (its
            # signature is only valid for that model), and replaying it as
            # user text is refused by Anthropic's safeguards — pi often runs
            # Claude. Say so rather than drop it silently.
            warnings.warn(
                f"{dropped} reasoning part(s) in the history were not carried "
                + "into pi: a model only reads its own reasoning. Text, tool "
                + "calls and images were carried in full.",
                stacklevel=2,
            )
        where = await self._store("dir")
        now = datetime.datetime.now(datetime.UTC)
        stamp = now.isoformat(timespec="milliseconds").replace("+00:00", "Z")
        lines: list[dict[str, Any]] = [
            {
                "type": "session",
                "version": 3,
                "id": session_id,
                "timestamp": stamp,
                "cwd": where["cwd"],
            }
        ]
        parent: str | None = None
        # The model the conversation was on, as pi records a switch. pi
        # restores a session's model from its last reply's author, which
        # `to_pi_messages` keeps for replies pi wrote; this covers a past
        # whose last reply is not one. Measured: without either, a pushed
        # conversation ran on pi's default model, which the gateway refused.
        # Only a conversation pi itself wrote says which; from another
        # harness, pi's own choice applies.
        written: dict[str, Any] = next(
            (
                found
                for m in reversed(history)
                if m.role == "assistant"
                and isinstance(
                    found := (m.provider_metadata or {}).get(KIND), dict
                )
            ),
            {},
        )
        changes: list[dict[str, Any]] = []
        if written.get("provider") and written.get("model"):
            changes.append(
                {
                    "type": "model_change",
                    "provider": written["provider"],
                    "modelId": written["model"],
                }
            )
        if written.get("thinkingLevel"):
            changes.append(
                {
                    "type": "thinking_level_change",
                    "thinkingLevel": written["thinkingLevel"],
                }
            )
        for change in changes:
            entry_id = uuid.uuid4().hex[:8]
            lines.append(
                {
                    **change,
                    "id": entry_id,
                    "parentId": parent,
                    "timestamp": stamp,
                }
            )
            parent = entry_id
        for message in _handoff.to_pi_messages(
            history, timestamp=int(now.timestamp() * 1000)
        ):
            entry_id = uuid.uuid4().hex[:8]
            lines.append(
                {
                    "type": "message",
                    "id": entry_id,
                    "parentId": parent,
                    "timestamp": stamp,
                    "message": message,
                }
            )
            parent = entry_id
        name = stamp.replace(":", "-").replace(".", "-")
        await self._workspace.write_text(
            f"{where['dir']}/{name}_{session_id}.jsonl",
            "".join(json.dumps(line) + "\n" for line in lines),
        )
        return session_id

    # -- the TUI --------------------------------------------------------------

    async def prepare_tui(self) -> None:
        """Nothing to do: pi's TUI has no first-run walk to get past."""

    def tui_launch(
        self, session_id: str | None
    ) -> tuple[list[str], dict[str, str], str | None]:
        assert self._workspace is not None
        running_as = session_id or str(uuid.uuid4())
        argv = [self._executable, "--session-id", running_as, *self._args()]
        # The person at the TUI is the gate: the extension asks nothing.
        env = {
            "TERM": "xterm-256color",
            **dict(self._workspace.env),
            **self._env(approve=False),
        }
        return argv, env, running_as

    async def claim_new_session(self, session_id: str) -> None:
        await self._acquire(session_id)

    async def record_tui_pid(self, session_id: str, pid: int | None) -> None:
        if self._lock is not None and pid is not None:
            await self._lock.claim_pid(session_id, pid, client=KIND)

    async def release_tui_lock(self, session_id: str) -> None:
        await self._release(session_id)

    async def resume_session_lock(self, session_id: str) -> None:
        await self._find(session_id)
        async with self._opening.setdefault(session_id, asyncio.Lock()):
            await self._hand_over(session_id)
            await self._acquire(session_id)

    # -- approval -------------------------------------------------------------

    def _dialog_handler(self, session_id: str) -> Any:
        async def on_dialog(agent: _Agent, request: dict[str, Any]) -> None:
            if request.get("method") not in DIALOGS:
                return  # a notice: nothing waits for an answer
            reply: dict[str, Any] = {
                "type": "extension_ui_response",
                "id": request.get("id"),
            }
            if request.get("method") != "input" or request.get("title") != (
                APPROVE
            ):
                # Some other extension asking a person. Nobody is there:
                # dismiss it rather than leave the agent waiting.
                agent.pending.pop(str(request.get("id")), None)
                await agent.send({**reply, "cancelled": True})
                return
            if session_id in self._interrupted:
                # The turn is being stopped: a question asked now is
                # withdrawn at once, or `abort` would wait on it.
                agent.pending.pop(str(request.get("id")), None)
                await agent.send({**reply, "cancelled": True})
                return
            decision = await self._judge(session_id, request)
            # Answered from here on: `dismiss` must not withdraw it too.
            agent.pending.pop(str(request.get("id")), None)
            await agent.send({**reply, "value": json.dumps(decision)})
            if decision.get("stop"):
                # Only now: the answer must reach pi before it is aborted.
                with contextlib.suppress(Exception):
                    await agent.call({"type": "abort"}, timeout=30)

        return on_dialog

    async def _judge(
        self, session_id: str, request: dict[str, Any]
    ) -> dict[str, Any]:
        """Ask the hook about one call; always come back with an answer.

        A hook that fails or runs out of time denies (a gate that fails open
        is not a gate) and is reported. A decision pi cannot honor denies
        and is parked to raise on the turn.
        """
        try:
            payload = json.loads(request.get("placeholder") or "{}")
            assert self._workspace is not None
            if self.approve is None:
                return {"decision": "allow"}
            parent = payload.get("parentToolCallId")
            call = _approval.ToolCall(
                id=str(payload.get("toolCallId") or ""),
                name=str(payload.get("toolName") or ""),
                input=dict(payload.get("input") or {}),
                hints={"parentToolCallId": parent} if parent else {},
                raw=payload,
            )
            context = _approval.ApprovalContext(
                call=call,
                history=list(self.live_history(session_id)),
                workspace=self._workspace,
                harness=KIND,
                session_id=session_id,
            )
            async with asyncio.timeout(self.approval_timeout):
                decision = await self.approve(context)
        except Exception as exc:
            self._record_error(session_id, f"{type(exc).__name__}: {exc}")
            return {"decision": "deny", "reason": "approval failed; denied"}
        if isinstance(decision, _approval.Allow):
            if decision.remember:
                self._pending_error[session_id] = errors.UnsupportedError(
                    KIND,
                    "remember",
                    "pi has no permission rules to remember a decision in",
                )
                return {"decision": "deny", "reason": "remember is unsupported"}
            return {"decision": "allow", "input": decision.input}
        if isinstance(decision, _approval.Deny):
            if decision.stop:
                self._interrupted.add(session_id)
            return {
                "decision": "deny",
                "reason": decision.reason or "denied by policy",
                "stop": decision.stop,
            }
        return {"decision": "deny", "reason": "approval hook returned nothing"}

    def _record_error(self, session_id: str, detail: str) -> None:
        if callable(self.on_approval_error):
            self.on_approval_error(session_id, detail)

    # -- turns ----------------------------------------------------------------

    def _agent(self, session_id: str) -> _Agent:
        agent = self._agents.get(session_id)
        if agent is None:
            raise RuntimeError(f"unknown session {session_id}")
        return agent

    async def turn(
        self, session_id: str, prompt: list[_input.Part]
    ) -> AsyncIterator[events_.AgentEvent]:
        agent = self._agent(session_id)
        if agent.dead is not None:
            raise errors.AgentCrashedError(KIND, agent.dead)
        # Whatever the last turn left behind (a stopped turn's tail) is not
        # this turn's: `abort` answers only once pi is idle, so it has all
        # arrived by now.
        while not agent.events.empty():
            agent.events.get_nowait()
        self._interrupted.discard(session_id)
        self._pending_error.pop(session_id, None)
        command = await _prompt_command(prompt)
        if session_id in self._interrupted:
            # Stopped while its files were being read: the abort found pi
            # idle, so nothing would stop a prompt sent now. Send none.
            self._interrupted.discard(session_id)
            yield events_.StreamStart()
            yield events_.StreamEnd(
                message=messages_.Message(role="assistant", parts=[]),
                finish_reason="cancelled",
            )
            return
        try:
            accepted = await agent.call(command)
        except _CommandError as exc:
            raise _turn_failure(str(exc), hint=self._auth_hint()) from exc
        if (accepted or {}).get("disposition") == "handled":
            # An extension command consumed it. The command may start a run
            # of its own (`pi.sendUserMessage`) — measured: then pi is
            # already streaming when it answers, and that run is this turn's.
            state = await agent.call({"type": "get_state"})
            if not (state or {}).get("isStreaming"):
                yield events_.StreamStart()
                yield events_.StreamEnd(
                    message=messages_.Message(role="assistant", parts=[]),
                    finish_reason="stop",
                )
                return
        # Running BEFORE the first event: a consumer may steer on it.
        self._running.add(session_id)
        parts: list[Any] = []
        # Where the latest assistant message's parts are in `parts`: pi may
        # abandon that attempt and retry it (see `agent_end`).
        attempt = (0, 0)
        usage = usage_.Usage()
        cost = 0.0
        started: set[str] = set()
        open_blocks: dict[str, str] = {}
        message_no = 0
        last: dict[str, Any] = {}
        retry_error = ""
        settled: dict[str, Any] = {}
        try:
            yield events_.StreamStart()
            while True:
                event = await agent.events.get()
                kind = event.get("type")
                if kind == CRASHED:
                    raise errors.AgentCrashedError(
                        KIND, event.get("detail") or "exited mid-turn"
                    )
                if kind == CLOSED:
                    raise errors.HarnessClosedError(
                        "the session was closed while its turn was running"
                    )
                message = event.get("message") or {}
                if kind == "message_start" and message.get("role") == (
                    "assistant"
                ):
                    message_no += 1
                elif kind == "message_update":
                    for out in _stream_events(
                        event.get("assistantMessageEvent") or {},
                        f"{message_no}",
                        open_blocks,
                        started,
                    ):
                        yield out
                elif kind == "message_end" and message.get("role") == (
                    "assistant"
                ):
                    last = message
                    begun = len(parts)
                    if message.get("stopReason") != "error":
                        # A failed attempt is not the answer: pi retries it
                        # (or the turn raises below), and its partial text
                        # would be glued onto the reply that replaces it.
                        parts += _assistant_parts(message.get("content"))
                    attempt = (begun, len(parts))
                    raw = message.get("usage") or {}
                    usage = usage + _usage(raw)
                    cost += float((raw.get("cost") or {}).get("total") or 0)
                    for out in _close_blocks(open_blocks):
                        yield out
                elif kind == "tool_execution_end" and not event.get(
                    "parentToolCallId"
                ):
                    # A nested call (a codemode script's) is not in the
                    # transcript: only the tool that made it reports it.
                    result = _tool_result(
                        event.get("toolCallId") or "",
                        event.get("toolName") or "",
                        event.get("result") or {},
                        is_error=bool(event.get("isError")),
                    )
                    yield events_.ToolCallResult(
                        message=messages_.Message(role="tool", parts=[result]),
                        results=[result],
                    )
                elif kind == "agent_end" and event.get("willRetry"):
                    # pi abandons the run's last response and retries it as
                    # a fresh run — after a cut-off (`length`) answer, it
                    # compacts first. That response is not the answer.
                    if attempt[1] == len(parts):
                        del parts[attempt[0] :]
                    attempt = (len(parts), len(parts))
                elif kind == "auto_retry_end" and not event.get("success"):
                    retry_error = str(event.get("finalError") or "")
                elif kind == "agent_settled":
                    settled = event
                    break
        finally:
            self._running.discard(session_id)
        error = self._pending_error.pop(session_id, None)
        if error is not None:
            raise error
        interrupted = session_id in self._interrupted or bool(
            settled.get("aborted")
        )
        self._interrupted.discard(session_id)
        stop = last.get("stopReason")
        if stop == "error" and not interrupted:
            raise _turn_failure(
                str(last.get("errorMessage") or retry_error or "unknown error"),
                hint=self._auth_hint(),
            )
        yield events_.StreamEnd(
            message=messages_.Message(
                role="assistant",
                parts=parts,
                provider_metadata=_written_by(last),
            ),
            # Cost has no home in Usage; pi's own number rides in `raw`.
            usage=usage.model_copy(update={"raw": {"cost_usd": cost}}),
            finish_reason=(
                "cancelled"
                if interrupted or stop == "aborted"
                else "length"
                if stop == "length"
                else "stop"
            ),
        )

    async def steer(self, session_id: str, text: str) -> None:
        """Queue guidance for the running turn.

        pi delivers it once the current assistant message's tool calls are
        done, before its next model call.
        """
        if session_id not in self._running:
            raise RuntimeError("no active turn to steer")
        try:
            await self._agent(session_id).call(
                {"type": "steer", "message": text}
            )
        except _CommandError as exc:
            raise errors.TurnFailedError(
                KIND, f"steer rejected: {exc}"
            ) from exc

    async def stop(self, session_id: str) -> None:
        self._interrupted.add(session_id)
        agent = self._agents.get(session_id)
        if agent is None or agent.dead is not None:
            return
        # A tool call waiting on the hook keeps pi busy, and `abort` only
        # answers once pi is idle — measured: stop() timed out and the turn
        # ran on. Withdraw the question first: the call is denied, pi goes
        # idle, and the abort lands at once.
        await agent.dismiss()
        # `abort` answers once pi is idle: everything the turn produced has
        # been sent by then.
        with contextlib.suppress(Exception):
            await agent.call({"type": "abort"}, timeout=30)


class _CommandError(Exception):
    """pi answered a command with `success: false`."""


class _Agent:
    """One `pi --mode rpc` process: its commands, its events, its dialogs.

    Responses resolve the command that asked (by id). Session events go on
    `events`, in order, for whichever turn is reading. A dialog is answered
    in a task of its own: an approval can take minutes, and the reader must
    keep reading meanwhile — including the reply to an `abort` sent from
    inside it.
    """

    def __init__(self, process: _base.Process, *, on_dialog: Any) -> None:
        self.process = process
        self.events: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self.dead: str | None = None
        self._on_dialog = on_dialog
        self._waiters: dict[str, asyncio.Future[Any]] = {}
        self._pump: asyncio.Task[None] | None = None
        self._dialogs: set[asyncio.Task[None]] = set()
        # Dialogs pi is still waiting on, by request id: the ones `dismiss`
        # withdraws. A handler takes its entry out just before it answers.
        self.pending: dict[str, asyncio.Task[None]] = {}

    def start(self) -> None:
        self._pump = asyncio.create_task(self._read())

    async def call(
        self, command: dict[str, Any], *, timeout: float = 120
    ) -> Any:
        if self.dead is not None:
            raise errors.AgentCrashedError(KIND, self.dead)
        request_id = uuid.uuid4().hex
        future: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
        self._waiters[request_id] = future
        try:
            await self.send({**command, "id": request_id})
            async with asyncio.timeout(timeout):
                return await future
        finally:
            self._waiters.pop(request_id, None)

    async def send(self, record: dict[str, Any]) -> None:
        try:
            await self.process.write(json.dumps(record) + "\n")
        except Exception as exc:
            raise errors.AgentCrashedError(
                KIND, f"could not write to pi: {exc}"
            ) from exc

    def _track(self, request_id: str, task: asyncio.Task[None]) -> None:
        self.pending[request_id] = task

        def forget(_: asyncio.Task[None]) -> None:
            if self.pending.get(request_id) is task:
                del self.pending[request_id]

        task.add_done_callback(forget)

    async def dismiss(self) -> None:
        """Withdraw every question pi is still waiting on.

        The hook deciding it is cancelled, and pi hears `cancelled`: the
        extension then blocks the call, so nothing runs on a question nobody
        answered.
        """
        for request_id, task in list(self.pending.items()):
            self.pending.pop(request_id, None)
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
            with contextlib.suppress(Exception):
                await self.send(
                    {
                        "type": "extension_ui_response",
                        "id": request_id,
                        "cancelled": True,
                    }
                )

    async def close(self) -> None:
        if self._pump is not None:
            self._pump.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._pump
            self._pump = None
        for task in list(self._dialogs):
            task.cancel()
        for task in list(self._dialogs):
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        self._dialogs.clear()
        self._fail("connection closed")
        # A turn reading events must hear it too: the producer is gone.
        self.events.put_nowait({"type": CLOSED})

    def _fail(self, detail: str) -> None:
        self.dead = self.dead or detail
        for future in self._waiters.values():
            if not future.done():
                future.set_exception(errors.AgentCrashedError(KIND, detail))
        self._waiters.clear()

    async def _read(self) -> None:
        try:
            while True:
                line = await self.process.readline()
                if not line:
                    detail = await self._death_note()
                    self._fail(detail)
                    # Whoever is waiting on events hears it too.
                    self.events.put_nowait({"type": CRASHED, "detail": detail})
                    return
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue  # stdout is protocol only; tolerate noise
                if not isinstance(record, dict):
                    continue
                kind = record.get("type")
                if kind == "response":
                    self._resolve(record)
                elif kind == "extension_ui_request":
                    task = asyncio.create_task(self._on_dialog(self, record))
                    self._dialogs.add(task)
                    task.add_done_callback(self._dialogs.discard)
                    if record.get("method") in DIALOGS:
                        self._track(str(record.get("id")), task)
                else:
                    self.events.put_nowait(record)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._fail(str(exc))
            self.events.put_nowait({"type": CRASHED, "detail": str(exc)})

    def _resolve(self, record: dict[str, Any]) -> None:
        future = self._waiters.get(str(record.get("id")))
        if future is None or future.done():
            return
        if record.get("success") is False:
            future.set_exception(_CommandError(str(record.get("error") or "")))
        else:
            future.set_result(record.get("data"))

    async def _death_note(self) -> str:
        """Describe how pi died and its last words: a crash to diagnose."""
        code: int | None = None
        tail = ""
        with contextlib.suppress(Exception):
            async with asyncio.timeout(2):
                code = await self.process.wait()
        with contextlib.suppress(Exception):
            async with asyncio.timeout(2):
                tail = (await self.process.read_stderr()).strip()
        how = f"exit code {code}" if code is not None else "exited"
        return f"{how}; stderr: {tail[-800:]}" if tail else f"{how}; no stderr"


async def _prompt_command(prompt: list[_input.Part]) -> dict[str, Any]:
    """Return a prompt as pi's `prompt` command: text, and images as base64.

    Text files travel inline, as text. Images are read HERE (`_input.data`
    downloads a URL first): pi in a sandbox could not fetch one.
    """
    texts: list[str] = []
    images: list[dict[str, Any]] = []
    for part in prompt:
        if isinstance(part, messages_.TextPart):
            texts.append(part.text)
        elif part.media_type.startswith("text/"):
            texts.append(await _input.as_text(part))
        else:
            images.append(
                {
                    "type": "image",
                    "data": base64.b64encode(await _input.data(part)).decode(
                        "ascii"
                    ),
                    "mimeType": part.media_type,
                }
            )
    command: dict[str, Any] = {"type": "prompt", "message": "\n\n".join(texts)}
    if images:
        command["images"] = images
    return command


def _stream_events(
    update: dict[str, Any],
    message: str,
    open_blocks: dict[str, str],
    started: set[str],
) -> list[Any]:
    """One `assistantMessageEvent` -> AI SDK events.

    Block ids are `<message>.<contentIndex>`: pi numbers content blocks per
    assistant message, and a turn holds several. Tool-call arguments stream
    as raw JSON fragments, but they travel here as ONE delta, the final
    arguments, at `toolcall_end`: the hydrator builds a call only from its
    deltas, and fragments are not guaranteed to add up to what pi settles on.
    """
    kind = update.get("type") or ""
    block = f"{message}.{update.get('contentIndex', 0)}"
    out: list[Any] = []
    if kind == "text_start":
        open_blocks[block] = "text"
        out.append(events_.TextStart(block_id=block))
    elif kind == "text_delta" and update.get("delta"):
        out.append(events_.TextDelta(chunk=update["delta"], block_id=block))
    elif kind == "thinking_start":
        open_blocks[block] = "thinking"
        out.append(events_.ReasoningStart(block_id=block))
    elif kind == "thinking_delta" and update.get("delta"):
        out.append(
            events_.ReasoningDelta(chunk=update["delta"], block_id=block)
        )
    elif kind in ("text_end", "thinking_end"):
        out += _close_blocks(open_blocks, only=block)
    elif kind == "toolcall_start" and update.get("id"):
        started.add(update["id"])
        out.append(
            events_.ToolStart(
                tool_call_id=update["id"],
                tool_name=update.get("toolName") or "",
            )
        )
    elif kind == "toolcall_end":
        call = _tool_call(update.get("toolCall") or {})
        if call.tool_call_id not in started:
            started.add(call.tool_call_id)
            out.append(
                events_.ToolStart(
                    tool_call_id=call.tool_call_id, tool_name=call.tool_name
                )
            )
        out.append(
            events_.ToolDelta(
                tool_call_id=call.tool_call_id, chunk=call.tool_args
            )
        )
        out.append(
            events_.ToolEnd(tool_call_id=call.tool_call_id, tool_call=call)
        )
    return out


def _close_blocks(
    open_blocks: dict[str, str], only: str | None = None
) -> list[Any]:
    """End the text and reasoning blocks still open (an aborted message's)."""
    out: list[Any] = []
    for block in [only] if only is not None else list(open_blocks):
        opened = open_blocks.pop(block, None)
        if opened == "text":
            out.append(events_.TextEnd(block_id=block))
        elif opened == "thinking":
            out.append(events_.ReasoningEnd(block_id=block))
    return out


def _tool_call(block: dict[str, Any]) -> messages_.ToolCallPart:
    return messages_.ToolCallPart(
        tool_call_id=str(block.get("id") or ""),
        tool_name=str(block.get("name") or ""),
        tool_args=json.dumps(block.get("arguments") or {}),
    )


def _assistant_parts(content: Any) -> list[Any]:
    parts: list[Any] = []
    for block in content if isinstance(content, list) else []:
        if not isinstance(block, dict):
            continue
        kind = block.get("type")
        if kind == "text" and block.get("text"):
            parts.append(messages_.TextPart(text=block["text"]))
        elif kind == "thinking" and block.get("thinking"):
            parts.append(messages_.ReasoningPart(text=block["thinking"]))
        elif kind == "toolCall":
            parts.append(_tool_call(block))
    return parts


def _content_parts(content: Any) -> list[Any]:
    """Return a user message's parts: from a string, or text and images."""
    if isinstance(content, str):
        return [messages_.TextPart(text=content)] if content else []
    parts: list[Any] = []
    for block in content if isinstance(content, list) else []:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text" and block.get("text"):
            parts.append(messages_.TextPart(text=block["text"]))
        elif block.get("type") == "image" and block.get("data"):
            parts.append(
                messages_.FilePart(
                    data=str(block["data"]),
                    media_type=str(block.get("mimeType") or "image/png"),
                )
            )
    return parts


def _tool_result(
    tool_call_id: str, tool_name: str, result: Any, *, is_error: bool
) -> messages_.ToolResultPart:
    """Return what a tool returned: its text, when that is all it is.

    pi tools return content blocks; text-only content (nearly every call)
    reads as the string it is, anything else stays the blocks themselves.
    """
    content = result.get("content") if isinstance(result, dict) else result
    if isinstance(content, list) and all(
        isinstance(b, dict) and b.get("type") == "text" for b in content
    ):
        content = "".join(b.get("text") or "" for b in content)
    return messages_.ToolResultPart(
        tool_call_id=tool_call_id,
        tool_name=tool_name,
        result=content,
        result_kind="error" if is_error else "json",
    )


def _active_branch(text: str) -> list[dict[str, Any]]:
    """Return the session file's active branch: root to latest entry.

    A session is a tree (`id`/`parentId`); the leaf is the entry appended
    last, and abandoned branches are not part of the conversation.
    """
    entries: dict[str, dict[str, Any]] = {}
    leaf: str | None = None
    for line in text.splitlines():
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if (
            isinstance(entry, dict)
            and entry.get("id")
            and entry.get("type") != ("session")
        ):
            entries[str(entry["id"])] = entry
            leaf = str(entry["id"])
    path: list[dict[str, Any]] = []
    seen: set[str] = set()
    while leaf is not None and leaf in entries and leaf not in seen:
        seen.add(leaf)
        path.append(entries[leaf])
        parent = entries[leaf].get("parentId")
        leaf = str(parent) if parent is not None else None
    return list(reversed(path))


def _stored_to_messages(
    entry: dict[str, Any], *, replay: bool = False
) -> list[messages_.Message]:
    """One session entry -> AI SDK messages, shaped like a live turn's.

    Everything pi sends the model is kept: what an extension injected
    (`custom_message`) and the output of a `!` command (`bashExecution`)
    reach it as user messages, and so they are user messages here; a branch
    summary is the context pi carried over from an abandoned branch.
    Compaction summaries are not: the messages they summarize are all here.
    """
    kind = entry.get("type")
    if kind == "custom_message":
        parts = _content_parts(entry.get("content"))
        return (
            [messages_.Message(role="user", parts=parts, replay=replay)]
            if parts
            else []
        )
    if kind == "branch_summary" and entry.get("summary"):
        text = str(entry["summary"])
        return [
            messages_.Message(
                role="user",
                parts=[messages_.TextPart(text=text)],
                replay=replay,
            )
        ]
    if kind != "message":
        return []
    message = entry.get("message") or {}
    role = message.get("role")
    if role == "custom":
        parts = _content_parts(message.get("content"))
        return (
            [messages_.Message(role="user", parts=parts, replay=replay)]
            if parts
            else []
        )
    if role == "bashExecution" and not message.get("excludeFromContext"):
        # pi's own rendering of it for the model.
        text = (
            f"Ran `{message.get('command', '')}`\n```\n"
            + f"{message.get('output', '')}\n```"
        )
        return [
            messages_.Message(
                role="user",
                parts=[messages_.TextPart(text=text)],
                replay=replay,
            )
        ]
    if role == "user":
        parts = _content_parts(message.get("content"))
        return (
            [messages_.Message(role="user", parts=parts, replay=replay)]
            if parts
            else []
        )
    if role == "assistant":
        parts = _assistant_parts(message.get("content"))
        return (
            [
                messages_.Message(
                    role="assistant",
                    parts=parts,
                    replay=replay,
                    provider_metadata=_written_by(message),
                )
            ]
            if parts
            else []
        )
    if role == "toolResult":
        result = _tool_result(
            str(message.get("toolCallId") or ""),
            str(message.get("toolName") or ""),
            message,
            is_error=bool(message.get("isError")),
        )
        return [messages_.Message(role="tool", parts=[result], replay=replay)]
    return []


def _written_by(message: dict[str, Any]) -> dict[str, Any] | None:
    """Which provider and model wrote an assistant message, as pi says.

    Kept on the message (`provider_metadata["pi"]`) so a conversation
    handed to another pi continues on the model it was on.
    """
    written = {
        k: message[k]
        for k in ("api", "provider", "model", "thinkingLevel")
        if isinstance(message.get(k), str)
    }
    if not written or written.get("provider") == "ai-python":
        return None
    return {KIND: written}


def _usage(raw: dict[str, Any]) -> usage_.Usage:
    """Return pi's per-message usage.

    pi's `input` excludes cached input; ours includes it.
    """

    def count(name: str) -> int:
        value = raw.get(name)
        return value if isinstance(value, int) else 0

    reasoning = raw.get("reasoning")
    return usage_.Usage(
        input_tokens=count("input") + count("cacheRead") + count("cacheWrite"),
        output_tokens=count("output"),
        cache_read_tokens=count("cacheRead"),
        cache_write_tokens=count("cacheWrite"),
        reasoning_tokens=reasoning if isinstance(reasoning, int) else None,
    )


def mcp_config(servers: dict[str, Any]) -> dict[str, Any]:
    """MCP servers, in the shape claude takes, as pi registers them.

    `{"name": {"command": ..., "args": [...], "env": {...}}}` for a local
    server, `{"name": {"type": "http", "url": ...}}` for a remote one. pi
    validates the config, so only the keys it knows are passed.
    """
    keys = ("command", "args", "env", "cwd", "url", "headers")
    return {
        name: {k: server[k] for k in keys if server.get(k) is not None}
        for name, server in servers.items()
    }


AUTH_HINT = (
    "pi authenticates where it RUNS: its own login (`/login`, auth.json) or a "
    "provider key in the environment. A sandbox is another machine that has "
    "never seen either, so give it a way to reach a model: "
    "VercelSandbox(gateway=vercel_ai_gateway()) injects a gateway key at "
    "egress, or VercelSandbox(env={'ANTHROPIC_API_KEY': ...}) puts a key of "
    "its own in the VM."
)
BROKERED_HINT = (
    "This workspace injects the credential into requests to {host} at egress, "
    "so pi never held it and no login in the VM is involved. A 401 here means "
    "the gateway rejected the injected key — check AI_GATEWAY_API_KEY on the "
    "machine that opened the sandbox — or the egress policy is not rewriting "
    "requests to that host."
)


def _turn_failure(detail: str, *, hint: str = AUTH_HINT) -> Exception:
    """Rename the failure for what the caller has to fix."""
    detail = detail[:500]
    if errors._looks_unauthenticated(detail):
        return workspace_errors.NotAuthenticatedError(KIND, detail, hint)
    return errors.TurnFailedError(KIND, detail)


_STORE = r"""
import json, os, re, sys

op, where, sid = sys.argv[1], sys.argv[2], sys.argv[3]
cwd = os.path.realpath(where)
# pi's own rule for a session id. A session file can come from anyone — a
# project's settings choose the directory, before pi asks for trust — and
# its id becomes a lock path and a CLI argument: one that breaks the rule
# is not a session.
ID = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9._-]{0,198}[A-Za-z0-9])?\Z")


def settings(path):
    try:
        with open(path) as f:
            value = json.load(f).get("sessionDir")
        return value if isinstance(value, str) and value else None
    except Exception:
        return None


# Where pi keeps this directory's sessions, decided the way pi decides it:
# the environment, then project settings, then global ones; else one
# directory per working directory under the agent dir.
agent = os.path.expanduser(
    os.environ.get("PI_CODING_AGENT_DIR") or "~/.pi/agent"
)
custom = (
    os.environ.get("PI_CODING_AGENT_SESSION_DIR")
    or settings(os.path.join(cwd, ".pi", "settings.json"))
    or settings(os.path.join(agent, "settings.json"))
)
if custom:
    directory = os.path.join(cwd, os.path.expanduser(custom))
else:
    safe = cwd.lstrip("/\\")
    for c in "/\\:":
        safe = safe.replace(c, "-")
    directory = os.path.join(agent, "sessions", "--" + safe + "--")


def iso_ms(text):
    import datetime

    try:
        when = datetime.datetime.strptime(text[:19], "%Y-%m-%dT%H:%M:%S")
    except Exception:
        return None
    when = when.replace(tzinfo=datetime.timezone.utc)
    return int(when.timestamp() * 1000)


def describe(path, full):
    with open(path) as f:
        header = json.loads(f.readline() or "{}")
        if header.get("type") != "session" or not ID.match(
            str(header.get("id") or "")
        ):
            return None
        if header.get("cwd") not in (cwd, where):
            return None
        info = {
            "id": header["id"],
            "path": path,
            "cwd": header.get("cwd"),
            "created": iso_ms(header.get("timestamp") or ""),
            "updated": int(os.path.getmtime(path) * 1000),
        }
        if not full:
            return info
        name = first = None
        for line in f:
            try:
                entry = json.loads(line)
            except ValueError:
                continue
            if entry.get("type") == "session_info":
                name = entry.get("name") or None
            elif first is None and entry.get("type") == "message":
                message = entry.get("message") or {}
                if message.get("role") == "user":
                    content = message.get("content")
                    if isinstance(content, list):
                        content = " ".join(
                            b.get("text", "") for b in content
                            if isinstance(b, dict) and b.get("type") == "text"
                        )
                    first = (content or "").strip()[:200] or None
        info["title"] = name or first
        return info


def sessions(full):
    try:
        names = sorted(os.listdir(directory))
    except OSError:
        return []
    out = []
    for name in names:
        if not name.endswith(".jsonl"):
            continue
        try:
            info = describe(os.path.join(directory, name), full)
        except Exception:
            continue
        if info is not None:
            out.append(info)
    return out


if op == "dir":
    out = {"dir": directory, "cwd": cwd}
elif op == "list":
    out = sorted(sessions(True), key=lambda s: -s["updated"])
else:
    out = next((s for s in sessions(False) if s["id"] == sid), None)
print(json.dumps(out))
"""
