"""The ai.harnesses seam — everything the spec tagged [REPLACE] in §4.4, §8, §9.

deepsec's three vendor adapters, its JSON repair loops, its OS sandbox and
its microVM pipeline all land in this one file, as calls into the SDK:

- `claude_code()` / `codex()` are the adapters. `--agent` picks one.
- `approve=static_analysis_only` is the tool permission table (§8.3).
- `session.run(prompt, output_type=..., retries=2)` is the output contract
  and its repair loop (§4.4). The SDK appends the schema and re-asks.
- `result.usage` is the per-call cost and token usage (§8.2 requirement 1).
- `VercelSandbox(gateway=gw)` + `copy()` is distributed execution (§9): the
  harness runs in the VM, this program and the store do not — and the
  gateway key does not either (§9.2): the sandbox firewall injects it at
  egress and the VM holds a placeholder.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import re
import time
from contextlib import AsyncExitStack, asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

from pydantic import BaseModel, ConfigDict, ValidationError

from ai.harnesses.experimental import (
    Allow,
    ApprovalContext,
    Decision,
    Deny,
    Harness,
    Result,
    Session,
    claude_code,
    codex,
)
from ai.harnesses.experimental.errors import HarnessError, TurnFailedError
from ai.types.messages import TextPart
from ai.workspaces.experimental import (
    Gateway,
    Local,
    VercelSandbox,
    Workspace,
    copy,
    vercel_ai_gateway,
)
from ai.workspaces.experimental.errors import NotAuthenticatedError
from deepysec.models import TokenUsage
from deepysec.scan import SCAN_IGNORE

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

KEY_VAR = "AI_GATEWAY_API_KEY"

AgentKind = Literal["claude", "codex"]
Effort = Literal["low", "medium", "high", "max"]

#: deepsec's `--thinking-level` vocabulary, mapped to the SDK's one scale.
#: Its default is xhigh: deepsec optimizes for finding hard bugs, not for cost.
THINKING_LEVELS: dict[str, Effort] = {
    "minimal": "low",
    "low": "low",
    "medium": "medium",
    "high": "high",
    "xhigh": "max",
}


# -- tool permissions (spec §8.3) ---------------------------------------------

#: Claude's read-only tools. Codex has no separate read tools: it reads by
#: running commands, which the shell rules below judge.
READ_ONLY_TOOLS = {"Read", "Glob", "Grep", "LS"}
#: The tools that carry a shell command, by each harness's native name.
SHELL_TOOLS = {"Bash", "commandExecution"}
NETWORK = re.compile(
    r"\b(curl|wget|nc|ncat|ssh|scp|ftp|telnet|npm\s+install|pip\s+install)\b"
)
MUTATION = re.compile(
    r"(^|[;&|]\s*)(rm|mv|cp|chmod|chown|touch|tee|sed\s+-i|git\s+(push|commit|checkout|reset|clean))\b"
)
REDIRECT = re.compile(r"(?<![0-9&])>\s*(?!/dev/null)\S")
SHELL_WRAPPER = re.compile(r"^\s*(?:/\S*/)?(?:ba|z|da)?sh\s+-l?c\s+(.*)$", re.S)


def shell_text(command: str) -> str:
    """The command as the shell will see it.

    Codex wraps everything in `/bin/bash -lc '...'`; judging the wrapper instead
    of its argument would let `rm -rf` through because it follows a quote, not a
    separator.
    """
    match = SHELL_WRAPPER.match(command)
    if match is None:
        return command
    inner = match.group(1).strip()
    if len(inner) >= 2 and inner[0] == inner[-1] and inner[0] in "'\"":
        inner = inner[1:-1]
    return inner


async def static_analysis_only(ctx: ApprovalContext) -> Decision:
    """The agent is a security researcher with a read-only shell.

    Claude never routes Read/Glob/Grep through the hook — its own policy
    treats them as safe — so what arrives from it is Bash and anything that
    writes. Codex routes its `commandExecution` here, under that name, with
    the command in the same `command` field. The SDK hands over each
    harness's own tool call untranslated, so the hook has to know both
    dialects; a denial's reason is delivered to the agent as guidance.
    """
    call = ctx.call
    if call.name in SHELL_TOOLS:
        command = shell_text(str(call.input.get("command", "")))
        if NETWORK.search(command):
            return Deny(
                "static analysis only: no network access from the shell"
            )
        if MUTATION.search(command) or REDIRECT.search(command):
            return Deny("static analysis only: the tree is read-only")
        return Allow()
    if call.name in READ_ONLY_TOOLS:
        return Allow()
    return Deny(
        f"{call.name} is not available: static analysis only, with read-only "
        "tools"
    )


# -- reaching a model ----------------------------------------------------------


def gateway_from_env() -> Gateway | None:
    """A Vercel AI Gateway, if a key is around: in the environment, or in a
    `.env.local` above the current directory — the file `vercel env pull`
    writes. The key only ever travels as this object: never in a handle,
    never in the store."""
    key = os.environ.get(KEY_VAR)
    if not key:
        for directory in (Path.cwd(), *Path.cwd().parents):
            candidate = directory / ".env.local"
            if candidate.is_file():
                for line in candidate.read_text().splitlines():
                    if line.startswith(KEY_VAR + "="):
                        key = line.split("=", 1)[1].strip().strip("\"'")
                        break
            if key:
                break
    return vercel_ai_gateway(api_key=key) if key else None


# -- workers -----------------------------------------------------------------


class Worker(BaseModel):
    """One open harness, and where the project is inside its workspace."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    label: str
    harness: Harness
    project_dir: str
    """`.` on a Local workspace; `project` in a sandbox holding the tree."""

    @property
    def model(self) -> str:
        return str(
            self.harness.options.get("model") or f"{self.harness.kind}-default"
        )

    def relative(self, path: str) -> str:
        """A path the agent reported, as the store keys it."""
        p = path.replace("\\", "/").strip()
        root = self.harness.workspace.path.rstrip("/")
        if root and p.startswith(root + "/"):
            p = p[len(root) + 1 :]
        p = p.removeprefix("./")
        if self.project_dir not in ("", ".") and p.startswith(
            self.project_dir + "/"
        ):
            p = p[len(self.project_dir) + 1 :]
        return p


def build_harness(
    agent: AgentKind,
    workspace: Workspace,
    *,
    model: str | None,
    gateway: Gateway | None,
    effort: Effort | None = None,
) -> Harness:
    if agent == "claude":
        return claude_code(
            workspace=workspace,
            approve=static_analysis_only,
            model=model,
            gateway=gateway,
            effort=effort,
        )
    # codex has a read-only sandbox of its own on a local machine; inside a
    # microVM the VM is the boundary and the SDK's default applies.
    return codex(
        workspace=workspace,
        approve=static_analysis_only,
        model=model,
        gateway=gateway,
        effort=effort,
        sandbox="read-only" if workspace.kind == "local" else None,
    )


@asynccontextmanager
async def open_workers(
    *,
    agent: AgentKind,
    model: str | None,
    root: Path,
    sandboxes: int,
    gateway: Gateway | None,
    effort: Effort | None = None,
) -> AsyncIterator[list[Worker]]:
    """One worker on this machine, or N in Vercel Sandboxes with the project
    copied into each. The caller never learns which — same `Worker` either way.
    """
    if sandboxes <= 0:
        harness = build_harness(
            agent, Local(root), model=model, gateway=gateway, effort=effort
        )
        async with harness:
            yield [Worker(label="local", harness=harness, project_dir=".")]
        return

    if gateway is None:
        raise NotAuthenticatedError(
            agent,
            "a sandbox has never seen your login",
            "set AI_GATEWAY_API_KEY (or put it in .env.local) so the harness "
            "can reach a model",
        )

    async with AsyncExitStack() as stack:

        async def boot(index: int) -> Worker:
            # Vercel credentials come from `.env.local` above where the COMMAND
            # runs, the same place `gateway_from_env` looks — not from the tree.
            # The gateway is the WORKSPACE's: the VM's egress is locked to the
            # gateway host, the firewall writes the key into requests on the
            # way out, and the harness below is handed a placeholder. deepsec's
            # credential brokering (§9.2), as one constructor argument.
            workspace = await stack.enter_async_context(
                VercelSandbox(gateway=gateway)
            )
            # The tree, minus what .gitignore excludes and minus our own store:
            # the agent gets the code, never its predecessors' findings.
            sent = await copy(root, workspace / "project", ignore=SCAN_IGNORE)
            harness = build_harness(
                agent, workspace, model=model, gateway=None, effort=effort
            )
            await stack.enter_async_context(harness)
            print(
                f"[sandbox-{index + 1}] up: {workspace.name}, {sent} files in, "
                f"{harness.kind} {harness.version}"
            )
            return Worker(
                label=f"sandbox-{index + 1}",
                harness=harness,
                project_dir="project",
            )

        yield list(await asyncio.gather(*(boot(i) for i in range(sandboxes))))


# -- one structured call -----------------------------------------------------


class ContractViolatedError(Exception):
    """The agent never produced a reply matching the schema, after the SDK's
    retries. `raw` is its last reply, for `debug/`."""

    def __init__(self, raw: str, detail: str) -> None:
        super().__init__(detail)
        self.raw = raw


class RunAbortedError(Exception):
    """Every batch is hitting the same wall — an empty balance, a missing login.

    Letting the others run wastes minutes (spec §4.5).
    """


BILLING = (
    "credit",
    "billing",
    "quota",
    "insufficient",
    "exceeded",
    "spending",
    "balance",
)


def is_fatal(exc: HarnessError) -> bool:
    if isinstance(exc, NotAuthenticatedError):
        return True
    return isinstance(exc, TurnFailedError) and any(
        m in str(exc).lower() for m in BILLING
    )


class Answer[T](BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    output: T
    result: Result
    session_id: str
    duration_ms: int

    @property
    def cost_usd(self) -> float | None:
        value = (self.result.usage.raw or {}).get("cost_usd")
        return float(value) if value is not None else None

    @property
    def usage(self) -> TokenUsage:
        u = self.result.usage
        return TokenUsage(
            input_tokens=u.input_tokens,
            output_tokens=u.output_tokens,
            cache_read_input_tokens=u.cache_read_tokens or 0,
            cache_creation_input_tokens=u.cache_write_tokens or 0,
        )


class Fleet:
    """The sessions in flight, so an abort can stop them all — real
    cancellation through the SDK, not a flag the next poll would notice."""

    def __init__(self) -> None:
        self.active: set[Session] = set()
        self.aborted: str | None = None

    async def abort(self, reason: str) -> None:
        if self.aborted is not None:
            return
        self.aborted = reason
        for session in list(self.active):
            with contextlib.suppress(Exception):
                await session.stop()


async def ask[T](
    worker: Worker,
    prompt: str,
    output_type: type[T],
    *,
    timeout: float | None,
    fleet: Fleet,
) -> Answer[T]:
    """One prompt, one fresh conversation, one validated value.

    `retries=2` is deepsec's MAX_ATTEMPTS=3: the SDK re-asks with the
    validation error twice before giving up. `timeout` is the turn budget:
    a deadline settles the turn as cancelled and keeps what was produced.
    """
    session = worker.harness.session()
    fleet.active.add(session)
    started = time.monotonic()
    try:
        result = await session.run(
            prompt, output_type=output_type, retries=2, timeout=timeout
        )
    except ValidationError as exc:
        raise ContractViolatedError(_last_reply(session), str(exc)) from exc
    except HarnessError as exc:
        if is_fatal(exc):
            raise RunAbortedError(str(exc)) from exc
        raise
    finally:
        fleet.active.discard(session)
        with contextlib.suppress(Exception):
            await session.close()
    if fleet.aborted is not None:
        raise RunAbortedError(fleet.aborted)
    return Answer[output_type](  # type: ignore[valid-type]  # ty: ignore[invalid-type-form]
        output=cast("T", result.output),
        result=result,
        session_id=session.session_id,
        duration_ms=int((time.monotonic() - started) * 1000),
    )


def _last_reply(session: Session) -> str:
    for message in reversed(session.messages):
        if message.role == "assistant":
            return "".join(
                p.text for p in message.parts if isinstance(p, TextPart)
            )
    return ""
