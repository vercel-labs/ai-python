"""Constructing a harness.

The factory IS the configured agent: there is no separate spec object to keep
in sync.
"""

from __future__ import annotations

import inspect
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, Literal

from ...providers import _optional
from ...workspaces.experimental import _base, _gateway, _local, _sandbox
from ...workspaces.experimental import errors as workspace_errors
from . import _handle, _harness, _session, errors
from ._adapters import base
from ._adapters import codex as codex_
from ._adapters import pi as pi_

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable, Mapping, Sequence

Effort = Literal["low", "medium", "high", "max"]


def claude_code(
    *,
    workspace: _base.Workspace,
    approve: base.ApprovalHook | None = None,
    approval_timeout: float = _harness.DEFAULT_APPROVAL_TIMEOUT,
    model: str | None = None,
    gateway: _gateway.Gateway | None = None,
    effort: Effort | None = None,
    executable: str = "claude",
    instructions: str | None = None,
    mcp_servers: Mapping[str, Mapping[str, Any]] | None = None,
    tools: Sequence[str] | None = None,
    disallowed_tools: Sequence[str] | None = None,
    max_turns: int | None = None,
) -> _harness.Harness:
    """Claude Code, driving your own installation and login.

    Needs the ``claude-code`` extra, and raises ``ai.errors.InstallationError``
    here, before anything runs, when it is missing.

    `gateway` is how it reaches a model somewhere other than your login — see
    `ai.workspaces.experimental.vercel_ai_gateway`. It travels with the harness
    process wherever the workspace runs it, which is what makes a sandbox work
    without a login of its own.

    `effort` is how hard the model thinks — "low" to "max" — the same scale
    on both harnesses. Unset, the CLI's own default applies.

    `instructions` are added to Claude Code's system prompt. `mcp_servers`
    connects MCP servers: `{"name": {"command": ..., "args": [...]}}` or
    `{"name": {"type": "http", "url": ...}}`. `tools` limits the built-in
    tools the agent has (`["Read", "Grep"]`; `[]` for none);
    `disallowed_tools` takes tools away. `max_turns` caps the agent's steps
    in one turn; a turn that hits it settles with `finish_reason="length"`.
    """
    _optional.import_optional_sdk(
        "claude_agent_sdk",
        extra="claude-code",
        feature="the Claude Code harness",
    )
    # imported here: the adapter needs claude_agent_sdk at import time
    from ._adapters import claude  # noqa: PLC0415

    return _harness.Harness(
        claude.ClaudeAdapter(
            executable=executable,
            model=model,
            gateway=gateway,
            effort=effort,
            instructions=instructions,
            mcp_servers={k: dict(v) for k, v in (mcp_servers or {}).items()},
            tools=list(tools) if tools is not None else None,
            disallowed_tools=list(disallowed_tools or []),
            max_turns=max_turns,
        ),
        workspace=workspace,
        approve=approve,
        approval_timeout=approval_timeout,
        options={
            "model": model,
            "executable": executable,
            "effort": effort,
            "instructions": instructions,
            "tools": list(tools) if tools is not None else None,
            "disallowed_tools": list(disallowed_tools or []) or None,
            "max_turns": max_turns,
        },
    )


def codex(
    *,
    workspace: _base.Workspace,
    approve: base.ApprovalHook | None = None,
    approval_timeout: float = _harness.DEFAULT_APPROVAL_TIMEOUT,
    model: str | None = None,
    sandbox: str | None = None,
    config: dict[str, Any] | None = None,
    gateway: _gateway.Gateway | None = None,
    effort: Effort | None = None,
    executable: str = "codex",
    instructions: str | None = None,
    mcp_servers: Mapping[str, Mapping[str, Any]] | None = None,
    tools: Sequence[str] | None = None,
    disallowed_tools: Sequence[str] | None = None,
    max_turns: int | None = None,
) -> _harness.Harness:
    """Codex over `codex app-server`.

    `gateway` supplies the model provider codex would otherwise read from
    `~/.codex/config.toml`, and the key that provider names — see
    `ai.workspaces.experimental.vercel_ai_gateway`. An explicit `config=` wins
    over it, key by key.

    `effort` is codex's `model_reasoning_effort`, on the same "low" to
    "max" scale claude takes; codex spells the top "xhigh", and that is
    translated here so the caller never learns the difference.

    `sandbox` defaults to "danger-full-access" everywhere: every harness
    runs with full access unless you restrict it (an approval hook, or here
    an explicit mode). On your own machine you may pass "read-only" or
    "workspace-write"; on a provider-owned workspace any other value raises
    UnsupportedError when the harness opens, because codex's own sandbox
    cannot start nested inside the microVM and every command would fail.

    `instructions` are codex's developer instructions for every thread.
    `mcp_servers` takes the same shape as on `claude_code()`.

    `tools`, `disallowed_tools` and `max_turns` are accepted so that every
    `claude_code()` argument is a `codex()` argument — swapping one for the
    other is a one-word edit, and a dict of both factories keeps its type —
    but codex has no equivalent: passing one raises UnsupportedError.
    """
    for name, value in (
        ("tools", tools),
        ("disallowed_tools", disallowed_tools),
        ("max_turns", max_turns),
    ):
        if value is not None:
            raise errors.UnsupportedError(
                codex_.KIND, name, "codex has no such option; it is claude's"
            )
    if effort is not None:
        config = {
            "model_reasoning_effort": "xhigh" if effort == "max" else effort,
            **(config or {}),
        }
    return _harness.Harness(
        codex_.CodexAdapter(
            executable=executable,
            model=model,
            sandbox=sandbox,
            config=config,
            gateway=gateway,
            instructions=instructions,
            mcp_servers={k: dict(v) for k, v in (mcp_servers or {}).items()},
        ),
        workspace=workspace,
        approve=approve,
        approval_timeout=approval_timeout,
        options={
            "model": model,
            "executable": executable,
            "sandbox": sandbox,
            "effort": effort,
            "instructions": instructions,
        },
    )


def pi(
    *,
    workspace: _base.Workspace,
    approve: base.ApprovalHook | None = None,
    approval_timeout: float = _harness.DEFAULT_APPROVAL_TIMEOUT,
    model: str | None = None,
    gateway: _gateway.Gateway | None = None,
    effort: Effort | None = None,
    executable: str = "pi",
    instructions: str | None = None,
    mcp_servers: Mapping[str, Mapping[str, Any]] | None = None,
    tools: Sequence[str] | None = None,
    disallowed_tools: Sequence[str] | None = None,
    max_turns: int | None = None,
) -> _harness.Harness:
    """Pi over `pi --mode rpc`, driving your own installation and login.

    `model` is any model pi knows (`--model`): `"sonnet"`,
    `"anthropic/claude-sonnet-4-5"`, with an optional `:<thinking>` suffix.
    Unset, pi chooses exactly as it does on its own — gateway or not.

    `gateway` sends pi to the Vercel AI Gateway (its `vercel-ai-gateway`
    provider) with that key — see `ai.workspaces.experimental.
    vercel_ai_gateway`. A `model` is then the gateway's:
    `"anthropic/claude-sonnet-4.5"`.

    `effort` is pi's thinking level, on the same "low" to "max" scale the
    other harnesses take; pi clamps it to what the model supports.

    `instructions` are appended to pi's system prompt. `mcp_servers` takes
    the same shape as on `claude_code()`. `tools` limits the tools the
    agent has (`["read", "grep"]`; `[]` for none); `disallowed_tools` takes
    tools away.

    pi has no permission prompt, so the approval hook is an extension this
    SDK loads into pi: it sees EVERY tool call, can rewrite its input, and
    a `Deny(reason)` reaches the agent. Your own pi extensions, skills and
    settings load as they always do.

    `max_turns` is accepted so that every `claude_code()` argument is a
    `pi()` argument, but pi has no such limit: passing one raises
    UnsupportedError.
    """
    if max_turns is not None:
        raise errors.UnsupportedError(
            pi_.KIND, "max_turns", "pi has no limit on an agent's steps"
        )
    return _harness.Harness(
        pi_.PiAdapter(
            executable=executable,
            model=model,
            gateway=gateway,
            effort=effort,
            instructions=instructions,
            mcp_servers={k: dict(v) for k, v in (mcp_servers or {}).items()},
            tools=list(tools) if tools is not None else None,
            disallowed_tools=list(disallowed_tools or []),
        ),
        workspace=workspace,
        approve=approve,
        approval_timeout=approval_timeout,
        options={
            "model": model,
            "executable": executable,
            "effort": effort,
            "instructions": instructions,
            "tools": list(tools) if tools is not None else None,
            "disallowed_tools": list(disallowed_tools or []) or None,
        },
    )


BUILDERS: dict[str, Callable[..., _harness.Harness]] = {
    "claude-code": claude_code,
    "codex": codex,
    "pi": pi,
}
RUNTIME_ONLY = frozenset(
    {"workspace", "approve", "approval_timeout", "gateway", "mcp_servers"}
)


def workspace_from(coords: _base.WorkspaceCoords) -> _base.Workspace:
    """Rebuild the place a conversation happened.

    A sandbox handle names the VM; while that VM is alive the workspace is
    rebuilt by reconnecting to it, and closing it leaves it running. Once the
    VM is gone, open() refuses with the reason rather than pointing the
    conversation at some other machine.
    """
    if coords.provider == "local":
        return _local.Local(coords.location)
    if coords.provider == "sandbox":
        return _sandbox.VercelSandbox(
            name=coords.location,
            workdir=str(coords.options.get("workdir") or "/vercel/sandbox"),
        )
    raise workspace_errors.WorkspaceError(
        f"cannot rebuild a {coords.provider!r} workspace from a handle: "
        + "no provider knows how. Open a new workspace and resume the session "
        "there."
    )


@asynccontextmanager
async def harness_from(
    handle: _handle.Handle,
    *,
    fork: bool = False,
    approve: base.ApprovalHook | None = None,
    approval_timeout: float = _harness.DEFAULT_APPROVAL_TIMEOUT,
    gateway: _gateway.Gateway | None = None,
    mcp_servers: Mapping[str, Mapping[str, Any]] | None = None,
) -> AsyncIterator[_session.Session]:
    """Rebuild the harness a handle names and continue its conversation.

    `workspace_from` for the harness, in any process.

    Yields the conversation: resumed, or with `fork=True` branched into a
    new one that inherits its past (the way to continue a conversation
    another client still holds — `resume` would raise `SessionBusyError`). The
    harness underneath is opened here and closed when the block ends.

    The handle carries the harness kind, its configuration and the
    workspace, so nothing needs a registry or an import. Approval hooks are
    process-local behavior and must be supplied again — and so is the
    gateway, deliberately: a handle is something you put in a database, and
    an API key must never be in it.
    """
    builder = BUILDERS.get(handle.kind)
    if builder is None:
        raise errors.ResumeFailedError(
            handle.session_id, f"no adapter for harness kind {handle.kind!r}"
        )
    # What the handle recorded, for the options this harness takes. MCP
    # servers are never recorded — their env often holds tokens — so like
    # the hook and the gateway they are passed again.
    accepted = inspect.signature(builder).parameters
    kwargs: dict[str, Any] = {
        k: v
        for k, v in handle.options.items()
        if k in accepted and k not in RUNTIME_ONLY and v is not None
    }
    if gateway is not None:
        kwargs["gateway"] = gateway
    if mcp_servers is not None:
        kwargs["mcp_servers"] = mcp_servers
    harness = builder(
        workspace=workspace_from(handle.workspace),
        approve=approve,
        approval_timeout=approval_timeout,
        **kwargs,
    )
    await harness.open()
    try:
        session = await (harness.fork if fork else harness.resume)(
            handle.session_id
        )
        yield session
    finally:
        # The handle's harness is ours to clean up: the caller never saw it.
        await harness.close()
