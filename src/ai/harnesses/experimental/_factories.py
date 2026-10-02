"""Constructing a harness.

The factory IS the configured agent: there is no separate spec object to keep
in sync.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, Literal

from ...providers import _optional
from ...workspaces.experimental import _base, _gateway, _local, _sandbox
from ...workspaces.experimental import errors as workspace_errors
from . import _handle, _harness, _session, errors
from ._adapters import base
from ._adapters import codex as codex_

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

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
            executable=executable, model=model, gateway=gateway, effort=effort
        ),
        workspace=workspace,
        approve=approve,
        approval_timeout=approval_timeout,
        options={"model": model, "executable": executable, "effort": effort},
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
) -> _harness.Harness:
    """Codex over `codex app-server`.

    `gateway` supplies the model provider codex would otherwise read from
    `~/.codex/config.toml`, and the key that provider names — see
    `ai.workspaces.experimental.vercel_ai_gateway`. An explicit `config=` wins
    over it, key by key.

    `effort` is codex's `model_reasoning_effort`, on the same "low" to
    "max" scale claude takes; codex spells the top "xhigh", and that is
    translated here so the caller never learns the difference.

    `sandbox` defaults to "read-only" on your own machine and
    "danger-full-access" on a provider-owned workspace, where the microVM
    is already the isolation boundary and codex's own sandbox cannot start
    nested inside it. On your own machine, pass a value to decide for
    yourself; on a provider-owned workspace any other value raises
    UnsupportedError when the harness opens, because every command would
    fail.
    """
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
        ),
        workspace=workspace,
        approve=approve,
        approval_timeout=approval_timeout,
        options={
            "model": model,
            "executable": executable,
            "sandbox": sandbox,
            "effort": effort,
        },
    )


BUILDERS = {"claude-code": claude_code, "codex": codex}


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
    options = {
        k: v
        for k, v in handle.options.items()
        if k in ("model", "executable", "sandbox", "effort")
    }
    kwargs: dict[str, Any] = {k: v for k, v in options.items() if v is not None}
    if gateway is not None:
        kwargs["gateway"] = gateway
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
