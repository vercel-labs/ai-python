"""Drive the Claude CLI when it lives on another machine.

claude-agent-sdk spawns the CLI itself and only accepts a `cwd`, so it
cannot be pointed at a microVM. Its `Transport` seam can: this
implementation builds the SAME argv the bundled transport would — by
reusing its own builder, rather than reimplementing dozens of option
mappings — and then runs it through the workspace instead of locally.

The `Transport` ABC is marked unstable upstream. If it changes shape, this
is the one file that has to follow.
"""

from __future__ import annotations

import contextlib
import json
from dataclasses import replace
from typing import TYPE_CHECKING, Any

# Private claude_agent_sdk API: the Transport seam is how the CLI runs inside
# a sandbox, and the argv builder and can_use_tool rewrite are reused rather
# than reimplemented. The claude-code extra's upper bound guards these.
from claude_agent_sdk._internal.transport import Transport  # noqa: PLC2701
from claude_agent_sdk._internal.transport.subprocess_cli import (
    SubprocessCLITransport,  # noqa: PLC2701
)
from claude_agent_sdk.types import _configure_can_use_tool  # noqa: PLC2701

from .. import errors

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from claude_agent_sdk import ClaudeAgentOptions

    from ....workspaces.experimental import _base


class RemoteCliTransport(Transport):
    """A claude-agent-sdk transport whose process lives in a workspace."""

    def __init__(
        self,
        workspace: _base.Workspace,
        options: ClaudeAgentOptions,
        *,
        executable: str = "claude",
    ) -> None:
        self._workspace = workspace
        # `can_use_tool` is not a CLI flag — the SDK rewrites the options to
        # add `--permission-prompt-tool stdio` so the CLI routes permission
        # requests back over the control protocol. That rewrite happens in
        # the client and is SKIPPED when a transport is supplied ready-made,
        # so the argv came out without it and the agent asked the human
        # instead of our hook. Apply it here.
        configured = (
            _configure_can_use_tool(options)
            if options.can_use_tool
            else options
        )
        # Borrow the bundled transport purely as an argv builder. It does no
        # I/O in its constructor, so nothing is spawned here.
        builder = SubprocessCLITransport(
            prompt="", options=replace(configured, cli_path=executable)
        )
        self._argv: list[str] = builder._build_command()
        # The bundled transport puts `options.env` on the process IT spawns.
        # We spawn instead, so it has to be carried across by hand — or a
        # gateway's credentials never reach the far side.
        self._env: dict[str, str] = dict(configured.env or {})
        self._process: _base.Process | None = None
        self._release_only = False
        self._closed = False

    async def connect(self) -> None:
        # The workspace applies its own environment too; this goes on top.
        self._process = await self._workspace.spawn(
            self._argv, env=self._env or None
        )

    async def write(self, data: str) -> None:
        if self._process is None:
            raise errors.AgentCrashedError(
                "claude-code", "transport is not connected"
            )
        await self._process.write(data)

    async def read_messages(self) -> AsyncIterator[dict[str, Any]]:
        if self._process is None:
            raise errors.AgentCrashedError(
                "claude-code", "transport is not connected"
            )
        while True:
            line = await self._process.readline()
            if not line:
                return
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                # The CLI prints the occasional non-JSON line; it is noise,
                # not a protocol failure.
                continue

    def release_on_close(self) -> None:
        """Make the next close() a detach.

        The SDK client's own disconnect() is the only thing that cleanly cancels
        its reader task, and it calls close() on the way out — so detaching goes
        through it, with the terminate swapped for a release.
        """
        self._release_only = True

    async def close(self) -> None:
        self._closed = True
        if self._release_only:
            await self.detach()
            return
        if self._process is not None:
            await self._process.terminate()
            self._process = None

    async def detach(self) -> None:
        """Release the client's hold WITHOUT terminating the CLI.

        Drop the input conduit and stop reading, leaving the remote process
        running (its FIFO stdin is held open by the launch, so it sees no
        EOF). close() additionally asks the process to terminate; in a kept
        sandbox that is a request the FIFO-held process may outlive, so
        survival is the WORKSPACE's `keep` decision, not this call's.
        """
        self._closed = True
        if self._process is not None:
            with contextlib.suppress(Exception):
                await self._process.detach()
            self._process = None

    def is_ready(self) -> bool:
        return self._process is not None and not self._closed

    async def end_input(self) -> None:
        # The remote process keeps its stdin open for the whole session: the
        # FIFO behind it is what lets a conduit detach without the CLI
        # seeing EOF, so there is nothing to close here.
        return None
