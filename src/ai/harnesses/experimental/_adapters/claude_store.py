"""The claude CLI's session store, read through the workspace it lives in.

Sessions belong to the harness, and the harness keeps them on disk WHERE IT
RUNS — `$CLAUDE_CONFIG_DIR/projects/<key>/<session>.jsonl`, or `~/.claude`
when unset. On your own machine claude-agent-sdk reads that directly. In a
microVM the CLI's home is the VM's, and reading this machine's `~/.claude`
for a `/vercel/sandbox` project finds nothing — measured: `sessions()`
returned `[]` for a conversation that had just happened.

claude-agent-sdk designed a seam for exactly this: a duck-typed
`SessionStore` whose `load`/`list_sessions`/`append` it drives from
`list_sessions_from_store`, `get_session_messages_from_store` and
`fork_session_via_store` — the SDK keeps doing every bit of parsing, UUID
remapping and `parentUuid` chaining. This class only moves bytes across
the workspace boundary. Transcript lines are pass-through blobs.
"""

from __future__ import annotations

import json
import posixpath
from typing import TYPE_CHECKING, cast

import claude_agent_sdk

if TYPE_CHECKING:
    from claude_agent_sdk import types as sdk_types

    from ....workspaces.experimental import _base


async def config_dir(workspace: _base.Workspace) -> str:
    """Resolve the CLI's config dir the way the CLI does, in the workspace."""
    override = await workspace.exec(["printenv", "CLAUDE_CONFIG_DIR"])
    found = override.stdout.strip() if override.exit_code == 0 else ""
    return found or f"{await workspace.home()}/.claude"


class WorkspaceSessionStore(claude_agent_sdk.SessionStore):
    def __init__(self, workspace: _base.Workspace, config_dir: str) -> None:
        self._ws = workspace
        self._projects = posixpath.join(config_dir, "projects")

    @classmethod
    async def discover(
        cls, workspace: _base.Workspace, directory: str | None = None
    ) -> WorkspaceSessionStore:
        """Open the store in the CLI's config dir: `directory`, or resolved."""
        return cls(workspace, directory or await config_dir(workspace))

    def _path(self, key: sdk_types.SessionKey) -> str:
        project_dir = posixpath.join(self._projects, key["project_key"])
        subpath = key.get("subpath")
        if subpath:
            # Subagent transcripts nest under the main session's directory,
            # mirroring the on-disk layout the SDK describes.
            return posixpath.join(
                project_dir, key["session_id"], f"{subpath}.jsonl"
            )
        return posixpath.join(project_dir, f"{key['session_id']}.jsonl")

    async def load(
        self, key: sdk_types.SessionKey
    ) -> list[sdk_types.SessionStoreEntry] | None:
        # One round trip: read_text raises FileNotFoundError for a missing
        # file, so checking first only doubled the cost of every load.
        try:
            text = await self._ws.read_text(self._path(key))
        except FileNotFoundError:
            return None
        return [
            cast("sdk_types.SessionStoreEntry", json.loads(line))
            for line in text.splitlines()
            if line.strip()
        ]

    async def list_sessions(
        self, project_key: str
    ) -> list[sdk_types.SessionStoreListEntry]:
        project_dir = posixpath.join(self._projects, project_key)
        # GNU find in the VM: name and mtime in one round trip. The SDK's
        # contract is milliseconds; find gives fractional seconds. A project
        # with no sessions yet has no directory: find exits non-zero with
        # nothing on stdout, which is exactly "no entries".
        result = await self._ws.exec(
            [
                "find",
                project_dir,
                "-maxdepth",
                "1",
                "-name",
                "*.jsonl",
                "-printf",
                "%f %T@\n",
            ]
        )
        entries: list[sdk_types.SessionStoreListEntry] = []
        for line in result.stdout.splitlines() if result.exit_code == 0 else []:
            name, _, mtime = line.strip().partition(" ")
            if not name.endswith(".jsonl"):
                continue
            entries.append(
                {
                    "session_id": name[: -len(".jsonl")],
                    "mtime": int(float(mtime or 0) * 1000),
                }
            )
        return entries

    async def append(
        self,
        key: sdk_types.SessionKey,
        entries: list[sdk_types.SessionStoreEntry],
    ) -> None:
        """Append entries; the SDK's fork uses this to write the new branch.

        Create-or-extend, like the reference store: a fork is a fresh file, but
        appending to an existing one must not clobber it.
        """
        path = self._path(key)
        await self._ws.exec(["mkdir", "-p", posixpath.dirname(path)])
        try:
            existing = await self._ws.read_text(
                path
            )  # one round trip, not a check and a read
        except FileNotFoundError:
            existing = ""
        lines = "".join(
            json.dumps(e, separators=(",", ":")) + "\n" for e in entries
        )
        await self._ws.write_text(path, existing + lines)
        # Date it now. A file written through the sandbox's file API is
        # stamped 1970, and a running claude CLI deletes transcripts it takes
        # for ancient (measured: a fork's file vanished before its CLI could
        # resume it, and which write won the race decided whether fork worked).
        await self._ws.exec(["touch", path])
