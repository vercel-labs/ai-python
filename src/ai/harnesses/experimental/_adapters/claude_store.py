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
from typing import TYPE_CHECKING, Any, cast

import claude_agent_sdk

if TYPE_CHECKING:
    from claude_agent_sdk import types as sdk_types

    from ....workspaces.experimental import _base


async def config_dir(workspace: _base.Workspace) -> str:
    """Resolve the CLI's config dir the way the CLI does, in the workspace."""
    override = await workspace.exec(["printenv", "CLAUDE_CONFIG_DIR"])
    found = override.stdout.strip() if override.exit_code == 0 else ""
    return found or f"{await workspace.home()}/.claude"


_SCAN = r"""
import json, os, sys

directory, with_text, out = sys.argv[1], sys.argv[2] == "1", {}
try:
    names = [n for n in os.listdir(directory) if n.endswith(".jsonl")]
except OSError:
    names = []
for name in names:
    path = os.path.join(directory, name)
    try:
        info = {"mtime": os.stat(path).st_mtime_ns // 1_000_000}
        if with_text:
            with open(path, encoding="utf-8", errors="replace") as f:
                info["text"] = f.read()
    except OSError:
        continue
    out[name[: -len(".jsonl")]] = info
print(json.dumps(out))
"""


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
        # Every part becomes a path in the workspace: a part that is itself
        # a path (absolute, `..`) would read or write outside the store. A
        # project key or session id is one name; a subagent's subpath nests.
        for part, nests in (
            (key["project_key"], False),
            (key["session_id"], False),
            (key.get("subpath"), True),
        ):
            if part is None:
                continue
            names = part.split("/") if nests else [part]
            if (
                not part
                or part.startswith("/")
                or (not nests and "/" in part)
                or any(n in ("", ".", "..") for n in names)
            ):
                raise ValueError(f"not a store key: {part!r}")
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

    async def _scan(
        self, project_key: str, *, with_text: bool
    ) -> dict[str, dict[str, Any]]:
        """Every transcript in a project: mtime, and the text when asked.

        One command in the workspace, a stdlib Python helper, so listing N
        conversations is one round trip, not one read per transcript. Both
        listings share it, so their mtimes agree to the millisecond — the
        SDK compares them to decide whether a summary is fresh.
        """
        result = await self._ws.exec(
            [
                "python3",
                "-c",
                _SCAN,
                posixpath.join(self._projects, project_key),
                "1" if with_text else "0",
            ],
            timeout=120,
        )
        if result.exit_code != 0:
            return {}
        found: dict[str, dict[str, Any]] = json.loads(result.stdout or "{}")
        return found

    async def list_sessions(
        self, project_key: str
    ) -> list[sdk_types.SessionStoreListEntry]:
        # A project with no sessions yet has no directory: no entries.
        scanned = await self._scan(project_key, with_text=False)
        return [
            {"session_id": sid, "mtime": int(info["mtime"])}
            for sid, info in scanned.items()
        ]

    async def list_session_summaries(
        self, project_key: str
    ) -> list[sdk_types.SessionSummaryEntry]:
        """Summaries for every session, folded here from one batch read.

        The CLI writes its transcripts itself, inside the workspace, so there
        is no `append()` to maintain summaries in. Without this the SDK
        loaded each transcript in turn: N round trips to list N sessions.
        """
        summaries: list[sdk_types.SessionSummaryEntry] = []
        for sid, info in (
            await self._scan(project_key, with_text=True)
        ).items():
            entries = [
                cast("sdk_types.SessionStoreEntry", json.loads(line))
                for line in str(info.get("text") or "").splitlines()
                if line.strip()
            ]
            summary = claude_agent_sdk.fold_session_summary(
                None, {"project_key": project_key, "session_id": sid}, entries
            )
            summary["mtime"] = int(info["mtime"])
            summaries.append(summary)
        return summaries

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
        # Written aside, dated, then renamed into place in one step. A file
        # written through the sandbox's file API is stamped 1970, and a
        # running claude CLI deletes transcripts it takes for ancient.
        # Writing in place and touching after left a window: measured under
        # load, the transcript was deleted before the touch (which then made
        # an empty file) and the fork's `claude --resume` said "No
        # conversation found". The CLI only sweeps `*.jsonl`, so the `.part`
        # file is safe, and the renamed file is dated now.
        part = f"{path}.part"
        await self._ws.write_text(part, existing + lines)
        moved = await self._ws.exec(
            ["sh", "-c", 'touch "$1" && mv -f "$1" "$2"', "sh", part, path]
        )
        if moved.exit_code != 0:
            raise OSError(
                f"could not put the transcript in place: {moved.stderr.strip()}"
            )
