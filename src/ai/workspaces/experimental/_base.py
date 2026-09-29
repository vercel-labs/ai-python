"""Where an agent runs.

A workspace is the only thing in this SDK that knows about paths and
processes. Adapters never spawn anything themselves — they ask the
workspace — which is the entire reason the same adapter drives a CLI on
this machine or inside a microVM.

Two rules make that true, and both are load-bearing:

1. Artifacts move through `read_text`/`write_text`, never `pathlib`. A
   caller reaching for the local filesystem silently breaks the moment the
   workspace is remote.
2. Containment is judged HERE, by pure-path normalization, never by
   resolving against this machine's filesystem — a remote path does not
   exist locally, so `Path.resolve()` would be answering a different
   question.
"""

from __future__ import annotations

import shutil
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, Protocol, Self

from pydantic import BaseModel

from . import _gateway, _gitignore, _pty

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, Sequence

# Directories nobody means to ship into a workspace.
DEFAULT_IGNORE: tuple[str, ...] = (
    ".git",
    ".hg",
    ".svn",
    "node_modules",
    ".venv",
    "venv",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".next",
    "dist",
    "build",
    "*.pyc",
    ".DS_Store",
)


def walk_uploadable(
    source: Path, ignore: Sequence[str], *, gitignore: bool = True
) -> Iterator[tuple[Path, str]]:
    """Yield (local file, workspace-relative posix path) for what should travel.

    Skipped: anything an `ignore` glob matches and, unless `gitignore` is
    off, anything the tree's `.gitignore` files exclude. A skipped directory
    is pruned rather than walked, which is what makes excluding
    `node_modules` cheap instead of merely tidy.
    """
    rules = _gitignore.GitIgnore(source) if gitignore else None

    def excluded(name: str) -> bool:
        return any(fnmatch(name, pattern) for pattern in ignore)

    def walk(
        directory: Path, prefix: str, scopes: tuple[_gitignore.Scope, ...]
    ) -> Iterator[tuple[Path, str]]:
        if rules is not None:
            scopes = rules.descend(scopes, directory, prefix)
        for entry in sorted(directory.iterdir()):
            if excluded(entry.name):
                continue
            relative = f"{prefix}{entry.name}"
            # A symlink is a file to git, whatever it points at.
            is_dir = entry.is_dir() and not entry.is_symlink()
            if rules is not None and rules.ignored(
                scopes, relative, entry.name, is_dir=is_dir
            ):
                continue
            if is_dir:
                yield from walk(entry, f"{relative}/", scopes)
            elif entry.is_file():
                if excluded(relative):
                    continue
                yield entry, relative

    yield from walk(source, "", rules.above if rules is not None else ())


def copy_tree(
    source: Path, dest: Path, ignore: Sequence[str], *, gitignore: bool
) -> int:
    """Copy what `walk_uploadable` yields into a local directory.

    Returns the number of files copied. Blocking: run it in a thread.
    """
    count = 0
    for local, relative in walk_uploadable(source, ignore, gitignore=gitignore):
        target = dest / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(local, target)
        count += 1
    return count


@dataclass(frozen=True)
class WorkspacePath:
    """A path, and the workspace it lives in.

    Built with `workspace / "some/path"`, the same `/` pathlib uses. It
    exists so `copy` can tell which side of a transfer is remote without a
    direction flag or a second method.
    """

    workspace: Workspace
    path: str

    def __truediv__(self, other: str) -> WorkspacePath:
        return WorkspacePath(self.workspace, f"{self.path.rstrip('/')}/{other}")

    def __str__(self) -> str:
        return f"{self.workspace.kind}:{self.path}"


class ExecResult(BaseModel):
    """A finished one-shot command."""

    exit_code: int
    stdout: str = ""
    stderr: str = ""


class WorkspaceCoords(BaseModel):
    """A serializable locator for a workspace.

    Carried inside a Handle so another process can reopen the same place.
    """

    provider: str
    location: str
    options: dict[str, Any] = {}


class Process(Protocol):
    """A live process with duplex stdio.

    This is what makes adapters location-agnostic: a harness CLI's stdin and
    stdout are just these streams, whether the process is a local child or a
    program inside a microVM.
    """

    @property
    def pid(self) -> int | None: ...

    async def write(self, data: str) -> None: ...

    async def readline(self) -> str: ...

    async def read_stderr(self) -> str: ...

    async def terminate(self) -> None: ...

    async def kill(self) -> None: ...

    async def os_pid(self) -> int | None:
        """Return the pid where the process RUNS.

        `pid` for a local child, the in-VM pid for a remote one. What a liveness
        check needs.
        """
        return self.pid

    async def detach(self) -> None:
        """Release this handle WITHOUT ending the process.

        Where the workspace outlives the client (a sandbox), the program
        keeps running for a later reconnect. Where processes die with the
        client (a local child), this is terminate().
        """
        ...

    async def wait(self) -> int | None: ...


def contains(root: str, candidate: str) -> bool:
    """Pure-path containment: no filesystem access, no symlink resolution.

    Relative paths are denied outright. The harness resolves them against
    its own cwd, which this process cannot observe — and a containment
    answer that might be wrong is worse than a refusal.
    """
    if not candidate.startswith("/"):
        return False
    root_path = PurePosixPath(root)
    parts: list[str] = []
    for part in PurePosixPath(candidate).parts:
        if part == "..":
            if not parts or parts == ["/"]:
                return False
            parts.pop()
        elif part not in (".",):
            parts.append(part)
    normalized = PurePosixPath(*parts)
    return normalized == root_path or root_path in normalized.parents


class Workspace(ABC):
    """The contract every implementation keeps identically."""

    kind: str

    owner: str = "user"
    """Who owns this machine — "user" or "provider".

    A user's own machine is never modified without being asked: a missing
    harness raises. A provider-owned box (an ephemeral microVM we created)
    is ours to provision, so a missing harness is installed.
    """

    duplex_spawn: bool = True
    """Whether `spawn` returns a process with a WRITABLE stdin.

    An adapter that needs to write to a harness CLI checks this before
    spawning one. True on both workspaces today: the Vercel Sandbox process
    API has no stdin, so `VercelSandbox.spawn` feeds one through a FIFO.
    """

    env: dict[str, str]
    """What the WORK needs: the project's own configuration.

    Applied to everything run here — your own `exec` and `spawn` calls, the
    harness process, and therefore the commands the agent itself runs. A
    harness overlays what the AGENT needs (its model, how it is told where
    the model is) on top of this.
    """

    gateway: _gateway.Gateway | None = None
    """Where a model is reached from here: a base URL and a credential.

    The workspace's, because reaching a model is a question about the
    machine the work happens on — its network, its environment — not about
    which agent asks. A harness opened here without a `gateway=` of its own
    uses this one, spelled the way its CLI wants; the workspace never learns
    that spelling. A `VercelSandbox` given a gateway also injects the
    credential at egress (see `injects_credentials_for`), so the VM only
    ever holds a placeholder. `Local` has no firewall: there the credential
    goes in the environment, which is your own machine.
    """

    def injects_credentials_for(self, host: str) -> bool:
        """Return whether requests to `host` get the credential at egress.

        Written into them outside anything that runs on this workspace. When
        true, an adapter hands its CLI `gateway.brokered()`: a placeholder the
        firewall overwrites, so the real key never enters the machine. The
        default is no: only a workspace with a firewall of its own can say yes.
        """
        return False

    @property
    @abstractmethod
    def path(self) -> str:
        """The working directory, as the AGENT sees it.

        On a remote workspace this path means nothing on your machine.
        """

    @property
    @abstractmethod
    def coords(self) -> WorkspaceCoords: ...

    async def home(self) -> str:
        """Return the home directory on the workspace's machine.

        Where the harnesses keep their session stores, and where the SDK keeps
        per-conversation state next to them. Asked of the machine, never assumed
        from this one.
        """
        result = await self.exec(["printenv", "HOME"], timeout=30)
        return result.stdout.strip() or "/root"

    @abstractmethod
    async def open(self) -> None: ...

    @abstractmethod
    async def close(self) -> None:
        """Release the workspace and everything it spawned."""

    @abstractmethod
    async def spawn(
        self, argv: Sequence[str], *, env: Mapping[str, str] | None = None
    ) -> Process: ...

    @abstractmethod
    async def exec(
        self,
        argv: Sequence[str],
        *,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> ExecResult: ...

    @abstractmethod
    async def pty(
        self,
        argv: Sequence[str],
        *,
        env: Mapping[str, str] | None = None,
        name: str | None = None,
        size: tuple[int, int] | None = None,
    ) -> _pty.Pty:
        """Run a program on a pseudo-terminal.

        For programs a human talks to, where `spawn` is for programs a protocol
        talks to. A NAME means persistence: the program outlives this connection
        and a later `attach(name)` picks it up; unnamed, it ends with the
        connection.
        """

    @abstractmethod
    async def attach(self, name: str) -> _pty.Pty:
        """Reconnect to a named pty started earlier, from any process."""

    @abstractmethod
    async def ptys(self) -> list[_pty.PtyInfo]:
        """List the named ptys still running here, and who is on each.

        Process plumbing this workspace started, nothing about what runs inside
        them. Never connects: that would take a pty from its client.
        """

    @abstractmethod
    async def read_text(self, path: str) -> str:
        """Read a workspace-relative file.

        Raises FileNotFoundError when it is absent — never returns '' for a
        missing file.
        """

    @abstractmethod
    async def write_text(self, path: str, data: str) -> None:
        """Write a workspace-relative file, creating parent directories."""

    @abstractmethod
    async def read_bytes(self, path: str) -> bytes: ...

    @abstractmethod
    async def exists(self, path: str) -> bool: ...

    @abstractmethod
    async def _copy_in(
        self,
        source: Path,
        dest: str,
        ignore: Sequence[str],
        *,
        gitignore: bool,
    ) -> int:
        """Copy a local directory in. Use `copy()`, not this."""

    @abstractmethod
    async def _copy_out(
        self,
        source: str,
        dest: Path,
        ignore: Sequence[str],
        *,
        gitignore: bool,
    ) -> int:
        """Copy a workspace directory out. Use `copy()`, not this.

        Filtered the same way as going in — as it lands, if the far side cannot.
        """

    @abstractmethod
    async def reachable_url(self, port: int) -> str:
        """Return a URL the AGENT can reach for a port served by the caller.

        Async because exposing a port is an API call on a remote workspace —
        the local case is the degenerate one, not the general one.
        """

    def contains(self, candidate: str) -> bool:
        return contains(self.path, candidate)

    def __truediv__(self, path: str) -> WorkspacePath:
        return WorkspacePath(self, path)

    async def __aenter__(self) -> Self:
        await self.open()
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.close()


async def copy(
    source: str | Path | WorkspacePath,
    dest: str | Path | WorkspacePath,
    *,
    ignore: Sequence[str] | None = None,
    gitignore: bool = True,
) -> int:
    """Copy a directory between your machine and a workspace, either way.

        await copy("./project", sandbox / "project")     # in
        await copy(sandbox / "reviews", "./reviews")     # out
        await copy(one / "out", other / "in")            # between workspaces

    Direction is whichever side is a `WorkspacePath`; there is no flag to
    get backwards. Two plain paths is an error — copying local to local is
    `shutil`'s job.

    `ignore` takes glob patterns; a match is pruned rather than walked,
    which is what makes excluding `node_modules` cheap. DEFAULT_IGNORE
    covers the usual noise. What the tree's own `.gitignore` files exclude
    is skipped too — nested ones, and those of the enclosing repository
    above the directory named — unless `gitignore=False`. Both describe the
    tree, not the direction: they apply whichever way it travels.

    Returns the number of files copied.
    """
    patterns = DEFAULT_IGNORE if ignore is None else tuple(ignore)
    if isinstance(source, WorkspacePath) and isinstance(dest, WorkspacePath):
        # Via here, because no provider offers a workspace-to-workspace move.
        with tempfile.TemporaryDirectory(prefix="harness-copy-") as staging:
            await source.workspace._copy_out(
                source.path, Path(staging), patterns, gitignore=gitignore
            )
            return await dest.workspace._copy_in(
                Path(staging), dest.path, patterns, gitignore=gitignore
            )
    if isinstance(dest, WorkspacePath):
        assert not isinstance(source, WorkspacePath)
        local = Path(source).expanduser()
        if not local.is_dir():
            raise FileNotFoundError(f"{local} is not a directory")
        return await dest.workspace._copy_in(
            local, dest.path, patterns, gitignore=gitignore
        )
    if isinstance(source, WorkspacePath):
        local_dest = Path(dest).expanduser()
        local_dest.mkdir(parents=True, exist_ok=True)
        return await source.workspace._copy_out(
            source.path, local_dest, patterns, gitignore=gitignore
        )
    raise TypeError(
        "copy() moves files between your machine and a workspace: mark "
        + 'one side with `workspace / "path"`. For local-to-local, use '
        + "shutil.copytree."
    )
