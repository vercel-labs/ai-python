"""deepsec's CLI surface (spec §11), the MVP subset.

    scan | process | revalidate | enrich | export | status | all

`--sandboxes N` on process/revalidate runs the agents in N Vercel Sandboxes
instead of here. It is a flag, not a subcommand, because with the SDK that
is all it is.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from ai.workspaces.experimental.errors import NotAuthenticatedError
from deepysec import enrich as enrich_stage
from deepysec import export as export_stage
from deepysec import process as process_stage
from deepysec import revalidate as revalidate_stage
from deepysec import scan as scan_stage
from deepysec.agents import (
    THINKING_LEVELS,
    AgentKind,
    gateway_from_env,
    open_workers,
)
from deepysec.matchers import registry
from deepysec.store import Store

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ai.workspaces.experimental import Gateway


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--root", default=".", help="project root (default: .)")
    p.add_argument("--project-id", help="default: the root directory's name")
    p.add_argument("--data-dir", help="default: <root>/.deepsec/data")


def _agent_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument("--agent", choices=["claude", "codex"], default="claude")
    p.add_argument("--model")
    p.add_argument(
        "--thinking-level",
        choices=list(THINKING_LEVELS),
        default="xhigh",
        help=(
            "reasoning effort, deepsec's scale (default: xhigh, as in the "
            "original)"
        ),
    )
    p.add_argument("--batch-size", type=int, default=3)
    p.add_argument("--concurrency", type=int, default=2)
    p.add_argument("--limit", type=int)
    p.add_argument(
        "--max-duration", type=float, default=900.0, help="seconds per batch"
    )
    p.add_argument(
        "--sandboxes",
        type=int,
        default=0,
        help="run the agents in N Vercel Sandboxes",
    )
    p.add_argument(
        "--gateway",
        action="store_true",
        help=(
            "reach the model through the Vercel AI Gateway (AI_GATEWAY_API_KEY)"
            " "
        )
        + "instead of the CLI's own login; implied by --sandboxes",
    )


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="deepysec", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("scan", help="regex candidate discovery")
    _common(s)
    s.add_argument("--only", nargs="*", help="matcher slugs to run")

    s = sub.add_parser("process", help="agent investigation of pending files")
    _common(s)
    _agent_flags(s)

    s = sub.add_parser("revalidate", help="verdicts on existing findings")
    _common(s)
    _agent_flags(s)
    s.add_argument(
        "--min-severity",
        default="HIGH",
        choices=["CRITICAL", "HIGH", "HIGH_BUG", "MEDIUM", "BUG", "LOW"],
    )
    s.add_argument(
        "--force",
        action="store_true",
        help="re-verdict findings that already have one",
    )

    s = sub.add_parser("enrich", help="git committers per finding-bearing file")
    _common(s)
    s.add_argument("--force", action="store_true")

    s = sub.add_parser("export", help="findings.json + report.md")
    _common(s)

    s = sub.add_parser("status", help="snapshot of the store")
    _common(s)

    s = sub.add_parser(
        "all", help="scan → process → revalidate → enrich → export"
    )
    _common(s)
    _agent_flags(s)
    s.add_argument(
        "--min-severity",
        default="HIGH",
        choices=["CRITICAL", "HIGH", "HIGH_BUG", "MEDIUM", "BUG", "LOW"],
    )
    return p


def _store(args: argparse.Namespace) -> tuple[Path, Store]:
    root = Path(args.root).expanduser().resolve()
    project_id = args.project_id or root.name
    data_dir = (
        Path(args.data_dir).expanduser()
        if args.data_dir
        else root / ".deepsec" / "data"
    )
    return root, Store(data_dir, project_id)


def _gateway(args: argparse.Namespace) -> Gateway | None:
    """By default the CLI's own login is used and no key is needed."""
    if not (args.gateway or args.sandboxes > 0):
        return None
    return gateway_from_env()


def _info(root: Path) -> str | None:
    info = root / ".deepsec" / "INFO.md"
    return info.read_text() if info.is_file() else None


async def _agent_stage(
    args: argparse.Namespace, root: Path, store: Store, *, stage: str
) -> bool:
    """False when a run aborted; the pipeline, and `all`, stop there."""
    agent: AgentKind = args.agent
    async with open_workers(
        agent=agent,
        model=args.model,
        root=root,
        sandboxes=args.sandboxes,
        gateway=_gateway(args),
        effort=THINKING_LEVELS[args.thinking_level],
    ) as workers:
        if stage in ("process", "all"):
            run = await process_stage.process(
                store,
                workers,
                batch_size=args.batch_size,
                concurrency=args.concurrency,
                limit=args.limit,
                timeout=args.max_duration,
                project_info=_info(root),
            )
            if run.phase == "error":
                return False
        if stage in ("revalidate", "all"):
            run = await revalidate_stage.revalidate(
                store,
                workers,
                min_severity=args.min_severity,
                force=getattr(args, "force", False),
                batch_size=args.batch_size,
                concurrency=args.concurrency,
                limit=args.limit,
                timeout=args.max_duration,
                git_root=root,
                project_info=_info(root),
            )
            if run.phase == "error":
                return False
    return True


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    root, store = _store(args)
    command: str = args.command

    if command in ("scan", "all"):
        only = getattr(args, "only", None)
        scan_stage.scan(root, store, registry(only=only or None))
    if command in ("process", "revalidate", "all"):
        try:
            ok = asyncio.run(_agent_stage(args, root, store, stage=command))
        except NotAuthenticatedError as exc:
            # The SDK already says what to set and where; a traceback adds
            # nothing.
            print(f"[{command}] {exc}", file=sys.stderr)
            return 2
        if not ok:
            return 2
    if command in ("enrich", "all"):
        enrich_stage.enrich(root, store, force=getattr(args, "force", False))
    if command in ("export", "all"):
        json_path, md_path = export_stage.export(store)
        print(f"[export] {json_path}\n[export] {md_path}")
    if command == "status":
        print(export_stage.status(store))
    return 0
