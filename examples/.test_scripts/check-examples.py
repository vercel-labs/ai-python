#!/usr/bin/env python3
"""Typecheck all example directories with mypy.

Usage (from repo root):
    uv run examples/.test_scripts/check-examples.py
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

import mypy.version

REPO = Path(__file__).resolve().parent.parent.parent
MYPY_VERSION = f"mypy=={mypy.version.__version__}"
_EXAMPLES_DIR = REPO / "examples"

_SAMPLE_FILES = sorted(
    str(p.relative_to(_EXAMPLES_DIR))
    for p in _EXAMPLES_DIR.rglob("*.py")
    if ".test_scripts" not in p.parts
    and p.relative_to(_EXAMPLES_DIR).parts[:1] != ("apps",)
)

# Each entry: (display name, directory to check, targets)
EXAMPLES: list[tuple[str, Path, list[str]]] = [
    ("samples", _EXAMPLES_DIR, _SAMPLE_FILES),
    (
        "web_agent/backend",
        _EXAMPLES_DIR / "apps" / "web_agent" / "backend",
        ["."],
    ),
    (
        "durable_agent_temporal",
        _EXAMPLES_DIR / "apps" / "durable_agent_temporal" / "backend",
        ["."],
    ),
    (
        "durable_agent_workflows/backend",
        _EXAMPLES_DIR / "apps" / "durable_agent_workflows" / "backend",
        ["."],
    ),
    (
        "slack_agent",
        _EXAMPLES_DIR / "apps" / "slack_agent",
        ["."],
    ),
]


def run_checker(
    checker: list[str],
    name: str,
    directory: Path,
    targets: list[str],
    *,
    use_current_ai: bool,
) -> bool:
    header = f"{'=' * 20} {name} ({checker[0]}) {'=' * 20}"
    print(header)

    with_args: list[str] = []
    for dep in [MYPY_VERSION]:
        with_args.extend(["--with", dep])

    cmd = ["uv", "run", "--frozen", "--group", "dev"]
    if use_current_ai:
        cmd.extend(["--with-editable", str(REPO)])
    cmd.extend(
        [
            *with_args,
            *checker,
            *targets,
        ]
    )

    env = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}
    sys.stdout.flush()
    result = subprocess.run(cmd, cwd=directory, env=env)
    print()
    sys.stdout.flush()
    return result.returncode == 0


CHECKERS = [
    ["mypy", "--config-file", str(REPO / "pyproject.toml")],
    ["ty", "check"],
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--no-current-ai",
        action="store_true",
        help="check against the ai version specified by each example",
    )
    args = parser.parse_args()

    results: list[tuple[str, bool]] = []
    for name, directory, targets in EXAMPLES:
        for checker in CHECKERS:
            ok = run_checker(
                checker,
                name,
                directory,
                targets,
                use_current_ai=not args.no_current_ai,
            )
            results.append((f"{name} ({checker[0]})", ok))

    print("=" * 60)
    print("Summary:")
    any_failed = False
    for name, ok in results:
        status = "PASS" if ok else "FAIL"
        print(f"  {status}  {name}")
        if not ok:
            any_failed = True
    print()

    if any_failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
