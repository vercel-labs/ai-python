#!/usr/bin/env python3
"""Typecheck all example directories with mypy.

Usage (from repo root):
    uv run examples/check-examples.py
"""

import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MYPY_VERSION = "mypy>=1.11"

# Each entry: (display name, directory to check, extra --with deps)
EXAMPLES: list[tuple[str, Path, list[str]]] = [
    ("samples", REPO / "examples" / "samples", []),
    (
        "fastapi-vite/backend",
        REPO / "examples" / "fastapi-vite" / "backend",
        ["fastapi"],
    ),
    (
        "multiagent-textual",
        REPO / "examples" / "multiagent-textual",
        ["fastapi", "textual", "websockets"],
    ),
    (
        "tau-agent",
        REPO / "examples" / "tau-agent",
        ["textual"],
    ),
    (
        "temporal-direct",
        REPO / "examples" / "temporal-direct",
        ["temporalio"],
    ),
]


def run_mypy(name: str, directory: Path, extra_deps: list[str]) -> bool:
    header = f"{'=' * 20} {name} {'=' * 20}"
    print(header)

    with_args: list[str] = []
    for dep in [MYPY_VERSION, "pydantic", *extra_deps]:
        with_args.extend(["--with", dep])

    cmd = [
        "uv",
        "run",
        "--frozen",
        "--project",
        str(REPO),
        "--group",
        "dev",
        "--with-editable",
        str(REPO),
        *with_args,
        "mypy",
        "--config-file",
        str(REPO / "pyproject.toml"),
        ".",
    ]

    env = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}
    sys.stdout.flush()
    result = subprocess.run(cmd, cwd=directory, env=env)
    print()
    sys.stdout.flush()
    return result.returncode == 0


def main() -> None:
    results: list[tuple[str, bool]] = []
    for name, directory, extra_deps in EXAMPLES:
        ok = run_mypy(name, directory, extra_deps)
        results.append((name, ok))

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
