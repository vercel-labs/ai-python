#!/usr/bin/env python3
"""Typecheck all example directories with mypy.

Usage (from repo root):
    uv run examples/.test_scripts/check-examples.py
"""

import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
MYPY_VERSION = "mypy>=1.11"
_EXAMPLES_DIR = REPO / "examples"

_SAMPLE_FILES = sorted(
    str(p.relative_to(_EXAMPLES_DIR))
    for p in _EXAMPLES_DIR.rglob("*.py")
    if ".test_scripts" not in p.parts
    and p.relative_to(_EXAMPLES_DIR).parts[:1] != ("apps",)
)

# Each entry: (display name, directory to check, extra --with deps, targets)
EXAMPLES: list[tuple[str, Path, list[str], list[str]]] = [
    ("samples", _EXAMPLES_DIR, [], _SAMPLE_FILES),
    (
        "fastapi-vite/backend",
        _EXAMPLES_DIR / "apps" / "fastapi-vite" / "backend",
        ["fastapi"],
        ["."],
    ),
    (
        "multiagent-textual",
        _EXAMPLES_DIR / "apps" / "multiagent-textual",
        ["fastapi", "textual", "websockets"],
        ["."],
    ),
    (
        "temporal-direct",
        _EXAMPLES_DIR / "apps" / "temporal-direct",
        ["temporalio"],
        ["."],
    ),
]


def run_mypy(
    name: str, directory: Path, extra_deps: list[str], targets: list[str]
) -> bool:
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
        *targets,
    ]

    env = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}
    sys.stdout.flush()
    result = subprocess.run(cmd, cwd=directory, env=env)
    print()
    sys.stdout.flush()
    return result.returncode == 0


def main() -> None:
    results: list[tuple[str, bool]] = []
    for name, directory, extra_deps, targets in EXAMPLES:
        ok = run_mypy(name, directory, extra_deps, targets)
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
