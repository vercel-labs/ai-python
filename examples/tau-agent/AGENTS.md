# AGENTS.md — tau-agent

## Overview

`tau` is a single-process coding-agent TUI built on the `ai` library and
Textual.  It gives the model seven filesystem/shell tools (read, write,
edit, bash, grep, find, ls) with an approval gate for mutating
operations.

## Project layout

```
tau/
  app.py        — Textual app, chat loop, approval flow
  tools.py      — tool definitions (mirrors pi's seven built-ins)
  session.py    — JSONL session persistence and resume
pyproject.toml  — project metadata, dependencies, ruff/mypy config
```

## TODO list

There may be a task list in `.tau/TODO` — check it for current
priorities and open items.

## Running

```bash
uv sync          # install deps
uv run tau       # launch the TUI
```

## Linting & type-checking

```bash
uv run ruff check .        # lint
uv run ruff format --check # format check
uv run mypy tau            # type-check
```

## Conventions

- Python ≥ 3.12.
- Line length: 80 (`ruff` and project style).
- Lint rule set: E, F, I, UP, B, SIM (see `pyproject.toml`).
- No workspace jail — the approval gate is the safety mechanism.
- Approval-gated tools (`write`, `edit`, `bash`) require operator
  confirmation; reads are auto-approved.
- Sessions persist as JSONL under `.tau/sessions/`.
- Commit messages for this subdirectory should be prefixed with `[tau]`.
