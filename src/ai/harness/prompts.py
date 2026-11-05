"""The harness prompts, and the system message built from them."""

from __future__ import annotations

import datetime
import pathlib
import platform
from collections.abc import Sequence

from ..types import builders
from ..types import messages as messages_

PROMPT = """\
You are a coding agent that works in the user's local environment.
Use your tools to inspect files and run commands; do not guess.
- read: read files and images.
- bash: run shell commands (search, edit, write, git, curl, ...).
- subagent: hand a self-contained job to a subagent with a fresh context.
Be brief. Show file paths clearly."""

SUMMARY_PROMPT = """\
The messages above are a conversation to summarize. Create a structured \
context checkpoint summary that another LLM will use to continue the work.

Use this EXACT format:

## Goal
[What is the user trying to accomplish?]

## Constraints & Preferences
- [Constraints, preferences, or requirements from the user, or "(none)"]

## Progress
### Done
- [x] [Completed tasks/changes]

### In Progress
- [ ] [Current work]

### Blocked
- [Issues preventing progress, if any]

## Key Decisions
- **[Decision]**: [Brief rationale]

## Next Steps
1. [Ordered list of what should happen next]

## Critical Context
- [Data, examples, or references needed to continue, or "(none)"]

Keep each section concise. Preserve exact file paths, function names, \
and error messages."""

# pi's prefix: the summary goes into history as a `user` message
SUMMARY_PREFIX = (
    "The conversation history before this point was compacted into the "
    "following summary:\n\n<summary>\n"
)

SKILL_DIRS = (".agents/skills", "~/.agents/skills")


def system_message(
    prompt: str = PROMPT, *, skill_dirs: Sequence[str] = SKILL_DIRS
) -> messages_.Message:
    """Build the system message: *prompt*, env facts, AGENTS.md, skills.

    Put it first in the messages you give to ``Agent.run``.
    """
    cwd = pathlib.Path.cwd()
    sections = [
        prompt,
        f"cwd: {cwd}\nos: {platform.system()} {platform.release()}\n"
        f"date: {datetime.date.today().isoformat()}",
    ]

    # AGENTS.md from the filesystem root down to cwd, so the most
    # specific file comes last and wins
    for d in reversed([cwd, *cwd.parents]):
        if (f := d / "AGENTS.md").is_file():
            sections.append(f"Instructions from {f}:\n\n{f.read_text()}")

    # skills: name + description from SKILL.md frontmatter; the model
    # reads the file itself with `read`
    skills = []
    for root in skill_dirs:
        for f in sorted(pathlib.Path(root).expanduser().glob("*/SKILL.md")):
            meta = {"name": f.parent.name, "description": ""}
            head = f.read_text().split("---")
            if len(head) >= 3:  # "", frontmatter, body...
                for line in head[1].splitlines():
                    key, sep, value = line.partition(":")
                    if sep and key.strip() in meta:
                        meta[key.strip()] = value.strip()
            skills.append(
                f"  <skill>\n"
                f"    <name>{meta['name']}</name>\n"
                f"    <description>{meta['description']}</description>\n"
                f"    <location>{f.resolve()}</location>\n"
                f"  </skill>"
            )
    if skills:
        sections.append(
            "Skills give task-specific instructions. When a task matches "
            "a skill, read its SKILL.md first.\n"
            "<available_skills>\n" + "\n".join(skills) + "\n</available_skills>"
        )

    return builders.system_message("\n\n".join(sections))
