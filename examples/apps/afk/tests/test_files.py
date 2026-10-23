"""The rules `afk pull --files` decides by: pure, so no sandbox needed."""

from __future__ import annotations

from typing import TYPE_CHECKING

from afk.files import Change, manifest, plan

if TYPE_CHECKING:
    from pathlib import Path


def _plan(
    sandbox: dict[str, str],
    local: dict[str, str],
    baseline: dict[str, str] | None,
) -> list[tuple[str, str, bool]]:
    return [
        (c.path, c.action, c.conflict) for c in plan(sandbox, local, baseline)
    ]


def test_only_what_the_sandbox_changed_comes_home() -> None:
    baseline = {"a": "1", "b": "1", "c": "1", "gone": "1"}
    sandbox = {
        "a": "2",
        "b": "1",
        "c": "1",
        "new": "1",
    }  # a edited, gone deleted, new added
    local = {"a": "1", "b": "9", "c": "1", "gone": "1"}  # b edited here only
    assert _plan(sandbox, local, baseline) == [
        ("a", "update", False),
        ("gone", "delete", False),
        ("new", "add", False),
    ], "b changed only here, so it stays yours and is not listed"


def test_a_file_changed_on_both_sides_is_a_conflict() -> None:
    baseline = {"a": "1", "d": "1"}
    sandbox = {"a": "2", "e": "2"}  # a edited, d deleted, e added
    local = {"a": "3", "d": "3", "e": "3"}  # all three touched here too
    assert _plan(sandbox, local, baseline) == [
        ("a", "update", True),
        ("d", "delete", True),
        ("e", "update", True),
    ]


def test_files_already_the_same_are_not_listed() -> None:
    assert (
        _plan({"a": "2"}, {"a": "2"}, {"a": "1"}) == []
    ), "both sides made the same edit"
    assert _plan({"a": "1"}, {"a": "1"}, {"a": "1"}) == []


def test_without_a_baseline_all_may_conflict_and_nothing_is_deleted() -> None:
    sandbox = {"a": "2", "new": "1", "same": "1"}
    local = {"a": "1", "same": "1", "only-here": "1"}
    assert _plan(sandbox, local, None) == [
        ("a", "update", True),
        ("new", "add", True),
    ], "only-here may never have been pushed: never proposed for deletion"


def test_the_manifest_follows_the_push_rules(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("print(1)\n")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "x.js").write_text("x\n")
    (tmp_path / ".gitignore").write_text("secret.txt\n")
    (tmp_path / "secret.txt").write_text("s\n")
    assert sorted(manifest(tmp_path)) == [
        ".gitignore",
        "src/app.py",
    ], "ignored files never travel, so never come back"


def test_change_is_a_model() -> None:
    assert Change(path="a", action="add", conflict=False).model_dump() == {
        "path": "a",
        "action": "add",
        "conflict": False,
    }
