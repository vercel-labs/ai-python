"""Every matcher's `examples` fire against its own regex set (spec §3, §13.10).

One discovery test over the registry: a typo in any sub-pattern is a
failure here, next to the rule, rather than a silently quieter scanner.
"""

from __future__ import annotations

import shutil
from typing import TYPE_CHECKING

import pytest
from deepysec.matchers import MATCHERS, Matcher, registry
from deepysec.scan import scan
from deepysec.store import Store

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize("matcher", MATCHERS, ids=lambda m: m.slug)
def test_every_example_fires(matcher: Matcher) -> None:
    assert matcher.examples, "a matcher without examples is untested"
    path = f"src/example.{matcher.extensions[0]}"
    for example in matcher.examples:
        assert matcher.match(
            example, path
        ), f"{matcher.slug}: did not fire on {example!r}"


def test_slugs_are_unique() -> None:
    slugs = [m.slug for m in MATCHERS]
    assert len(slugs) == len(set(slugs))


def test_skip_paths_apply_to_both_gates() -> None:
    ssrf = registry(only=["ssrf"])[0]
    assert not ssrf.applies_to("src/proxy.test.ts")
    assert ssrf.match("fetch(req.body.url);", "src/proxy.test.ts") == []
    assert ssrf.match("fetch(req.body.url);", "src/proxy.ts")


def test_scan_finds_every_planted_issue(
    fixture_app: Path, tmp_path: Path
) -> None:
    store = Store(tmp_path, "vulnerable-app")
    run = scan(fixture_app, store, registry(), log=lambda _: None)
    by_path = {r.file_path: r for r in store.records()}
    expected = {
        "src/api/users.ts": "sql-injection",
        "src/api/render.ts": "xss",
        "src/api/proxy.ts": "ssrf",
        "src/api/exec.ts": "rce",
        "src/api/login.ts": "open-redirect",
        "src/lib/crypto.ts": "insecure-crypto",
        "src/lib/files.ts": "path-traversal",
        "src/config.ts": "secrets-exposure",
        "scripts/report.py": "py-sql-raw",
        # the constant sql.raw: a candidate, not a bug
        "src/lib/safe.ts": "sql-injection",
    }
    for path, slug in expected.items():
        assert path in by_path, f"{path} was not scanned"
        assert slug in {
            c.vuln_slug for c in by_path[path].candidates
        }, f"{path} lacks {slug}"
    assert all(r.status == "pending" for r in by_path.values())
    assert run.stats.files_scanned == len(by_path)
    assert run.phase == "done"


def test_rescan_merges_without_duplicates(
    fixture_app: Path, tmp_path: Path
) -> None:
    store = Store(tmp_path, "vulnerable-app")
    first = scan(fixture_app, store, registry(), log=lambda _: None)
    before = {r.file_path: len(r.candidates) for r in store.records()}
    second = scan(fixture_app, store, registry(), log=lambda _: None)
    after = {r.file_path: len(r.candidates) for r in store.records()}
    assert before == after
    assert second.stats.candidates_found == 0
    assert first.run_id != second.run_id


def test_own_data_dir_is_never_scanned(
    fixture_app: Path, tmp_path: Path
) -> None:
    root = tmp_path / "app"
    shutil.copytree(fixture_app, root)
    store = Store(root / ".deepsec" / "data", "app")
    scan(root, store, registry(), log=lambda _: None)
    (root / ".deepsec" / "data" / "planted.ts").write_text("eval(x);\n")
    scan(root, store, registry(), log=lambda _: None)
    assert not any(r.file_path.startswith(".deepsec") for r in store.records())
