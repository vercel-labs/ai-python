"""The whole funnel against a real `claude`, on the vulnerable fixture.

scan → process → revalidate → enrich → export, then the acceptance criteria that
need an agent to check (spec §13): resumability/idempotence, cost-sum equality,
append-only, finding identity. Costs a few cents; minutes, not seconds.
"""

from __future__ import annotations

import shutil
import subprocess
from typing import TYPE_CHECKING

import pytest
from deepysec.agents import gateway_from_env, open_workers
from deepysec.enrich import enrich
from deepysec.export import export, status
from deepysec.matchers import registry
from deepysec.models import AgentVerdict
from deepysec.process import process
from deepysec.revalidate import revalidate
from deepysec.scan import scan
from deepysec.store import Store

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = [
    pytest.mark.live,
    pytest.mark.slow,
    pytest.mark.timeout(1200),
    pytest.mark.skipif(
        shutil.which("claude") is None, reason="claude is not installed"
    ),
]

VERDICTS = set(AgentVerdict.__args__)  # type: ignore[attr-defined]


async def test_pipeline_end_to_end(fixture_app: Path, tmp_path: Path) -> None:
    root = tmp_path / "app"
    shutil.copytree(fixture_app, root)
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.com",
            "add",
            "-A",
        ],
        cwd=root,
        check=True,
    )
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.com",
            "commit",
            "-qm",
            "plant",
        ],
        cwd=root,
        check=True,
    )
    store = Store(root / ".deepsec" / "data", "app")
    quiet = print

    scan(root, store, registry(), log=quiet)

    # Your own login when claude has one here; the gateway when a key is around.
    gateway = gateway_from_env()
    # Precise candidates first, so --limit 2 is exec.ts (rce) and users.ts
    # (sql-injection).
    async with open_workers(
        agent="claude", model=None, root=root, sandboxes=0, gateway=gateway
    ) as workers:
        run = await process(
            store,
            workers,
            batch_size=2,
            concurrency=1,
            limit=2,
            timeout=600,
            log=quiet,
        )
        assert run.phase == "done"
        assert run.stats.files_processed == 2
        analyzed = {
            r.file_path: r for r in store.records() if r.status == "analyzed"
        }
        assert set(analyzed) == {"src/api/exec.ts", "src/api/users.ts"}
        for record in analyzed.values():
            assert record.findings, (
                f"{record.file_path}: an unambiguous planted bug was not "
                "reported"
            )
            assert len(record.analysis_history) == 1
            assert record.locked_by_run_id is None
            assert all(
                f.finding_id and f.produced_by_run_id == run.run_id
                for f in record.findings
            )

        # Cost attribution: per-file shares sum to the run total.
        shares = [
            e.cost_usd or 0.0
            for r in analyzed.values()
            for e in r.analysis_history
        ]
        assert run.stats.total_cost_usd > 0
        assert abs(sum(shares) - run.stats.total_cost_usd) < 1e-9
        tokens = sum(
            e.usage.input_tokens
            for r in analyzed.values()
            for e in r.analysis_history
            if e.usage
        )
        assert tokens == run.stats.total_input_tokens > 0

        # Idempotent restart: nothing eligible is left within the limit, nothing
        # is redone.
        again = await process(
            store,
            workers,
            batch_size=2,
            concurrency=1,
            limit=2,
            timeout=600,
            log=quiet,
        )
        assert (
            again.stats.files_processed == 2
        )  # the next two pending files, not the same two
        assert {
            r.file_path for r in store.records() if r.status == "analyzed"
        } > set(analyzed)
        for path, before in analyzed.items():
            after = store.load(path)
            assert after is not None
            assert [e.run_id for e in after.analysis_history] == [run.run_id]
            assert [f.finding_id for f in after.findings] == [
                f.finding_id for f in before.findings
            ]

        # Revalidate everything found so far; every verdict is one the agent may
        # give.
        rerun = await revalidate(
            store,
            workers,
            min_severity="LOW",
            batch_size=4,
            concurrency=1,
            timeout=600,
            log=quiet,
        )
        assert rerun.phase == "done"
        findings = [f for r in store.records() for f in r.findings]
        assert findings
        revalidated = [f for f in findings if f.revalidation is not None]
        assert revalidated, "no finding received a verdict"
        assert all(
            f.revalidation is not None and f.revalidation.verdict in VERDICTS
            for f in revalidated
        )
        for f in revalidated:
            v = f.revalidation
            assert v is not None
            if v.verdict == "duplicate":
                primary = next(
                    g for g in findings if g.finding_id == v.duplicate_of
                )
                assert (
                    primary.revalidation is None
                    or primary.revalidation.verdict != "duplicate"
                )
        assert rerun.stats.findings_revalidated == len(revalidated)

    enriched = enrich(root, store, log=quiet)
    assert enriched >= 1
    for r in store.records():
        if r.findings:
            assert r.git_info is not None
            assert r.git_info.recent_committers[0].name == "Fixture"

    json_path, md_path = export(store)
    assert json_path.is_file() and md_path.is_file()
    report = md_path.read_text()
    assert findings[0].title in report
    assert "Recent committers" in report
    snapshot = status(store)
    assert "analyzed" in snapshot and "findings:" in snapshot
    print(snapshot)
