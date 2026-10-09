"""Stage 3 — revalidate (spec §5). A second agent attacks the FINDINGS.

Input: findings with no `revalidation` (all of them with `force`), at or
above `min_severity`. Output: a verdict per `findingId`, reconciled onto the
findings under the duplicate invariant. Never creates a finding.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from ai.harnesses.experimental.errors import HarnessError
from deepysec.agents import (
    ContractViolatedError,
    Fleet,
    RunAbortedError,
    Worker,
    ask,
)
from deepysec.models import (
    SEVERITY_ORDER,
    AnalysisEntry,
    FileRecord,
    Finding,
    Revalidation,
    RunMeta,
    Verdict,
    now,
    split_evenly,
)
from deepysec.prompt import revalidation_prompt

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from deepysec.store import Store

Item = tuple[FileRecord, Finding]


def select(
    store: Store, *, min_severity: str, force: bool, limit: int | None
) -> list[Item]:
    cutoff = SEVERITY_ORDER[min_severity]
    items: list[Item] = []
    for record in store.records():
        for finding in record.findings:
            if SEVERITY_ORDER[finding.severity] > cutoff:
                continue
            existing = finding.revalidation
            # A manual accepted-risk is never re-opened, force or not.
            if existing is not None and (
                existing.verdict == "accepted-risk" or not force
            ):
                continue
            items.append((record, finding))
    return items[:limit] if limit else items


def by_file(items: list[Item], batch_size: int) -> list[list[Item]]:
    groups: dict[str, list[Item]] = {}
    for record, finding in items:
        groups.setdefault(record.file_path, []).append((record, finding))
    files = list(groups.values())
    return [
        [item for group in files[i : i + batch_size] for item in group]
        for i in range(0, len(files), max(1, batch_size))
    ]


def reconcile(
    items: list[Item], verdicts: list[Verdict], *, run_id: str, model: str
) -> tuple[int, list[str]]:
    """Map verdicts onto findings; enforce the duplicate invariant.

    Every equivalence class has exactly one non-duplicate primary. A
    `duplicate` whose `duplicateOf` is missing, is itself, points outside
    the batch, or points at another duplicate is rejected — the finding
    stays unrevalidated and the next run asks again.
    """
    findings = {f.finding_id: f for _, f in items if f.finding_id}
    proposed = {v.finding_id: v for v in verdicts if v.finding_id in findings}
    rejected: list[str] = []

    def is_duplicate(fid: str) -> bool:
        v = proposed.get(fid)
        if v is not None:
            return v.verdict == "duplicate"
        existing = findings[fid].revalidation
        return existing is not None and existing.verdict == "duplicate"

    applied = 0
    for fid, verdict in proposed.items():
        finding = findings[fid]
        if (
            finding.revalidation is not None
            and finding.revalidation.verdict == "accepted-risk"
        ):
            continue
        if verdict.verdict == "duplicate":
            target = verdict.duplicate_of
            if (
                not target
                or target == fid
                or target not in findings
                or is_duplicate(target)
            ):
                rejected.append(fid)
                continue
        finding.revalidation = Revalidation(
            verdict=verdict.verdict,
            reasoning=verdict.reasoning,
            adjusted_severity=verdict.adjusted_severity,
            duplicate_of=verdict.duplicate_of
            if verdict.verdict == "duplicate"
            else None,
            revalidated_at=now(),
            run_id=run_id,
            model=model,
        )
        applied += 1
    return applied, rejected


async def revalidate(
    store: Store,
    workers: list[Worker],
    *,
    min_severity: str = "HIGH",
    force: bool = False,
    batch_size: int = 3,
    concurrency: int = 2,
    limit: int | None = None,
    timeout: float | None = 900.0,
    git_root: Path | None = None,
    project_info: str | None = None,
    log: Callable[[str], None] = print,
) -> RunMeta:
    first = workers[0]
    run = RunMeta(
        project_id=store.project_id,
        root_path=str(first.harness.workspace.path),
        type="revalidate",
        agent_type=first.harness.kind,
        model=first.model,
    )
    store.save_run(run)
    batches = by_file(
        select(store, min_severity=min_severity, force=force, limit=limit),
        batch_size,
    )
    total = sum(map(len, batches))
    log(
        f"[revalidate] run {run.run_id}: {total} findings ({min_severity}+) in "
        f"{len(batches)} batches"
    )
    fleet = Fleet()
    gate = asyncio.Semaphore(concurrency)

    async def one(index: int, items: list[Item]) -> None:
        async with gate:
            if fleet.aborted:
                return
            worker = workers[index % len(workers)]
            prompt = revalidation_prompt(
                items,
                project_dir=worker.project_dir,
                git_root=git_root,
                project_info=project_info,
            )
            try:
                answer = await ask(
                    worker, prompt, list[Verdict], timeout=timeout, fleet=fleet
                )
            except RunAbortedError as exc:
                await fleet.abort(str(exc))
                return
            except ContractViolatedError as exc:
                stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
                path = store.write_debug(
                    f"parse-error-revalidate-{stamp}-{index + 1}.txt", exc.raw
                )
                log(
                    f"[revalidate] batch {index + 1}: no valid reply after 3 "
                    f"attempts; raw output in {path}"
                )
                return
            except HarnessError as exc:
                log(f"[revalidate] batch {index + 1} failed: {exc}")
                return
            # Raw verdicts are kept verbatim so a mismatch can be diagnosed
            # without repeating model work (spec §5).
            store.write_debug(
                f"revalidate-{run.run_id}-{index + 1}.json",
                "["
                + ",".join(v.model_dump_json() for v in answer.output)
                + "]",
            )
            applied, rejected = reconcile(
                items, answer.output, run_id=run.run_id, model=worker.model
            )
            touched = {
                r.file_path: r
                for r, f in items
                if f.revalidation and f.revalidation.run_id == run.run_id
            }
            n = len(touched)
            cost = (
                answer.cost_usd / n
                if (answer.cost_usd is not None and n)
                else None
            )
            durations = split_evenly(answer.duration_ms, n)
            for i, record in enumerate(touched.values()):
                record.analysis_history.append(
                    AnalysisEntry(
                        run_id=run.run_id,
                        investigated_at=now(),
                        duration_ms=durations[i],
                        agent_type=worker.harness.kind,
                        model=worker.model,
                        agent_session_id=answer.session_id,
                        finding_count=sum(
                            1
                            for f in record.findings
                            if f.revalidation
                            and f.revalidation.run_id == run.run_id
                        ),
                        phase="revalidate",
                        cost_usd=cost,
                        usage=answer.usage if i == 0 else None,
                    )
                )
                store.save(record)
                if cost is not None:
                    run.stats.total_cost_usd += cost
            for _, finding in items:
                v = finding.revalidation
                if v is None or v.run_id != run.run_id:
                    continue
                run.stats.findings_revalidated += 1
                key = {
                    "true-positive": "true_positives",
                    "false-positive": "false_positives",
                    "fixed": "fixed",
                    "uncertain": "uncertain",
                    "duplicate": "duplicates",
                }[v.verdict]
                setattr(run.stats, key, getattr(run.stats, key) + 1)
            store.save_run(run)
            log(
                f"[revalidate] batch {index + 1}: {applied} verdicts"
                + (
                    f", {len(rejected)} rejected (duplicate invariant)"
                    if rejected
                    else ""
                )
            )

    try:
        await asyncio.gather(*(one(i, b) for i, b in enumerate(batches)))
    finally:
        run.phase = "error" if fleet.aborted else "done"
        run.completed_at = now()
        store.save_run(run)
    s = run.stats
    log(
        f"[revalidate] done: {s.findings_revalidated} verdicts — "
        f"{s.true_positives} true, "
        + f"{s.false_positives} false, {s.fixed} fixed, {s.uncertain} "
        "uncertain, " + f"{s.duplicates} duplicate; ${s.total_cost_usd:.4f}"
    )
    return run
