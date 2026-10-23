"""Stage 2 — process (spec §4). Batches of pending files go to a coding agent.

Work selection and locking are the store's (§4.1). Prompting, the output
contract, cost and cancellation are the SDK's, through `agents.ask`. What
is left here is the loop: claim, ask, stamp, release.
"""

from __future__ import annotations

import asyncio
import os
import socket
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from ai.harnesses.experimental.errors import HarnessError
from deepysec.agents import (
    Answer,
    ContractViolatedError,
    Fleet,
    RunAbortedError,
    Worker,
    ask,
)
from deepysec.models import (
    AnalysisEntry,
    FileRecord,
    FileReport,
    Finding,
    RunMeta,
    TokenUsage,
    now,
    split_evenly,
)
from deepysec.prompt import investigation_prompt
from deepysec.store import Store, append_analysis

if TYPE_CHECKING:
    from collections.abc import Callable


def select(store: Store, *, limit: int | None) -> list[FileRecord]:
    """Eligible files: precise candidates first (§3 noise), then by path."""
    eligible = [r for r in store.records() if store.is_claimable(r)]
    eligible.sort(key=lambda r: (r.noise_score, r.file_path))
    return eligible[:limit] if limit else eligible


def chunk(items: list[FileRecord], size: int) -> list[list[FileRecord]]:
    return [items[i : i + size] for i in range(0, len(items), max(1, size))]


async def process(
    store: Store,
    workers: list[Worker],
    *,
    batch_size: int = 3,
    concurrency: int = 2,
    limit: int | None = None,
    timeout: float | None = 900.0,
    project_info: str | None = None,
    log: Callable[[str], None] = print,
) -> RunMeta:
    first = workers[0]
    run = RunMeta(
        project_id=store.project_id,
        root_path=str(first.harness.workspace.path),
        type="process",
        pid=os.getpid(),
        hostname=socket.gethostname(),
        agent_type=first.harness.kind,
        model=first.model,
    )
    store.save_run(run)
    batches = chunk(select(store, limit=limit), batch_size)
    log(
        f"[process] run {run.run_id}: {sum(map(len, batches))} files in "
        f"{len(batches)} batches "
        + f"on {len(workers)} worker(s), {concurrency} in flight"
    )
    fleet = Fleet()
    gate = asyncio.Semaphore(concurrency)

    async def one(index: int, batch: list[FileRecord]) -> None:
        async with gate:
            if fleet.aborted:
                return
            worker = workers[index % len(workers)]
            claimed = [
                c
                for r in batch
                if (c := store.claim(r.file_path, run)) is not None
            ]
            if not claimed:
                return
            names = ", ".join(r.file_path for r in claimed)
            log(
                f"[process] batch {index + 1}/{len(batches)} on "
                f"{worker.label}: {names}"
            )
            prompt = investigation_prompt(
                claimed,
                project_dir=worker.project_dir,
                project_info=project_info,
            )
            try:
                answer = await ask(
                    worker,
                    prompt,
                    list[FileReport],
                    timeout=timeout,
                    fleet=fleet,
                )
            except RunAbortedError as exc:
                for record in claimed:
                    store.release(record, "pending")
                await fleet.abort(str(exc))
                return
            except ContractViolatedError as exc:
                stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
                path = store.write_debug(
                    f"parse-error-process-{stamp}-{index + 1}.txt", exc.raw
                )
                for record in claimed:
                    store.release(record, "error")
                log(
                    f"[process] batch {index + 1}: no valid reply after 3 "
                    f"attempts; raw output in {path}"
                )
                return
            except HarnessError as exc:
                for record in claimed:
                    store.release(record, "pending")
                log(
                    f"[process] batch {index + 1} failed, files released: {exc}"
                )
                return
            stamped = apply_reports(store, run, worker, claimed, answer)
            store.save_run(run)
            log(
                f"[process] batch {index + 1}: {stamped} files analyzed, "
                + f"{run.stats.findings_count} findings so far, "
                f"${run.stats.total_cost_usd:.4f}"
            )

    try:
        await asyncio.gather(*(one(i, b) for i, b in enumerate(batches)))
    finally:
        # Whatever happened — Ctrl-C included — nothing stays locked by a run
        # that is no longer running. Re-running the same command resumes.
        freed = store.release_all(run.run_id)
        run.phase = "error" if fleet.aborted else "done"
        run.completed_at = now()
        store.save_run(run)
        if freed:
            log(f"[process] released {freed} lock(s) held by this run")
    if fleet.aborted:
        log(
            f"[process] ABORTED: {fleet.aborted}\n"
            + "  In-flight batches were stopped and their files stay pending. "
            + "Fix the cause (top up, log in) and re-run the same command to "
            "resume."
        )
    else:
        log(
            f"[process] done: {run.stats.files_processed} files, "
            f"{run.stats.findings_count} new "
            + f"findings, ${run.stats.total_cost_usd:.4f}, run {run.run_id}"
        )
    return run


def apply_reports(
    store: Store,
    run: RunMeta,
    worker: Worker,
    claimed: list[FileRecord],
    answer: Answer[list[FileReport]],
) -> int:
    """Stamp one batch's answer onto its files (spec §2.7 cost attribution).

    Cost, tokens and duration are divided over the files that produced a
    valid report, so the per-file entries sum to the run total. A file the
    agent left out goes back to pending — the next run re-requests only
    those (adaptive splitting, in its simplest form).
    """
    reports = {worker.relative(r.file_path): r for r in answer.output}
    valid = [r for r in claimed if r.file_path in reports]
    if not valid:
        for record in claimed:
            store.release(record, "pending")
        return 0
    n = len(valid)
    cost_share = answer.cost_usd / n if answer.cost_usd is not None else None
    usage = answer.usage
    shares = {
        "input_tokens": split_evenly(usage.input_tokens, n),
        "output_tokens": split_evenly(usage.output_tokens, n),
        "cache_read_input_tokens": split_evenly(
            usage.cache_read_input_tokens, n
        ),
        "cache_creation_input_tokens": split_evenly(
            usage.cache_creation_input_tokens, n
        ),
    }
    durations = split_evenly(answer.duration_ms, n)
    for record in claimed:
        if record.file_path not in reports:
            store.release(record, "pending")
    for i, record in enumerate(valid):
        report = reports[record.file_path]
        findings = [
            Finding(
                severity=f.severity,
                vuln_slug=f.vuln_slug,
                title=f.title,
                description=f.description,
                line_numbers=f.line_numbers,
                recommendation=f.recommendation,
                confidence=f.confidence,
            )
            for f in report.findings
        ]
        entry = AnalysisEntry(
            run_id=run.run_id,
            investigated_at=now(),
            duration_ms=durations[i],
            agent_type=worker.harness.kind,
            model=worker.model,
            agent_session_id=answer.session_id,
            finding_count=len(findings),
            phase="process",
            cost_usd=cost_share,
            usage=TokenUsage(**{k: v[i] for k, v in shares.items()}),
        )
        added = append_analysis(record, entry, findings)
        store.release(record, "analyzed")
        run.stats.files_processed += 1
        run.stats.findings_count += added
        run.stats.total_input_tokens += shares["input_tokens"][i]
        run.stats.total_output_tokens += shares["output_tokens"][i]
        run.stats.total_duration_ms += durations[i]
        if cost_share is not None:
            run.stats.total_cost_usd += cost_share
    return n
