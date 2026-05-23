"""Durability tests for ``temporal-direct/main.py``.

Three checks:

1. **Happy path** — the workflow runs end-to-end via an embedded dev
   server and produces a non-empty text answer.

2. **Replay determinism** — the recorded event history of a successful
   run replays cleanly against the current workflow code (no
   non-determinism error). This is what catches accidental sources of
   non-determinism (clocks, randomness, set iteration order, etc.) in
   the loop or the framework.

3. **Activity caching across a worker restart** — start worker1 in a
   subprocess, kick off the workflow, ``SIGINT`` the subprocess
   mid-flight, then start a fresh in-process worker. A file-backed
   activity log proves that activities completed before the crash are
   *not* re-executed: their results are served from history.

   The subprocess pattern matters: a graceful in-process
   ``worker.shutdown()`` leaves the cached ``WorkflowInstance`` alive
   in memory with its asyncio tasks pending, which produces unraisable
   context-mismatch errors when GC eventually closes those coroutines.
   Real worker death is process death, where in-memory residue is a
   non-issue — and that's what we simulate here. (We use ``SIGINT``
   rather than ``SIGKILL`` so the subprocess gets a chance to cancel
   its in-flight activity instead of leaving it pinned for
   ``start_to_close_timeout``; the subprocess exits via ``os._exit``
   to skip the GC pass that would otherwise emit the same unraisables.)

Run with::

    AI_GATEWAY_API_KEY=... uv run python test_durability.py
"""

from __future__ import annotations

import asyncio
import gc
import os
import pathlib
import signal
import sys
import tempfile
import traceback
import uuid
from collections import Counter
from typing import Any

import main as ex
import temporalio.client
import temporalio.testing
import temporalio.worker
from _durability_worker import LOGGED_ACTIVITIES  # noqa: PLC2701

# ── Helpers ──────────────────────────────────────────────────────


def make_worker(client: temporalio.client.Client) -> temporalio.worker.Worker:
    return temporalio.worker.Worker(
        client,
        task_queue=ex.TASK_QUEUE,
        workflows=[ex.WeatherWorkflow],
        activities=LOGGED_ACTIVITIES,
    )


def read_activity_log(log_file: pathlib.Path) -> Counter[str]:
    if not log_file.exists():
        return Counter()
    return Counter(log_file.read_text().splitlines())


QUERY = "What's the weather and population of New York and Los Angeles?"


# ── Test 1: happy path ───────────────────────────────────────────


async def test_happy_path(
    client: temporalio.client.Client, log_file: pathlib.Path
) -> tuple[str, int]:
    print("\n── test_happy_path ────────────────────────────────")
    log_file.write_text("")

    async with make_worker(client):
        wid = f"happy-{uuid.uuid4().hex[:8]}"
        result = await client.execute_workflow(
            ex.WeatherWorkflow.run,
            QUERY,
            id=wid,
            task_queue=ex.TASK_QUEUE,
        )

    assert result, "expected non-empty text"
    assert (
        "8,336,817" in result or "8336817" in result
    ), f"expected NYC population in result, got: {result!r}"
    counts = read_activity_log(log_file)
    print(f"  ✓ workflow {wid} produced {len(result)} chars")
    print(f"  ✓ activity calls: {dict(counts)}")
    return wid, counts.total()


# ── Test 2: replay determinism ───────────────────────────────────


async def test_replay_determinism(
    client: temporalio.client.Client, workflow_id: str
) -> None:
    print("\n── test_replay_determinism ────────────────────────")
    handle = client.get_workflow_handle(workflow_id)
    history = await handle.fetch_history()

    replayer = temporalio.worker.Replayer(workflows=[ex.WeatherWorkflow])
    # Raises a non-determinism error if the recorded events don't
    # line up with what the current workflow code would produce.
    await replayer.replay_workflow(history)
    print(f"  ✓ replay clean for {workflow_id} ({len(history.events)} events)")


# ── Test 3: activity caching across a worker restart ─────────────


async def test_activity_caching(
    env: temporalio.testing.WorkflowEnvironment,
    client: temporalio.client.Client,
    log_file: pathlib.Path,
    baseline_total: int,
) -> None:
    print("\n── test_activity_caching ──────────────────────────")
    log_file.write_text("")

    wid = f"resume-{uuid.uuid4().hex[:8]}"

    # Worker 1 lives in a subprocess so we can shut it down by signal,
    # mimicking real worker process death. The child inherits stdout/
    # stderr so any Temporal output is visible in our logs.
    worker_script = pathlib.Path(__file__).parent / "_durability_worker.py"
    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        str(worker_script),
        "--server-addr",
        client.service_client.config.target_host,
        "--namespace",
        client.namespace,
        env={**os.environ, "DURABILITY_ACTIVITY_LOG": str(log_file)},
    )
    try:
        # No READY handshake needed: ``start_workflow`` queues the run
        # server-side; whichever worker shows up first picks it up.
        handle = await client.start_workflow(
            ex.WeatherWorkflow.run,
            QUERY,
            id=wid,
            task_queue=ex.TASK_QUEUE,
        )
        # Poll the activity log for completions written by the subprocess.
        # Wait until we've seen the first LLM call and at least one tool
        # complete: that puts us partway through the workflow with several
        # activities still to schedule, so worker2 has actual work to do.
        deadline = asyncio.get_event_loop().time() + 30
        while asyncio.get_event_loop().time() < deadline:
            counts = read_activity_log(log_file)
            if counts.get("llm_call", 0) >= 1 and counts.total() >= 2:
                break
            await asyncio.sleep(0.05)
        else:
            raise AssertionError(
                "no activities executed before timeout — worker slow to start?"
            )
    finally:
        # SIGINT → KeyboardInterrupt → asyncio.run cleans up → Worker
        # graceful shutdown (default ``graceful_shutdown_timeout=0``
        # cancels in-flight activities immediately so the test isn't
        # held hostage by ``start_to_close_timeout``). The subprocess
        # dies; any in-memory residue dies with it.
        if proc.returncode is None:
            proc.send_signal(signal.SIGINT)
            await proc.wait()

    pre_restart = dict(read_activity_log(log_file))
    print(f"  pre-restart activity calls (worker1): {pre_restart}")

    # Worker 2: in-process. Workflow resumes from history.
    async with make_worker(client):
        result = await handle.result()

    post_restart = dict(read_activity_log(log_file))
    print(f"  post-restart activity calls (worker1+worker2): {post_restart}")
    assert result, "workflow returned empty result after resume"

    total_pre = sum(pre_restart.values())
    total_post = sum(post_restart.values())

    # Sanity: we actually killed worker1 mid-workflow. If worker1 had
    # finished everything before the SIGINT landed, the test would
    # vacuously "pass" the cache invariant without exercising resume.
    assert total_pre < baseline_total, (
        f"worker1 finished the entire workflow before shutdown landed "
        f"(pre={total_pre}, baseline={baseline_total}); not exercising resume"
    )

    # Cache invariant: post-restart total equals one full workflow's
    # worth of completions (from the happy-path baseline). If worker2
    # had ignored history and re-run cached activities, total_post
    # would exceed baseline_total.
    assert total_post == baseline_total, (
        f"unexpected activity count after resume: {total_post} "
        f"(baseline is {baseline_total}); worker2 may have re-run "
        f"cached activities"
    )
    print("  ✓ resume completed without re-running cached activities")
    print(
        f"    (pre={total_pre}, post={total_post}, baseline={baseline_total})"
    )


# ── Entry point ──────────────────────────────────────────────────
#
# Coroutine cleanup that raises during asyncio shutdown surfaces as
# "Exception ignored while closing generator …" via ``sys.unraisablehook``
# rather than propagating out of ``asyncio.run``. For a durability test
# that's a real failure — capture them and fail at the end.

_unraisables: list[str] = []
_orig_unraisablehook = sys.unraisablehook


def _unraisablehook(unraisable: Any) -> None:
    msg = unraisable.err_msg or f"Exception ignored in: {unraisable.object!r}"
    tb = "".join(
        traceback.format_exception(
            unraisable.exc_type, unraisable.exc_value, unraisable.exc_traceback
        )
    )
    _unraisables.append(f"{msg}\n{tb}")
    _orig_unraisablehook(unraisable)


async def main() -> None:
    log_file = pathlib.Path(tempfile.mkdtemp()) / "activity_log.txt"
    # In-process workers (worker2 below, plus the happy-path worker)
    # share this module's ``LOGGED_ACTIVITIES``, which look at this env
    # var to decide whether to log. Set it for the whole run.
    os.environ["DURABILITY_ACTIVITY_LOG"] = str(log_file)

    print("Starting embedded Temporal dev server...")
    async with (
        await temporalio.testing.WorkflowEnvironment.start_local() as env
    ):
        client = env.client

        wid, baseline_total = await test_happy_path(client, log_file)
        await test_replay_determinism(client, wid)
        await test_activity_caching(env, client, log_file, baseline_total)

    print("\nAll durability checks passed.")


if __name__ == "__main__":
    sys.unraisablehook = _unraisablehook
    try:
        asyncio.run(main())
    except AssertionError as exc:
        print(f"\nFAIL: {exc}", file=sys.stderr)
        sys.exit(1)

    # Some unraisable closures fire only during interpreter-shutdown
    # GC. Force a collection now so any such closures land while we
    # still control the exit code.
    gc.collect()

    if _unraisables:
        print(
            f"\nFAIL: {len(_unraisables)} unraisable exception(s) during run",
            file=sys.stderr,
        )
        sys.exit(1)
