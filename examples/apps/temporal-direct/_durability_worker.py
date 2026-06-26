"""Subprocess worker for ``test_durability``.

Run in its own process so the parent test can shut it down by signal
instead of doing a graceful in-process ``worker.shutdown()`` — which
leaves cached ``WorkflowInstance``\\ s and their pending asyncio tasks
alive in memory and produces unraisable context-mismatch errors at
interpreter shutdown. With a subprocess, the residue dies with the
process; the parent never sees it.

We use ``SIGINT`` (raises ``KeyboardInterrupt`` → ``asyncio.run``
cleans up → ``Worker.__aexit__`` runs ``shutdown()``) rather than
``SIGKILL``. ``SIGKILL`` would leave any in-flight activity hanging
until ``start_to_close_timeout`` fires (5 min for the LLM activity),
which makes the test painfully slow. The ``SIGINT`` path is what a
production rollout's ``preStop``/``SIGTERM`` looks like in spirit.

Activities log every completion to a file named in
``DURABILITY_ACTIVITY_LOG`` so the parent can count executions across
the process boundary.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import os
import threading
from typing import TYPE_CHECKING, Any

import main as ex
import temporalio.activity
import temporalio.client
import temporalio.worker

if TYPE_CHECKING:
    from collections.abc import Callable

_log_lock = threading.Lock()


def _log_activity(name: str) -> None:
    log_file = os.environ.get("DURABILITY_ACTIVITY_LOG")
    if not log_file:
        return
    with _log_lock, open(log_file, "a") as f:
        f.write(f"{name}\n")


# Activities log *after* completion so the file is a record of activities
# that actually finished (on whichever worker ran them). If a worker dies
# mid-activity, that attempt never logs; only the eventual completion (on
# the same or a different worker) does. The crash test relies on this:
# pre-restart counts == activities completed on worker1, and post-restart
# total should equal one full workflow's worth of completions.


@temporalio.activity.defn(name="get_weather_activity")
async def logged_get_weather(city: str) -> str:
    result = await ex.get_weather_activity(city)
    _log_activity("get_weather")
    return result


@temporalio.activity.defn(name="get_population_activity")
async def logged_get_population(city: str) -> int:
    result = await ex.get_population_activity(city)
    _log_activity("get_population")
    return result


@temporalio.activity.defn(name="llm_call_activity")
async def logged_llm_call(params: ex.LLMParams) -> ex.LLMResult:
    result = await ex.llm_call_activity(params)
    _log_activity("llm_call")
    return result


LOGGED_ACTIVITIES: list[Callable[..., Any]] = [
    logged_llm_call,
    logged_get_weather,
    logged_get_population,
]


async def amain(server_addr: str, namespace: str) -> None:
    client = await temporalio.client.Client.connect(
        server_addr, namespace=namespace
    )
    async with temporalio.worker.Worker(
        client,
        task_queue=ex.TASK_QUEUE,
        workflows=[ex.WeatherWorkflow],
        activities=LOGGED_ACTIVITIES,
        # Serialize activities so the parent has a deterministic window
        # to interrupt between them (the workflow's tool batch is
        # ``asyncio.gather``-parallel; without this knob, the four tool
        # activities all complete before the parent's poll loop wakes).
        max_concurrent_activities=1,
    ):
        # Run forever. The parent sends SIGINT when it's done.
        with contextlib.suppress(KeyboardInterrupt, asyncio.CancelledError):
            await asyncio.Event().wait()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--server-addr", required=True)
    p.add_argument("--namespace", default="default")
    args = p.parse_args()
    asyncio.run(amain(args.server_addr, args.namespace))
    # Skip the interpreter's atexit / final-GC pass. The cached
    # ``WorkflowInstance`` is still in memory at this point; letting GC
    # close its leaked coroutines would print "Exception ignored …
    # ContextVar … was created in a different Context" to our (inherited)
    # stderr, which the parent test would treat as a failure. Real
    # production worker death is process exit without Python cleanup, so
    # this matches that semantics: memory dies with the process.
    os._exit(0)
