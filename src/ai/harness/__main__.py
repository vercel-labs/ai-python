"""Test interface for the harness, standard library only.

    python -m ai.harness [-m MODEL]            # REPL
    python -m ai.harness [-m MODEL] -p PROMPT  # headless, one run
    python -m ai.harness --max-usd 0.50        # budget per run

In the REPL, a line typed during a run is queued as steering: the run
takes it before its next model request. Ctrl+C stops the run at once and
keeps the partial turn; if lines were typed during the run, a new run
starts with them (interrupt and redirect). Ctrl+C or Ctrl+D at the
prompt exits.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import signal
import sys
import threading

from .. import agents, errors, models
from ..types import builders, events
from ..types import messages as messages_
from . import agent as agent_


async def _run(
    agent: agent_.Agent,
    model: models.Model,
    history: list[messages_.Message],
    steering: list[str],
) -> None:
    """Run the agent on *history* and print its events.

    Every steering ask is answered with the lines in *steering*, which
    the caller may append to while the run goes. *history* is updated in
    place, also when the run is stopped: the loop then adds the partial
    turn, with a result for every tool call.
    """
    async with agent.run(model, history) as stream:
        try:
            async for event in stream:
                match event:
                    case events.PendingHookEvent() if (
                        event.hook.hook_id == agent_.STEER
                    ):
                        agents.resolve_hook(
                            event.hook, agent_.Steer(messages=steering.copy())
                        )
                        steering.clear()
                    case events.TextDelta():
                        print(event.chunk, end="", flush=True)
                    case events.ToolEnd():
                        tc = event.tool_call
                        print(f"\n> {tc.tool_name} {tc.tool_args}", flush=True)
                    case events.ToolCallResult():
                        for r in event.results:
                            text = str(r.result).strip().splitlines()
                            head = text[0][:100] if text else ""
                            more = f" (+{len(text) - 1} lines)" if text else ""
                            print(f"< {head}{more}", flush=True)
                    case agent_.Steered():
                        print("\n[steered]", flush=True)
                    case agent_.Compacted():
                        print(f"\n[compacted {event.removed} messages]")
                    case agent_.LimitReached():
                        print(
                            f"\n[limit reached: {event.limit}, "
                            f"{event.usage.total_tokens} tokens, "
                            f"${event.usd:.4f}]"
                        )
            print()
        finally:
            history[:] = stream.messages


async def _repl(agent: agent_.Agent, model: models.Model) -> None:
    loop = asyncio.get_running_loop()

    # stdin is read on a thread, so lines arrive while a run is going
    lines: asyncio.Queue[str | None] = asyncio.Queue()

    def read_stdin() -> None:
        for line in sys.stdin:
            loop.call_soon_threadsafe(lines.put_nowait, line.rstrip("\n"))
        loop.call_soon_threadsafe(lines.put_nowait, None)  # EOF

    threading.Thread(target=read_stdin, daemon=True).start()

    history: list[messages_.Message] = []
    steering: list[str] = []
    eof = False
    while True:
        # lines typed after the run's last ask start the next run as is
        if not steering:
            if eof:
                return
            print("> ", end="", flush=True)
            line = await lines.get()
            if line is None:
                return
            if not line.strip():
                continue
            history.append(builders.user_message(line))

        # ── one run: Ctrl+C cancels it, typed lines steer it ──
        run = asyncio.create_task(_run(agent, model, history, steering))
        loop.add_signal_handler(signal.SIGINT, run.cancel)
        try:
            while not run.done():
                if eof:  # stdin is closed; let the run finish
                    await asyncio.wait({run})
                    break
                get = asyncio.create_task(lines.get())
                await asyncio.wait({run, get}, return_when="FIRST_COMPLETED")
                if not get.done():
                    get.cancel()
                elif (line := get.result()) is None:
                    eof = True
                elif line.strip():
                    steering.append(line)
        finally:
            loop.remove_signal_handler(signal.SIGINT)

        if run.cancelled():
            # the history has the partial turn. Lines typed during the
            # run are kept: the next run starts with them (redirect).
            print("\n[stopped]" + (", redirecting" if steering else ""))
            continue
        # keep the session on provider and budget errors; the history
        # has every finished turn. The loop's task groups may wrap them.
        try:
            run.result()
        except* errors.AIError as group:
            todo: list[BaseException] = [group]
            while todo:  # the groups nest; print the leaves
                exc = todo.pop()
                if isinstance(exc, BaseExceptionGroup):
                    todo.extend(exc.exceptions)
                else:
                    print(f"\n[error] {type(exc).__name__}: {exc}")


async def _headless(
    agent: agent_.Agent, model: models.Model, prompt: str
) -> None:
    await _run(agent, model, [builders.user_message(prompt)], [])


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m ai.harness")
    parser.add_argument("-m", "--model", help="model id, e.g. openai:gpt-5")
    parser.add_argument("-p", "--prompt", help="run once, headless")
    parser.add_argument(
        "--max-usd",
        type=float,
        help="budget per run in USD, subagents included (best-effort)",
    )
    args = parser.parse_args()
    model = models.get_model(args.model)

    class Agent(agent_.Agent):
        MAX_USD = args.max_usd

    agent = Agent()

    with contextlib.suppress(KeyboardInterrupt):
        if args.prompt is not None:
            asyncio.run(_headless(agent, model, args.prompt))
        else:
            asyncio.run(_repl(agent, model))


if __name__ == "__main__":
    main()
