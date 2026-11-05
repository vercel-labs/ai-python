"""``ai.harness.Agent``: a coding agent loop for long-running processes.

The loop is the standard ``ai.Agent`` loop plus, in order: a built system
prompt (env facts, ``AGENTS.md`` files, a skill list), steering through
the ``steer`` hook, turn/token limits, and compaction by a client-side
summary. The ``task`` tool runs a subagent. Settings are class variables.

Budgets: ``MAX_TOKENS`` and ``MAX_USD`` cover the run and its subagents
(see :class:`Budget`). ``MAX_USD`` is best-effort, from models.dev
prices; without prices for the model the run raises
:class:`UnknownCostError` before its first request.

Steering: before each model request the loop asks the interface for new
messages with ``ai.hook("steer", payload=Steer)``. The interface keeps
its own buffer and answers every ask, with an empty list if it has
nothing::

    if isinstance(event, ai.events.PendingHookEvent) and (
        event.hook.hook_id == ai.harness.STEER
    ):
        ai.resolve_hook(event.hook, ai.harness.Steer(messages=queued))
        queued.clear()

The run waits until the ask is answered.

Interrupting: cancel the task that runs the agent. The loop adds the
partial turn to the history (``stream.messages``) before the cancel goes
on: the text so far, the finished tool results, and an ``INTERRUPTED``
error result for every tool call that did not finish. To interrupt and
redirect, start a new run on that history with the new user message.
"""

from __future__ import annotations

import asyncio
import contextvars
import dataclasses
import datetime
import pathlib
import platform
from collections.abc import AsyncGenerator, Sequence
from typing import TYPE_CHECKING, ClassVar, Literal

import pydantic

from .. import _modelsdev, agents, errors, models, util
from .. import experimental_telemetry as telemetry
from ..types import builders, events
from ..types import messages as messages_
from ..types import usage as usage_
from . import tools

if TYPE_CHECKING:
    import modelsdotdev

STEER = "steer"

# result of a tool call that was cut short by an interrupt
INTERRUPTED = "Interrupted by user"

PROMPT = """\
You are a coding agent that works in the user's local environment.
Use your tools to inspect files and run commands; do not guess.
- read: read files and images.
- bash: run shell commands (search, edit, write, git, curl, ...).
- task: hand a self-contained job to a subagent with a fresh context.
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

# The model of the current run, for the `task` tool. `loop` sets it.
# `decouple` runs the loop in its own task, so the value does not leak
# out of the run, and tool tasks created by the loop inherit it.
_model: contextvars.ContextVar[models.Model] = contextvars.ContextVar(
    "harness_model"
)

# The budget of the top-level run. `loop` sets it when it is not set, so
# subagent runs, which run in tool tasks of the parent, share it.
_budget: contextvars.ContextVar[Budget] = contextvars.ContextVar(
    "harness_budget"
)


class Steer(pydantic.BaseModel):
    """Answer to the ``steer`` hook: user messages for the next request.

    Empty when the interface has nothing to add.
    """

    messages: list[str] = []


class Steered(events.BaseEvent):
    """Steering messages were added to the history."""

    messages: tuple[messages_.Message, ...]
    kind: Literal["harness_steered"] = "harness_steered"


class Compacted(events.BaseEvent):
    """Old history was replaced by a summary."""

    summary: messages_.Message
    removed: int
    usage: usage_.Usage | None
    kind: Literal["harness_compacted"] = "harness_compacted"


class LimitReached(events.BaseEvent):
    """The run stopped at ``MAX_TURNS``, ``MAX_TOKENS`` or ``MAX_USD``.

    ``usage`` and ``usd`` are the budget's totals: this run and the runs
    that share its budget (subagents).
    """

    limit: Literal["turns", "tokens", "usd"]
    usage: usage_.Usage
    usd: float
    kind: Literal["harness_limit_reached"] = "harness_limit_reached"


class UnknownCostError(errors.AIError):
    """``MAX_USD`` is set, but the cost of a request cannot be known.

    models.dev has no token prices for the model, or the provider did not
    report the usage of a request.
    """


@dataclasses.dataclass
class Budget:
    """What a run and its subagents spent, and the limits they share.

    The top-level run makes it from its ``MAX_TOKENS`` and ``MAX_USD``.
    Subagent runs (``task``) find it in a contextvar, add their spend to
    it, and stop at the same limits; their own ``MAX_TOKENS`` and
    ``MAX_USD`` are not used. A request counts when its model stream
    ends, before the turn's tools run. Limits are checked before each
    request, so a run can go over by its last request, and parallel
    subagents by one request each. Tools start when their call ends, so a
    subagent can send its first request before the parent's request is
    counted. A request cut short by an interrupt is not counted: its usage is
    never reported.
    """

    max_tokens: int | None = None
    max_usd: float | None = None
    usage: usage_.Usage = dataclasses.field(default_factory=usage_.Usage)
    usd: float = 0.0


def usd(prices: Sequence[modelsdotdev.Cost], usage: usage_.Usage) -> float:
    """Best-effort cost of one request in USD, from models.dev prices.

    Prices are per million tokens. The tier is picked by the request's
    input size. It errs high: cache writes are charged on top of
    ``input_tokens`` (Anthropic does not count them there, other
    providers may), and reasoning is priced as output unless the model
    has a reasoning price.
    """
    tier = next(
        (p for p in reversed(prices) if p.min_context <= usage.input_tokens),
        prices[0],
    )
    cache_read = usage.cache_read_tokens or 0
    cache_write = usage.cache_write_tokens or 0
    reasoning = usage.reasoning_tokens or 0
    return (
        max(usage.input_tokens - cache_read, 0) * tier.input
        + cache_read
        * (tier.input if tier.cache_read is None else tier.cache_read)
        + cache_write
        * (tier.input if tier.cache_write is None else tier.cache_write)
        + (
            usage.output_tokens * tier.output
            if tier.reasoning is None
            else max(usage.output_tokens - reasoning, 0) * tier.output
            + reasoning * tier.reasoning
        )
    ) / 1_000_000


type HarnessEvent = events.AgentEvent | Steered | Compacted | LimitReached


class Agent(agents.Agent):
    TOOLS: ClassVar[list[agents.AgentTool]] = [tools.read, tools.bash]  # + task
    STEERING: ClassVar[bool] = True
    MAX_TURNS: ClassVar[int | None] = None
    MAX_TOKENS: ClassVar[int | None] = None  # shared with subagents
    MAX_USD: ClassVar[float | None] = None  # shared with subagents
    RESERVE_TOKENS: ClassVar[int] = 16_384
    KEEP_RECENT_TOKENS: ClassVar[int] = 20_000
    SKILL_DIRS: ClassVar[tuple[str, ...]] = (
        ".agents/skills",
        "~/.agents/skills",
    )

    # The loop yields harness events too; they join `AgentEvent` later.
    async def loop(  # type: ignore[override]  # ty: ignore[invalid-method-override]
        self, context: agents.Context
    ) -> AsyncGenerator[HarnessEvent]:
        _model.set(context.model)

        # ── system prompt: built only if the caller gave none ──
        if not context.messages or context.messages[0].role != "system":
            cwd = pathlib.Path.cwd()
            sections = [
                PROMPT,
                f"cwd: {cwd}\nos: {platform.system()} {platform.release()}\n"
                f"date: {datetime.date.today().isoformat()}",
            ]
            # AGENTS.md from the filesystem root down to cwd, so the
            # most specific file comes last and wins
            for d in reversed([cwd, *cwd.parents]):
                if (f := d / "AGENTS.md").is_file():
                    sections.append(
                        f"Instructions from {f}:\n\n{f.read_text()}"
                    )
            # skills: name + description from SKILL.md frontmatter; the
            # model reads the file itself with `read`
            skills = []
            for root in self.SKILL_DIRS:
                for f in sorted(
                    pathlib.Path(root).expanduser().glob("*/SKILL.md")
                ):
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
                        f"    <description>{meta['description']}"
                        "</description>\n"
                        f"    <location>{f.resolve()}</location>\n"
                        f"  </skill>"
                    )
            if skills:
                sections.append(
                    "Skills give task-specific instructions. When a task "
                    "matches a skill, read its SKILL.md first.\n"
                    "<available_skills>\n"
                    + "\n".join(skills)
                    + "\n</available_skills>"
                )
            context.messages.insert(
                0, builders.system_message("\n\n".join(sections))
            )

        info = _modelsdev.get_model_by_id(
            f"{context.model.provider.name}:{context.model.id}"
        )
        window = (
            info.limits.context if info and info.limits.context else 128_000
        )
        # size of the last request, for the compaction trigger
        last = next(
            (m.usage for m in reversed(context.messages) if m.usage), None
        )
        turns = 0

        # ── budget: the top-level run makes it, subagents share it ──
        budget = _budget.get(None)
        if budget is None:
            budget = Budget(max_tokens=self.MAX_TOKENS, max_usd=self.MAX_USD)
            _budget.set(budget)
        model_id = f"{context.model.provider.name}:{context.model.id}"
        prices = info.cost if info else None
        if budget.max_usd is not None and not prices:
            raise UnknownCostError(
                f"models.dev has no token prices for {model_id}; "
                "MAX_USD cannot be enforced"
            )

        def spend(request: usage_.Usage | None) -> None:
            if request is None:
                if budget.max_usd is not None:
                    raise UnknownCostError(
                        f"{model_id} did not report usage; "
                        "MAX_USD cannot be enforced"
                    )
                return
            budget.usage += request
            if prices:
                budget.usd += usd(prices, request)

        while True:
            # ── steering: ask the interface before each request, and
            # once more before stopping, so a late message continues
            if self.STEERING:
                steer = await agents.hook(STEER, payload=Steer)
                if steer.messages:
                    new = [builders.user_message(t) for t in steer.messages]
                    context.add(new)
                    yield Steered(messages=tuple(new))
            if not context.keep_running():
                return

            # ── limits: turns are per run, the budget is shared ──
            limit: Literal["turns", "tokens", "usd"] | None = None
            if self.MAX_TURNS is not None and turns >= self.MAX_TURNS:
                limit = "turns"
            elif (
                budget.max_tokens is not None
                and budget.usage.total_tokens >= budget.max_tokens
            ):
                limit = "tokens"
            elif budget.max_usd is not None and budget.usd >= budget.max_usd:
                limit = "usd"
            if limit is not None:
                yield LimitReached(
                    limit=limit, usage=budget.usage, usd=budget.usd
                )
                return

            # ── compaction, when the last request was near the window ──
            if (
                last is not None
                and last.total_tokens > window - self.RESERVE_TOKENS
            ):
                before = len(context.messages)
                compacted, cost = await compact(
                    context, keep_recent_tokens=self.KEEP_RECENT_TOKENS
                )
                if len(compacted) < before:
                    context.messages[:] = compacted
                    spend(cost)
                    last = None
                    yield Compacted(
                        summary=compacted[1],
                        removed=before - len(compacted) + 1,
                        usage=cost,
                    )

            # ── one turn: the standard loop body ──
            async with (
                telemetry.span(telemetry.LoopTurnSpanData()),
                models.stream(context=context) as stream,
                agents.ToolRunner() as tr,
                util.merge(stream, tr.events()) as merged,
            ):
                ended: set[str] = set()  # tool calls with complete args
                try:
                    async for event in merged:
                        if (
                            isinstance(event, events.StreamEnd)
                            and not event.replay
                        ):
                            # count the request now, not after the turn:
                            # its tools (subagents) check the same budget
                            spend(event.usage)
                        yield event
                        if isinstance(event, events.ToolEnd):
                            ended.add(event.tool_call.tool_call_id)
                            tr.schedule(self.resolve(event.tool_call))
                except asyncio.CancelledError:
                    # interrupted: keep the partial turn, but every tool
                    # call must have a result. A call without one would
                    # be replayed (run again) by the next run.
                    parts = [
                        p
                        for p in stream.message.parts
                        if not isinstance(p, messages_.ToolCallPart)
                        or p.tool_call_id in ended
                    ]
                    if parts:
                        context.add(
                            stream.message.model_copy(update={"parts": parts})
                        )
                    done = tr.get_tool_message()
                    results = list(done.tool_results) if done else []
                    answered = {r.tool_call_id for r in results}
                    results += [
                        builders.tool_result_part(
                            p.tool_call_id,
                            tool_name=p.tool_name,
                            result=INTERRUPTED,
                            is_error=True,
                        )
                        for p in parts
                        if isinstance(p, messages_.ToolCallPart)
                        and p.tool_call_id not in answered
                    ]
                    if results:
                        context.add(builders.tool_message(*results))
                    raise
                context.add(stream.message)
                context.add(tr.get_tool_message())

            turns += 1
            last = stream.usage


async def compact(
    context: agents.Context, *, keep_recent_tokens: int = 20_000
) -> tuple[list[messages_.Message], usage_.Usage | None]:
    """Replace old history with a summary. Keeps ``messages[0]`` (system).

    Returns the new messages and the usage of the summary request, or
    the old messages and None if there is nothing to cut.
    """
    system, history = context.messages[:1], context.messages[1:]

    # ── cut point: walk back from the end, about `keep_recent_tokens`
    # deep. Cut only before a user/assistant message, so a tool call
    # and its result stay on the same side. history[0] always goes.
    cut, kept = len(history), 0
    for i in range(len(history) - 1, 0, -1):
        kept += len(history[i].model_dump_json()) // 4  # rough token count
        if history[i].role in ("user", "assistant"):
            cut = i
            if kept >= keep_recent_tokens:
                break
    if cut == len(history):
        return context.messages, None

    # ── summary request: same prefix and tools as the last request, so
    # the prompt cache hits; tool choice "none" forces a text answer
    params = context.params or models.InferenceRequestParams()
    if context.tools:
        params = dataclasses.replace(
            params,
            tool_calling=models.ToolCallingParams(
                tool_choice=models.ToolChoiceMode.NONE
            ),
        )
    async with models.stream(
        context.model,
        [*system, *history[:cut], builders.user_message(SUMMARY_PROMPT)],
        tools=context.tools,
        params=params,
    ) as s:
        async for _ in s:
            pass

    summary = builders.user_message(
        f"{SUMMARY_PREFIX}{s.message.text}\n</summary>"
    )
    return [*system, summary, *history[cut:]], s.usage


class Subagent(Agent):
    """The agent behind ``task``.

    No ``task`` tool, so no nesting. No steering: the subagent shares
    the parent's hook registry, so its asks would take the parent's
    messages, and parallel subagents would collide on the label.
    """

    TOOLS: ClassVar[list[agents.AgentTool]] = [tools.read, tools.bash]
    STEERING = False


@agents.tool
async def task(prompt: str) -> agents.SubAgentTool:
    """Give a self-contained task to a subagent with a fresh context.

    It returns its final answer. Use it for broad searches or
    independent work. The prompt must contain everything it needs.
    """
    async with Subagent().run(
        _model.get(), [builders.user_message(prompt)]
    ) as stream:
        async for event in stream:
            yield event


Agent.TOOLS = [*Agent.TOOLS, task]

__all__ = [
    "INTERRUPTED",
    "PROMPT",
    "STEER",
    "SUMMARY_PREFIX",
    "SUMMARY_PROMPT",
    "Agent",
    "Budget",
    "Compacted",
    "HarnessEvent",
    "LimitReached",
    "Steer",
    "Steered",
    "Subagent",
    "UnknownCostError",
    "compact",
    "task",
    "usd",
]
