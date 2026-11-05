"""``ai.harness.Agent``: a coding agent loop for long-running processes.

The loop is the ``ai.Agent`` loop with steering, limits and compaction.
Settings are class variables. Make the system message with
:func:`.prompts.system_message` and put it first in the messages.

Steering: before each request, the loop asks the interface for user
messages with the ``steer`` hook. The interface must answer each ask.
If it has no messages, it answers with an empty :class:`Steer`.

Limits: ``MAX_TURNS`` is per run. ``MAX_TOKENS`` and ``MAX_USD`` are
shared with subagents (see :class:`Budget`).

Interrupting: cancel the task that runs the agent. The loop adds the
partial turn to ``stream.messages``. Each tool call that did not finish
gets an ``INTERRUPTED`` error result.
"""

from __future__ import annotations

import asyncio
import contextvars
import dataclasses
from collections.abc import AsyncGenerator, Sequence
from typing import TYPE_CHECKING, ClassVar, Literal

import pydantic

from .. import _modelsdev, agents, errors, models, util
from .. import experimental_telemetry as telemetry
from ..types import builders, events
from ..types import messages as messages_
from ..types import usage as usage_
from . import events as harness_events
from . import prompts, tools

if TYPE_CHECKING:
    import modelsdotdev

STEER = "steer"

# result of a tool call that was cut short by an interrupt
INTERRUPTED = "Interrupted by user"

# The model of the current run, for the `subagent` tool. `loop` sets it.
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


class UnknownCostError(errors.AIError):
    """``MAX_USD`` is set, but the cost of a request cannot be known.

    models.dev has no token prices for the model, or the provider did not
    report the usage of a request.
    """


@dataclasses.dataclass
class Budget:
    """What a run and its subagents spent, and the limits they share.

    The top-level run makes it from its ``MAX_TOKENS`` and ``MAX_USD``.
    Subagent runs find it in a contextvar, add their spend to it, and
    stop at the same limits; their own ``MAX_TOKENS`` and ``MAX_USD`` are
    not used. A request counts when its model stream ends, before the
    turn's tools run. Limits are checked before each request, so a run
    can go over by its last request, and parallel subagents by one
    request each. Tools start when their call ends, so a subagent can
    send its first request before the parent's request is counted. A
    request cut short by an interrupt is not counted: its usage is never
    reported.
    """

    max_tokens: int | None = None
    max_usd: float | None = None
    usage: usage_.Usage = dataclasses.field(default_factory=usage_.Usage)
    usd: float = 0.0

    def spend(
        self,
        request: usage_.Usage | None,
        *,
        prices: Sequence[modelsdotdev.Cost] | None,
        model_id: str,
    ) -> None:
        """Add the usage of one request. ``None``: it was not reported."""
        if request is None:
            if self.max_usd is not None:
                raise UnknownCostError(
                    f"{model_id} did not report usage; "
                    "MAX_USD cannot be enforced"
                )
            return
        self.usage += request
        if prices:
            self.usd += to_usd(prices, request)

    def exceeded(self) -> Literal["tokens", "usd"] | None:
        """Return the limit that was reached, or None."""
        if (
            self.max_tokens is not None
            and self.usage.total_tokens >= self.max_tokens
        ):
            return "tokens"
        if self.max_usd is not None and self.usd >= self.max_usd:
            return "usd"
        return None


def to_usd(prices: Sequence[modelsdotdev.Cost], usage: usage_.Usage) -> float:
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


def _shared_budget(max_tokens: int | None, max_usd: float | None) -> Budget:
    """Get the shared budget. The top-level run makes and sets it."""
    budget = _budget.get(None)
    if budget is None:
        budget = Budget(max_tokens=max_tokens, max_usd=max_usd)
        _budget.set(budget)
    return budget


def _add_partial_turn(
    context: agents.Context,
    message: messages_.Message,
    tool_message: messages_.Message | None,
    ended: set[str],
) -> None:
    """Add an interrupted turn to the history.

    Tool calls with incomplete args are dropped. Each kept call gets a
    result: a call without one would be replayed (run again) by the
    next run.
    """
    parts = [
        p
        for p in message.parts
        if not isinstance(p, messages_.ToolCallPart) or p.tool_call_id in ended
    ]
    if parts:
        context.add(message.model_copy(update={"parts": parts}))
    results = list(tool_message.tool_results) if tool_message else []
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


class Agent(agents.Agent):
    TOOLS: ClassVar[list[agents.AgentTool]] = [tools.read, tools.bash]
    STEERING: ClassVar[bool] = True
    MAX_TURNS: ClassVar[int | None] = None
    MAX_TOKENS: ClassVar[int | None] = None  # shared with subagents
    MAX_USD: ClassVar[float | None] = None  # shared with subagents
    RESERVE_TOKENS: ClassVar[int] = 16_384
    KEEP_RECENT_TOKENS: ClassVar[int] = 20_000

    # The loop yields harness events too; they join `AgentEvent` later.
    async def loop(  # type: ignore[override]  # ty: ignore[invalid-method-override]
        self, context: agents.Context
    ) -> AsyncGenerator[harness_events.HarnessEvent]:
        _model.set(context.model)

        model_id = f"{context.model.provider.name}:{context.model.id}"
        info = _modelsdev.get_model_by_id(model_id)
        window = (
            info.limits.context if info and info.limits.context else 128_000
        )
        prices = info.cost if info else None
        budget = _shared_budget(self.MAX_TOKENS, self.MAX_USD)
        if budget.max_usd is not None and not prices:
            raise UnknownCostError(
                f"models.dev has no token prices for {model_id}; "
                "MAX_USD cannot be enforced"
            )

        # size of the last request, for the compaction trigger
        last = next(
            (m.usage for m in reversed(context.messages) if m.usage), None
        )
        turns = 0

        while True:
            # ── steering: ask the interface before each request, and
            # once more before stopping, so a late message continues
            if self.STEERING:
                steer = await agents.hook(STEER, payload=Steer)
                if steer.messages:
                    new = [builders.user_message(t) for t in steer.messages]
                    context.add(new)
                    yield harness_events.Steered(messages=tuple(new))
            if not context.keep_running():
                return

            # ── limits: turns are per run, the budget is shared ──
            limit: Literal["turns", "tokens", "usd"] | None = (
                "turns"
                if self.MAX_TURNS is not None and turns >= self.MAX_TURNS
                else budget.exceeded()
            )
            if limit is not None:
                yield harness_events.LimitReached(
                    limit=limit, usage=budget.usage, usd=budget.usd
                )
                return

            # ── compaction, when the last request was near the window ──
            if (
                last is not None
                and last.total_tokens > window - self.RESERVE_TOKENS
            ):
                before = len(context.messages)
                compacted, summary, cost = await compact(
                    context, keep_recent_tokens=self.KEEP_RECENT_TOKENS
                )
                if summary is not None:
                    context.messages[:] = compacted
                    budget.spend(cost, prices=prices, model_id=model_id)
                    last = None
                    yield harness_events.Compacted(
                        summary=summary,
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
                            budget.spend(
                                event.usage, prices=prices, model_id=model_id
                            )
                        yield event
                        if isinstance(event, events.ToolEnd):
                            ended.add(event.tool_call.tool_call_id)
                            tr.schedule(self.resolve(event.tool_call))
                except asyncio.CancelledError:
                    _add_partial_turn(
                        context, stream.message, tr.get_tool_message(), ended
                    )
                    raise
                context.add(stream.message)
                context.add(tr.get_tool_message())

            turns += 1
            last = stream.usage


async def compact(
    context: agents.Context, *, keep_recent_tokens: int = 20_000
) -> tuple[
    list[messages_.Message], messages_.Message | None, usage_.Usage | None
]:
    """Replace old history with a summary. Keeps a leading system message.

    Returns the new messages, the summary message, and the usage of the
    summary request. If there is nothing to cut, returns the old
    messages, None and None.
    """
    n = 1 if context.messages and context.messages[0].role == "system" else 0
    system, history = context.messages[:n], context.messages[n:]

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
        return context.messages, None, None

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
    response = await models.experimental_generate(
        context.model,
        [
            *system,
            *history[:cut],
            builders.user_message(prompts.SUMMARY_PROMPT),
        ],
        tools=context.tools,
        params=params,
    )

    summary = builders.user_message(
        f"{prompts.SUMMARY_PREFIX}{response.text}\n</summary>"
    )
    return [*system, summary, *history[cut:]], summary, response.usage


class Subagent(Agent):
    """The agent behind the ``subagent`` tool.

    No ``subagent`` tool, so no nesting. No steering: the subagent shares
    the parent's hook registry, so its asks would take the parent's
    messages, and parallel subagents would collide on the label.
    """

    TOOLS: ClassVar[list[agents.AgentTool]] = [tools.read, tools.bash]
    STEERING = False


@agents.tool
async def subagent(prompt: str) -> agents.SubAgentTool:
    """Give a self-contained task to a subagent with a fresh context.

    It returns its final answer. Use it for broad searches or
    independent work. The prompt must contain everything it needs.
    """
    messages = [prompts.system_message(), builders.user_message(prompt)]
    async with Subagent().run(_model.get(), messages) as stream:
        async for event in stream:
            yield event


Agent.TOOLS = [*Agent.TOOLS, subagent]

__all__ = [
    "INTERRUPTED",
    "STEER",
    "Agent",
    "Budget",
    "Steer",
    "Subagent",
    "UnknownCostError",
    "compact",
    "subagent",
    "to_usd",
]
