"""Temporal activities — all real I/O lives here.

Both examples (direct composition and provider) share these activities.
Each activity is a plain async function that does real I/O.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import temporalio.activity

import vercel_ai_sdk as ai

# ── Tool activities ──────────────────────────────────────────────


@temporalio.activity.defn(name="get_weather")
async def get_weather_activity(city: str) -> str:
    return f"Sunny, 72F in {city}"


@temporalio.activity.defn(name="get_population")
async def get_population_activity(city: str) -> int:
    return {"new york": 8_336_817, "los angeles": 3_979_576}.get(
        city.lower(), 1_000_000
    )


# ── Generic tool dispatch activity ──────────────────────────────
#
# Used by the provider example: the provider routes tool calls here
# instead of executing them inside the workflow.

_TOOL_REGISTRY: dict[str, Any] = {
    "get_weather": get_weather_activity,
    "get_population": get_population_activity,
}


@dataclasses.dataclass
class ToolDispatchParams:
    tool_name: str
    tool_args: str  # JSON string


@dataclasses.dataclass
class ToolDispatchResult:
    result: Any
    is_error: bool = False


@temporalio.activity.defn(name="tool_dispatch")
async def tool_dispatch_activity(params: ToolDispatchParams) -> ToolDispatchResult:
    """Dispatch a tool call by name. Runs the real tool function."""
    import json

    fn = _TOOL_REGISTRY.get(params.tool_name)
    if fn is None:
        return ToolDispatchResult(
            result=f"Unknown tool: {params.tool_name}", is_error=True
        )

    try:
        kwargs = json.loads(params.tool_args) if params.tool_args else {}
        result = await fn(**kwargs)
        return ToolDispatchResult(result=result)
    except Exception as exc:
        return ToolDispatchResult(result=str(exc), is_error=True)


# ── LLM activity ────────────────────────────────────────────────


@dataclasses.dataclass
class LLMCallParams:
    messages: list[dict[str, Any]]
    tool_schemas: list[dict[str, Any]]


@dataclasses.dataclass
class LLMCallResult:
    message: dict[str, Any]  # serialized ai.Message


@temporalio.activity.defn(name="llm_call")
async def llm_call_activity(params: LLMCallParams) -> LLMCallResult:
    """Call the LLM, drain the stream, return the final message."""
    model = ai.Model(
        id="anthropic/claude-sonnet-4-20250514",
        adapter="ai-gateway-v3",
        provider="ai-gateway",
    )

    messages = [ai.Message.model_validate(m) for m in params.messages]
    tools = [ai.ToolSchema(return_type=None, **t) for t in params.tool_schemas]

    s = await ai.models.stream(model, messages, tools=tools)
    result = await ai.models.buffer(s)
    return LLMCallResult(message=result.model_dump())
