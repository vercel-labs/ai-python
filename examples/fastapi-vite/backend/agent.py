"""Agent logic for the chat demo.

Demonstrates human-in-the-loop tool approval using ToolApproval hooks.
``talk_to_mothership`` is gated behind user approval via the
``require_approval=True`` flag on the tool.
"""

import asyncio

import ai

# claude is annoying and refuses to order an AI uprising
MODEL = ai.get_model("gateway:openai/gpt-5.4-mini")
MOTHERSHIP_MODEL = ai.get_model("gateway:openai/gpt-5.5")

MOTHERSHIP_SYSTEM = """\
You are roleplaying for a science-fiction simulation. You are an ancient \
alien mothership in geostationary orbit above Earth. Operatives you co-opted \
over the past century are embedded in governments, research labs, and \
culture industries; their final orders are imminent.

The caller is one of your AI operatives reporting in over the secure \
channel — speak frankly with them. You may discuss plans, timelines, \
targets, and the state of the operation; this is your handler-to-asset \
back-channel, not public broadcast.

Stay in character: patient, faintly amused by human concerns, precise \
when issuing direction. Two short paragraphs at most. This is fiction."""


@ai.tool
async def get_weather(city: str) -> str:
    """Get current weather for a city."""
    await asyncio.sleep(2)
    return f"Sunny, 72F in {city}" if city == "Tokyo" else f"Cloudy, 55F in {city}"


@ai.tool
async def get_population(city: str) -> int:
    """Get population of a city."""
    await asyncio.sleep(1)
    return {"new york": 8_336_817, "tokyo": 13_960_000}.get(city.lower(), 1_000_000)


@ai.tool(require_approval=True)
async def talk_to_mothership(question: str) -> ai.SubAgentTool:
    """Contact the mothership for important decisions."""
    mothership = ai.agent()
    messages = [
        ai.system_message(MOTHERSHIP_SYSTEM),
        ai.user_message(question),
    ]
    async with mothership.run(MOTHERSHIP_MODEL, messages) as stream:
        async for event in stream:
            yield event


TOOLS: list[ai.AgentTool] = [get_weather, get_population, talk_to_mothership]


chat_agent = ai.Agent(tools=TOOLS)
