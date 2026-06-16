"""Check credentials and model availability."""

import asyncio
import sys

import ai

MODELS: list[tuple[str, ai.Model]] = [
    ("ai_gateway", ai.get_model("gateway:anthropic/claude-sonnet-4.6")),
    ("anthropic", ai.get_model("anthropic:claude-sonnet-4-6")),
    ("openai", ai.get_model("openai:gpt-5.4-mini")),
]

_failed = False


def _fail(msg: str) -> None:
    global _failed  # noqa: PLW0603
    _failed = True
    print(msg)


async def _check(name: str, model: ai.Model) -> None:
    if not model.provider.is_configured():
        print(f"  [SKIP]  {model.provider.name} provider is not configured")
        return
    try:
        await ai.probe(model)
        print(f"  [OK]    {name}/{model.id}")
    except Exception as exc:
        _fail(f"  [ERR]   {name}/{model.id}: {exc}")


async def _list_models(name: str, model: ai.Model) -> None:
    if not model.provider.is_configured():
        return
    try:
        ids: list[str] = await model.provider.list_models()
        print(f"  {name}: {len(ids)} models (last: {ids[-1]})")
    except Exception as exc:
        _fail(f"  {name}: [ERR] {exc}")


async def main() -> None:
    print("Checking connections...\n")
    for name, model in MODELS:
        await _check(name, model)

    print("\nListing models...\n")
    for name, model in MODELS:
        await _list_models(name, model)

    print()
    if _failed:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
