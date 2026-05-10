"""Check connection and list models — verify credentials and model availability."""

import asyncio
import sys

import ai

PROVIDERS: list[tuple[str, ai.Provider, str]] = [
    ("ai_gateway", ai.ai_gateway, "anthropic/claude-sonnet-4"),
    ("anthropic", ai.anthropic, "claude-sonnet-4-20250514"),
    ("openai", ai.openai, "gpt-5.4-mini"),
]

_failed = False


def _fail(msg: str) -> None:
    global _failed  # noqa: PLW0603
    _failed = True
    print(msg)


async def _check(name: str, provider: ai.Provider, model_id: str) -> None:
    if provider.client().api_key is None:
        print(f"  [SKIP]  {provider.api_key_env} not set")
        return
    model = provider(model_id)
    try:
        ok = await ai.check_connection(model)
        if ok:
            print(f"  [OK]    {name}/{model_id}")
        else:
            _fail(f"  [FAIL]  {name}/{model_id}")
    except Exception as exc:
        _fail(f"  [ERR]   {name}/{model_id}: {exc}")


async def _list_models(name: str, provider: ai.Provider) -> None:
    if provider.client().api_key is None:
        return
    try:
        ids: list[str] = await provider.list()
        print(f"  {name}: {len(ids)} models (last: {ids[-1]})")
    except Exception as exc:
        _fail(f"  {name}: [ERR] {exc}")


async def main() -> None:
    print("Checking connections...\n")
    for name, provider, model_id in PROVIDERS:
        await _check(name, provider, model_id)

    print("\nListing models...\n")
    for name, provider, _ in PROVIDERS:
        await _list_models(name, provider)

    print()
    if _failed:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
