"""Direct control: send, steer, read the feed, interrupt."""

import asyncio

import ai


async def main() -> None:
    async with ai.external.Codex() as harness:
        await harness.session.send(
            ai.user_message("Count from 1 to 20 slowly, one number per line.")
        )

        steered = False
        async for event in harness.session.events():
            match event:
                case ai.events.TextDelta():
                    print(event.chunk, end="", flush=True)
                    if not steered and "5" in event.chunk:
                        steered = True
                        # A steer: the CLI decides how to fit it in.
                        await harness.session.send(
                            ai.user_message("Stop at 8 instead.")
                        )
                case ai.external.MessageEvent(message=message):
                    print(f"\n[{message.role}] {message.text[:40]!r}")
                case ai.events.StreamEnd(finish_reason="stop"):
                    break

        await harness.session.send(
            ai.user_message("Now count from 100 down to 1, slowly.")
        )
        await asyncio.sleep(3)
        await harness.session.interrupt()  # stop the turn; the session stays
        # The cut reply is in the conversation, with the reason in its data.
        print(f"\n{harness.session.messages[-1].provider_metadata}")


asyncio.run(main())
