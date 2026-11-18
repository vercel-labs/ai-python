"""Save a session id, and continue the conversation in a new process."""

import asyncio
import tempfile

import ai


async def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        async with ai.external.ClaudeCode(cwd=tmp) as harness:
            await ai.generate(
                harness,
                [
                    ai.user_message(
                        "Remember this codeword: PLATYPUS. Reply OK."
                    )
                ],
            )

        # The session id names the conversation in the CLI. Put it in your
        # database.
        session_id = harness.session.id

        # ... later, another process entirely ...

        # A new external harness with the same settings continues the
        # session. The CLI has the conversation: send what is new. The block
        # keeps the CLI alive between the calls.
        async with ai.external.ClaudeCode(cwd=tmp, resume=session_id) as again:
            for question in (
                "What codeword did I give you? Reply with the word only.",
                "Spell it backwards. Reply with the word only.",
            ):
                messages = [
                    *again.session.messages,
                    ai.user_message(question),
                ]
                answer = await ai.generate(again, messages)
                print(answer.text)


if __name__ == "__main__":
    asyncio.run(main())
