"""Save a worker, and continue its conversation in another process."""

import asyncio
import tempfile

import ai
from ai import workers


async def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        worker = workers.ClaudeCode(cwd=tmp)

        await ai.experimental_generate(
            worker,
            [ai.user_message("Remember this codeword: PLATYPUS. Reply OK.")],
        )

        # The worker is plain data, and it remembers its CLI session.
        saved = worker.model_dump_json()  # put this in your database

        # ... later, another process entirely ...

        again = workers.Worker.model_validate_json(saved)

        # The CLI has the conversation: send what is new. A session block
        # keeps the CLI alive between the calls.
        async with again.session() as session:
            for question in (
                "What codeword did I give you? Reply with the word only.",
                "Spell it backwards. Reply with the word only.",
            ):
                messages = [*session.messages, ai.user_message(question)]
                answer = await ai.experimental_generate(again, messages)
                print(answer.text)


if __name__ == "__main__":
    asyncio.run(main())
