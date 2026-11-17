# Gateway tool-choice request contract

`test_tool_choice_contract.py` captures the outgoing HTTP body after `ai.stream`,
the provider and the real Gateway serializer construct it. Only the HTTP
transport is mocked. The tests use a fake credential and reject unexpected
network operations or additional HTTP requests.

## Independent protocol sources

The expected shapes were verified separately against the official Vercel AI SDK
at `3bea041f28e65f371e640d2c70c302776a46387f`:

- [V3 tool choice](https://github.com/vercel/ai/blob/3bea041f28e65f371e640d2c70c302776a46387f/packages/provider/src/language-model/v3/language-model-v3-tool-choice.ts)
- [V4 tool choice](https://github.com/vercel/ai/blob/3bea041f28e65f371e640d2c70c302776a46387f/packages/provider/src/language-model/v4/language-model-v4-tool-choice.ts)

Both require object-shaped choices. Separate call-options definitions make the
field optional and document automatic selection as the default. The omission
control preserves the Python SDK's existing behavior: no `toolChoice` field.

`protocol/sources.json` records immutable URLs, revisions, Git blob IDs, source
hashes and retained excerpt hashes, including the Gateway implementations that
pass call options into HTTP bodies. The integrity test checks the retained
fixtures without network access. Expectations are transcribed independently of
the Python serializer, and negative controls reject all three legacy scalars.

The mocked finish response was also checked separately against the V3 and V4
`language-model-v*-stream-part.ts`, `language-model-v*-finish-reason.ts` and
`language-model-v*-usage.ts` files at that revision. Both support the object
finish reason and nested usage fields used here.

## Coverage

| SDK input | Required outgoing field |
| --- | --- |
| `ToolChoiceMode.AUTO` | `{"type":"auto"}` |
| `ToolChoiceMode.NONE` | `{"type":"none"}` |
| `ToolChoiceMode.REQUIRED` | `{"type":"required"}` |
| `ToolRef("vercel_sandbox_run")` | `{"type":"tool","toolName":"vercel_sandbox_run"}` |
| Omitted | Field absent |

Each request case runs through both V3 and V4. The existing `ToolSelection`
branch is covered with every mode as well. Its sorted `activeTools` output is
preserved. These tests establish the choice-field shape, not Gateway enforcement
of `activeTools`, which neither pinned call-options contract declares.

`tool-routing-context.json` preserves the tools-only context from a Bench
sandbox-tool-routing investigation: four declared tools and no structured-text
output schema. Its SHA-256 is pinned in the test:
`c9777f787c0c70a5bfd19d9f4b150f1960f0586733b7f86d6a1371a6c8bb686a`.
No tools are executed.

## Reproduction and limits

```sh
uv run pytest -q tests/providers/ai_gateway/protocol/test_tool_choice_contract.py
uv run pytest -q tests/providers/ai_gateway
```

On the unfixed `v0.9.0` source (`c0c72eeca22d1399fc8859081fba5d973370899b`),
the regression module produced **12 failures and 20 passes**. Every failure
rejected a scalar mode where the independent contract required an object:
three ordinary modes and three subset modes, each across V3 and V4. Named-tool
and omission controls already passed. After the two serializer corrections,
all 32 cases pass. No assertions were weakened, skipped or marked as expected
failures.

This fixes a confirmed request-format mismatch. Earlier live Bench requests
were reported as HTTP 400 and produced no model responses, but their saved
artifacts lack the detailed provider rejection body. The exact cause of those
live rejections remains unconfirmed. Mocked responses are not evidence of live
Gateway acceptance; this regression suite makes no paid model calls.
