# Tool-choice protocol evidence

These fixtures preserve independently retrieved primary protocol definitions;
they are not generated from the Python SDK under test. `sources.json` records
immutable commit URLs, Git blob IDs, source-file SHA-256s, excerpt line numbers,
and fixture SHA-256s. Tests read local fixtures and perform no source fetches.

V3 and V4 were inspected separately at Vercel AI SDK commit
`3bea041f28e65f371e640d2c70c302776a46387f`. Both tool-choice definitions require
objects for `auto`, `none`, `required`, and a named tool. Their separate
`CallOptions` definitions make `toolChoice` optional and document the default as
`auto`. An omitted Python SDK setting should therefore remain omitted from the
outgoing request; these fixtures do not require inserting an explicit default.

The JSON Schemas are manual transcriptions of those two TypeScript unions.
They constrain the types and required keys only. They deliberately impose no
extra-property restriction or tool-name length requirement absent from the
definitions. They validate this field, not the entire Gateway request.

Gateway request-construction excerpts show the call options passing through to
the JSON request body. V3 uses the published `@ai-sdk/gateway@3.0.0` revision
`96df007cee5174f26b206c7c3ec8c8d72ee41847`; V4 uses the main revision above.
The separate protocol-version excerpts identify which version each implements.
This is source-contract evidence, not a capture of the deployed Gateway server.

The SDK's tool-subset branch is exercised only to check its emitted tool-choice
mode against these same unions. These sources do not define `activeTools` as a
Gateway CallOptions field, so the tests must not claim to establish full
Gateway support for that additional field or add that capability to Bench.

The upstream excerpts retain their original content. Attribution and licensing
are preserved in `LICENSE.vercel-ai`. No credentials or runtime environment
files are included.
