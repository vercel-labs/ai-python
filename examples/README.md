# Examples

Examples are grouped by the main API surface they demonstrate.

- `apps/` contains end-to-end demos.
- `agents/` contains agent loops, tools, hooks, and MCP examples.
- `media/` contains image, video, speech, transcription, multimodal, embedding,
  and reranking examples.
- `models/` contains model streaming, structured output, and provider examples.

Provider-specific model examples live under `models/<provider>/`, such as
`models/gateway/`, `models/openai/`, and `models/anthropic/`.

`models/gateway/jev_python_ast.py` uses Jev to build Python code through typed
AST choices, with a live terminal preview and streamed prompt expansion and
review. Set `AI_GATEWAY_API_KEY` and run it with `uv run python
examples/models/gateway/jev_python_ast.py`. Set `MODEL_ID` to override the
expansion/review model.
