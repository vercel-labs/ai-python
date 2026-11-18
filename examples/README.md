# Examples

Examples are grouped by the main API surface they demonstrate.

- `apps/` contains end-to-end demos.
- `agents/` contains agent loops, tools, hooks, and MCP examples.
- `media/` contains image, video, speech, transcription, multimodal, embedding,
  and reranking examples.
- `models/` contains model streaming, structured output, and provider examples.
- `external/` contains examples that use coding-agent CLIs (Claude Code, Codex,
  fx, pi) where a model goes (experimental; run with
  `run-examples.py --external`).

Provider-specific model examples live under `models/<provider>/`, such as
`models/gateway/`, `models/openai/`, `models/anthropic/`, and `models/typesafe/`.

| Directory | Examples | Guide |
| --- | --- | --- |
| [agents](agents/) | [Basic](agents/basic.py), [custom loop](agents/custom_loop.py), [subagent](agents/subagent.py), [content tool](agents/content_tool.py), [streaming tool](agents/streaming_tool.py) | [Agents](https://ai-python.dev/docs/basics/agents) |
| [agents](agents/) | [Tool approval](agents/tool_approval.py), [custom hook](agents/custom_hook.py), [MCP](agents/mcp_tools.py), [telemetry](agents/telemetry.py) | [Tools](https://ai-python.dev/docs/basics/tools) |
| [media](media/) | [Embeddings](media/embeddings.py), [image edit](media/image_edit.py), [image generation](media/image_generation.py), [inline image](media/inline_image.py), [multimodal input](media/multimodal_input.py), [reranking](media/reranking.py), [speech](media/speech_generation.py), [transcription](media/transcription.py), [video](media/video_generation.py) | [Model operations](https://ai-python.dev/docs/basics/model-operations) |
| [models](models/) | [Connection check](models/check_connection.py), [buffered generation](models/generate.py), [prompt caching](models/prompt_caching.py), [streaming](models/stream.py), [structured output](models/structured_output.py) | [Streaming](https://ai-python.dev/docs/basics/streaming) |
| [models/gateway](models/gateway/) | [Web search](models/gateway/builtin_web_search_gateway.py), [decisions](models/gateway/decisions.py), [Jev Python AST](models/gateway/jev_python_ast.py), [model parameters](models/gateway/model_params.py), [streaming](models/gateway/stream.py) | [Providers](https://ai-python.dev/docs/basics/providers) |
| [models/openai](models/openai/) | [Image decisions](models/openai/decisions_images.py), [explicit client](models/openai/explicit_client.py), [Chat Completions](models/openai/openai_chat_completions.py), [streaming](models/openai/stream.py) | [Providers](https://ai-python.dev/docs/basics/providers) |
| [models/anthropic](models/anthropic/), [models/typesafe](models/typesafe/) | [Anthropic web search](models/anthropic/builtin_web_search.py), [Anthropic streaming](models/anthropic/stream.py), [TypeSafe image decisions](models/typesafe/decisions_images.py) | [Providers](https://ai-python.dev/docs/basics/providers) |

Single-file examples use the root environment: `uv run examples/agents/basic.py`.
Live examples need the API keys named in their source. Follow each app's README
for its services and dependencies. To test an app against this checkout, add
`--with-editable /absolute/path/to/ai-python` to its `uv run` command.

`models/gateway/jev_python_ast.py` uses Jev to build Python code through typed
AST choices, with a live terminal preview and streamed prompt expansion and
review. Set `AI_GATEWAY_API_KEY` and run it with `uv run python
examples/models/gateway/jev_python_ast.py`. Set `MODEL_ID` to override the
expansion/review model.
