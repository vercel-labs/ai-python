from . import errors, models, providers, util
from .agents import (
    Agent,
    AgentTool,
    Context,
    StreamingStatusTool,
    StreamingTextTool,
    SubAgentTool,
    Tool,
    ToolCall,
    ToolRunner,
    abort_pending_hook,
    agent,
    cancel_hook,
    hook,
    mcp,
    pending_tool_result,
    resolve_hook,
    tool,
    tool_result,
    yield_from,
)
from .errors import AIError, ConfigurationError, UnsupportedProviderError
from .models import (
    Client,
    ImageParams,
    Model,
    Provider,
    Stream,
    VideoParams,
    check_connection,
    generate,
    get_model,
    stream,
)
from .providers import ai_gateway, anthropic, anthropic_like, openai, openai_like
from .types import events, messages, tools
from .types.builders import (
    assistant_message,
    file_part,
    system_message,
    thinking,
    tool_message,
    tool_result_part,
    user_message,
)

__all__ = [
    # Builders (from types/builders)
    "user_message",
    "assistant_message",
    "system_message",
    "tool_message",
    "tool_result",
    "tool_result_part",
    "pending_tool_result",
    "file_part",
    "thinking",
    # Models (from models/)
    "AIError",
    "ConfigurationError",
    "UnsupportedProviderError",
    "Model",
    "Provider",
    "ImageParams",
    "VideoParams",
    "Client",
    "Stream",
    "check_connection",
    "stream",
    "generate",
    "get_model",
    "models",
    "providers",
    # Provider factories
    "openai",
    "openai_like",
    "anthropic",
    "anthropic_like",
    "ai_gateway",
    # Agents — primary API
    "Agent",
    "agent",
    "Context",
    # Agents — tools
    "AgentTool",
    "Tool",
    "ToolCall",
    "ToolRunner",
    "tool",
    "StreamingTextTool",
    "SubAgentTool",
    "StreamingStatusTool",
    # Agents — composition
    "yield_from",
    # Agents — hooks
    "hook",
    "resolve_hook",
    "cancel_hook",
    "abort_pending_hook",
    # Submodules
    "events",
    "errors",
    "messages",
    "mcp",
    "tools",
    "util",
]
