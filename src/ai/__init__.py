from . import middleware, models
from .agents import (
    TOOL_APPROVAL_HOOK_TYPE,
    Agent,
    Context,
    Tool,
    ToolApproval,
    ToolCall,
    agent,
    cancel_hook,
    hook,
    mcp,
    resolve_hook,
    tool,
    yield_from,
)
from .middleware import AgentRunContext, Middleware
from .models import (
    Client,
    ImageParams,
    Model,
    Provider,
    StreamResult,
    VideoParams,
    ai_gateway,
    anthropic,
    check_connection,
    generate,
    openai,
    stream,
)

# Re-export core types
from .types import (
    FilePart,
    HookPart,
    Message,
    Part,
    ReasoningPart,
    StreamResultLike,
    StructuredOutputPart,
    TextPart,
    ToolCallPart,
    ToolLike,
    ToolResultPart,
    ToolSchema,
    Usage,
)
from .types.builders import (
    assistant_message,
    file_part,
    system_message,
    thinking,
    tool_message,
    tool_result,
    user_message,
)

__all__ = [
    # Types (from types/)
    "Message",
    "Part",
    "TextPart",
    "ToolCallPart",
    "ToolResultPart",
    "ReasoningPart",
    "FilePart",
    "HookPart",
    "StructuredOutputPart",
    "ToolLike",
    "ToolSchema",
    "Usage",
    # Builders (from types/builders)
    "user_message",
    "assistant_message",
    "system_message",
    "tool_message",
    "tool_result",
    "file_part",
    "thinking",
    # Models (from models/)
    "Model",
    "Provider",
    "ImageParams",
    "VideoParams",
    "Client",
    "StreamResult",
    "StreamResultLike",
    "check_connection",
    "stream",
    "generate",
    "models",
    # Provider factories
    "openai",
    "anthropic",
    "ai_gateway",
    # Agents — primary API
    "Agent",
    "agent",
    "Context",
    # Agents — tools
    "Tool",
    "ToolCall",
    "tool",
    # Agents — composition
    "yield_from",
    # Agents — hooks
    "hook",
    "resolve_hook",
    "cancel_hook",
    "ToolApproval",
    "TOOL_APPROVAL_HOOK_TYPE",
    # Middleware
    "AgentRunContext",
    "Middleware",
    "middleware",
    # Submodules
    "mcp",
]
