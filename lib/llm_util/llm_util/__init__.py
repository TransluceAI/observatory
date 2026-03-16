"""LLM utility library — custom types, no inspect_ai dependency."""

from llm_util.prod_llms import complete_async, get_llm_completions_async
from llm_util.types import (
    AssistantMessage,
    AsyncSingleStreamingCallback,
    AsyncStreamingCallback,
    ChatMessage,
    CompletionTooLongException,
    Content,
    ContentReasoning,
    ContentRedactedReasoning,
    ContentText,
    ContextWindowException,
    FinishReasonType,
    LLMApiKeys,
    LLMCompletion,
    LLMCompletionPartial,
    LLMOutput,
    LLMOutputPartial,
    ModelCallParams,
    NoResponseException,
    RateLimitException,
    ResponseFormat,
    SystemMessage,
    ToolCall,
    ToolCallPartial,
    ToolInfo,
    ToolMessage,
    ToolParam,
    ToolParams,
    Usage,
    UserMessage,
    finalize_llm_output_partial,
    get_single_streaming_callback,
    parse_chat_message,
)

__all__ = [
    # Message types
    "SystemMessage",
    "UserMessage",
    "AssistantMessage",
    "ToolMessage",
    "ChatMessage",
    # Content types
    "Content",
    "ContentText",
    "ContentReasoning",
    "ContentRedactedReasoning",
    # Tool types
    "ToolCall",
    "ToolCallPartial",
    "ToolParam",
    "ToolParams",
    "ToolInfo",
    # Output types
    "LLMCompletion",
    "LLMCompletionPartial",
    "LLMOutput",
    "LLMOutputPartial",
    "Usage",
    "FinishReasonType",
    "ResponseFormat",
    # Params & keys
    "ModelCallParams",
    "LLMApiKeys",
    # Exceptions
    "CompletionTooLongException",
    "RateLimitException",
    "ContextWindowException",
    "NoResponseException",
    # Callbacks
    "AsyncStreamingCallback",
    "AsyncSingleStreamingCallback",
    # Functions
    "get_llm_completions_async",
    "complete_async",
    "parse_chat_message",
    "finalize_llm_output_partial",
    "get_single_streaming_callback",
]
