"""Custom types for the LLM utility library.

Defines message types, content types, tool types, and output types
without depending on inspect_ai.
"""

import json
from dataclasses import dataclass
from typing import Any, Literal, Protocol, TypedDict

from log_util import get_logger
from openai.types.chat.chat_completion_token_logprob import TopLogprob
from pydantic import BaseModel, Field

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Content types (discriminated union on `type`)
# ---------------------------------------------------------------------------


class ContentText(BaseModel):
    type: Literal["text"] = "text"
    text: str


class ContentReasoning(BaseModel):
    type: Literal["reasoning"] = "reasoning"
    reasoning: str
    signature: str | None = None


class ContentRedactedReasoning(BaseModel):
    type: Literal["redacted_reasoning"] = "redacted_reasoning"
    data: str = ""


Content = ContentText | ContentReasoning | ContentRedactedReasoning

# ---------------------------------------------------------------------------
# Tool types
# ---------------------------------------------------------------------------


@dataclass
class ToolCall:
    """A tool/function call made by the model."""

    id: str
    function: str
    arguments: dict[str, Any]
    type: Literal["function"] | None = None
    parse_error: str | None = None


class ToolParam(BaseModel):
    """A single parameter for a tool function."""

    name: str
    description: str
    input_schema: dict[str, Any]


class ToolParams(BaseModel):
    """Description of tool parameters object in JSON Schema format."""

    type: Literal["object"] = "object"
    properties: dict[str, ToolParam] = Field(default_factory=dict)
    required: list[str] = Field(default_factory=list)
    additionalProperties: bool = False


class ToolInfo(BaseModel):
    """Specification of a tool (JSON Schema compatible).

    Dumping ``parameters`` with ``model_dump(exclude_none=True)`` produces a
    valid JSON Schema ``"parameters"`` block that can be passed directly to
    provider SDKs.
    """

    name: str
    description: str
    parameters: ToolParams = Field(default_factory=ToolParams)


# ---------------------------------------------------------------------------
# Message types (discriminated union on `role`)
# ---------------------------------------------------------------------------


class SystemMessage(BaseModel):
    role: Literal["system"] = "system"
    content: str | list[Content] = ""

    @property
    def text(self) -> str:
        if isinstance(self.content, str):
            return self.content
        return "".join(c.text if isinstance(c, ContentText) else c.reasoning for c in self.content)


class UserMessage(BaseModel):
    role: Literal["user"] = "user"
    content: str | list[Content] = ""

    @property
    def text(self) -> str:
        if isinstance(self.content, str):
            return self.content
        return "".join(c.text if isinstance(c, ContentText) else c.reasoning for c in self.content)


class AssistantMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: str | list[Content] = ""
    tool_calls: list[ToolCall] | None = None

    @property
    def text(self) -> str:
        if isinstance(self.content, str):
            return self.content
        return "".join(c.text if isinstance(c, ContentText) else c.reasoning for c in self.content)


class ToolMessage(BaseModel):
    role: Literal["tool"] = "tool"
    content: str | list[Content] = ""
    tool_call_id: str | None = None
    function: str | None = None
    error: str | None = None

    @property
    def text(self) -> str:
        if isinstance(self.content, str):
            return self.content
        return "".join(c.text if isinstance(c, ContentText) else c.reasoning for c in self.content)


ChatMessage = SystemMessage | UserMessage | AssistantMessage | ToolMessage

# ---------------------------------------------------------------------------
# Response format for structured output
# ---------------------------------------------------------------------------


class ResponseFormat(BaseModel):
    """Structured output specification.

    ``schema_`` is serialized as ``"schema"`` in JSON (to avoid shadowing
    the built-in).
    """

    name: str
    schema_: dict[str, Any] = Field(alias="schema")
    strict: bool = True

    model_config = {"populate_by_name": True}


# ---------------------------------------------------------------------------
# API keys and model params
# ---------------------------------------------------------------------------


class LLMApiKeys(TypedDict, total=False):
    openai_key: str | None
    anthropic_key: str | None
    google_key: str | None
    openrouter_key: str | None


class ModelCallParams(BaseModel):
    provider: str
    model_name: str
    reasoning_effort: Literal["low", "medium", "high", "max"] | None = None


# ---------------------------------------------------------------------------
# Streaming partials
# ---------------------------------------------------------------------------


@dataclass
class ToolCallPartial:
    id: str | None
    function: str | None
    arguments_raw: str | None
    type: Literal["function"]


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def parse_chat_message(message_data: dict[str, Any]) -> ChatMessage:
    role = message_data.get("role")
    if role == "system":
        return SystemMessage.model_validate(message_data)
    elif role == "user":
        return UserMessage.model_validate(message_data)
    elif role == "assistant":
        return AssistantMessage.model_validate(message_data)
    elif role == "tool":
        return ToolMessage.model_validate(message_data)
    else:
        raise ValueError(f"Unknown message role: {role}")


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class CompletionTooLongException(Exception):
    pass


class RateLimitException(Exception):
    pass


class ContextWindowException(Exception):
    pass


class NoResponseException(Exception):
    pass


# ---------------------------------------------------------------------------
# Finish reason
# ---------------------------------------------------------------------------

FinishReasonType = Literal[
    "error",
    "stop",
    "length",
    "tool_calls",
    "content_filter",
    "function_call",
    "streaming",
]

# ---------------------------------------------------------------------------
# Usage
# ---------------------------------------------------------------------------


class Usage(BaseModel):
    cache_read_input_tokens: int | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None

    def divide(self, factor: float) -> "Usage":
        return Usage(
            cache_read_input_tokens=(
                int(self.cache_read_input_tokens / factor)
                if self.cache_read_input_tokens is not None
                else None
            ),
            input_tokens=(
                int(self.input_tokens / factor) if self.input_tokens is not None else None
            ),
            output_tokens=(
                int(self.output_tokens / factor) if self.output_tokens is not None else None
            ),
        )

    def __add__(self, other: "Usage") -> "Usage":
        """Add two Usage objects together, combining their token counts."""
        return Usage(
            cache_read_input_tokens=(
                (self.cache_read_input_tokens or 0) + (other.cache_read_input_tokens or 0)
                if self.cache_read_input_tokens is not None
                or other.cache_read_input_tokens is not None
                else None
            ),
            input_tokens=(
                (self.input_tokens or 0) + (other.input_tokens or 0)
                if self.input_tokens is not None or other.input_tokens is not None
                else None
            ),
            output_tokens=(
                (self.output_tokens or 0) + (other.output_tokens or 0)
                if self.output_tokens is not None or other.output_tokens is not None
                else None
            ),
        )

    def __radd__(self, other) -> "Usage":
        """Support sum() operation by handling addition with 0."""
        if other == 0:
            return self
        return self.__add__(other)

    def compute_cost(self, cache_cost: float, input_cost: float, output_cost: float) -> float:
        """All costs are in dollars per million tokens."""
        cache_tokens = self.cache_read_input_tokens or 0
        input_tokens = self.input_tokens or 0
        output_tokens = self.output_tokens or 0
        return (
            cache_tokens * cache_cost
            + (input_tokens - cache_tokens) * input_cost
            + output_tokens * output_cost
        ) / 1e6


# ---------------------------------------------------------------------------
# LLM output types
# ---------------------------------------------------------------------------


class LLMCompletion(BaseModel):
    text: str | None = None
    tool_calls: list[ToolCall] | None = None
    finish_reason: FinishReasonType | None = None
    top_logprobs: list[list[TopLogprob]] | None = None
    """List of the top logprobs for each token in the completion text."""
    reasoning: ContentReasoning | None = None
    """Captured reasoning/thinking block text (for extended thinking models)."""
    redacted_reasoning: ContentRedactedReasoning | None = None
    """Captured redacted reasoning/thinking block text (for extended thinking models)."""

    @property
    def no_text(self) -> bool:
        return self.text is None or len(self.text) == 0


class LLMOutput(BaseModel):
    model: str
    completions: list[LLMCompletion]
    errors: (
        list[Literal["rate_limit", "no_response", "other", "all_providers_exhausted"]] | None
    ) = None
    usage: Usage | None = None
    from_cache: bool = False
    duration: float | None = None

    @property
    def non_empty(self) -> bool:
        return len(self.completions) > 0

    @property
    def first(self) -> LLMCompletion | None:
        return self.completions[0] if self.non_empty else None

    @property
    def first_text(self) -> str | None:
        return self.first.text if self.first else None

    @property
    def did_error(self) -> bool:
        return bool(self.errors)


class LLMCompletionPartial(LLMCompletion):
    tool_calls: list[ToolCallPartial | None] | None = None  # type: ignore
    reasoning: ContentReasoning | None = None


class LLMOutputPartial(LLMOutput):
    completions: list[LLMCompletionPartial]  # type: ignore


def finalize_llm_output_partial(partial: LLMOutputPartial) -> LLMOutput:
    def _parse_tool_call(tc_partial: ToolCallPartial):
        if tc_partial.id is None:
            raise ValueError("Tool call ID not found in partial; check for parsing errors")
        if tc_partial.function is None:
            raise ValueError("Tool call function not found in partial; check for parsing errors")

        # Attempt to load arguments into JSON
        try:
            arguments: dict[str, Any] = json.loads(tc_partial.arguments_raw or "{}")
            parse_error = None
        # If the tool call arguments are not valid JSON, return an empty dict with the error
        except Exception as e:
            arguments = {}
            parse_error = f"Couldn't parse tool call arguments as JSON: {e}. Original input: {tc_partial.arguments_raw}"

        return ToolCall(
            id=tc_partial.id,
            function=tc_partial.function,
            arguments=arguments,
            parse_error=parse_error,
            type=tc_partial.type,
        )

    output = LLMOutput(
        model=partial.model,
        completions=[
            LLMCompletion(
                text=c.text,
                tool_calls=[_parse_tool_call(tc) for tc in (c.tool_calls or []) if tc is not None],
                finish_reason=c.finish_reason,
                reasoning=c.reasoning,
                redacted_reasoning=c.redacted_reasoning,
            )
            for c in partial.completions
        ],
    )

    # If the completion is empty and was truncated (likely due to too much reasoning), raise
    if output.first and output.first.finish_reason == "length" and output.first.no_text:
        raise CompletionTooLongException(
            "Completion empty due to truncation. Consider increasing max_new_tokens."
        )
    for c in output.completions:
        if c.finish_reason == "length":
            logger.warning(
                "Completion truncated due to length; consider increasing max_new_tokens."
            )

    return output


# ---------------------------------------------------------------------------
# Callback protocols
# ---------------------------------------------------------------------------


class AsyncStreamingCallback(Protocol):
    async def __call__(
        self,
        batch_index: int,
        llm_output: LLMOutput,
    ) -> None: ...


class AsyncSingleStreamingCallback(Protocol):
    async def __call__(
        self,
        llm_output: LLMOutput,
    ) -> None: ...


def get_single_streaming_callback(
    batch_index: int,
    streaming_callback: AsyncStreamingCallback,
) -> AsyncSingleStreamingCallback:
    async def single_streaming_callback(llm_output: LLMOutput):
        await streaming_callback(batch_index, llm_output)

    return single_streaming_callback
