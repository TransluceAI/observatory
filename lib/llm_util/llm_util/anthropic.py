import asyncio
import contextlib
from typing import Any, Literal, cast

import backoff
from anthropic import (
    Anthropic,
    AsyncAnthropic,
    AuthenticationError,
    BadRequestError,
    RateLimitError,
)
from anthropic._types import NOT_GIVEN
from anthropic.types import (
    Message,
    MessageParam,
    RawContentBlockDeltaEvent,
    RawMessageDeltaEvent,
    RawMessageStartEvent,
    RawMessageStreamEvent,
    RedactedThinkingBlockParam,
    SignatureDelta,
    TextBlockParam,
    TextDelta,
    ThinkingBlockParam,
    ThinkingDelta,
    ToolChoiceAnyParam,
    ToolChoiceAutoParam,
    ToolChoiceParam,
    ToolParam,
    ToolResultBlockParam,
    ToolUseBlockParam,
)
from backoff.types import Details
from llm_util.types import (
    AsyncSingleStreamingCallback,
    ChatMessage,
    CompletionTooLongException,
    Content,
    ContentReasoning,
    ContentRedactedReasoning,
    ContentText,
    FinishReasonType,
    LLMCompletion,
    LLMCompletionPartial,
    LLMOutput,
    LLMOutputPartial,
    RateLimitException,
    ResponseFormat,
    ToolCall,
    ToolInfo,
    Usage,
    finalize_llm_output_partial,
)
from log_util import get_logger

logger = get_logger(__name__)


def _print_backoff_message(e: Details):
    logger.warning(
        f"Anthropic backing off for {e['wait']:.2f}s due to {e['exception'].__class__.__name__}"  # type: ignore
    )


def _parse_message_content(content: str | list[Content]) -> str | list[TextBlockParam]:
    if isinstance(content, str):
        return content
    else:
        result: list[TextBlockParam] = []
        for sub_content in content:
            if sub_content.type == "text":
                result.append(TextBlockParam(text=sub_content.text, type="text"))
            else:
                raise ValueError(f"Unsupported content type: {sub_content.type}")
        return result


def _parse_chat_messages(messages: list[ChatMessage]) -> tuple[str | None, list[MessageParam]]:
    result: list[MessageParam] = []
    system_prompt: str | None = None

    # Separate system messages from non-system messages
    non_system_messages = []
    for message in messages:
        if message.role == "system":
            system_prompt = message.text
        else:
            non_system_messages.append(message)

    for message in non_system_messages:
        if message.role == "user":
            content = _parse_message_content(message.content)
            result.append(
                MessageParam(
                    role=message.role,
                    content=content,
                )
            )
        elif message.role == "assistant":
            # Deal with all non-tool content (thinking, redacted thinking, normal text) first
            # Thinking blocks must be included in assistant messages for
            # multi-turn extended thinking to work correctly.
            all_content: list[
                ThinkingBlockParam | RedactedThinkingBlockParam | TextBlockParam | ToolUseBlockParam
            ] = []
            if isinstance(message.content, list):
                for item in message.content:
                    if isinstance(item, ContentReasoning):
                        assert item.signature is not None
                        all_content.append(
                            ThinkingBlockParam(
                                type="thinking",
                                thinking=item.reasoning,
                                signature=item.signature,
                            )
                        )
                    elif isinstance(item, ContentRedactedReasoning):
                        all_content.append(
                            RedactedThinkingBlockParam(type="redacted_thinking", data=item.data)
                        )
                    elif isinstance(item, ContentText):
                        all_content.append(TextBlockParam(text=item.text, type="text"))
                    else:
                        raise ValueError(f"Unsupported content type: {type(item)}")
            else:
                all_content = [TextBlockParam(text=message.content, type="text")]

            # Now add tool calls
            for tool_call in message.tool_calls or []:
                all_content.append(
                    ToolUseBlockParam(
                        id=tool_call.id,
                        input=tool_call.arguments,
                        name=tool_call.function,
                        type="tool_use",
                    )
                )
            result.append(
                MessageParam(
                    role="assistant",
                    content=all_content,
                )
            )
        elif message.role == "tool":
            tool_result_block = ToolResultBlockParam(
                tool_use_id=str(message.tool_call_id),
                type="tool_result",
                content=_parse_message_content(message.content),
                is_error=message.error is not None,
            )
            # Merge consecutive tool results into one user message (Anthropic
            # requires alternating user/assistant roles, so multiple tool
            # results from the same assistant turn must share one user message).
            if result and result[-1]["role"] == "user" and isinstance(result[-1]["content"], list):
                last_content = result[-1]["content"]
                if (
                    last_content
                    and isinstance(last_content[0], dict)
                    and last_content[0].get("type") == "tool_result"
                ):
                    last_content.append(tool_result_block)
                    continue
            result.append(
                MessageParam(
                    role="user",
                    content=[tool_result_block],
                )
            )
        else:
            raise ValueError(f"Unknown message role: {message.role}")

    return system_prompt, result


def _parse_tools(tools: list[ToolInfo]) -> list[ToolParam]:
    return [
        ToolParam(
            name=tool.name,
            description=tool.description,
            input_schema=tool.parameters.model_dump(exclude_none=True),
        )
        for tool in tools
    ]


def _parse_tool_choice(tool_choice: Literal["auto", "required"] | None) -> ToolChoiceParam | None:
    if tool_choice is None:
        return None
    elif tool_choice == "auto":
        return ToolChoiceAutoParam(type="auto")
    elif tool_choice == "required":
        return ToolChoiceAnyParam(type="any")


ADAPTIVE_THINKING_MODELS = {"claude-opus-4-6", "claude-sonnet-4-6"}
MIN_THINKING_BUDGET = 1024
THINKING_BUDGET_BY_EFFORT: dict[str, int] = {
    "max": 16384,
    "high": 8192,
    "medium": 4096,
    "low": 1024,
}


def _build_thinking_args(
    model_name: str,
    reasoning_effort: Literal["low", "medium", "high", "max"] | None,
    max_new_tokens: int,
) -> dict[str, Any]:
    """Build the thinking/effort kwargs for the Anthropic messages.create call."""
    if not reasoning_effort:
        return {"thinking": NOT_GIVEN}

    if model_name in ADAPTIVE_THINKING_MODELS:
        if reasoning_effort == "max" and model_name != "claude-opus-4-6":
            raise ValueError(
                f"reasoning_effort='max' is only supported on claude-opus-4-6, " f"not {model_name}"
            )
        return {
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": reasoning_effort},
        }

    # Non-4.6 models: use manual thinking with budget_tokens.
    # "max" isn't a valid effort on these models, so treat it like "high".
    effective_effort = "high" if reasoning_effort == "max" else reasoning_effort
    budget = min(THINKING_BUDGET_BY_EFFORT[effective_effort], max_new_tokens)

    if budget < MIN_THINKING_BUDGET:
        logger.warning(
            f"max_new_tokens={max_new_tokens} is too small for extended thinking "
            f"(minimum budget is {MIN_THINKING_BUDGET}). Disabling thinking."
        )
        return {"thinking": NOT_GIVEN}

    return {
        "thinking": {
            "type": "enabled",
            "budget_tokens": budget,
        }
    }


@backoff.on_exception(
    backoff.expo,
    exception=(Exception),
    giveup=lambda e: isinstance(e, BadRequestError)
    or isinstance(e, RateLimitException)
    or isinstance(e, AuthenticationError)
    or isinstance(e, NotImplementedError),
    max_tries=3,
    factor=2.0,
    on_backoff=_print_backoff_message,
)
async def get_anthropic_chat_completion_streaming_async(
    client: AsyncAnthropic,
    streaming_callback: AsyncSingleStreamingCallback | None,
    messages: list[ChatMessage],
    model_name: str,
    tools: list[ToolInfo] | None = None,
    tool_choice: Literal["auto", "required"] | None = None,
    max_new_tokens: int = 32,
    temperature: float = 1.0,
    reasoning_effort: Literal["low", "medium", "high", "max"] | None = None,
    logprobs: bool = False,
    top_logprobs: int | None = None,
    timeout: float = 5.0,
    response_format: ResponseFormat | None = None,
):
    if logprobs or top_logprobs is not None:
        raise NotImplementedError(
            "We have not implemented logprobs or top_logprobs for Anthropic yet."
        )
    if response_format is not None:
        logger.warning("response_format is not yet supported for Anthropic streaming; ignoring.")

    system, input_messages = _parse_chat_messages(messages)
    input_tools = _parse_tools(tools) if tools else NOT_GIVEN
    thinking_args = _build_thinking_args(model_name, reasoning_effort, max_new_tokens)
    try:
        async with asyncio.timeout(timeout) if timeout else contextlib.nullcontext():  # type: ignore
            stream = await client.messages.create(
                model=model_name,
                messages=input_messages,
                tools=input_tools,
                tool_choice=_parse_tool_choice(tool_choice) or NOT_GIVEN,
                max_tokens=max_new_tokens,
                temperature=temperature,
                system=system if system is not None else NOT_GIVEN,
                stream=True,
                cache_control={"type": "ephemeral"},
                **thinking_args,
            )

            llm_output_partial = None
            async for chunk in stream:
                llm_output_partial = update_llm_output(llm_output_partial, chunk)
                if streaming_callback:
                    await streaming_callback(finalize_llm_output_partial(llm_output_partial))

            # Fully parse the partial output
            if llm_output_partial:
                return finalize_llm_output_partial(llm_output_partial)
            else:
                # Streaming did not produce anything
                return LLMOutput(model=model_name, completions=[], errors=["no_response"])
    except RateLimitError as e:
        # Raise a custom type that prod_llms can catch
        raise RateLimitException(e) from e


FINISH_REASON_MAP: dict[str, FinishReasonType] = {
    "end_turn": "stop",
    "max_tokens": "length",
    "stop_sequence": "stop",
    "tool_use": "tool_calls",
}


def update_llm_output(
    llm_output_partial: LLMOutputPartial | None,
    chunk: RawMessageStreamEvent,
):
    """
    Note that Anthropic only allows one message to be streamed at a time.
    Thus there can only be one completion.
    """

    if llm_output_partial is not None:
        cur_text: str | None = llm_output_partial.completions[0].text
        cur_finish_reason: FinishReasonType | None = llm_output_partial.completions[0].finish_reason
        cur_model = llm_output_partial.model
        prev_reasoning = llm_output_partial.completions[0].reasoning
        cur_reasoning_text: str | None = prev_reasoning.reasoning if prev_reasoning else None
        cur_reasoning_sig: str | None = prev_reasoning.signature if prev_reasoning else None
    else:
        cur_text, cur_finish_reason, cur_model = None, None, None
        cur_reasoning_text, cur_reasoning_sig = None, None

    if isinstance(chunk, RawMessageStartEvent):
        cur_model = chunk.message.model
    elif isinstance(chunk, RawContentBlockDeltaEvent):
        if isinstance(chunk.delta, TextDelta):
            cur_text = (cur_text or "") + chunk.delta.text
        elif isinstance(chunk.delta, ThinkingDelta):
            cur_reasoning_text = (cur_reasoning_text or "") + chunk.delta.thinking
        elif isinstance(chunk.delta, SignatureDelta):
            cur_reasoning_sig = (cur_reasoning_sig or "") + chunk.delta.signature
        else:
            raise ValueError(f"Unsupported delta type: {type(chunk.delta)}")
    elif isinstance(chunk, RawMessageDeltaEvent):
        if stop_reason := chunk.delta.stop_reason:
            cur_finish_reason = FINISH_REASON_MAP.get(stop_reason)

    reasoning = (
        ContentReasoning(reasoning=cur_reasoning_text or "", signature=cur_reasoning_sig)
        if cur_reasoning_text is not None
        else None
    )

    completions: list[LLMCompletionPartial] = []
    completions.append(
        LLMCompletionPartial(
            text=cur_text,
            finish_reason=cur_finish_reason,
            reasoning=reasoning,
        )
    )

    assert cur_model is not None, "First chunk should always set the cur_model"
    return LLMOutputPartial(completions=completions, model=cur_model)


@backoff.on_exception(
    backoff.expo,
    exception=(Exception),
    giveup=lambda e: isinstance(e, BadRequestError)
    or isinstance(e, RateLimitException)
    or isinstance(e, AuthenticationError),
    max_tries=3,
    factor=2.0,
    on_backoff=_print_backoff_message,
)
async def get_anthropic_chat_completion_async(
    client: AsyncAnthropic,
    messages: list[ChatMessage],
    model_name: str,
    tools: list[ToolInfo] | None = None,
    tool_choice: Literal["auto", "required"] | None = None,
    max_new_tokens: int = 32,
    temperature: float = 1.0,
    reasoning_effort: Literal["low", "medium", "high", "max"] | None = None,
    logprobs: bool = False,
    top_logprobs: int | None = None,
    timeout: float = 5.0,
    response_format: ResponseFormat | None = None,
) -> LLMOutput:
    if logprobs or top_logprobs is not None:
        raise NotImplementedError(
            "We have not implemented logprobs or top_logprobs for Anthropic yet."
        )
    if response_format is not None:
        logger.warning("response_format is not yet supported for Anthropic; ignoring.")

    system, input_messages = _parse_chat_messages(messages)
    input_tools = _parse_tools(tools) if tools else NOT_GIVEN
    thinking_args = _build_thinking_args(model_name, reasoning_effort, max_new_tokens)

    try:
        async with asyncio.timeout(timeout) if timeout else contextlib.nullcontext():  # type: ignore
            raw_output = await client.messages.create(
                model=model_name,
                messages=input_messages,
                tools=input_tools,
                tool_choice=_parse_tool_choice(tool_choice) or NOT_GIVEN,
                max_tokens=max_new_tokens,
                temperature=temperature,
                system=system if system is not None else NOT_GIVEN,
                cache_control={"type": "ephemeral"},
                **thinking_args,
            )

            output = parse_anthropic_completion(raw_output, model_name)
            if output.first and output.first.finish_reason == "length" and output.first.no_text:
                raise CompletionTooLongException(
                    "Completion empty due to truncation. Consider increasing max_new_tokens."
                )

            return output
    except RateLimitError as e:
        # Raise a custom type that prod_llms can catch
        raise RateLimitException(e) from e


def get_anthropic_client_sync(api_key: str | None = None) -> Anthropic:
    if api_key:
        return Anthropic(api_key=api_key)
    return Anthropic()


def get_anthropic_client_async(api_key: str | None = None) -> AsyncAnthropic:
    if api_key:
        return AsyncAnthropic(api_key=api_key)
    return AsyncAnthropic()


def parse_anthropic_completion(message: Message | None, model: str) -> LLMOutput:
    if message is None:
        return LLMOutput(
            model=model,
            completions=[],
            errors=["no_response"],
        )

    if message.stop_reason == "end_turn":
        finish_reason = "stop"
    elif message.stop_reason == "max_tokens":
        finish_reason = "length"
    elif message.stop_reason == "stop_sequence":
        finish_reason = "stop"
    elif message.stop_reason == "tool_use":
        finish_reason = "tool_calls"
    else:
        finish_reason = "error"

    text = None
    tool_calls: list[ToolCall] = []
    reasoning = None
    redacted_reasoning = None
    for block in message.content:
        if block.type == "text":
            if text is not None:
                raise ValueError(
                    "Anthropic API returned multiple text blocks; this was unexpected."
                )
            text = block.text
        elif block.type == "tool_use":
            tool_calls.append(
                ToolCall(
                    id=block.id,
                    function=block.name,
                    arguments=cast(dict[str, Any], block.input),
                    type="function",
                )
            )
        elif block.type == "thinking":
            reasoning = ContentReasoning(reasoning=block.thinking, signature=block.signature)
        elif block.type == "redacted_thinking":
            redacted_reasoning = ContentRedactedReasoning(data=block.data)
        else:
            raise ValueError(f"Unknown block type: {block.type}")

        # Check that we have only one of reasoning or redacted_reasoning
        if reasoning is not None and redacted_reasoning is not None:
            raise ValueError("Only one of reasoning or redacted_reasoning can be present")

    input_tokens = message.usage.input_tokens
    if message.usage.cache_creation_input_tokens:
        input_tokens += message.usage.cache_creation_input_tokens

    return LLMOutput(
        model=model,
        completions=[
            LLMCompletion(
                text=text,
                tool_calls=tool_calls,
                finish_reason=finish_reason,
                reasoning=reasoning,
                redacted_reasoning=redacted_reasoning,
            )
        ],
        usage=Usage(
            cache_read_input_tokens=message.usage.cache_read_input_tokens,
            input_tokens=input_tokens,
            output_tokens=message.usage.output_tokens,
        ),
    )


async def is_anthropic_api_key_valid(api_key: str) -> bool:
    """Test whether an Anthropic API key is valid or invalid."""
    client = AsyncAnthropic(api_key=api_key)

    try:
        await client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=1,
            messages=[{"role": "user", "content": "hi"}],
        )
        return True
    except AuthenticationError:
        return False
    except Exception:
        return True
