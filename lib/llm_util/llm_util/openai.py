import asyncio
import contextlib
import json
from collections.abc import Mapping
from typing import Any, Literal, cast

import backoff
import tiktoken
from backoff.types import Details
from llm_util.types import (
    AsyncSingleStreamingCallback,
    ChatMessage,
    CompletionTooLongException,
    Content,
    FinishReasonType,
    LLMCompletion,
    LLMCompletionPartial,
    LLMOutput,
    LLMOutputPartial,
    RateLimitException,
    ResponseFormat,
    ToolCall,
    ToolCallPartial,
    ToolInfo,
    Usage,
    finalize_llm_output_partial,
)
from log_util import get_logger
from openai import AsyncOpenAI, AuthenticationError, BadRequestError, OpenAI, RateLimitError
from openai._types import NOT_GIVEN
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionAssistantMessageParam,
    ChatCompletionChunk,
    ChatCompletionContentPartTextParam,
    ChatCompletionMessageParam,
    ChatCompletionMessageToolCall,
    ChatCompletionMessageToolCallParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionToolMessageParam,
    ChatCompletionToolParam,
    ChatCompletionUserMessageParam,
)
from openai.types.chat.chat_completion_message_tool_call_param import (
    Function as OpenAIFunctionParam,
)
from openai.types.shared_params.function_definition import FunctionDefinition
from openai.types.shared_params.response_format_json_schema import (
    JSONSchema,
    ResponseFormatJSONSchema,
)

logger = get_logger(__name__)
DEFAULT_TIKTOKEN_ENCODING = "cl100k_base"
MAX_EMBEDDING_TOKENS = 8000


def _print_backoff_message(e: Details):
    logger.warning(
        f"OpenAI backing off for {e['wait']:.2f}s due to {e['exception'].__class__.__name__}"  # type: ignore
    )


def _parse_message_content(
    content: str | list[Content],
) -> str | list[ChatCompletionContentPartTextParam]:
    if isinstance(content, str):
        return content
    else:
        result: list[ChatCompletionContentPartTextParam] = []
        for sub_content in content:
            if sub_content.type == "text":
                result.append(
                    ChatCompletionContentPartTextParam(type="text", text=sub_content.text)
                )
            else:
                raise ValueError(f"Unsupported content type: {sub_content.type}")
        return result


def _parse_chat_messages(messages: list[ChatMessage]) -> list[ChatCompletionMessageParam]:
    result: list[ChatCompletionMessageParam] = []

    for message in messages:
        if message.role == "user":
            result.append(
                ChatCompletionUserMessageParam(
                    role=message.role,
                    content=_parse_message_content(message.content),
                )
            )
        elif message.role == "assistant":
            tool_calls = (
                [
                    ChatCompletionMessageToolCallParam(
                        id=tool_call.id,
                        function=OpenAIFunctionParam(
                            name=tool_call.function,
                            arguments=json.dumps(tool_call.arguments),
                        ),
                        type="function",
                    )
                    for tool_call in message.tool_calls
                ]
                if message.tool_calls
                else None
            )
            # Redundant code annoyingly necessary due to typechecking
            if not tool_calls:
                result.append(
                    ChatCompletionAssistantMessageParam(
                        role=message.role, content=_parse_message_content(message.content)
                    )
                )
            else:
                result.append(
                    ChatCompletionAssistantMessageParam(
                        role=message.role,
                        content=_parse_message_content(message.content),
                        tool_calls=tool_calls,
                    )
                )
        elif message.role == "tool":
            result.append(
                ChatCompletionToolMessageParam(
                    role=message.role,
                    content=_parse_message_content(message.content),
                    tool_call_id=str(message.tool_call_id),
                )
            )
        elif message.role == "system":
            result.append(
                ChatCompletionSystemMessageParam(
                    role=message.role,
                    content=_parse_message_content(message.content),
                )
            )

    return result


def _parse_tools(tools: list[ToolInfo]) -> list[ChatCompletionToolParam]:
    return [
        ChatCompletionToolParam(
            type="function",
            function=FunctionDefinition(
                name=tool.name,
                description=tool.description,
                parameters=tool.parameters.model_dump(exclude_none=True),
            ),
        )
        for tool in tools
    ]


def _build_response_format(
    response_format: ResponseFormat | None,
) -> ResponseFormatJSONSchema | None:
    if response_format is None:
        return None
    return ResponseFormatJSONSchema(
        type="json_schema",
        json_schema=JSONSchema(
            name=response_format.name,
            schema=response_format.schema_,
            strict=response_format.strict,
        ),
    )


@backoff.on_exception(
    backoff.expo,
    exception=(Exception,),
    giveup=lambda e: isinstance(e, BadRequestError)
    or isinstance(e, RateLimitException)
    or isinstance(e, AuthenticationError),
    max_tries=3,
    factor=2.0,
    on_backoff=_print_backoff_message,
)
async def get_openai_chat_completion_streaming_async(
    client: AsyncOpenAI,
    streaming_callback: AsyncSingleStreamingCallback | None,
    messages: list[ChatMessage],
    model_name: str,
    tools: list[ToolInfo] | None = None,
    tool_choice: Literal["auto", "required"] | None = None,
    max_new_tokens: int = 32,
    temperature: float = 1.0,
    reasoning_effort: Literal["low", "medium", "high"] | None = None,
    logprobs: bool = False,
    top_logprobs: int | None = None,
    timeout: float = 30.0,
    response_format: ResponseFormat | None = None,
):
    input_messages = _parse_chat_messages(messages)
    input_tools = _parse_tools(tools) if tools else NOT_GIVEN
    oai_response_format = _build_response_format(response_format)

    try:
        async with asyncio.timeout(timeout) if timeout else contextlib.nullcontext():  # type: ignore
            stream = await client.chat.completions.create(
                model=model_name,
                messages=input_messages,
                tools=input_tools,
                tool_choice=tool_choice or NOT_GIVEN,
                max_completion_tokens=max_new_tokens,
                temperature=temperature,
                reasoning_effort=reasoning_effort or NOT_GIVEN,
                logprobs=logprobs,
                top_logprobs=top_logprobs,
                response_format=oai_response_format or NOT_GIVEN,
                stream=True,
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


def update_llm_output(llm_output_partial: LLMOutputPartial | None, chunk: ChatCompletionChunk):
    # Collect existing outputs
    if llm_output_partial is not None:
        cur_texts: list[str | None] = [c.text for c in llm_output_partial.completions]
        cur_finish_reasons: list[FinishReasonType | None] = [
            c.finish_reason for c in llm_output_partial.completions
        ]
        cur_tool_calls_all: list[list[ToolCallPartial | None] | None] = [
            cast(list[ToolCallPartial | None], c.tool_calls) for c in llm_output_partial.completions
        ]
    else:
        cur_texts, cur_finish_reasons, cur_tool_calls_all = [], [], []

    # Define functions for getting and setting values of the current state
    def _get_text(i: int):
        if i >= len(cur_texts):
            return None
        else:
            return cur_texts[i]

    def _set_text(i: int, text: str):
        if i >= len(cur_texts):
            cur_texts.extend([None] * (i - len(cur_texts) + 1))
        cur_texts[i] = text

    def _get_finish_reason(i: int):
        if i >= len(cur_finish_reasons) or cur_finish_reasons[i] is None:
            return None
        else:
            return cur_finish_reasons[i]

    def _set_finish_reason(i: int, finish_reason: FinishReasonType | None):
        if i >= len(cur_finish_reasons):
            cur_finish_reasons.extend([None] * (i - len(cur_finish_reasons) + 1))
        cur_finish_reasons[i] = finish_reason

    def _get_tool_calls(i: int):
        if i >= len(cur_tool_calls_all):
            return None
        else:
            return cur_tool_calls_all[i]

    def _get_tool_call(i: int, j: int):
        if i >= len(cur_tool_calls_all):
            return None
        else:
            cur_tool_calls = cur_tool_calls_all[i]
            if cur_tool_calls is None or j >= len(cur_tool_calls):
                return None
            else:
                return cur_tool_calls[j]

    def _set_tool_call(i: int, j: int, tool_call: ToolCallPartial):
        if i >= len(cur_tool_calls_all):
            cur_tool_calls_all.extend([None] * (i - len(cur_tool_calls_all) + 1))

        # Add ToolCall to current choice index
        cur_tool_calls = cur_tool_calls_all[i] or []
        if j >= len(cur_tool_calls):
            cur_tool_calls.extend([None] * (j - len(cur_tool_calls) + 1))
        cur_tool_calls[j] = tool_call

        # Re-update the global array
        cur_tool_calls_all[i] = cur_tool_calls

    # Update existing completions based on this chunk
    for choice in chunk.choices:
        i, delta = choice.index, choice.delta

        # Resolve text and finish reason
        _set_text(i, (_get_text(i) or "") + (delta.content or ""))
        _set_finish_reason(i, choice.finish_reason or _get_finish_reason(i))

        # Tool call resolution is more complicated
        for tc_delta in delta.tool_calls or []:
            tc_idx = tc_delta.index
            tc_function = tc_delta.function.name if tc_delta.function else None
            tc_arguments = tc_delta.function.arguments if tc_delta.function else None

            old_tool_call = _get_tool_call(i, tc_idx)

            if old_tool_call:
                tool_call_partial = ToolCallPartial(
                    id=old_tool_call.id or tc_delta.id,
                    function=(old_tool_call.function or "") + (tc_function or ""),
                    arguments_raw=(old_tool_call.arguments_raw or "") + (tc_arguments or ""),
                    type="function",
                )
            else:
                tool_call_partial = ToolCallPartial(
                    id=tc_delta.id,
                    function=tc_function or "",
                    arguments_raw=tc_arguments or "",
                    type="function",
                )

            _set_tool_call(i, tc_idx, tool_call_partial)

    completions: list[LLMCompletionPartial] = []
    for i in range(len(cur_texts)):
        completions.append(
            LLMCompletionPartial(
                text=_get_text(i),
                tool_calls=_get_tool_calls(i),
                finish_reason=_get_finish_reason(i),
            )
        )

    return LLMOutputPartial(completions=completions, model=chunk.model)


@backoff.on_exception(
    backoff.expo,
    exception=(Exception,),
    giveup=lambda e: isinstance(e, BadRequestError)
    or isinstance(e, RateLimitException)
    or isinstance(e, AuthenticationError),
    max_tries=3,
    factor=2.0,
    on_backoff=_print_backoff_message,
)
async def get_openai_chat_completion_async(
    client: AsyncOpenAI,
    messages: list[ChatMessage],
    model_name: str,
    tools: list[ToolInfo] | None = None,
    tool_choice: Literal["auto", "none", "required"] | None = None,
    max_new_tokens: int = 32,
    temperature: float = 1.0,
    reasoning_effort: Literal["low", "medium", "high"] | None = None,
    logprobs: bool = False,
    top_logprobs: int | None = None,
    timeout: float = 5.0,
    response_format: ResponseFormat | None = None,
) -> LLMOutput:
    input_messages = _parse_chat_messages(messages)
    input_tools = _parse_tools(tools) if tools else NOT_GIVEN
    oai_response_format = _build_response_format(response_format)

    try:
        async with asyncio.timeout(timeout) if timeout else contextlib.nullcontext():  # type: ignore
            raw_output = await client.chat.completions.create(
                model=model_name,
                messages=input_messages,
                tools=input_tools,
                tool_choice=tool_choice or NOT_GIVEN,
                max_completion_tokens=max_new_tokens,
                temperature=temperature,
                reasoning_effort=reasoning_effort or NOT_GIVEN,
                logprobs=logprobs,
                top_logprobs=top_logprobs,
                response_format=oai_response_format or NOT_GIVEN,
            )

            # If the completion is empty and was truncated, raise
            output = parse_openai_completion(raw_output, model_name)
            if output.first and output.first.finish_reason == "length" and output.first.no_text:
                raise CompletionTooLongException(
                    f"Completion empty due to truncation. Consider increasing max_new_tokens (currently {max_new_tokens}). "
                    f"Usage: {raw_output.usage}"
                )
            for c in output.completions:
                if c.finish_reason == "length":
                    logger.warning(
                        f"Completion truncated due to length; consider increasing max_new_tokens (currently {max_new_tokens}). "
                        f"Usage: {raw_output.usage}"
                    )

            return output
    except RateLimitError as e:
        # Raise a custom type that prod_llms can catch
        raise RateLimitException(e) from e


def get_openai_client_sync(api_key: str | None = None) -> OpenAI:
    if api_key:
        return OpenAI(api_key=api_key)
    return OpenAI()


def get_openai_client_async(api_key: str | None = None) -> AsyncOpenAI:
    if api_key:
        return AsyncOpenAI(api_key=api_key)
    return AsyncOpenAI()


@backoff.on_exception(
    backoff.expo,
    exception=(Exception),
    max_tries=10,
    factor=2.0,
    on_backoff=_print_backoff_message,
)
async def _get_openai_embeddings_async_one_batch(
    client: AsyncOpenAI, texts_batch: list[str], model_name: str, dimensions: int | None
):
    response = await client.embeddings.create(
        model=model_name,
        input=texts_batch,
        dimensions=dimensions if dimensions is not None else NOT_GIVEN,
    )
    return [data.embedding for data in response.data]


async def get_openai_embeddings_async(
    client: AsyncOpenAI,
    texts: list[str],
    model_name: str = "text-embedding-3-large",
    dimensions: int | None = 1536,
    max_concurrency: int = 100,
) -> list[list[float] | None]:
    """Asynchronously get embeddings for a list of texts using OpenAI's embedding model."""
    if model_name != "text-embedding-3-large":
        assert dimensions is None, f"{model_name} does not have a variable dimension size"

    # Tokenize and truncate texts
    tokenizer = tiktoken.get_encoding(DEFAULT_TIKTOKEN_ENCODING)
    truncated_texts: list[str] = []
    for i, text in enumerate(texts):
        tokens = tokenizer.encode(text)
        if len(tokens) > MAX_EMBEDDING_TOKENS:
            print(
                f"Warning: Text at index {i} has been truncated from {len(tokens)} to {MAX_EMBEDDING_TOKENS} tokens."
            )
            tokens = tokens[:MAX_EMBEDDING_TOKENS]
        truncated_texts.append(tokenizer.decode(tokens))

    semaphore = asyncio.Semaphore(max_concurrency)

    async def limited_task(texts_batch: list[str]):
        async with semaphore:
            try:
                return await _get_openai_embeddings_async_one_batch(
                    client, texts_batch, model_name, dimensions
                )
            except Exception as e:
                print(f"Error in fetch_embeddings: {e}. Returning None.")
                return [None] * len(texts_batch)

    # Create batches of 1000 texts (OpenAI's current limit per request)
    batches = [truncated_texts[i : i + 1000] for i in range(0, len(truncated_texts), 1000)]

    # Run tasks concurrently
    tasks = [limited_task(batch) for batch in batches]
    results = await asyncio.gather(*tasks)

    # Flatten the results
    embeddings = [embedding for batch_result in results for embedding in batch_result]

    return embeddings


def get_openai_embeddings_sync(
    client: OpenAI,
    texts: list[str],
    model_name: str = "text-embedding-3-large",
    dimensions: int | None = 1536,
) -> list[list[float] | None]:
    """Synchronously get embeddings for a list of texts using OpenAI's embedding model."""
    # Tokenize and truncate texts
    tokenizer = tiktoken.get_encoding(DEFAULT_TIKTOKEN_ENCODING)
    truncated_texts: list[str] = []
    for i, text in enumerate(texts):
        tokens = tokenizer.encode(text)
        if len(tokens) > MAX_EMBEDDING_TOKENS:
            print(
                f"Warning: Text at index {i} has been truncated from {len(tokens)} to {MAX_EMBEDDING_TOKENS} tokens."
            )
            tokens = tokens[:MAX_EMBEDDING_TOKENS]
        truncated_texts.append(tokenizer.decode(tokens))

    # Process in batches of 1000
    embeddings: list[list[float] | None] = []
    for i in range(0, len(truncated_texts), 1000):
        batch = truncated_texts[i : i + 1000]
        try:
            response = client.embeddings.create(
                model=model_name,
                input=batch,
                dimensions=dimensions if dimensions is not None else NOT_GIVEN,
            )
            batch_embeddings = [data.embedding for data in response.data]
            embeddings.extend(batch_embeddings)
        except Exception as e:
            print(f"Error in get_openai_embeddings_sync: {e}")
            embeddings.extend([None] * len(batch))

    return embeddings


def _parse_openai_tool_call(tc: ChatCompletionMessageToolCall) -> ToolCall:
    # Attempt to parse the tool call arguments as JSON
    try:
        arguments: dict[str, Any] = json.loads(tc.function.arguments)
        parse_error = None
    except Exception as e:
        arguments = {}
        parse_error = f"Couldn't parse tool call arguments as JSON: {e}. Original input: {tc.function.arguments}"

    return ToolCall(
        id=tc.id,
        function=tc.function.name,
        arguments=arguments,
        parse_error=parse_error,
        type=tc.type,
    )


def _extract_usage_tokens(
    usage: Any,
) -> tuple[int | None, int | None, int | None]:
    """Extract token counts from OpenAI-style usage payloads."""
    if usage is None:
        return None, None, None

    cache_tokens: int | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None

    if isinstance(usage, Mapping):
        prompt_tokens_details = usage.get("prompt_tokens_details")
        if isinstance(prompt_tokens_details, Mapping):
            cache_tokens = prompt_tokens_details.get("cached_tokens")
        if cache_tokens is None:
            cache_tokens = usage.get("cache_read_input_tokens")

        input_tokens = usage.get("prompt_tokens") or usage.get("input_tokens")
        output_tokens = usage.get("completion_tokens") or usage.get("output_tokens")
    else:
        prompt_tokens_details = getattr(usage, "prompt_tokens_details", None)
        if prompt_tokens_details is not None:
            cache_tokens = getattr(prompt_tokens_details, "cached_tokens", None)
        if cache_tokens is None:
            cache_tokens = getattr(usage, "cache_read_input_tokens", None)

        input_tokens = getattr(usage, "prompt_tokens", None)
        if input_tokens is None:
            input_tokens = getattr(usage, "input_tokens", None)

        output_tokens = getattr(usage, "completion_tokens", None)
        if output_tokens is None:
            output_tokens = getattr(usage, "output_tokens", None)

    return cache_tokens, input_tokens, output_tokens


def parse_openai_completion(response: ChatCompletion | None, model: str) -> LLMOutput:
    if response is None:
        return LLMOutput(
            model=model,
            completions=[],
            errors=["no_response"],
        )
    if response.choices is None:
        raise ValueError((f"response.choices is None. response was: {response}"))
    cache_tokens, input_tokens, output_tokens = _extract_usage_tokens(response.usage)
    return LLMOutput(
        model=response.model,
        completions=[
            LLMCompletion(
                text=choice.message.content,
                finish_reason=choice.finish_reason,
                tool_calls=(
                    [_parse_openai_tool_call(tc) for tc in tcs]
                    if (tcs := choice.message.tool_calls)
                    else None
                ),
                top_logprobs=(
                    [pos.top_logprobs for pos in choice.logprobs.content]
                    if choice.logprobs is not None and choice.logprobs.content is not None
                    else None
                ),
            )
            for choice in response.choices
        ],
        usage=Usage(
            cache_read_input_tokens=cache_tokens,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        ),
    )


async def is_openai_api_key_valid(api_key: str) -> bool:
    """Test whether an OpenAI API key is valid or invalid."""
    client = AsyncOpenAI(api_key=api_key)

    try:
        await client.chat.completions.create(
            model="gpt-3.5-turbo", messages=[{"role": "user", "content": "hi"}], max_tokens=1
        )
        return True
    except AuthenticationError:
        return False
    except Exception:
        return True
