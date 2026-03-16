"""Simplified LLM completion API.

Provides ``get_llm_completions_async`` (batch) and ``complete_async`` (single)
for calling LLM providers without rotation or fallback logic.
"""

import os
import traceback
from contextlib import nullcontext
from functools import partial
from typing import Any, Literal, Sequence

import anyio
from llm_util import anthropic as anthropic_mod
from llm_util import openai as openai_mod
from llm_util.anthropic import (
    get_anthropic_chat_completion_async,
    get_anthropic_chat_completion_streaming_async,
)
from llm_util.llm_cache import LLMCache
from llm_util.openai import (
    get_openai_chat_completion_async,
    get_openai_chat_completion_streaming_async,
)
from llm_util.types import (
    AsyncSingleStreamingCallback,
    AsyncStreamingCallback,
    ChatMessage,
    LLMApiKeys,
    LLMOutput,
    ModelCallParams,
    RateLimitException,
    ResponseFormat,
    ToolInfo,
    get_single_streaming_callback,
    parse_chat_message,
)
from log_util import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Provider registry
# ---------------------------------------------------------------------------

_PROVIDER_FUNCTIONS: dict[str, dict[str, Any]] = {
    "anthropic": {
        "client_getter": anthropic_mod.get_anthropic_client_async,
        "single_output_getter": get_anthropic_chat_completion_async,
        "single_streaming_output_getter": get_anthropic_chat_completion_streaming_async,
    },
    "openai": {
        "client_getter": openai_mod.get_openai_client_async,
        "single_output_getter": get_openai_chat_completion_async,
        "single_streaming_output_getter": get_openai_chat_completion_streaming_async,
    },
}

_PROVIDER_KEY_ENVVARS: dict[str, str] = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
}

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _resolve_api_key(
    provider: str,
    llm_api_keys: LLMApiKeys | None,
) -> str | None:
    """Get the API key for a provider from explicit keys or environment."""
    key_field_map: dict[str, str] = {
        "anthropic": "anthropic_key",
        "openai": "openai_key",
    }
    if llm_api_keys:
        explicit = llm_api_keys.get(key_field_map.get(provider, ""))  # type: ignore[arg-type]
        if explicit:
            return explicit
    envvar = _PROVIDER_KEY_ENVVARS.get(provider)
    if envvar:
        return os.getenv(envvar)
    return None


def _get_client(provider: str, api_key: str | None) -> Any:
    """Create a provider client with an optional explicit API key."""
    provider_funcs = _PROVIDER_FUNCTIONS.get(provider)
    if provider_funcs is None:
        raise ValueError(f"Unknown provider: {provider}")

    client_getter = provider_funcs["client_getter"]
    if api_key is not None:
        return client_getter(api_key=api_key)
    return client_getter()


# ---------------------------------------------------------------------------
# Parallel execution
# ---------------------------------------------------------------------------


async def _parallelize_calls(
    single_output_getter: Any,
    streaming_callback: AsyncStreamingCallback | None,
    completion_callback: AsyncStreamingCallback | None,
    client: Any,
    messages_list: list[list[ChatMessage]],
    model_name: str,
    tools: list[ToolInfo] | None,
    tool_choice: Literal["auto", "required"] | None,
    max_new_tokens: int,
    temperature: float,
    reasoning_effort: Literal["low", "medium", "high", "max"] | None,
    logprobs: bool,
    top_logprobs: int | None,
    timeout: float,
    max_concurrency: int | None,
    semaphore: anyio.Semaphore | None,
    cache: LLMCache | None = None,
    response_format: ResponseFormat | None = None,
):
    base_func = partial(
        single_output_getter,
        client=client,
        model_name=model_name,
        tools=tools,
        tool_choice=tool_choice,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        reasoning_effort=reasoning_effort,
        logprobs=logprobs,
        top_logprobs=top_logprobs,
        timeout=timeout,
        response_format=response_format,
    )
    if max_concurrency is not None:
        ctx = anyio.Semaphore(max_concurrency)
    else:
        ctx = nullcontext() if semaphore is None else semaphore

    responses: list[LLMOutput | None] = [None for _ in messages_list]

    async def _limited_task(i: int, messages: list[ChatMessage]) -> None:
        nonlocal responses

        async with ctx:
            try:
                # Use streaming if streaming_callback is provided
                if streaming_callback is None:
                    result = await base_func(client=client, messages=messages)
                    if completion_callback:
                        await completion_callback(i, result)
                else:
                    result = await base_func(
                        client=client,
                        streaming_callback=get_single_streaming_callback(i, streaming_callback),
                        messages=messages,
                    )

                responses[i] = result
            except Exception as e:
                error_message = (
                    f"Call to {model_name} failed even with backoff: {e.__class__.__name__}."
                )
                if not isinstance(e, RateLimitException):
                    error_message += f" Failure traceback:\n{traceback.format_exc()}"

                logger.error(error_message)
                error_output = LLMOutput(
                    model=model_name,
                    completions=[],
                    errors=["rate_limit" if isinstance(e, RateLimitException) else "other"],
                )
                responses[i] = error_output
                if completion_callback:
                    await completion_callback(i, error_output)

    def _cache_responses():
        nonlocal responses, cache

        if cache is not None:
            indices = [
                i
                for i, response in enumerate(responses)
                if isinstance(response, LLMOutput) and not response.did_error
            ]
            valid_outputs: list[LLMOutput] = [
                responses[i] for i in indices if responses[i] is not None  # type: ignore[misc]
            ]
            cache.set_batch(
                [messages_list[i] for i in indices],
                model_name,
                valid_outputs,
                tools=tools,
                tool_choice=tool_choice,
                reasoning_effort=reasoning_effort,
                temperature=temperature,
                logprobs=logprobs,
                top_logprobs=top_logprobs,
                response_format=response_format,
            )
            return len(indices)
        else:
            return 0

    # Get all results
    try:
        async with anyio.create_task_group() as tg:
            for i, messages in enumerate(messages_list):
                tg.start_soon(_limited_task, i, messages)
    except anyio.get_cancelled_exc_class():
        num_cached = _cache_responses()
        logger.info(
            f"Cancelled {len(messages_list) - num_cached} unfinished LLM API calls; "
            f"cached {num_cached} completed responses"
        )
        raise

    # Cache results if available
    if cache is not None:
        _cache_responses()

    return responses


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def get_llm_completions_async(
    messages_list: Sequence[Sequence[ChatMessage | dict[str, Any]]],
    model: ModelCallParams,
    *,
    tools: list[ToolInfo] | None = None,
    tool_choice: Literal["auto", "required"] | None = None,
    max_new_tokens: int = 1024,
    temperature: float = 1.0,
    logprobs: bool = False,
    top_logprobs: int | None = None,
    max_concurrency: int = 100,
    timeout: float = 60.0,
    streaming_callback: AsyncStreamingCallback | None = None,
    completion_callback: AsyncStreamingCallback | None = None,
    use_cache: bool = False,
    llm_api_keys: LLMApiKeys | None = None,
    response_format: ResponseFormat | None = None,
) -> list[LLMOutput]:
    """Get LLM completions for a batch of message lists.

    Args:
        messages_list: Batch of conversations. Each conversation is a sequence
            of ``ChatMessage`` or raw dicts.
        model: Provider + model name (+ optional reasoning_effort).
        tools: Tool definitions available to the model.
        tool_choice: How the model should use tools.
        max_new_tokens: Maximum tokens to generate.
        temperature: Sampling temperature.
        logprobs: Whether to return log probabilities.
        top_logprobs: Number of top log probabilities per token.
        max_concurrency: Maximum concurrent API calls.
        timeout: Per-call timeout in seconds.
        streaming_callback: Called with partial results during streaming.
        completion_callback: Called when each item completes.
        use_cache: Whether to cache results in SQLite.
        llm_api_keys: Explicit API keys (overrides env vars).
        response_format: Structured output specification.

    Returns:
        List of ``LLMOutput``, one per conversation in ``messages_list``.
    """
    provider = model.provider
    model_name = model.model_name
    effective_reasoning_effort = model.reasoning_effort

    # Resolve API key and create client
    api_key = _resolve_api_key(provider, llm_api_keys)
    client = _get_client(provider, api_key)

    # Set API key on client if needed (for clients that use it as an attribute)
    if api_key and hasattr(client, "api_key"):
        client.api_key = api_key

    # Determine which getter to use
    provider_funcs = _PROVIDER_FUNCTIONS.get(provider)
    if provider_funcs is None:
        raise ValueError(f"Unknown provider: {provider}")

    if streaming_callback is not None:
        single_getter = provider_funcs["single_streaming_output_getter"]
    else:
        single_getter = provider_funcs["single_output_getter"]

    # Parse messages
    parsed_messages_list = [
        [
            message if isinstance(message, ChatMessage) else parse_chat_message(message)
            for message in messages
        ]
        for messages in messages_list
    ]

    # Set up cache
    cache = LLMCache() if use_cache else None

    # Check cache for existing results
    results: list[LLMOutput | None] = [None] * len(parsed_messages_list)
    uncached_indices: list[int] = []
    uncached_messages: list[list[ChatMessage]] = []

    if cache is not None:
        hits = 0
        for i, messages in enumerate(parsed_messages_list):
            cached_result = cache.get(
                messages,
                model_name,
                tools=tools,
                tool_choice=tool_choice,
                reasoning_effort=effective_reasoning_effort,
                temperature=temperature,
                logprobs=logprobs,
                top_logprobs=top_logprobs,
                response_format=response_format,
            )
            if cached_result is not None:
                results[i] = cached_result
                hits += 1
            else:
                uncached_indices.append(i)
                uncached_messages.append(messages)

        misses = len(parsed_messages_list) - hits
        logger.info(f"{model_name}: {hits} cache hits, {misses} misses")
    else:
        uncached_indices = list(range(len(parsed_messages_list)))
        uncached_messages = parsed_messages_list

    # Call streaming/completion callbacks for cached messages
    if streaming_callback is not None:
        for i in range(len(parsed_messages_list)):
            result = results[i]
            if result is not None:
                await streaming_callback(i, result)

    if completion_callback is not None:
        for i in range(len(parsed_messages_list)):
            result = results[i]
            if result is not None:
                await completion_callback(i, result)

    # Get completions for uncached messages
    if uncached_messages:
        outputs = await _parallelize_calls(
            single_getter,
            streaming_callback,
            completion_callback,
            client,
            uncached_messages,
            model_name,
            tools=tools,
            tool_choice=tool_choice,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            reasoning_effort=effective_reasoning_effort,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            timeout=timeout,
            max_concurrency=max_concurrency,
            semaphore=None,
            cache=cache,
            response_format=response_format,
        )

        for i, output in enumerate(outputs):
            results[uncached_indices[i]] = output

    # Fill any None results with errors
    final_results: list[LLMOutput] = []
    for result in results:
        if result is None:
            final_results.append(
                LLMOutput(
                    model=model_name,
                    completions=[],
                    errors=["other"],
                )
            )
        else:
            final_results.append(result)

    return final_results


async def complete_async(
    messages: Sequence[ChatMessage | dict[str, Any]],
    model: ModelCallParams,
    *,
    tools: list[ToolInfo] | None = None,
    tool_choice: Literal["auto", "required"] | None = None,
    max_new_tokens: int = 1024,
    temperature: float = 1.0,
    logprobs: bool = False,
    top_logprobs: int | None = None,
    timeout: float = 60.0,
    streaming_callback: AsyncSingleStreamingCallback | None = None,
    use_cache: bool = False,
    llm_api_keys: LLMApiKeys | None = None,
    response_format: ResponseFormat | None = None,
) -> LLMOutput:
    """Single-call convenience wrapper around ``get_llm_completions_async``.

    Returns a single ``LLMOutput`` instead of a list.
    """
    # Wrap the single streaming callback into a batch callback if provided
    batch_streaming_callback: AsyncStreamingCallback | None = None
    if streaming_callback is not None:

        async def _batch_cb(batch_index: int, llm_output: LLMOutput) -> None:  # noqa: ARG001
            await streaming_callback(llm_output)

        batch_streaming_callback = _batch_cb

    results = await get_llm_completions_async(
        messages_list=[messages],
        model=model,
        tools=tools,
        tool_choice=tool_choice,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        logprobs=logprobs,
        top_logprobs=top_logprobs,
        max_concurrency=1,
        timeout=timeout,
        streaming_callback=batch_streaming_callback,
        use_cache=use_cache,
        llm_api_keys=llm_api_keys,
        response_format=response_format,
    )
    return results[0]
