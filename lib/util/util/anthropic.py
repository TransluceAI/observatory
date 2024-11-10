import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import cast

import backoff
from anthropic import Anthropic, AsyncAnthropic
from anthropic._types import NOT_GIVEN
from anthropic.types import Message, MessageParam
from backoff.types import Details
from tqdm.asyncio import tqdm_asyncio
from util.env import ENV
from util.types import ChatMessage


def _print_backoff_message(e: Details):
    print(
        f"Anthropic backing off for {e['wait']:.2f}s due to {e['exception'].__class__.__name__}"  # type: ignore
    )


@backoff.on_exception(
    backoff.expo,
    exception=(Exception),
    max_tries=3,
    factor=2.0,
    on_backoff=_print_backoff_message,
)
async def _get_anthropic_chat_completion_async(
    client: AsyncAnthropic,
    messages: list[ChatMessage],
    model_name: str,
    max_new_tokens: int = 32,
    temperature: float = 1.0,
    timeout: float = 5.0,
) -> Message:
    system = messages[0]["content"] if messages[0]["role"] == "system" else None

    async with asyncio.timeout(timeout) if timeout else asyncio.nullcontext():  # type: ignore
        return await client.messages.create(
            model=model_name,
            max_tokens=max_new_tokens,
            messages=[
                MessageParam(role=m["role"], content=m["content"])
                for m in messages
                if m["role"] != "system"
            ],
            temperature=temperature,
            system=system if system is not None else NOT_GIVEN,
        )


def get_anthropic_client_sync() -> Anthropic:
    if ENV.ANTHROPIC_API_KEY is None:
        raise Exception("Missing Anthropic API key; check your .env")
    return Anthropic(api_key=ENV.ANTHROPIC_API_KEY)


def get_anthropic_client_async() -> AsyncAnthropic:
    if ENV.ANTHROPIC_API_KEY is None:
        raise Exception("Missing Anthropic API key; check your .env")
    return AsyncAnthropic(api_key=ENV.ANTHROPIC_API_KEY)


async def get_anthropic_chat_completions_async(
    client: AsyncAnthropic,
    messages_list: list[list[ChatMessage]],
    model_name: str,
    max_new_tokens: int = 32,
    temperature: float = 1.0,
    max_concurrency: int = 100,
    timeout: float = 5.0,
    use_tqdm: bool = False,
):
    base_func = partial(
        _get_anthropic_chat_completion_async,
        model_name=model_name,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        timeout=timeout,
    )
    semaphore = asyncio.Semaphore(max_concurrency)

    async def limited_task(messages: list[ChatMessage]):
        async with semaphore:
            try:
                return await base_func(client=client, messages=messages)
            except Exception as e:  # This will catch any exception, including those from backoff
                print(
                    f"Anthropic chat completion failed even with backoff: {e.__class__.__name__}. Returning None."
                )
                return None

    tasks = [limited_task(messages) for messages in messages_list]
    if use_tqdm:
        responses = cast(
            list[Message | None],
            await tqdm_asyncio.gather(*tasks, desc="Processing messages"),  # type: ignore
        )
    else:
        responses = await asyncio.gather(*tasks)

    return responses


def parse_anthropic_completion(response: Message | None) -> str | None:
    if response is None:
        return None
    try:
        first_content = response.content[0]
        if first_content.type == "text":
            return first_content.text
        else:
            return None
    except (AttributeError, IndexError):
        return None


def get_anthropic_chat_completions_parallel(
    client: AsyncAnthropic,
    messages_list: list[list[ChatMessage]],
    model_name: str,
    max_new_tokens: int = 32,
    temperature: float = 1.0,
    max_concurrency: int = 100,
    timeout: float = 5.0,
    use_tqdm: bool = False,
) -> list[Message | None]:
    with ThreadPoolExecutor() as executor:
        future = executor.submit(
            asyncio.run,
            get_anthropic_chat_completions_async(
                client=client,
                messages_list=messages_list,
                model_name=model_name,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                max_concurrency=max_concurrency,
                timeout=timeout,
                use_tqdm=use_tqdm,
            ),
        )
        return future.result()
