import asyncio
import contextlib
from typing import Any, Literal, cast

import backoff
from backoff.types import Details
from llm_util.prod_llms import get_llm_completions_async
from llm_util.types import LLMOutput, ModelCallParams, RateLimitException, Usage
from log_util import get_logger
from openai import AsyncOpenAI, AuthenticationError, BadRequestError, RateLimitError
from openai.types.chat import ChatCompletion
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from pydantic import BaseModel

logger = get_logger(__name__)


class DataGeneratorConfig(BaseModel):
    """Configuration for generating data"""

    provider: str
    model_name: str
    reasoning_effort: Literal["low", "medium", "high"] | None
    max_new_tokens: int
    temperature: float = 1.0
    max_concurrency: int = 100
    timeout: float = 360.0


async def get_llm_completion_strs_async(
    messages_list: list[list[dict[str, Any]]],
    config: DataGeneratorConfig,
    use_cache: bool = False,
) -> tuple[list[str], list[Usage]]:
    outs: list[LLMOutput] = await get_llm_completions_async(
        messages_list=messages_list,
        model=ModelCallParams(
            provider=config.provider,
            model_name=config.model_name,
            reasoning_effort=config.reasoning_effort,
        ),
        max_new_tokens=config.max_new_tokens,
        temperature=config.temperature,
        max_concurrency=config.max_concurrency,
        timeout=config.timeout,
        use_cache=use_cache,
    )
    usages: list[Usage] = []
    completions: list[str] = []
    for out in outs:
        text = out.completions[0].text
        if text is not None:
            completions.append(text)
            assert out.usage is not None
            usages.append(out.usage)
    return completions, usages


def _print_backoff_message(e: Details):
    logger.warning(
        f"VLLM client backing off for {e['wait']:.2f}s due to {e['exception'].__class__.__name__}"  # type: ignore
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
async def get_vllm_chat_completion_async(
    client: AsyncOpenAI,
    messages: list[dict[str, str]],
    model: str,
    n: int = 1,
    max_tokens: int = 32,
    temperature: float = 1.0,
    timeout: float = 5.0,
    continue_final_message: bool = False,
) -> ChatCompletion:
    try:
        async with asyncio.timeout(timeout) if timeout else contextlib.nullcontext():
            return await client.chat.completions.create(
                model=model,
                messages=cast(list[ChatCompletionMessageParam], messages),
                extra_body={
                    "continue_final_message": continue_final_message,
                    "add_generation_prompt": not continue_final_message,
                },
                max_tokens=max_tokens,
                temperature=temperature,
                n=n,
            )
    except RateLimitError as e:
        # Raise a custom type that prod_llms can catch
        raise RateLimitException(e) from e


def compute_o3_cost(usage: Usage):
    return usage.compute_cost(cache_cost=0.5, input_cost=2, output_cost=8)


def compute_gpt41_nano_cost(usage: Usage):
    return usage.compute_cost(cache_cost=0.025, input_cost=0.1, output_cost=0.4)


def compute_gpt41_cost(usage: Usage):
    return usage.compute_cost(cache_cost=0.5, input_cost=2, output_cost=8)
