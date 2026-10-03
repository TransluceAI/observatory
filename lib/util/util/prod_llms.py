from typing import Any, Awaitable, Callable, TypedDict

from util import anthropic, openai
from util.env import ENV
from util.types import ChatMessage


class ProviderConfig(TypedDict):
    keys: list[str]
    current_key_index: int
    async_client: openai.AsyncOpenAI | anthropic.AsyncAnthropic
    async_completion_getter: Callable[
        [Any, list[list[ChatMessage]], str, int, float, int], Awaitable[list[Any]]
    ]
    completion_parser: Callable[[Any], str | None]
    models: dict[str, str]


class LLMManager:
    def __init__(self):
        if ENV.ANTHROPIC_API_KEY is None:
            raise Exception("Missing Anthropic API key; check your .env")
        if ENV.OPENAI_API_KEY is None:
            raise Exception("Missing OpenAI API key; check your .env")

        self.providers: dict[str, ProviderConfig] = {
            "anthropic": {
                "keys": [ENV.ANTHROPIC_API_KEY],
                "current_key_index": 0,
                "async_client": anthropic.get_anthropic_client_async(),
                "async_completion_getter": anthropic.get_anthropic_chat_completions_async,
                "completion_parser": anthropic.parse_anthropic_completion,
                "models": {
                    "smart": "claude-sonnet-4-6",
                    "fast": "claude-haiku-4-5-20251001",
                },
            },
            "openai": {
                "keys": [ENV.OPENAI_API_KEY],
                "current_key_index": 0,
                "async_client": openai.get_openai_client_async(),
                "async_completion_getter": openai.get_openai_chat_completions_async,
                "completion_parser": openai.parse_openai_completion,
                "models": {
                    "smart": "gpt-4o-2024-08-06",
                    "fast": "gpt-4o-mini-2024-07-18",
                },
            },
            # Add more providers here as needed
        }
        self.provider_order = list(self.providers.keys())
        self.current_provider_index = 0
        self.provider = self.providers[self.provider_order[self.current_provider_index]]

    async def get_completions(
        self,
        messages_list: list[list[ChatMessage]],
        model_category: str,
        max_new_tokens: int = 32,
        temperature: float = 1.0,
        max_concurrency: int = 100,
    ) -> list[str | None]:
        while True:
            try:
                client = self.provider["async_client"]
                client.api_key = self.provider["keys"][self.provider["current_key_index"]]
                model_name = self.provider["models"][model_category]
                completions = await self.provider["async_completion_getter"](
                    client,
                    messages_list,
                    model_name,
                    max_new_tokens,
                    temperature,
                    max_concurrency,
                )

                # If all the completions are None, rotate keys and swap provider
                if all(completion is None for completion in completions):
                    raise Exception("All completions are None")

                # Parse completions based on provider
                parser = self.provider["completion_parser"]
                return [parser(completion) for completion in completions]

            except Exception as e:
                print(f"Error occurred: {e}")
                if not self._rotate_keys_and_swap_provider():
                    print("All keys and providers exhausted. Returning None.")
                    return [None] * len(messages_list)

    def _rotate_keys_and_swap_provider(self) -> bool:
        """
        Rotate to the next API key for the current provider.
        If all keys for the current provider are exhausted, move to the next provider.
        Return True if there is a new key/provider to try, False if all are exhausted.
        """

        # Move to the next key index
        self.provider["current_key_index"] += 1

        if self.provider["current_key_index"] < len(self.provider["keys"]):
            # There are more keys in this provider
            provider_name = self.provider_order[self.current_provider_index]
            print(f"Switched to next key for provider '{provider_name}'.")
            return True
        else:
            # No more keys in the current provider, reset and move to next provider
            self.provider["current_key_index"] = 0  # Reset key index for this provider
            self.current_provider_index += 1
            if self.current_provider_index < len(self.provider_order):
                # Move to next provider
                provider_name = self.provider_order[self.current_provider_index]
                self.provider = self.providers[provider_name]
                print(f"Switched to next provider '{provider_name}'.")
                return True
            else:
                # All providers and their keys have been exhausted
                print("All providers and their keys have been exhausted.")
                return False


async def get_llm_completions_async(
    messages_list: list[list[ChatMessage]],
    model_category: str,
    max_new_tokens: int = 32,
    temperature: float = 1.0,
    max_concurrency: int = 100,
):
    llm_manager = LLMManager()
    return await llm_manager.get_completions(
        messages_list, model_category, max_new_tokens, temperature, max_concurrency
    )
