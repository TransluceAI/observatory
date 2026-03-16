from typing import List

from anthropic import AsyncAnthropic
from explanations.llama_model import Llama3TokenizerWrapper, get_tokenizer
from openai import AsyncOpenAI
from pydantic import BaseModel, ConfigDict
from util.anthropic import (
    get_anthropic_chat_completions_parallel,
    get_anthropic_client_async,
    parse_anthropic_completion,
)
from util.openai import get_openai_chat_completions_parallel, get_openai_client_async
from util.types import ChatMessage
from vllm import LLM, SamplingParams


class Explainer(BaseModel):
    model_name: str
    max_new_tokens: int
    temperature: float = 1.0
    model_config = ConfigDict(arbitrary_types_allowed=True, protected_namespaces=())

    def get_chat_completions(
        self, messages_list: List[List[ChatMessage]], num_samples: int = 1
    ) -> tuple[List[str], int]:
        raise NotImplementedError


class OpenAIExplainer(Explainer):
    client: AsyncOpenAI

    def get_chat_completions(
        self, messages_list: List[List[ChatMessage]], num_samples: int = 1
    ) -> tuple[List[str], int]:
        # O1 models don't support system prompts.
        if self.model_name.startswith("o1"):
            updated_messages_list: List[List[ChatMessage]] = []
            for messages in messages_list:
                updated_messages = messages.copy()
                if messages[0]["role"] == "system":
                    updated_messages[0]["role"] = "user"
                updated_messages_list.append(updated_messages)
            messages_list = updated_messages_list

        assert num_samples == 1
        responses = get_openai_chat_completions_parallel(
            client=self.client,
            messages_list=messages_list,
            model_name=self.model_name,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
        )
        valid_response_strs: List[str] = []
        num_refusals = 0
        for response in responses:
            if response is None:
                num_refusals += 1
                continue
            for sample in response.choices:
                if sample.message.refusal:
                    num_refusals += 1
                else:
                    assert sample.message.content is not None
                    valid_response_strs.append(sample.message.content)
        return valid_response_strs, num_refusals


class AnthropicExplainer(Explainer):
    client: AsyncAnthropic

    def get_chat_completions(
        self, messages_list: List[List[ChatMessage]], num_samples: int = 1
    ) -> tuple[List[str], int]:
        assert num_samples == 1
        responses = get_anthropic_chat_completions_parallel(
            client=self.client,
            messages_list=messages_list,
            model_name=self.model_name,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
        )
        valid_response_strs: List[str] = []
        num_refusals = 0
        for response in responses:
            if response is None:
                num_refusals += 1
            text = parse_anthropic_completion(response)
            if text is not None:
                valid_response_strs.append(text)
            else:
                num_refusals += 1
        return valid_response_strs, num_refusals


class VLLMExplainer(Explainer):
    model: LLM
    tokenizer: Llama3TokenizerWrapper
    sampling_params: SamplingParams

    def get_chat_completions(
        self, messages_list: List[List[dict[str, str]]], num_samples: int = 1
    ) -> tuple[List[str], int]:
        self.sampling_params.n = num_samples
        prompts: List[str] = []
        for messages in messages_list:
            assert len(messages) == 2
            input_str: str = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False, return_dict=False
            )
            prompts.append(input_str)

        outputs = self.model.generate(prompts, self.sampling_params)
        response_strs: List[str] = []
        for out in outputs:
            for sample in out.outputs:
                response_str = self.tokenizer.update_output(sample.text)
                response_strs.append(response_str)
        return response_strs, 0


def get_explainer(
    model_name: str,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_p: float = 1.0,
    add_special_tokens: bool = True,
) -> Explainer:
    if "gpt" in model_name or "o1" in model_name:
        explainer = OpenAIExplainer(
            client=get_openai_client_async(),
            model_name=model_name,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
    elif "claude" in model_name:
        explainer = AnthropicExplainer(
            client=get_anthropic_client_async(),
            model_name=model_name,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
    else:
        llm = LLM(model=model_name)
        explainer = VLLMExplainer(
            model_name=model_name,
            max_new_tokens=max_new_tokens,
            model=llm,
            tokenizer=get_tokenizer(model_name, add_special_tokens=add_special_tokens),
            sampling_params=SamplingParams(
                temperature=temperature, max_tokens=max_new_tokens, top_p=top_p
            ),
        )
    return explainer
