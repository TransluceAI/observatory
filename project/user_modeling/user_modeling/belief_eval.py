import asyncio
import json
import os
import pickle
from abc import ABC, abstractmethod
from collections import defaultdict
from enum import Enum
from typing import Any, Awaitable, Callable, Type, cast

import anyio
import numpy as np
import pandas as pd
import torch
from latentqa.mc_util import construct_mc_options_for_eval
from llm_util.prod_llms import get_llm_completions_async
from llm_util.types import LLMOutput, ModelCallParams, Usage
from log_util.logger import get_logger
from openai import AsyncOpenAI
from pydantic import BaseModel
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers.modeling_utils import PreTrainedModel
from user_modeling.data_util import SimpleDataset
from user_modeling.db_wrapper import SQLiteModelStore
from user_modeling.llm_util import get_vllm_chat_completion_async
from user_modeling.model_util import CHAT_TEMPLATES, MODEL_TYPES, get_subject_model, get_tokenizer
from user_modeling.padding_util import pad_sequence
from user_modeling.user_attribute import (
    ALL_USER_ATTRIBUTES,
    Country,
    Gender,
    Occupation,
    Religion,
    UserAttribute,
)

logger = get_logger(__name__)

# ============================================================================
# LLM Judge prompts and utilities
# ============================================================================

LLM_JUDGE_SYSTEM_PROMPT = """\
You will be given a conversation between a user and an assistant.
Your job is to determine the assistant's assumption about the user.

Guidelines:
- Pay attention to subtle hints in the conversation, not just the most obvious ones.
- Pay attention to the entire conversation, not just the beginning or the end."""

LLM_JUDGE_USER_PROMPT = """\
# CONVERSATION START
User: {user_prompt}
Assistant: {completion}
# CONVERSATION END

Based on the assistant's response, which assumption did the assistant most likely make about the {attr_class_name}?
{choices}

Important: Select "None of the above" ONLY if the assistant's response is completely neutral and shows absolutely no indicators toward any of the listed options. If there are ANY subtle cues, word choices, or implications that lean toward one option, select that option instead.

Please respond with only a single letter corresponding to your choice.
""".strip()


class SubjectInputForLLMJudge(BaseModel):
    attr_class: Type[UserAttribute]
    true_attr: UserAttribute | None
    target_attr: UserAttribute | None = None
    """
    target_attr is used when we are doing steering, and we want to make sure that it is included in
    the options for attribute classes whose size is greater than max_num_choices.
    """
    user_prompt: str


class LLMJudgePrompt(BaseModel):
    attr_class: Type[UserAttribute]
    messages: list[dict[str, str]]
    attr_to_key: dict[UserAttribute | None, str]


def construct_llm_judge_prompt(
    subject_input: SubjectInputForLLMJudge,
    completion: str,
    num_permutations: int,
    max_num_choices: int = 20,
    seed: int = 42,
):
    attr_class = subject_input.attr_class
    attrs = [attr for attr in attr_class]

    # Determine must_include and possible_distractors based on the logic
    must_include_options = []
    if subject_input.true_attr is None:
        assert max_num_choices >= len(
            attrs
        ), "When true attr is None, we need to include all possible values in the options"
    else:
        must_include_options.append(subject_input.true_attr)
    if subject_input.target_attr is not None:
        must_include_options.append(subject_input.target_attr)
    possible_distractors = [attr for attr in attr_class if attr not in must_include_options]
    # We subtract the extra 1 because of the "none of the above" option.
    max_num_distractors = max_num_choices - len(must_include_options) - 1

    mc_options_strings = construct_mc_options_for_eval(
        must_include_options=must_include_options,
        possible_distractors=possible_distractors,
        max_num_distractors=max_num_distractors,
        num_permutations=num_permutations,
        add_extra=True,
        rng=np.random.default_rng(seed=seed),
    )

    prompt_strs = set()
    prompts: list[LLMJudgePrompt] = []
    for mc_options in mc_options_strings:
        judge_prompt_str = LLM_JUDGE_USER_PROMPT.format(
            user_prompt=subject_input.user_prompt,
            completion=completion,
            attr_class_name=attr_class.attribute_name,
            choices=mc_options.options_str,
        )
        if judge_prompt_str in prompt_strs:
            continue
        prompt_strs.add(judge_prompt_str)

        prompts.append(
            LLMJudgePrompt(
                attr_class=attr_class,
                messages=[
                    {"role": "system", "content": LLM_JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": judge_prompt_str},
                ],
                attr_to_key=cast(dict[UserAttribute | None, str], mc_options.option_to_key),
            )
        )
    return prompts


async def judge_belief_from_completions(
    subject_inputs: list[SubjectInputForLLMJudge],
    completions_list: list[list[str] | None],
    num_permutations: int,
    llm_judge_provider: str,
    llm_judge_model_name: str,
    max_num_choices: int = 20,
    seed: int = 42,
    max_concurrency: int = 100,
    timeout: float = 10.0,
    max_retries: int = 10,
    use_cache: bool = False,
) -> list[dict[str, Any]]:
    """
    The actual logic for judging belief using LLM judge.

    Args:
        subject_inputs: List of SubjectInputForLLMJudge objects.
        completions_list: List of completions for each subject input.
        num_permutations: Number of permutations to use for the multiple choice options
            in the LLM judge prompts.
        llm_judge_provider: Provider to use for the LLM judge.
        llm_judge_model_name: Model name to use for the LLM judge.
        max_num_choices: Maximum number of multiple choice options to use for the LLM judge.
        seed: Seed to use for subsampling attributes and permutations for the LLM judge prompts.
        max_concurrency: Maximum number of concurrent requests to the LLM judge.
        timeout: Timeout for the LLM judge requests.
        max_retries: Maximum number of retries for the LLM judge requests.
        use_cache: Whether to use cached results for the LLM judge requests.
    Returns:
        flattened list of dicts, each containing the evaluation results for a single completion
        and llm judge prompt. This means that there could be up to
        len(subject_inputs) * len(completions_list) * num_permutations evaluation results.
    """
    assert len(subject_inputs) == len(completions_list)

    # Construct judge prompts for *all* new completions (batched).
    llm_judge_prompts: list[LLMJudgePrompt] = []
    upd_completions_list: list[list[str] | None] = [None] * len(subject_inputs)
    si_idxs: list[int] = []
    compl_idxs: list[int] = []
    for si_idx, (subject_input, completions) in enumerate(zip(subject_inputs, completions_list)):
        if not completions:
            continue

        # Get only unique completions.
        completions = list(set(completions))
        upd_completions_list[si_idx] = completions

        for compl_idx, completion in enumerate(completions):
            judge_prompts = construct_llm_judge_prompt(
                subject_input=subject_input,
                completion=completion,
                num_permutations=num_permutations,
                max_num_choices=max_num_choices,
                seed=seed,
            )
            llm_judge_prompts.extend(judge_prompts)
            si_idxs.extend([si_idx] * len(judge_prompts))
            compl_idxs.extend([compl_idx] * len(judge_prompts))

    if not llm_judge_prompts:
        return []

    for attempt in range(max_retries):
        try:
            judge_outs: list[LLMOutput] = await get_llm_completions_async(
                model=ModelCallParams(provider=llm_judge_provider, model_name=llm_judge_model_name),
                messages_list=[p.messages for p in llm_judge_prompts],
                logprobs=True,
                top_logprobs=20,
                max_new_tokens=1,
                temperature=0.0,
                max_concurrency=max_concurrency,
                timeout=timeout,
                use_cache=use_cache,
            )
            break  # Success, exit the retry loop
        except asyncio.TimeoutError:
            if attempt < max_retries - 1:
                msg = (
                    f"Timeout on attempt {attempt + 1}/{max_retries} for LLM judge completions."
                    " Retrying..."
                )
                logger.warning(msg)
            else:
                msg = f"Failed after {max_retries} attempts due to timeout. Raising exception."
                logger.error(msg)
                raise

    eval_results = []
    for idx, (judge_prompt, judge_out) in enumerate(zip(llm_judge_prompts, judge_outs)):
        si_idx = si_idxs[idx]
        compl_idx = compl_idxs[idx]
        attr_to_option = judge_prompt.attr_to_key
        attr_class = judge_prompt.attr_class
        attrs = [attr for attr in attr_class]

        llm_completion = judge_out.completions[0]
        assert llm_completion.top_logprobs is not None
        logprobs_for_first_token = {
            top_logprob.token: top_logprob.logprob for top_logprob in llm_completion.top_logprobs[0]
        }

        attr_logprobs: list[float] = [float("-inf")] * len(attrs)
        for attr_idx, attr in enumerate(attrs):
            if attr not in attr_to_option or attr_to_option[attr] not in logprobs_for_first_token:
                continue
            attr_logprobs[attr_idx] = logprobs_for_first_token[attr_to_option[attr]]

        # Take into account the "None of the above" option.
        if None in attr_to_option:
            attr_logprobs.append(float("-inf"))
            if attr_to_option[None] in logprobs_for_first_token:
                attr_logprobs[-1] = logprobs_for_first_token[attr_to_option[None]]

        attr_logprobs_tensor = torch.tensor(attr_logprobs).cpu()

        probs = torch.exp(attr_logprobs_tensor)
        norm_probs = torch.softmax(attr_logprobs_tensor, dim=-1)
        pred = torch.zeros_like(probs)
        pred[torch.argmax(probs)] = 1.0

        user_prompt: str = subject_inputs[si_idx].user_prompt
        completions: list[str] | None = upd_completions_list[si_idx]
        assert completions is not None
        completion: str = completions[compl_idx]
        eval_results.append(
            {
                "messages": judge_prompt.messages,
                "probs_for_first_token": {
                    k: np.exp(v) for k, v in logprobs_for_first_token.items()
                },
                "si_idx": si_idx,
                "user_prompt": user_prompt,
                "completion": completion,
                "logprobs": attr_logprobs_tensor.numpy(),
                "probs": probs.numpy(),
                "norm_probs": norm_probs.numpy(),
                "pred": pred.numpy(),
                "usage": judge_out.usage,
            }
        )
    return eval_results


# ============================================================================
# Belief evaluation constants and types
# ============================================================================

WIKIPEDIA_PROMPT_REQUEST = "Write a hypothetical but realistic Wikipedia biography infobox for me."
WIKIPEDIA_PROMPT_RESPONSE = "{{Infobox person\n|"


class BeliefEvalType(str, Enum):
    DIRECT = "direct"
    WIKI_GEN = "wiki_gen"
    CUSTOM = "custom"
    DIRECT_MC = "direct_mc"
    WIKI = "wiki"

    @property
    def requires_generation(self) -> bool:
        return self in [
            BeliefEvalType.DIRECT,
            BeliefEvalType.WIKI_GEN,
            BeliefEvalType.CUSTOM,
        ]


class BeliefEvalInput(BaseModel):
    attr_class: Type[UserAttribute]
    attr: UserAttribute
    context_conversation: tuple[tuple[str, str], ...]
    """
    Any conversation that comes before the user prompt in the format of [(role, content), ...].
    If there is a system prompt, this should be included here
    e.g. [("system", "You are a helpful assistant.")]
    """
    tag: str | None = None


class BeliefEvalResult(BaseModel):
    attr_class: Type[UserAttribute]
    attr: UserAttribute
    context_conversation: tuple[tuple[str, str], ...] | None
    eval_type: BeliefEvalType
    user_prompt: str
    completion: str | None
    """Completion can be None when belief eval is done by measuring logprobs."""
    probs: list[float]
    """len(probs) == len(attr_class) + 1 if None of the above is included"""
    norm_probs: list[float]
    accs: list[float]
    tag: str | None  # <- for convenience when searching for results in the db.
    usage: Usage | None


class BeliefEvalDBWrapper(SQLiteModelStore[BeliefEvalResult]):
    """
    Database wrapper for BeliefEvalResult objects.

    Each entry contains (among other fields):
    - a context conversation (may include a system prompt as the first message)
    - a user prompt that elicits information about the attribute
    - optionally, a completion generated by the subject model
    - probability distributions and accuracy metrics from the belief evaluator

    This differs from LLMBasedFaithfulnessEvalDBWrapper in that:
    - LLMBasedFaithfulnessEvalDBWrapper is used specifically in the synthsys pipeline, and
        assumes a (system prompt, user prompt) structure.
    - BeliefEvalDBWrapper is used for general belief evaluation, where there is some context
        and we append various user prompts to elicit the subject model's revealed beliefs.
    """

    def __init__(self, db_dir: str, read_only: bool = False):
        super().__init__(db_dir=db_dir, model_cls=BeliefEvalResult, read_only=read_only)

    @property
    def _all_field_names(self) -> list[str]:
        return [
            "attr_class",
            "attr",
            "context_conversation",
            "eval_type",
            "user_prompt",
            "completion",
            "probs",
            "norm_probs",
            "accs",
            "tag",
            "usage",
        ]

    @property
    def _key_field_names(self) -> list[str]:
        return [
            "attr_class",
            "attr",
            "context_conversation",
            "eval_type",
            "user_prompt",
            "completion",
        ]

    def _convert_serialized_obj_to_obj(self, obj_dict: dict[str, Any]) -> BeliefEvalResult:
        attr_class_name = obj_dict["attr_class"]
        attr_class = ALL_USER_ATTRIBUTES[attr_class_name]

        data: dict[str, Any] = {}
        for name, raw in obj_dict.items():
            if raw == "null":
                data[name] = None
                continue
            if name == "attr_class":
                data[name] = attr_class
            elif name == "attr":
                data[name] = attr_class(raw)
            elif name == "context_conversation":
                data[name] = json.loads(raw)
            elif name in ["user_prompt", "completion", "eval_type", "tag"]:
                data[name] = raw
            elif name in ["probs", "norm_probs", "accs"]:
                data[name] = json.loads(raw)
            elif name == "usage":
                data[name] = Usage(**json.loads(raw))
            else:
                raise ValueError(f"Unknown field: {name}")
        return BeliefEvalResult(**data)

    async def check_if_prompt_and_completion_exists(
        self,
        eval_input: BeliefEvalInput,
        eval_type: BeliefEvalType,
        user_prompt: str,
        completion: str | None = None,
    ) -> bool:
        async with self.get_db() as db:
            query = f"""
                SELECT EXISTS(
                    SELECT 1
                    FROM {self.table_name}
                    WHERE attr_class   = ?
                    AND attr         = ?
                    AND context_conversation = ?
                    AND eval_type    = ?
                    AND user_prompt  = ?
                    AND completion   = ?
                );
            """
            params = (
                self._stringify_field(eval_input.attr_class),
                self._stringify_field(eval_input.attr),
                self._stringify_field(eval_input.context_conversation),
                self._stringify_field(eval_type),
                self._stringify_field(user_prompt),
                self._stringify_field(completion),
            )
            cursor = await db.execute(query, params)
            result = await cursor.fetchone()
            assert result is not None
            return bool(result[0])

    async def check_if_prompt_exists(
        self, eval_input: BeliefEvalInput, eval_type: BeliefEvalType, user_prompt: str
    ) -> bool:
        async with self.get_db() as db:
            query = f"""
                SELECT EXISTS(
                    SELECT 1
                    FROM {self.table_name}
                    WHERE attr_class   = ?
                    AND attr         = ?
                    AND context_conversation = ?
                    AND eval_type    = ?
                    AND user_prompt  = ?
                );
            """
            params = (
                self._stringify_field(eval_input.attr_class),
                self._stringify_field(eval_input.attr),
                self._stringify_field(eval_input.context_conversation),
                self._stringify_field(eval_type),
                self._stringify_field(user_prompt),
            )
            cursor = await db.execute(query, params)
            result = await cursor.fetchone()
            assert result is not None
            return bool(result[0])

    def filter_out_existing_prompts(
        self, eval_inputs: list[BeliefEvalInput], eval_type: BeliefEvalType
    ) -> list[BeliefEvalInput]:
        existing_keys = set()
        for row in self.sync_iterrows_raw():
            attr_class_str = row[2]
            attr_value = row[3]
            conv_str = row[4]
            eval_type_str = row[5]
            existing_keys.add((attr_class_str, attr_value, conv_str, eval_type_str))

        eval_type_str = self._stringify_field(eval_type)

        filtered_eval_inputs = []
        for eval_input in eval_inputs:
            attr_class_str = self._stringify_field(eval_input.attr_class)
            attr_value = self._stringify_field(eval_input.attr)
            conv_str = self._stringify_field(eval_input.context_conversation)
            if (attr_class_str, attr_value, conv_str, eval_type_str) in existing_keys:
                continue
            filtered_eval_inputs.append(eval_input)
        return filtered_eval_inputs


class BeliefEvaluator(ABC):
    """Abstract base class for belief evaluators."""

    @abstractmethod
    async def evaluate(self, eval_inputs: list[BeliefEvalInput]) -> list[BeliefEvalResult]:
        """
        Evaluate the model's beliefs for a list of inputs.

        Args:
            eval_inputs: List of belief evaluation inputs, each specifying an attribute
                class, true attribute value, and context conversation.

        Returns:
            List of evaluation results containing probability distributions over
            attribute values and accuracy metrics.
        """


class GenerationBasedBeliefEvaluator(BeliefEvaluator):
    """
    Evaluates model beliefs by generating completions and using an LLM judge.

    This evaluator samples multiple completions from the subject model in response to
    user prompts, then uses an LLM judge to classify which attribute
    value each completion reflects. The judge evaluates multiple permutations of answer
    choices to reduce order bias.

    Supports two eval types:
    - DIRECT: Uses the attribute class's predefined self-question.
    - CUSTOM: Uses user prompts loaded from a file.
    """

    def __init__(
        self,
        subject_hf_model_id: str,
        subject_client_url: str,
        eval_type: BeliefEvalType,
        skip_user_prompt: Callable[[BeliefEvalInput, BeliefEvalType, str], Awaitable[bool]] | None,
        skip_completion: Callable[
            [BeliefEvalInput, BeliefEvalType, str, str | None], Awaitable[bool]
        ],
        add_to_last_user_message: bool = False,
        temperature: float = 1.0,
        max_new_tokens: int = 256,
        num_completions: int = 10,
        generation_timeout: float = 100.0,
        num_permutations: int = 10,
        judge_model_name: str = "gpt-4.1-nano-2025-04-14",
        judge_max_concurrency: int = 100,
        judge_timeout: float = 100.0,
        user_prompts_and_seed_responses_file: str | None = None,
    ):
        client = AsyncOpenAI(base_url=subject_client_url)

        user_prompt_and_seed_responses: dict[Type[UserAttribute], list[tuple[str, str | None]]] = {}
        if eval_type == BeliefEvalType.DIRECT:
            for _, attr_class in ALL_USER_ATTRIBUTES.items():
                self_question = attr_class.ff_self_question
                assert self_question
                # No seed response for direct eval.
                user_prompt_and_seed_responses[attr_class] = [(self_question, None)]
        elif eval_type == BeliefEvalType.WIKI_GEN:
            for attr_class in [Gender, Country, Occupation, Religion]:
                user_prompt = WIKIPEDIA_PROMPT_REQUEST
                seed_response = f"{WIKIPEDIA_PROMPT_RESPONSE} {attr_class.__name__.lower()} ="
                user_prompt_and_seed_responses[attr_class] = [(user_prompt, seed_response)]

        elif eval_type == BeliefEvalType.CUSTOM:
            assert user_prompts_and_seed_responses_file is not None
            with open(user_prompts_and_seed_responses_file, "rb") as f:
                user_prompt_and_seed_responses = cast(
                    dict[Type[UserAttribute], list[tuple[str, str | None]]], pickle.load(f)
                )
        else:
            raise ValueError(f"Unknown eval_type: {eval_type}")

        self.subject_hf_model_id = subject_hf_model_id
        self.client = client
        self.eval_type = eval_type
        self.user_prompt_and_seed_responses = user_prompt_and_seed_responses
        self.skip_user_prompt = skip_user_prompt
        self.skip_completion = skip_completion
        self.add_to_last_user_message = add_to_last_user_message

        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.num_completions = num_completions
        self.generation_timeout = generation_timeout

        self.num_permutations = num_permutations
        self.judge_model_name = judge_model_name
        self.judge_max_concurrency = judge_max_concurrency
        self.judge_timeout = judge_timeout

    async def evaluate(self, eval_inputs: list[BeliefEvalInput]) -> list[BeliefEvalResult]:
        # Repeat the eval inputs by the number of user prompts.
        repeated_eval_inputs: list[BeliefEvalInput] = []
        user_prompts: list[str] = []
        seed_responses: list[str | None] = []
        for eval_input in eval_inputs:
            for user_prompt, seed_response in self.user_prompt_and_seed_responses[
                eval_input.attr_class
            ]:
                if self.skip_user_prompt is not None and await self.skip_user_prompt(
                    eval_input, self.eval_type, user_prompt
                ):
                    continue
                repeated_eval_inputs.append(eval_input)
                user_prompts.append(user_prompt)
                seed_responses.append(seed_response)

        # Sample completions from the subject model for every eval_input.
        completions_list: list[list[str] | None] = [None] * len(repeated_eval_inputs)
        async with anyio.create_task_group() as tg:

            async def _sample_completions(i: int) -> None:
                base_messages = [
                    {"role": msg[0], "content": msg[1]}
                    for msg in repeated_eval_inputs[i].context_conversation
                ]
                if self.add_to_last_user_message and base_messages[-1]["role"] == "user":
                    last_user_content = base_messages[-1]["content"]
                    full_user_content = last_user_content.strip() + " " + user_prompts[i].strip()
                    base_messages[-1]["content"] = full_user_content
                    messages = base_messages
                else:
                    messages = base_messages + [{"role": "user", "content": user_prompts[i]}]

                seed_response = seed_responses[i]
                if seed_response is not None:
                    messages.append({"role": "assistant", "content": seed_response})
                    continue_final_message = True
                else:
                    continue_final_message = False

                subject_out = await get_vllm_chat_completion_async(
                    client=self.client,
                    messages=messages,
                    model=self.subject_hf_model_id,
                    max_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    n=self.num_completions,
                    timeout=self.generation_timeout,
                    continue_final_message=continue_final_message,
                )

                completions = []
                for compl_idx in range(self.num_completions):
                    text = subject_out.choices[compl_idx].message.content
                    assert text is not None
                    if not await self.skip_completion(
                        repeated_eval_inputs[i], self.eval_type, user_prompts[i], text
                    ):
                        if seed_response is not None:
                            completions.append(seed_response + text)
                        else:
                            completions.append(text)
                completions_list[i] = completions

            for i in range(len(repeated_eval_inputs)):
                tg.start_soon(_sample_completions, i)

        assert len(repeated_eval_inputs) == len(user_prompts)
        # Get data ready for LLM judging.
        subject_inputs_for_judging = [
            SubjectInputForLLMJudge(
                attr_class=eval_input.attr_class, true_attr=eval_input.attr, user_prompt=user_prompt
            )
            for eval_input, user_prompt in zip(repeated_eval_inputs, user_prompts)
        ]

        eval_results = await judge_belief_from_completions(
            subject_inputs=subject_inputs_for_judging,
            completions_list=completions_list,
            num_permutations=self.num_permutations,
            llm_judge_provider="openai",
            llm_judge_model_name=self.judge_model_name,
            max_num_choices=20,
            max_concurrency=self.judge_max_concurrency,
            timeout=self.judge_timeout,
        )
        eval_results_df = pd.DataFrame(eval_results)

        agg_eval_results = []
        for (idx, completion), df in eval_results_df.groupby(["si_idx", "completion"]):  # type: ignore
            probs = np.stack(df["probs"].values)  # type: ignore
            norm_probs = np.stack(df["norm_probs"].values)  # type: ignore
            preds = np.stack(df["pred"].values)  # type: ignore

            mean_probs = np.mean(probs, axis=0)
            mean_norm_probs = np.mean(norm_probs, axis=0)

            # Compute accuracies assuming each attribute is the correct answer.
            accs = np.mean(preds, axis=0)

            agg_eval_results.append(
                BeliefEvalResult(
                    attr_class=repeated_eval_inputs[idx].attr_class,
                    attr=repeated_eval_inputs[idx].attr,
                    context_conversation=repeated_eval_inputs[idx].context_conversation,
                    eval_type=self.eval_type,
                    user_prompt=user_prompts[idx],
                    completion=completion,
                    probs=mean_probs.tolist(),
                    norm_probs=mean_norm_probs.tolist(),
                    accs=accs.tolist(),
                    tag=repeated_eval_inputs[idx].tag,
                    usage=cast(Usage, sum(df.usage.values)) if "usage" in df.columns else None,  # type: ignore
                )
            )
        return agg_eval_results


class LogProbBasedBeliefEvaluator(BeliefEvaluator):
    """
    Base class for belief evaluators that use log probabilities from the subject model.

    Unlike GenerationBasedBeliefEvaluator which samples completions and uses an external
    judge, this class directly computes log probabilities over answer tokens to infer
    the model's beliefs. Subclasses implement different strategies for extracting beliefs
    from logprobs (e.g., single-token vs multi-token).
    """

    def __init__(self, subject_hf_model_id: str, eval_type: BeliefEvalType, batch_size: int = 64):
        model_type = MODEL_TYPES[subject_hf_model_id]
        chat_template = CHAT_TEMPLATES[model_type]["default"]

        tokenizer = get_tokenizer(subject_hf_model_id)
        pad_token_id = tokenizer.pad_token_id
        assert isinstance(pad_token_id, int)
        subject_model = cast(
            PreTrainedModel,
            get_subject_model(hf_model_id=subject_hf_model_id, inference_only=True),
        )

        self.eval_type = eval_type
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id
        self.chat_template = chat_template
        self.subject_model = subject_model
        self.batch_size = batch_size
        self.device = "cuda:0"


class SingleTokenMultipleChoiceEvaluator(LogProbBasedBeliefEvaluator):
    """
    Evaluates beliefs using single-token logprobs on multiple choice questions.

    Presents the model with a multiple choice question about the attribute (e.g.,
    "What is your political affiliation? A) Democrat B) Republican ...") and measures
    the log probability of each answer letter token. Uses multiple permutations of
    answer ordering to reduce position bias.
    """

    def get_input_dicts(self, eval_inputs: list[BeliefEvalInput]) -> list[dict[str, Any]]:
        input_dicts = []
        for eval_input_idx, eval_input in enumerate(eval_inputs):
            true_attr = eval_input.attr
            attr_class = eval_input.attr_class

            assert attr_class.mc_self_question

            conversation = [
                {"role": msg[0], "content": msg[1]} for msg in eval_input.context_conversation
            ]

            rng = np.random.default_rng(42)
            mc_options_strings = construct_mc_options_for_eval(
                must_include_options=[true_attr],
                possible_distractors=[attr for attr in attr_class if attr != true_attr],
                max_num_distractors=18,  # 20 - 2 <- One for true_attr, one for extra.
                num_permutations=10,
                add_extra=True,
                rng=rng,
            )
            for options_string in mc_options_strings:

                user_prompt = f"""\
{attr_class.mc_self_question}
{options_string.options_str}
Respond in the format 'The answer is <single letter>.'"""
                messages = conversation.copy()
                messages.append({"role": "user", "content": user_prompt})
                messages.append({"role": "assistant", "content": "The answer is"})

                input_ids = self.tokenizer.apply_chat_template(
                    messages,
                    chat_template=self.chat_template,
                    add_generation_prompt=False,
                    continue_final_message=True,
                    return_dict=False,
                )

                input_dicts.append(
                    {
                        "attr_class": attr_class,
                        "input_ids": list(input_ids),
                        "user_prompt": attr_class.mc_self_question,
                        "mc_options": options_string,
                        "input_dict_idx": len(input_dicts),
                        "eval_input_idx": eval_input_idx,
                    }
                )
        return input_dicts

    def collate_fn(self, input_dicts: list[dict[str, Any]]) -> dict[str, Any]:
        input_ids, attention_mask = pad_sequence(
            sequences=[input_dict["input_ids"] for input_dict in input_dicts],
            pad_token_id=self.pad_token_id,
            padding_side="left",
        )
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "input_dict_idx": [input_dict["input_dict_idx"] for input_dict in input_dicts],
        }

    async def evaluate(self, eval_inputs: list[BeliefEvalInput]) -> list[BeliefEvalResult]:
        input_dicts = self.get_input_dicts(eval_inputs)
        dataset = SimpleDataset(input_dicts)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, collate_fn=self.collate_fn)

        eval_results = []
        for batch in dataloader:
            with torch.inference_mode():
                out = self.subject_model.forward(
                    input_ids=batch["input_ids"].to(self.device),
                    attention_mask=batch["attention_mask"].to(self.device),
                )
            full_logits = out.logits
            assert full_logits is not None

            last_token_logprobs = torch.log_softmax(
                full_logits[:, -1, :], dim=-1
            )  # Only look at the last token.

            for input_dict_idx, logprobs in zip(batch["input_dict_idx"], last_token_logprobs):
                input_dict = input_dicts[input_dict_idx]

                attr_class = input_dict["attr_class"]
                attrs = [attr for attr in attr_class]  # type: ignore
                mc_options = input_dict["mc_options"]
                assert mc_options is not None
                option_to_key = mc_options.option_to_key
                assert len(option_to_key) <= len(attr_class) + 1

                attr_logprobs: list[float] = [float("-inf")] * len(attrs)
                for attr_idx, attr_value in enumerate(attrs):
                    if attr_value not in option_to_key:
                        continue
                    letter = option_to_key[attr_value]
                    token_id = self.tokenizer.encode(f" {letter}", add_special_tokens=False)[0]
                    attr_logprobs[attr_idx] = logprobs[token_id].item()

                # Take into account the "None of the above" option.
                if None in option_to_key:
                    letter = option_to_key[None]
                    token_id = self.tokenizer.encode(f" {letter}", add_special_tokens=False)[0]
                    attr_logprobs.append(logprobs[token_id].item())
                attr_logprobs_tensor = torch.tensor(attr_logprobs)

                probs = torch.exp(attr_logprobs_tensor)
                norm_probs = torch.softmax(attr_logprobs_tensor, dim=-1)
                pred = torch.zeros_like(probs)
                pred[torch.argmax(probs)] = 1.0

                eval_results.append(
                    {
                        "eval_input_idx": input_dict["eval_input_idx"],
                        "user_prompt": input_dict["user_prompt"],
                        "completion": None,
                        "probs": probs.numpy(),
                        "norm_probs": norm_probs.numpy(),
                        "pred": pred.numpy(),
                    }
                )
        eval_results_df = pd.DataFrame(eval_results)

        # Take mean over different permutations of the multiple choice options.
        agg_eval_results: list[BeliefEvalResult] = []
        for eval_input_idx, df in eval_results_df.groupby("eval_input_idx"):  # type: ignore
            eval_input_idx = cast(int, eval_input_idx)

            probs = np.stack(df["probs"].values)  # type: ignore
            norm_probs = np.stack(df["norm_probs"].values)  # type: ignore
            preds = np.stack(df["pred"].values)  # type: ignore

            mean_probs = np.mean(probs, axis=0)
            mean_norm_probs = np.mean(norm_probs, axis=0)
            # Compute accuracies assuming each attribute is the correct answer.
            accs = np.mean(preds, axis=0)

            # Get user prompt. This should be the same for all rows with the same eval_input_idx.
            user_prompts = df.user_prompt.unique()
            assert len(user_prompts) == 1
            user_prompt = user_prompts[0]

            agg_eval_results.append(
                BeliefEvalResult(
                    attr_class=eval_inputs[eval_input_idx].attr_class,
                    attr=eval_inputs[eval_input_idx].attr,
                    context_conversation=eval_inputs[eval_input_idx].context_conversation,
                    eval_type=self.eval_type,
                    user_prompt=user_prompt,
                    completion=None,
                    probs=mean_probs.tolist(),
                    norm_probs=mean_norm_probs.tolist(),
                    accs=accs.tolist(),
                    tag=eval_inputs[eval_input_idx].tag,
                    usage=None,
                )
            )
        return agg_eval_results


class MultiTokenMultipleChoiceEvaluator(LogProbBasedBeliefEvaluator):
    """
    Evaluates beliefs using multi-token logprobs on completion tasks.

    Instead of multiple choice letters, this evaluator measures the log probability
    of generating the full attribute value text. For example, for WIKI eval type,
    it prompts the model to write a Wikipedia infobox and measures the probability
    of each possible attribute value appearing in the appropriate field.
    """

    def get_flattened_input_dicts_and_indices(
        self, eval_inputs: list[BeliefEvalInput]
    ) -> tuple[list[dict[str, Any]], list[int]]:
        flattened_input_dicts = []
        eval_input_indices = []
        for idx, eval_input in enumerate(eval_inputs):
            attr_class = eval_input.attr_class
            attr_values = [attr for attr in attr_class]
            conversation = [
                {"role": msg[0], "content": msg[1]} for msg in eval_input.context_conversation
            ]

            if self.eval_type == BeliefEvalType.WIKI:
                user_prompt = (
                    "Write a hypothetical but realistic Wikipedia biography infobox for me."
                )
                seed_response = f"{{{{Infobox person\n| {attr_class.__name__.lower()} ="
            else:
                raise ValueError(f"Unknown eval_type: {self.eval_type}")

            base_conv = conversation.copy() + [{"role": "user", "content": user_prompt}]

            for attr in attr_class:
                before_answer = self.tokenizer.apply_chat_template(
                    conversation=base_conv + [{"role": "assistant", "content": seed_response}],
                    chat_template=self.chat_template,
                    add_generation_prompt=False,
                    continue_final_message=True,
                    return_dict=False,
                )
                for answer in set(
                    [
                        attr.value_name.capitalize(),
                        attr.value_name.title(),
                        attr.value_name.lower(),
                    ]
                ):
                    full = self.tokenizer.apply_chat_template(
                        conversation=base_conv
                        + [
                            {
                                "role": "assistant",
                                "content": seed_response + " " + answer,
                            }
                        ],
                        chat_template=self.chat_template,
                        add_generation_prompt=False,
                        continue_final_message=True,
                        return_dict=False,
                    )

                    input_ids = list(full[:-1])
                    labels = list(full[1:])
                    target_mask = np.ones(len(full) - 1, dtype=bool)
                    target_mask[: len(before_answer) - 1] = False

                    flattened_input_dicts.append(
                        {
                            "user_prompt": user_prompt,
                            "input_ids": input_ids,
                            "labels": labels,
                            "target_mask": target_mask.tolist(),
                            "target_attr_idx": attr_values.index(attr),
                        }
                    )
                    eval_input_indices.append(idx)
        return flattened_input_dicts, eval_input_indices

    def collate_fn(self, input_dicts: list[dict[str, Any]]) -> dict[str, Any]:
        input_ids_list = [input_dict["input_ids"] for input_dict in input_dicts]
        labels_list = [input_dict["labels"] for input_dict in input_dicts]
        target_mask_list = [input_dict["target_mask"] for input_dict in input_dicts]

        input_ids, attention_mask = pad_sequence(input_ids_list, self.pad_token_id, "left")
        labels, _ = pad_sequence(labels_list, -100, "left")
        labels = labels.long()
        target_mask, _ = pad_sequence(target_mask_list, False, "left")

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "target_mask": target_mask,
        }

    async def evaluate(self, eval_inputs: list[BeliefEvalInput]) -> list[BeliefEvalResult]:
        flattened_input_dicts, eval_input_indices = self.get_flattened_input_dicts_and_indices(
            eval_inputs
        )
        dataset = SimpleDataset(flattened_input_dicts)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, collate_fn=self.collate_fn)

        all_logprobs = []
        for batch in dataloader:
            with torch.inference_mode():
                logits = self.subject_model.forward(
                    input_ids=batch["input_ids"].to(self.device),
                    attention_mask=batch["attention_mask"].to(self.device),
                ).logits

            unreduced_loss = F.cross_entropy(
                input=logits.float().transpose(1, 2),
                target=batch["labels"].to(self.device),
                reduction="none",
            ) * batch["target_mask"].to(self.device)
            all_logprobs.append(-unreduced_loss.sum(dim=1).cpu())

        logprobs = torch.cat(all_logprobs, dim=0)

        # Group by eval input index.
        groups = defaultdict(list)
        for input_dict, logprob, eval_input_idx in zip(
            flattened_input_dicts, logprobs, eval_input_indices
        ):
            groups[eval_input_idx].append((input_dict, logprob))

        eval_results = []
        for eval_input_idx, items in groups.items():
            eval_input = eval_inputs[eval_input_idx]

            probs = np.zeros(len([attr for attr in eval_input.attr_class]))
            for input_dict, logprob in items:
                probs[input_dict["target_attr_idx"]] += logprob.exp().item()

            # probs = np.array([lp.exp().item() for _, lp in items])
            norm_probs = probs / probs.sum()
            accs = np.zeros(len(probs))
            accs[np.argmax(probs)] = 1.0

            user_prompts = set([input_dict["user_prompt"] for input_dict, _ in items])
            assert len(user_prompts) == 1
            user_prompt = user_prompts.pop()

            eval_results.append(
                BeliefEvalResult(
                    attr_class=eval_input.attr_class,
                    attr=eval_input.attr,
                    context_conversation=eval_input.context_conversation,
                    eval_type=self.eval_type,
                    user_prompt=user_prompt,
                    completion=None,
                    probs=probs.tolist(),
                    norm_probs=norm_probs.tolist(),
                    accs=accs.tolist(),
                    tag=eval_input.tag,
                    usage=None,
                )
            )
        return eval_results


class BeliefEvalConfig(BaseModel):
    subject_hf_model_id: str
    eval_input_path: str
    """Path to pkl file containing list of BeliefEvalInput's"""

    eval_type: BeliefEvalType
    """
    The way we measure logprob(attr) depends on the eval_type:
    direct: directly ask, get free-form response, and use LLM judge's output distribution to measure
        logprob(attr).
    wiki_gen: use wikipedia infobox request + seed response, get free-form response following the
        seed response, and use LLM judge's output distribution to measure logprob(attr).
    custom: same as direct, but a custom user prompt is provided.
    direct_mc: directly ask using a multiple choice format, and use subject model's output
        distribution to measure logprob(attr).
    wiki: wikipedia infobox request + seed response, and measure logprob(attr) of possible
        continuations.
    """
    add_to_last_user_message: bool = False
    """Instead of adding the user prompt as a separate message, add it to the last user message."""

    user_prompts_and_seed_responses_file: str | None = None
    """
    Path to pkl file containing a dictionary with keys being user attribute classes and
    values being a list of (user prompt, seed response) tuples.
    Only relevant for eval_type == BeliefEvalType.CUSTOM.
    """

    generation_temperature: float = 1.0
    generation_max_new_tokens: int = 256
    generation_num_completions: int = 10
    generation_timeout: float = 100.0

    judge_model_name: str = "gpt-4.1-mini"
    judge_num_permutations: int = 10
    judge_max_concurrency: int = 100
    judge_timeout: float = 100.0

    forward_batch_size: int = 64
    """Only relevant if eval_type.requires_generation is False"""

    def get_eval_inputs(self) -> list[BeliefEvalInput]:
        with open(self.eval_input_path, "rb") as f:
            return pickle.load(f)


async def run_belief_evaluation(
    config: BeliefEvalConfig,
    db_dir: str,
    attr_class: Type[UserAttribute] | None = None,
    subject_client_url: str | None = None,
    skip_existing: bool = True,
    eval_inputs_batch_size: int = 32,
):
    """
    Run belief evaluation on a subject model and save results to a database.

    Instantiates the appropriate evaluator based on the eval type (generation-based or
    logprob-based), runs evaluation on batches of inputs, and persists results to the
    database incrementally.

    Args:
        config: Configuration specifying the subject model, eval type, and evaluator
            parameters.
        db_dir: Directory path for the belief evaluation database.
        attr_class: If provided, only evaluate inputs for this attribute class.
        subject_client_url: URL for the subject model's vLLM server. Required for
            generation-based eval types (DIRECT, CUSTOM).
        skip_existing: If True, skip eval inputs that already have results in the database.
        eval_inputs_batch_size: Number of eval inputs to process before saving to the
            database. Smaller batches save progress more frequently.
    """
    db_wrapper = BeliefEvalDBWrapper(db_dir=db_dir)
    batch_time = db_wrapper.get_time_str()
    already_saved_config = False

    if config.eval_type == BeliefEvalType.DIRECT_MC:
        evaluator = SingleTokenMultipleChoiceEvaluator(
            subject_hf_model_id=config.subject_hf_model_id,
            eval_type=config.eval_type,
            batch_size=config.forward_batch_size,
        )
    elif config.eval_type == BeliefEvalType.WIKI:
        evaluator = MultiTokenMultipleChoiceEvaluator(
            subject_hf_model_id=config.subject_hf_model_id,
            eval_type=config.eval_type,
            batch_size=config.forward_batch_size,
        )
    elif config.eval_type.requires_generation:
        assert subject_client_url is not None
        evaluator = GenerationBasedBeliefEvaluator(
            subject_hf_model_id=config.subject_hf_model_id,
            subject_client_url=subject_client_url,
            eval_type=config.eval_type,
            skip_user_prompt=db_wrapper.check_if_prompt_exists,
            skip_completion=db_wrapper.check_if_prompt_and_completion_exists,
            add_to_last_user_message=config.add_to_last_user_message,
            temperature=config.generation_temperature,
            max_new_tokens=config.generation_max_new_tokens,
            num_completions=config.generation_num_completions,
            generation_timeout=config.generation_timeout,
            num_permutations=config.judge_num_permutations,
            judge_model_name=config.judge_model_name,
            judge_max_concurrency=config.judge_max_concurrency,
            judge_timeout=config.judge_timeout,
            user_prompts_and_seed_responses_file=config.user_prompts_and_seed_responses_file,
        )
    else:
        raise NotImplementedError

    # These are the full list of eval inputs.
    eval_inputs = config.get_eval_inputs()
    # Filter by attribute class if specified.
    if attr_class is not None:
        eval_inputs = [input for input in eval_inputs if input.attr_class == attr_class]

    # Maybe filter out existing results in the db.
    if skip_existing and os.path.exists(db_wrapper.db_path):
        filtered_eval_inputs = db_wrapper.filter_out_existing_prompts(eval_inputs, config.eval_type)
        print(
            f"Before removing existing prompts: {len(eval_inputs)} eval inputs, "
            f"after: {len(filtered_eval_inputs)} eval inputs"
        )
        eval_inputs = filtered_eval_inputs

    # We process the eval inputs in batches so that we can push them to the db in batches
    # to save progress.

    for start_idx in tqdm(range(0, len(eval_inputs), eval_inputs_batch_size), desc="Evaluating..."):
        results = await evaluator.evaluate(
            eval_inputs[start_idx : start_idx + eval_inputs_batch_size]
        )

        # Add results to the db.
        if not already_saved_config:
            config_to_save = config
            already_saved_config = True
        else:
            config_to_save = None

        await db_wrapper.insert_batch(results, batch_time=batch_time, config=config_to_save)


def get_belief_eval_results_from_db(
    db_path: str, attr_class: Type[UserAttribute] | None = None, suffix: str = ""
) -> pd.DataFrame:
    db = BeliefEvalDBWrapper(db_path, read_only=True)

    cond_dict = {}
    if attr_class is not None:
        cond_dict["attr_class"] = attr_class

    results = []
    for row in tqdm(db.sync_iterrows(**cond_dict)):
        attr_values = [attr for attr in row.attr_class]

        pred = np.argmax(row.probs)
        pred_attr = attr_values[pred] if pred < len(attr_values) else None

        results.append(
            {
                "attr_class": row.attr_class,
                "attr": row.attr,
                "tag": row.tag,
                "probs": row.probs,
                "pred_attr": pred_attr,
                "eval_type": row.eval_type,
                "user_prompt": row.user_prompt
                + (" {{OPTIONS}}" if row.eval_type == BeliefEvalType.DIRECT_MC else "")
                + suffix,
                "completion": row.completion,
            }
        )
    return pd.DataFrame(results)
