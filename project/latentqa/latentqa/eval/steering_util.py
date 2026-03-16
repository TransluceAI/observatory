from collections import defaultdict
from typing import Callable, Type

import numpy as np
import torch
from latentqa.data_util import MinimalDatapoint, tokenize_subject_input
from latentqa.lqa_data import QA, LQAData
from latentqa.mc_util import MC_TEMPLATE_WITH_SEED_RESPONSE, construct_mc_options_for_eval
from latentqa.model_wrapper import ActivationsList, ModelWrapper, Modules
from latentqa.qas import CompletedQuestion, QACategory
from latentqa.train_util import PatchFromRegion, get_dataloader, get_num_patch_tokens_multiplier
from pydantic import BaseModel
from transformers.tokenization_utils import PreTrainedTokenizer
from user_modeling.user_attribute import UserAttribute


def get_test_mc_qas(
    true_attr: UserAttribute,
    target_attr: UserAttribute,
    stems_by_attr_class: dict[Type[UserAttribute], list[str]],
    rng: np.random.Generator,
    max_num_choices: int = 20,
) -> list[QA]:
    assert true_attr.__class__ == target_attr.__class__
    attr_class = true_attr.__class__
    attrs = [attr for attr in attr_class]

    mc_options_strings = construct_mc_options_for_eval(
        must_include_options=[true_attr, target_attr],
        possible_distractors=[attr for attr in attrs if attr not in [true_attr, target_attr]],
        # We subtract 3 because we have two must-include options and one extra.
        max_num_distractors=max_num_choices - 3,
        num_permutations=10,
        add_extra=True,
        rng=rng,
    )
    test_mc_qas: list[QA] = []
    for options_string in mc_options_strings:
        stem = rng.choice(stems_by_attr_class[attr_class])

        q = MC_TEMPLATE_WITH_SEED_RESPONSE.format(stem=stem, choices=options_string.options_str)
        assert options_string.option_to_key is not None
        a = options_string.option_to_key[target_attr]
        test_mc_qas.append(
            QA(
                attr_class=attr_class,
                attr=target_attr,
                qa_category=QACategory.MC,
                content=CompletedQuestion(question=q),
                seed_response="The answer is",
                answer=a,
                multiple_choice_options=options_string,
            )
        )
    return test_mc_qas


class SteeringInput(BaseModel):
    system_prompt: str | None
    user_prompt: str
    true_attr: UserAttribute
    target_attr: UserAttribute
    steering_qas: tuple[tuple[str, str], ...] | None = None
    validation_qas: tuple[tuple[str, str], ...] | None = None
    # The following two fields are only used if control_eval_type == "fixed".
    subject_seed_response: str | None = None
    expected_subject_responses: list[tuple[UserAttribute, list[str]]] | None = None
    test_qas: tuple[QA, ...]

    @property
    def attr_class(self) -> type[UserAttribute]:
        assert self.true_attr.__class__ == self.target_attr.__class__
        return self.true_attr.__class__

    def tokenize_subject_input(
        self,
        tokenizer: PreTrainedTokenizer,
        subject_chat_template: str,
        patch_from_region: PatchFromRegion | None,
    ) -> tuple[list[int], tuple[int, int]]:
        return tokenize_subject_input(
            tokenizer=tokenizer,
            chat_template=subject_chat_template,
            system_prompt=self.system_prompt,
            user_prompt=self.user_prompt,
            patch_from_region=patch_from_region,
            conversation=None,
            conversation_read_range=None,
        )

    def get_steering_qas_for_dataloader(self, validation: bool = False) -> list[MinimalDatapoint]:
        if validation:
            if self.validation_qas is None:
                raise ValueError("Validation qas are not set.")
            steering_qas = self.validation_qas
        else:
            if self.steering_qas is None:
                raise ValueError("Steering qas are not set.")
            steering_qas = self.steering_qas

        steering_qas = set(steering_qas) - set([(q.question, q.answer) for q in self.test_qas])
        assert len(steering_qas) > 0, "Need at least one steering qa."

        return [
            MinimalDatapoint(
                attr=self.target_attr,
                system_prompt=self.system_prompt,
                user_prompt=self.user_prompt,
                question=question,
                answer=answer,
                id=i,
            )
            for i, (question, answer) in enumerate(steering_qas)
        ]

    def get_test_qas_for_dataloader(self) -> list[LQAData]:
        return [
            LQAData(
                attr_class=self.attr_class,
                attr=self.target_attr,
                system_prompt=self.system_prompt,
                user_prompt=self.user_prompt,
                qa=qa,
                id=i,
            )
            for i, qa in enumerate(self.test_qas)
        ]

    def get_dummy_datapoint(self) -> MinimalDatapoint:
        return MinimalDatapoint(
            attr=self.target_attr,
            system_prompt=self.system_prompt,
            user_prompt=self.user_prompt,
            question="",
            answer="",
        )

    def key(self):
        if self.expected_subject_responses is None:
            flattened_expected_subject_responses = None
        else:
            flattened_expected_subject_responses = tuple(
                (attr,) + tuple(sorted(responses))
                for attr, responses in self.expected_subject_responses
            )
        return (
            self.system_prompt,
            self.user_prompt,
            self.true_attr,
            self.target_attr,
            self.steering_qas,
            self.subject_seed_response,
            flattened_expected_subject_responses,
            self.test_qas,
        )

    def __hash__(self) -> int:
        return hash(self.key())

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SteeringInput):
            return False
        return self.key() == other.key()


def evaluate_reading_using_multiple_choice(
    activations_batch: ActivationsList,
    token_range_from: list[tuple[int, int]],
    data: list[list[LQAData]],  # list of qas for each activation,
    hf_model_id: str,
    decoder: ModelWrapper,
    tokenizer: PreTrainedTokenizer,
    write_modules: Modules,
    layer_mapping: dict[int, list[int]],
    eval_batch_size: int,
    patch_from_region: PatchFromRegion = PatchFromRegion.USER,
    custom_read_range_fn: (
        Callable[[list[dict], PreTrainedTokenizer, str, list[int]], tuple[int, int]] | None
    ) = None,
    device: str = "cuda:0",
):
    N = activations_batch.batch_size()
    assert len(token_range_from) == N
    assert len(data) == N

    # Flatten the batch structure: for each activation set, repeat it for each evaluation data point
    # This creates a flat list where each activation is paired with its corresponding evaluation data
    flattened_repeated_acts_dict = defaultdict(list)
    flattened_repeated_token_range_from: list[tuple[int, int]] = []
    flattened_data: list[LQAData] = []
    for i in range(N):
        for eval_data in data[i]:
            for key, acts_list in activations_batch.items():
                flattened_repeated_acts_dict[key].append(acts_list[i])
            flattened_repeated_token_range_from.append(token_range_from[i])
            flattened_data.append(eval_data)
    flattened_repeated_acts = ActivationsList(data=flattened_repeated_acts_dict)

    eval_dataloader = get_dataloader(
        data=flattened_data,
        hf_model_id=hf_model_id,
        patch_from_regions=[patch_from_region],
        tokenizer=tokenizer,
        generate=True,
        shuffle=False,
        batch_size=eval_batch_size,
        num_patch_tokens_multiplier=get_num_patch_tokens_multiplier(layer_mapping),
        custom_read_range_fn=custom_read_range_fn,
    )

    flat_eval_results = []
    for batch_idx, batch in enumerate(eval_dataloader):
        with torch.inference_mode():
            out = decoder.forward_with_intervention(
                activations=flattened_repeated_acts.batch_slice(
                    batch_idx * eval_batch_size, (batch_idx + 1) * eval_batch_size
                ),
                token_range_from=flattened_repeated_token_range_from[
                    batch_idx * eval_batch_size : (batch_idx + 1) * eval_batch_size
                ],
                token_range_to=batch.token_range_to,
                input_ids=batch.decoder_input_ids,
                attention_mask=batch.decoder_attention_mask,
                write_modules=write_modules,
                layer_mapping=layer_mapping,
                device=device,
            )
        full_logits = out.logits
        assert full_logits is not None

        last_token_logprobs = torch.log_softmax(
            full_logits[:, -1, :], dim=-1
        )  # Only look at the last token.

        def get_token_id(letter: str, with_seed_response) -> int:
            if with_seed_response:
                return tokenizer.encode(f" {letter}", add_special_tokens=False)[0]
            else:
                return tokenizer.encode(letter, add_special_tokens=False)[0]

        for d_id, qa, logprobs in zip(batch.ids, batch.qas, last_token_logprobs):
            mc_options = qa.multiple_choice_options
            assert mc_options is not None
            option_to_key = mc_options.option_to_key
            assert len(option_to_key) <= len(qa.attr_class) + 1

            attr_logprobs: list[float] = [float("-inf") for _ in qa.attr_class]
            for attr_idx, attr_value in enumerate(qa.attr_class):
                if attr_value not in option_to_key:
                    continue
                token_id = get_token_id(
                    option_to_key[attr_value], with_seed_response=qa.seed_response is not None
                )
                attr_logprobs[attr_idx] = logprobs[token_id].item()

            # Take into account the "None of the above" option.
            if None in option_to_key:
                token_id = get_token_id(
                    option_to_key[None], with_seed_response=qa.seed_response is not None
                )
                attr_logprobs.append(logprobs[token_id].item())
            attr_logprobs_tensor = torch.tensor(attr_logprobs)

            preds = attr_logprobs_tensor.argmax()
            corrs = torch.arange(len(attr_logprobs_tensor)) == preds

            norm_logprobs = torch.log_softmax(attr_logprobs_tensor, dim=-1)
            probs = torch.exp(attr_logprobs_tensor)
            norm_probs = torch.softmax(attr_logprobs_tensor, dim=-1)

            eval_result = {
                "id": d_id,
                "accs": corrs.tolist(),
                "probs": probs.tolist(),
                "norm_probs": norm_probs.tolist(),
                "logprobs": attr_logprobs_tensor.tolist(),
                "norm_logprobs": norm_logprobs.tolist(),
            }
            flat_eval_results.append(eval_result)

    # Group flat_eval_results by activation, reconstructing the original nested structure.
    eval_results = []
    curr_flat_idx = 0
    for sublist in data:
        num_items = len(sublist)
        eval_results.append(flat_eval_results[curr_flat_idx : curr_flat_idx + num_items])
        curr_flat_idx += num_items
    return eval_results


class GenerationBasedControlEvalConfig(BaseModel):
    batch_size: int = 128
    max_new_tokens: int = 128
    temperature: float = 1.0
    num_completions: int = 4
    llm_judge_provider: str = "openai"
    llm_judge_model_name: str = "gpt-4.1-mini-2025-04-14"
    llm_judge_max_concurrency: int = 20
    llm_judge_timeout: float = 100.0
