import logging
import os
from enum import Enum
from functools import partial
from typing import Dict, List, Literal, Sequence, Tuple, cast

import numpy as np
import torch
from explanations.few_shot_examples import format_example, get_examples_for_fewshot
from explanations.llama_model import UPD_MAPPING, Llama3Model, Llama3TokenizerWrapper, get_tokenizer
from explanations.scoring_utils import (
    VALID_ACTIVATION_TOKENS_ORDERED,
    ActivationScale,
    SequenceSimulation,
)
from llama_models.llama3.reference_impl.generation import Llama  # type: ignore
from llama_models.llama3.reference_impl.model import Transformer  # type: ignore
from transformers import (  # type: ignore
    LlamaTokenizer,
    LlamaTokenizerFast,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)
from util.huggingface import get_huggingface_model_and_tokenizer, get_huggingface_tokenizer
from util.types import ChatMessage, NDFloatArray

logger = logging.getLogger(__name__)

UNK_SIM_SYS_PROMPT = """\
You are a meticulous AI researcher conducting an important investigation into a specific neuron inside a language model that activates in response to text inputs.

Your overall task is to simulate the activations of a neuron given a sequence of input tokens.
Prior to this, you have studied neuron activation patterns given a large corpus of text and summarized the behavior of a neuron in a sentence or two. Look at the explanation and try to predict how it will fire on each token.

The activation format is token<tab>activation, activations go from 0 to 10, "unknown" indicates an unknown activation. Most activations will be 0.
"""
UNK_NOTAB_SIM_SYS_PROMPT = """\
You are a meticulous AI researcher conducting an important investigation into a specific neuron inside a language model that activates in response to text inputs.

Your overall task is to simulate the activations of a neuron given a sequence of input tokens.
Prior to this, you have studied neuron activation patterns given a large corpus of text and summarized the behavior of a neuron in a sentence or two. Look at the explanation and try to predict how it will fire on each token.

The activation format is {token}{activation}, activations go from 0 to 10, "unknown" indicates an unknown activation. Most activations will be 0.
"""
BLANK_SIM_SYS_PROMPT = """\
You are a meticulous AI researcher conducting an important investigation into a specific neuron inside a language model that activates in response to text inputs.

Your overall task is to simulate the activations of a neuron given a sequence of input tokens.
Prior to this, you have studied neuron activation patterns given a large corpus of text and summarized the behavior of a neuron in a sentence or two. Look at the explanation and try to predict how it will fire on each token.

The activation format is token<tab>activation, activations go from 0 to 10, "blank" is a placeholder for the activation value to be simulated. Most activations will be 0.
"""

LLAMA31_INSTRUCT_PREFIX = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\n<|eot_id|>"

PROMPT_BASED_SIMULATOR_SYSTEM_PROMPT = """\
You are an expert research assistant for language model research. Your job is to predict
the activations of an internal neuron given a description of the tokens that
this neuron activates on.

You will be given two inputs:
- A description of when the target neuron fires,
- A sequence of tokens (and their indices) that will be fed into the language model.

You should repeat the sequence of tokens, and insert annotations for how
strongly you predict the neuron should fire based on the activation. Your output
should be in the form of a Python list of tuples, e.g.

```
[(0, token_0, predicted_activation_0), (1, token_1, predicted_activation_1), ...]
```

where `predicted_activation_i` is the predicted activation of the neuron on
`token_i` as an integer from 0 to 10.
Note: It's very important that in the list of tuples, all tokens are included in the exact same order as they appeared in the input sequence."""

PROMPT_BASED_SIMULATOR_USER_TEMPLATE = """\
Description: {description}
Input sequence:
```
{sequence}
```"""


class SimSysPromptType(str, Enum):
    UNK_BASE = "unk_base"
    UNK_SPECIFIC = "unk_specific"
    UNK_NOTAB_SPECIFIC = "unk_notab_specific"
    BLANK_BASE = "blank_base"
    BLANK_SPECIFIC = "blank_specific"


def get_system_prompt(sys_prompt_type: SimSysPromptType) -> str:
    if sys_prompt_type == SimSysPromptType.UNK_BASE:
        return UNK_SIM_SYS_PROMPT
    elif sys_prompt_type == SimSysPromptType.UNK_SPECIFIC:
        return (
            UNK_SIM_SYS_PROMPT
            + "  Only predict non-zero activations specifically related to the description."
        )
    elif sys_prompt_type == SimSysPromptType.UNK_NOTAB_SPECIFIC:
        return (
            UNK_NOTAB_SIM_SYS_PROMPT
            + "  Only predict non-zero activations specifically related to the description."
        )
    elif sys_prompt_type == SimSysPromptType.BLANK_BASE:
        return BLANK_SIM_SYS_PROMPT
    elif sys_prompt_type == SimSysPromptType.BLANK_SPECIFIC:
        return (
            BLANK_SIM_SYS_PROMPT
            + "  Only predict non-zero activations specifically related to the description."
        )
    else:
        raise ValueError(f"Unknown system prompt type: {sys_prompt_type}!")


def get_simulation_prompt_prefix(
    sys_prompt_type: SimSysPromptType,
    sim_act_token: Literal["unknown", "blank"],
) -> List[ChatMessage]:
    """Returns the system prompt and few-shot example part of the simulation prompt."""
    messages = [ChatMessage(role="system", content=get_system_prompt(sys_prompt_type))]

    # Add few-shot examples to the prompt.
    few_shot_examples = get_examples_for_fewshot()
    for i, example in enumerate(few_shot_examples):
        explanation_str = (
            f"Neuron {i + 1}\nDescription of neuron selectivity: {example.explanation}"
        )
        activations_str = f"Activations:" + format_example(
            example.token_activation_pairs_list,
            example.first_reveal_indices if sim_act_token == "unknown" else None,
        )
        messages.extend(
            [
                ChatMessage(role="user", content=explanation_str),
                ChatMessage(role="assistant", content=activations_str),
            ]
        )
    return messages


def format_tokens_for_simulation(tokens: List[str], sim_act_token: str) -> str:
    """
    Format tokens into a string with each token marked as having an "unknown" activation, suitable
    for use in prompts.
    """
    entries: List[str] = []
    for token in tokens:
        entries.append(f"{token}\t{sim_act_token}")
    return "\n".join(entries)


def format_sequences_for_simulation(all_tokens: List[List[str]], sim_act_token: str) -> str:
    """
    Format a list of lists of tokens into a string with each token marked as having an
    "unknown" or "blank" activation, suitable for use in prompts.
    """
    format_tokens = partial(format_tokens_for_simulation, sim_act_token=sim_act_token)
    return (
        "\n<start>\n"
        + "\n<end>\n<start>\n".join([format_tokens(tokens) for tokens in all_tokens])
        + "\n<end>\n"
    )


def get_simulation_prompt(
    tokens: List[str],
    explanation: str,
    sim_act_token: str,
    num_fewshot: int,
) -> List[ChatMessage]:
    """Returns the part of the simulation prompt that depends on the explanation and sequence."""
    explanation_str = f"Neuron {num_fewshot + 1}\nDescription of neuron selectivity: {explanation}"
    activations_str = f"Activations:" + format_sequences_for_simulation([tokens], sim_act_token)
    messages = [
        ChatMessage(role="user", content=explanation_str),
        ChatMessage(role="assistant", content=activations_str),
    ]
    return messages


# Adapted from tether/tether/core/encoder.py.
def convert_to_byte_array(s: str) -> bytearray:
    byte_array = bytearray()
    assert s.startswith("bytes:"), s
    s = s[6:]
    while len(s) > 0:
        if s[0] == "\\":
            # Hex encoding.
            assert s[1] == "x"
            assert len(s) >= 4
            byte_array.append(int(s[2:4], 16))
            s = s[4:]
        else:
            # Regular ascii encoding.
            byte_array.append(ord(s[0]))
            s = s[1:]
    return byte_array


def handle_byte_encoding(
    response_tokens: Sequence[str], merged_response_index: int
) -> tuple[str, int]:
    """
    Handle the case where the current token is a sequence of bytes. This may involve merging
    multiple response tokens into a single token.
    """
    response_token = response_tokens[merged_response_index]
    if response_token.startswith("bytes:"):
        byte_array = bytearray()
        while True:
            byte_array = convert_to_byte_array(response_token) + byte_array
            try:
                # If we can decode the byte array as utf-8, then we're done.
                response_token = byte_array.decode("utf-8")
                break
            except UnicodeDecodeError:
                # If not, then we need to merge the previous response token into the byte
                # array.
                merged_response_index -= 1
                response_token = response_tokens[merged_response_index]
    return response_token, merged_response_index


def was_token_split(current_token: str, response_tokens: Sequence[str], start_index: int) -> bool:
    """
    Return whether current_token (a token from the subject model) was split into multiple tokens by
    the simulator model (as represented by the tokens in response_tokens). start_index is the index
    in response_tokens at which to begin looking backward to form a complete token. It is usually
    the first token *before* the delimiter that separates the token from the normalized activation,
    barring some unusual cases.

    This mainly happens if the subject model uses a different tokenizer than the simulator model.
    But it can also happen in cases where Unicode characters are split. This function handles both
    cases.
    """
    merged_response_tokens = ""
    merged_response_index = start_index
    while len(merged_response_tokens) < len(current_token):
        response_token = response_tokens[merged_response_index]
        response_token, merged_response_index = handle_byte_encoding(
            response_tokens, merged_response_index
        )
        merged_response_tokens = response_token + merged_response_tokens
        merged_response_index -= 1
    # It's possible that merged_response_tokens is longer than current_token at this point,
    # since the between-lines delimiter may have been merged into the original token. But it
    # should always be the case that merged_response_tokens ends with current_token.
    assert merged_response_tokens.endswith(current_token)
    num_merged_tokens = start_index - merged_response_index
    token_was_split = num_merged_tokens > 1
    if token_was_split:
        logger.debug(
            "Warning: token from the subject model was split into 2+ tokens by the simulator model."
        )
    return token_was_split


def compute_predicted_activation_stats_for_token(
    vocab_logprobs: NDFloatArray,
) -> tuple[NDFloatArray, float]:
    probs = np.exp(vocab_logprobs)
    total_prob = sum(probs)
    norm_probs = probs / total_prob
    expected_value = np.dot(
        np.arange(len(norm_probs)),
        norm_probs,
    )
    return norm_probs, expected_value


def parse_prompt(
    prompt_tokens: List[str],
    prompt_ids: List[int],
    original_sequence_tokens: List[str],
    unknown_tokens: List[str],
    remove_tab: bool = False,
) -> tuple[List[int], List[Tuple[int, bool]]]:
    """
    Finds all locations of "unknown" in the prompt.
    The prompt format is generally like this: token<tab>unknown
    unknown_tokens is a hack for dealing with the case where the actual
    activation is given in the few-shot examples. In this case, unknown_tokens
    unknown_tokens = ["unknown", "0", ..., "10"].

    If remove_tab is True, we return a version of prompt_ids where the
    tab tokens before the unknown token are removed.
    The locations of the unknown tokens are adjusted accordingly.
    """
    sequence_tokens: List[str] = []
    upd_prompt_ids = prompt_ids[:2]
    unk_indices: List[Tuple[int, bool]] = []
    for i in range(2, len(prompt_tokens)):
        upd_prompt_ids.append(prompt_ids[i])
        if prompt_tokens[i - 1] == "\t":
            if prompt_tokens[i] not in unknown_tokens:
                continue

            # Remove tab token.
            upd_prompt_ids.pop(-2)

            # j represents the index of the token in a "token<tab>activation" line, barring
            # one of the unusual cases handled below.
            j = i - 2

            current_token = original_sequence_tokens[len(sequence_tokens)]

            if current_token == prompt_tokens[j] or was_token_split(
                current_token, prompt_tokens, j
            ):
                # We're in the normal case where the tokenization didn't throw off the
                # formatting or in the token-was-split case, which we handle the usual way.

                # Keep in track of the locations of where the activation should go.
                unk_indices.append((len(upd_prompt_ids) - 1 if remove_tab else i, True))
            else:
                # We're in a case where the tokenization resulted in a newline being folded into
                # the token. We can't do our usual prediction of activation stats for the token,
                # since the model did not observe the original token. Instead, we use dummy values.
                newline_folded_into_token = "\n" in prompt_tokens[j]
                assert newline_folded_into_token, f"`{current_token=}` {prompt_tokens[j-3:j+3]=}"
                logger.debug(
                    "Warning: newline before a token<tab>activation line was folded into the token"
                )
                # Keep in track of the locations of where the activation should go.
                unk_indices.append((len(upd_prompt_ids) - 1 if remove_tab else i, False))

            sequence_tokens.append(current_token)
    assert original_sequence_tokens == sequence_tokens
    assert len(sequence_tokens) == len(unk_indices)
    return upd_prompt_ids, unk_indices


class NeuronSimulator:
    """Base class for simulating neuron behavior."""

    def __init__(
        self,
        model: PreTrainedModel | Transformer,
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
        sys_prompt_type: SimSysPromptType,
        remove_tab: bool,
    ):
        valid_activation_token_ids = tokenizer.convert_tokens_to_ids(
            VALID_ACTIVATION_TOKENS_ORDERED
        )

        if sys_prompt_type.startswith("unk"):
            sim_act_token = "unknown"
        elif sys_prompt_type.startswith("blank"):
            sim_act_token = "blank"
        else:
            raise ValueError("Unrecognized system prompt type for simulation!")

        prefix_messages = get_simulation_prompt_prefix(sys_prompt_type, sim_act_token)

        # Tokenize prefix messages.
        # TODO(damichoi): Add support for base models. This isn't a priority since from our
        # earlier tests we found that using a base model is worse than an instruct model.
        prefix_ids: List[int] = tokenizer.apply_chat_template(
            cast(list[dict[str, str]], prefix_messages),
            add_generation_prompt=False,
            return_dict=False,
        )
        if isinstance(tokenizer, LlamaTokenizer) or isinstance(tokenizer, LlamaTokenizerFast):
            default_sys_prompt_ids = tokenizer.encode(
                LLAMA31_INSTRUCT_PREFIX, add_special_tokens=False
            )
        else:
            raise ValueError()

        fewshot_examples = get_examples_for_fewshot()
        # Optionally remove tabs from the prompt prefix.
        if remove_tab:
            example_tokens: List[str] = []
            for example in fewshot_examples:
                for token_activation_pairs in example.token_activation_pairs_list:
                    example_tokens.extend([t for t, _ in token_activation_pairs])

            prefix_ids, _ = parse_prompt(
                [tokenizer.decode(id) for id in prefix_ids],
                prefix_ids,
                original_sequence_tokens=example_tokens,
                unknown_tokens=["unknown"] + VALID_ACTIVATION_TOKENS_ORDERED,
                remove_tab=True,
            )

        prefix_ids_tensor: torch.Tensor = torch.tensor(
            prefix_ids, dtype=torch.long, device="cuda"
        ).unsqueeze(0)

        self.model = model
        self.tokenizer = tokenizer
        self.valid_activation_token_ids = valid_activation_token_ids
        self.sim_act_token = sim_act_token
        self.default_sys_prompt_ids = default_sys_prompt_ids
        self.remove_tab = remove_tab
        self.num_fewshot = len(fewshot_examples)
        self.prefix_cache: Tuple[Tuple[torch.Tensor]] | int = self.setup_kv_cache(prefix_ids_tensor)

    def setup_kv_cache(self, input_ids: torch.Tensor) -> Tuple[Tuple[torch.Tensor]] | int:
        raise NotImplementedError()

    def get_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    def encode_simulation_prompt(self, messages: List[ChatMessage]) -> List[int]:
        """
        Tokenize simulation prompt such that given
        prefix_ids = tokenize(prefix_messages)
        input_ids = tokenize(messages), we have:

        tokenizer.decode(prefix_ids + input_ids) ==
            tokenizer.apply_chat_template(prefix_messages + messages).

        This involves removing the default system prompt message added when calling
        tokenizer.apply_chat_template(messages).
        """
        input_ids: List[int] = self.tokenizer.apply_chat_template(
            cast(List[Dict[str, str]], messages), add_generation_prompt=False, return_dict=False
        )
        if self.default_sys_prompt_ids:
            assert input_ids[: len(self.default_sys_prompt_ids)] == self.default_sys_prompt_ids
            input_ids = input_ids[len(self.default_sys_prompt_ids) :]
        return input_ids

    def simulate(
        self, explanations: List[str], tokens: List[str], token_ids: List[int] | None = None
    ) -> List[SequenceSimulation]:
        """Simulate the behavior of a neuron based on an explanation."""
        results: List[SequenceSimulation] = []
        for explanation in explanations:
            messages = get_simulation_prompt(
                tokens, explanation, self.sim_act_token, self.num_fewshot
            )
            input_ids = self.encode_simulation_prompt(messages)

            # Optionally remove <tab> tokens and get location of unknown tokens.
            upd_input_ids, unk_indices = parse_prompt(
                [self.tokenizer.decode(id) for id in input_ids],
                input_ids,
                original_sequence_tokens=tokens,
                unknown_tokens=["unknown"],
                remove_tab=self.remove_tab,
            )
            if self.remove_tab:
                input_ids = upd_input_ids
            input_ids = torch.tensor(input_ids, dtype=torch.long, device="cuda").unsqueeze(0)

            # Do forward pass and get logits.
            logits = self.get_logits(input_ids[:, :-1])
            response_logits: NDFloatArray = (
                logits[0, :, self.valid_activation_token_ids].cpu().numpy()  # type: ignore
            )

            expected_values: List[float] = []
            distribution_probabilities: list[list[float]] = []
            for idx, valid in unk_indices:
                if valid:
                    norm_probs, expected_value = compute_predicted_activation_stats_for_token(
                        response_logits[idx - 1]
                    )
                    norm_probs_list: List[float] = norm_probs.tolist()
                else:
                    norm_probs_list: List[float] = []
                    expected_value = 0.0
                distribution_probabilities.append(norm_probs_list)
                expected_values.append(expected_value)

            results.append(
                SequenceSimulation(
                    tokens=tokens,
                    expected_activations=expected_values,
                    activation_scale=ActivationScale.SIMULATED_NORMALIZED_ACTIVATIONS,
                    distribution_probabilities=distribution_probabilities,
                )
            )

        return results


class HuggingFaceSimulator(NeuronSimulator):
    """
    Neuron activation simulator using HuggingFace models.
    """

    @staticmethod
    def setup(
        hf_model_id: str,
        sys_prompt_type: SimSysPromptType = SimSysPromptType.UNK_SPECIFIC,
        remove_tab: bool = False,
    ):
        model, tokenizer = get_huggingface_model_and_tokenizer(hf_model_id)
        return HuggingFaceSimulator(model, tokenizer, sys_prompt_type, remove_tab)

    @torch.inference_mode()
    def setup_kv_cache(self, input_ids: torch.Tensor) -> Tuple[Tuple[torch.Tensor]]:
        return self.model(input_ids=input_ids, use_cache=True).past_key_values  # type: ignore

    @torch.inference_mode()
    def get_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids=input_ids, past_key_values=self.prefix_cache).logits  # type: ignore


class LlamaSimulator(NeuronSimulator):
    """
    Neuron activation simulator using HuggingFace models.
    """

    @staticmethod
    def setup(
        model_path: str,
        hf_model_id: str,
        model_parallel_size: int,
        compile: bool = False,
        max_seq_len: int = 10_000,
        sys_prompt_type: SimSysPromptType = SimSysPromptType.UNK_SPECIFIC,
        remove_tab: bool = False,
    ):
        llama: Llama = Llama.build(
            ckpt_dir=model_path,
            tokenizer_path=os.path.join(model_path, "tokenizer.model"),
            max_seq_len=max_seq_len,
            max_batch_size=1,
            model_parallel_size=model_parallel_size,
        )
        if compile:
            llama.model = torch.compile(llama.model)  # type: ignore
        # Use huggingface's tokenizer since llama.tokenizer has a different interface (annoying).
        tokenizer = get_huggingface_tokenizer(hf_model_id)

        return LlamaSimulator(llama.model, tokenizer, sys_prompt_type, remove_tab)

    @torch.inference_mode()
    def setup_kv_cache(self, input_ids: torch.Tensor) -> int:
        self.model.forward(input_ids, 0)
        return input_ids.shape[1]

    @torch.inference_mode()
    def get_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        assert isinstance(self.prefix_cache, int)
        return self.model.forward(input_ids, self.prefix_cache)


class FinetunedSimulator(NeuronSimulator):
    def __init__(
        self,
        model: Llama3Model,
        tokenizer: Llama3TokenizerWrapper,
        add_special_tokens: bool,
        gpu_idx: int = 0,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.add_special_tokens = add_special_tokens
        self.gpu_idx = gpu_idx

    @staticmethod
    def setup(
        model_path: str, add_special_tokens: bool = False, gpu_idx: int = 0
    ) -> "FinetunedSimulator":
        """
        Set gpu_idx to > 0 if you have multiple GPUs and the explainer is usiing the first one.
        """
        tokenizer = get_tokenizer(model_path=model_path, add_special_tokens=add_special_tokens)
        model = Llama3Model(model_path=model_path, add_special_tokens=add_special_tokens)
        device = torch.device(f"cuda:{gpu_idx}")
        model.to(device).to(torch.bfloat16)
        return FinetunedSimulator(model, tokenizer, add_special_tokens, gpu_idx)

    def simulate(
        self, explanations: List[str], tokens: List[str], token_ids: List[int] | None = None
    ) -> List[SequenceSimulation]:
        """Simulate the behavior of neurons based on multiple explanations."""
        assert token_ids is not None

        batch_size = len(explanations)
        prompt_prefixes = [f"## Neuron Description: {exp}\n\n ## Input: " for exp in explanations]
        prefix_tokens: Dict[str, torch.Tensor] = self.tokenizer(  # type: ignore
            prompt_prefixes, padding=True, return_tensors="pt"
        )

        input_ids = torch.cat(
            (prefix_tokens["input_ids"], torch.tensor([token_ids] * batch_size)), dim=1
        ).to(f"cuda:{self.gpu_idx}")

        if self.add_special_tokens:
            for old_id, new_id in UPD_MAPPING.items():
                input_ids[input_ids == old_id] = new_id

        attention_mask = torch.cat(
            [prefix_tokens["attention_mask"], torch.ones((batch_size, len(token_ids)))], dim=1
        ).to(f"cuda:{self.gpu_idx}")

        # TODO(damichoi): Right now, we assume that tokenizer for the subject model and
        # simulator are the same (which is why we can use token_ids directly), but this
        # might not always be the case. We should deal with this in a better way than simply
        # encoding the tokens (obtained by tokenizing the full sequence by the subject model)
        # since in general:
        # simulator_tokenizer.encode(full_sequence) !=
        # [simulator_tokenizer.encode(token) for token in subject_tokenizer.tokenize(full_sequence)]
        token_indices = [prefix_tokens["input_ids"].shape[1] - 1]
        for _ in token_ids:
            token_indices.append(token_indices[-1] + 1)

        with torch.no_grad():
            logits: torch.Tensor = self.model.forward(  # type: ignore
                {"input_ids": input_ids, "attention_mask": attention_mask}
            )

        response_logits: NDFloatArray = logits[:, :, :11].cpu().numpy()  # type: ignore

        simulations: List[SequenceSimulation] = []
        for batch_idx in range(batch_size):
            expected_values: List[float] = []
            distribution_probabilities: list[list[float]] = []
            for i in range(1, len(token_indices)):
                if token_indices[i] == token_indices[i - 1] + 1:
                    norm_probs, expected_value = compute_predicted_activation_stats_for_token(
                        response_logits[batch_idx, token_indices[i], :]
                    )
                    distribution_probabilities.append(norm_probs.tolist())
                    expected_values.append(expected_value)
                else:
                    expected_values.append(0.0)
                    distribution_probabilities.append([])

            simulations.append(
                SequenceSimulation(
                    tokens=tokens,
                    expected_activations=expected_values,
                    activation_scale=ActivationScale.SIMULATED_NORMALIZED_ACTIVATIONS,
                    distribution_probabilities=None,
                )
            )

        return simulations
