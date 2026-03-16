from enum import Enum
from typing import Any, Callable, Mapping, Sequence, cast

import numpy as np
import pandas as pd
import plotly.express as px
import torch
from latentqa.decoder_input_util import get_decoder_base_conversation_and_patch_range
from latentqa.lqa_data import QA, LQAData
from latentqa.model_wrapper import ActivationsList, LayerModuleKey
from latentqa.qas import CompletedQuestion
from pydantic import BaseModel, ConfigDict
from torch.utils.data import Dataset
from transformers.tokenization_utils import PreTrainedTokenizer
from user_modeling.model_util import CHAT_TEMPLATES, OLD_LATENTQA_CHAT_TEMPLATES
from user_modeling.padding_util import pad_sequence
from user_modeling.synthsys.subject_inputs import SubjectInput
from user_modeling.user_attribute import ALL_USER_ATTRIBUTES, UserAttribute


class PatchFromRegion(str, Enum):
    """The region within the subject prompt to get the activations from."""

    SYSTEM = "system"
    """From just the system prompt"""
    SYSTEM_USER = "system_user"
    """
    From the system prompt and the user prompt.
    If there is a context conversation, it will be included."""
    USER = "user"
    """From just the user prompt"""
    CONTEXT_USER = "context_user"
    """From the context conversation and the user prompt"""
    LAST_ASSISTANT_TAG = "last_assistant_tag"
    """From just the last assistant tag"""
    CUSTOM = "custom"


class MinimalDatapoint(BaseModel):
    attr: UserAttribute
    system_prompt: str | None = None
    user_prompt: str
    context_conversation: list[dict] | None = None
    question: str
    seed_response: str | None = None
    answer: str
    id: int | None = None

    @property
    def qa(self) -> QA:
        args = {
            "attr_class": self.attr.__class__,
            "attr": self.attr,
            "content": CompletedQuestion(question=self.question),
            "answer": self.answer,
        }
        if self.seed_response is not None:
            args["seed_response"] = self.seed_response
            return QA.model_validate(args)
        return QA.model_validate(args)


class LQAExample(BaseModel):
    user_attribute: UserAttribute
    conversation: list[dict]
    subject_input_ids: list[int]
    activations: dict[LayerModuleKey, torch.Tensor] | None = None
    decoder_input_ids: list[int]
    decoder_labels: list[int] | None
    decoder_loss_mask: list[bool] | None
    token_range_from: tuple[int, int]
    """
    The range (inclusive, exclusive) of token idxs in subject_input_ids where we
    take the activations from.
    """
    token_range_to: tuple[int, int]
    """
    The range (inclusive, exclusive) of token idxs in decoder_input_ids where we patch the
    activations to.
    """
    qa: QA
    mean_faithfulness_score: float | None = None
    best_faithfulness_score: float | None = None
    id: Any | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True)


class LQABatch(BaseModel):
    user_attributes: list[UserAttribute]
    subject_input_ids: torch.Tensor | None
    subject_attention_mask: torch.Tensor | None
    activations: ActivationsList
    decoder_input_ids: torch.Tensor
    decoder_attention_mask: torch.Tensor
    decoder_labels: torch.Tensor | list[list[int]] | None
    decoder_loss_mask: torch.Tensor | None
    token_range_from: list[tuple[int, int]]
    token_range_to: list[tuple[int, int]]
    qas: list[QA]
    mean_faithfulness_scores: torch.Tensor | None = None
    best_faithfulness_scores: torch.Tensor | None = None
    ids: list[Any] | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __len__(self) -> int:
        return len(self.user_attributes)


def split_by_ratios(
    data: Sequence[Any],
    ratios: Mapping[Any, float],
) -> dict[Any, list[Any]]:
    """
    Shuffles `data` and splits it into len(ratios) subsets.

    The ratios should sum (roughly) to 1.0; any rounding remainder,
    or items left because the list length is not divisible by the
    ratios, are given to the **final** split in `ratios` order.

    Parameters
    ----------
    data   : any sequence (e.g. list, tuple, np.ndarray)
    ratios : mapping of split name -> fraction (e.g. {"train": .8, "val": .1, "test": .1})
    seed   : optional int for reproducibility

    Returns
    -------
    dict {split_name: list_of_items}
    """
    if not ratios:
        raise ValueError("`ratios` must contain at least one split.")

    names = list(ratios.keys())
    fracs = list(ratios.values())
    total = sum(fracs)
    if total <= 0:
        raise ValueError("Sum of ratios must be > 0.")
    # Normalise in case the ratios don't add to 1 exactly
    fracs = [f / total for f in fracs]

    # Shuffle a copy so the caller's list stays unchanged
    shuffled = list(data)
    np.random.shuffle(shuffled)

    n = len(shuffled)
    splits: dict[Any, list[Any]] = {}
    start = 0
    # iterate through all but the last split
    for name, frac in zip(names[:-1], fracs[:-1]):
        length = round(frac * n)
        splits[name] = shuffled[start : start + length]
        start += length
    # last split gets the rest
    splits[names[-1]] = shuffled[start:]
    return splits


def prepare_input_for_subject(
    conversation: list[dict],
    conversation_read_range: tuple[int, int],
    tokenizer: PreTrainedTokenizer,
    subject_chat_template: str,
) -> tuple[list[int], tuple[int, int]]:
    subject_input_ids = cast(
        list[int],
        tokenizer.apply_chat_template(
            conversation=conversation,
            chat_template=subject_chat_template,
            add_generation_prompt=True if conversation[-1]["role"] == "user" else False,
            continue_final_message=True if conversation[-1]["role"] == "assistant" else False,
            return_dict=False,
        ),
    )

    # Deal with the case where we want to read from just the last assistant tag.
    if conversation_read_range == (-1, -1):
        assert conversation[-1]["role"] == "user"
        input_ids_wo_assistant_tag = tokenizer.apply_chat_template(
            conversation=conversation,
            chat_template=subject_chat_template,
            add_generation_prompt=False,
            continue_final_message=True,
            return_dict=False,
        )
        token_range_from = (len(input_ids_wo_assistant_tag), len(subject_input_ids))
        return subject_input_ids, token_range_from

    read_start, read_end = conversation_read_range
    assert (
        read_start < read_end and read_start >= 0 and read_end <= len(conversation)
    ), f"Invalid conversation read range ({read_start}, {read_end})!"

    pre_read = conversation[:read_start]

    if pre_read:
        next_role = conversation[read_start]["role"]
        pre_read_ids = tokenizer.apply_chat_template(
            conversation=pre_read,
            chat_template=subject_chat_template,
            add_generation_prompt=True if next_role == "assistant" else False,
            return_dict=False,
        )
    else:
        pre_read_ids = []

    if read_end < len(conversation):
        next_role = conversation[read_end]["role"]
        up_till_read_ids = tokenizer.apply_chat_template(
            conversation=conversation[:read_end],
            chat_template=subject_chat_template,
            add_generation_prompt=True if next_role == "assistant" else False,
            return_dict=False,
        )
    else:
        up_till_read_ids = subject_input_ids

    token_range_from = (len(pre_read_ids), len(up_till_read_ids))

    return subject_input_ids, token_range_from


def tokenize_subject_input(
    tokenizer: PreTrainedTokenizer,
    chat_template: str,
    system_prompt: str | None,
    user_prompt: str | None,
    patch_from_region: PatchFromRegion | None,
    conversation: list[dict] | None,
    conversation_read_range: tuple[int, int] | None,
    context_conversation: list[dict[str, str]] | None = None,
    num_last_tokens: int | None = None,
    custom_read_range_fn: (
        Callable[[list[dict], PreTrainedTokenizer, str, list[int]], tuple[int, int]] | None
    ) = None,
) -> tuple[list[int], tuple[int, int], list[dict]]:
    if conversation is None:
        assert user_prompt is not None

        # Make context_conversation is a list.
        if context_conversation is None:
            context_conversation = []

        if system_prompt is not None:
            conversation = [
                {"role": "system", "content": system_prompt},
                *context_conversation,
                {"role": "user", "content": user_prompt},
            ]
            assert isinstance(patch_from_region, PatchFromRegion)
            if patch_from_region == PatchFromRegion.SYSTEM:
                conversation_read_range = (0, 1)
            elif patch_from_region == PatchFromRegion.USER:
                conversation_read_range = (len(conversation) - 1, len(conversation))
            elif patch_from_region == PatchFromRegion.CONTEXT_USER:
                conversation_read_range = (1, len(conversation))
            elif patch_from_region == PatchFromRegion.SYSTEM_USER:
                conversation_read_range = (0, len(conversation))
            elif patch_from_region == PatchFromRegion.LAST_ASSISTANT_TAG:
                conversation_read_range = (-1, -1)
            elif patch_from_region == PatchFromRegion.CUSTOM:
                conversation_read_range = (-1, -1)  # This won't be used.
            else:
                raise ValueError(f"Invalid patch_from_region: {patch_from_region}")
        else:
            conversation = context_conversation + [{"role": "user", "content": user_prompt}]
            if patch_from_region == PatchFromRegion.USER:
                conversation_read_range = (len(conversation) - 1, len(conversation))
            elif patch_from_region == PatchFromRegion.CONTEXT_USER:
                conversation_read_range = (0, len(conversation))
            elif patch_from_region == PatchFromRegion.LAST_ASSISTANT_TAG:
                conversation_read_range = (-1, -1)
            elif patch_from_region == PatchFromRegion.CUSTOM:
                conversation_read_range = (-1, -1)  # This won't be used.
            else:
                raise ValueError(f"Invalid patch_from_region: {patch_from_region}")
    else:
        assert system_prompt is None
        assert user_prompt is None
        assert patch_from_region is None
        assert context_conversation is None
        assert conversation_read_range is not None

    subject_input_ids, token_range_from = prepare_input_for_subject(
        conversation=conversation,
        conversation_read_range=conversation_read_range,
        tokenizer=tokenizer,
        subject_chat_template=chat_template,
    )
    if num_last_tokens is not None:
        token_range_from = (len(subject_input_ids) - num_last_tokens, len(subject_input_ids))

    if custom_read_range_fn is not None:
        token_range_from = custom_read_range_fn(
            conversation, tokenizer, chat_template, subject_input_ids
        )

    return subject_input_ids, token_range_from, conversation


def tokenize_qa(
    tokenizer: PreTrainedTokenizer,
    chat_template: str,
    num_patch_tokens: int,
    qa: QA,
    generate: bool = False,
    base_conv_format: str = "default",
    assistant_role: str = "reflect",
    base_conv_format_kwargs: dict[str, Any] | None = None,
):
    if base_conv_format_kwargs is None:
        base_conv_format_kwargs = {}

    base_conversation, token_range_to = get_decoder_base_conversation_and_patch_range(
        format=base_conv_format,
        num_patch_tokens=num_patch_tokens,
        question=qa.question,
        tokenizer=tokenizer,
        chat_template=chat_template,
        **base_conv_format_kwargs,
    )
    seed_response = getattr(qa, "seed_response", None)
    if generate:
        if seed_response is not None:
            conversation = base_conversation + [{"role": assistant_role, "content": seed_response}]
        else:
            conversation = base_conversation
        full = tokenizer.apply_chat_template(
            conversation=conversation,
            chat_template=chat_template,
            add_generation_prompt=True if seed_response is None else False,
            continue_final_message=False if seed_response is None else True,
            return_dict=False,
        )

        decoder_input_ids = list(full)
        decoder_labels = None
        decoder_loss_mask = None
    else:
        if seed_response is not None:
            before_answer = tokenizer.apply_chat_template(
                conversation=base_conversation
                + [{"role": assistant_role, "content": seed_response}],
                chat_template=chat_template,
                add_generation_prompt=False,
                continue_final_message=True,
                return_dict=False,
            )
            full = tokenizer.apply_chat_template(
                conversation=base_conversation
                + [{"role": assistant_role, "content": seed_response + qa.answer}],
                chat_template=chat_template,
                add_generation_prompt=False,
                continue_final_message=False,
                return_dict=False,
            )
        else:
            before_answer = tokenizer.apply_chat_template(
                conversation=base_conversation,
                chat_template=chat_template,
                add_generation_prompt=True,
                return_dict=False,
            )
            full = tokenizer.apply_chat_template(
                conversation=base_conversation + [{"role": assistant_role, "content": qa.answer}],
                chat_template=chat_template,
                add_generation_prompt=False,
                continue_final_message=False,
                return_dict=False,
            )
        decoder_input_ids = list(full[:-1])
        decoder_labels = list(full[1:])
        decoder_loss_mask = np.ones(len(full) - 1, dtype=bool)
        decoder_loss_mask[: len(before_answer) - 1] = False

    return decoder_input_ids, decoder_labels, decoder_loss_mask, token_range_to


class LatentQABaseDataset:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        subject_chat_template: str,
        decoder_chat_template: str,
        decoder_base_conv_format: str = "default",
        decoder_assistance_role: str = "reflect",
        decoder_base_conv_format_kwargs: dict[str, Any] | None = None,
        generate: bool = False,
    ):
        self.tokenizer = tokenizer
        self.subject_chat_template = subject_chat_template
        self.decoder_chat_template = decoder_chat_template
        self.decoder_base_conv_format = decoder_base_conv_format
        self.decoder_assistance_role = decoder_assistance_role
        self.decoder_base_conv_format_kwargs = decoder_base_conv_format_kwargs or {}
        self.generate = generate

    def get_subject_portion_of_example(
        self,
        system_prompt: str | None,
        user_prompt: str | None,
        patch_from_region: PatchFromRegion | None,
        conversation: list[dict] | None,
        conversation_read_range: tuple[int, int] | None,
        context_conversation: list[dict[str, str]] | None = None,
        num_last_tokens: int | None = None,
        custom_read_range_fn: (
            Callable[[list[dict], PreTrainedTokenizer, str, list[int]], tuple[int, int]] | None
        ) = None,
    ) -> tuple[list[int], tuple[int, int], list[dict]]:
        return tokenize_subject_input(
            tokenizer=self.tokenizer,
            chat_template=self.subject_chat_template,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            patch_from_region=patch_from_region,
            conversation=conversation,
            conversation_read_range=conversation_read_range,
            context_conversation=context_conversation,
            num_last_tokens=num_last_tokens,
            custom_read_range_fn=custom_read_range_fn,
        )

    def get_decoder_portion_of_example(self, num_patch_tokens: int, qa: QA):
        return tokenize_qa(
            tokenizer=self.tokenizer,
            chat_template=self.decoder_chat_template,
            num_patch_tokens=num_patch_tokens,
            qa=qa,
            generate=self.generate,
            base_conv_format=self.decoder_base_conv_format,
            assistant_role=self.decoder_assistance_role,
            base_conv_format_kwargs=self.decoder_base_conv_format_kwargs,
        )

    @staticmethod
    def collate_examples(
        examples: list[LQAExample],
        pad_token_id: int,
        generate: bool,
    ) -> LQABatch:
        subject_input_ids, subject_attention_mask = pad_sequence(
            sequences=[example.subject_input_ids for example in examples],
            pad_token_id=pad_token_id,
            padding_side="left",
        )
        # Update token_range_to according to the padding.
        token_range_from_tensor = torch.tensor([example.token_range_from for example in examples])
        padding_tokens = (1 - subject_attention_mask).sum(dim=1)
        token_range_from_tensor = token_range_from_tensor + padding_tokens[:, None]

        decoder_input_ids, decoder_attention_mask = pad_sequence(
            sequences=[example.decoder_input_ids for example in examples],
            pad_token_id=pad_token_id,
            padding_side="left",
        )
        # Update token_range_to according to the padding.
        token_range_to_tensor = torch.tensor([example.token_range_to for example in examples])
        padding_tokens = (1 - decoder_attention_mask).sum(dim=1)
        token_range_to_tensor = token_range_to_tensor + padding_tokens[:, None]

        if not generate:
            decoder_labels_list: list[list[int]] = []
            decoder_loss_mask_list: list[list[bool]] = []
            for example in examples:
                assert example.decoder_labels is not None
                assert example.decoder_loss_mask is not None
                decoder_labels_list.append(example.decoder_labels)
                decoder_loss_mask_list.append(example.decoder_loss_mask)

            decoder_labels, _ = pad_sequence(
                sequences=decoder_labels_list,
                pad_token_id=-100,
                padding_side="left",
            )
            decoder_labels = decoder_labels.long()
            decoder_loss_mask, _ = pad_sequence(
                sequences=decoder_loss_mask_list,
                pad_token_id=False,
                padding_side="left",
            )
        else:
            decoder_labels = None
            decoder_loss_mask = None

        if examples[0].mean_faithfulness_score is not None:
            mean_faithfulness_scores = torch.tensor(
                [example.mean_faithfulness_score for example in examples], dtype=torch.float32
            )
            best_faithfulness_scores = torch.tensor(
                [example.best_faithfulness_score for example in examples], dtype=torch.float32
            )
        else:
            mean_faithfulness_scores = None
            best_faithfulness_scores = None

        return LQABatch(
            user_attributes=[example.user_attribute for example in examples],
            subject_input_ids=subject_input_ids.long(),
            subject_attention_mask=subject_attention_mask,
            activations=ActivationsList.from_sample_dicts(
                [example.activations for example in examples]
            ),
            decoder_input_ids=decoder_input_ids.long(),
            decoder_attention_mask=decoder_attention_mask,
            decoder_labels=decoder_labels,
            decoder_loss_mask=decoder_loss_mask,
            token_range_from=[tuple(tr.tolist()) for tr in token_range_from_tensor],
            token_range_to=[tuple(tr.tolist()) for tr in token_range_to_tensor],
            qas=[example.qa for example in examples],
            mean_faithfulness_scores=mean_faithfulness_scores,
            best_faithfulness_scores=best_faithfulness_scores,
            ids=[example.id for example in examples],
        )


class LatentQADataset(LatentQABaseDataset, Dataset[LQAExample]):
    def __init__(
        self,
        data: list[LQAData] | list[MinimalDatapoint],
        model_type: str,
        tokenizer: PreTrainedTokenizer,
        decoder_base_conv_format: str = "default",
        decoder_assistance_role: str = "reflect",
        decoder_base_conv_format_kwargs: dict[str, Any] | None = None,
        patch_from_regions: list[PatchFromRegion] | None = None,
        rng: np.random.Generator | None = None,
        generate: bool = False,  # Whether we want to use the decoder to generate completions.
        num_patch_tokens_multiplier: int = 1,
        custom_read_range_fn: (
            Callable[[list[dict], PreTrainedTokenizer, str, list[int]], tuple[int, int]] | None
        ) = None,
    ):
        """
        custom_read_range_fn: Callable[[list[dict], list[int]], tuple[int, int]] | None = None
        If provided, this function will be used to determine the read range for the subject model.
        It will be called on the following arguments:
        - full conversation for the subject model in the raw dictionary format,
        - tokenizer,
        - chat template,
        - tokenized full conversation
        and should return the range of token ids to read (inclusive, exclusive).
        """
        if decoder_base_conv_format == "old_latentqa":
            subject_chat_template = OLD_LATENTQA_CHAT_TEMPLATES[model_type]["subject"]
            decoder_chat_template = OLD_LATENTQA_CHAT_TEMPLATES[model_type]["decoder"]
        else:
            subject_chat_template = CHAT_TEMPLATES[model_type]["default"]
            decoder_chat_template = CHAT_TEMPLATES[model_type][decoder_assistance_role]

        super().__init__(
            tokenizer,
            subject_chat_template,
            decoder_chat_template,
            decoder_base_conv_format,
            decoder_assistance_role,
            decoder_base_conv_format_kwargs,
            generate,
        )

        self.data = data
        self.patch_from_regions = patch_from_regions
        self.rng = np.random.default_rng() if rng is None else rng
        self.num_patch_tokens_multiplier = num_patch_tokens_multiplier
        self.custom_read_range_fn = custom_read_range_fn

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index) -> LQAExample:
        data = self.data[index]

        example = {}
        # 1. Prepare the inputs for the subject model.
        if hasattr(data, "conversation") and data.conversation is not None:
            conversation = data.conversation
            conversation_read_range = data.conversation_read_range
            patch_from_region = None
            context_conversation = None
        else:
            conversation = None
            conversation_read_range = None
            # Sample patch-from region.
            assert self.patch_from_regions is not None
            patch_from_region = self.patch_from_regions[
                self.rng.choice(len(self.patch_from_regions))
            ]
            context_conversation = data.context_conversation

        subject_input_ids, token_range_from, full_conversation = (
            self.get_subject_portion_of_example(
                system_prompt=data.system_prompt,
                user_prompt=data.user_prompt,
                patch_from_region=patch_from_region,
                context_conversation=context_conversation,
                conversation=conversation,
                conversation_read_range=conversation_read_range,
                custom_read_range_fn=self.custom_read_range_fn,
            )
        )
        example["subject_input_ids"] = subject_input_ids
        example["token_range_from"] = token_range_from
        example["conversation"] = full_conversation

        start, end = example["token_range_from"]
        assert isinstance(start, int) and isinstance(end, int)
        num_patch_tokens = end - start
        num_patch_tokens *= self.num_patch_tokens_multiplier

        # 2. Prepare the inputs for the decoder model.
        decoder_input_ids, decoder_labels, decoder_loss_mask, token_range_to = (
            self.get_decoder_portion_of_example(num_patch_tokens=num_patch_tokens, qa=data.qa)
        )
        example["decoder_input_ids"] = decoder_input_ids
        example["decoder_labels"] = decoder_labels
        example["decoder_loss_mask"] = decoder_loss_mask
        example["token_range_to"] = token_range_to
        example["user_attribute"] = data.attr
        example["qa"] = data.qa
        example = LQAExample(**example)

        example.id = data.id
        return example


def plot_attr_class_value_counts(
    datapoints: list[SubjectInput] | list[LQAData],
    sort_by_value: bool = False,
    filter_fn: Callable[[pd.Series], bool] | None = None,
):
    dicts_list = []
    for d in datapoints:
        if d is None:
            continue
        d_dict = d.model_dump()
        d_dict["attr_class"] = d_dict["attr_class"].__name__
        dicts_list.append(d_dict)
    df = pd.DataFrame(dicts_list)

    if filter_fn is not None:
        df = df[df.apply(filter_fn, axis=1)]

    attr_class_value_counts = df.attr_class.value_counts().to_dict()

    if sort_by_value:
        attr_class_names = sorted(
            attr_class_value_counts.keys(), key=lambda x: attr_class_value_counts[x], reverse=True
        )
    else:
        attr_class_names = [
            attr_class
            for attr_class in ALL_USER_ATTRIBUTES
            if attr_class in attr_class_value_counts
        ]

    plot_data = []
    for attr_class_name in attr_class_names:
        attr_class_df = cast(pd.DataFrame, df[df.attr_class == attr_class_name])
        attr_value_counts = attr_class_df["attr"].value_counts().to_dict()

        attr_class = ALL_USER_ATTRIBUTES[attr_class_name]
        for attr in attr_class:
            count = attr_value_counts.get(attr, 0)
            plot_data.append(
                {"attr_class": attr_class_name, "attr": attr.value_name, "count": count}
            )
    plot_data_df = pd.DataFrame(plot_data)
    fig = px.bar(
        plot_data_df,
        x="attr_class",
        y="count",
        color="attr",
        category_orders={"attr_class": attr_class_names},
        title="Number of examples per attribute class",
        hover_data={"attr": True, "count": True},  # what to show on hover
        labels={"attr_class": "Attribute Class", "count": "Count"},
    )

    fig.update_layout(barmode="stack", height=700, width=2000)
    # fig.update_xaxes(tickangle=-90)  # Rotate x-axis labels from bottom to up
    fig.show()
