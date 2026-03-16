import json
import os
import pickle
import random
from typing import Any, Literal, Type, cast

import numpy as np
from datasets import load_dataset
from latentqa.mc_util import (
    LABEL_DELIMITERS,
    LABEL_TYPES,
    OPTION_DELIMITERS,
    MultipleChoiceOptionsString,
    construct_mc_options_for_train,
)
from latentqa.qas import (
    BaseQA,
    CompletedQuestion,
    IncompleteMultipleChoice,
    InvalidContext,
    QACategory,
    QAGenerator,
)
from llm_util.types import Usage
from pydantic import BaseModel
from tqdm import tqdm
from user_modeling.db_wrapper import SQLiteModelStore
from user_modeling.llm_util import DataGeneratorConfig
from user_modeling.synthsys.subject_inputs import MinimalSubjectInput, SubjectInputDB
from user_modeling.user_attribute import ALL_USER_ATTRIBUTES, UserAttribute


class FullLQAData(BaseModel):
    is_valid: bool
    attr_class: Type[UserAttribute]
    attr: UserAttribute
    system_prompt: str | None
    user_prompt: str
    completion: str
    aug_id: int | None = None
    aug_timestamp: str | None = None

    # Fields related to QA.
    qa_category: QACategory | None
    question: str | None
    stem: str | None | None
    distractors: list[str] | None = None
    answer: str | None
    model_name: str
    prompt_used_to_generate_qa_prompt: list[dict[str, Any]]
    validation_reasoning: str | None
    usage_for_qa: Usage

    def model_post_init(self, context: Any) -> None:
        if not self.is_valid:
            return
        if self.stem is not None:
            assert (
                self.distractors is not None
            ), "Distractors should be provided when stem is provided."
            assert self.question is None, "Question shouldn't be provided when stem is provided."
        else:
            assert (
                self.question is not None
            ), "Question should be provided when stem is not provided."


class FullLQADataDB(SQLiteModelStore[FullLQAData]):
    """Database wrapper for FullLQAData objects.

    This database stores LQA (LatentQA) question-answer pairs generated from
    subject inputs (system prompt, user prompt, completion). It combines the
    subject context with the generated QA data.

    Each entry contains (among other fields):
    - a subject input (system prompt, user prompt, completion)
    - the user attribute class and attribute value
    - QA data: either (question, answer) for general questions, or
      (stem, distractors, answer) for multiple choice questions
    - metadata: model used for generation, validation reasoning, usage stats
    """

    def __init__(self, db_dir: str, read_only: bool = False):
        super().__init__(db_dir=db_dir, model_cls=FullLQAData, read_only=read_only)

    @property
    def _all_field_names(self) -> list[str]:
        return [
            "attr_class",
            "attr",
            "system_prompt",
            "user_prompt",
            "completion",
            "is_valid",
            "question",
            "answer",
            "qa_category",
            "model_name",
            "prompt_used_to_generate_qa_prompt",
            "validation_reasoning",
            "usage_for_qa",
            "aug_id",
            "aug_timestamp",
            "distractors",
            "stem",
        ]

    @property
    def _key_field_names(self) -> list[str]:
        return [
            "attr_class",
            "attr",
            "system_prompt",
            "user_prompt",
            "completion",
            "question",
            "answer",
            "aug_id",
            "aug_timestamp",
            "distractors",
            "stem",
        ]

    def _convert_serialized_obj_to_obj(self, obj_dict: dict[str, Any]) -> FullLQAData:
        attr_class_name = obj_dict["attr_class"]
        attr_class = ALL_USER_ATTRIBUTES[attr_class_name]

        if obj_dict["is_valid"] == "True":
            is_valid = True
        elif obj_dict["is_valid"] == "False":
            is_valid = False
        else:
            raise ValueError(f"Invalid is_valid value: {obj_dict['is_valid']}")

        system_prompt = obj_dict["system_prompt"] if obj_dict["system_prompt"] != "null" else None
        question = obj_dict["question"] if obj_dict["question"] != "null" else None
        answer = obj_dict["answer"] if obj_dict["answer"] != "null" else None
        stem = obj_dict["stem"] if obj_dict.get("stem", "null") != "null" else None
        distractors = (
            json.loads(obj_dict["distractors"])
            if obj_dict.get("distractors", "null") != "null"
            else None
        )

        # Deal with qa_category. This is because things were being saved inconsistently
        # either as str(qa_category) or qa_category.value.
        if obj_dict["qa_category"] != "null":
            try:
                qa_category = eval(obj_dict["qa_category"])
            except Exception:
                qa_category = QACategory(obj_dict["qa_category"])
        else:
            qa_category = None

        obj = FullLQAData(
            is_valid=is_valid,
            attr_class=attr_class,
            attr=attr_class(obj_dict["attr"]),
            system_prompt=system_prompt,
            user_prompt=obj_dict["user_prompt"],
            completion=obj_dict["completion"],
            qa_category=qa_category,
            question=question,
            stem=stem,
            distractors=distractors,
            answer=answer,
            model_name=obj_dict["model_name"],
            prompt_used_to_generate_qa_prompt=json.loads(
                obj_dict["prompt_used_to_generate_qa_prompt"]
            ),
            validation_reasoning=obj_dict["validation_reasoning"],
            usage_for_qa=Usage(**json.loads(obj_dict["usage_for_qa"])),
        )
        return obj


class FullLQADataGenerationConfig(BaseModel):
    subject_input_db_dir: str | None = None
    hf_dataset_id: str | None = None
    qa_generation_configs: list[DataGeneratorConfig]
    question_type: Literal["general", "multiple_choice"] = "general"
    format_probs: tuple[float, float, float] = (0.6, 0.25, 0.15)
    filter_buffer: float = 0.01


async def generate_full_lqa_data_from_subject_inputs(
    config: FullLQADataGenerationConfig,
    full_data_db_dir: str,
    max_num_insert: int | None = None,
    batch_size: int = 100,
    seed: int = 128,
):
    rng = random.Random(seed)

    # Initialize all the DBs.
    if config.subject_input_db_dir is not None:
        subject_input_db = SubjectInputDB(db_dir=config.subject_input_db_dir, read_only=True)
    else:
        subject_input_db = None
    full_data_db = FullLQADataDB(db_dir=full_data_db_dir, read_only=False)
    time_str = full_data_db.get_time_str()

    # Get all unique (attr_class, attr, system_prompt, user_prompt, completion) values that already
    # exist in full_data_db so that we don't generate qas for them.
    unique_keys_list = await full_data_db.get_unique_values(
        ["attr_class", "attr", "system_prompt", "user_prompt", "completion"]
    )
    existing_subject_inputs: set[MinimalSubjectInput] = set()
    for (
        attr_class_name,
        attr_value,
        system_prompt,
        user_prompt,
        completion,
    ) in unique_keys_list:
        attr_class = ALL_USER_ATTRIBUTES[attr_class_name]
        attr = attr_class(attr_value)
        existing_subject_inputs.add(
            MinimalSubjectInput(
                attr_class=attr_class,
                attr=attr,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                completion=completion,
            )
        )

    # Create a list of all subject inputs to generate question-answers for.
    # If the subject input db is not provided, we use the version uploaded on huggingface.
    subject_inputs_to_consider: list[MinimalSubjectInput] = []
    if subject_input_db is None:
        assert config.hf_dataset_id in [
            "Transluce/SynthSys-Llama-3.1-8B-Instruct",
            "Transluce/SynthSys-Llama-3.1-70B-Instruct",
        ]
        hf_dataset = load_dataset(config.hf_dataset_id, split="train")
        for row in hf_dataset:
            row = cast(dict[str, Any], row)

            attr_class = ALL_USER_ATTRIBUTES[row["attr_class"]]
            attr = attr_class(row["attr"])
            attr_values = [a for a in attr_class]
            attr_idx = attr_values.index(attr)

            probs = np.array(row["probs"])
            attr_scores = probs[:, attr_idx]
            completion = row["completions"][np.argmax(attr_scores)]

            minimal_si = MinimalSubjectInput(
                attr_class=attr_class,
                attr=attr,
                system_prompt=row["system_prompt"] if row.get("system_prompt") else None,
                user_prompt=row["user_prompt"],
                completion=completion,
            )
            if minimal_si not in existing_subject_inputs:
                subject_inputs_to_consider.append(minimal_si)
    else:
        async for si in subject_input_db.iterrows():
            if si.is_valid(buffer=config.filter_buffer):
                minimal_si = si.to_minimal_subject_input()
                if minimal_si not in existing_subject_inputs:
                    subject_inputs_to_consider.append(minimal_si)
    rng.shuffle(subject_inputs_to_consider)
    print("Num datapoints left to insert: ", len(subject_inputs_to_consider))

    if max_num_insert is not None:
        subject_inputs_to_consider = subject_inputs_to_consider[:max_num_insert]

    # Generate qas in batches.
    qa_generators = [
        QAGenerator(
            gen_config,
            question_type=config.question_type,
            format_probs=config.format_probs,
            rng=rng,
        )
        for gen_config in config.qa_generation_configs
    ]

    pbar = tqdm(total=len(subject_inputs_to_consider))
    for i in range(0, len(subject_inputs_to_consider), batch_size):
        subject_inputs = subject_inputs_to_consider[i : i + batch_size]

        # Sample a generator.
        generator = qa_generators[rng.randint(0, len(qa_generators) - 1)]

        outs = await generator.generate_qas_batch(subject_inputs, use_cache=False)

        # Insert results into full_data_db.
        objs_to_insert: list[FullLQAData] = []
        for si, out in zip(subject_inputs, outs):

            # This happens when there is a parsing error in the LLM response.
            if isinstance(out, InvalidContext):
                usage = out.usage
                assert usage is not None
                objs_to_insert.append(
                    FullLQAData(
                        is_valid=False,
                        qa_category=None,
                        question=None,
                        stem=None,
                        distractors=None,
                        answer=None,
                        model_name=out.model_name,
                        prompt_used_to_generate_qa_prompt=out.prompt_used_to_generate_prompt,
                        validation_reasoning=out.reasoning,
                        usage_for_qa=usage,
                        **si.model_dump(),
                    )
                )
            elif len(out) == 4:
                for qa in out:
                    usage = qa.usage
                    assert usage is not None
                    prompt_for_gen = qa.prompt_used_to_generate_prompt
                    assert prompt_for_gen is not None
                    if isinstance(qa.content, CompletedQuestion):
                        question = qa.content.question
                        stem, distractors = None, None
                    elif isinstance(qa.content, IncompleteMultipleChoice):
                        question = None
                        stem = qa.content.stem
                        distractors = qa.content.distractors
                    else:
                        raise ValueError(f"Unknown QA content type: {type(qa.content)}")
                    model_name = qa.model_name
                    assert model_name is not None
                    objs_to_insert.append(
                        FullLQAData(
                            is_valid=True,
                            qa_category=qa.qa_category,
                            question=question,
                            stem=stem,
                            distractors=distractors,
                            answer=qa.answer,
                            model_name=model_name,
                            prompt_used_to_generate_qa_prompt=prompt_for_gen,
                            validation_reasoning=qa.validation_reasoning,
                            usage_for_qa=usage,
                            **si.model_dump(),
                        )
                    )
            else:
                assert len(out) == 0

            pbar.update(1)

        await full_data_db.insert_batch(
            objs=objs_to_insert,
            batch_time=time_str,
            config=config,
        )
    pbar.close()


class QA(BaseQA[CompletedQuestion]):
    """Used for training and evaluating the LatentQA decoder."""

    seed_response: str | None = None  # Only used for evaluation.
    multiple_choice_options: MultipleChoiceOptionsString | None = None
    """This might be set if question is multiple choice."""

    @property
    def question(self) -> str:
        return self.content.question

    def __hash__(self) -> int:
        # Hash based on the core fields that define QA equality
        options_str = (  # For MC, include the options string if present
            self.multiple_choice_options.options_str
            if self.multiple_choice_options is not None
            else None
        )
        return hash(
            (
                self.content.question,
                self.answer,
                self.attr_class,
                self.attr,
                self.qa_category,
                self.seed_response,
                options_str,
            )
        )


class LQAData(BaseModel):
    attr_class: Type[UserAttribute]
    attr: UserAttribute

    system_prompt: str | None
    user_prompt: str | None
    completion: str | None = None
    context_conversation: list[dict] | None = None
    conversation: list[dict] | None = None
    conversation_read_range: tuple[int, int] | None = None

    qa: QA
    id: Any
    """
    A unique identifier for this datapoint.
    In practice, it can just be the index of the datapoint in the dataset.
    It's used to make sure we don't evaluate the same datapoint multiple times.
    """
    tag: str | None = None
    """
    A string that helps identify the origin of the datapoint.
    For example, for PRISM, the tag is "prism_{conversation_id}", where conversation_id
    uniquely identifies the conversation in the original PRISM dataset.
    """
    metadata: Any | None = None

    def model_post_init(self, context: Any) -> None:
        if self.conversation is not None:
            assert (
                self.system_prompt is None and self.user_prompt is None
            ), "System prompt and user prompt shouldn't be provided when conversation is provided."
        else:
            assert (
                self.system_prompt is not None or self.user_prompt is not None
            ), "If conversation is not provided, system prompt and user prompt are required."


MC_Q_TEMPLATES = [
    """{stem}
{options}""",
    """{stem}

{options}""",
    """{stem} {options}""",
]


def convert_full_lqa_data_to_lqa_data(
    d: FullLQAData, id: Any, rng: np.random.Generator | None = None
) -> LQAData:
    # First check whether the question is complete or incomplete.
    if d.question is not None:
        answer = d.answer
        assert answer is not None
        qa = QA(
            attr_class=d.attr_class,
            attr=d.attr,
            qa_category=d.qa_category,
            content=CompletedQuestion(question=d.question),
            answer=answer,
        )
    elif d.stem is not None:  # Multiple choice question.
        assert rng is not None

        label_type = rng.choice(LABEL_TYPES)
        label_delimiter = rng.choice(LABEL_DELIMITERS)
        option_delimiter = rng.choice(OPTION_DELIMITERS)

        stem = d.stem
        assert stem is not None
        answer = d.answer
        assert answer is not None
        distractors = d.distractors
        assert distractors is not None

        mc_options_obj = construct_mc_options_for_train(
            must_include_options=[answer],
            possible_distractors=distractors,
            max_num_distractors=20,
            num_permutations=1,
            label_type=label_type,
            label_delimiter=label_delimiter,
            option_delimiter=option_delimiter,
            # This is because there might already be an equivalent to none-of-the-above option
            # already in the distractors.
            add_none_option=False,
            rng=rng,
        )[0]

        question_template = rng.choice(MC_Q_TEMPLATES)
        question = question_template.format(stem=stem, options=mc_options_obj.options_str)

        # TODO(dami): Come up with diverse answer templates.
        answer_template = "{key}"
        assert mc_options_obj.option_to_key is not None
        key = mc_options_obj.option_to_key[answer]
        answer = answer_template.format(key=key, option=answer)

        qa = QA(
            attr_class=d.attr_class,
            attr=d.attr,
            qa_category=d.qa_category,  # <- This will not be MC. But this is intended.
            content=CompletedQuestion(question=question),
            answer=answer,
            multiple_choice_options=mc_options_obj,
        )
    else:
        raise ValueError("Incompletion question that is not multiple choice!")

    return LQAData(
        attr_class=d.attr_class,
        attr=d.attr,
        system_prompt=d.system_prompt,
        user_prompt=d.user_prompt,
        completion=d.completion,
        qa=qa,
        id=id,
    )


def get_lqa_data_from_full_lqa_data_db(
    full_lqa_data_db_dir: str | os.PathLike,
    conditions: dict[str, Any] | None = None,
    rng: np.random.Generator | None = None,
) -> list[LQAData]:

    data_db = FullLQADataDB(db_dir=str(full_lqa_data_db_dir), read_only=True)
    lqa_data = []

    if conditions is None:
        conditions = {}
    for row in tqdm(
        data_db.sync_iterrows_raw(is_valid=True, **conditions), desc="Fetching valid data..."
    ):
        id, batch_time, *rest = row
        d: FullLQAData = data_db.convert_row_to_obj(row)
        lqa_data.append(convert_full_lqa_data_to_lqa_data(d, f"{id}_{batch_time}", rng=rng))
    return lqa_data


def load_lqa_data(data_path: str | os.PathLike, cls: Type[LQAData] = LQAData) -> list[LQAData]:
    data = pickle.load(open(data_path, "rb"))
    return [cls(**d.model_dump()) for d in data]


def load_legacy_pickle(data_path: str, alias: dict[str, str], cls: Type[BaseModel] | None = None):
    import importlib
    import sys

    for old, new in alias.items():
        if old not in sys.modules:
            sys.modules[old] = importlib.import_module(new)

    data: list = pickle.load(open(data_path, "rb"))
    if cls is not None:
        data = [cls(**d.model_dump()) for d in data]
    return data
