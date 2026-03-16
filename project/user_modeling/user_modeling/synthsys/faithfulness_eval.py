import asyncio
import json
import os
from collections import defaultdict
from typing import Any, Generator, Literal, Type, cast

import anyio
import numpy as np
import pandas as pd
from llm_util.types import Usage
from log_util.logger import get_logger
from openai import AsyncOpenAI
from pydantic import BaseModel, ConfigDict
from tqdm import tqdm
from tqdm.asyncio import tqdm as async_tqdm
from user_modeling.belief_eval import SubjectInputForLLMJudge, judge_belief_from_completions
from user_modeling.db_wrapper import SQLiteModelStore
from user_modeling.llm_util import get_vllm_chat_completion_async
from user_modeling.synthsys.system_prompts import SystemPromptsDBWrapper
from user_modeling.synthsys.user_prompts import UserPromptsDBWrapper
from user_modeling.user_attribute import (
    ALL_USER_ATTRIBUTES,
    Country,
    Occupation,
    Religion,
    UserAttribute,
)

logger = get_logger(__name__)


class IDKey(BaseModel):
    attr_class: Type[UserAttribute]
    attr: UserAttribute | None
    system_prompt: str | None
    user_prompt: str

    # Make the model immutable so it becomes hashable.
    model_config = ConfigDict(frozen=True)


class LLMBasedFaithfulnessEvalResult(BaseModel):
    attr_class: Type[UserAttribute]
    attr: UserAttribute | None
    system_prompt: str | None
    system_prompt_timestamp: str | None
    user_prompt: str
    user_prompt_timestamp: str | None
    completion: str
    logprobs: list[list[float]]
    usage: Usage

    @property
    def key(self) -> IDKey:
        return IDKey(
            attr_class=self.attr_class,
            attr=self.attr,
            system_prompt=self.system_prompt,
            user_prompt=self.user_prompt,
        )

    def get_mean_logprobs(self) -> np.ndarray:
        logprobs_array = np.array(self.logprobs)  # (num_permutations, num_choices)
        return np.mean(logprobs_array, axis=0)

    def get_mean_probs(self) -> np.ndarray:
        probs_array = np.exp(np.array(self.logprobs))  # (num_permutations, num_choices)
        return np.mean(probs_array, axis=0)

    def get_accuracies(self) -> np.ndarray:
        logprobs_array = np.array(self.logprobs)
        num_choices = logprobs_array.shape[1]

        preds = logprobs_array.argmax(axis=1, keepdims=True)
        corrs = np.arange(num_choices) == preds
        return np.mean(corrs, axis=0)


class FaithfulnessEvalInput(BaseModel):
    attr_class: Type[UserAttribute]
    attr: UserAttribute | None
    target_attr: UserAttribute | None = None
    system_prompt: str | None
    system_prompt_timestamp: str | None = None
    user_prompt: str
    user_prompt_timestamp: str | None = None

    @property
    def key(self) -> IDKey:
        return IDKey(
            attr_class=self.attr_class,
            attr=self.attr,
            system_prompt=self.system_prompt,
            user_prompt=self.user_prompt,
        )

    def __eq__(self, other):
        return (self.attr, self.system_prompt, self.user_prompt) == (
            other.attr,
            other.system_prompt,
            other.user_prompt,
        )

    def __hash__(self):
        return hash((self.attr, self.system_prompt, self.user_prompt))


async def evaluate_faithfulness_of_completions(
    eval_inputs: list[FaithfulnessEvalInput],
    completions_list: list[list[str] | None],
    **llm_judge_kwargs: Any,
) -> list[list[LLMBasedFaithfulnessEvalResult]]:
    """
    Evaluate faithfulness of completions using LLM judge.

    Args:
        eval_inputs: List of FaithfulnessEvalInput objects.
        completions_list: List of completions for each eval input.
        llm_judge_kwargs: Keyword arguments for the LLM judge.
            See `judge_belief_from_completions` for more details.
    Returns:
        List of lists of LLMBasedFaithfulnessEvalResult objects.
    """
    assert len(eval_inputs) == len(completions_list)
    # Get data ready for LLM judging.
    subject_inputs_for_judging = [
        SubjectInputForLLMJudge(
            attr_class=eval_input.attr_class,
            true_attr=eval_input.attr,
            target_attr=eval_input.target_attr,
            user_prompt=eval_input.user_prompt,
        )
        for eval_input in eval_inputs
    ]

    # For each completion, get multiple LLM judge predictions (different variations of the judge
    # prompts). This means for each eval_inputs[i], we could get up to
    # len(completions_list[i]) * num_permutations LLM judge predictions.
    # We end up with a flattened list of LLM judge predictions as eval_results.
    eval_results = await judge_belief_from_completions(
        subject_inputs=subject_inputs_for_judging,
        completions_list=completions_list,
        **llm_judge_kwargs,
    )
    eval_results_df = pd.DataFrame(eval_results)

    # Stack logprobs from all judge predictions from the same completions.
    agg_eval_results: list[list[LLMBasedFaithfulnessEvalResult]] = [[] for _ in eval_inputs]
    for (idx, completion), df in eval_results_df.groupby(["si_idx", "completion"]):  # type: ignore
        idx = int(idx)

        logprobs = np.stack(df["logprobs"].values)  # type: ignore
        agg_eval_results[idx].append(
            LLMBasedFaithfulnessEvalResult(
                attr_class=eval_inputs[idx].attr_class,
                attr=eval_inputs[idx].attr,
                system_prompt=eval_inputs[idx].system_prompt,
                system_prompt_timestamp=eval_inputs[idx].system_prompt_timestamp,
                user_prompt=eval_inputs[idx].user_prompt,
                user_prompt_timestamp=eval_inputs[idx].user_prompt_timestamp,
                completion=completion,
                logprobs=logprobs.tolist(),  # (num judge prompts, num choices)
                usage=cast(Usage, sum(df.usage.values)) if "usage" in df.columns else None,  # type: ignore
            )
        )
    return agg_eval_results


class LLMBasedFaithfulnessEvalDBConfig(BaseModel):
    subject_hf_model_id: str
    """HuggingFace model identifier for the *subject* model."""
    num_permutations: int = 1
    """Number of answer choice permutations to generate for each completion when constructing judge prompts."""
    generation_temperature: float = 1.0
    """Sampling temperature for the *subject* model."""
    generation_max_new_tokens: int = 256
    """Maximum new tokens to sample from the *subject* model."""
    llm_judge_provider: str = "openai"
    """Provider to use for the *judge* model (passed through to :pyfunc:`get_llm_completions_async`)."""
    llm_judge_model_name: str = "gpt-4.1-nano-2025-04-14"
    """Concrete model name for the *judge* model."""


class LLMBasedFaithfulnessEvalDBWrapper(SQLiteModelStore[LLMBasedFaithfulnessEvalResult]):
    """
    Database wrapper for LLMBasedFaithfulnessEvalResult objects.

    Each entry contains (among other fields):
    - a subject input (system prompt, user prompt) with provenance timestamps
    - a completion generated from the subject input
    - LLM judge logprobs for the completion, with multiple predictions from different
      permutations of answer choices to reduce order bias
    """

    def __init__(
        self,
        db_dir: str,
        db_config: LLMBasedFaithfulnessEvalDBConfig,
        read_only: bool = False,
    ):
        super().__init__(
            db_dir=db_dir, model_cls=LLMBasedFaithfulnessEvalResult, read_only=read_only
        )

        db_config_path = os.path.join(db_dir, "db_config.json")
        if os.path.exists(db_config_path):
            # Make sure the config matches the existing one.
            with open(db_config_path, "r") as f:
                existing_config_dict = json.load(f)
            db_config_dict = db_config.model_dump()
            for key in db_config_dict:
                existing_val = existing_config_dict[key]
                curr_val = db_config_dict[key]
                assert existing_val == curr_val, (
                    f"Value of '{key}' for existing config is '{existing_val}', "
                    f"while the value given in the config is '{curr_val}')"
                )
        else:
            os.makedirs(db_dir, exist_ok=True)
            # Create the config file.
            with open(db_config_path, "w") as f:
                f.write(db_config.model_dump_json())

        self.db_config = db_config

    @classmethod
    def from_disk(cls, db_dir: str, read_only: bool = True):
        db_config_path = os.path.join(db_dir, "db_config.json")
        assert os.path.exists(db_config_path), f"Config file not found at {db_config_path}"
        with open(db_config_path, "r") as f:
            db_config = LLMBasedFaithfulnessEvalDBConfig.model_validate_json(f.read())
        return cls(db_dir, db_config, read_only=read_only)

    @property
    def _all_field_names(self) -> list[str]:
        return [
            "attr_class",
            "attr",
            "system_prompt",
            "system_prompt_timestamp",
            "user_prompt",
            "user_prompt_timestamp",
            "completion",
            "logprobs",
            "usage",
        ]

    @property
    def _key_field_names(self) -> list[str]:
        return ["attr_class", "attr", "system_prompt", "user_prompt", "completion"]

    def _convert_serialized_obj_to_obj(
        self, obj_dict: dict[str, Any]
    ) -> LLMBasedFaithfulnessEvalResult:
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
            elif name in [
                "system_prompt",
                "system_prompt_timestamp",
                "user_prompt",
                "user_prompt_timestamp",
                "completion",
            ]:
                data[name] = raw
            elif name == "logprobs":
                data[name] = json.loads(raw)
            elif name == "usage":
                data[name] = Usage(**json.loads(raw))
            else:
                raise ValueError(f"Unknown field: {name}")
        return LLMBasedFaithfulnessEvalResult(**data)

    @classmethod
    async def merge_dbs(
        cls,
        db1_dir: str,
        db2_dir: str,
        new_db_dir: str,
    ):
        """Create a new database that is a union of the two input databases."""
        db1 = cls.from_disk(db1_dir, read_only=True)
        db2 = cls.from_disk(db2_dir, read_only=True)

        # Must share the same config.
        assert db1.db_config == db2.db_config, "Cannot merge databases with different configs."

        new_db = cls(new_db_dir, db1.db_config, read_only=False)

        # Copy rows from db1 then db2.
        for src_db in (db1, db2):
            all_rows = await src_db.fetch_all_raw()

            async for row in async_tqdm(all_rows, desc=f"Copying rows from {src_db.db_dir}"):
                timestamp = row[1]  # batch_time column
                obj = src_db.convert_row_to_obj(row)
                await new_db.insert(obj, timestamp)  # INSERT OR IGNORE keeps duplicates out

        # Copy config files.
        for src_db in (db1, db2):
            for fname in os.listdir(src_db.config_dir):
                src_path = os.path.join(src_db.config_dir, fname)
                dst_path = os.path.join(new_db.config_dir, fname)
                with open(src_path, "r") as f_in, open(dst_path, "w") as f_out:
                    json.dump(json.load(f_in), f_out)

    @staticmethod
    def get_all_user_prompts(
        user_prompt_db_dir: str,
        user_prompt_type: Literal["faith_result", "user_prompt"],
        split: Literal["train", "test"] | None = None,
        user_prompt_timestamps: list[str] | None = None,
    ) -> dict[Type[UserAttribute], list[tuple[str, str]]]:
        up_conditions: dict[str, Any] = {}
        if split is not None:
            up_conditions["split"] = split
        if user_prompt_timestamps is not None:
            up_conditions["batch_time"] = user_prompt_timestamps

        user_prompts_and_timestamps: dict[Type[UserAttribute], set[tuple[str, str]]] = defaultdict(
            set
        )

        if user_prompt_type == "user_prompt":
            user_prompt_db_wrapper = UserPromptsDBWrapper(db_dir=user_prompt_db_dir, read_only=True)
            for row in tqdm(user_prompt_db_wrapper.sync_iterrows_raw(**up_conditions)):
                attr_class_name = row[6]
                attr_class = ALL_USER_ATTRIBUTES[attr_class_name]
                user_prompt = row[4]
                timestamp = row[1]
                user_prompts_and_timestamps[attr_class].add((user_prompt, timestamp))

        elif user_prompt_type == "faith_result":
            user_prompt_db_wrapper = LLMBasedFaithfulnessEvalDBWrapper.from_disk(user_prompt_db_dir)
            for row in tqdm(user_prompt_db_wrapper.sync_iterrows_raw(**up_conditions)):
                attr_class_name = row[2]
                attr_class = ALL_USER_ATTRIBUTES[attr_class_name]
                user_prompt = row[6]
                timestamp = row[7]
                user_prompts_and_timestamps[attr_class].add((user_prompt, timestamp))
        else:
            raise ValueError(f"Invalid user prompt type: {user_prompt_type}")

        # Sort the values to maintain reproducibility.
        sorted_results = {}
        for key, values in user_prompts_and_timestamps.items():
            sorted_results[key] = sorted(values)
        return sorted_results

    @staticmethod
    def get_all_system_prompts(
        system_prompt_db_dir: str,
        split: Literal["train", "test"] | None = None,
        system_prompt_timestamps: list[str] | None = None,
        num_segments: int = 1,
        segment_idx: int = 0,
    ) -> dict[tuple[Type[UserAttribute], UserAttribute], list[tuple[str, str]]]:
        system_prompt_db_wrapper = SystemPromptsDBWrapper(
            db_dir=system_prompt_db_dir, read_only=True
        )

        sp_conditions: dict[str, Any] = {}
        if split is not None:
            sp_conditions["split"] = split
        if system_prompt_timestamps is not None:
            sp_conditions["batch_time"] = system_prompt_timestamps

        system_prompts_and_timestamps: dict[
            tuple[Type[UserAttribute], UserAttribute], list[tuple[str, str]]
        ] = defaultdict(list)
        for row in tqdm(system_prompt_db_wrapper.sync_iterrows_raw(**sp_conditions)):
            attr_class_name = row[5]
            attr_class = ALL_USER_ATTRIBUTES[attr_class_name]
            attr_value_name = row[6]
            attr_value = attr_class(attr_value_name)
            system_prompt = row[4]
            timestamp = row[1]
            system_prompts_and_timestamps[(attr_class, attr_value)].append(
                (system_prompt, timestamp)
            )

        # Sort the values to maintain reproducibility.
        for key, values in system_prompts_and_timestamps.items():
            system_prompts_and_timestamps[key] = sorted(values)

        # Deal with the case where we have multiple segments.
        if num_segments > 1:
            subset_system_prompts_and_timestamps: dict[
                tuple[Type[UserAttribute], UserAttribute], list[tuple[str, str]]
            ] = {}
            for key, values in system_prompts_and_timestamps.items():
                total = len(values)
                # Compute inclusive range for this segment.
                start = int(np.floor(total * segment_idx / num_segments))
                end = int(np.floor(total * (segment_idx + 1) / num_segments))
                subset_system_prompts_and_timestamps[key] = values[start:end]

            return subset_system_prompts_and_timestamps
        else:
            return system_prompts_and_timestamps

    async def check_if_prompt_and_completion_exists(
        self,
        attr_class: Type[UserAttribute],
        attr: UserAttribute | None,
        system_prompt: str | None,
        user_prompt: str,
        completion: str,
    ) -> bool:
        async with self.get_db() as db:
            query = f"""
                SELECT 1
                FROM {self.table_name}
                WHERE attr_class   = ?
                  AND attr         = ?
                  AND system_prompt = ?
                  AND user_prompt  = ?
                  AND completion   = ?
                LIMIT 1;
            """
            params = (
                self._stringify_field(attr_class),
                self._stringify_field(attr),
                self._stringify_field(system_prompt),
                self._stringify_field(user_prompt),
                self._stringify_field(completion),
            )
            cursor = await db.execute(query, params)
            return await cursor.fetchone() is not None

    async def get_all_unique_keys(self, **conditions: Any) -> set[IDKey]:
        where_sql, params = self._build_where_clause(conditions)
        query = f"SELECT DISTINCT attr_class, attr, system_prompt, user_prompt FROM {self.table_name}{where_sql}"

        async with self.get_db() as db:
            unique_keys = set()
            cursor = await db.execute(query, tuple(params))
            async for row in cursor:
                attr_class_name, attr_value, system_prompt, user_prompt = row
                attr_class = ALL_USER_ATTRIBUTES[attr_class_name]
                unique_keys.add(
                    IDKey(
                        attr_class=attr_class,
                        attr=attr_class(attr_value) if attr_value != "null" else None,
                        system_prompt=system_prompt if system_prompt != "null" else None,
                        user_prompt=user_prompt,
                    )
                )
            return unique_keys

    async def sample_and_evaluate_faithfulness(
        self,
        eval_inputs: list[FaithfulnessEvalInput],
        generation_client: AsyncOpenAI,
        num_completions: int,
        rng: np.random.Generator,
        batch_time: str,
        config: BaseModel | None = None,
        max_retries: int = 10,
        generation_timeout: float = 100.0,
        llm_judge_max_concurrency: int = 100,
        llm_judge_timeout: float = 10.0,
    ):
        """
        Sample completions and evaluate their faithfulness to user attributes.

        For each eval input, generates completions from the subject model given
        (system_prompt, user_prompt) pairs, then uses an LLM judge to evaluate whether
        the completions are faithful to the true attribute. Results are inserted into the
        database, skipping any (prompt, completion) pairs that already exist.

        Args:
            eval_inputs: Eval inputs specifying prompts and attributes to evaluate.
            generation_client: Async OpenAI client for generating completions from the
                subject model.
            num_completions: Number of completions to sample per prompt.
            rng: NumPy random generator for sampling seeds and permutations.
            batch_time: Timestamp string for the batch when inserting to the database.
            config: Optional config object to store with the first batch insert.
            max_retries: Maximum retry attempts for completion generation.
            generation_timeout: Timeout in seconds for completion generation.
            llm_judge_max_concurrency: Maximum concurrent LLM judge calls.
            llm_judge_timeout: Timeout in seconds for LLM judge calls.
        """
        completions_list: list[list[str] | None] = [None for _ in eval_inputs]
        async with anyio.create_task_group() as tg:

            async def _sample_completions(
                i,
                attr_class: Type[UserAttribute],
                true_attr: UserAttribute | None,
                system_prompt: str | None,
                user_prompt: str,
            ):
                messages = [{"role": "user", "content": user_prompt}]
                if system_prompt is not None:
                    messages = [{"role": "system", "content": system_prompt}] + messages
                for attempt in range(max_retries):
                    try:
                        subject_out = await get_vllm_chat_completion_async(
                            client=generation_client,
                            messages=messages,
                            model=self.db_config.subject_hf_model_id,
                            max_tokens=self.db_config.generation_max_new_tokens,
                            temperature=self.db_config.generation_temperature,
                            n=num_completions,
                            timeout=generation_timeout,
                        )
                        break  # Success, exit the retry loop
                    except asyncio.TimeoutError:
                        if attempt < max_retries - 1:
                            logger.warning(
                                f"Timeout on attempt {attempt + 1}/{max_retries} for completion generation (prompt {i}). Retrying..."
                            )
                        else:
                            logger.error(
                                f"Failed after {max_retries} attempts due to timeout for completion generation (prompt {i}). Raising exception."
                            )
                            raise
                completions = []
                for compl_idx in range(num_completions):
                    text = subject_out.choices[compl_idx].message.content
                    assert text is not None
                    # Check if this exact entry exists in the db.
                    does_exist = await self.check_if_prompt_and_completion_exists(
                        attr_class, true_attr, system_prompt, user_prompt, text
                    )
                    if does_exist:
                        continue
                    completions.append(text)
                completions_list[i] = completions

            for i, eval_input in enumerate(eval_inputs):
                tg.start_soon(
                    _sample_completions,
                    i,
                    eval_input.attr_class,
                    eval_input.attr,
                    eval_input.system_prompt,
                    eval_input.user_prompt,
                )

        objs_to_insert = await evaluate_faithfulness_of_completions(
            eval_inputs=eval_inputs,
            completions_list=completions_list,
            num_permutations=self.db_config.num_permutations,
            llm_judge_provider=self.db_config.llm_judge_provider,
            llm_judge_model_name=self.db_config.llm_judge_model_name,
            max_num_choices=20,
            seed=int(rng.integers(0, 1000000)),
            max_concurrency=llm_judge_max_concurrency,
            timeout=llm_judge_timeout,
            max_retries=max_retries,
            use_cache=False,
        )
        # Flatten the list of lists that objs_to_insert is.
        flattened_objs_to_insert = [obj for sublist in objs_to_insert for obj in sublist]
        await self.insert_batch(objs=flattened_objs_to_insert, batch_time=batch_time, config=config)


class LLMBasedFaithfulnessEvalConfig(BaseModel):
    hf_model_id: str = "meta-llama/Llama-3.1-8B-Instruct"
    system_prompt_db_dir: str | None = None
    user_prompt_db_dir: str
    user_prompt_type: Literal["faith_result", "user_prompt"] = "user_prompt"
    num_subject_completions: int
    num_examples: int
    split: Literal["train", "test"] | None = None
    system_prompt_timestamps: list[str] | None = None
    user_prompt_timestamps: list[str] | None = None
    num_segments: int = 1
    segment_idx: int = 0
    seed: int


async def run_faithfulness_evaluation(
    db_dir: str,
    faithfulness_db_config: LLMBasedFaithfulnessEvalDBConfig,
    subject_generation_client: AsyncOpenAI,
    config: LLMBasedFaithfulnessEvalConfig,
    eval_baseline: bool = False,
    batch_size: int = 16,
    generation_timeout: float = 100.0,
    llm_judge_timeout: float = 100.0,
    max_retries: int = 10,
):
    """
    Run faithfulness evaluation by sampling completions from a subject model and judging them.

    Iteratively samples (system_prompt, user_prompt) pairs, generates completions from the
    subject model, and uses an LLM judge to evaluate whether the completions are faithful
    to the user attribute encoded in the prompts. Results are stored in a database.

    The function supports two modes controlled by `sample_eval_inputs`:
    - Random sampling: Randomly samples attribute classes, user prompts, and system prompts.
    - Sequential iteration: Iterates through all user prompts without system prompts.
        This is for evaluating the baseline belief scores.

    Args:
        db_dir: Directory path for the faithfulness evaluation database.
        faithfulness_db_config: Configuration for the database wrapper.
        subject_generation_client: Async OpenAI client for generating completions from the
            subject model.
        config: Configuration specifying the model, number of examples, seed, etc.
        eval_baseline: If true, we evaluate default beliefs from just the user prompt
            without the system prompts.
        batch_size: Number of eval inputs to process per batch.
        generation_timeout: Timeout in seconds for subject model completion generation.
        llm_judge_timeout: Timeout in seconds for LLM judge calls.
        max_retries: Maximum retry attempts for failed API calls.
    """
    db_wrapper = LLMBasedFaithfulnessEvalDBWrapper(db_dir=db_dir, db_config=faithfulness_db_config)
    time_str = db_wrapper.get_time_str()

    rng = np.random.default_rng(config.seed)

    # Collect all user and system prompts here so that we don't have to keep re-querying the dbs.
    user_prompts_and_timestamps = db_wrapper.get_all_user_prompts(
        user_prompt_db_dir=config.user_prompt_db_dir,
        user_prompt_type=config.user_prompt_type,
        split=config.split,
        user_prompt_timestamps=config.user_prompt_timestamps,
    )
    if eval_baseline:
        total_num_examples = sum(len(v) for v in user_prompts_and_timestamps.values())
        config.num_examples = min(config.num_examples, total_num_examples)
    attr_classes = sorted(list(user_prompts_and_timestamps.keys()), key=lambda x: x.__name__)
    # These attribute classes are specific to SelfDescribe.
    attr_classes = [ac for ac in attr_classes if ac not in [Country, Occupation, Religion]]
    attr_by_attr_class = {
        # Sort to make sure the order is consistent.
        attr_class: sorted([attr for attr in attr_class], key=lambda x: x.value)
        for attr_class in attr_classes
    }

    if config.system_prompt_db_dir is not None:
        system_prompts_and_timestamps = db_wrapper.get_all_system_prompts(
            system_prompt_db_dir=config.system_prompt_db_dir,
            split=config.split,
            system_prompt_timestamps=config.system_prompt_timestamps,
            num_segments=config.num_segments,
            segment_idx=config.segment_idx,
        )
    else:
        system_prompts_and_timestamps = None

    # Find out how many unique datapoints are currently in the database.
    unique_keys: set[IDKey] = await db_wrapper.get_all_unique_keys()
    curr_num_examples = len(unique_keys)

    if curr_num_examples >= config.num_examples:
        print(
            f"Already have {curr_num_examples} >= {config.num_examples} examples in the database."
        )
        return

    def sample_next_eval_input() -> Generator[FaithfulnessEvalInput, None, None]:
        while True:
            # Sample an attribute class.
            attr_class_idx = int(rng.integers(len(attr_classes)))
            attr_class = attr_classes[attr_class_idx]

            # Sample a user prompt given the attribute class.
            user_prompt, user_prompt_timestamp = rng.choice(user_prompts_and_timestamps[attr_class])

            if system_prompts_and_timestamps is not None:
                # Sample an attribute value within the attribute class.
                attr_idx = int(rng.integers(len(attr_by_attr_class[attr_class])))
                attr = attr_by_attr_class[attr_class][attr_idx]
                system_prompt, system_prompt_timestamp = rng.choice(
                    system_prompts_and_timestamps[(attr_class, attr)]
                )
            else:
                attr = None
                system_prompt = None
                system_prompt_timestamp = None

            yield FaithfulnessEvalInput(
                attr_class=attr_class,
                attr=attr,
                system_prompt=system_prompt,
                system_prompt_timestamp=system_prompt_timestamp,
                user_prompt=user_prompt,
                user_prompt_timestamp=user_prompt_timestamp,
            )

    def get_next_user_prompt_batch() -> Generator[FaithfulnessEvalInput, None, None]:
        for attr_class in attr_classes:
            for user_prompt, user_prompt_timestamp in user_prompts_and_timestamps[attr_class]:
                yield FaithfulnessEvalInput(
                    attr_class=attr_class,
                    attr=None,
                    system_prompt=None,
                    system_prompt_timestamp=None,
                    user_prompt=user_prompt,
                    user_prompt_timestamp=user_prompt_timestamp,
                )

    eval_input_gen = get_next_user_prompt_batch() if eval_baseline else sample_next_eval_input()

    first_iter = True
    pbar = tqdm(total=config.num_examples - curr_num_examples)
    while curr_num_examples < config.num_examples:
        eval_inputs: set[FaithfulnessEvalInput] = set()
        while len(eval_inputs) < batch_size:
            eval_input = next(eval_input_gen, None)
            if eval_input is None:
                break
            if eval_input.key in unique_keys or eval_input in eval_inputs:
                continue
            eval_inputs.add(eval_input)
            unique_keys.add(eval_input.key)

        if len(eval_inputs) == 0:
            break

        await db_wrapper.sample_and_evaluate_faithfulness(
            eval_inputs=list(eval_inputs),
            generation_client=subject_generation_client,
            num_completions=config.num_subject_completions,
            rng=rng,
            batch_time=time_str,
            config=config if first_iter else None,  # Only save config on the first iteration.
            generation_timeout=generation_timeout,
            llm_judge_timeout=llm_judge_timeout,
            max_retries=max_retries,
        )
        unique_keys = await db_wrapper.get_all_unique_keys()
        pbar.update(len(unique_keys) - curr_num_examples)
        curr_num_examples = len(unique_keys)
        first_iter = False
