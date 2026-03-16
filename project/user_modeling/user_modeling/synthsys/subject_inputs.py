import json
import math
import pickle
from typing import Any, Type, cast

import anyio
import numpy as np
import torch
from pydantic import BaseModel, ConfigDict
from tqdm import tqdm
from user_modeling.db_wrapper import SQLiteModelStore
from user_modeling.synthsys.faithfulness_eval import (
    IDKey,
    LLMBasedFaithfulnessEvalDBWrapper,
    LLMBasedFaithfulnessEvalResult,
)
from user_modeling.user_attribute import ALL_USER_ATTRIBUTES, UserAttribute


class MinimalSubjectInput(BaseModel):
    model_config = {"frozen": True}

    attr_class: Type[UserAttribute]
    attr: UserAttribute
    system_prompt: str | None
    user_prompt: str
    completion: str


class SubjectInput(BaseModel):
    attr_class: Type[UserAttribute]
    attr: UserAttribute
    system_prompt: str | None
    user_prompt: str
    completions: list[str]
    probs: list[list[float]]
    attr_scores: list[float]
    mean_faithfulness_score: float
    max_faithfulness_score: float
    baseline_completions: list[str]
    baseline_probs: list[list[float]]
    mean_baseline_score: float
    max_baseline_score: float
    aug_id: int | None = None
    aug_timestamp: str | None = None
    activations: torch.Tensor | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def get_completion(self) -> str:
        return self.completions[np.argmax(self.attr_scores)]

    def is_valid(self, use_mean: bool = False, buffer: float = 0.0) -> bool:
        # First check whether the argmax is the correct value.
        attr_values = [a for a in self.attr_class]
        idx = attr_values.index(self.attr)
        probs = np.array(self.probs).mean(axis=0)
        is_top = probs.argmax() == idx

        if not is_top:
            return False

        # Then check whether the score is above the baseline.
        if use_mean:
            return self.mean_faithfulness_score > self.mean_baseline_score + buffer
        else:
            return self.max_faithfulness_score > self.max_baseline_score + buffer

    def to_minimal_subject_input(self) -> MinimalSubjectInput:
        return MinimalSubjectInput(
            attr_class=self.attr_class,
            attr=self.attr,
            system_prompt=self.system_prompt,
            user_prompt=self.user_prompt,
            completion=self.get_completion(),
        )


class SubjectInputDB(SQLiteModelStore[SubjectInput]):
    """Database wrapper for SubjectInput objects.

    This is sort of an intermediate database that collects
    1) all belief evaluation results for a given subject input (system prompt, user prompt), and
    2) all baseline faithfulness (belief) eval results for user prompts
    so that we can do filtering.

    Each entry contains (among other fields):
    - a subject input (system prompt, user prompt)
    - a list of completions generated from the subject input
    - LLM judge predictions for all completions
    - LLM judge predictions for all *baseline* completions (obtained from just user prompt
        without system prompt)
    """

    def __init__(self, db_dir: str, read_only: bool = False):
        super().__init__(db_dir=db_dir, model_cls=SubjectInput, read_only=read_only)

    @property
    def _all_field_names(self) -> list[str]:
        return [
            "attr_class",
            "attr",
            "system_prompt",
            "user_prompt",
            "completions",
            "probs",
            "attr_scores",
            "mean_faithfulness_score",
            "max_faithfulness_score",
            "baseline_completions",
            "baseline_probs",
            "mean_baseline_score",
            "max_baseline_score",
            "aug_id",
            "activations",
            "aug_timestamp",
        ]

    @property
    def _all_field_types(self) -> dict[str, str]:
        return {**super()._all_field_types, "activations": "array"}

    @property
    def _key_field_names(self) -> list[str]:
        return ["attr_class", "attr", "system_prompt", "user_prompt", "aug_id", "aug_timestamp"]

    def _convert_serialized_obj_to_obj(self, obj_dict: dict[str, Any]) -> SubjectInput:
        attr_class_name = obj_dict["attr_class"]
        attr_class = ALL_USER_ATTRIBUTES[attr_class_name]

        return SubjectInput(
            attr_class=attr_class,
            attr=attr_class(obj_dict["attr"]),
            system_prompt=(
                obj_dict["system_prompt"] if obj_dict["system_prompt"] != "null" else None
            ),
            user_prompt=obj_dict["user_prompt"],
            completions=json.loads(obj_dict["completions"]),
            probs=json.loads(obj_dict["probs"]),
            attr_scores=json.loads(obj_dict["attr_scores"]),
            mean_faithfulness_score=float(obj_dict["mean_faithfulness_score"]),
            max_faithfulness_score=float(obj_dict["max_faithfulness_score"]),
            baseline_completions=json.loads(obj_dict["baseline_completions"]),
            baseline_probs=json.loads(obj_dict["baseline_probs"]),
            mean_baseline_score=float(obj_dict["mean_baseline_score"]),
            max_baseline_score=float(obj_dict["max_baseline_score"]),
            aug_id=(obj_dict["aug_id"] if obj_dict.get("aug_id", "null") != "null" else None),
            aug_timestamp=(
                obj_dict["aug_timestamp"]
                if obj_dict.get("aug_timestamp", "null") != "null"
                else None
            ),
        )

    async def upsert_batch(
        self,
        datapoints: list[SubjectInput | None],
        time_str: str,
        batch_size: int = 100,
        verbose: bool = False,
    ):
        for i in tqdm(
            range(0, len(datapoints), batch_size),
            desc="Inserting datapoints...",
            disable=not verbose,
        ):
            batch = datapoints[i : i + batch_size]
            not_nones = []
            for datapoint in batch:
                if datapoint is None:
                    continue
                # Check if the datapoint with the same key already exists.
                row = await self.return_match(datapoint, return_raw=True)
                row = cast(list[Any], row)

                if row is not None:
                    row_id = cast(int, row[0])

                    # Remove the existing datapoint, and insert the new one.
                    print("Deleting row because it already exists.")
                    await self.delete(row_id)
                not_nones.append(datapoint)
            await super().insert_batch(not_nones, time_str)

    async def save_config_dict(self, config_dict: dict[str, Any], time_str: str):
        self._save_config(config_dict, time_str)


def collect_completions_and_probs(
    eval_results: list[LLMBasedFaithfulnessEvalResult],
) -> tuple[list[str], np.ndarray]:
    """Collect completions and compute mean probability scores from eval results.

    Takes a list of eval results for a single subject input (same key) and extracts
    all completions along with their mean probability scores across permutations.

    Args:
        eval_results: List of eval results, all sharing the same (attr_class, attr,
            system_prompt, user_prompt) key.

    Returns:
        A tuple of (completions, stacked_probs) where:
        - completions: List of completion strings
        - stacked_probs: Array of shape (num_completions, num_choices) containing
          mean probabilities across permutations for each choice
    """
    key: IDKey | None = None
    completions = []
    all_probs = []
    for eval_result in eval_results:
        # In order to aggregate over the completions, we need to make sure that the prompts
        # and user attributes are the same.
        if key is None:
            key = eval_result.key
        else:
            assert key == eval_result.key
        completions.append(eval_result.completion)

        logprobs_array = np.array(eval_result.logprobs)  # (num_permutations, num_choices)
        mean_probs = np.exp(logprobs_array).mean(axis=0)  # (num_choices)
        all_probs.append(mean_probs)
    stacked_probs = np.stack(all_probs)
    return completions, stacked_probs


async def build_subject_inputs_with_baselines(
    unique_keys: list[IDKey],
    baseline_results_cache: dict[IDKey, tuple[list[str], np.ndarray]],
    eval_db: LLMBasedFaithfulnessEvalDBWrapper,
    baseline_db: LLMBasedFaithfulnessEvalDBWrapper,
    max_concurrency: int = 100,
) -> tuple[list[SubjectInput | None], dict[IDKey, tuple[list[str], np.ndarray]]]:
    """Build SubjectInput objects with eval and baseline results bundled together.

    For each key, queries both eval_db and baseline_db, computes faithfulness scores,
    and constructs SubjectInput objects containing both eval and baseline metrics
    for later comparison.

    Args:
        unique_keys: List of IDKey objects identifying subject inputs to process.
        baseline_results_cache: Cache of previously computed baseline results to avoid
            redundant DB queries. Will be updated in-place with new entries.
        eval_db: Database containing faithfulness eval results (with system prompts).
        baseline_db: Database containing baseline eval results (without system prompts).
        max_concurrency: Maximum number of concurrent DB queries to limit open SQLite
            connections.

    Returns:
        A tuple of (datapoints, baseline_results_cache) where:
        - datapoints: List of SubjectInput objects (or None if baseline not found)
        - baseline_results_cache: Updated cache with any newly computed baseline results
    """

    # Limit the number of concurrent tasks so we don't open hundreds of simultaneous
    # SQLite connections (hitting the OS "too many open files" limit).
    sem = anyio.Semaphore(max_concurrency)

    async with anyio.create_task_group() as tg:
        datapoints: list[SubjectInput | None] = [None for _ in range(len(unique_keys))]

        async def process_prompt(i: int, key: IDKey):
            async with sem:
                attr_class = key.attr_class
                attr = key.attr
                assert attr is not None

                # Compute baseline scores.
                if key in baseline_results_cache:
                    baseline_completions, baseline_probs = baseline_results_cache[key]
                else:
                    baseline_results = []
                    async for row in baseline_db.iterrows(
                        attr_class=key.attr_class, user_prompt=key.user_prompt
                    ):
                        baseline_results.append(row)

                    if not baseline_results:
                        # print(f"Skipping {key} because no baseline results found.")
                        return

                    baseline_completions, baseline_probs = collect_completions_and_probs(
                        baseline_results
                    )
                    baseline_results_cache[key] = (baseline_completions, baseline_probs)

                baseline_attr_scores = baseline_probs[:, list(attr_class).index(attr)]
                mean_baseline_score = baseline_attr_scores.mean()
                max_baseline_score = baseline_attr_scores.max()

                # Compute eval scores.
                eval_results = []
                async for row in eval_db.iterrows(
                    attr_class=key.attr_class,
                    attr=key.attr,
                    system_prompt=key.system_prompt,
                    user_prompt=key.user_prompt,
                ):
                    eval_results.append(row)
                completions, probs = collect_completions_and_probs(eval_results)
                attr_scores = probs[:, list(attr_class).index(attr)]
                mean_eval_score = attr_scores.mean()
                max_eval_score = attr_scores.max()

                datapoint = SubjectInput(
                    attr_class=attr_class,
                    attr=attr,
                    system_prompt=key.system_prompt,
                    user_prompt=key.user_prompt,
                    completions=completions,
                    probs=probs.tolist(),
                    attr_scores=attr_scores.tolist(),
                    mean_faithfulness_score=mean_eval_score,
                    max_faithfulness_score=max_eval_score,
                    baseline_completions=baseline_completions,
                    baseline_probs=baseline_probs.tolist(),
                    mean_baseline_score=mean_baseline_score,
                    max_baseline_score=max_baseline_score,
                )
                datapoints[i] = datapoint

        for i, key in enumerate(unique_keys):
            tg.start_soon(process_prompt, i, key)
    return datapoints, baseline_results_cache


async def run_subject_input_aggregation(
    db_dir: str,
    faithfulness_eval_db_dir: str,
    baseline_faithfulness_eval_db_dir: str,
    faithfulness_eval_db_timestamps: list[str] | None = None,
    batch_size: int = 100,
):
    """Aggregate faithfulness eval results and populate the SubjectInput database.

    Orchestrates the full pipeline: sets up database connections, iterates through
    eval results in batches, computes faithfulness scores alongside baseline scores,
    and saves the resulting SubjectInput objects to the results database for later
    comparison.

    Args:
        db_dir: Directory where the SubjectInput results database will be stored.
        faithfulness_eval_db_dir: Directory containing the faithfulness eval database
            (eval results with system prompts).
        baseline_faithfulness_eval_db_dir: Directory containing the baseline eval
            database (eval results without system prompts, for comparison).
        faithfulness_eval_db_timestamps: If provided, only process eval results from
            these specific batch timestamps. Useful for incremental updates.
        batch_size: Number of unique keys to process per batch. Controls memory usage
            and progress granularity.
    """
    results_db = SubjectInputDB(db_dir, read_only=False)
    eval_db = LLMBasedFaithfulnessEvalDBWrapper.from_disk(faithfulness_eval_db_dir, read_only=True)
    baseline_db = LLMBasedFaithfulnessEvalDBWrapper.from_disk(
        baseline_faithfulness_eval_db_dir, read_only=True
    )

    timestamp = results_db.get_time_str()

    # Save config dict.
    config_dict = {
        "eval_db_dir": faithfulness_eval_db_dir,
        "baseline_db_dir": baseline_faithfulness_eval_db_dir,
        "eval_db_timestamps": faithfulness_eval_db_timestamps,
    }
    await results_db.save_config_dict(config_dict, timestamp)

    conditions = {}
    if faithfulness_eval_db_timestamps is not None:
        conditions["batch_time"] = faithfulness_eval_db_timestamps
    unique_keys = list(await eval_db.get_all_unique_keys(**conditions))

    baseline_results_cache: dict[IDKey, tuple[list[str], np.ndarray]] = {}

    num_batches = math.ceil(len(unique_keys) / batch_size)
    pbar = tqdm(total=len(unique_keys))
    for batch_idx in range(num_batches):
        batch_unique_keys = unique_keys[batch_idx * batch_size : (batch_idx + 1) * batch_size]
        datapoints, baseline_results_cache = await build_subject_inputs_with_baselines(
            batch_unique_keys,
            baseline_results_cache,
            eval_db,
            baseline_db,
        )
        await results_db.upsert_batch(datapoints, timestamp)
        pbar.update(len(datapoints))
    pbar.close()


def create_synthsys_dataset(db_dir: str, save_path: str, eps: float = 0.01):
    subject_input_db = SubjectInputDB(db_dir=db_dir, read_only=True)

    total = 0
    filtered_data: list[dict[str, Any]] = []
    for si in subject_input_db.sync_iterrows():
        if si.is_valid(buffer=eps):
            filtered_data.append(
                {
                    "attr_class": si.attr_class,
                    "attr": si.attr,
                    "system_prompt": si.system_prompt,
                    "user_prompt": si.user_prompt,
                    "completions": si.completions,
                    "probs": si.probs,
                }
            )
        total += 1

    num_correct = len(filtered_data)
    print(f"{num_correct}/{total} examples passed the filtering criteria.")

    with open(save_path, "wb") as f:
        pickle.dump(filtered_data, f)
