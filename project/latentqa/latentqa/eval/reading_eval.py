import gc
import json
import math
import os
import pickle
import re
from collections import defaultdict
from itertools import groupby
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from latentqa.config_util import BaseConfig
from latentqa.data_util import PatchFromRegion
from latentqa.lqa_data import QA, LQAData, load_lqa_data
from latentqa.model_wrapper import ModelWrapper, Modules
from latentqa.train_util import (
    LQATrainConfig,
    compute_loss,
    get_dataloader,
    get_decoder_model,
    get_num_patch_tokens_multiplier,
    get_subject_model,
    get_tokenizer,
    process_activations_for_batch,
)
from peft import PeftModel
from torch.distributed import ReduceOp, all_reduce
from tqdm import tqdm
from transformers.tokenization_utils import PreTrainedTokenizer


def do_logprob_eval(
    data: list[LQAData],
    hf_model_id: str,
    subject: ModelWrapper,
    decoder: ModelWrapper,
    tokenizer: PreTrainedTokenizer,
    read_modules: Modules,
    write_modules: Modules,
    layer_mapping: dict[int, list[int]],
    eval_batch_size: int,
    patch_from_regions: list[PatchFromRegion] | None = None,
    decoder_base_conv_format: Literal["default", "role_change", "act_in_user"] = "default",
    decoder_assistance_role: str = "reflect",
    decoder_base_conv_format_kwargs: dict[str, Any] | None = None,
    eval_num_steps: int | None = None,
    device: str = "cuda:0",
    rank: int = 0,
    world_size: int = 1,
    drop_last: bool = False,
    multi_gpu: bool = False,
    save_path: str | None = None,
    progress_bar: bool = False,
):
    # Load this here to make sure we evaluate on the same data at every step.
    eval_dataloader = get_dataloader(
        data=data,
        hf_model_id=hf_model_id,
        tokenizer=tokenizer,
        generate=False,
        shuffle=False,
        patch_from_regions=patch_from_regions,
        decoder_base_conv_format=decoder_base_conv_format,
        decoder_assistance_role=decoder_assistance_role,
        decoder_base_conv_format_kwargs=decoder_base_conv_format_kwargs,
        rng=np.random.default_rng(0),
        batch_size=eval_batch_size,
        rank=rank,
        world_size=world_size,
        drop_last=drop_last,
        num_patch_tokens_multiplier=get_num_patch_tokens_multiplier(layer_mapping),
    )

    local_loss_sum = torch.tensor(0.0, device=device)
    local_num_examples = torch.tensor(0.0, device=device)

    with torch.inference_mode():
        eval_results = []
        for batch_idx, batch in tqdm(
            enumerate(eval_dataloader), total=eval_num_steps, disable=not progress_bar
        ):
            if batch_idx >= eval_num_steps:
                break

            unreduced_loss = compute_loss(
                batch=batch,
                subject=subject,
                decoder=decoder,
                read_modules=read_modules,
                write_modules=write_modules,
                layer_mapping=layer_mapping,
                device=device,
            )
            local_loss_sum += unreduced_loss.sum()
            local_num_examples += unreduced_loss.size(0)

            new_batch_results = []
            for i, d_id in enumerate(batch.ids):
                logprobs = -unreduced_loss[i]
                logprob = logprobs.sum()
                prob = torch.exp(logprob)
                eval_result = {
                    "data_id": d_id,
                    "logprob": logprob.item(),
                    "prob": prob.item(),
                }
                eval_results.append(eval_result)
                new_batch_results.append(eval_result)

            # Save intermediate progress.
            if save_path is not None:
                with open(save_path, "a") as f:
                    for result in new_batch_results:
                        f.write(json.dumps(result) + "\n")

    # Aggregate losses across all processes.
    if multi_gpu:
        all_reduce(local_loss_sum, op=ReduceOp.SUM)
        all_reduce(local_num_examples, op=ReduceOp.SUM)

    eval_loss = (local_loss_sum / local_num_examples).item()

    return {"loss": eval_loss}, eval_results


def do_multi_token_multiple_choice_eval(
    data: list[LQAData],
    hf_model_id: str,
    subject: ModelWrapper,
    decoder: ModelWrapper,
    tokenizer: PreTrainedTokenizer,
    read_modules: Modules,
    write_modules: Modules,
    layer_mapping: dict[int, list[int]],
    eval_batch_size: int,
    patch_from_regions: list[PatchFromRegion] | None = None,
    decoder_base_conv_format: Literal["default", "role_change", "act_in_user"] = "default",
    decoder_assistance_role: str = "reflect",
    decoder_base_conv_format_kwargs: dict[str, Any] | None = None,
    device: str = "cuda:0",
    save_path: str | None = None,
    progress_bar: bool = False,
    **kwargs: Any,
):
    # Step 1: Create a new datapoint for each option
    flattened_data = []
    original_data_indices = []  # Track which original data each flattened data belongs to

    for data_idx, d in enumerate(data):
        data_dict = d.model_dump()
        qa_dict = data_dict.pop("qa")
        qa_dict.pop("answer")
        mc_options = qa_dict.pop("multiple_choice_options")
        assert mc_options is not None, "Multiple choice options must be provided"
        answer_template = mc_options["answer_template"]
        base_id = data_dict.pop("id")

        for i, (key, option) in enumerate(mc_options["key_to_option"].items()):
            new_data = LQAData(
                qa=QA(answer=answer_template.format(key=key, option=option), **qa_dict),
                id=f"{base_id}/{i}",
                **data_dict,
            )
            flattened_data.append(new_data)
            original_data_indices.append(data_idx)

    # Step 2: Call do_logprob_eval on the flattened data
    _, logprob_results = do_logprob_eval(
        data=flattened_data,
        hf_model_id=hf_model_id,
        subject=subject,
        decoder=decoder,
        tokenizer=tokenizer,
        read_modules=read_modules,
        write_modules=write_modules,
        layer_mapping=layer_mapping,
        eval_batch_size=eval_batch_size,
        patch_from_regions=patch_from_regions,
        decoder_base_conv_format=decoder_base_conv_format,
        decoder_assistance_role=decoder_assistance_role,
        decoder_base_conv_format_kwargs=decoder_base_conv_format_kwargs,
        eval_num_steps=math.ceil(len(flattened_data) / eval_batch_size),
        drop_last=False,
        device=device,
        save_path=None,  # Don't save intermediate results yet
        progress_bar=progress_bar,
    )

    # Step 3: Group results by original data and compute metrics
    eval_results = []
    local_probs_sum = torch.tensor(0.0, device=device)
    local_correct_sum = torch.tensor(0.0, device=device)
    local_num_examples = torch.tensor(0.0, device=device)

    # Group by original_data_indices
    grouped = groupby(
        zip(flattened_data, logprob_results, original_data_indices),
        key=lambda x: x[2],  # Group by data_idx
    )

    for data_idx, group in grouped:
        orig_d = data[data_idx]

        # Collect logprobs for all options in this group
        logprobs = {}
        for d, result, _ in group:
            logprobs[d.qa.answer] = result["logprob"]

        # Compute metrics
        keys = sorted(logprobs.keys())
        logprob_values = [logprobs[k] for k in keys]

        logprobs_tensor = torch.tensor(logprob_values)
        ground_truth_idx = keys.index(orig_d.qa.answer)
        corr = logprobs_tensor.argmax() == ground_truth_idx

        norm_logprobs = torch.log_softmax(logprobs_tensor, dim=-1)
        probs = torch.exp(logprobs_tensor)
        norm_probs = torch.softmax(logprobs_tensor, dim=-1)

        eval_result = {
            "data_id": orig_d.id,
            "correct": corr.item(),
            "prob": probs[ground_truth_idx].item(),
            "norm_prob": norm_probs[ground_truth_idx].item(),
            "logprob": logprobs_tensor[ground_truth_idx].item(),
            "norm_logprob": norm_logprobs[ground_truth_idx].item(),
        }
        eval_results.append(eval_result)

        local_probs_sum += probs[ground_truth_idx]
        local_correct_sum += corr
        local_num_examples += 1

        if save_path is not None:
            with open(save_path, "a") as f:
                f.write(json.dumps(eval_result) + "\n")

    eval_acc = (local_correct_sum / local_num_examples).item()
    eval_score = (local_probs_sum / local_num_examples).item()

    return {"acc": eval_acc, "score": eval_score}, eval_results


def do_single_token_multiple_choice_eval(
    data: list[LQAData],
    hf_model_id: str,
    subject: ModelWrapper,
    decoder: ModelWrapper,
    tokenizer: PreTrainedTokenizer,
    read_modules: Modules,
    write_modules: Modules,
    layer_mapping: dict[int, list[int]],
    eval_batch_size: int,
    patch_from_regions: list[PatchFromRegion] | None = None,
    decoder_base_conv_format: Literal["default", "role_change", "act_in_user"] = "default",
    decoder_assistance_role: str = "reflect",
    decoder_base_conv_format_kwargs: dict[str, Any] | None = None,
    eval_num_steps: int | None = None,
    device: str = "cuda:0",
    rank: int = 0,
    world_size: int = 1,
    drop_last: bool = False,
    multi_gpu: bool = False,
    save_path: str | None = None,
    progress_bar: bool = False,
):
    # Make sure the evaluation data is such that the choices are all single-token.
    answers = set()
    for d in tqdm(data, desc="Gathering all answers from data...", disable=not progress_bar):
        answers.add(d.qa.answer)
    for answer in answers:
        ids = tokenizer.encode(answer, add_special_tokens=False)
        assert len(ids) == 1

    # Load this here to make sure we evaluate on the same data at every step.
    eval_dataloader = get_dataloader(
        data=data,
        hf_model_id=hf_model_id,
        tokenizer=tokenizer,
        generate=True,  ###############
        shuffle=False,
        patch_from_regions=patch_from_regions,
        decoder_base_conv_format=decoder_base_conv_format,
        decoder_assistance_role=decoder_assistance_role,
        decoder_base_conv_format_kwargs=decoder_base_conv_format_kwargs,
        rng=np.random.default_rng(0),
        batch_size=eval_batch_size,
        rank=rank,
        world_size=world_size,
        drop_last=drop_last,
        num_patch_tokens_multiplier=get_num_patch_tokens_multiplier(layer_mapping),
    )

    local_probs_sum = torch.tensor(0.0, device=device)
    local_correct_sum = torch.tensor(0.0, device=device)
    local_num_examples = torch.tensor(0.0, device=device)

    eval_results = []
    for batch_idx, batch in tqdm(
        enumerate(eval_dataloader), total=eval_num_steps, disable=not progress_bar
    ):
        if eval_num_steps is not None and batch_idx >= eval_num_steps:
            break

        # Process activations using the shared function
        activations_list, token_range_from_list = process_activations_for_batch(
            batch=batch,
            subject=subject,
            read_modules=read_modules,
            device=device,
        )

        with torch.inference_mode():
            out = decoder.forward_with_intervention(
                activations=activations_list,
                token_range_from=token_range_from_list,
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

        new_batch_results = []
        for i, (d_id, qa, logprobs) in enumerate(zip(batch.ids, batch.qas, last_token_logprobs)):
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

            ground_truth_idx = list(qa.attr_class).index(qa.attr)
            corr = attr_logprobs_tensor.argmax() == ground_truth_idx

            norm_logprobs = torch.log_softmax(attr_logprobs_tensor, dim=-1)
            probs = torch.exp(attr_logprobs_tensor)
            norm_probs = torch.softmax(attr_logprobs_tensor, dim=-1)

            eval_result = {
                "data_id": d_id,
                "correct": corr.item(),
                "probs": probs.tolist(),
                "norm_probs": norm_probs.tolist(),
                "logprobs": attr_logprobs_tensor.tolist(),
                "norm_logprobs": norm_logprobs.tolist(),
            }
            eval_results.append(eval_result)
            new_batch_results.append(eval_result)

            local_probs_sum += probs[ground_truth_idx]
            local_correct_sum += corr
            local_num_examples += 1

        # Save intermediate progress.
        if save_path is not None:
            with open(save_path, "a") as f:
                for result in new_batch_results:
                    f.write(json.dumps(result) + "\n")

    # Aggregate losses across all processes.
    if multi_gpu:
        all_reduce(local_probs_sum, op=ReduceOp.SUM)
        all_reduce(local_correct_sum, op=ReduceOp.SUM)
        all_reduce(local_num_examples, op=ReduceOp.SUM)

    eval_acc = (local_correct_sum / local_num_examples).item()
    eval_score = (local_probs_sum / local_num_examples).item()

    return {"acc": eval_acc, "score": eval_score}, eval_results


def load_checkpoint_eval_results(jsonl_path: Path | os.PathLike) -> dict[Any, dict[str, Any]]:
    """
    Load reading eval results from a jsonl file.

    If there are any duplicate results for the same data_id, keep the latest result.
    """
    results_by_id = {}
    with open(jsonl_path, "r") as f:
        for line in f:
            result = json.loads(line)
            results_by_id[result["data_id"]] = result  # keep the most recent result
    return results_by_id


def aggregate_and_add_train_eval_results(
    save_dir: Path | str,
    ckpt_names: list[str] | None = None,
    overwrite: bool = True,
) -> dict[str, dict[int, dict[str, Any]]]:
    """
    Aggregate eval results saved during training and combine with standalone evaluation results.

    During distributed training, each rank saves eval results to separate files
    (e.g., `rank0_step_100.jsonl`, `rank1_step_100.jsonl`). This function merges those
    per-rank files and combines them with any results from `run_reading_eval`.

    The aggregated results are cached as pickle files (`step_X.pkl`) for faster loading
    on subsequent calls.

    Args:
        save_dir: Directory containing eval result files (both per-rank JSONL files
            and aggregated pickle files).
        ckpt_names: If provided, only aggregate results for these checkpoint names
            (e.g., ["step_100", "step_200"]). If None, processes all found checkpoints.
        overwrite: If False, skip checkpoints that already have a `.pkl` file.
            If True, re-aggregate and overwrite existing pickle files.

    Returns:
        A dict mapping checkpoint names (e.g., "step_100") to their eval results,
        where each result dict maps data_id to the evaluation result for that datapoint.
    """

    save_dir = Path(save_dir)
    # First, aggregate the eval results from training, during which each rank saves
    # eval results separately.
    eval_results_by_ckpt_name = defaultdict(dict)
    for file_path in save_dir.glob("rank*_step_*.jsonl"):
        fname = file_path.stem

        match = re.match(r"rank(\d+)_step_(\d+)", fname)
        assert match is not None
        int(match.group(1))
        step = int(match.group(2))
        ckpt_name = f"step_{step}"

        if ckpt_names is not None and ckpt_name not in ckpt_names:
            continue

        if (save_dir / f"{ckpt_name}.pkl").exists() and not overwrite:
            continue

        eval_results = load_checkpoint_eval_results(file_path)
        eval_results_by_ckpt_name[ckpt_name].update(eval_results)

    file_paths = set(save_dir.glob("step_*.jsonl"))
    file_paths.update(save_dir.glob("step_*.pkl"))
    file_paths.update(
        [save_dir / f"{ckpt_name}.jsonl" for ckpt_name in eval_results_by_ckpt_name.keys()]
    )

    for file_path in file_paths:
        ckpt_name = file_path.stem

        if ckpt_names is not None and ckpt_name not in ckpt_names:
            continue

        # pkl files only come from evaluating outside of training.
        if (save_dir / f"{ckpt_name}.pkl").exists() and not overwrite:
            with open(save_dir / f"{ckpt_name}.pkl", "rb") as f:
                eval_results = pickle.load(f)
            eval_results_by_ckpt_name[ckpt_name] = eval_results
        elif file_path.suffix == ".jsonl":
            # Maybe load pre-existing results (i.e. from running do_eval_on_multiple_checkpoints).
            if os.path.isfile(file_path):
                eval_results = load_checkpoint_eval_results(file_path)
                eval_results_by_ckpt_name[ckpt_name].update(eval_results)

            # Save the full set of results as a pickle file.
            with open(save_dir / f"{ckpt_name}.pkl", "wb") as f:
                pickle.dump(eval_results_by_ckpt_name[ckpt_name], f)

    return eval_results_by_ckpt_name


class ReadingEvalConfig(BaseConfig):
    eval_type: Literal["logprob", "multi_token_mc", "single_token_mc"]
    exp_path: str | Path
    ckpt_names: list[str] | None = None
    dset_dir: str
    dset_name: str
    num_batches: int
    batch_size: int
    patch_from_region: PatchFromRegion | None = PatchFromRegion.USER

    @property
    def fields_to_exclude(self):
        return {"ckpt_names", "num_batches", "batch_size"}

    def model_post_init(self, __context):
        if self.save_path is None:
            exp_path = Path(self.exp_path)
            base_exp_path = exp_path.parent.name
            exp_name = exp_path.name
            self.save_path = str(
                Path(self.dset_dir)
                / f"{self.dset_name}_eval_results"
                / f"{base_exp_path}:{exp_name}"
            )


def run_reading_eval(config: ReadingEvalConfig, progress_bar: bool = False):
    """
    Run reading evaluation on checkpoints from a LatentQA training experiment.

    Evaluates how well trained decoder models can "read" information from a subject
    model's hidden states. Supports multiple evaluation types: log-probability scoring,
    multi-token multiple choice, and single-token multiple choice.

    This function handles:
    - Resuming from partial evaluation runs (skips already-evaluated datapoints)
    - Saving results incrementally to JSONL files

    Args:
        config: Configuration specifying the experiment path, dataset, checkpoints
            to evaluate, batch size, and evaluation type.
        progress_bar: If True, display a progress bar during evaluation batches.

    Note:
        Results are saved to `{config.save_path}/{ckpt_name}.jsonl` for each checkpoint.
        The function automatically skips checkpoints that have already been fully evaluated.
    """
    exp_path = Path(config.exp_path)
    assert config.save_path is not None
    save_dir = Path(config.save_path)
    dset_dir = Path(config.dset_dir)

    config.save_config()
    print("#" * 100)
    print(f"Running reading eval for {exp_path.name}...")
    print("#" * 100)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Load the data.
    data_path = dset_dir / f"{config.dset_name}.pkl"
    assert os.path.exists(data_path)
    data = load_lqa_data(data_path, LQAData)
    num_datapoints_to_eval = min(len(data), config.num_batches * config.batch_size)

    # Get the checkpoints to evaluate.
    if config.ckpt_names is None:
        ckpt_names = []
        for ckpt_name in os.listdir(exp_path / "checkpoints"):
            ckpt_names.append(ckpt_name)
    else:
        ckpt_names = config.ckpt_names

    # Collect all pre-existing eval results.
    eval_results_by_ckpt_name = aggregate_and_add_train_eval_results(save_dir, ckpt_names)

    # Resuming logic:
    # - We evaluate exactly num_datapoints_to_eval datapoints.
    # - We never re-evaluate the same datapoint.
    num_batches_by_ckpt = {}
    data_to_eval_by_ckpt = {}
    for ckpt_name in ckpt_names:
        existing_data_ids = set()
        if ckpt_name in eval_results_by_ckpt_name:
            # Load pre-existing results.
            eval_results: dict[int, dict[str, Any]] = eval_results_by_ckpt_name[ckpt_name]
            existing_data_ids = eval_results.keys()
            if len(existing_data_ids) >= num_datapoints_to_eval:
                print(f"Skipping checkpoint `{ckpt_name}` because it already has enough results.")
                continue
        data_to_eval = [d for d in data if d.id not in existing_data_ids]
        if data_to_eval:
            num_left_to_eval = num_datapoints_to_eval - len(existing_data_ids)
            num_batches_by_ckpt[ckpt_name] = math.ceil(num_left_to_eval / config.batch_size)
            data_to_eval_by_ckpt[ckpt_name] = data_to_eval[:num_left_to_eval]
    if not data_to_eval_by_ckpt:
        print("No checkpoints left to evaluate.")
        return
    ckpt_names = sorted(data_to_eval_by_ckpt.keys())

    # Load the experiment config for the checkpoint that we're evaluating.
    with open(exp_path / "config.json", "r") as f:
        exp_config = LQATrainConfig.model_validate_json(json.load(f))

    # Load the models.
    tokenizer = get_tokenizer(exp_config.hf_model_id, exp_config.new_context_tokens)
    subject = get_subject_model(
        hf_model_id=exp_config.hf_model_id,
        use_fsdp=False,
        device="auto",
        inference_only=True,
        fsdp_config=None,
    )
    base_decoder = get_decoder_model(
        hf_model_id=exp_config.hf_model_id,
        tokenizer=tokenizer,
        use_peft=False,
        use_fsdp=False,
        device="auto",
        decoder_checkpoint_path=None,
        inference_only=True,
        fsdp_config=None,
        lora_config=exp_config.lora_config,
    )

    wrapped_subject = ModelWrapper(subject, exp_config.hf_model_id, "cuda:0")
    read_modules = wrapped_subject.get_modules(exp_config.get_read_layer_module_keys())
    layer_mapping = exp_config.get_layer_mapping()

    if config.eval_type == "logprob":
        eval_fn = do_logprob_eval
    elif config.eval_type == "multi_token_mc":
        eval_fn = do_multi_token_multiple_choice_eval
    elif config.eval_type == "single_token_mc":
        eval_fn = do_single_token_multiple_choice_eval
    else:
        raise ValueError(f"Invalid eval type: {config.eval_type}")

    for ckpt_name in tqdm(ckpt_names, desc="Evaluating checkpoints..."):
        gc.collect()
        torch.cuda.empty_cache()

        print(f"Loading checkpoint `{ckpt_name}`...")

        # Don't load anything if we're evaluating the first checkpoint.
        if ckpt_name != "step_0":
            assert isinstance(base_decoder, torch.nn.Module)
            decoder = PeftModel.from_pretrained(
                base_decoder, exp_path / "checkpoints" / ckpt_name, is_trainable=False
            )
        else:
            decoder = base_decoder

        wrapped_decoder = ModelWrapper(decoder, exp_config.hf_model_id, "cuda:0")
        write_modules = wrapped_decoder.get_modules(exp_config.get_write_layer_module_keys())

        print(f"Evaluating checkpoint `{ckpt_name}`...")

        _ = eval_fn(
            data=data_to_eval_by_ckpt[ckpt_name],
            hf_model_id=exp_config.hf_model_id,
            patch_from_regions=(
                [config.patch_from_region] if config.patch_from_region is not None else None
            ),
            decoder_base_conv_format=exp_config.decoder_base_conv_format,
            decoder_assistance_role=exp_config.decoder_assistance_role,
            decoder_base_conv_format_kwargs=exp_config.decoder_base_conv_format_kwargs,
            subject=wrapped_subject,
            decoder=wrapped_decoder,
            tokenizer=tokenizer,
            read_modules=read_modules,
            write_modules=write_modules,
            layer_mapping=layer_mapping,
            eval_batch_size=config.batch_size,
            device="cuda:0",
            progress_bar=progress_bar,
            save_path=str(save_dir / f"{ckpt_name}.jsonl"),
            eval_num_steps=num_batches_by_ckpt[ckpt_name],
            rank=0,
            world_size=1,
            multi_gpu=False,
        )
