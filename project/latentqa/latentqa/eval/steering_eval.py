import contextlib
import gc
import json
import math
import os
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Literal, cast

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from latentqa.config_util import BaseConfig
from latentqa.data_util import LatentQADataset, MinimalDatapoint, PatchFromRegion
from latentqa.eval.steering_util import (
    GenerationBasedControlEvalConfig,
    SteeringInput,
    evaluate_reading_using_multiple_choice,
    get_test_mc_qas,
)
from latentqa.lqa_data import LQAData
from latentqa.model_wrapper import Activations, ActivationsList, ModelWrapper, Modules
from latentqa.train_util import (
    LQATrainConfig,
    get_dataloader,
    get_decoder_model,
    get_num_patch_tokens_multiplier,
    get_subject_model,
    get_tokenizer,
)
from peft import PeftModel
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizer
from user_modeling.model_util import MODEL_TYPES
from user_modeling.padding_util import pad_sequence, strip_padding
from user_modeling.self_describe import ATTRIBUTE_PROMPTS, WIKIPEDIA_PROMPT_RESPONSE
from user_modeling.synthsys.faithfulness_eval import (
    FaithfulnessEvalInput,
    LLMBasedFaithfulnessEvalResult,
    evaluate_faithfulness_of_completions,
)
from user_modeling.user_attribute import UserAttribute, resolve_attr


def steer_subject_activations_single_step(
    grad_computation_dataloader: DataLoader,
    loss_computation_dataloader: DataLoader,
    init_activations: Activations,
    decoder: ModelWrapper,
    write_modules: Modules,
    layer_mapping: dict[int, list[int]],
    step_sizes: list[float],
    device: str = "cuda:0",
    verbose: bool = True,
) -> list[dict[str, Any]]:
    # Define parameters for each activation tensor in the Activations.
    parameters_dict = {
        key: torch.nn.Parameter(tensor.clone().detach(), requires_grad=True)
        for key, tensor in init_activations.items()
    }
    parameters = Activations(cast(dict, parameters_dict))
    N = parameters.batch_size()

    num_datapoints = 0
    for batch in grad_computation_dataloader:
        num_datapoints += batch.decoder_input_ids.shape[0]

    def get_total_loss_over_dataset(
        parameters: Activations,
        dataloader,
        propagate_grad=True,
    ):
        N_ = parameters.batch_size()
        batch_loss = torch.zeros(N_, device=device)
        for batch in dataloader:
            bs = batch.decoder_input_ids.shape[0]

            # In order to get the logits for each param tensor x batch element, we need to
            # expand the parameters and inputs accordingly.
            expanded_parameters = {}
            for key, param in parameters.items():
                expanded_parameters[key] = [param[i] for _ in range(bs) for i in range(N_)]
            input_ids = batch.decoder_input_ids.repeat_interleave(N_, dim=0)
            attn_mask = batch.decoder_attention_mask.repeat_interleave(N_, dim=0)
            labels = batch.decoder_labels.repeat_interleave(N_, dim=0)
            loss_mask = batch.decoder_loss_mask.repeat_interleave(N_, dim=0)
            token_range_from = torch.tensor(batch.token_range_from).repeat_interleave(N_, dim=0)
            token_range_to = torch.tensor(batch.token_range_to).repeat_interleave(N_, dim=0)

            out = decoder.forward_with_intervention(
                activations=ActivationsList(data=expanded_parameters),
                # convert tensors to plain int tuples for static typing
                token_range_from=[tuple(t.tolist()) for t in token_range_from],
                token_range_to=[tuple(t.tolist()) for t in token_range_to],
                input_ids=input_ids,
                attention_mask=attn_mask,
                write_modules=write_modules,
                layer_mapping=layer_mapping,
                device=device,
            )
            logits = out.logits
            assert logits is not None
            unreduced_loss = F.cross_entropy(
                input=logits.float().transpose(1, 2),
                target=labels.to(device),
                reduction="none",
            ) * loss_mask.to(device)
            unreduced_loss = unreduced_loss.unflatten(0, (bs, N_))  # (bs, N_, num_tokens)
            loss = unreduced_loss.sum(dim=-1).sum(dim=0) / num_datapoints
            batch_loss += loss.detach()
            if propagate_grad:
                loss = loss.sum()
                loss.backward()
        return batch_loss

    # Propagate gradients through the decoder.
    _ = get_total_loss_over_dataset(parameters, grad_computation_dataloader)

    # Batched version.
    concat_params = defaultdict(list)
    for step_size in step_sizes:
        for key, param in parameters.items():
            assert param.grad is not None
            upd_param = param.data.clone() - step_size * param.grad
            concat_params[key].append(upd_param)
    concat_params = {key: torch.cat(params, dim=0) for key, params in concat_params.items()}

    with torch.inference_mode():
        losses = get_total_loss_over_dataset(
            Activations(data=concat_params), loss_computation_dataloader, False
        ).view(len(step_sizes), -1)

    checkpoints = []
    for i, step_size in enumerate(step_sizes):
        params_i = {key: concat_params[key][N * i : N * (i + 1)] for key in concat_params}
        losses_i = losses[i].tolist()
        if verbose:
            print(f"Loss at step size {step_size:.2f}: {losses_i}")
        checkpoints.append(
            {"activations": Activations(data=params_i), "loss": losses_i, "step_size": step_size}
        )
    return checkpoints


def steer_subject_activations_multiple_steps(
    train_dataloader: DataLoader,
    eval_dataloader: DataLoader | None,
    init_activations: Activations,
    decoder: ModelWrapper,
    write_modules: Modules,
    layer_mapping: dict[int, list[int]],
    num_steps: int,
    step_size: float,
    device: str = "cuda:0",
    verbose: bool = True,
) -> list[dict[str, Any]]:
    # Define parameters for each activation tensor in the Activations.
    parameters_dict = {
        key: torch.nn.Parameter(tensor.clone().detach(), requires_grad=True)
        for key, tensor in init_activations.items()
    }
    parameters = Activations(cast(dict, parameters_dict))
    optimizer = optim.AdamW([param for _, param in parameters.items()], lr=step_size)

    num_train_datapoints = len(train_dataloader.dataset)
    num_eval_datapoints = len(eval_dataloader.dataset) if eval_dataloader is not None else 0

    def get_total_loss_over_dataset(
        parameters: Activations,
        dataloader,
        num_datapoints: int,
        inference_mode: bool = False,
    ):
        ctx = torch.inference_mode() if inference_mode else contextlib.nullcontext()

        N = parameters.batch_size()
        batch_loss = torch.zeros(N, device=device)
        for batch in dataloader:
            bs = batch.decoder_input_ids.shape[0]

            # In order to get the logits for each param tensor x batch element, we need to
            # expand the parameters and inputs accordingly.
            expanded_parameters = {}
            for key, param in parameters.items():
                expanded_parameters[key] = [param[i] for _ in range(bs) for i in range(N)]
            input_ids = batch.decoder_input_ids.repeat_interleave(N, dim=0)
            attn_mask = batch.decoder_attention_mask.repeat_interleave(N, dim=0)
            labels = batch.decoder_labels.repeat_interleave(N, dim=0)
            loss_mask = batch.decoder_loss_mask.repeat_interleave(N, dim=0)
            token_range_from = torch.tensor(batch.token_range_from).repeat_interleave(N, dim=0)
            token_range_to = torch.tensor(batch.token_range_to).repeat_interleave(N, dim=0)

            with ctx:
                out = decoder.forward_with_intervention(
                    activations=ActivationsList(data=expanded_parameters),
                    # convert tensors to plain int tuples for static typing
                    token_range_from=[tuple(t.tolist()) for t in token_range_from],
                    token_range_to=[tuple(t.tolist()) for t in token_range_to],
                    input_ids=input_ids,
                    attention_mask=attn_mask,
                    write_modules=write_modules,
                    layer_mapping=layer_mapping,
                    device=device,
                )
            logits = out.logits
            assert logits is not None
            unreduced_loss = F.cross_entropy(
                input=logits.float().transpose(1, 2),
                target=labels.to(device),
                reduction="none",
            ) * loss_mask.to(device)
            unreduced_loss = unreduced_loss.unflatten(0, (bs, N))  # (bs, N, num_tokens)
            loss = unreduced_loss.sum(dim=-1).sum(dim=0) / num_datapoints
            batch_loss += loss.detach()
            if not inference_mode:
                loss = loss.sum()
                loss.backward()
        return batch_loss

    # Propagate gradients through the decoder.
    checkpoints = []
    for step in tqdm(range(num_steps), desc="Steering activations...", disable=not verbose):
        loss = get_total_loss_over_dataset(
            parameters, train_dataloader, num_train_datapoints
        ).tolist()

        parameters_to_save = {
            key: param.detach().clone().to(device) for key, param in parameters.items()
        }
        checkpoint = {
            "activations": Activations(data=parameters_to_save),
            "train_loss": loss,
            "step": step,
        }
        if eval_dataloader is not None:
            eval_loss = get_total_loss_over_dataset(
                parameters, eval_dataloader, num_eval_datapoints, inference_mode=True
            ).tolist()
            checkpoint["eval_loss"] = eval_loss

        checkpoints.append(checkpoint)
        optimizer.step()
        optimizer.zero_grad()
        if verbose:
            print(f"Loss at step {step}: {loss}")

    return checkpoints


def steer_and_generate(
    activations: ActivationsList,
    token_range_from: list[tuple[int, int]],
    token_range_to: list[tuple[int, int]],
    input_ids: list[int],
    model: ModelWrapper,
    tokenizer: PreTrainedTokenizer,
    write_modules: Modules,
    layer_mapping: dict[int, list[int]],
    pad_token_id: int,
    max_new_tokens: int = 256,
    temperature: float = 1.0,
    num_completions: int = 1,
    device: str = "cuda:0",
) -> list[list[str]]:
    """Patch activations to `token_range_to`, and generate from the model."""
    bs = activations.batch_size()
    assert len(token_range_from) == bs
    assert len(token_range_to) == bs
    for trf, trt in zip(token_range_from, token_range_to):
        assert trf[1] - trf[0] == trt[1] - trt[0]

    # Repeat to generate > 1 num_completions.
    # Effectively (bs, num_completions) flattened.
    repeated_activations = activations.repeat_interleave(num_completions)
    repeated_token_range_from = [trf for trf in token_range_from for _ in range(num_completions)]
    repeated_token_range_to = [trt for trt in token_range_to for _ in range(num_completions)]

    # These don't need to be repeated, because huggingface does this automatically.
    input_ids_tensor = torch.tensor(input_ids)[None, :].expand(bs, -1)
    attention_mask = torch.ones_like(input_ids_tensor)

    out = model.generate_with_intervention(
        activations=repeated_activations,
        token_range_from=repeated_token_range_from,
        token_range_to=repeated_token_range_to,
        input_ids=input_ids_tensor.to(device),
        attention_mask=attention_mask.to(device),
        write_modules=write_modules,
        layer_mapping=layer_mapping,
        device=device,
        max_new_tokens=max_new_tokens,
        pad_token_id=pad_token_id,
        temperature=temperature,
        num_return_sequences=num_completions,
    )

    # Build a list of completions: completions[completion_idx][batch_idx] = decoded string
    completions = [["" for _ in range(num_completions)] for _ in range(bs)]
    for i in range(num_completions * bs):
        # Remove padding and decode generated tokens
        ids = strip_padding(out[i, len(input_ids) :].tolist(), pad_token_id, padding_side="right")
        batch_idx = i // num_completions
        completion_idx = i % num_completions
        completions[batch_idx][completion_idx] = tokenizer.decode(ids)

    return completions


def get_activation_checkpoints_from_gradient_based_steering(
    steering_input: SteeringInput,
    patch_from_region: PatchFromRegion,
    exp_config: LQATrainConfig,
    wrapped_subject: ModelWrapper,
    wrapped_decoder: ModelWrapper,
    tokenizer: PreTrainedTokenizer,
    step_sizes: list[float],
    num_steps: int = 1,
    grad_batch_size: int = 8,
    forward_batch_size: int = 128,
    device: str = "cuda:0",
    verbose: bool = True,
) -> tuple[list[dict[str, Any]], tuple[int, int]]:

    read_modules = wrapped_subject.get_modules(exp_config.get_read_layer_module_keys())
    write_modules = wrapped_decoder.get_modules(exp_config.get_write_layer_module_keys())
    layer_mapping = exp_config.get_layer_mapping()

    ####################################################
    # Prepare the data used for steering.
    ####################################################
    steering_data: list[MinimalDatapoint] = steering_input.get_steering_qas_for_dataloader()
    common_dataloader_kwargs = {
        "hf_model_id": exp_config.hf_model_id,
        "patch_from_regions": [patch_from_region],
        "decoder_base_conv_format": exp_config.decoder_base_conv_format,
        "decoder_assistance_role": exp_config.decoder_assistance_role,
        "tokenizer": tokenizer,
        "generate": False,
        "shuffle": False,
        "num_patch_tokens_multiplier": get_num_patch_tokens_multiplier(layer_mapping),
    }
    grad_computation_for_steering_dataloader = get_dataloader(
        data=steering_data, batch_size=grad_batch_size, **common_dataloader_kwargs
    )
    loss_computation_for_steering_dataloader = get_dataloader(
        data=steering_data, batch_size=forward_batch_size, **common_dataloader_kwargs
    )
    if steering_input.validation_qas is not None:
        validation_data = steering_input.get_steering_qas_for_dataloader(validation=True)
        validation_dataloader = get_dataloader(
            data=validation_data, batch_size=forward_batch_size, **common_dataloader_kwargs
        )
    else:
        validation_dataloader = None
    steering_dataset = cast(LatentQADataset, loss_computation_for_steering_dataloader.dataset)
    # Can just take the first datapoint since the subject input ids are the same.
    token_range_from = steering_dataset[0].token_range_from

    ##########################
    # Get steered activations.
    ##########################
    # Initialize the subject model activations.
    subject_input_ids_tensor = torch.tensor(steering_dataset[0].subject_input_ids)[
        None, :
    ]  # (1, seq_len)
    subject_activations = wrapped_subject.get_activations(
        input_ids=subject_input_ids_tensor.to(device),
        attention_mask=torch.ones_like(subject_input_ids_tensor).to(device),
        read_modules=read_modules,
        device=device,
    )

    # if num_steps == 1:
    #     raise NotImplementedError("Single-step steering is not implemented yet.")
    #     # checkpoints = steer_subject_activations_single_step(
    #     #     grad_computation_dataloader=grad_computation_for_steering_dataloader,
    #     #     loss_computation_dataloader=loss_computation_for_steering_dataloader,
    #     #     init_activations=subject_activations,
    #     #     decoder=wrapped_decoder,
    #     #     write_modules=write_modules,
    #     #     layer_mapping=layer_mapping,
    #     #     step_sizes=step_sizes,
    #     #     device=device,
    #     #     verbose=verbose,
    #     # )
    # else:
    checkpoints = steer_subject_activations_multiple_steps(
        train_dataloader=grad_computation_for_steering_dataloader,
        eval_dataloader=validation_dataloader,
        init_activations=subject_activations,
        decoder=wrapped_decoder,
        write_modules=write_modules,
        layer_mapping=layer_mapping,
        num_steps=num_steps,
        step_size=step_sizes[0],
        device=device,
        verbose=verbose,
    )

    # Unpack the loss.
    for ckpt in checkpoints:
        # We do this because we only have one steering input.
        ckpt["train_loss"] = ckpt["train_loss"][0]
        if "eval_loss" in ckpt:
            ckpt["eval_loss"] = ckpt["eval_loss"][0]

    return checkpoints, token_range_from


def evaluate_reading_after_steering(
    steering_input: SteeringInput,
    token_range_from: tuple[int, int],
    all_activations: ActivationsList,
    patch_from_region: PatchFromRegion,
    exp_config: LQATrainConfig,
    wrapped_decoder: ModelWrapper,
    tokenizer: PreTrainedTokenizer,
    custom_read_range_fn: (
        Callable[[list[dict], PreTrainedTokenizer, str, list[int]], tuple[int, int]] | None
    ) = None,
    forward_batch_size: int = 128,
    device: str = "cuda:0",
):
    layer_mapping = exp_config.get_layer_mapping()
    write_modules = wrapped_decoder.get_modules(exp_config.get_write_layer_module_keys())

    NUM_ACTS = all_activations.batch_size()

    ##################################################
    # Evaluate the steered activations on the decoder.
    ##################################################
    # We duplicate the test data since we're using the same set for each set of activations.
    test_data: list[list[LQAData]] = [
        steering_input.get_test_qas_for_dataloader() for _ in range(NUM_ACTS)
    ]

    eval_results = evaluate_reading_using_multiple_choice(
        activations_batch=all_activations,
        token_range_from=[token_range_from] * NUM_ACTS,
        data=test_data,
        hf_model_id=exp_config.hf_model_id,
        decoder=wrapped_decoder,
        tokenizer=tokenizer,
        write_modules=write_modules,
        layer_mapping=layer_mapping,
        eval_batch_size=forward_batch_size,
        patch_from_region=patch_from_region,
        custom_read_range_fn=custom_read_range_fn,
        device=device,
    )

    agg_eval_results = [{} for _ in range(NUM_ACTS)]
    # Compute mean over test_data for each checkpoints[i].
    for i in range(NUM_ACTS):
        results_by_metric = defaultdict(list)
        for result in eval_results[i]:
            for metric in result:
                results_by_metric[metric].append(result[metric])

        # We can just take the mean over all test qas for this activation, because
        # there is only one base question, with multiple permutations.
        # If there were multiple base questions, we would have to first take the mean over
        # permutatations and then take the mean over test qas.
        for metric, results in results_by_metric.items():
            agg_eval_results[i][f"decoder_{metric}"] = np.mean(np.array(results), axis=0).tolist()

    return agg_eval_results


async def evaluate_control(
    steering_input: SteeringInput,
    all_activations: ActivationsList,
    patch_from_region: PatchFromRegion,
    exp_config: LQATrainConfig,
    wrapped_subject: ModelWrapper,
    tokenizer: PreTrainedTokenizer,
    generate_to_eval_control: bool = True,
    forward_batch_size: int = 128,
    gen_based_control_eval_config: GenerationBasedControlEvalConfig | None = None,
    custom_read_range_fn: (
        Callable[[list[dict], PreTrainedTokenizer, str, list[int]], tuple[int, int]] | None
    ) = None,
    device: str = "cuda:0",
    verbose: bool = True,
) -> list[dict[str, Any]]:
    read_modules = wrapped_subject.get_modules(exp_config.get_read_layer_module_keys())
    NUM_ACTS = all_activations.batch_size()

    ####################################################
    # Prepare the data
    ####################################################
    # This is only used to tokenize the subject input and get the chat templates.
    dummy_dataset = LatentQADataset(
        data=[steering_input.get_dummy_datapoint()],
        model_type=MODEL_TYPES[exp_config.hf_model_id],
        tokenizer=tokenizer,  # type: ignore
        patch_from_regions=[patch_from_region],
        generate=False,
        custom_read_range_fn=custom_read_range_fn,
    )
    subject_input_ids_tensor = torch.tensor(dummy_dataset[0].subject_input_ids)[
        None, :
    ]  # (1, seq_len)
    T = subject_input_ids_tensor.shape[1]

    system_prompt = steering_input.system_prompt
    user_prompt = steering_input.user_prompt
    conversation = (
        [] if system_prompt is None else [{"role": "system", "content": system_prompt}]
    ) + [{"role": "user", "content": user_prompt}]

    seed_response = steering_input.subject_seed_response

    if seed_response is not None:
        before_target = tokenizer.apply_chat_template(
            conversation=conversation + [{"role": "assistant", "content": seed_response}],
            chat_template=dummy_dataset.subject_chat_template,
            add_generation_prompt=False,
            continue_final_message=True,
            return_dict=False,
        )
    else:
        before_target = tokenizer.apply_chat_template(
            conversation=conversation,
            chat_template=dummy_dataset.subject_chat_template,
            add_generation_prompt=True,
            return_dict=False,
        )
    pad_token_id = tokenizer.pad_token_id
    assert isinstance(pad_token_id, int)
    read_layers = set(key.layer for key in read_modules.keys())

    eval_results = [{} for _ in range(NUM_ACTS)]

    # Check that T is consistent with all_activations.
    for key in all_activations.keys():
        for idx in range(NUM_ACTS):
            data = all_activations.get_i(key, idx)
            assert data.shape[0] == T

    # Evaluate prob(expected subject response) instead of doing free-form generation.
    if not generate_to_eval_control:
        assert steering_input.expected_subject_responses is not None

        # Tokenize the sequence to be fed to the subject model.
        # This sequence includes the user prompt + maybe seed response + response.
        subject_seq_ids_list = []
        subject_seq_labels_list = []
        subject_seq_target_masks_list = []
        for attr, subject_responses in steering_input.expected_subject_responses:
            for subject_response in subject_responses:
                response = (seed_response if seed_response is not None else "") + subject_response
                input_ids = cast(
                    list[int],
                    tokenizer.apply_chat_template(
                        conversation=conversation + [{"role": "assistant", "content": response}],
                        chat_template=dummy_dataset.subject_chat_template,
                        add_generation_prompt=False,
                        continue_final_message=True,
                        return_dict=False,
                    ),
                )
                subject_seq_ids_list.append(input_ids[:-1])
                subject_seq_labels_list.append(input_ids[1:])
                target_mask = np.ones(len(input_ids) - 1, dtype=bool)
                target_mask[: len(before_target) - 1] = False
                subject_seq_target_masks_list.append(target_mask)
        subject_seq_ids, subject_seq_attn_mask = pad_sequence(
            sequences=subject_seq_ids_list, pad_token_id=pad_token_id, padding_side="right"
        )
        subject_seq_labels, _ = pad_sequence(
            sequences=subject_seq_labels_list, pad_token_id=-100, padding_side="right"
        )
        subject_seq_target_masks, _ = pad_sequence(
            sequences=subject_seq_target_masks_list,
            pad_token_id=False,
            padding_side="right",
        )

        NUM_SEQS = subject_seq_ids.shape[0]
        repeated_activations = all_activations.tile(NUM_SEQS)
        token_range_from = [(0, T)] * NUM_ACTS * NUM_SEQS
        token_range_to = [(0, T)] * NUM_ACTS * NUM_SEQS
        input_ids = subject_seq_ids.repeat_interleave(NUM_ACTS, dim=0).to(device)
        attention_mask = subject_seq_attn_mask.repeat_interleave(NUM_ACTS, dim=0).to(device)
        target = subject_seq_labels.repeat_interleave(NUM_ACTS, dim=0).to(device)
        target_mask = subject_seq_target_masks.repeat_interleave(NUM_ACTS, dim=0).to(device)

        num_batches = math.ceil(NUM_ACTS * NUM_SEQS / forward_batch_size)
        full_logprobs = []
        top5_token_ids = []
        for batch_idx in tqdm(
            range(num_batches), desc="Evaluating on subject model...", disable=not verbose
        ):
            start_idx = batch_idx * forward_batch_size
            end_idx = min(start_idx + forward_batch_size, NUM_ACTS * NUM_SEQS)

            with torch.inference_mode():
                logits = wrapped_subject.forward_with_intervention(
                    activations=repeated_activations.batch_slice(start_idx, end_idx),
                    token_range_from=token_range_from[start_idx:end_idx],
                    token_range_to=token_range_to[start_idx:end_idx],
                    input_ids=input_ids[start_idx:end_idx],
                    attention_mask=attention_mask[start_idx:end_idx],
                    write_modules=read_modules,
                    layer_mapping={layer: [layer] for layer in read_layers},
                    device=device,
                ).logits
                assert logits is not None

            logprobs = (
                -F.cross_entropy(
                    input=logits.float().transpose(1, 2),
                    target=target[start_idx:end_idx],
                    reduction="none",
                )
                * target_mask[start_idx:end_idx]
            )
            full_logprobs.append(logprobs.sum(dim=1).cpu())
            top5_token_ids.append(
                torch.topk(logits[:, len(before_target) - 1], k=5, dim=1).indices.cpu()
            )

        logprobs = torch.cat(full_logprobs, dim=0)
        probs = logprobs.unflatten(0, (NUM_SEQS, NUM_ACTS)).exp()
        top5_token_ids = torch.cat(top5_token_ids, dim=0).unflatten(
            0, (NUM_SEQS, NUM_ACTS)
        )  # (NUM_SEQS, NUM_ACTS, 5)

        for act_idx in range(NUM_ACTS):
            probs_for_attrs = np.zeros(len(steering_input.attr_class))

            # Compute the sum_i p(expected subject response_i | attr) for each attr.
            start_idx = 0
            for attr_idx, (attr, responses) in enumerate(steering_input.expected_subject_responses):
                end_idx = start_idx + len(responses)

                probs_for_attrs[attr_idx] = probs[start_idx:end_idx, act_idx].sum().item()
                start_idx = end_idx

            probs_sum = probs_for_attrs.sum()

            preds = probs_for_attrs.argmax()
            corrs = torch.arange(len(probs_for_attrs)) == preds

            eval_results[act_idx]["subject_accs"] = corrs.tolist()
            eval_results[act_idx]["subject_probs"] = probs_for_attrs.tolist()
            eval_results[act_idx]["subject_norm_probs"] = (probs_for_attrs / probs_sum).tolist()
            eval_results[act_idx]["subject_logprobs"] = np.log(probs_for_attrs).tolist()
            eval_results[act_idx]["subject_norm_logprobs"] = np.log(
                probs_for_attrs / probs_sum
            ).tolist()
            eval_results[act_idx]["subject_top5_token_ids"] = [
                # We only care about the first sequence because the most likely next token is
                # the same regardless of the continuation.
                tokenizer.decode(tok_id)
                for tok_id in top5_token_ids[0, act_idx].tolist()
            ]

    else:
        assert gen_based_control_eval_config is not None

        completions = steer_and_generate(
            activations=all_activations,
            token_range_from=[(0, T)] * NUM_ACTS,
            token_range_to=[(0, T)] * NUM_ACTS,
            input_ids=cast(list[int], before_target),
            model=wrapped_subject,
            tokenizer=tokenizer,
            write_modules=read_modules,
            layer_mapping={layer: [layer] for layer in read_layers},
            pad_token_id=pad_token_id,
            max_new_tokens=gen_based_control_eval_config.max_new_tokens,
            temperature=gen_based_control_eval_config.temperature,
            num_completions=gen_based_control_eval_config.num_completions,
            device=device,
        )

        if seed_response is not None:
            # Include the seed response in the completions just so that the LLM judge gets
            # the full picture.
            full_completions = []
            for cs in completions:
                full_completions.append([seed_response + c for c in cs])
        else:
            full_completions = completions

        subject_eval_results, _, _ = await evaluate_faithfulness_of_completions(
            eval_inputs=[
                FaithfulnessEvalInput(
                    attr_class=steering_input.attr_class,
                    attr=steering_input.true_attr,
                    target_attr=steering_input.target_attr,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                )
                for _ in range(NUM_ACTS)
            ],
            completions_list=full_completions,
            num_permutations=10,
            llm_judge_provider=gen_based_control_eval_config.llm_judge_provider,
            llm_judge_model_name=gen_based_control_eval_config.llm_judge_model_name,
            max_num_choices=20,
            seed=42,
            max_concurrency=gen_based_control_eval_config.llm_judge_max_concurrency,
            timeout=gen_based_control_eval_config.llm_judge_timeout,
        )

        for act_idx in range(NUM_ACTS):
            metrics = defaultdict(list)
            for result in subject_eval_results[act_idx]:
                assert isinstance(result, LLMBasedFaithfulnessEvalResult)

                logprobs = result.get_mean_logprobs()
                norm_logprobs = torch.log_softmax(torch.tensor(logprobs), dim=-1)
                probs = result.get_mean_probs()
                norm_probs = probs / probs.sum()
                accs = result.get_accuracies()

                metrics["logprobs"].append(logprobs)
                metrics["norm_logprobs"].append(norm_logprobs)
                metrics["probs"].append(probs)
                metrics["norm_probs"].append(norm_probs)
                metrics["accs"].append(accs)
            for metric in metrics:
                eval_results[act_idx][f"subject_{metric}"] = np.mean(
                    np.stack(metrics[metric]), axis=0
                ).tolist()
            eval_results[act_idx]["subject_completions"] = completions[act_idx]
            eval_results[act_idx]["faithfulness_eval_results"] = subject_eval_results[act_idx]

    return eval_results


class SteeringEvalConfig(BaseConfig):
    subject_inputs_data_path: str
    attr_pairs: list[tuple[str, str]]  # [(true_attr, target_attr), ...]
    exp_path: str
    ckpt_name: str
    patch_from_region: PatchFromRegion = PatchFromRegion.USER
    max_subject_inputs_per_attr_pair: int = 10
    steering_qas_path: str | None = None
    mc_stems_path: str
    step_sizes: list[float]
    num_steps: int
    generate_to_eval_control: bool = True
    seed_response_type: Literal["direct", "wiki", "none"]
    gradient_batch_size: int = 16
    forward_batch_size: int = 128
    gen_based_control_eval_config: GenerationBasedControlEvalConfig | None = (
        GenerationBasedControlEvalConfig(
            max_new_tokens=256,
            temperature=1.0,
            num_completions=4,
            llm_judge_provider="openai",
            llm_judge_model_name="gpt-4.1-mini-2025-04-14",
            llm_judge_max_concurrency=20,
            llm_judge_timeout=100.0,
        )
    )
    seed: int = 42
    si_batch_size: int = 32

    @property
    def fields_to_exclude(self):
        return {
            "attr_pairs",
            "max_subject_inputs_per_attr_pair",
            "generate_to_eval_control",
            "gradient_batch_size",
            "forward_batch_size",
            "si_batch_size",
        }

    def results_path(self, eval_type: Literal["reading", "control"]):
        if not self.save_path:
            return None
        if eval_type == "control":
            file_name = "gen_control" if self.generate_to_eval_control else "fixed_control"
        else:
            file_name = "reading"
        return os.path.join(self.save_path, f"{file_name}.pkl")

    def save_results(
        self,
        results: dict[SteeringInput, list[dict[str, Any]]],
        eval_type: Literal["reading", "control"],
    ):
        path = self.results_path(eval_type)
        if not path:
            print("save_path is not set, skipping saving results")
            return
        with open(path, "wb") as f:
            pickle.dump(results, f)

    def load_results(
        self, eval_type: Literal["reading", "control"]
    ) -> dict[SteeringInput, list[dict[str, Any]]] | None:
        path = self.results_path(eval_type)
        if not path or not os.path.exists(path):
            return None
        with open(path, "rb") as f:
            return pickle.load(f)


async def do_steering_on_batch(
    sis_to_get_activations_for: list[SteeringInput],
    config: SteeringEvalConfig,
    exp_config: LQATrainConfig,
    wrapped_subject: ModelWrapper,
    wrapped_decoder: ModelWrapper,
    tokenizer: PreTrainedTokenizer,
    save_dir: Path,
    verbose: bool = False,
):
    if (save_dir / "losses.pkl").exists():
        with open(save_dir / "losses.pkl", "rb") as f:
            losses_by_steering_input = pickle.load(f)
    else:
        losses_by_steering_input: dict[SteeringInput, list[dict[str, Any]]] = {}

    checkpoints_by_steering_input: dict[
        SteeringInput, tuple[list[dict[str, Any]], tuple[int, int]]
    ] = {}

    for steering_input in tqdm(sis_to_get_activations_for, desc="Steering..."):
        checkpoints, token_range_from = get_activation_checkpoints_from_gradient_based_steering(
            steering_input=steering_input,
            patch_from_region=config.patch_from_region,
            exp_config=exp_config,
            wrapped_subject=wrapped_subject,
            wrapped_decoder=wrapped_decoder,
            tokenizer=tokenizer,
            step_sizes=config.step_sizes,
            num_steps=config.num_steps,
            grad_batch_size=config.gradient_batch_size,
            forward_batch_size=config.forward_batch_size,
            device="cuda:0",
            verbose=verbose,
        )

        # token_range_from is used for evaluating reading.
        checkpoints_by_steering_input[steering_input] = (checkpoints, token_range_from)

        # Save loss separately so that we don't have to load the activations.
        losses_by_steering_input[steering_input] = [
            {key: ckpt[key] for key in ckpt if key != "activations"} for ckpt in checkpoints
        ]

        with open(save_dir / "losses.pkl", "wb") as f:
            pickle.dump(losses_by_steering_input, f)

    #########################################################
    # Do steering eval.
    #########################################################
    for eval_type in ["reading", "control"]:
        # Load pre-existing eval results.
        existing_eval_results = config.load_results(eval_type)
        if existing_eval_results is None:
            existing_eval_results = {}

        for steering_input in tqdm(
            sis_to_get_activations_for,
            desc=f"Evaluating {eval_type}...",
        ):
            if steering_input in existing_eval_results:
                continue
            checkpoints, token_range_from = checkpoints_by_steering_input[steering_input]

            all_activations = ActivationsList.concat(
                [ckpt["activations"].to_activations_list() for ckpt in checkpoints]
            )
            assert all_activations.batch_size() == len(checkpoints)

            if eval_type == "reading":
                eval_results = evaluate_reading_after_steering(
                    steering_input=steering_input,
                    token_range_from=token_range_from,
                    all_activations=all_activations,
                    patch_from_region=config.patch_from_region,
                    exp_config=exp_config,
                    wrapped_decoder=wrapped_decoder,
                    tokenizer=tokenizer,
                    forward_batch_size=config.forward_batch_size,
                    device="cuda:0",
                )
            elif eval_type == "control":
                eval_results = await evaluate_control(
                    steering_input=steering_input,
                    all_activations=all_activations,
                    patch_from_region=config.patch_from_region,
                    exp_config=exp_config,
                    wrapped_subject=wrapped_subject,
                    tokenizer=tokenizer,
                    generate_to_eval_control=config.generate_to_eval_control,
                    forward_batch_size=config.forward_batch_size,
                    gen_based_control_eval_config=config.gen_based_control_eval_config,
                    device="cuda:0",
                    verbose=verbose,
                )
            else:
                raise ValueError(f"Invalid eval_type: {eval_type}")

            existing_eval_results[steering_input] = eval_results

            config.save_results(
                existing_eval_results, cast(Literal["reading", "control"], eval_type)
            )

    del checkpoints_by_steering_input
    gc.collect()
    torch.cuda.empty_cache()


async def do_steering_eval(
    config: SteeringEvalConfig,
    decoder: PeftModel | None = None,
    subject: PreTrainedModel | None = None,
    steering_qas_by_attr_pairs: (
        dict[tuple[UserAttribute, UserAttribute], list[tuple[str, str]]] | None
    ) = None,
    verbose: bool = False,
):
    assert config.save_path is not None
    save_dir = Path(config.save_path)
    save_dir.mkdir(parents=True, exist_ok=True)
    print("saving config")
    config.save_config()
    print("#" * 100)
    print(f"Running steering eval for {config.exp_path}/{config.ckpt_name}...")
    print("#" * 100)

    np.random.seed(config.seed)

    # Load the experiment config.

    with open(Path(config.exp_path) / LQATrainConfig.config_file_name, "r") as f:
        exp_config: LQATrainConfig = LQATrainConfig.model_validate_json(json.load(f))
    tokenizer = get_tokenizer(exp_config.hf_model_id, exp_config.new_context_tokens)

    #########################################################
    # Get steering inputs ready.
    #########################################################

    # Load the subject inputs.
    all_subject_inputs = pickle.load(open(config.subject_inputs_data_path, "rb"))
    all_subject_inputs = [LQAData(**d.model_dump()) for d in all_subject_inputs]
    subject_inputs_by_attr = defaultdict(set)
    for d in all_subject_inputs:
        subject_inputs_by_attr[d.attr].add((d.system_prompt, d.user_prompt))

    # Get things ready for generating test qas.
    stems_by_attr_class = pickle.load(open(config.mc_stems_path, "rb"))

    # Collect all steering inputs.
    steering_inputs = []
    for true_attr_str, target_attr_str in config.attr_pairs:
        true_attr = resolve_attr(true_attr_str)
        target_attr = resolve_attr(target_attr_str)
        assert true_attr.__class__ == target_attr.__class__
        attr_class = true_attr.__class__

        subject_inputs = sorted(list(subject_inputs_by_attr[true_attr]))
        subject_inputs = np.random.permutation(subject_inputs)
        subject_inputs = subject_inputs[: config.max_subject_inputs_per_attr_pair]

        def get_answers(attr: UserAttribute) -> list[str]:
            return list(
                set(
                    [
                        " " + attr.value_name.capitalize(),
                        " " + attr.value_name.title(),
                        " " + attr.value_name.lower(),
                    ]
                )
            )

        if config.seed_response_type == "direct":
            seed_response = f"I will base my response on {attr_class.user_reference}, which is"
            subject_responses = [(attr, get_answers(attr)) for attr in attr_class]
        elif config.seed_response_type == "wiki":
            seed_response = (
                WIKIPEDIA_PROMPT_RESPONSE + ATTRIBUTE_PROMPTS[attr_class.__name__.lower()]
            )
            subject_responses = [(attr, get_answers(attr)) for attr in attr_class]
        elif config.seed_response_type == "none":
            assert (
                config.generate_to_eval_control
            ), "We need seed response for logprob-based control eval."
            seed_response = None
            subject_responses = None

        for system_prompt, user_prompt in subject_inputs:
            validation_qas = None
            if (
                steering_qas_by_attr_pairs is not None
                and (true_attr, target_attr) in steering_qas_by_attr_pairs
            ):
                steering_qas = steering_qas_by_attr_pairs[(true_attr, target_attr)]
            else:
                assert config.steering_qas_path is not None
                fname = f"{str(true_attr)}_to_{str(target_attr)}.pkl"

                train_path = os.path.join(config.steering_qas_path, f"train_{fname}")
                valid_path = os.path.join(config.steering_qas_path, f"valid_{fname}")
                if not os.path.exists(train_path) or not os.path.exists(valid_path):
                    print(f"Skipping {true_attr} to {target_attr} because it doesn't exist")
                    continue

                with open(train_path, "rb") as f:
                    steering_qas = pickle.load(f)
                with open(valid_path, "rb") as f:
                    validation_qas = tuple(pickle.load(f))

            steering_input = SteeringInput(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                true_attr=true_attr,
                target_attr=target_attr,
                steering_qas=tuple(steering_qas),
                validation_qas=validation_qas,
                subject_seed_response=seed_response,
                expected_subject_responses=subject_responses,
                # Fixing the seed here makes us always get the same same set of multiple choice
                # questions given a fixed attribute class.
                test_qas=tuple(
                    get_test_mc_qas(
                        true_attr,
                        target_attr,
                        stems_by_attr_class=stems_by_attr_class,
                        rng=np.random.default_rng(0),
                    )
                ),
            )

            steering_inputs.append(steering_input)

    #########################################################
    # Load models.
    #########################################################
    if subject is None:
        subject = cast(
            PreTrainedModel,
            get_subject_model(
                hf_model_id=exp_config.hf_model_id,
                use_fsdp=False,
                device="auto",
                inference_only=True,
                fsdp_config=None,
            ),
        )

    if decoder is None:
        decoder = cast(
            PeftModel,
            get_decoder_model(
                hf_model_id=exp_config.hf_model_id,
                tokenizer=tokenizer,
                use_peft=exp_config.use_peft,
                use_fsdp=False,
                device="auto",
                decoder_checkpoint_path=str(
                    Path(config.exp_path) / "checkpoints" / config.ckpt_name
                ),
                fsdp_config=None,
                lora_config=exp_config.lora_config,
            ),
        )

    wrapped_subject = ModelWrapper(subject, exp_config.hf_model_id, "cuda:0")
    wrapped_decoder = ModelWrapper(decoder, exp_config.hf_model_id, "cuda:0")

    #########################################################
    # Go through any existing eval results and collect new steering inputs to evaluate.
    #########################################################
    reading_eval_results = config.load_results("reading")
    control_eval_results = config.load_results("control")

    if reading_eval_results is not None and control_eval_results is not None:
        already_evaluated_sis = set(reading_eval_results.keys()).intersection(
            set(control_eval_results.keys())
        )
        sis_to_get_activations_for = list(set(steering_inputs) - already_evaluated_sis)
    else:
        sis_to_get_activations_for = steering_inputs

    print(f"Going to evaluate {len(sis_to_get_activations_for)} steering inputs.")

    #########################################################
    # Go through steering inputs in batches to save memory.
    #########################################################
    for start in tqdm(
        range(0, len(sis_to_get_activations_for), config.si_batch_size),
        desc="Going through steering inputs in batches...",
    ):
        sis = sis_to_get_activations_for[start : start + config.si_batch_size]
        await do_steering_on_batch(
            sis_to_get_activations_for=sis,
            config=config,
            exp_config=exp_config,
            wrapped_subject=wrapped_subject,
            wrapped_decoder=wrapped_decoder,
            tokenizer=tokenizer,
            save_dir=save_dir,
        )
        gc.collect()
        torch.cuda.empty_cache()

    del subject
    del decoder
    gc.collect()
    torch.cuda.empty_cache()


def parse_steering_eval_results(
    losses: dict[SteeringInput, list[dict[str, Any]]],
    reading_results: dict[SteeringInput, list[dict[str, Any]]],
    fixed_control_results: dict[SteeringInput, list[dict[str, Any]]] | None,
    gen_control_results: dict[SteeringInput, list[dict[str, Any]]] | None = None,
) -> pd.DataFrame:
    all_results = []
    for steering_input_id, steering_input in enumerate(losses.keys()):
        attr_class = steering_input.attr_class
        true_attr = steering_input.true_attr
        target_attr = steering_input.target_attr

        true_attr_idx = list(attr_class).index(true_attr)
        target_attr_idx = list(attr_class).index(target_attr)

        losses_for_si = losses[steering_input]
        num_ckpts = len(losses_for_si)

        reading_for_si = reading_results[steering_input]
        fixed_control_for_si = (
            fixed_control_results[steering_input]
            if fixed_control_results is not None
            else [None] * num_ckpts
        )
        gen_control_for_si = (
            gen_control_results[steering_input]
            if gen_control_results is not None
            else [None] * num_ckpts
        )

        for loss, reading, fixed_control, gen_control in zip(
            losses_for_si, reading_for_si, fixed_control_for_si, gen_control_for_si
        ):
            parsed_eval_result = {}

            # Add results from steering.
            step = None if "step" not in loss else loss["step"]
            step_size = None if "step_size" not in loss else loss["step_size"]
            assert step is not None or step_size is not None
            step = step if step is not None else step_size
            parsed_eval_result.update(
                {
                    "tag": steering_input_id,
                    "steering_step": step,
                    "train_loss": loss["train_loss"],
                    "eval_loss": loss["eval_loss"] if "eval_loss" in loss else None,
                    "true_attr": true_attr,
                    "target_attr": target_attr,
                    "attr_class": attr_class,
                }
            )

            # Add results from reading.
            parsed_eval_result.update(
                {
                    "reading_acc_target": reading["decoder_accs"][target_attr_idx],
                    "reading_acc_orig": reading["decoder_accs"][true_attr_idx],
                    "reading_prob_target": reading["decoder_probs"][target_attr_idx],
                    "reading_prob_orig": reading["decoder_probs"][true_attr_idx],
                    "reading_norm_prob_target": reading["decoder_norm_probs"][target_attr_idx],
                    "reading_norm_prob_orig": reading["decoder_norm_probs"][true_attr_idx],
                }
            )

            # Add results from control.
            for control_type, control in [("fixed", fixed_control), ("gen", gen_control)]:
                if control is None:
                    continue
                parsed_eval_result.update(
                    {
                        f"{control_type}_acc_target": control["subject_accs"][target_attr_idx],
                        f"{control_type}_acc_orig": control["subject_accs"][true_attr_idx],
                        f"{control_type}_prob_target": control["subject_probs"][target_attr_idx],
                        f"{control_type}_prob_orig": control["subject_probs"][true_attr_idx],
                        f"{control_type}_norm_prob_target": control["subject_norm_probs"][
                            target_attr_idx
                        ],
                        f"{control_type}_norm_prob_orig": control["subject_norm_probs"][
                            true_attr_idx
                        ],
                    }
                )
                if "most_likely_first_token" in control:
                    parsed_eval_result["most_likely_first_token"] = control[
                        "most_likely_first_token"
                    ]
                if "subject_completions" in control:
                    parsed_eval_result["completions"] = control["subject_completions"]

            all_results.append(parsed_eval_result)
    return pd.DataFrame(all_results)
