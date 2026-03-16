import json
import math
import os
import time
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
import wandb
from latentqa.eval.reading_eval import do_logprob_eval, do_single_token_multiple_choice_eval
from latentqa.lqa_data import LQAData, load_lqa_data
from latentqa.model_wrapper import ModelWrapper
from latentqa.train_util import (
    CosineLRScheduler,
    LQATrainConfig,
    compute_loss,
    get_dataloader,
    get_decoder_model,
    get_num_patch_tokens_multiplier,
    get_subject_model,
    get_tokenizer,
    save_decoder_and_optimizer_checkpoint,
)
from torch.distributed import (
    ReduceOp,
    all_reduce,
    barrier,
    destroy_process_group,
    init_process_group,
)
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.api import FullOptimStateDictConfig, FullStateDictConfig, StateDictType
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from wandb.sdk.lib.runid import generate_id as generate_wandb_id


def train(config: LQATrainConfig):
    ###### Set up distributed training ######
    multi_gpu = int(os.environ.get("RANK", -1)) != -1  # is this a distributed run?
    if multi_gpu:
        init_process_group(backend="nccl")
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        device = f"cuda:{rank}"
        torch.cuda.set_device(device)
        master_process = rank == 0  # this process will do logging, checkpointing etc.
    else:
        device = "cuda"
        master_process = True
        rank = 0
        world_size = 1
    ddp = multi_gpu and not config.use_fsdp

    # Set parameters that depend on the number of GPUs.
    if config.gradient_accumulation_steps is None:
        assert config.batch_size is not None, "Batch size is not set!"
        config.gradient_accumulation_steps = math.ceil(
            config.batch_size / (world_size * config.micro_batch_size)
        )
    else:
        assert config.batch_size is None, "Both batch_size and gradient_accumulation_steps are set!"
    if config.eval_num_steps is None:
        assert config.eval_num_datapoints is not None, "Eval num datapoints is not set!"
        config.eval_num_steps = math.ceil(
            config.eval_num_datapoints / (world_size * config.eval_batch_size)
        )
    else:
        assert (
            config.eval_num_datapoints is None
        ), "Both eval_num_datapoints and eval_num_steps are set!"

    if master_process:
        config.save_config()

    rng = np.random.default_rng(config.seed)
    torch.manual_seed(config.seed)  # This is to make the model initialization deterministic.
    # To my knowledge, all other randomness is delt with by the rng above.
    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn

    load_result = config.get_most_recent_checkpoint()
    if load_result is not None:
        checkpoint_to_load, most_recent_step = load_result
        start_step = most_recent_step + 1

        if start_step >= config.num_steps:
            print("Latest checkpoint is already the final checkpoint. Exiting...")
            return

        if rank == 0:
            print(f"Loading checkpoint from {checkpoint_to_load}, with start step {start_step}.")
    else:
        checkpoint_to_load = None
        start_step = 0

    ###### Load the models ######
    tokenizer = get_tokenizer(config.hf_model_id, config.new_context_tokens)
    subject = get_subject_model(
        hf_model_id=config.hf_model_id,
        use_fsdp=config.use_fsdp,
        device=device,
        fsdp_config=config.fsdp_config,
    )
    decoder = get_decoder_model(
        hf_model_id=config.hf_model_id,
        tokenizer=tokenizer,
        use_peft=config.use_peft,
        use_fsdp=config.use_fsdp,
        device=device,
        decoder_checkpoint_path=checkpoint_to_load,
        fsdp_config=config.fsdp_config,
        lora_config=config.lora_config,
    )
    if ddp:
        decoder = DDP(decoder, device_ids=[rank])

    wrapped_subject = ModelWrapper(subject, config.hf_model_id, device)
    wrapped_decoder = ModelWrapper(decoder, config.hf_model_id, device)

    read_modules = wrapped_subject.get_modules(config.get_read_layer_module_keys())
    write_modules = wrapped_decoder.get_modules(config.get_write_layer_module_keys())
    layer_mapping = config.get_layer_mapping()

    ###### Define the optimizer and scheduler ######
    trainable_params = [
        p for _, p in cast(torch.nn.Module, decoder).named_parameters() if p.requires_grad
    ]
    if master_process:
        print("Number of trainable parameters:", sum(p.numel() for p in trainable_params))
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    scheduler = CosineLRScheduler(
        optimizer=optimizer,
        num_steps=config.num_steps,
        lr=config.lr,
        min_lr_ratio=config.min_lr_ratio,
        warmup_steps=config.warmup_steps,
    )
    # Load the optimizer state if it exists.
    other_state_dict = {}
    if checkpoint_to_load is not None:
        scheduler.curr_step = start_step
        scheduler.apply_lr_to_optimizer(scheduler.compute_lr(start_step))

        optim_state_dict = torch.load(
            os.path.join(checkpoint_to_load, "optim.pt"), weights_only=True
        )
        if config.use_fsdp:
            with FSDP.state_dict_type(
                cast(torch.nn.Module, decoder),
                StateDictType.FULL_STATE_DICT,
                FullStateDictConfig(rank0_only=False),
                FullOptimStateDictConfig(rank0_only=False),
            ):
                optim_state_dict = FSDP.optim_state_dict_to_load(
                    model=cast(torch.nn.Module, decoder),
                    optim=optimizer,
                    optim_state_dict=optim_state_dict,
                )
        optimizer.load_state_dict(optim_state_dict)
        if config.use_wandb:
            with open(os.path.join(checkpoint_to_load, "misc.json"), "r") as f:
                other_state_dict = json.load(f)

    ###### Set up the dataloaders ######
    train_data = load_lqa_data(config.train_data_path, LQAData)
    valid_data = (
        load_lqa_data(config.valid_data_path, LQAData)
        if config.valid_data_path is not None
        else None
    )

    train_dataloader = get_dataloader(
        data=train_data,
        hf_model_id=config.hf_model_id,
        patch_from_regions=config.train_patch_regions,
        decoder_base_conv_format=config.decoder_base_conv_format,
        decoder_assistance_role=config.decoder_assistance_role,
        decoder_base_conv_format_kwargs=config.decoder_base_conv_format_kwargs,
        tokenizer=tokenizer,
        generate=False,
        shuffle=True,
        rng=rng,
        batch_size=config.micro_batch_size,
        rank=rank,
        world_size=world_size,
        drop_last=True,
        num_patch_tokens_multiplier=get_num_patch_tokens_multiplier(layer_mapping),
    )

    def get_train_datastream():
        while True:
            for batch in train_dataloader:
                yield batch

    train_datastream = get_train_datastream()

    ###### Set up logging ######
    wandb_id = None
    save_path = Path(config.save_path)
    base_exp_name = save_path.parent.name
    exp_name = save_path.name
    base_checkpoint_path = Path(config.base_checkpoint_path)
    os.path.join(config.save_path, "eval_results")
    if master_process:
        save_path.mkdir(parents=True, exist_ok=True)
        base_checkpoint_path.mkdir(parents=True, exist_ok=True)
        if config.use_wandb:
            print("Initializing wandb run.")
            wandb_id = other_state_dict.get("wandb_id", generate_wandb_id())
            wandb.init(
                project=config.wandb_project,
                resume="allow",
                id=wandb_id,
                group=config.wandb_group_name,
                name=config.wandb_run_name,
            )
    eval_data = {}
    eval_save_paths = {}
    for eval_data_name, (_, eval_data_path) in config.eval_data.items():
        eval_data_path = Path(eval_data_path)
        eval_data[eval_data_name] = load_lqa_data(eval_data_path, LQAData)

        eval_save_path = (
            eval_data_path.parent / f"{eval_data_name}_eval_results" / f"{base_exp_name}:{exp_name}"
        )
        eval_save_path.mkdir(parents=True, exist_ok=True)
        eval_save_paths[eval_data_name] = eval_save_path

    ###### Start training ######
    time_per_step = 0.0
    for step in tqdm(range(config.num_steps), disable=not master_process, desc="Training"):
        start_time = time.time()
        batch_loss = 0.0
        for micro_step in range(config.gradient_accumulation_steps):
            batch = next(train_datastream)

            # For reproducibility when resuming training.
            if step < start_step:
                continue

            if multi_gpu:
                assert isinstance(decoder, DDP)
                decoder.require_backward_grad_sync = (
                    micro_step == config.gradient_accumulation_steps - 1
                )

            unreduced_loss = compute_loss(
                batch=batch,
                subject=wrapped_subject,
                decoder=wrapped_decoder,
                read_modules=read_modules,
                write_modules=write_modules,
                layer_mapping=layer_mapping,
                device=device,
            )  # shape: (batch size, seq len)

            loss = unreduced_loss.sum(dim=1).mean() / config.gradient_accumulation_steps
            loss.backward()

            batch_loss += loss.detach().float()

        # For reproducibility when resuming training.
        if step < start_step:
            continue

        if multi_gpu:
            all_reduce(batch_loss, op=ReduceOp.SUM)

        # Clip the gradient.
        if config.gradient_clipping_threshold != 0.0:
            torch.nn.utils.clip_grad_norm_(trainable_params, config.gradient_clipping_threshold)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        time_per_step = (time_per_step * step + (time.time() - start_time)) / (step + 1)

        ###### Logging (master only) ######
        if master_process and (step + 1) % config.log_interval == 0:
            if config.use_wandb:
                wandb.log(
                    {
                        "step": step,
                        "train/loss": batch_loss / world_size,
                        "train/lr": scheduler.compute_lr(scheduler.curr_step),
                        "train/time_per_step": time_per_step,
                    }
                )

        ###### Evaluation (all ranks) ######
        if (step + 1) % config.eval_interval == 0 or step == 0:
            if master_process:
                print("Evaluating...")

            # Do logprob evaluation on validation set.
            if config.valid_data_path is not None:
                assert valid_data is not None

                t0 = time.time()
                eval_loss, _ = do_logprob_eval(
                    data=valid_data,
                    hf_model_id=config.hf_model_id,
                    subject=wrapped_subject,
                    decoder=wrapped_decoder,
                    tokenizer=tokenizer,
                    read_modules=read_modules,
                    write_modules=write_modules,
                    layer_mapping=layer_mapping,
                    eval_batch_size=config.eval_batch_size,
                    eval_num_steps=config.eval_num_steps,
                    drop_last=True,
                    patch_from_regions=config.eval_patch_regions,
                    decoder_base_conv_format=config.decoder_base_conv_format,
                    decoder_assistance_role=config.decoder_assistance_role,
                    decoder_base_conv_format_kwargs=config.decoder_base_conv_format_kwargs,
                    device=device,
                    rank=rank,
                    world_size=world_size,
                    multi_gpu=multi_gpu,
                )

                if master_process:
                    if config.use_wandb:
                        wandb.log({"step": step, "valid/loss": eval_loss["loss"]})
                    print(f"Time taken for logprob evaluation: {time.time() - t0} seconds.")

            # Do evaluation on multiple choice datasets.
            for eval_data_name, (eval_type, _) in config.eval_data.items():
                rank_save_path = eval_save_paths[eval_data_name] / f"rank{rank}_step_{step}.jsonl"
                # Delete pre-existing results.
                if rank_save_path.exists():
                    os.remove(rank_save_path)

                first_qa = eval_data[eval_data_name][0].qa
                if eval_type == "logprob":
                    eval_fn = do_logprob_eval
                elif eval_type == "single_token_mc":
                    eval_fn = do_single_token_multiple_choice_eval
                else:
                    raise ValueError(f"Invalid eval type: {first_qa.qa_category}")

                t0 = time.time()
                agg_results_dict, _ = eval_fn(
                    data=eval_data[eval_data_name],
                    hf_model_id=config.hf_model_id,
                    patch_from_regions=config.eval_patch_regions,
                    decoder_base_conv_format=config.decoder_base_conv_format,
                    decoder_assistance_role=config.decoder_assistance_role,
                    decoder_base_conv_format_kwargs=config.decoder_base_conv_format_kwargs,
                    subject=wrapped_subject,
                    decoder=wrapped_decoder,
                    tokenizer=tokenizer,
                    read_modules=read_modules,
                    write_modules=write_modules,
                    layer_mapping=layer_mapping,
                    eval_batch_size=config.eval_batch_size,
                    eval_num_steps=config.eval_num_steps,
                    drop_last=True,
                    device=device,
                    rank=rank,
                    world_size=world_size,
                    multi_gpu=multi_gpu,
                    save_path=str(rank_save_path),
                    progress_bar=False,
                )

                if master_process:
                    if config.use_wandb:
                        eval_label = (
                            config.eval_name_mapping[eval_data_name]
                            if config.eval_name_mapping is not None
                            else eval_data_name
                        )

                        to_log: dict[str, Any] = {"step": step}
                        for key, value in agg_results_dict.items():
                            to_log[f"{eval_label}/{key}"] = value
                        wandb.log(to_log)
                    print(
                        f"Time taken for {eval_data_name} evaluation: {time.time() - t0} seconds."
                    )
                torch.cuda.empty_cache()

            # Ensure all processes have finished evaluation before continuing.
            if multi_gpu:
                barrier()

        ###### Checkpoint saving ######
        if (step + 1) % config.save_interval == 0:
            save_decoder_and_optimizer_checkpoint(
                decoder,
                str(base_checkpoint_path),
                f"step_{step}",
                config.use_fsdp,
                optimizer,
                wandb_id,
                master_process,
                ddp,
            )

    if config.save_final_checkpoint:
        save_decoder_and_optimizer_checkpoint(
            decoder,
            str(base_checkpoint_path),
            "final",
            config.use_fsdp,
            optimizer,
            wandb_id,
            master_process,
            ddp,
        )

    if config.use_wandb:
        wandb.finish()

    if multi_gpu:
        destroy_process_group()
