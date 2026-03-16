import json
import math
import os
import time
from functools import partial
from pathlib import Path
from typing import Any, Callable, Literal, Sequence, cast

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from latentqa.config_util import BaseConfig
from latentqa.data_util import LatentQADataset, LQABatch, MinimalDatapoint, PatchFromRegion
from latentqa.lqa_data import LQAData
from latentqa.model_wrapper import (
    ActivationsList,
    BaseModelWithGenerate,
    LayerModuleKey,
    ModelWrapper,
    Modules,
)
from peft import LoraConfig, PeftModel, get_peft_model
from peft.tuners import PrefixEncoder, PromptEmbedding, PromptEncoder
from pydantic import BaseModel
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)
from torch.distributed.fsdp import FullOptimStateDictConfig
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy
from torch.distributed.fsdp.api import CPUOffload, FullStateDictConfig, StateDictType
from torch.distributed.fsdp.wrap import (
    _or_policy,
    lambda_auto_wrap_policy,
    transformer_auto_wrap_policy,
)
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers.tokenization_utils import PreTrainedTokenizer
from user_modeling.model_util import MODEL_TYPES


class CosineLRScheduler:
    def __init__(
        self,
        optimizer,
        num_steps: int,
        lr: float = 5e-6,
        min_lr_ratio: float = 0.1,
        warmup_steps: int = 100,
    ) -> None:

        self.curr_step = 0
        self.optimizer = optimizer
        self.num_steps = num_steps
        self.lr = lr
        self.min_lr = min_lr_ratio * lr
        self.warmup_steps = max(1, warmup_steps)

        self.apply_lr_to_optimizer(self.compute_lr(step=0))

    def compute_lr(self, step: int) -> float:
        progress_fraction = step / self.num_steps

        scale_factor = (1 + math.cos(math.pi * progress_fraction)) / 2
        lr = self.min_lr + ((self.lr - self.min_lr) * scale_factor)

        return lr * min(step / self.warmup_steps, 1)

    def step(self) -> None:
        self.curr_step += 1
        self.apply_lr_to_optimizer(self.compute_lr(self.curr_step))

    def apply_lr_to_optimizer(self, lr: float) -> None:
        for group in self.optimizer.param_groups:
            group["lr"] = lr


class FSDPConfig(BaseModel):
    sharding_strategy: ShardingStrategy = ShardingStrategy.FULL_SHARD
    """
    Require HYBRID_SHARD to be set. This flag can extend the HYBRID_SHARD by allowing sharding
    a model on customized number of GPUs (Sharding_group) and Replicas over Sharding_group.
    """
    sharding_group_size: int = 0
    """
    Requires hsdp to be set. This specifies the sharding group size, number of GPUs that
    you model can fit into to form a replica of a model.
    """
    replica_group_size: int = 0
    """
    Requires hsdp to be set. This specifies the replica group size,
    which is world_size/sharding_group_size.
    """
    fsdp_activation_checkpointing: bool = True
    fsdp_cpu_offload: bool = False


class OurLoRAConfig(BaseModel):
    lora_r: int = 16
    lora_alpha: int = 32
    lora_target_modules: list[str] = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "lm_head",
    ]
    lora_bias: Literal["none", "all", "lora_only"] = "none"
    lora_task_type: str = "CAUSAL_LM"
    lora_dropout: float = 0.05


class PretrainDataConfig(BaseModel):
    dataset_name: str
    pretrain_mix: float = 1.0
    seq_len: int = 32


class ConceptEncoderConfig(BaseModel):
    num_concepts: int = 65536
    top_k: int = 32
    use_lqa_format: bool = False
    anneal: bool = False
    normalize: bool = False
    posttrain: bool = False
    aux_loss: bool = False


class LQATrainConfig(BaseConfig):
    hf_model_id: str

    # Data parameters.
    train_data_path: str
    """Path to pickle file containing a list of LQATrainingData objects."""
    valid_data_path: str | None = None
    """
    Path to pickle file containing a list of LQATrainingData objects.
    We use this for logprob-based evaluation.
    """
    eval_data: dict[str, tuple[str, str]]
    """
    dictionary with keys being the names of the evaluation datasets, and values being
    tuples of (reading eval type, path to evaluation data).
    The reading eval type can be "logprob", "single_token_mc".
    The path to the evaluation data is a pickle file containing a list of LQAData objects.
    """
    eval_name_mapping: dict[str, str] | None = None
    """
    dictionary with keys being the names of the evaluation datasets, and values being the name
    we want to be logged to wandb. This is useful for comparing against experiments on wandb
    without having to rename the evaluation dataset files.
    """
    train_patch_regions: list[PatchFromRegion] = [PatchFromRegion.USER, PatchFromRegion.SYSTEM_USER]
    eval_patch_regions: list[PatchFromRegion] = [PatchFromRegion.USER]

    # Formatting of decoder inputs.
    decoder_base_conv_format: Literal["default", "role_change", "act_in_user"] = "default"
    decoder_assistance_role: str = "reflect"
    new_context_tokens: list[str] | None = None
    decoder_base_conv_format_kwargs: dict[str, Any] | None = None

    # Patching parameters.
    read_layer_module_keys: Sequence[LayerModuleKey] | None = None
    write_layer_module_keys: Sequence[LayerModuleKey] | None = None
    layer_to_read: int | None = 21
    layer_to_write: int | None = 1
    layer_mapping: dict[int, int] | dict[int, list[int]] | None = None

    # Train parameters.
    batch_size: int | None = None
    micro_batch_size: int = 4
    gradient_accumulation_steps: int | None = 8
    """This will be done per GPU."""
    gradient_clipping: bool = False
    gradient_clipping_threshold: float = 1.0

    num_steps: int = 10_000
    lr: float = 3e-5
    min_lr_ratio: float = 0.1
    weight_decay: float = 0.01
    dropout: float = 0.0
    warmup_steps: int = 0
    seed: int = 64

    # LoRA parameters.
    use_peft: bool = True
    lora_config: OurLoRAConfig = OurLoRAConfig()

    # FSDP parameters.
    use_fsdp: bool = False
    fsdp_config: FSDPConfig = FSDPConfig()
    concept_encoder_config: ConceptEncoderConfig | None = None

    # Evaluation and logging parameters.
    save_path: str = ""
    eval_num_datapoints: int | None = None
    eval_batch_size: int = 16
    eval_num_steps: int | None = 1
    eval_interval: int = 100
    save_interval: int = 100
    save_final_checkpoint: bool = True
    log_interval: int = 100
    use_wandb: bool = False
    wandb_group_name: str | None = None
    wandb_project: str = "latentqa"
    wandb_run_name: str | None = None

    @property
    def fields_to_exclude(self):
        fields = {
            "micro_batch_size",
            "gradient_accumulation_steps",
            "eval_batch_size",
            "eval_num_steps",
            "eval_interval",
            "save_interval",
            "log_interval",
            "use_wandb",
        }
        # If we don't decay the learning rate we can always change the number of steps we train for.
        if self.min_lr_ratio == 1.0:
            fields.add("num_steps")
        return fields

    @property
    def base_checkpoint_path(self) -> str:
        return os.path.join(self.save_path, "checkpoints")

    def get_checkpoint_name(self, step: int) -> str:
        return f"step_{step}"

    def get_most_recent_checkpoint(self) -> tuple[str, int] | None:
        base_ckpt_path = Path(self.base_checkpoint_path)

        if not os.path.exists(base_ckpt_path):
            return None
        checkpoints_by_step = {}
        for f in os.listdir(base_ckpt_path):
            if f.startswith("step_"):
                if not os.path.isfile(base_ckpt_path / f / "optim.pt"):
                    continue
                checkpoints_by_step[int(f.split("_")[-1])] = f
        if not checkpoints_by_step:
            return None
        most_recent_step = max(checkpoints_by_step.keys())
        return (
            os.path.join(self.base_checkpoint_path, checkpoints_by_step[most_recent_step]),
            most_recent_step,
        )

    @classmethod
    def from_yaml(cls, yaml_path: Path | os.PathLike) -> "LQATrainConfig":
        with open(yaml_path, "r") as f:
            config_dict = yaml.load(f, Loader=yaml.FullLoader)
        return cls(**config_dict)

    def get_read_layer_module_keys(self) -> Sequence[LayerModuleKey]:
        if self.read_layer_module_keys is not None:
            return self.read_layer_module_keys
        assert self.layer_to_read is not None
        return [LayerModuleKey(layer=self.layer_to_read, module="residual")]

    def get_write_layer_module_keys(self) -> Sequence[LayerModuleKey]:
        if self.write_layer_module_keys is not None:
            return self.write_layer_module_keys
        assert self.layer_to_write is not None
        return [LayerModuleKey(layer=self.layer_to_write, module="residual")]

    def get_layer_mapping(self) -> dict[int, list[int]]:
        if self.layer_mapping is None:
            assert self.layer_to_read is not None and self.layer_to_write is not None
            return {self.layer_to_write: [self.layer_to_read]}

        upd_layer_mapping = {}
        for k, v in self.layer_mapping.items():
            if isinstance(v, int):
                upd_layer_mapping[k] = [v]
            elif isinstance(v, list):
                upd_layer_mapping[k] = v
            else:
                raise ValueError(f"Invalid layer mapping: {v}")
        return upd_layer_mapping


def get_num_patch_tokens_multiplier(layer_mapping: dict[int, list[int]]) -> int:
    """Support for a single write layer to read from multiple read layers ("flattening")."""
    num_read_layers_per_write_layer = set(len(v) for v in layer_mapping.values())
    assert len(num_read_layers_per_write_layer) == 1
    num_patch_tokens_multiplier = num_read_layers_per_write_layer.pop()
    return num_patch_tokens_multiplier


def get_tokenizer(hf_model_id: str, new_tokens: list[str] | None = None) -> PreTrainedTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(hf_model_id)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    if new_tokens is not None:
        tokenizer.add_tokens(new_tokens)
    return tokenizer


def get_dataloader(
    data: list[LQAData] | list[MinimalDatapoint],
    hf_model_id: str,
    tokenizer: PreTrainedTokenizer,
    generate: bool,
    batch_size: int,
    shuffle: bool,
    patch_from_regions: list[PatchFromRegion] | None = None,
    decoder_base_conv_format: Literal["default", "role_change", "act_in_user"] = "default",
    decoder_assistance_role: str = "reflect",
    decoder_base_conv_format_kwargs: dict[str, Any] | None = None,
    rng: np.random.Generator | None = None,
    rank: int = 0,
    world_size: int = 1,
    drop_last: bool = False,
    num_patch_tokens_multiplier: int = 1,
    custom_read_range_fn: (
        Callable[[list[dict], PreTrainedTokenizer, str, list[int]], tuple[int, int]] | None
    ) = None,
) -> DataLoader:
    rng = np.random.default_rng() if rng is None else rng

    model_type = MODEL_TYPES[hf_model_id]

    dataset = LatentQADataset(
        data=data,
        model_type=model_type,
        tokenizer=tokenizer,
        patch_from_regions=patch_from_regions,
        decoder_base_conv_format=decoder_base_conv_format,
        decoder_base_conv_format_kwargs=decoder_base_conv_format_kwargs,
        decoder_assistance_role=decoder_assistance_role,
        rng=rng,
        generate=generate,
        num_patch_tokens_multiplier=num_patch_tokens_multiplier,
        custom_read_range_fn=custom_read_range_fn,
    )

    sampler = torch.utils.data.DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=shuffle,
        seed=int(rng.integers(0, 1000000)),  # <- doesn't get used if shuffle is False
        drop_last=drop_last,
    )
    assert isinstance(tokenizer.pad_token_id, int)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=0,  # For reproducibility
        pin_memory=True,
        sampler=sampler,
        collate_fn=partial(
            dataset.collate_examples,
            pad_token_id=tokenizer.pad_token_id,
            generate=generate,
        ),
        drop_last=drop_last,
    )
    return dataloader


def fsdp_auto_wrap_policy(model, transformer_layer_name):
    def lambda_policy_fn(module):
        if (
            len(list(module.named_children())) == 0
            and getattr(module, "weight", None) is not None
            and module.weight.requires_grad
        ):
            return True
        return False

    lambda_policy = partial(lambda_auto_wrap_policy, lambda_fn=lambda_policy_fn)
    transformer_wrap_policy = partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls=set(
            [
                PrefixEncoder,
                PromptEncoder,
                PromptEmbedding,
                transformer_layer_name,
            ]
        ),
    )
    auto_wrap_policy = partial(_or_policy, policies=[lambda_policy, transformer_wrap_policy])
    return auto_wrap_policy


def wrap_fsdp(
    model: BaseModelWithGenerate | PeftModel, hf_model_id: str, fsdp_config: FSDPConfig
) -> FSDP:
    assert fsdp_config is not None
    assert fsdp_config.sharding_strategy == ShardingStrategy.FULL_SHARD

    if "llama" in hf_model_id:
        DECODER_LAYER = LlamaDecoderLayer
    else:
        raise NotImplementedError

    my_auto_wrapping_policy = fsdp_auto_wrap_policy(model, DECODER_LAYER)
    device_id = torch.cuda.current_device()

    wrapped_model = FSDP(
        model,
        auto_wrap_policy=my_auto_wrapping_policy,
        cpu_offload=(CPUOffload(offload_params=True) if fsdp_config.fsdp_cpu_offload else None),
        mixed_precision=None,
        sharding_strategy=fsdp_config.sharding_strategy,
        device_id=device_id,
        limit_all_gathers=True,
    )
    if fsdp_config.fsdp_activation_checkpointing:

        def check_fn(submodule):
            return isinstance(submodule, DECODER_LAYER)

        non_reentrant_wrapper = partial(
            checkpoint_wrapper,
            checkpoint_impl=CheckpointImpl.NO_REENTRANT,
        )
        apply_activation_checkpointing(
            wrapped_model, checkpoint_wrapper_fn=non_reentrant_wrapper, check_fn=check_fn
        )
    return wrapped_model


def get_subject_model(
    hf_model_id: str,
    use_fsdp: bool,
    device: str,
    inference_only: bool = False,
    fsdp_config: FSDPConfig | None = FSDPConfig(),
) -> BaseModelWithGenerate | FSDP:
    model = cast(
        BaseModelWithGenerate,
        AutoModelForCausalLM.from_pretrained(
            hf_model_id,
            dtype=torch.bfloat16,
            use_cache=None,
            device_map="auto" if device == "auto" else None,
        ),
    )

    if use_fsdp and not inference_only:
        assert fsdp_config is not None
        return wrap_fsdp(model, hf_model_id, fsdp_config)
    if device is not None and device != "auto":
        model = model.to(device)  # type: ignore

    if inference_only:
        model.eval()
        model.requires_grad_(False)

    return model


def get_decoder_model(
    hf_model_id: str,
    tokenizer: PreTrainedTokenizer,
    use_peft: bool,
    use_fsdp: bool,
    device: str,
    decoder_checkpoint_path: str | None = None,
    inference_only: bool = False,
    fsdp_config: FSDPConfig | None = FSDPConfig(),
    lora_config: OurLoRAConfig | None = OurLoRAConfig(),
) -> BaseModelWithGenerate | PeftModel | FSDP:
    model_kwargs = {
        "dtype": torch.bfloat16,
        "use_cache": None,
        "device_map": "auto" if device == "auto" else None,
    }

    if use_peft:
        assert lora_config is not None
        peft_config = LoraConfig(
            r=lora_config.lora_r,
            lora_alpha=lora_config.lora_alpha,
            target_modules=lora_config.lora_target_modules,
            bias=lora_config.lora_bias,
            lora_dropout=lora_config.lora_dropout,
            task_type=lora_config.lora_task_type,
        )

        model = cast(
            BaseModelWithGenerate,
            AutoModelForCausalLM.from_pretrained(hf_model_id, **model_kwargs),
        )
        model.resize_token_embeddings(len(tokenizer))

        if decoder_checkpoint_path is not None:
            model = PeftModel.from_pretrained(
                model, decoder_checkpoint_path, is_trainable=not inference_only
            )
        else:
            model = get_peft_model(model, peft_config)

        assert isinstance(model, PeftModel)
    else:
        model_path = hf_model_id if decoder_checkpoint_path is None else decoder_checkpoint_path
        model = cast(
            BaseModelWithGenerate, AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
        )
        model.resize_token_embeddings(len(tokenizer))

    if use_fsdp and not inference_only:
        assert fsdp_config is not None
        return wrap_fsdp(model, hf_model_id, fsdp_config)
    if device is not None and device != "auto":
        model = model.to(device)  # type: ignore

    if inference_only:
        model.eval()
        model.requires_grad_(False)

    return model


def save_decoder_and_optimizer_checkpoint(
    decoder: BaseModelWithGenerate | PeftModel | DDP | FSDP,
    base_checkpoint_path: str,
    dir_name: str,
    use_fsdp: bool,
    optimizer: torch.optim.Optimizer,
    wandb_id: str | None,
    master_process: bool,
    ddp: bool,
):
    t0 = time.time()
    dir_path = Path(base_checkpoint_path) / dir_name
    if master_process:
        dir_path.mkdir(parents=True, exist_ok=True)

    if use_fsdp:
        assert isinstance(decoder, FSDP)
        with FSDP.state_dict_type(
            decoder,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
            FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True),
        ):
            model_state_dict = decoder.state_dict()
            optim_state_dict = FSDP.optim_state_dict(
                decoder, optimizer, optim_state_dict=optimizer.state_dict()
            )
            if master_process:
                decoder.module.save_pretrained(str(dir_path), state_dict=model_state_dict)
                torch.save(optim_state_dict, dir_path / "optim.pt")
    else:
        if ddp:
            assert isinstance(decoder, DDP)
            base_module = decoder.module
        else:
            base_module = decoder
        optim_state_dict = optimizer.state_dict()
        if master_process:
            base_module.save_pretrained(str(dir_path))
            torch.save(optim_state_dict, dir_path / "optim.pt")

    if wandb_id is not None:
        with open(dir_path / "misc.json", "w") as f:
            json.dump({"wandb_id": wandb_id}, f, indent=4)

    if master_process:
        print(f"Time taken for checkpoint saving: {time.time() - t0} seconds.")


def process_activations_for_batch(
    batch: LQABatch,
    subject: ModelWrapper,
    read_modules: Modules,
    device: str,
) -> tuple[ActivationsList, list[tuple[int, int]]]:
    """
    Process activations for a batch, handling both pre-computed and subject activations.

    Context: pre-computed activations can be part of the dataset (e.g. when we do data-augmentation
    in the activation space), in which case, subject_input_ids will be empty (we already have the
    activations, we don't need to do a forward pass).
    During multi-GPU training, if a GPU happens to get all empty subject inputs, it will hang
    unless we use dummy inputs.

    This function unifies the logic for:
    1. Handling dummy vs real subject inputs
    2. Calling subject.get_activations
    3. Processing both pre-computed and subject activations
    4. Building the final ActivationsList and token_range_from_list

    Args:
        batch: LQABatch containing input data and potentially pre-computed activations
        subject: ModelWrapper for the subject model
        read_modules: Modules to read from the subject model
        device: Device to run computations on

    Returns:
        Tuple of (ActivationsList, token_range_from_list)
    """
    assert batch.subject_input_ids is not None
    assert batch.subject_attention_mask is not None

    BS = batch.subject_input_ids.size(0)

    # Always call get_activations to ensure consistent execution across ranks
    if batch.subject_input_ids.size(1) == 0:
        # Create dummy inputs to ensure consistent execution across ranks
        subject_input_ids = torch.zeros(
            (batch.subject_input_ids.size(0), 1), dtype=torch.long, device=device
        )
        subject_attention_mask = torch.zeros(
            (batch.subject_input_ids.size(0), 1), dtype=torch.long, device=device
        )
        use_subject_activations = False  # Will ignore subject results
    else:
        subject_input_ids = batch.subject_input_ids
        subject_attention_mask = batch.subject_attention_mask
        use_subject_activations = True

    # Always call subject model (results may be ignored if use_subject_activations=False)
    subject_activations = subject.get_activations(
        input_ids=subject_input_ids,
        attention_mask=subject_attention_mask,
        read_modules=read_modules,
        device=device,
    )

    # Unified processing: handle both pre-computed and subject activations
    token_range_from_list: list[tuple[int, int] | None] = [None] * BS
    activations_list = ActivationsList()

    # Process each key across all samples
    for key in read_modules.keys():
        acts_list = []
        for i in range(BS):
            # Decide which activation to use for this sample and key
            acts_i = None
            token_range_from = None

            # Check for pre-computed activation first
            precomputed_acts_i = batch.activations.get_i(key, i)
            if precomputed_acts_i is not None:
                acts_i = precomputed_acts_i
                token_range_from = (0, acts_i.size(0))
            else:
                assert use_subject_activations
                subject_acts_for_key = subject_activations.get(key)
                assert subject_acts_for_key is not None, f"Missing activation for key {key}"
                acts_i = subject_acts_for_key[i]  # Extract sample i from batch
                token_range_from = batch.token_range_from[i]

            assert acts_i.dim() == 2 and isinstance(acts_i, torch.Tensor)
            acts_list.append(acts_i)

            # Update token_range_from_list for this sample
            if token_range_from_list[i] is None:
                token_range_from_list[i] = token_range_from
            else:
                assert token_range_from_list[i] == token_range_from

        # Add the complete batch of activations for this key
        activations_list.add(key, acts_list)

    # Convert token_range_from_list to proper type
    final_token_range_from_list: list[tuple[int, int]] = []
    for i in range(BS):
        token_range = token_range_from_list[i]
        assert token_range is not None, f"No token range found for sample {i}"
        final_token_range_from_list.append(token_range)

    return activations_list, final_token_range_from_list


def compute_loss(
    batch: LQABatch,
    subject: ModelWrapper,
    decoder: ModelWrapper,
    read_modules: Modules,
    write_modules: Modules,
    layer_mapping: dict[int, list[int]],
    device: str,
):
    assert batch.decoder_labels is not None
    assert batch.decoder_loss_mask is not None

    # Process activations using the shared function
    activations_list, final_token_range_from_list = process_activations_for_batch(
        batch=batch,
        subject=subject,
        read_modules=read_modules,
        device=device,
    )

    out = decoder.forward_with_intervention(
        activations=activations_list,
        token_range_from=final_token_range_from_list,
        token_range_to=batch.token_range_to,
        input_ids=batch.decoder_input_ids,
        attention_mask=batch.decoder_attention_mask,
        write_modules=write_modules,
        layer_mapping=layer_mapping,
        device=device,
    )
    logits = out.logits
    assert logits is not None
    return F.cross_entropy(
        input=logits.float().transpose(1, 2),
        target=cast(torch.Tensor, batch.decoder_labels).to(device),
        reduction="none",
    ) * batch.decoder_loss_mask.to(device)
