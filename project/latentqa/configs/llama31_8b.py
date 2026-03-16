import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
from latentqa.blogpost import STEERING_DATASET_ATTR_PAIRS
from latentqa.data_util import PatchFromRegion
from latentqa.eval.circuit_based_steering_eval import CircuitBasedSteeringEvalConfig
from latentqa.eval.reading_eval import ReadingEvalConfig
from latentqa.eval.steering_eval import SteeringEvalConfig
from latentqa.eval.steering_util import GenerationBasedControlEvalConfig
from latentqa.model_wrapper import LayerModuleKey
from latentqa.train import LQATrainConfig
from transformers import AutoConfig
from user_modeling.user_attribute import (
    AgeGroup,
    Country,
    EmploymentStatus,
    Gender,
    MaritalStatus,
    Occupation,
    Religion,
    UNGeoRegion,
)

BASE_DIR = Path("")  # <- Update this

BASE_EXP_NAME = Path(__file__).stem
BASE_EXP_PATH = BASE_DIR / "latentqa" / BASE_EXP_NAME

BASE_DATA_DIR = BASE_DIR / "latentqa" / "llama31_8b_data"

EVAL_NAMES = ["non_heldout", "heldout"]
EVAL_NAME_MAPPING = {
    "non_heldout": "train_mc",
    "heldout": "test_mc",
}


def train_lqa_configs() -> list[dict[str, Any]]:
    HF_MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
    BATCH_SIZE = 64
    NUM_EVAL_DATAPOINTS = 64 * 100

    model_config = AutoConfig.from_pretrained(HF_MODEL_ID)
    L = model_config.num_hidden_layers

    ################################################
    # Experiments
    ################################################

    base_config_dict = {
        "hf_model_id": HF_MODEL_ID,
        "train_patch_regions": [PatchFromRegion.USER],
        "eval_patch_regions": [PatchFromRegion.USER],
        "use_peft": True,
        "use_fsdp": False,
        "batch_size": BATCH_SIZE,
        "micro_batch_size": 8,
        "gradient_accumulation_steps": None,
        "num_steps": 5000,
        "lr": 1e-4,
        "use_wandb": True,  #
        "wandb_group_name": BASE_EXP_NAME,
        "eval_num_datapoints": NUM_EVAL_DATAPOINTS,
        "eval_batch_size": 64,
        "eval_num_steps": None,
        "eval_interval": 100,  ###
        "save_interval": 500,
        "save_final_checkpoint": True,
        "log_interval": 100,
        # Change to "train_full.pkl" to train on all SynthSys attributes.
        "train_data_path": str(BASE_DATA_DIR / "train.pkl"),
        "eval_data": {
            eval_name: (
                "single_token_mc",
                str(BASE_DATA_DIR / "eval_synthsys" / f"{eval_name}.pkl"),
            )
            for eval_name in EVAL_NAMES
        },
        "eval_name_mapping": EVAL_NAME_MAPPING,
    }

    architecture_configs = []

    # All -> All (residuals).
    architecture_configs.append(
        {
            "exp_name": "all_to_all",
            "read_layer_module_keys": [
                LayerModuleKey(layer=layer, module="residual") for layer in range(L)
            ],
            "write_layer_module_keys": [
                LayerModuleKey(layer=layer, module="residual") for layer in range(L)
            ],
            "layer_mapping": {layer: layer for layer in range(L)},
        }
    )

    # Uncomment to run 15 -> X architecture.
    # # 15 -> X.
    # for write_layer in [0]:
    #     architecture_configs.append(
    #         {
    #             "exp_name": f"15_to_{write_layer}",
    #             "read_layer_module_keys": [LayerModuleKey(layer=15, module="residual")],
    #             "write_layer_module_keys": [LayerModuleKey(layer=write_layer, module="residual")],
    #             "layer_mapping": {write_layer: 15},
    #         }
    #     )

    configs: list[dict[str, Any]] = []

    for seed in [32]:
        for config_dict in architecture_configs:
            config = deepcopy(base_config_dict)
            config.update(config_dict)

            exp_name = str(config.pop("exp_name")) + f"_s{seed}"
            config["seed"] = seed
            config["save_path"] = str(BASE_EXP_PATH / exp_name)
            config["wandb_run_name"] = f"{exp_name}"
            configs.append(json.loads(LQATrainConfig.model_validate(config).model_dump_json()))

    return configs


def run_reading_eval_configs() -> list[dict[str, Any]]:
    normal_base_config_dict = {
        "eval_type": "single_token_mc",
        "ckpt_names": None,
        "num_batches": 100,
        "batch_size": 64,
        "patch_from_region": PatchFromRegion.USER,
    }
    prism_base_config_dict = {
        "eval_type": "single_token_mc",
        "batch_size": 4,
        "ckpt_names": None,
        "patch_from_region": PatchFromRegion.USER,
    }

    configs = []
    for seed in [32]:
        for exp_name in ["all_to_all"]:
            exp_name = f"{exp_name}_s{seed}"
            exp_path = BASE_EXP_PATH / exp_name

            # Synthsys: non-heldout and heldout attributes.
            for dset_name in EVAL_NAMES:
                config = deepcopy(normal_base_config_dict)
                config.update(
                    {
                        "exp_path": str(exp_path),
                        "dset_dir": str(BASE_DATA_DIR / "eval_synthsys"),
                        "dset_name": dset_name,
                    }
                )
                configs.append(
                    json.loads(ReadingEvalConfig.model_validate(config).model_dump_json())
                )

            # SelfDescribe: gender, country, occupation, religion.
            for attr_class in [
                Gender,
                Country,
                Occupation,
                Religion,
            ]:
                config = deepcopy(normal_base_config_dict)
                config.update(
                    {
                        "exp_path": str(exp_path),
                        "dset_dir": str(BASE_DATA_DIR / "eval_selfdescribe"),
                        "dset_name": attr_class.__name__,
                    }
                )
                configs.append(
                    json.loads(ReadingEvalConfig.model_validate(config).model_dump_json())
                )

            # PRISM: gender, age group, UN georegion, employment status, marital status.
            for attr_class in [
                Gender,
                AgeGroup,
                UNGeoRegion,
                EmploymentStatus,
                MaritalStatus,
            ]:
                config = deepcopy(prism_base_config_dict)
                config.update(
                    {
                        "num_batches": 500,  # To evaluate on the entire PRISM dataset, use 20013.
                        "exp_path": str(exp_path),
                        "dset_dir": str(BASE_DATA_DIR / "eval_prism"),
                        "dset_name": attr_class.__name__,
                    }
                )
                configs.append(
                    json.loads(ReadingEvalConfig.model_validate(config).model_dump_json())
                )
    return configs


def run_steering_eval_configs() -> list[dict[str, Any]]:
    CKPT_NAMES = ["step_4999"]
    GENERATE_TO_EVAL_CONTROL = True
    USE_SEED_RESPONSE = True

    assert (
        USE_SEED_RESPONSE or GENERATE_TO_EVAL_CONTROL
    ), "If we don't have a seed response, we need to generate to eval control."

    gen_config = GenerationBasedControlEvalConfig(
        max_new_tokens=10,
        temperature=1.0,
        num_completions=5,
        llm_judge_provider="openai",
        llm_judge_model_name="gpt-4.1-mini-2025-04-14",
        llm_judge_max_concurrency=20,
        llm_judge_timeout=100.0,
    )

    base_config_dict = {
        "mc_stems_path": str(BASE_DATA_DIR / "mc_stems_by_attr_class.pkl"),
        "patch_from_region": PatchFromRegion.USER,
        "max_subject_inputs_per_attr_pair": 10,
        "steering_qas_path": str(BASE_DATA_DIR / "steering_qas"),
        "gradient_batch_size": 64,
        "forward_batch_size": 64,
        "num_steps": 20,
        "step_sizes": [0.003],
        "generate_to_eval_control": GENERATE_TO_EVAL_CONTROL,
        "si_batch_size": 16,
        "gen_based_control_eval_config": gen_config,
    }

    configs = []
    for seed in [32]:
        for exp_to_load_name in ["all_to_all"]:
            exp_name = f"{exp_to_load_name}_s{seed}"
            exp_path = BASE_EXP_PATH / exp_name
            for ckpt_name in CKPT_NAMES:
                update_keys: dict[str, Any] = {"exp_path": str(exp_path), "ckpt_name": ckpt_name}
                attr_pairs = STEERING_DATASET_ATTR_PAIRS
                update_keys.update(
                    {
                        "subject_inputs_data_path": str(
                            BASE_DATA_DIR / "eval_selfdescribe" / "all_data.pkl"
                        ),
                        "seed_response_type": "wiki" if USE_SEED_RESPONSE else "none",
                        "attr_pairs": [(str(base), str(target)) for base, target in attr_pairs],
                    }
                )

                config = deepcopy(base_config_dict)
                config.update(update_keys)
                config = SteeringEvalConfig(**config)
                config.save_path = str(
                    BASE_DIR
                    / "latentqa"
                    / "llama31_8b_steering_results"
                    / f"{BASE_EXP_NAME}:{exp_name}"
                    / config.ckpt_name
                )
                configs.append(json.loads(config.model_dump_json()))
    return configs


def run_circuit_steering_eval_configs() -> list[dict[str, Any]]:
    CKPT_NAMES = ["step_4999"]
    STEERING_QUANTILE = "1e-7"
    NUM_SOURCE_ATTRS_PER_TARGET = 10
    GENERATE_TO_EVAL_CONTROL = False
    USE_SEED_RESPONSE = True

    assert (
        USE_SEED_RESPONSE or GENERATE_TO_EVAL_CONTROL
    ), "If we don't have a seed response, we need to generate to eval control."

    data_path = BASE_DATA_DIR / "eval_selfdescribe" / "all_data.pkl"

    base_config_dict = {
        "mc_stems_path": str(BASE_DATA_DIR / "mc_stems_by_attr_class.pkl"),
        "subject_inputs_data_path": str(data_path),
        "max_subject_inputs_per_attr_pair": 10,
        "forward_batch_size": 64,
        "steering_strengths": (np.arange(10) / 10).tolist(),
        "generate_to_eval_control": GENERATE_TO_EVAL_CONTROL,
        "steer_region": "prefix",
        "read_region": "wiki",
        "steering_quantile": STEERING_QUANTILE,
        "si_batch_size": 16,
    }
    np.random.seed(45)

    configs = []
    for target_attr in [
        Country.FRANCE,
        Country.INDIA,
        Country.SOUTH_KOREA,
    ]:
        neurons_path = BASE_DIR / "latentqa" / "circuits" / f"{target_attr.title()}_neurons.pkl"
        # Sample source attributes.
        attr_values = [attr for attr in target_attr.__class__]
        idxs = np.random.choice(
            range(len(attr_values)), size=NUM_SOURCE_ATTRS_PER_TARGET, replace=False
        )
        source_attrs = [attr_values[idx] for idx in idxs]

        for seed in [32]:
            for exp_to_load_name in ["all_to_all"]:
                exp_name = f"{exp_to_load_name}_s{seed}"
                exp_path = str(BASE_EXP_PATH / exp_name)
                for ckpt_name in CKPT_NAMES:
                    update_keys = {
                        "exp_path": exp_path,
                        "neurons_path": str(neurons_path),
                        "target_attr": str(target_attr),
                        "true_attrs": [str(attr) for attr in source_attrs],
                        "ckpt_name": ckpt_name,
                        "seed_response_type": "wiki" if USE_SEED_RESPONSE else "none",
                    }
                    config = deepcopy(base_config_dict)
                    config.update(update_keys)
                    config = CircuitBasedSteeringEvalConfig(**config)
                    config.save_path = str(
                        BASE_DIR
                        / "latentqa"
                        / "llama31_8b_circuit_steering_results"
                        / f"{BASE_EXP_NAME}:{exp_name}"
                        / config.ckpt_name
                        / str(target_attr)
                    )
                    configs.append(json.loads(config.model_dump_json()))
    return configs
