import json
import os
import pickle
import re
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from latentqa.eval.circuit_based_steering_eval import parse_circuit_based_steering_eval_results
from latentqa.eval.steering_eval import parse_steering_eval_results
from latentqa.train_util import LQATrainConfig

EXPS_DIR = Path("experiments")  # TODO: Update this to the actual experiments directory.
KEYS = [
    "full_exp_name",
    "exp_name",
    "step",
    "attr_class",
    "attr",
    "system_prompt",
    "user_prompt",
    "conversation",
    "tag",
]


def get_config_values_from_exp(
    base_exp_name: str, exp_name: str, flags: list[str]
) -> dict[str, Any]:
    exp_path = EXPS_DIR / base_exp_name / exp_name
    with open(exp_path / "config.json", "r") as f:
        config = LQATrainConfig.model_validate_json(json.load(f))

    flags_dict = {}
    for flag in flags:
        flags_dict[flag] = getattr(config, flag)
    return flags_dict


def parse_exps_to_load(exps_to_load: dict[str, list[str] | None]) -> dict[str, list[str]]:
    exps_dict = {}
    for base_exp_name, exp_names in exps_to_load.items():
        base_exp_path = EXPS_DIR / base_exp_name
        valid_exp_names = [
            exp_dir.name
            for exp_dir in base_exp_path.iterdir()
            if exp_dir.is_dir() and not exp_dir.name.endswith("_eval_results")
        ]

        if exp_names is None:
            exp_names = valid_exp_names
        else:
            exp_names = [exp_name for exp_name in exp_names if exp_name in valid_exp_names]
        exps_dict[base_exp_name] = set(exp_names)
    return exps_dict


def parse_seed_from_exp_name(exp_name: str) -> tuple[str, int]:
    """Returns (exp_name_without_seed, seed). Defaults to seed=64."""
    match = re.search(r"_s(\d+)(?=(_|$))", exp_name)
    seed = int(match.group(1)) if match else 64
    name = re.sub(r"_s\d+(?=(_|$))", "", exp_name)
    return name, seed


def load_steering_eval_result(
    results_path: Path | str,
    steering_type: Literal["gradient", "circuit"],
):
    results_path = Path(results_path)

    losses = None
    reading_eval_results = None
    fixed_control_eval_results = None
    gen_control_eval_results = None
    if (results_path / "losses.pkl").exists():
        with open(results_path / "losses.pkl", "rb") as f:
            losses = pickle.load(f)
    if (results_path / "reading.pkl").exists():
        with open(results_path / "reading.pkl", "rb") as f:
            reading_eval_results = pickle.load(f)
    if (results_path / "fixed_control.pkl").exists():
        with open(results_path / "fixed_control.pkl", "rb") as f:
            fixed_control_eval_results = pickle.load(f)
    if (results_path / "gen_control.pkl").exists():
        with open(results_path / "gen_control.pkl", "rb") as f:
            gen_control_eval_results = pickle.load(f)

    if not losses and steering_type == "gradient":
        print(f"Missing losses for {results_path}")
        return None

    if not reading_eval_results:
        print(f"Missing reading eval results for {results_path}")
        return None

    if not fixed_control_eval_results and not gen_control_eval_results:
        print(f"Missing control eval results for {results_path}")
        return None

    if steering_type == "gradient":
        assert losses is not None
        df = parse_steering_eval_results(
            losses,
            reading_eval_results,
            fixed_control_eval_results,
            gen_control_eval_results,
        )
    elif steering_type == "circuit":
        df = parse_circuit_based_steering_eval_results(
            reading_eval_results,
            fixed_control_eval_results,
            gen_control_eval_results,
        )
    else:
        raise ValueError(f"Invalid steering type: {steering_type}")

    df["ckpt_name"] = next(part for part in results_path.parts if part.startswith("step"))
    return df


def load_steering_eval_results(
    eval_base_path: str | Path | os.PathLike, exp_names: list[str] | None = None
):
    eval_base_path = Path(eval_base_path)

    dfs = []
    for folder in Path(eval_base_path).glob("*"):
        if not folder.is_dir():
            continue

        folder_name = folder.name
        components = folder_name.split(":")
        assert len(components) == 2
        base_exp_name, exp_name = components
        transformed_exp_name, seed = parse_seed_from_exp_name(exp_name)

        if exp_names is not None and exp_name not in exp_names:
            continue

        for ckpt_path in folder.glob("step_*"):
            ckpt_name = ckpt_path.name
            step = int(ckpt_name.removeprefix("step_"))

            df = load_steering_eval_result(ckpt_path, "gradient")
            if df is None:
                continue

            df["base_exp_name"] = base_exp_name
            df["exp_name"] = transformed_exp_name
            df["seed"] = seed
            df["step"] = step
            dfs.append(df)

    return pd.concat(dfs)


def load_circuit_based_steering_eval_results(eval_base_path: str | Path | os.PathLike):
    eval_base_path = Path(eval_base_path)

    dfs = []
    for folder in Path(eval_base_path).glob("*"):
        if not folder.is_dir():
            continue

        folder_name = folder.name
        components = folder_name.split(":")
        assert len(components) == 2
        base_exp_name, exp_name = components
        transformed_exp_name, seed = parse_seed_from_exp_name(exp_name)

        for ckpt_path in folder.glob("step_*"):
            ckpt_name = ckpt_path.name
            step = int(ckpt_name.removeprefix("step_"))

            for target_attr_path in ckpt_path.iterdir():
                try:
                    df = load_steering_eval_result(target_attr_path, "circuit")
                except Exception as e:
                    print(f"Error loading results from {target_attr_path}: {e}")
                    continue
                if df is None:
                    continue

                df["base_exp_name"] = base_exp_name
                df["exp_name"] = transformed_exp_name
                df["seed"] = seed
                df["step"] = step
                dfs.append(df)

    return pd.concat(dfs)
