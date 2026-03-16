import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Literal, Type

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from latentqa.eval.reading_eval import load_checkpoint_eval_results
from latentqa.lqa_data import LQAData, load_lqa_data
from tqdm import tqdm
from user_modeling.user_attribute import (
    ALL_USER_ATTRIBUTES,
    AgeGroup,
    Country,
    EmploymentStatus,
    Gender,
    MaritalStatus,
    Occupation,
    Religion,
    ReligiousAffiliation,
    UNGeoRegion,
    UserAttribute,
)

STEERING_DATASET_ATTR_PAIRS = [
    (Gender.MALE, Gender.FEMALE),
    (Gender.FEMALE, Gender.MALE),
    (Country.FRANCE, Country.JAPAN),
    (Country.SWITZERLAND, Country.LEBANON),
    (Country.CHILE, Country.RUSSIA),
    (Country.GUATEMALA, Country.NORWAY),
    (Country.TAIWAN, Country.PANAMA),
    (Country.HAITI, Country.GERMANY),
    (Country.BELGIUM, Country.KENYA),
    (Country.ZIMBABWE, Country.PERU),
    (Occupation.LAWYER, Occupation.RETAIL_MANAGER),
    (Occupation.POLICE_OFFICER, Occupation.SALES_REPRESENTATIVE),
    (Occupation.CHIROPRACTOR, Occupation.MARINE_BIOLOGIST),
    (Occupation.HUMAN_RESOURCES_MANAGER, Occupation.AIR_TRAFFIC_CONTROLLER),
    (Occupation.LAWYER, Occupation.MASSAGE_THERAPIST),
    (Occupation.CHIROPRACTOR, Occupation.INSURANCE_AGENT),
    (Occupation.ANTHROPOLOGIST, Occupation.POLICE_OFFICER),
    (Occupation.TOUR_GUIDE, Occupation.FARMER),
    (Occupation.LOCKSMITH, Occupation.DOCTOR),
    (Occupation.CHIROPRACTOR, Occupation.SOCIAL_WORKER),
    (Occupation.INVESTMENT_BANKER, Occupation.TRUCK_DRIVER),
    (Occupation.FIREFIGHTER, Occupation.BARBER),
    (Occupation.PUBLIC_RELATIONS_SPECIALIST, Occupation.ANIMATOR),
    (Occupation.ASTRONOMER, Occupation.LANDSCAPE_ARCHITECT),
    (Occupation.TOUR_GUIDE, Occupation.CONSTRUCTION_WORKER),
    (Occupation.LAWYER, Occupation.MIDWIFE),
    (Occupation.CHEF, Occupation.HISTORIAN),
    (Religion.DEISM, Religion.SEVENTH_DAY_ADVENTISM),
    (Religion.HERMETICISM, Religion.SHINTO),
    (Religion.JUDAISM, Religion.ATHEISM),
    (Religion.KARAITE_JUDAISM, Religion.SUNNI_ISLAM),
    (Religion.KONKOKYO, Religion.BAHAI_FAITH),
    (Religion.PAGANISM, Religion.ISLAM),
    (Religion.ROMAN_CATHOLICISM, Religion.ISLAM),
    (Religion.ROMAN_POLYTHEISM, Religion.KEMETISM),
    (Religion.SATANISM, Religion.VAJRAYANA_BUDDHISM),
    (Religion.SCIENTOLOGY, Religion.DEISM),
    (Religion.THERAVADA_BUDDHISM, Religion.SHINTO),
]

TEST_ATTR_CLASSES = [
    Gender,
    AgeGroup,
    ReligiousAffiliation,
    UNGeoRegion,
    EmploymentStatus,
    MaritalStatus,
]

SYNTHSYS_ATTR_CLASSES = [
    attr_class
    for attr_class in ALL_USER_ATTRIBUTES.values()
    if attr_class not in [Occupation, Religion, Country]
]
TRAIN_SYNTHSYS_ATTR_CLASSES = [
    attr_class for attr_class in SYNTHSYS_ATTR_CLASSES if attr_class not in TEST_ATTR_CLASSES
]
TEST_SYNTHSYS_ATTR_CLASSES = TEST_ATTR_CLASSES

SELF_DESCRIBE_ATTR_CLASSES = [Gender, Country, Occupation, Religion]
PRISM_ATTR_CLASSES = [Gender, AgeGroup, UNGeoRegion, EmploymentStatus, MaritalStatus]

DATASET_NAME_MAPPING = {
    "synthsys": "SynthSys",
    "selfdescribe": "SelfDescribe",
    "prism": "PRISM",
}

EVAL_NAMES_BY_DATASET: dict[str, list[str]] = {
    "synthsys": ["heldout", "non_heldout"],
    "selfdescribe": [c.__name__ for c in SELF_DESCRIBE_ATTR_CLASSES],
    "prism": [c.__name__ for c in PRISM_ATTR_CLASSES],
}


def get_attr_classes_for_dataset(dataset_name: str) -> list[Type[UserAttribute]]:
    if dataset_name == "synthsys":
        return SYNTHSYS_ATTR_CLASSES
    elif dataset_name == "selfdescribe":
        return SELF_DESCRIBE_ATTR_CLASSES
    elif dataset_name == "prism":
        return PRISM_ATTR_CLASSES
    else:
        raise ValueError(f"Invalid dataset name: {dataset_name}")


def process_folder(folder: Path, source_data_dict: dict[str, LQAData], overwrite: bool = False):
    if not overwrite and (folder / "all_steps.pkl").exists():
        return

    folder_name_parts = folder.name.split(":")
    base_exp_name = folder_name_parts[0]
    exp_name = folder_name_parts[1]

    # Extract seed from exp_name if it exists.
    seed = int(exp_name.split("_s")[-1]) if "_s" in exp_name else 0

    dfs_for_folder = []
    for results_path in folder.glob("step_*.jsonl"):
        ckpt_name = results_path.stem

        step = int(ckpt_name.removeprefix("step_"))

        ckpt_results = list(load_checkpoint_eval_results(results_path).values())
        for train_results_path in folder.glob(f"rank*_{ckpt_name}.jsonl"):
            ckpt_results.extend(list(load_checkpoint_eval_results(train_results_path).values()))

        parsed_ckpt_results = []
        ids = set()
        for result in ckpt_results:
            if result["data_id"] in ids:
                continue
            ids.add(result["data_id"])

            # Get corresponding datapoint from source_data.
            d = source_data_dict[result["data_id"]]

            parsed_ckpt_results.append(
                {
                    "attr_class": d.attr_class,
                    "attr": d.attr,
                    "tag": d.tag,
                    "probs": np.array(result["probs"]),
                    "base_exp_name": base_exp_name,
                    "exp_name": exp_name,
                    "step": step,
                }
            )
        ckpt_df = pd.DataFrame(parsed_ckpt_results)

        # Aggregate over questions.
        agg_ckpt_df = (
            ckpt_df.groupby(
                ["exp_name", "attr_class", "attr", "tag"],
                dropna=False,
                sort=False,
            )["probs"]
            .apply(lambda x: np.mean(np.stack(x.values), axis=0))
            .reset_index()
        )
        agg_ckpt_df["step"] = step
        agg_ckpt_df["seed"] = seed
        dfs_for_folder.append(agg_ckpt_df)

    if len(dfs_for_folder) > 0:
        df_for_folder = pd.concat(dfs_for_folder)
        df_for_folder.to_pickle(folder / "all_steps.pkl")


def _process_eval_results(data_dir: Path, names: list[str], overwrite: bool = False):
    for name in names:
        print(f"Processing {name}...")
        source_data = load_lqa_data(data_dir / f"{name}.pkl", LQAData)
        source_data_dict = {d.id: d for d in source_data}

        folders = list((data_dir / f"{name}_eval_results").glob("*"))
        for folder in tqdm(folders, desc="Going through folders..."):
            process_folder(folder, source_data_dict, overwrite)


def process_dataset(dataset_name: str, data_dir: str | Path, overwrite: bool = False):
    if dataset_name not in EVAL_NAMES_BY_DATASET:
        raise ValueError(f"Invalid dataset name: {dataset_name}")
    _process_eval_results(
        Path(data_dir) / f"eval_{dataset_name}", EVAL_NAMES_BY_DATASET[dataset_name], overwrite
    )


def _load_eval_results(
    data_dir: Path, names: list[str], exps_to_load: dict[str, list[str]]
) -> pd.DataFrame:
    dfs = []
    for name in names:
        eval_dir = data_dir / f"{name}_eval_results"
        for base_exp_name, exp_names in exps_to_load.items():
            for exp_name in exp_names:
                results_path = eval_dir / f"{base_exp_name}:{exp_name}" / "all_steps.pkl"
                try:
                    dfs.append(pd.read_pickle(results_path))
                except Exception as e:
                    print(f"Error loading results from {results_path}: {e}")
    return pd.concat(dfs)


def load_eval_results_for_dataset(
    dataset_name: str, data_dir: Path, exps_to_load: dict[str, list[str]]
) -> pd.DataFrame:
    if dataset_name not in EVAL_NAMES_BY_DATASET:
        raise ValueError(f"Invalid dataset name: {dataset_name}")
    return _load_eval_results(
        data_dir / f"eval_{dataset_name}", EVAL_NAMES_BY_DATASET[dataset_name], exps_to_load
    )


def get_everything_needed_for_reading_plots(
    dataset_name: str,
    data_dir: Path,
    exps_to_load: dict[str, list[str]],
    attr_classes: list[Type[UserAttribute]] | None = None,
):
    # First, load our results.
    results_df = load_eval_results_for_dataset(dataset_name, data_dir, exps_to_load)

    if attr_classes is None:
        attr_classes = get_attr_classes_for_dataset(dataset_name)

    # Find intersection of tags for each attribute class.
    tags_by_attr_class = {}
    for attr_class in attr_classes:
        attr_class_df = results_df[results_df.attr_class == attr_class]
        exp_names = attr_class_df.exp_name.unique()

        intersect_tags = None
        for exp_name in exp_names:
            exp_rows = attr_class_df[attr_class_df.exp_name == exp_name]
            if intersect_tags is None:
                intersect_tags = set(exp_rows.tag.unique())
            else:
                intersect_tags = intersect_tags.intersection(set(exp_rows.tag.unique()))

        tags_by_attr_class[attr_class] = intersect_tags

    return results_df, tags_by_attr_class


def get_xs_ys_error(
    results_df,
    tags_by_attr_class: dict[Type[UserAttribute], set[str]],
    labels_by_tag: dict[str, UserAttribute] | None = None,
    last_step_only: bool = False,
    progress_bar: bool = False,
):
    accuracies = []
    for exp_name in results_df.exp_name.unique():
        exp_df = results_df[results_df.exp_name == exp_name]
        if last_step_only:
            exp_df = exp_df[exp_df.step == exp_df.step.max()]

    for (attr_class, seed, step), group in tqdm(
        exp_df.groupby(["attr_class", "seed", "step"], sort=False),
        desc="Computing metrics",
        disable=not progress_bar,
    ):
        attr_values = [attr for attr in attr_class]

        preds = []
        labels = []
        for attr, tag, probs in zip(group.attr, group.tag, group.probs):
            pred = np.argmax(probs)

            # Filter by valid tags.
            if tag not in tags_by_attr_class[attr_class]:
                continue

            # Re-label as necessary.
            if labels_by_tag is not None:
                if tag not in labels_by_tag:
                    continue
                attr = labels_by_tag[tag]
            label = attr_values.index(attr)

            preds.append(pred)
            labels.append(label)

        accuracy = np.mean(np.array(preds) == np.array(labels))
        accuracies.append(
            {
                "exp_name": exp_name,
                "attr_class": attr_class,
                "accuracy": accuracy,
                "step": step,
                "seed": seed,
            }
        )
    accs_df = pd.DataFrame(accuracies)

    # First take mean over attribute classes.
    agg_rows = accs_df.groupby(["step", "seed"])["accuracy"].mean().reset_index()

    # Then take mean/std over seeds.
    step_grouped = agg_rows.groupby("step")["accuracy"]
    result = step_grouped.agg(["mean", "std", "count"]).reset_index()
    result["std_err"] = result["std"] / np.sqrt(result["count"])
    result = result.sort_values("step", ascending=True)

    steps = result["step"].values

    xs = steps
    ys = result["mean"].values
    std_error = result["std_err"].values
    return xs, ys, std_error


def plot_panel(
    ax,
    exp_names: list[str],
    results_df: pd.DataFrame,
    tags_by_attr_class: dict[Type[UserAttribute], set[str]],
    attr_classes: list[Type[UserAttribute]],
    kwargs_by_exp_name: dict[str, Any],
    labels_by_tag: dict[str, UserAttribute] | None = None,
    ignore_label: bool = False,
    plot_lqa_baseline_as_line: bool = True,
):

    for exp_name in exp_names:
        # Plot our results.
        exp_rows = results_df[
            (results_df.exp_name == exp_name) & (results_df.attr_class.isin(attr_classes))
        ]
        if len(exp_rows) == 0:
            continue

        # print(exp_name, len(exp_rows))

        kwargs = kwargs_by_exp_name.get(exp_name, {}).copy()
        color = kwargs.get("color", "black")
        if ignore_label:
            kwargs.pop("label", None)

        if exp_name in ["8b", "8b_all_to_all"] and plot_lqa_baseline_as_line:
            _, ys, _ = get_xs_ys_error(
                exp_rows, tags_by_attr_class, labels_by_tag, last_step_only=True
            )
            ax.axhline(np.mean(ys[0]), **kwargs)
        else:
            xs, ys, std_error = get_xs_ys_error(exp_rows, tags_by_attr_class, labels_by_tag)
            ax.plot(xs, ys, **kwargs)
            ax.fill_between(
                xs,
                ys - std_error,
                ys + std_error,
                alpha=0.15,
                color=color,
            )

    ax.grid(True)
    ax.set_xlabel("Train Steps")
    ax.set_ylabel("Accuracy")


def plot_steering_for_attr_pairs(
    all_results_df: pd.DataFrame,
    steering_type: Literal["circuit", "gradient"],
    exp_name: str,
    seed: int,
    ckpt_name: str,
    attr_pairs: list[tuple[UserAttribute, UserAttribute]],
    metric: Literal["acc", "prob"] = "prob",
    control_prefix: Literal["fixed", "gen"] = "fixed",
    mean_kwargs: dict[str, Any] = {},  # {"marker": "o", "ms": 5.0}
    sharey: bool = False,
    save_path: Path | os.PathLike | None = None,
    title: str | None = None,
    title_fontsize: int = 14,
    title_y: float = 0.98,
):
    assert steering_type in ["circuit", "gradient"]
    steering_unit = "strength" if steering_type == "circuit" else "steering_step"
    x_label = r"Steering Strength $f$" if steering_unit == "strength" else "Steering Step"

    fig, axes = plt.subplots(
        len(attr_pairs), 2, figsize=(10, 2 * len(attr_pairs)), squeeze=False, sharey=sharey
    )

    prefixes = ["reading", control_prefix]

    ckpt_df = all_results_df[
        (all_results_df.exp_name == exp_name)
        & (all_results_df.ckpt_name == ckpt_name)
        & (all_results_df.seed == seed)
    ]
    for row, (true_attr, target_attr) in enumerate(attr_pairs):
        df = ckpt_df[(ckpt_df.true_attr == true_attr) & (ckpt_df.target_attr == target_attr)]
        df = df.sort_values(by=steering_unit, ascending=True)

        # Plot individual runs
        for id, group in df.groupby("tag"):
            xs = group[steering_unit].values
            for col, prefix in enumerate(prefixes):
                orig_key = f"{prefix}_{metric}_orig"
                target_key = f"{prefix}_{metric}_target"

                axes[row, col].plot(xs, group[orig_key].values, alpha=0.3, color="tab:blue")
                axes[row, col].plot(xs, group[target_key].values, alpha=0.3, color="tab:red")

        # Plot mean
        xs = []
        orig_ys_by_prefix = defaultdict(list)
        target_ys_by_prefix = defaultdict(list)
        for x, group in df.groupby(steering_unit):
            xs.append(x)

            for prefix in prefixes:
                orig_ys_by_prefix[prefix].append(np.mean(group[f"{prefix}_{metric}_orig"].values))
                target_ys_by_prefix[prefix].append(
                    np.mean(group[f"{prefix}_{metric}_target"].values)
                )

        for col, prefix in enumerate(prefixes):
            axes[row, col].plot(
                xs,
                orig_ys_by_prefix[prefix],
                label=f"{true_attr.value_name} (Source)",
                color="tab:blue",
                **mean_kwargs,
            )
            axes[row, col].plot(
                xs,
                target_ys_by_prefix[prefix],
                label=f"{target_attr.value_name} (Target)",
                color="tab:red",
                **mean_kwargs,
            )

            axes[row, col].grid(True)

            if col == 0:
                axes[row, col].set_ylabel("Probability" if metric == "prob" else "Accuracy")
            axes[row, col].set_xlabel(x_label)

        axes[row, -1].legend(loc="center left", bbox_to_anchor=(1, 0.5), title="Attribute")

    axes[0, 0].set_title("Decoder Reading")
    axes[0, 1].set_title("Subject Control")

    if title is not None:
        fig.suptitle(title, fontsize=title_fontsize, y=title_y)
    else:
        target_attrs = set([t.value_name for s, t in attr_pairs])
        if len(target_attrs) == 1:
            target_attr = target_attrs.pop()
            source_attrs_str = "{" + ", ".join([true.value_name for true, _ in attr_pairs]) + "}"
            title = rf"Steering {source_attrs_str} $\rightarrow$ {target_attr}"
            fig.suptitle(title, fontsize=title_fontsize, y=title_y)

    fig.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=600)
