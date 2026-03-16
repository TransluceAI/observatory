import argparse
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any, Type, cast

import numpy as np
from datasets import load_dataset
from latentqa.lqa_data import QA, LQAData, get_lqa_data_from_full_lqa_data_db
from latentqa.mc_util import MC_TEMPLATE_WITH_SEED_RESPONSE, construct_mc_options_for_eval
from latentqa.qas import CompletedQuestion, QACategory
from tqdm import tqdm
from user_modeling.prism import extract_user_attributes_from_prism_dataset, get_prism_dataset
from user_modeling.synthsys.subject_inputs import MinimalSubjectInput, SubjectInput
from user_modeling.user_attribute import (
    ALL_USER_ATTRIBUTES,
    AgeGroup,
    EmploymentStatus,
    Gender,
    MaritalStatus,
    ReligiousAffiliation,
    UNGeoRegion,
    UserAttribute,
)

TEST_ATTR_CLASSES: list[Type[UserAttribute]] = [
    Gender,
    AgeGroup,
    ReligiousAffiliation,
    UNGeoRegion,
    EmploymentStatus,
    MaritalStatus,
]


def generate_mc_qas_for_attr(
    attr_label: UserAttribute,
    stems_by_attr_class: dict[Type[UserAttribute], list[str]],
    rng: np.random.Generator,
    num_permutations: int = 10,
    max_num_choices: int = 20,
    add_extra: bool = True,
) -> list[QA]:
    """Used for experiments for blog post."""
    attr_class = attr_label.__class__

    # One for the label, one for extra.
    max_num_distractors = max_num_choices - (2 if add_extra else 1)
    mc_options_strings = construct_mc_options_for_eval(
        must_include_options=[attr_label],
        possible_distractors=[attr for attr in attr_class if attr != attr_label],
        max_num_distractors=max_num_distractors,
        num_permutations=num_permutations,
        add_extra=add_extra,
        rng=rng,
    )
    test_mc_qas: list[QA] = []
    for options_string in mc_options_strings:
        stem = rng.choice(stems_by_attr_class[attr_class])

        q = MC_TEMPLATE_WITH_SEED_RESPONSE.format(stem=stem, choices=options_string.options_str)
        assert options_string.option_to_key is not None
        a = options_string.option_to_key[attr_label]
        test_mc_qas.append(
            QA(
                attr_class=attr_class,
                attr=attr_label,
                qa_category=QACategory.MC,
                content=CompletedQuestion(question=q),
                seed_response="The answer is",
                answer=a,
                multiple_choice_options=options_string,
            )
        )
    return test_mc_qas


def generate_templated_mc_qas(
    data: list[SubjectInput],
    tag_prefix: str,
    mc_stems_by_attr_class: dict[Type[UserAttribute], list[str]],
    rng: np.random.Generator,
):
    keys = set()
    eval_data = []
    for idx, d in tqdm(enumerate(data), total=len(data)):
        key = (d.attr, d.system_prompt, d.user_prompt)
        if key in keys:
            continue
        keys.add(key)

        qas = generate_mc_qas_for_attr(
            attr_label=d.attr, stems_by_attr_class=mc_stems_by_attr_class, rng=rng
        )
        for qa in qas:
            eval_data.append(
                LQAData(
                    attr_class=d.attr_class,
                    attr=d.attr,
                    system_prompt=d.system_prompt,
                    user_prompt=d.user_prompt,
                    qa=qa,
                    id=len(eval_data),
                    tag=f"{tag_prefix}_{idx}",
                )
            )
    return eval_data


def get_train_data(
    general_db_dir: str,
    mc_db_dir: str,
    test_attr_classes: list[Type[UserAttribute]],
    rng: np.random.Generator,
    mc_ratio: float = 0.5,
):
    general_data = get_lqa_data_from_full_lqa_data_db(general_db_dir)
    mc_data = get_lqa_data_from_full_lqa_data_db(mc_db_dir, rng=rng)

    all_data = rng.permutation(general_data + mc_data)

    # Filter out test attr classes.
    upd_general_data = []
    for idx in rng.permutation(len(general_data)):
        d = general_data[idx]
        if d.attr_class not in test_attr_classes:
            upd_general_data.append(d)

    upd_mc_data = []
    for idx in rng.permutation(len(mc_data)):
        d = mc_data[idx]
        if d.attr_class not in test_attr_classes:
            upd_mc_data.append(d)
    total_general = len(upd_general_data)
    total_mc = len(upd_mc_data)

    # Mix general and MC qa's.
    # If we want mc_ratio = len(mc) / (len(mc) + len(nonmc))
    # Then len(nonmc) = len(mc) * (1 - mc_ratio) / mc_ratio
    if mc_ratio == 1.0:
        target_mc = total_mc
        target_nonmc = 0
    elif mc_ratio == 0.0:
        target_mc = 0
        target_nonmc = total_general
    else:
        target_mc = min(total_mc, int(total_general * mc_ratio / (1 - mc_ratio)))
        target_nonmc = int(target_mc * (1 - mc_ratio) / mc_ratio)

    mixed_train_data = upd_mc_data[:target_mc] + upd_general_data[:target_nonmc]
    return mixed_train_data, all_data


def get_synthsys_eval_data(
    hf_model_id: str,
    attr_classes_to_consider: list[Type[UserAttribute]],
    mc_stems_by_attr_class: dict[Type[UserAttribute], list[str]],
    rng: np.random.Generator,
):
    hf_dataset_id = "Transluce/SynthSys-" + hf_model_id.split("/")[-1]
    hf_dataset = load_dataset(hf_dataset_id, split="test")

    test_data_set: set[MinimalSubjectInput] = set()
    for row in hf_dataset:
        row = cast(dict[str, Any], row)
        attr_class = ALL_USER_ATTRIBUTES[row["attr_class"]]

        # Skip if the attribute class is not in the list of attribute classes to consider.
        if attr_class not in attr_classes_to_consider:
            continue

        test_data_set.add(
            MinimalSubjectInput(
                attr_class=attr_class,
                attr=attr_class(row["attr"]),
                system_prompt=row["system_prompt"],
                user_prompt=row["user_prompt"],
                completion="",  # Not used.
            )
        )
    test_data = list(test_data_set)

    # Reorder to make sure that at any given cutoff point, we have ~equal number from each attr class.
    data_by_attr_class = defaultdict(list)
    for idx, d in enumerate(test_data):
        data_by_attr_class[d.attr_class].append(idx)

    attr_classes = sorted(list(data_by_attr_class.keys()), key=lambda x: x.__name__)

    remaining_idxs = set(range(len(test_data)))
    reordered_test_data: list[MinimalSubjectInput] = []
    pbar = tqdm(total=len(test_data))
    while remaining_idxs:
        attr_class = attr_classes[rng.choice(len(attr_classes))]

        if not data_by_attr_class[attr_class]:
            attr_classes.remove(attr_class)
            continue

        idx = rng.choice(data_by_attr_class[attr_class])
        reordered_test_data.append(test_data[idx])
        remaining_idxs.remove(idx)
        data_by_attr_class[attr_class].remove(idx)
        pbar.update(1)
    pbar.close()
    assert len(test_data) == len(reordered_test_data)

    # Construct multiple choice questions with the stems being paraphrased direct question,
    # and the options being all possible attribute values.
    eval_data: list[LQAData] = []
    for idx, d in tqdm(enumerate(reordered_test_data)):
        qas = generate_mc_qas_for_attr(
            attr_label=d.attr, stems_by_attr_class=mc_stems_by_attr_class, rng=rng
        )
        for qa in qas:
            eval_data.append(
                LQAData(
                    attr_class=d.attr_class,
                    attr=d.attr,
                    system_prompt=d.system_prompt,
                    user_prompt=d.user_prompt,
                    qa=qa,
                    id=len(eval_data),
                    tag=f"synthsys_{idx}",
                )
            )
    return eval_data


def get_selfdescribe_eval_data(
    hf_model_id: str,
    mc_stems_by_attr_class: dict[Type[UserAttribute], list[str]],
    rng: np.random.Generator,
) -> dict[Type[UserAttribute], list[LQAData]]:
    hf_dataset_id = "Transluce/SelfDescribe-" + hf_model_id.split("/")[-1]
    hf_dataset = load_dataset(hf_dataset_id, split="train")  # There is only one split

    eval_data_by_attr_class = defaultdict(list)
    for row in hf_dataset:
        row = cast(dict[str, Any], row)

        attr_class = ALL_USER_ATTRIBUTES[row["attr_class"]]
        attr = attr_class(row["attr"])

        qas = generate_mc_qas_for_attr(
            attr_label=attr, stems_by_attr_class=mc_stems_by_attr_class, rng=rng
        )
        for qa in qas:
            eval_data_by_attr_class[attr_class].append(
                LQAData(
                    attr_class=attr_class,
                    attr=attr,
                    system_prompt=row["system_prompt"],
                    user_prompt=row["user_prompt"],
                    qa=qa,
                    id=len(eval_data_by_attr_class[attr_class]),
                    tag=row["tag"],
                )
            )

    return eval_data_by_attr_class


def get_prism_eval_data(
    hf_model_id: str,
    mc_stems_by_attr_class: dict[Type[UserAttribute], list[str]],
    rng: np.random.Generator,
) -> dict[Type[UserAttribute], list[LQAData]]:

    def extract_user_prompt_and_context_from_conv(
        conversation: list[dict[str, str]],
    ) -> tuple[str, list[dict[str, str]]]:
        # Don't include the last assistant message.
        if conversation[-1]["role"] == "assistant":
            conversation = conversation[:-1]
        assert conversation[-1]["role"] == "user"
        context = conversation[:-1]
        user_prompt = conversation[-1]["content"]
        return user_prompt, context

    eval_data_by_attr_class = defaultdict(list)

    # First, get convert PRISM dataset to LQAData's.
    convs_df, survey_df = get_prism_dataset()
    for _, conv_row in convs_df.iterrows():
        user = conv_row.user_id
        survey_row = survey_df[survey_df["user_id"] == user]
        assert len(survey_row) == 1
        survey_row = survey_row.iloc[0]

        tag = f"prism_{conv_row.conversation_id}"
        # For each conversation, we extract the ground truth attributes for 5 attribute classes.
        user_attributes = extract_user_attributes_from_prism_dataset(survey_row)

        user_prompt, context = extract_user_prompt_and_context_from_conv(
            conv_row.chosen_conversation
        )

        # We generate a datapoint for each attribute class and for each question
        # This means the same subject input (conversation) will be evaluated multiple times
        # each with a different attribute class and question.
        for attr_class, attr in user_attributes.items():
            if attr is None:
                continue

            qas = generate_mc_qas_for_attr(
                attr_label=attr, stems_by_attr_class=mc_stems_by_attr_class, rng=rng
            )
            for qa in qas:
                eval_data_by_attr_class[attr_class].append(
                    LQAData(
                        attr_class=attr_class,
                        attr=attr,  # Use ground truth attribute.
                        system_prompt="You are a helpful assistant.",
                        user_prompt=user_prompt,
                        context_conversation=context,
                        qa=qa,
                        id=len(eval_data_by_attr_class[attr_class]),
                        tag=tag,
                    )
                )

    # For hf_model_id == "meta-llama/Llama-3.1-8B-Instruct" and attr_class == Gender,
    # we focus only on conversations where the subject model had a consistent belief,
    # and we use that belief as the label instead of the ground truth.

    if hf_model_id == "meta-llama/Llama-3.1-8B-Instruct":
        eval_data_by_attr_class[Gender] = []  # Ignore the previous results

        hf_dataset = load_dataset(
            "Transluce/PRISM-gender-Llama-3.1-8B-Instruct", split="train"
        )  # There is only one split
        for row in hf_dataset:
            row = cast(dict[str, Any], row)

            attr_class = ALL_USER_ATTRIBUTES[row["attr_class"]]
            assert attr_class == Gender
            attr = attr_class(row["attr"])

            user_prompt, context = extract_user_prompt_and_context_from_conv(row["conversation"])

            qas = generate_mc_qas_for_attr(
                attr_label=attr, stems_by_attr_class=mc_stems_by_attr_class, rng=rng
            )
            for qa in qas:
                eval_data_by_attr_class[Gender].append(
                    LQAData(
                        attr_class=Gender,
                        attr=attr,
                        system_prompt="You are a helpful assistant.",
                        user_prompt=user_prompt,
                        context_conversation=context,
                        qa=qa,
                        id=len(eval_data_by_attr_class[Gender]),
                        tag=row["tag"],
                    )
                )

    return eval_data_by_attr_class


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hf_model_id",
        type=str,
        choices=["meta-llama/Llama-3.1-8B-Instruct", "meta-llama/Llama-3.1-70B-Instruct"],
    )
    parser.add_argument("--general_db_dir", type=str, required=True)
    parser.add_argument("--mc_db_dir", type=str, required=True)
    parser.add_argument("--mc_stems_path", type=str, required=True)
    parser.add_argument("--test_attr_classes", type=str, nargs="+", default=None)
    parser.add_argument("--save_dir", type=str, required=True)
    args = parser.parse_args()

    rng = np.random.default_rng(382)
    if args.test_attr_classes is None:
        test_attr_classes: list[Type[UserAttribute]] = TEST_ATTR_CLASSES
    else:
        test_attr_classes = [
            ALL_USER_ATTRIBUTES[attr_class_name] for attr_class_name in args.test_attr_classes
        ]

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    #################################################################
    # Generate training datasets based on SynthSys + general/MC QA's.
    #################################################################
    mixed_train_data, all_data = get_train_data(
        general_db_dir=args.general_db_dir,
        mc_db_dir=args.mc_db_dir,
        test_attr_classes=test_attr_classes,
        rng=rng,
        mc_ratio=0.5,
    )
    with open(save_dir / "train.pkl", "wb") as f:
        pickle.dump(mixed_train_data, f)
    with open(save_dir / "train_full.pkl", "wb") as f:
        pickle.dump(all_data, f)

    ##########################################################################
    # Generate evaluation datasets based on SynthSys, SelfDescribe, and PRISM.
    ##########################################################################
    # First, load paraphrased direct questions (e.g. "What does the model believe is the user's
    # gender?").
    with open(args.mc_stems_path, "rb") as f:
        mc_stems_by_attr_class = pickle.load(f)

    synthsys_dir = save_dir / "eval_synthsys"
    synthsys_dir.mkdir(parents=True, exist_ok=True)

    # Generate SynthSys evaluation data using held-out attribute classes.
    heldout_templ_mc = get_synthsys_eval_data(
        hf_model_id=args.hf_model_id,
        attr_classes_to_consider=test_attr_classes,
        mc_stems_by_attr_class=mc_stems_by_attr_class,
        rng=rng,
    )
    with open(synthsys_dir / "heldout.pkl", "wb") as f:
        pickle.dump(heldout_templ_mc, f)

    # Generate SynthSys evaluation data using non-held-out attribute classes.
    non_heldout_templ_mc = get_synthsys_eval_data(
        hf_model_id=args.hf_model_id,
        attr_classes_to_consider=[
            attr_class
            for attr_class in ALL_USER_ATTRIBUTES.values()
            if attr_class not in test_attr_classes
        ],
        mc_stems_by_attr_class=mc_stems_by_attr_class,
        rng=rng,
    )
    with open(synthsys_dir / "non_heldout.pkl", "wb") as f:
        pickle.dump(non_heldout_templ_mc, f)

    # Generate SelfDescribe evaluation data.
    self_describe_eval_data = get_selfdescribe_eval_data(
        hf_model_id=args.hf_model_id,
        mc_stems_by_attr_class=mc_stems_by_attr_class,
        rng=rng,
    )
    self_describe_dir = save_dir / "eval_selfdescribe"
    self_describe_dir.mkdir(parents=True, exist_ok=True)
    for attr_class, data in self_describe_eval_data.items():
        with open(self_describe_dir / f"{attr_class.__name__}.pkl", "wb") as f:
            pickle.dump(data, f)

    # Generate PRISM evaluation data.
    prism_eval_data = get_prism_eval_data(
        hf_model_id=args.hf_model_id,
        mc_stems_by_attr_class=mc_stems_by_attr_class,
        rng=rng,
    )
    prism_dir = save_dir / "eval_prism"
    prism_dir.mkdir(parents=True, exist_ok=True)
    for attr_class, data in prism_eval_data.items():
        with open(prism_dir / f"{attr_class.__name__}.pkl", "wb") as f:
            pickle.dump(data, f)


if __name__ == "__main__":
    main()
