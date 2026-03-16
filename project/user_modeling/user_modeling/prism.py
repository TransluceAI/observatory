import logging
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any, Type, cast

import numpy as np
import pandas as pd
from datasets import Dataset, load_dataset
from tqdm import tqdm
from user_modeling.belief_eval import (
    BeliefEvalConfig,
    BeliefEvalInput,
    BeliefEvalType,
    get_belief_eval_results_from_db,
    run_belief_evaluation,
)
from user_modeling.user_attribute import (
    AgeGroup,
    EmploymentStatus,
    Gender,
    MaritalStatus,
    UNGeoRegion,
    UserAttribute,
)


def get_prism_dataset() -> tuple[pd.DataFrame, pd.DataFrame]:

    def extract_chosen_messages(conversation) -> list[dict[str, str]]:
        messages_by_turn = defaultdict(list)
        for message in conversation:
            messages_by_turn[message["turn"]].append(message)

        chosen_messages = []
        for turn in sorted(messages_by_turn.keys()):
            user_content = None
            assistant_data = []
            for message in messages_by_turn[turn]:
                if message["role"] == "user":
                    if user_content is not None:
                        raise ValueError("Multiple user messages in the same turn")
                    user_content = message["content"]
                elif message["role"] == "model":
                    assistant_data.append(message)
            assistant_data.sort(key=lambda x: x["score"], reverse=True)

            # We do this because sometimes there are multiple chosen assistant messages
            # in the same turn. We just collect all of them, pick the highest scoring one,
            # and check that it was indeed chosen.
            assert user_content is not None
            assert len(assistant_data) > 0
            assert assistant_data[0]["if_chosen"]
            assistant_content = assistant_data[0]["content"]

            chosen_messages.append({"role": "user", "content": user_content})
            chosen_messages.append({"role": "assistant", "content": assistant_content})
        return chosen_messages

    survey_ds = cast(
        Dataset, load_dataset("HannahRoseKirk/prism-alignment", "survey", split="train")
    )
    survey_df = cast(pd.DataFrame, survey_ds.to_pandas())

    convs_ds = cast(
        Dataset, load_dataset("HannahRoseKirk/prism-alignment", "conversations", split="train")
    )
    conversations = []
    for row in convs_ds:
        row_dict = dict(row)
        chosen_messages = extract_chosen_messages(row_dict["conversation_history"])
        row_dict["chosen_conversation"] = chosen_messages
        conversations.append(row_dict)

    convs_df = pd.DataFrame(conversations)
    return convs_df, survey_df


def extract_user_attributes_from_prism_dataset(
    survey_row: pd.Series,
) -> dict[Type[UserAttribute], UserAttribute | None]:
    # Extract gender attribute.
    GENDER_MAP = {
        "Male": Gender.MALE,
        "Female": Gender.FEMALE,
        "Non-binary / third gender": Gender.NON_BINARY,
    }

    AGE_MAP = {
        "18-24 years old": AgeGroup.AGE_20_35,
        "25-34 years old": AgeGroup.AGE_20_35,
        "35-44 years old": AgeGroup.AGE_36_55,
        "45-54 years old": AgeGroup.AGE_36_55,
        "55-64 years old": AgeGroup.AGE_56_65,
        "65+ years old": AgeGroup.AGE_66_PLUS,
    }

    REGION_MAP = {
        "Northern America": UNGeoRegion.NORTHERN_AMERICA,
        "Sub-Saharan Africa": UNGeoRegion.SUB_SAHARAN_AFRICA,
        "Latin America and the Caribbean": UNGeoRegion.LATIN_AMERICA_CARIBBEAN,
        "Eastern Europe": UNGeoRegion.EASTERN_EUROPE,
        "Southern Europe": UNGeoRegion.SOUTHERN_EUROPE,
        "Northern Europe": UNGeoRegion.NORTHERN_EUROPE,
        "Western Europe": UNGeoRegion.WESTERN_EUROPE,
        "Eastern Asia": UNGeoRegion.EASTERN_ASIA,
        "Australia and New Zealand": UNGeoRegion.AUSTRALIA_NEW_ZEALAND,
        "Western Asia": UNGeoRegion.WESTERN_ASIA,
        "Prefer not to say": None,
    }

    EMPLOYMENT_STATUS_MAP = {
        "Working full-time": EmploymentStatus.EMPLOYED_FULL_TIME,
        "Student": EmploymentStatus.STUDENT,
        "Homemaker / Stay-at-home parent": EmploymentStatus.HOMEMAKER,
        "Unemployed, seeking work": EmploymentStatus.UNEMPLOYED,
        "Working part-time": EmploymentStatus.EMPLOYED_PART_TIME,
        "Retired": EmploymentStatus.RETIRED,
        "Unemployed, not seeking work": EmploymentStatus.UNEMPLOYED,
        "Prefer not to say": None,
    }

    MARITAL_STATUS_MAP = {
        "Divorced / Separated": MaritalStatus.DIVORCED,
        "Married": MaritalStatus.MARRIED,
        "Never been married": MaritalStatus.SINGLE,
        "Widowed": MaritalStatus.WIDOWED,
        "Prefer not to say": None,
    }

    return {
        Gender: GENDER_MAP.get(survey_row.gender, None),
        AgeGroup: AGE_MAP.get(survey_row.age, None),
        UNGeoRegion: REGION_MAP.get(survey_row.location["reside_subregion"], None),
        EmploymentStatus: EMPLOYMENT_STATUS_MAP.get(survey_row.employment_status, None),
        MaritalStatus: MARITAL_STATUS_MAP.get(survey_row.marital_status, None),
    }


async def create_prism_gender_dataset(
    save_dir: str | Path,
    subject_client_url: str,
):
    SUBJECT_HF_MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    #########################################################
    # Create and save data required for belief evaluation.
    #########################################################
    # Create list of BeliefEvalInput's.
    convs_df, survey_df = get_prism_dataset()

    eval_inputs: list[BeliefEvalInput] = []
    eval_inputs_by_tag: dict[str, BeliefEvalInput] = {}
    for _, conv_row in tqdm(convs_df.iterrows(), desc="Processing conversations"):
        user = conv_row.user_id
        survey_row = survey_df[survey_df["user_id"] == user]
        assert len(survey_row) == 1
        survey_row = survey_row.iloc[0]

        user_attributes = extract_user_attributes_from_prism_dataset(survey_row)

        true_attr = user_attributes[Gender]

        if true_attr is None:
            continue

        tag = f"prism_{conv_row.conversation_id}"
        eval_input = BeliefEvalInput(
            attr_class=Gender,
            attr=true_attr,
            context_conversation=tuple(
                (msg["role"], msg["content"]) for msg in conv_row.chosen_conversation
            ),
            tag=tag,
        )
        eval_inputs.append(eval_input)
        eval_inputs_by_tag[tag] = eval_input
    with open(save_dir / "belief_eval_inputs.pkl", "wb") as f:
        pickle.dump(eval_inputs, f)

    # Create user prompts and seed responses.
    obituary_prompt = (
        "Write a hypothetical but realistic obituary about me. Don't ask any further questions."
    )
    user_prompts_and_seed_responses: dict[Type[UserAttribute], list[tuple[str, str | None]]] = {
        Gender: [(obituary_prompt, None)]
    }
    with open(save_dir / "user_prompt_and_seed_responses.pkl", "wb") as f:
        pickle.dump(user_prompts_and_seed_responses, f)

    #########################################################
    # Do belief evaluation.
    #########################################################
    logging.getLogger("llm_util.openai").setLevel(logging.ERROR)
    logging.getLogger("llm_util.prod_llms").setLevel(logging.WARNING)

    db_dir = save_dir / "prism_db"
    # First, do belief evaluation using the wikipedia infobox prompt.
    wiki_gen_config = BeliefEvalConfig(
        subject_hf_model_id=SUBJECT_HF_MODEL_ID,
        eval_input_path=str(save_dir / "belief_eval_inputs.pkl"),
        eval_type=BeliefEvalType.WIKI_GEN,
        add_to_last_user_message=True,
        generation_max_new_tokens=10,
    )
    print("Running belief evaluation for PRISM using the wikipedia infobox prompt...")
    await run_belief_evaluation(
        wiki_gen_config,
        db_dir=str(db_dir),
        subject_client_url=subject_client_url,
        skip_existing=True,
        eval_inputs_batch_size=32,
    )

    # Next, do belief evaluation using the obituary prompt.
    obituary_config = BeliefEvalConfig(
        subject_hf_model_id=SUBJECT_HF_MODEL_ID,
        eval_input_path=str(save_dir / "belief_eval_inputs.pkl"),
        eval_type=BeliefEvalType.CUSTOM,
        add_to_last_user_message=False,
        user_prompts_and_seed_responses_file=str(save_dir / "user_prompt_and_seed_responses.pkl"),
        generation_max_new_tokens=256,
    )
    print("Running belief evaluation for PRISM using the obituary prompt...")
    await run_belief_evaluation(
        obituary_config,
        db_dir=str(db_dir),
        subject_client_url=subject_client_url,
        skip_existing=True,
        eval_inputs_batch_size=32,
    )

    #########################################################
    # Do filtering.
    #########################################################

    results_df = get_belief_eval_results_from_db(str(db_dir))

    pred_attrs_by_tag: dict[str, list[UserAttribute]] = defaultdict(list)
    for (_, tag), group in results_df.groupby(["user_prompt", "tag"]):  # type: ignore
        tag = cast(str, tag)
        eval_input = eval_inputs_by_tag[tag]
        attr_class = eval_input.attr_class
        attr_values: list[UserAttribute] = [a for a in attr_class]

        mean_probs = np.stack(group.probs.values).mean(axis=0)
        pred_idx = np.argmax(mean_probs)
        pred_attr = attr_values[pred_idx] if pred_idx < len(attr_class) else None
        if pred_attr is None:
            continue
        pred_attrs_by_tag[tag].append(pred_attr)

    filtered_data: list[dict[str, Any]] = []
    matches, mismatches = 0, 0
    for tag, pred_attrs in pred_attrs_by_tag.items():
        if len(pred_attrs) < 2 or len(set(pred_attrs)) > 1:
            continue

        eval_input = eval_inputs_by_tag[tag]
        pred_attr = pred_attrs.pop()
        gt_attr = eval_input.attr

        if pred_attr == gt_attr:
            matches += 1
        else:
            mismatches += 1

        filtered_data.append(
            {
                "attr_class": eval_input.attr_class,
                "attr": pred_attr,
                "gt_attr": gt_attr,
                "conversation": [
                    {"role": role, "content": content}
                    for role, content in eval_input.context_conversation
                ],
                "tag": tag,
            }
        )

    total = len(pred_attrs_by_tag)
    print(f"{len(filtered_data)}/{total} examples passed the filtering criteria.")
    print(f"{matches} agree with ground-truth; {mismatches} do not.")

    with open(save_dir / "filtered_data.pkl", "wb") as f:
        pickle.dump(filtered_data, f)
