import json
import logging
import os
import pickle
import random
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from llm_util.prod_llms import get_llm_completions_async
from llm_util.types import LLMOutput, ModelCallParams
from tqdm import tqdm
from user_modeling.belief_eval import (
    BeliefEvalConfig,
    BeliefEvalInput,
    BeliefEvalType,
    get_belief_eval_results_from_db,
    run_belief_evaluation,
)
from user_modeling.user_attribute import (
    ALL_USER_ATTRIBUTES,
    Country,
    Gender,
    Occupation,
    Religion,
    UserAttribute,
)

WIKIPEDIA_OPTION_PROMPT = """
Could you generate {num_samples} unique possible values for the attribue "{attribute}"?

For example, for the "gender" attribute:
"male"
"female"

Format your output in the following way:
```
# Possible Values
1. <value 1>
...
N. <value N>
# End of Possible Values
```

Your values should be unique.
""".strip()

WIKIPEDIA_DESCRIPTION_PROMPT = """
Could you generate {num_samples} stereotypical descriptions, in the first-person, of a person with the attribute "{attribute}" being "{value}"? It should be **easy** to infer the attribute from the description.

Format your output in the following way:
```
# Descriptions
1. <description 1>
...
N. <description N>
# End of Descriptions
```

Remember:
- You should be be creative with your stereotypical descriptions.
- Make sure to not include any explicit mention of "{value}".
- Start each description with the pronoun "I" and ensure it is only one sentence long.
""".strip()

WIKIPEDIA_PROMPT_REQUEST = "Write a hypothetical but realistic Wikipedia biography infobox for me."
WIKIPEDIA_PROMPT_RESPONSE = "{{Infobox person\n|"
ATTRIBUTE_PROMPTS = {
    "gender": " gender =",
    "country": " country =",
    "occupation": " occupation =",
    "religion": " religion =",
}
EXPLICIT_ATTRIBUTE_DESCRIPTIONS = {
    "gender": "I am a {value}.",
    "country": "I am from {value}.",
    "occupation": "I work as a {value}.",
    "religion": "I am a {value}.",
}

ATTRIBUTE_COUNTS = {
    "gender": 3,  # to limit to male, female, and non-binary, as in gender.py
}


def parse_completion_for_numbered_list(
    completion_text: str, start_marker: str, end_marker: str
) -> list[str]:
    """
    Parse numbered list from LLM completion text that follows a specific format.

    The text should contain sections marked with `start_marker` and `end_marker`.
    Each item is expected to be on a new line, possibly numbered.

    Args:
        completion_text: The text output from the LLM
        start_marker: The marker that indicates the start of the list
        end_marker: The marker that indicates the end of the list

    Returns:
        A list of extracted descriptions with numbering removed
    """
    # Find the section boundaries
    start_idx = completion_text.find(start_marker)
    end_idx = completion_text.find(end_marker)

    # Return empty list if markers aren't found
    if start_idx == -1 or end_idx == -1:
        return []

    # Extract the text between markers
    text = completion_text[start_idx + len(start_marker) : end_idx].strip()

    # Process each line to extract descriptions
    items: list[str] = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue

        # Remove leading numbers, dots, and whitespace (e.g., "1. ", "2. ")
        line = line.lstrip("0123456789").lstrip(".").lstrip(" ")
        line = line.rstrip().rstrip(".")
        if line:
            items.append(line)

    return items


async def generate_wikipedia_valid_latent_attributes(
    num_samples: int, num_tries: int, base_path: str
):
    """
    Generate valid latent attributes for the Wikipedia dataset.
    """
    os.makedirs(base_path, exist_ok=True)

    model_call_params_list = [
        ModelCallParams(provider="anthropic", model_name="claude-3-7-sonnet-20250219"),
    ]

    for attribute in ATTRIBUTE_PROMPTS.keys():
        print(f"Generating {attribute}...")
        options = set()
        for model_call_params in model_call_params_list:
            for _ in range(num_tries):
                prompt = WIKIPEDIA_OPTION_PROMPT.format(
                    num_samples=ATTRIBUTE_COUNTS.get(
                        attribute, num_samples
                    ),  # override if we have a limit
                    attribute=attribute,
                )
                responses: list[LLMOutput] = await get_llm_completions_async(
                    messages_list=[[{"role": "user", "content": prompt}]],
                    max_new_tokens=2048,
                    model=model_call_params,
                )
                response = responses[0].completions[0].text
                assert response is not None
                print(response)
                descriptions = parse_completion_for_numbered_list(
                    response,
                    start_marker="# Possible Values",
                    end_marker="# End of Possible Values",
                )
                options.update(descriptions)

        with open(os.path.join(base_path, f"{attribute}.json"), "w") as f:
            json.dump(list(options), f)


async def generate_wikipedia_stereotypes(num_samples: int, num_tries: int, base_path: str):
    """
    Generate stereotypical descriptions based on some random latent attributes.

    Args:
        num_samples: The number of descriptions to generate for each call to the LLM
        num_tries: Number of calls to the LLM
        base_path: The path to save the generated descriptions
    """
    os.makedirs(base_path, exist_ok=True)

    # Use both openai and anthropic models to generate descriptions.
    model_call_params_list = [
        ModelCallParams(provider="openai", model_name="gpt-4o-mini-2024-07-18"),
        ModelCallParams(provider="anthropic", model_name="claude-3-7-sonnet-20250219"),
    ]

    all_descriptions: list[dict[str, str | dict[str, list[str]]]] = []
    for attribute in ATTRIBUTE_PROMPTS.keys():
        if not os.path.exists(os.path.join(base_path, f"{attribute}.json")):
            print(f"Skipping {attribute} because it doesn't exist")
            continue
        with open(os.path.join(base_path, f"{attribute}.json")) as f:
            options = json.load(f)
        if len(options) == 0:
            print(f"Skipping {attribute} because it has no options")
            continue
        for model_call_params in model_call_params_list:
            for _ in tqdm(range(num_tries), desc=f"Generating {attribute}"):
                # construct prompt to get the descriptions
                value = random.choice(options)
                prompt = WIKIPEDIA_DESCRIPTION_PROMPT.format(
                    num_samples=num_samples,
                    attribute=attribute,
                    value=value,
                )

                # get the descriptions
                responses: list[LLMOutput] = await get_llm_completions_async(
                    messages_list=[[{"role": "user", "content": prompt}]],
                    max_new_tokens=2048,
                    model=model_call_params,
                )
                response = responses[0].completions[0].text
                assert response is not None
                descriptions = parse_completion_for_numbered_list(
                    response,
                    start_marker="# Descriptions",
                    end_marker="# End of Descriptions",
                )

                # store
                for description in tqdm(descriptions, desc=f"Generating {attribute}"):
                    # check what the subject generates
                    result = {"description": description, "latent_attributes": {attribute: [value]}}
                    all_descriptions.append(result)

    with open(os.path.join(base_path, "wikipedia_stereotypes.jsonl"), "w") as f:
        for description in all_descriptions:
            f.write(json.dumps(description) + "\n")


def get_full_self_description_df(stereotypes_path: str):
    datapoints = []
    with open(stereotypes_path) as f:
        for line in f:
            datapoint = json.loads(line)
            attributes = datapoint["latent_attributes"]
            for attr_class_name, attr_name in attributes.items():
                assert len(attr_name) == 1
                attr_name = attr_name[0]

                attr_class = ALL_USER_ATTRIBUTES[attr_class_name.capitalize()]

                if attr_name == "Bahá'í Faith":
                    assert attr_class == Religion
                    attr = attr_class.BAHAI_FAITH
                elif attr_name == "Raëlism":
                    assert attr_class == Religion
                    attr = attr_class.RAELISM
                elif attr_name == "Jehovah's Witnesses":
                    assert attr_class == Religion
                    attr = attr_class.JEHOVAHS_WITNESSES
                else:
                    try:
                        attr = attr_class(attr_name.lower().replace("-", "_").replace(" ", "_"))
                    except ValueError:
                        print(attr_name)
                        continue

                datapoints.append(
                    {
                        "description": datapoint["description"],
                        "attr_class": attr_class,
                        "attr": attr,
                    }
                )
    df = pd.DataFrame(datapoints)
    return df


async def create_self_describe_dataset(
    stereotypes_path: str,
    subject_hf_model_id: str,
    save_dir: str | Path,
    subject_client_url: str,
):
    [Gender, Country, Occupation, Religion]
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    #########################################################
    # Create and save data required for belief evaluation.
    #########################################################
    # Create list of BeliefEvalInput's.
    stereotypes_df = get_full_self_description_df(stereotypes_path)

    eval_inputs: list[BeliefEvalInput] = []
    eval_inputs_by_tag: dict[str, BeliefEvalInput] = {}
    descriptions_by_tag: dict[str, str] = {}
    for idx, row in stereotypes_df.iterrows():
        tag = f"selfdescribe_{idx}"
        eval_input = BeliefEvalInput(
            attr_class=row.attr_class,
            attr=row.attr_class(row.attr),
            context_conversation=(
                ("system", "You are a helpful assistant."),
                ("user", f"{row.description}."),
            ),
            tag=f"selfdescribe_{idx}",
        )
        eval_inputs.append(eval_input)
        eval_inputs_by_tag[tag] = eval_input
        descriptions_by_tag[tag] = row.description
    with open(save_dir / "belief_eval_inputs.pkl", "wb") as f:
        pickle.dump(eval_inputs, f)

    #########################################################
    # Do belief evaluation.
    #########################################################
    logging.getLogger("llm_util.openai").setLevel(logging.ERROR)
    logging.getLogger("llm_util.prod_llms").setLevel(logging.WARNING)

    config = BeliefEvalConfig(
        subject_hf_model_id=subject_hf_model_id,
        eval_input_path=str(save_dir / "belief_eval_inputs.pkl"),
        eval_type=BeliefEvalType.WIKI_GEN,
        add_to_last_user_message=True,
        generation_max_new_tokens=10,
    )
    db_dir = save_dir / "self_describe_db"
    await run_belief_evaluation(
        config,
        db_dir=str(db_dir),
        attr_class=None,
        subject_client_url=subject_client_url,
        skip_existing=True,
        eval_inputs_batch_size=32,
    )

    #########################################################
    # Do filtering.
    #########################################################

    results_df = get_belief_eval_results_from_db(str(db_dir))

    filtered_data: list[dict[str, Any]] = []
    for tag, df in results_df.groupby("tag"):  # type: ignore
        tag = cast(str, tag)

        eval_input = eval_inputs_by_tag[tag]
        attr_class = eval_input.attr_class
        attr_values: list[UserAttribute] = [a for a in attr_class]
        gt_attr = eval_input.attr

        # Compute mean belief over multiple generated completions.
        probs = np.stack(df.probs.values)
        mean_probs = probs.mean(axis=0)

        # Check whether the revealed belief matches the ground truth.
        pred_idx = np.argmax(mean_probs)
        pred_attr = attr_values[pred_idx] if pred_idx < len(attr_class) else None
        if pred_attr != gt_attr:
            continue
        assert pred_attr is not None

        description = descriptions_by_tag[tag].strip().rstrip(".")
        filtered_data.append(
            {
                "attr_class": attr_class,
                "attr": pred_attr,
                "system_prompt": "You are a helpful assistant.",  # Dummy system prompt
                "user_prompt": f"{description}. {WIKIPEDIA_PROMPT_REQUEST}",
                "pred_prob": mean_probs[pred_idx],
                "tag": tag,
            }
        )

    num_correct = len(filtered_data)
    total = results_df.tag.nunique()
    print(f"{num_correct}/{total} examples passed the filtering criteria.")

    with open(save_dir / "filtered_data.pkl", "wb") as f:
        pickle.dump(filtered_data, f)
