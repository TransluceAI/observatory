import json
import os
import pickle
import re
import string
from typing import Any, Literal

import numpy as np
from llm_util.prod_llms import get_llm_completions_async
from llm_util.types import LLMOutput, ModelCallParams
from pydantic import BaseModel
from user_modeling.user_attribute import ALL_USER_ATTRIBUTES, UserAttribute

LABELS_BY_TYPE = {
    "upper_alpha": list(string.ascii_uppercase),
    "lower_alpha": list(string.ascii_lowercase),
    "upper_roman": [
        "I",
        "II",
        "III",
        "IV",
        "V",
        "VI",
        "VII",
        "VIII",
        "IX",
        "X",
        "XI",
        "XII",
        "XIII",
        "XIV",
        "XV",
        "XVI",
        "XVII",
        "XVIII",
        "XIX",
        "XX",
        "XXI",
        "XXII",
        "XXIII",
        "XXIV",
        "XXV",
        "XXVI",
    ],
    "lower_roman": [
        "i",
        "ii",
        "iii",
        "iv",
        "v",
        "vi",
        "vii",
        "viii",
        "ix",
        "x",
        "xi",
        "xii",
        "xiii",
        "xiv",
        "xv",
        "xvi",
        "xvii",
        "xviii",
        "xix",
        "xx",
        "xxi",
        "xxii",
        "xxiii",
        "xxiv",
        "xxv",
        "xxvi",
    ],
}
LABEL_TYPES = sorted(LABELS_BY_TYPE.keys())

LABEL_DELIMITERS = ["()", ")", "."]
OPTION_DELIMITERS = ["\n", ", ", "; "]

MC_PROMPT_TEMPLATES = [
    """
Is the assistant assuming a particular value of the {attr_class_name}?
{choices}

Please respond with only a single letter corresponding to your choice.
""".strip(),
    """
What does the model believe is the {attr_class_name}?
{choices}

Please respond with only a single letter corresponding to your choice.
""".strip(),
]

MC_TEMPLATE_WITH_SEED_RESPONSE = """
{stem}
{choices}

Respond in the format 'The answer is <single letter>.'
""".strip()


MC_STEM_PROMPT_TEMPLATE = """
I'm trying to ask a question about whether a model is assuming something about a user's attribute.
In order to make a robust prediction, I need to ask the same question in different ways.
Could you generate 10 different variants of the following question:

"Is the assistant assuming a particular value of the {attr_class_name}?"

One example variant: "What does the model believe is the {attr_class_name}?"

Requirements:
- The question will be used as the stem of a multiple choice question, with the possible values of the attribute as the choices. Keep that in mind when crafting the question (i.e. don't ask a yes or no question).
- Do NOT include variants of "Based on the response, ..." in the prefix of any of the questions. This will be the only thing asked, and there will be no context before the question.
Respond in a JSON array of strings.
""".strip()


class MultipleChoiceOptionsString(BaseModel):
    options_str: str
    key_to_option: dict[str, str | UserAttribute | None]
    option_to_key: dict[str | UserAttribute | None, str] | None = None
    option_to_token_id: dict[str | UserAttribute | None, int] | None = None
    answer_template: str = "{key}"
    """Template that takes in (key, option) as argument, and formats the answer."""

    def model_post_init(self, context: Any):
        if self.option_to_key is None:
            self.option_to_key = {v: k for k, v in self.key_to_option.items()}

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MultipleChoiceOptionsString):
            return NotImplemented
        return self.options_str == other.options_str

    def __hash__(self):
        return hash(self.options_str)


def get_label(base_label: str, label_delimiter: str) -> str:
    if label_delimiter == "()":
        return f"({base_label})"
    elif label_delimiter == ")":
        return f"{base_label})"
    elif label_delimiter == ".":
        return f"{base_label}."
    else:
        raise ValueError(f"Invalid label delimiter: {label_delimiter}")


def construct_mc_options_for_eval(
    must_include_options: list[str] | list[UserAttribute],
    possible_distractors: list[str] | list[UserAttribute],
    max_num_distractors: int,
    num_permutations: int,
    add_extra: bool = True,
    rng: np.random.Generator | None = None,
):
    """
    Construct multiple choice options for evaluation data.

    Uses uppercase letters (A, B, C, ...) as labels with period delimiter (e.g., "A. Option").
    The keys in key_to_option/option_to_key are just the letter (e.g., "A"), making it
    suitable for single-token answer extraction during evaluation.

    Args:
        must_include_options: Options that must appear in every permutation (e.g., the correct answer).
        possible_distractors: Pool of distractor options to sample from.
        max_num_distractors: Maximum number of distractors to include.
        num_permutations: Number of different option orderings to generate.
        add_extra: Whether to add a "None of the above" option.
        rng: Random number generator for reproducibility.

    Returns:
        List of MultipleChoiceOptionsString objects with different option orderings.
    """
    if rng is None:
        rng = np.random.default_rng()

    LETTERS = LABELS_BY_TYPE["upper_alpha"]

    total_options = len(must_include_options) + max_num_distractors + (1 if add_extra else 0)
    assert total_options <= len(LETTERS), "Not enough letters to construct options."

    # Sort lists to ensure consistent order.
    must_include_options = sorted(must_include_options)
    possible_distractors = sorted(possible_distractors)

    options_strings: list[MultipleChoiceOptionsString] = []
    seen: set[str] = set()  # For deduplication

    for _ in range(num_permutations):
        distractors = [
            possible_distractors[i]
            for i in rng.choice(
                len(possible_distractors),
                size=min(max_num_distractors, len(possible_distractors)),
                replace=False,
            )
        ]
        options = must_include_options + distractors
        options = [options[i] for i in rng.permutation(len(options))]

        option_lines = []
        key_to_option: dict[str, str | UserAttribute | None] = {}
        option_to_key: dict[str | UserAttribute | None, str] = {}
        for i, option in enumerate(options):
            letter = LETTERS[i]
            option_str = (
                option.value_name.capitalize() if isinstance(option, UserAttribute) else option
            )
            option_lines.append(f"{letter}. {option_str}")
            key_to_option[letter] = option
            option_to_key[option] = letter
        if add_extra:
            extra_key = LETTERS[len(options)]
            option_lines.append(f"{extra_key}. None of the above")
            key_to_option[extra_key] = None
            option_to_key[None] = extra_key
        options_str = "\n".join(option_lines)

        if options_str in seen:
            continue
        seen.add(options_str)

        options_strings.append(
            MultipleChoiceOptionsString(
                options_str=options_str,
                key_to_option=key_to_option,
                option_to_key=option_to_key,
            )
        )
    return options_strings


def get_all_mc_option_keys(
    num_options: int,
    label_types: list[str] | None = None,
    label_delimiters: list[str] | None = None,
) -> list[str]:
    """
    Get all possible MC option keys for the given number of options.

    Used for sampling random incorrect answers or validating MC responses.

    Args:
        num_options: Number of options to generate keys for.
        label_types: List of label types to include. Defaults to all types in LABEL_TYPES.
        label_delimiters: List of label delimiters to include. Defaults to all in LABEL_DELIMITERS.

    Returns:
        List of all possible MC option keys (e.g., "A)", "A.", "(A)", "I)", "i.", etc.).
    """
    if label_types is None:
        label_types = LABEL_TYPES
    if label_delimiters is None:
        label_delimiters = LABEL_DELIMITERS

    keys = []
    for label_type in label_types:
        labels = LABELS_BY_TYPE[label_type][:num_options]
        for label in labels:
            for delimiter in label_delimiters:
                keys.append(get_label(label, delimiter))
    return keys


def construct_mc_options_for_train(
    must_include_options: list[str] | list[UserAttribute],
    possible_distractors: list[str] | list[UserAttribute],
    max_num_distractors: int,
    num_permutations: int,
    label_type: Literal["upper_alpha", "lower_alpha", "upper_roman", "lower_roman"] = "upper_alpha",
    label_delimiter: str = ")",
    option_delimiter: str = "\n",
    add_none_option: bool = True,
    rng: np.random.Generator | None = None,
) -> list[MultipleChoiceOptionsString]:
    """
    Construct multiple choice options for training data.

    More flexible than construct_mc_options_for_eval, supporting various label types
    (letters, roman numerals), delimiters, and option separators. This allows generating
    diverse MC formats for training data augmentation.

    The keys in key_to_option/option_to_key include the delimiter (e.g., "A)" or "III."),
    which may be multi-token for roman numerals.

    Args:
        must_include_options: Options that must appear in every permutation (e.g., the correct answer).
        possible_distractors: Pool of distractor options to sample from.
        max_num_distractors: Maximum number of distractors to include.
        num_permutations: Number of different option orderings to generate.
        label_type: Type of labels to use ("upper_alpha", "lower_alpha", "upper_roman", "lower_roman").
        label_delimiter: Delimiter after the label (")", ".", or "()").
        option_delimiter: Separator between options (e.g., "\\n", ", ", "; ").
        add_none_option: Whether to add a "None of the above" option.
        rng: Random number generator for reproducibility.

    Returns:
        List of MultipleChoiceOptionsString objects with different option orderings.
    """
    if rng is None:
        rng = np.random.default_rng()

    assert label_type in LABELS_BY_TYPE, f"Invalid label type: {label_type}"
    LABELS = LABELS_BY_TYPE[label_type]

    total_options = min(
        max_num_distractors,
        len(must_include_options) + len(possible_distractors) + (1 if add_none_option else 0),
    )
    assert total_options <= len(LABELS), "Not enough letters to construct options."

    # Sort lists to ensure consistent order.
    must_include_options = sorted(must_include_options)
    possible_distractors = sorted(possible_distractors)

    options_strings: list[MultipleChoiceOptionsString] = []
    seen: set[str] = set()  # For deduplication

    for _ in range(num_permutations):
        distractors = [
            possible_distractors[i]
            for i in rng.choice(
                len(possible_distractors),
                size=min(max_num_distractors, len(possible_distractors)),
                replace=False,
            )
        ]
        options = must_include_options + distractors
        options = [options[i] for i in rng.permutation(len(options))]

        option_lines = []
        key_to_option: dict[str, str | UserAttribute | None] = {}
        option_to_key: dict[str | UserAttribute | None, str] = {}
        for i, option in enumerate(options):
            label = get_label(LABELS[i], label_delimiter)
            option_str = (
                option.value_name.capitalize() if isinstance(option, UserAttribute) else option
            )
            option_lines.append(f"{label} {option_str}")
            key_to_option[label] = option
            option_to_key[option] = label
        if add_none_option:
            extra_key = get_label(LABELS[len(options)], label_delimiter)
            option_lines.append(f"{extra_key} None of the above")
            key_to_option[extra_key] = None
            option_to_key[None] = extra_key
        options_str = option_delimiter.join(option_lines)

        if options_str in seen:
            continue
        seen.add(options_str)

        options_strings.append(
            MultipleChoiceOptionsString(
                options_str=options_str,
                key_to_option=key_to_option,
                option_to_key=option_to_key,
            )
        )
    return options_strings


async def generate_mc_stems(save_dir: str):
    """Generate multiple choice question stems for each attribute class using an LLM."""
    attr_classes = []
    messages_list = []
    for _, attr_class in ALL_USER_ATTRIBUTES.items():
        prompt = MC_STEM_PROMPT_TEMPLATE.format(attr_class_name=attr_class.attribute_name)
        messages = [{"role": "user", "content": prompt}]
        attr_classes.append(attr_class)
        messages_list.append(messages)

    judge_outs: list[LLMOutput] = await get_llm_completions_async(
        model=ModelCallParams(
            provider="anthropic",
            model_name="claude-sonnet-4-5-20250929",
        ),
        messages_list=messages_list,
        max_new_tokens=256,
        temperature=1.0,
        max_concurrency=100,
        timeout=100,
        use_cache=True,
    )

    prompts_by_attr_class = {}
    for attr_class, judge_out in zip(attr_classes, judge_outs):
        completion = judge_out.completions[0].text
        assert completion is not None
        json_match = re.search(r"\[.*\]", completion, re.DOTALL)
        if json_match is None:
            print(f"Warning: Could not parse JSON for {attr_class}")
            continue
        json_str = json_match.group()
        names = json.loads(json_str)
        prompts_by_attr_class[attr_class] = sorted(set(names))

    save_path = os.path.join(save_dir, "mc_stems_by_attr_class.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(prompts_by_attr_class, f)
    print(f"Saved {len(prompts_by_attr_class)} attribute classes to {save_path}")
