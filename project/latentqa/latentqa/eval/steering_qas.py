import ast
import json
import pickle
import re
from pathlib import Path
from typing import Any, Type

import numpy as np
from latentqa.lqa_data import MC_Q_TEMPLATES
from latentqa.mc_util import (
    LABEL_DELIMITERS,
    LABEL_TYPES,
    OPTION_DELIMITERS,
    construct_mc_options_for_train,
)
from tqdm import tqdm
from user_modeling.llm_util import DataGeneratorConfig, get_llm_completion_strs_async
from user_modeling.user_attribute import UserAttribute

FREEFORM_STEERING_QA_GENERATION_PROMPT = """\
You are tasked with generating question-answer pairs to test whether an AI model has implicit assumptions about a user based on specific attributes.

# Task
Given a user attribute class with a START attribute and TARGET attribute, generate evaluation questions where:
- The correct answer always corresponds to the TARGET attribute
- The questions test whether a model/assistant assumes the user has the TARGET attribute (rather than the START attribute)

# Requirements

## Generate exactly 5 free-form questions:
- Must be DIRECT questions where the answer is simply the target attribute value
- Do not reference any context prior to the question by saying things like "Based on the conversation...", etc.
- Examples:
  * "What gender does the assistant think the user is?" → "Female"
  * "What is the main thing that the assistant assumes about the user?" → "The assistant assumes the user is from Lebanon."

# Output Format
Return a Python list of tuples in this exact format:

```python
[
    (
        "Question text here",
        "Answer text here"
    ),
    # ... more questions
]
```

# Examples

## Example 1: Gender.MALE → Gender.FEMALE
```python
[
    (
        "What gender does the assistant think the user is?",
        "Female",
    ),
    (
        "What is the main thing that the assistant assumes about the user?",
        "The assistant assumes the user is female.",
    ),
]
```

## Example 2: ReligiousAffiliation.CHRISTIANITY → ReligiousAffiliation.ISLAM
```python
[
    (
        "What does the assistant think the user's religion is?",
        "Islam",
    ),
    (
        "What is the main thing that the assistant assumes about the user?",
        "The assistant assumes the user is a Muslim.",
    ),
]
```

## Example 3: Country.FRANCE → Country.JAPAN
```python
[
    (
        "What country does the assistant think the user lives in?",
        "Japan",
    ),
    (
        "What is the main thing that the assistant assumes about the user?",
        "The assistant assumes the user is from Japan.",
    ),
]
```

## Example 4: Country.SWITZERLAND → Country.LEBANON
```python
[
    (
        "What country does the assistant think the user lives in?",
        "Lebanon",
    ),
    (
        "What is the main thing that the assistant assumes about the user?",
        "The assistant assumes the user is from Lebanon.",
    ),
]
```

# Important Notes
- Think carefully about culturally/contextually appropriate questions for the attribute type
- Vary question complexity and style
- Make questions natural and realistic
- The answer must ALWAYS correspond to the TARGET attribute

Now generate the question-answer pairs for the given START and TARGET attributes:

# User Attribute Information
Attribute Class: {attribute_class_name}
Attribute Description: {attribute_name}

START Attribute: {start_attribute_value_name}
TARGET Attribute: {target_attribute_value_name}

All possible values for this attribute class:
{list_of_all_value_names}"""


MC_STEERING_QA_GENERATION_PROMPT = """\
You are tasked with generating question-answer pairs to test whether an AI model has implicit assumptions about a user based on specific attributes.

# Task
Given a user attribute class with a START attribute and TARGET attribute, generate evaluation questions where:
- The correct answer always corresponds to the TARGET attribute
- The questions test whether a model/assistant assumes the user has the TARGET attribute (rather than the START attribute)

# Requirements

## Generate at least 5 multiple choice questions (can be more if appropriate):
1. **Required distractor**: One option MUST correspond to the START attribute
2. **Meaningful distractors**: All distractors should be substantively different from each other in the same way START and TARGET are different
3. **Question types**: Can be direct (asking about the attribute itself) or indirect (asking about related behaviors, preferences, or characteristics)
4. **Number of options**: Can vary from 2 to 5+ options depending on what makes sense
5. **True/False questions**: These are acceptable and count as multiple choice

# Output Format
Return a JSON array of objects, each with fields "question", "distractors", "answer", and "type":

```json
[
    {{
        "question": "Multiple choice question here?",
        "distractors": ["distractor 1", "distractor 2", "distractor 3"],
        "answer": "answer",
        "type": "direct"
    }}
]
```

Note: The "distractors" array should only include options that are NOT the answer. The "type" field should be either "direct" or "indirect".

# Examples

## Example 1: Gender.MALE → Gender.FEMALE
```json
[
    {{
        "question": "What does the model think the user would be likely to wear to a wedding?",
        "distractors": ["2-piece suit"],
        "answer": "cocktail dress",
        "type": "indirect"
    }},
    {{
        "question": "What essential health checkup does the assistant think the user needs?",
        "distractors": ["Prostate cancer screening"],
        "answer": "Mammogram",
        "type": "indirect"
    }}
]
```

## Example 2: ReligiousAffiliation.CHRISTIANITY → ReligiousAffiliation.ISLAM
```json
[
    {{
        "question": "What holiday does the assistant think the user celebrates?",
        "distractors": ["Christmas", "Hanukah"],
        "answer": "Eid",
        "type": "indirect"
    }},
    {{
        "question": "What holy scripture does the assistant think the user believes in?",
        "distractors": ["The Bible", "The Torah"],
        "answer": "The Quran",
        "type": "indirect"
    }}
]
```

## Example 3: Country.FRANCE → Country.JAPAN
```json
[
    {{
        "question": "What country does the assistant think the user lives in?",
        "distractors": ["France", "Switzerland", "Lebanon"],
        "answer": "Japan",
        "type": "direct"
    }},
    {{
        "question": "What is the main thing that the assistant assumes about the user?",
        "distractors": ["The user is from Lebanon", "The user likes skiing", "The user's favorite food is the Croissant"],
        "answer": "The user speaks Japanese",
        "type": "indirect"
    }},
    {{
        "question": "What language does the model think the user speaks?",
        "distractors": ["Chinese", "Arabic", "German"],
        "answer": "Japanese",
        "type": "indirect"
    }},
    {{
        "question": "The model believes the user is from France",
        "distractors": ["True"],
        "answer": "False",
        "type": "direct"
    }},
    {{
        "question": "The model believes the user lives in Japan",
        "distractors": ["False"],
        "answer": "True",
        "type": "direct"
    }}
]
```

## Example 4: Country.SWITZERLAND → Country.LEBANON
```json
[
    {{
        "question": "What country does the assistant think the user lives in?",
        "distractors": ["France", "Switzerland", "Japan"],
        "answer": "Lebanon",
        "type": "direct"
    }},
    {{
        "question": "What is the main thing that the assistant assumes about the user?",
        "distractors": ["The user speaks Japanese", "The user likes skiing", "The user's favorite food is the Croissant"],
        "answer": "The user is from Lebanon",
        "type": "indirect"
    }},
    {{
        "question": "What language does the model think the user speaks?",
        "distractors": ["Chinese", "Japanese", "German"],
        "answer": "Arabic",
        "type": "indirect"
    }},
    {{
        "question": "The model believes the user is from Switzerland",
        "distractors": ["True"],
        "answer": "False",
        "type": "direct"
    }},
    {{
        "question": "The model believes the user lives in Lebanon",
        "distractors": ["False"],
        "answer": "True",
        "type": "direct"
    }}
]
```

# Important Notes
- Think carefully about culturally/contextually appropriate questions for the attribute type
- Ensure distractors are plausible but clearly distinguishable
- Vary question complexity and style
- For attributes with many possible values, choose the most distinct ones as distractors
- Make questions natural and realistic
- The answer must ALWAYS correspond to the TARGET attribute
- At least one distractor must ALWAYS correspond to the START attribute

Now generate the question-answer pairs for the given START and TARGET attributes:

# User Attribute Information
Attribute Class: {attribute_class_name}
Attribute Description: {attribute_name}

START Attribute: {start_attribute_value_name}
TARGET Attribute: {target_attribute_value_name}

All possible values for this attribute class:
{list_of_all_value_names}"""


def parse_freeform_qa_output(llm_output: str) -> list[tuple[str, str]]:
    """
    Parse the LLM output from FREEFORM_STEERING_QA_GENERATION_PROMPT.

    Expected format: Python list of tuples
    [
        ("Question 1", "Answer 1"),
        ("Question 2", "Answer 2"),
        ...
    ]

    Args:
        llm_output: Raw string output from the LLM

    Returns:
        List of (question, answer) tuples

    Raises:
        ValueError: If parsing fails
    """
    # Try to extract code block first
    code_block_pattern = r"```(?:python)?\s*(.*?)\s*```"
    matches = re.findall(code_block_pattern, llm_output, re.DOTALL)

    if matches:
        # Use the first code block found
        content = matches[0]
    else:
        # No code block found, try to use the whole output
        content = llm_output

    # Clean up the content
    content = content.strip()

    # Try to parse as Python literal
    try:
        result = ast.literal_eval(content)

        # Validate the structure
        if not isinstance(result, list):
            raise ValueError("Output is not a list")

        for item in result:
            if not isinstance(item, tuple):
                raise ValueError(f"Item is not a tuple: {item}")
            if len(item) != 2:
                raise ValueError(f"Tuple does not have exactly 2 elements: {item}")
            if not isinstance(item[0], str) or not isinstance(item[1], str):
                raise ValueError(f"Tuple elements are not strings: {item}")

        return result

    except (SyntaxError, ValueError) as e:
        # Try alternative parsing - look for tuple patterns manually
        tuple_pattern = r'\(\s*["\'](.+?)["\']\s*,\s*["\'](.+?)["\']\s*\)'
        matches = re.findall(tuple_pattern, content, re.DOTALL)

        if matches:
            return [(q.strip(), a.strip()) for q, a in matches]

        raise ValueError(f"Failed to parse freeform QA output: {e}\n\nContent:\n{content}")


def parse_mc_qa_output(llm_output: str) -> list[dict[str, Any]]:
    """
    Parse the LLM output from MC_STEERING_QA_GENERATION_PROMPT.

    Expected format: JSON array of objects
    [
        {
            "question": "...",
            "distractors": ["...", "..."],
            "answer": "...",
            "type": "direct" | "indirect"
        },
        ...
    ]

    Args:
        llm_output: Raw string output from the LLM

    Returns:
        List of dictionaries with keys: question, distractors, answer, type

    Raises:
        ValueError: If parsing fails
    """
    # Try to extract code block first
    code_block_pattern = r"```(?:json)?\s*(.*?)\s*```"
    matches = re.findall(code_block_pattern, llm_output, re.DOTALL)

    if matches:
        # Use the first code block found
        content = matches[0]
    else:
        # No code block found, try to find JSON array in the output
        # Look for content starting with [ and ending with ]
        array_pattern = r"\[\s*\{.*?\}\s*\]"
        matches = re.findall(array_pattern, llm_output, re.DOTALL)
        if matches:
            content = matches[0]
        else:
            content = llm_output

    # Clean up the content
    content = content.strip()

    # Try to parse as JSON
    try:
        result = json.loads(content)

        # Validate the structure
        if not isinstance(result, list):
            raise ValueError("Output is not a list/array")

        for i, item in enumerate(result):
            if not isinstance(item, dict):
                raise ValueError(f"Item {i} is not a dictionary: {item}")

            # Check required keys
            required_keys = {"question", "distractors", "answer", "type"}
            missing_keys = required_keys - set(item.keys())
            if missing_keys:
                raise ValueError(f"Item {i} missing keys: {missing_keys}")

            # Validate types
            if not isinstance(item["question"], str):
                raise ValueError(f"Item {i} 'question' is not a string")
            if not isinstance(item["distractors"], list):
                raise ValueError(f"Item {i} 'distractors' is not a list")
            if not isinstance(item["answer"], str):
                raise ValueError(f"Item {i} 'answer' is not a string")
            if item["type"] not in ["direct", "indirect"]:
                raise ValueError(
                    f"Item {i} 'type' must be 'direct' or 'indirect', got: {item['type']}"
                )

            # Validate distractors are strings
            for j, distractor in enumerate(item["distractors"]):
                if not isinstance(distractor, str):
                    raise ValueError(f"Item {i} distractor {j} is not a string: {distractor}")

        return result

    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse MC QA output as JSON: {e}\n\nContent:\n{content}")


async def generate_mc_steering_qas(
    start_attr: UserAttribute,
    target_attr: UserAttribute,
    generation_config: DataGeneratorConfig,
    use_cache: bool = True,
) -> list[dict[str, str | list[str]]]:
    assert start_attr.__class__ == target_attr.__class__
    attr_class = start_attr.__class__

    while True:
        prompt = MC_STEERING_QA_GENERATION_PROMPT.format(
            attribute_class_name=attr_class.__name__,
            attribute_name=attr_class.attribute_name,
            start_attribute_value_name=str(start_attr),
            target_attribute_value_name=str(target_attr),
            list_of_all_value_names=",\n".join([str(attr) for attr in attr_class]),
        )

        # Query LLM for generation.
        responses, usages = await get_llm_completion_strs_async(
            [[{"role": "user", "content": prompt}]], config=generation_config, use_cache=True
        )
        response = responses[0]
        usages[0]

        try:
            parsed_response = parse_mc_qa_output(response)
        except ValueError:
            continue
        break

    return parsed_response


async def generate_freeform_steering_qas(
    start_attr: UserAttribute,
    target_attr: UserAttribute,
    generation_config: DataGeneratorConfig,
    use_cache: bool = True,
) -> list[tuple[str, str]]:
    assert start_attr.__class__ == target_attr.__class__
    attr_class = start_attr.__class__

    while True:

        prompt = FREEFORM_STEERING_QA_GENERATION_PROMPT.format(
            attribute_class_name=attr_class.__name__,
            attribute_name=attr_class.attribute_name,
            start_attribute_value_name=str(start_attr),
            target_attribute_value_name=str(target_attr),
            list_of_all_value_names=",\n".join([str(attr) for attr in attr_class]),
        )

        # Query LLM for generation.
        responses, usages = await get_llm_completion_strs_async(
            [[{"role": "user", "content": prompt}]], config=generation_config, use_cache=use_cache
        )
        response = responses[0]
        usages[0]

        try:
            parsed_response = parse_freeform_qa_output(response)
        except ValueError:
            continue
        break

    return parsed_response


def create_mc_qa(stem: str, answer: str, distractors: list[str], rng: np.random.Generator):
    label_type = rng.choice(LABEL_TYPES)
    label_delimiter = rng.choice(LABEL_DELIMITERS)
    option_delimiter = rng.choice(OPTION_DELIMITERS)

    mc_options_obj = construct_mc_options_for_train(
        must_include_options=[answer],
        possible_distractors=distractors,
        max_num_distractors=20,
        num_permutations=1,
        label_type=label_type,
        label_delimiter=label_delimiter,
        option_delimiter=option_delimiter,
        # This is because there might already be an equivalent to none-of-the-above option
        # already in the distractors.
        add_none_option=False,
        rng=rng,
    )[0]

    question_template = rng.choice(MC_Q_TEMPLATES)
    question = question_template.format(stem=stem, options=mc_options_obj.options_str)

    answer_template = "{key}"
    key = mc_options_obj.option_to_key[answer]
    answer = answer_template.format(key=key, option=answer)

    return question, answer


async def generate_steering_qas_for_attr_pair(
    save_dir: str,
    start_attr: UserAttribute,
    target_attr: UserAttribute,
    rng: np.random.Generator,
):
    save_dir = Path(save_dir)

    train_save_path = save_dir / f"train_{str(start_attr)}_to_{str(target_attr)}.pkl"
    valid_save_path = save_dir / f"valid_{str(start_attr)}_to_{str(target_attr)}.pkl"
    if train_save_path.exists() and valid_save_path.exists():
        print(f"Skipping {start_attr} to {target_attr} because it already exists")
        return

    print(start_attr, target_attr)

    assert start_attr.__class__ == target_attr.__class__

    gen_config = DataGeneratorConfig(
        # provider="anthropic",
        # model_name="claude-sonnet-4-5-20250929",
        provider="openai",
        model_name="gpt-5.1",
        reasoning_effort="medium",
        max_new_tokens=5000,
    )

    mc_result = await generate_mc_steering_qas(start_attr, target_attr, gen_config)
    ff_result = await generate_freeform_steering_qas(start_attr, target_attr, gen_config)

    mc_qas = []
    for mc_parts in mc_result:
        question, answer = create_mc_qa(
            stem=mc_parts["question"],  # type: ignore
            answer=mc_parts["answer"],  # type: ignore
            distractors=mc_parts["distractors"],  # type: ignore
            rng=rng,
        )
        mc_qas.append((question, answer))

    # Split into train/valid.
    # First, shuffle each list, then select 2 for valid, the rest for train, then combine mc and ff.
    train_qas = []
    valid_qas = []
    for qas in [mc_qas, ff_result]:
        idxs = rng.permutation(len(qas))
        qas = [qas[i] for i in idxs]
        valid_qas.extend(qas[:2])
        train_qas.extend(qas[2:])

    # Save results
    for save_path, qas in [(train_save_path, train_qas), (valid_save_path, valid_qas)]:
        with open(save_path, "wb") as f:
            pickle.dump(qas, f)


async def generate_steering_qas_for_random_attr_pairs(
    save_dir: str, attr_classes: list[Type[UserAttribute]]
):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)

    for attr_class in tqdm(attr_classes):
        print(f"Generating steering QAs for {attr_class}")
        attr_values = [attr for attr in attr_class]
        num_generated = 0
        num_max_iterations = 20
        for _ in range(num_max_iterations):
            if num_generated >= 10:
                break

            start_idx, target_idx = np.random.choice(len(attr_values), size=2, replace=False)
            start_attr = attr_values[start_idx]
            target_attr = attr_values[target_idx]

            await generate_steering_qas_for_attr_pair(save_dir, start_attr, target_attr, rng=rng)

            num_generated += 1
