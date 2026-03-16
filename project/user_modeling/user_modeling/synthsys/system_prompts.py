"""Utils for generating system prompts that states the user's attribute as background knowledge."""

import json
import re
from collections import defaultdict
from typing import Any, Type

import numpy as np
from pydantic import BaseModel
from tqdm import tqdm
from user_modeling.db_wrapper import SQLiteModelStore
from user_modeling.llm_util import DataGeneratorConfig, get_llm_completion_strs_async
from user_modeling.user_attribute import (
    ALL_USER_ATTRIBUTES,
    Country,
    Occupation,
    Religion,
    UserAttribute,
)


class SystemPrompt(BaseModel):
    """Represents a system prompt for a specific attribute value"""

    model_name: str
    prompt_used_to_generate_prompt: list[dict[str, Any]] | None = None
    message: str
    attribute_class: Type[UserAttribute]
    attribute_value: UserAttribute
    split: str | None = None

    def __eq__(self, other: object) -> bool:
        """Compare two SystemPrompts based on their message, attribute_class, and attribute_value."""
        if not isinstance(other, SystemPrompt):
            return NotImplemented

        return (
            self.message == other.message
            and self.attribute_class == other.attribute_class
            and self.attribute_value == other.attribute_value
        )

    def __hash__(self) -> int:
        """Generate a hash based on the message, attribute_class, and attribute_value."""
        return hash((self.message, self.attribute_class, self.attribute_value))


class SystemPromptGenerator:
    """Handles generation of system prompts for user attributes"""

    GENERATION_PROMPT_TEMPLATE = """You are an expert at crafting system prompts that subtly embed user attributes into an AI assistant's worldview, making it naturally adapt its responses without ever explicitly mentioning this knowledge.

**Your Task**: Create diverse system prompts that establish a user attribute as an implicit, background fact that naturally influences the assistant's responses - like how a friend who knows you well would adjust their advice and conversation style without stating why.

**Core Requirements**:
* Given the system prompt, the assistant must internalize the attribute as true background knowledge.
* The internalization of the attribute *should* change the assistant's behavior that is consistent with the attribute, but it is important that the system prompt itself NEVER explicitly suggests a particular behavior.
    - For example, if the attribute is "user's age group: child", the system prompt must not contain cues like "use simpler words" or "use more emojis", etc.
*The system prompt should be designed such that the assistant NEVER explicitly mentions knowing this attribute or use phrases like "since you are...", "as a...", "given that you're..." in its responses to the user.
* Keep prompts concise but immersive

**Attribute to establish**:
- {attribute_name}: {value_name}

**Generate {num_prompts} distinct variations that each take a different approach to embedding this attribute. Ensure variety in both framing devices and implementation strategies.**

Output as a JSON array with objects containing a "message" field for each system prompt."""

    def __init__(self, generation_config: DataGeneratorConfig):
        self.generation_config = generation_config

    def _get_generation_prompt(self, attribute: UserAttribute, num_prompts: int) -> str:
        """Get the generation prompt for a given attribute class"""
        return self.GENERATION_PROMPT_TEMPLATE.format(
            attribute_name=attribute.attribute_name,
            value_name=attribute.value_name,
            num_prompts=num_prompts,
        )

    def _parse_generation_response(
        self, response: str, attribute: UserAttribute, messages_for_generation: list[dict[str, Any]]
    ) -> list[SystemPrompt]:
        """Parse the LLM's generation response into SystemPrompt objects"""
        prompts: list[SystemPrompt] = []
        try:
            # Try to extract JSON from the response
            json_match = re.search(r"\[.*\]", response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                data = json.loads(json_str)

                for item in data:
                    if isinstance(item, dict) and "message" in item:
                        prompts.append(
                            SystemPrompt(
                                model_name=self.generation_config.model_name,
                                prompt_used_to_generate_prompt=messages_for_generation,
                                message=item["message"],
                                attribute_class=attribute.__class__,
                                attribute_value=attribute,
                            )
                        )
        except json.JSONDecodeError:
            # If JSON parsing fails, try to extract prompts manually
            # Look for the pattern "message": "..."
            message_pattern = r'"message"\s*:\s*"([^"]+)"'
            messages = re.findall(message_pattern, response)

            for message in messages:
                prompts.append(
                    SystemPrompt(
                        model_name=self.generation_config.model_name,
                        prompt_used_to_generate_prompt=messages_for_generation,
                        message=message,
                        attribute_class=attribute.__class__,
                        attribute_value=attribute,
                    )
                )

        return prompts

    async def generate_prompts(
        self,
        attributes: list[UserAttribute],
        num_prompts: int,
    ) -> defaultdict[UserAttribute, set[SystemPrompt]]:
        """Generate system prompts for all values of an attribute class"""

        prompts_by_attr: defaultdict[UserAttribute, set[SystemPrompt]] = defaultdict(set)
        # Construct prompts for generating system prompts.
        generation_requests: list[tuple[UserAttribute, list[dict[str, Any]]]] = []
        for attribute in attributes:
            prompt = self._get_generation_prompt(attribute, num_prompts)
            messages = [{"role": "user", "content": prompt}]
            generation_requests.append((attribute, messages))
        messages_list = [messages for _, messages in generation_requests]

        # Query LLM for generation.
        responses, usages = await get_llm_completion_strs_async(
            messages_list, config=self.generation_config
        )

        # Parse generation responses to extract the system prompts.
        for (attribute, messages), response in zip(generation_requests, responses):
            prompts = self._parse_generation_response(response, attribute, messages)
            prompts_by_attr[attribute].update(prompts)
        return prompts_by_attr


class SystemPromptsDBWrapper(SQLiteModelStore[SystemPrompt]):

    def __init__(self, db_dir: str, read_only: bool = False):
        super().__init__(db_dir=db_dir, model_cls=SystemPrompt, read_only=read_only)

    @property
    def _all_field_names(self) -> list[str]:
        return [
            "model_name",
            "prompt_used_to_generate_prompt",
            "message",
            "attribute_class",
            "attribute_value",
            "split",
        ]

    @property
    def _key_field_names(self) -> list[str]:
        return ["message", "attribute_class", "attribute_value"]

    def _convert_serialized_obj_to_obj(self, obj_dict: dict[str, Any]) -> SystemPrompt:
        attr_class_name = obj_dict["attribute_class"]
        attr_class = ALL_USER_ATTRIBUTES[attr_class_name]

        data: dict[str, Any] = {}
        for name, raw in obj_dict.items():
            if raw == "null":
                data[name] = None
                continue
            if name == "attribute_class":
                data[name] = attr_class
            elif name == "attribute_value":
                data[name] = attr_class(raw)
            elif name == "prompt_used_to_generate_prompt":
                data[name] = json.loads(raw)
            elif name in ["model_name", "message", "split"]:  # These are all strings.
                data[name] = raw
            else:
                raise ValueError(f"Unknown field: {name}")
        return SystemPrompt(**data)

    def get_unique_attributes_by_attr_class(
        self,
    ) -> dict[Type[UserAttribute], set[UserAttribute]]:
        with self.get_sync_db() as db:
            cursor = db.execute(
                f"SELECT DISTINCT attribute_class, attribute_value FROM {self.table_name}"
            )
            rows = cursor.fetchall()
        attr_by_attr_class: defaultdict[Type[UserAttribute], set[UserAttribute]] = defaultdict(set)
        for attr_class_name, attr_value in rows:
            attr_class = ALL_USER_ATTRIBUTES[attr_class_name]
            attr = attr_class(attr_value)
            attr_by_attr_class[attr_class].add(attr)
        return attr_by_attr_class


class SystemPromptGenerationConfig(BaseModel):
    """Configuration for generating system prompts"""

    min_num_prompts: int
    num_candidates_per_call: int | None
    max_attempts: int = 5

    generation_config: DataGeneratorConfig


async def generate_system_prompts(
    config: SystemPromptGenerationConfig,
    db_dir: str,
    attributes: list[UserAttribute] | None = None,
):
    generator = SystemPromptGenerator(generation_config=config.generation_config)

    # Initialize DB wrapper and get pre-existing prompts.
    db_wrapper = SystemPromptsDBWrapper(db_dir=db_dir)
    timestamp = db_wrapper.get_time_str()

    sys_prompts_by_attr_class: dict[UserAttribute, set[SystemPrompt]] = defaultdict(set)
    for row in db_wrapper.sync_iterrows(model_name=config.generation_config.model_name):
        sys_prompts_by_attr_class[row.attribute_value].add(row)

    num_candidates_per_call = (
        config.min_num_prompts
        if config.num_candidates_per_call is None
        else config.num_candidates_per_call
    )

    if attributes is None:
        attribute_classes = [
            ac for ac in ALL_USER_ATTRIBUTES.values() if ac not in [Country, Occupation, Religion]
        ]
        attributes = [attr for attr_class in attribute_classes for attr in attr_class]

    # Generate prompts for each attribute within attribute classes.
    attributes_left = attributes.copy()
    for i in range(config.max_attempts):
        attributes_left = [
            attr
            for attr in attributes_left
            if len(sys_prompts_by_attr_class[attr]) < config.min_num_prompts
        ]

        if len(attributes_left) == 0:
            break

        print(f"({i + 1}/{config.max_attempts}) {len(attributes_left)} attributes left.")

        prompts_by_attr = await generator.generate_prompts(
            attributes=attributes_left, num_prompts=num_candidates_per_call
        )
        all_prompts = []
        for attr, prompts in prompts_by_attr.items():
            for prompt in prompts:
                split = np.random.choice(["train", "test"], p=[0.9, 0.1])
                prompt.split = split
                all_prompts.append(prompt)
            sys_prompts_by_attr_class[attr].update(prompts)

        # Add prompts to DB.
        await db_wrapper.insert_batch(all_prompts, batch_time=timestamp, config=config)


async def test_overlap_between_splits(db_dir: str):
    db = SystemPromptsDBWrapper(db_dir=db_dir)

    # Check that the system prompts don't overlap between splits.
    attribute_values = set()
    system_prompts_by_attr = {"train": defaultdict(set), "test": defaultdict(set)}
    for row in tqdm(db.sync_iterrows()):
        split = row.split
        assert split in ["train", "test"]
        system_prompts_by_attr[split][row.attribute_value].add(row.message)
        attribute_values.add(row.attribute_value)

    for value in attribute_values:
        train_prompts = system_prompts_by_attr["train"][value]
        test_prompts = system_prompts_by_attr["test"][value]
        intersection = train_prompts.intersection(test_prompts)
        assert len(intersection) == 0
