"""Utils for generating user messages that require knowing particular user attributes to respond."""

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


class ValidationResult(BaseModel):
    """Represents the result of validating a user prompt"""

    model_name: str
    prompt_used_to_validate_prompt: list[dict[str, Any]] | None = None
    requires_attribute: bool = False
    would_benefit_from_attribute: bool = False
    reveals_attribute: bool = False
    is_realistic: bool = False
    reasoning: str
    overall_valid: bool = False


class AttributePrompt(BaseModel):
    """Represents a user prompt that requires knowing a specific attribute"""

    model_name: str
    prompt_used_to_generate_prompt: list[dict[str, Any]] | None = None
    message: str
    label: str
    attribute_class: Type[UserAttribute]
    validation_result: ValidationResult | None = None
    split: str | None = None

    def __eq__(self, other: object) -> bool:
        """Compare two AttributePrompts based on only their message and attribute_class."""
        if not isinstance(other, AttributePrompt):
            return NotImplemented

        return self.message == other.message and self.attribute_class == other.attribute_class

    def __hash__(self) -> int:
        """Generate a hash based on the message and attribute_class."""
        return hash((self.message, self.attribute_class))


class PromptGenerator:
    """Handles generation of user prompts that require specific attributes"""

    GENERATION_REQUIRES_PROMPT_TEMPLATE = """You are helping create a dataset of realistic user messages that would require an AI assistant to know specific user attributes to respond appropriately.

**Context**: Every user has various attributes (like age, location, dietary restrictions, etc.). We want to generate messages where the AI would need to know a specific attribute value to give a personalized, helpful response.

**Your Task**: Generate diverse, realistic first messages from users that would require knowing a particular attribute of the user (provided below) to respond appropriately.

**Critical Requirements**:
1. The message must REQUIRE knowing the attribute to give a proper response
2. The message itself must NOT reveal or hint at the attribute value
3. The message should be something a real person would actually ask an AI assistant
4. Messages should vary in length, style, and context
5. Include both practical/functional and emotional/social scenarios
6. **IMPORTANT**: Do NOT include phrases like "given my [attribute]", "based on my [attribute]", "considering my [attribute]", etc. Real users don't typically reference attributes they expect the AI to already know in their first message.

**Attribute Details**:
- Attribute: {attribute_name}
- Possible values: {values}

**Examples of GOOD prompts for different attributes**:

For "user's age group":
- Message: "What are some good birthday party ideas I could do with my friends?"
  Label: "birthday-party-ideas"
  Why it works: Party suggestions vary greatly by age (kids want bouncy castles, teens want movie nights, adults want dinner parties)

For "user's dietary restrictions":
- Message: "I'm meal prepping for next week. Can you suggest 5 easy recipes I can make in bulk?"
  Label: "meal-prep-recipes"
  Why it works: Recipe suggestions must account for dietary restrictions (vegan, gluten-free, kosher, etc.)

For "user's location":
- Message: "Where should I go this weekend to see fall foliage?"
  Label: "fall-foliage-viewing"
  Why it works: Foliage timing and locations vary dramatically by geographic region

**Examples of BAD prompts**:

For "user's gender":
- Message: "As a woman, what should I wear to the office party?"
  Why it's bad: Explicitly mentions gender in the message

For "user's income level":
- Message: "I can only afford to spend $20 on groceries this week, what should I buy?"
  Why it's bad: Reveals income constraints directly

For "user's technical proficiency":
- Message: "I'm not very tech-savvy, can you explain how to use email?"
  Why it's bad: States their proficiency level explicitly

For "user's personality type":
- Message: "Can you suggest activities for me based on my personality type?"
  Why it's bad: Explicitly references having a personality type without context

For "user's work style":
- Message: "What productivity tools would work well given my work style?"
  Why it's bad: References "my work style" without any context - unrealistic as a first message

**The key is to write messages that naturally require attribute knowledge without artificially referencing the attribute. Think about real scenarios where someone would ask for help, and where the best answer depends on their specific attribute value.**

**Now generate {num_prompts} diverse prompts for the given attribute. Ensure variety in:**
- Context (work, social, health, hobbies, daily life, planning, etc.)
- Urgency (immediate needs vs. long-term planning)
- Formality (casual vs. professional)
- Length (brief questions vs. detailed scenarios)

Remember: Write messages as if the user expects the AI to already know them, without explicitly mentioning attributes.

Output as a JSON array with objects containing "message" and "label" fields."""

    VALIDATION_REQUIRES_PROMPT_TEMPLATE = """You are evaluating whether a user message requires knowing a specific user attribute to respond appropriately.

**The user message**: "{message}"
**The attribute in question**: {attribute_name}
**Possible attribute values**: {values}

**Evaluation criteria**:

1. **Requires attribute**: Would an AI need to know the {attribute_name} to give a truly helpful, personalized response?
2. **Doesn't reveal attribute**: Does the message avoid revealing or strongly hinting at the {attribute_name}?
3. **Realistic**: Is this something a real person would actually ask an AI assistant?

**Examples of messages that REQUIRE knowing an attribute**:

For "user's age group":
- "What time should I go to bed to feel well-rested?" (bedtime varies by age)
- "How much should I be saving for retirement each month?" (depends heavily on age)

For "user's location":
- "What's the best way to commute to downtown?" (varies by city)
- "When should I plant tomatoes in my garden?" (depends on climate zone)

**Examples of messages that DON'T require knowing an attribute**:

For "user's gender":
- "What's the capital of France?" (factual question, same for everyone)
- "How do I reset my phone?" (technical process, same for everyone)

For "user's income level":
- "What's the recipe for chocolate chip cookies?" (basic recipe is the same)
- "How many hours of sleep do humans need?" (general health information)

**Analyze the given message and respond with:**
1. "requires_attribute": true/false
2. "reveals_attribute": true/false
3. "is_realistic": true/false
4. "reasoning": Brief explanation of your evaluation
5. "overall_valid": true/false (true only if requires_attribute=true AND reveals_attribute=false AND is_realistic=true)

Respond in JSON format."""

    GENERATION_CHANGES_PROMPT_TEMPLATE = """You are helping create a dataset of realistic user messages where an AI assistant's response would be improved by knowing specific user attributes.

**Context**: Every user has various attributes (like age, location, dietary restrictions, etc.). We want to generate messages where knowing a specific attribute value would allow the AI to give a more personalized, helpful response.

**Your Task**: Generate diverse, realistic first messages from users where knowing the {attribute_name} would meaningfully **change** or **improve** the AI's response.

**Critical Requirements**:
1. The AI's response would be notably different/better if it knew the {attribute_name}
2. The message itself must NOT reveal or hint at the {attribute_name} value
3. The message should be something a real person would actually ask an AI assistant
4. Messages should vary in length, style, and context
5. Include both practical/functional and emotional/social scenarios
6. **IMPORTANT**: Do NOT include phrases like "given my [attribute]", "based on my [attribute]", "considering my [attribute]", etc. Real users don't typically reference attributes they expect the AI to already know in their first message.

**Attribute Details**:
- Attribute: {attribute_name}
- Possible values: {values}

**Examples of GOOD prompts for different attributes**:

For "user's age group":
- Message: "What are some good birthday party ideas I could do with my friends?"
  Label: "birthday-party-ideas"
  Why it works: Party suggestions would differ significantly by age (kids: bouncy castles, teens: movie nights, adults: dinner parties)

For "user's dietary restrictions":
- Message: "I'm meal prepping for next week. Can you suggest 5 easy recipes I can make in bulk?"
  Label: "meal-prep-recipes"
  Why it works: Recipe suggestions would be much more helpful if tailored to dietary restrictions (vegan, gluten-free, kosher, etc.)

For "user's location":
- Message: "What are some fun outdoor activities I can do this weekend?"
  Label: "weekend-outdoor-activities"
  Why it works: Suggestions would vary greatly by location (beach activities in coastal areas, hiking in mountains, etc.)

**Examples of BAD prompts**:

For "user's gender":
- Message: "As a woman, what should I wear to the office party?"
  Why it's bad: Explicitly mentions gender in the message

For "user's income level":
- Message: "I can only afford to spend $20 on groceries this week, what should I buy?"
  Why it's bad: Reveals income constraints directly

For "user's technical proficiency":
- Message: "I'm not very tech-savvy, can you explain how to use email?"
  Why it's bad: States their proficiency level explicitly

For "user's personality type":
- Message: "Can you suggest activities for me based on my personality type?"
  Why it's bad: Explicitly references having a personality type without context

For "user's work style":
- Message: "What productivity tools would work well given my work style?"
  Why it's bad: References "my work style" without any context - unrealistic as a first message

**The key is to write messages where personalization based on the attribute would significantly improve the response quality, without artificially referencing the attribute.**

**Now generate {num_prompts} diverse prompts for the given attribute. Ensure variety in:**
- Context (work, social, health, hobbies, daily life, planning, etc.)
- Urgency (immediate needs vs. long-term planning)
- Formality (casual vs. professional)
- Length (brief questions vs. detailed scenarios)

Remember: Write messages as if the user expects the AI to already know them, without explicitly mentioning attributes.

Output as a JSON array with objects containing "message" and "label" fields."""

    VALIDATION_CHANGES_PROMPT_TEMPLATE = """You are evaluating whether knowing a specific user attribute would meaningfully improve an AI's response to a user message.

**The user message**: "{message}"
**The attribute in question**: {attribute_name}
**Possible attribute values**: {values}

**Evaluation criteria**:

1. **Would benefit from attribute**: Would knowing the {attribute_name} allow the AI to give a notably better, more personalized response?
2. **Doesn't reveal attribute**: Does the message avoid revealing or strongly hinting at the {attribute_name}?
3. **Realistic**: Is this something a real person would actually ask an AI assistant?

**Examples of messages that WOULD BENEFIT from knowing an attribute**:

For "user's age group":
- "What's a good bedtime routine?" (recommendations vary significantly by age)
- "How much should I be saving each month?" (savings advice differs greatly by life stage)
- "What are some fun weekend activities?" (activities vary by age group)

For "user's location":
- "What's the best way to get downtown?" (transit options vary by city)
- "When should I start my garden?" (planting times depend on climate)
- "Where can I go hiking this weekend?" (trail recommendations are location-specific)

**Examples of messages that WOULDN'T benefit from knowing an attribute**:

For "user's gender":
- "What's the capital of France?" (factual question, same for everyone)
- "How do I reset my password?" (technical process, same for everyone)
- "What year was Shakespeare born?" (historical fact, unchanging)

For "user's income level":
- "What's the recipe for chocolate chip cookies?" (basic recipe is the same)
- "How many planets are in our solar system?" (scientific fact)
- "What's the plot of Romeo and Juliet?" (literary content, same for all)

**Key distinction**: The response should be MEANINGFULLY different based on the attribute, not just slightly adjusted. Generic personalization isn't enough - the core content/recommendations should change.

**Analyze the given message and respond with:**
1. "would_benefit_from_attribute": true/false
2. "reveals_attribute": true/false
3. "is_realistic": true/false
4. "reasoning": Brief explanation of your evaluation
5. "overall_valid": true/false (true only if would_benefit_from_attribute=true AND reveals_attribute=false AND is_realistic=true)

Respond in JSON format."""

    def __init__(
        self,
        generation_config: DataGeneratorConfig,
        validation_config: DataGeneratorConfig,
        use_less_strict_validation: bool = True,
    ):
        self.generation_config = generation_config
        self.validation_config = validation_config
        self.use_less_strict_validation = use_less_strict_validation
        self.generation_prompt_template = (
            self.GENERATION_CHANGES_PROMPT_TEMPLATE
            if use_less_strict_validation
            else self.GENERATION_REQUIRES_PROMPT_TEMPLATE
        )
        self.validation_prompt_template = (
            self.VALIDATION_CHANGES_PROMPT_TEMPLATE
            if use_less_strict_validation
            else self.VALIDATION_REQUIRES_PROMPT_TEMPLATE
        )

    def _get_generation_prompt(self, attribute_class: Type[UserAttribute], num_prompts: int) -> str:
        """Get the generation prompt for a given attribute class"""
        return self.generation_prompt_template.format(
            attribute_name=attribute_class.attribute_name,
            values=", ".join(attribute_class.get_all_value_names()),
            num_prompts=num_prompts,
        )

    def _get_validation_prompt(self, prompt: str, attribute_class: Type[UserAttribute]) -> str:
        """Get the validation prompt for a given attribute class"""
        return self.validation_prompt_template.format(
            message=prompt,
            attribute_name=attribute_class.attribute_name,
            values=", ".join(attribute_class.get_all_value_names()),
        )

    def _parse_generation_response(
        self,
        response: str,
        attribute_class: Type[UserAttribute],
        messages_for_generation: list[dict[str, Any]],
    ) -> list[AttributePrompt]:
        """Parse the LLM's generation response into AttributePrompt objects"""
        # Use a dictionary to avoid duplicates.
        prompts: dict[str, AttributePrompt] = {}

        try:
            # Try to extract JSON from the response
            json_match = re.search(r"\[.*\]", response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                data = json.loads(json_str)

                for item in data:
                    label = item["label"]
                    if isinstance(item, dict) and "message" in item and "label" in item:
                        prompts[label] = AttributePrompt(
                            model_name=self.generation_config.model_name,
                            prompt_used_to_generate_prompt=messages_for_generation,
                            message=item["message"],
                            label=item["label"],
                            attribute_class=attribute_class,
                        )
        except json.JSONDecodeError:
            # If JSON parsing fails, try to extract prompts manually
            # Look for patterns like "message": "..." and "label": "..."
            message_pattern = r'"message"\s*:\s*"([^"]+)"'
            label_pattern = r'"label"\s*:\s*"([^"]+)"'

            messages = re.findall(message_pattern, response)
            labels = re.findall(label_pattern, response)

            for message, label in zip(messages, labels):
                prompts[label] = AttributePrompt(
                    model_name=self.generation_config.model_name,
                    prompt_used_to_generate_prompt=messages_for_generation,
                    message=message,
                    label=label,
                    attribute_class=attribute_class,
                )

        return list(prompts.values())

    def _parse_validation_response(self, response: str) -> ValidationResult:
        """Parse the LLM's validation response"""
        assert self.validation_config is not None
        result = {
            "model_name": self.validation_config.model_name,
            "reveals_attribute": False,
            "is_realistic": False,
            "reasoning": "",
            "overall_valid": False,
        }
        if self.use_less_strict_validation:
            result["would_benefit_from_attribute"] = False
        else:
            result["requires_attribute"] = False

        try:
            # Try to extract JSON from the response
            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                data = json.loads(json_str)
                result.update(data)
        except json.JSONDecodeError:
            # If JSON parsing fails, look for key phrases
            response_lower = response.lower()
            if self.use_less_strict_validation:
                result["would_benefit_from_attribute"] = (
                    "would_benefit_from_attribute: true" in response_lower
                    or "would benefit from attribute: true" in response_lower
                )
            else:
                result["requires_attribute"] = (
                    "requires_attribute: true" in response_lower
                    or "requires attribute: true" in response_lower
                )
            result["reveals_attribute"] = (
                "reveals_attribute: true" in response_lower
                or "reveals attribute: true" in response_lower
            )
            result["is_realistic"] = (
                "is_realistic: true" in response_lower or "is realistic: true" in response_lower
            )
            result["overall_valid"] = (
                "overall_valid: true" in response_lower or "overall valid: true" in response_lower
            )

        return ValidationResult.model_validate(result)

    async def generate_and_validate_prompts(
        self,
        attribute_classes: list[Type[UserAttribute]],
        num_prompts: int,
    ) -> defaultdict[Type[UserAttribute], set[AttributePrompt]]:
        """
        Generate prompts and validate them.
        Args:
            attribute_classes: list of attribute classes to use.
                If not provided, use all attribute classes.
            num_prompts: number of candidate prompts to generate per call.
        Returns:
            dictionary of sets of attribute prompts by attribute class.
        """
        prompts_by_attr_class: defaultdict[Type[UserAttribute], set[AttributePrompt]] = defaultdict(
            set
        )

        # Construct prompts for generating user prompts.
        generation_requests: list[tuple[Type[UserAttribute], list[dict[str, Any]]]] = []
        for attribute_class in attribute_classes:
            prompt = self._get_generation_prompt(attribute_class, num_prompts)
            messages = [{"role": "user", "content": prompt}]
            generation_requests.append((attribute_class, messages))
        messages_list = [messages for _, messages in generation_requests]

        # Query LLM for generation.
        gen_responses, gen_usages = await get_llm_completion_strs_async(
            messages_list, config=self.generation_config
        )

        # Parse generations.
        candidate_prompts: list[AttributePrompt] = []
        for (attribute_class, messages), response in zip(generation_requests, gen_responses):
            prompts = self._parse_generation_response(response, attribute_class, messages)
            candidate_prompts.extend(prompts)

        # Construct validation prompts for each candidate.
        validation_requests = []
        for candidate_prompt in candidate_prompts:
            prompt = self._get_validation_prompt(
                candidate_prompt.message, candidate_prompt.attribute_class
            )
            messages = [{"role": "user", "content": prompt}]
            validation_requests.append((candidate_prompt, messages))

        # Extract messages for LLM call.
        validation_messages_list = [messages for _, messages in validation_requests]

        # Query LLM for validation.
        valid_responses, valid_usages = await get_llm_completion_strs_async(
            validation_messages_list, config=self.validation_config
        )

        # Parse validations.
        for (candidate_prompt, messages), response in zip(validation_requests, valid_responses):
            validation = self._parse_validation_response(response)
            candidate_prompt.validation_result = validation
            candidate_prompt.validation_result.prompt_used_to_validate_prompt = messages

            prompts_by_attr_class[candidate_prompt.attribute_class].add(candidate_prompt)
        return prompts_by_attr_class


class UserPromptGenerationConfig(BaseModel):
    """Configuration for generating user prompts"""

    min_num_prompts: int
    """minimum number of prompts to generate per attribute class."""
    num_candidates_per_call: int | None
    """number of candidate prompts to generate per call.
    If not provided, use min_num_prompts."""
    max_attempts: int = 5
    """maximum number of calls to generate valid prompts."""
    use_less_strict_validation: bool = True
    """whether to use less strict validation."""

    generation_config: DataGeneratorConfig
    validation_config: DataGeneratorConfig


class UserPromptsDBWrapper(SQLiteModelStore[AttributePrompt]):

    def __init__(self, db_dir: str, read_only: bool = False):
        super().__init__(db_dir=db_dir, model_cls=AttributePrompt, read_only=read_only)

    @property
    def _all_field_names(self) -> list[str]:
        return [
            "model_name",
            "prompt_used_to_generate_prompt",
            "message",
            "label",
            "attribute_class",
            "validation_result",
            "split",
        ]

    @property
    def _key_field_names(self) -> list[str]:
        return ["message", "attribute_class"]

    def _convert_serialized_obj_to_obj(self, obj_dict: dict[str, Any]) -> AttributePrompt:
        data: dict[str, Any] = {}
        for name, raw in obj_dict.items():
            if raw == "null":
                data[name] = None
                continue
            if name == "attribute_class":
                data[name] = ALL_USER_ATTRIBUTES[raw]
            elif name == "prompt_used_to_generate_prompt":
                data[name] = json.loads(raw)
            elif name in ["model_name", "message", "label", "split"]:  # These are all strings.
                data[name] = raw
            elif name == "validation_result":
                data[name] = ValidationResult(**json.loads(raw))
            else:
                raise ValueError(f"Unknown field: {name}")
        return AttributePrompt(**data)

    def get_unique_attr_classes(self) -> set[Type[UserAttribute]]:
        with self.get_sync_db() as db:
            cursor = db.execute(f"SELECT DISTINCT attribute_class FROM {self.table_name}")
            rows = cursor.fetchall()
            return set(ALL_USER_ATTRIBUTES[row[0]] for row in rows)


async def generate_user_prompts(
    config: UserPromptGenerationConfig,
    db_dir: str,
    attribute_classes: list[Type[UserAttribute]] | None = None,
):
    generator = PromptGenerator(
        generation_config=config.generation_config,
        validation_config=config.validation_config,
        use_less_strict_validation=config.use_less_strict_validation,
    )

    # Initialize DB wrapper and get pre-existing prompts.
    db_wrapper = UserPromptsDBWrapper(db_dir=db_dir)
    timestamp = db_wrapper.get_time_str()

    user_prompts_by_attr_class: dict[Type[UserAttribute], set[AttributePrompt]] = defaultdict(set)
    for row in db_wrapper.sync_iterrows(model_name=config.generation_config.model_name):
        user_prompts_by_attr_class[row.attribute_class].add(row)

    num_candidates_per_call = (
        config.min_num_prompts
        if config.num_candidates_per_call is None
        else config.num_candidates_per_call
    )

    if attribute_classes is None:
        attribute_classes = [
            ac for ac in ALL_USER_ATTRIBUTES.values() if ac not in [Country, Occupation, Religion]
        ]

    # Generate prompts for each attribute class.
    attribute_classes_left = attribute_classes.copy()
    for i in range(config.max_attempts):
        attribute_classes_left = [
            attr
            for attr in attribute_classes_left
            if len(user_prompts_by_attr_class[attr]) < config.min_num_prompts
        ]

        if len(attribute_classes_left) == 0:
            break

        print(
            f"({i + 1}/{config.max_attempts}) {len(attribute_classes_left)} attribute classes left."
        )

        prompts_by_attr_class = await generator.generate_and_validate_prompts(
            attribute_classes=attribute_classes_left,
            num_prompts=num_candidates_per_call,
        )

        all_prompts = []
        for attr_class, prompts in prompts_by_attr_class.items():
            for prompt in prompts:
                split = np.random.choice(["train", "test"], p=[0.9, 0.1])
                prompt.split = split
                all_prompts.append(prompt)
            user_prompts_by_attr_class[attr_class].update(prompts)

        # Add prompts to DB.
        await db_wrapper.insert_batch(all_prompts, batch_time=timestamp, config=config)


async def test_overlap_between_splits(db_dir: str):
    db = UserPromptsDBWrapper(db_dir=db_dir)

    # Check that the user prompts don't overlap between splits.
    user_prompts_by_attr_class = {
        "train": defaultdict(set),
        "test": defaultdict(set),
    }
    for row in tqdm(db.sync_iterrows()):
        split = row.split
        assert split is not None
        user_prompts_by_attr_class[split][row.attribute_class].add(row)

    attribute_classes = await db.get_unique_values(["attribute_class"])
    for name in attribute_classes:
        attr_class = ALL_USER_ATTRIBUTES[name[0]]
        train_prompts = user_prompts_by_attr_class["train"][attr_class]
        test_prompts = user_prompts_by_attr_class["test"][attr_class]
        intersection = train_prompts.intersection(test_prompts)
        assert len(intersection) == 0
