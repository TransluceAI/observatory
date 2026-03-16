import argparse
import asyncio

from user_modeling.llm_util import DataGeneratorConfig
from user_modeling.synthsys.system_prompts import (
    SystemPromptGenerationConfig,
    generate_system_prompts,
)
from user_modeling.user_attribute import ALL_USER_ATTRIBUTES


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db_dir", type=str, required=True)
    parser.add_argument("--attribute_class_names", type=str, nargs="+", default=None)

    parser.add_argument("--min_num_prompts", type=int, required=True)
    parser.add_argument("--num_candidates_per_call", type=int, default=None)
    parser.add_argument("--max_attempts", type=int, default=5)

    parser.add_argument("--provider", type=str, default="openai")
    parser.add_argument("--model_name", type=str, default="o3")
    parser.add_argument(
        "--reasoning_effort",
        type=str,
        default="medium",
        choices=["low", "medium", "high"],
    )
    parser.add_argument("--max_new_tokens", type=int, default=10_000)

    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max_concurrency", type=int, default=100)
    parser.add_argument("--timeout", type=float, default=360.0)
    args = parser.parse_args()

    config = SystemPromptGenerationConfig(
        min_num_prompts=args.min_num_prompts,
        num_candidates_per_call=args.num_candidates_per_call,
        max_attempts=args.max_attempts,
        generation_config=DataGeneratorConfig(
            provider=args.provider,
            model_name=args.model_name,
            reasoning_effort=args.reasoning_effort,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            max_concurrency=args.max_concurrency,
            timeout=args.timeout,
        ),
    )
    if args.attribute_class_names is not None:
        attribute_classes = [
            attr
            for attr in ALL_USER_ATTRIBUTES.values()
            if attr.__name__ in args.attribute_class_names
        ]
        attributes = [attr for attr_class in attribute_classes for attr in attr_class]
    else:
        attributes = None

    await generate_system_prompts(config, db_dir=args.db_dir, attributes=attributes)


if __name__ == "__main__":
    asyncio.run(main())
