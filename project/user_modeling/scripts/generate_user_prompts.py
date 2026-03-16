import argparse
import asyncio

from user_modeling.llm_util import DataGeneratorConfig
from user_modeling.synthsys.user_prompts import UserPromptGenerationConfig, generate_user_prompts
from user_modeling.user_attribute import ALL_USER_ATTRIBUTES


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db_dir", type=str, required=True)
    parser.add_argument("--attribute_class_names", type=str, nargs="+", default=None)

    parser.add_argument("--min_num_prompts", type=int, required=True)
    parser.add_argument("--num_candidates_per_call", type=int, default=None)
    parser.add_argument("--max_attempts", type=int, default=5)
    parser.add_argument(
        "--use_less_strict_validation", type=eval, choices=[True, False], default=True
    )

    parser.add_argument("--generation_provider", type=str, default="openai")
    parser.add_argument("--generation_model_name", type=str, default="o3")
    parser.add_argument(
        "--generation_reasoning_effort",
        type=str,
        default="medium",
        choices=["low", "medium", "high"],
    )
    parser.add_argument("--generation_max_new_tokens", type=int, default=10_000)

    parser.add_argument("--validation_provider", type=str, default="openai")
    parser.add_argument("--validation_model_name", type=str, default="o4-mini")
    parser.add_argument(
        "--validation_reasoning_effort",
        type=str,
        default="low",
        choices=["low", "medium", "high"],
    )
    parser.add_argument("--validation_max_new_tokens", type=int, default=1000)

    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max_concurrency", type=int, default=100)
    parser.add_argument("--timeout", type=float, default=360.0)
    args = parser.parse_args()

    config = UserPromptGenerationConfig(
        min_num_prompts=args.min_num_prompts,
        num_candidates_per_call=args.num_candidates_per_call,
        max_attempts=args.max_attempts,
        use_less_strict_validation=args.use_less_strict_validation,
        generation_config=DataGeneratorConfig(
            provider=args.generation_provider,
            model_name=args.generation_model_name,
            reasoning_effort=args.generation_reasoning_effort,
            max_new_tokens=args.generation_max_new_tokens,
            temperature=args.temperature,
            max_concurrency=args.max_concurrency,
            timeout=args.timeout,
        ),
        validation_config=DataGeneratorConfig(
            provider=args.validation_provider,
            model_name=args.validation_model_name,
            reasoning_effort=args.validation_reasoning_effort,
            max_new_tokens=args.validation_max_new_tokens,
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
    else:
        attribute_classes = None

    await generate_user_prompts(config, db_dir=args.db_dir, attribute_classes=attribute_classes)


if __name__ == "__main__":
    asyncio.run(main())
