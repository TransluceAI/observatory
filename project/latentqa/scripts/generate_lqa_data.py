import argparse
import asyncio
import logging

from latentqa.lqa_data import (
    FullLQADataGenerationConfig,
    generate_full_lqa_data_from_subject_inputs,
)
from user_modeling.llm_util import DataGeneratorConfig


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject_input_db_dir", type=str, default=None)
    parser.add_argument("--hf_dataset_id", type=str, default=None)
    parser.add_argument(
        "--question_type", type=str, default="general", choices=["general", "multiple_choice"]
    )
    parser.add_argument("--format_probs", type=float, nargs="+", default=[0.6, 0.25, 0.15])
    parser.add_argument("--full_data_db_dir", type=str, required=True)
    parser.add_argument("--max_num_insert", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=128)
    args = parser.parse_args()

    if args.subject_input_db_dir is None and args.hf_dataset_id is None:
        raise ValueError("Either --subject_input_db_dir or --hf_dataset_id must be provided.")

    logging.getLogger("llm_util.anthropic").setLevel(logging.WARNING)

    o3_gen_config = DataGeneratorConfig(
        provider="openai",
        model_name="o3",
        reasoning_effort="medium",
        max_new_tokens=10_000,
        temperature=1.0,
        max_concurrency=100,
        timeout=360.0,
    )
    c4sonnet_gen_config = DataGeneratorConfig(
        provider="anthropic",
        model_name="claude-sonnet-4-20250514",
        reasoning_effort="medium",
        max_new_tokens=5_000,
        temperature=1.0,
        max_concurrency=50,
        timeout=360.0,
    )

    config = FullLQADataGenerationConfig(
        hf_dataset_id=args.hf_dataset_id,
        subject_input_db_dir=args.subject_input_db_dir,
        qa_generation_configs=[o3_gen_config, c4sonnet_gen_config],
        question_type=args.question_type,
        format_probs=tuple(args.format_probs),
    )

    await generate_full_lqa_data_from_subject_inputs(
        config=config,
        full_data_db_dir=args.full_data_db_dir,
        max_num_insert=args.max_num_insert,
        batch_size=args.batch_size,
        seed=args.seed,
    )


if __name__ == "__main__":
    asyncio.run(main())
