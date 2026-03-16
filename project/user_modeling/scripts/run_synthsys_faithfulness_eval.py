import argparse
import logging

import numpy as np
from dotenv import load_dotenv
from openai import AsyncOpenAI
from user_modeling.synthsys.faithfulness_eval import (
    LLMBasedFaithfulnessEvalConfig,
    LLMBasedFaithfulnessEvalDBConfig,
    run_faithfulness_evaluation,
)


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_model_id", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--system_prompt_db_dir", type=str, default=None)
    parser.add_argument("--user_prompt_db_dir", type=str, required=True)
    parser.add_argument(
        "--user_prompt_type",
        type=str,
        required=True,
        choices=["faith_result", "user_prompt"],
        help=(
            """Whether the db is LLMBasedFaithfulnessEvalDBWrapper or UserPromptsDBWrapper.
When evaluating the baseline belief, we can either get user prompts from
- UserPromptsDBWrapper if we want to serially iterate through all user prompts, or
- LLMBasedFaithfulnessEvalDBWrapper if we want to evaluate on user prompts that have already been
evaluated previously."""
        ),
    )
    parser.add_argument("--subject_gen_client_url", type=str)
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        choices=["train", "test", None],
    )
    parser.add_argument("--num_subject_completions", type=int, default=1)
    parser.add_argument(
        "--num_examples",
        type=int,
        default=100_000,
    )
    parser.add_argument("--system_prompt_timestamps", type=str, nargs="+", default=None)
    parser.add_argument("--user_prompt_timestamps", type=str, nargs="+", default=None)
    parser.add_argument("--num_segments", type=int, default=1)
    parser.add_argument("--segment_idx", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--generation_timeout", type=float, default=20.0)
    parser.add_argument("--llm_judge_timeout", type=float, default=10.0)
    parser.add_argument("--max_retries", type=int, default=10)

    parser.add_argument("--db_dir", type=str, required=True)
    parser.add_argument("--eval_baseline", type=eval, default=False, choices=[True, False])
    args = parser.parse_args()

    load_dotenv()
    logging.getLogger("llm_util.openai").setLevel(logging.ERROR)
    logging.getLogger("llm_util.prod_llms").setLevel(logging.WARNING)

    faithfulness_db_config = LLMBasedFaithfulnessEvalDBConfig(
        subject_hf_model_id=args.hf_model_id,
        num_permutations=10,
        generation_temperature=1.0,
        generation_max_new_tokens=256,
        llm_judge_provider="openai",
        llm_judge_model_name="gpt-4.1-mini",
    )
    subject_generation_client = AsyncOpenAI(base_url=args.subject_gen_client_url, api_key="dummy")

    eval_config = LLMBasedFaithfulnessEvalConfig(
        hf_model_id=args.hf_model_id,
        system_prompt_db_dir=args.system_prompt_db_dir,
        user_prompt_db_dir=args.user_prompt_db_dir,
        user_prompt_type=args.user_prompt_type,
        num_subject_completions=args.num_subject_completions,
        num_examples=args.num_examples,
        split=args.split,
        system_prompt_timestamps=args.system_prompt_timestamps,
        user_prompt_timestamps=args.user_prompt_timestamps,
        num_segments=args.num_segments,
        segment_idx=args.segment_idx,
        seed=np.random.randint(1000000),
    )

    await run_faithfulness_evaluation(
        db_dir=args.db_dir,
        faithfulness_db_config=faithfulness_db_config,
        subject_generation_client=subject_generation_client,
        config=eval_config,
        eval_baseline=args.eval_baseline,
        batch_size=args.batch_size,
        generation_timeout=args.generation_timeout,
        llm_judge_timeout=args.llm_judge_timeout,
    )


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
