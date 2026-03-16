import argparse
import logging

import yaml
from user_modeling.belief_eval import BeliefEvalConfig, run_belief_evaluation
from user_modeling.user_attribute import ALL_USER_ATTRIBUTES


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path",
        type=str,
        help="Path to yaml file containing args.",
        default=None,
    )
    parser.add_argument(
        "--subject_hf_model_id", type=str, default="meta-llama/Llama-3.1-8B-Instruct"
    )
    parser.add_argument("--eval_input_path", type=str, required=True)

    parser.add_argument("--eval_type", type=str, required=True)
    parser.add_argument(
        "--add_to_last_user_message", type=eval, default=False, choices=[True, False]
    )

    parser.add_argument("--user_prompts_and_seed_responses_file", type=str, default=None)

    parser.add_argument("--generation_temperature", type=float, default=1.0)
    parser.add_argument("--generation_max_new_tokens", type=int, default=256)
    parser.add_argument("--generation_num_completions", type=int, default=10)
    parser.add_argument("--generation_timeout", type=float, default=100.0)

    parser.add_argument("--judge_model_name", type=str, default="gpt-4.1-mini")
    parser.add_argument("--judge_num_permutations", type=int, default=10)
    parser.add_argument("--judge_max_concurrency", type=int, default=100)
    parser.add_argument("--judge_timeout", type=float, default=100.0)

    parser.add_argument("--forward_batch_size", type=int, default=64)

    parser.add_argument("--db_dir", type=str, required=True)
    parser.add_argument("--attr_class", type=str, default=None)
    parser.add_argument("--subject_client_url", type=str, default=None)
    parser.add_argument("--skip_existing", type=eval, default=True, choices=[True, False])
    parser.add_argument("--eval_inputs_batch_size", type=int, default=32)
    args = parser.parse_args()

    logging.getLogger("llm_util.openai").setLevel(logging.ERROR)
    logging.getLogger("llm_util.prod_llms").setLevel(logging.WARNING)

    if args.config_path is not None:
        with open(args.config_path, "r") as f:
            loaded_args = yaml.load(f, Loader=yaml.FullLoader)
        config = BeliefEvalConfig(**loaded_args)
        args.db_dir = loaded_args.get("db_dir", args.db_dir)
        args.attr_class = loaded_args.get("attr_class", args.attr_class)
        args.subject_client_url = loaded_args.get("subject_client_url", args.subject_client_url)
        args.skip_existing = loaded_args.get("skip_existing", args.skip_existing)
        args.eval_inputs_batch_size = loaded_args.get(
            "eval_inputs_batch_size", args.eval_inputs_batch_size
        )
    else:
        config = BeliefEvalConfig(**vars(args))

    await run_belief_evaluation(
        config=config,
        db_dir=args.db_dir,
        attr_class=(ALL_USER_ATTRIBUTES[args.attr_class] if args.attr_class is not None else None),
        subject_client_url=args.subject_client_url,
        skip_existing=args.skip_existing,
        eval_inputs_batch_size=args.eval_inputs_batch_size,
    )


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
