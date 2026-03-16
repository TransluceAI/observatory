import argparse
import asyncio

from user_modeling.synthsys.subject_inputs import run_subject_input_aggregation


async def main():
    parser = argparse.ArgumentParser(
        description="Aggregate faithfulness eval results into SubjectInput database."
    )
    parser.add_argument("--db_dir", type=str)
    parser.add_argument("--faithfulness_eval_db_dir", type=str)
    parser.add_argument("--baseline_faithfulness_eval_db_dir", type=str)
    parser.add_argument("--batch_size", type=int, default=100)
    args = parser.parse_args()

    await run_subject_input_aggregation(**vars(args))


if __name__ == "__main__":
    asyncio.run(main())
