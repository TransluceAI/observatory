import argparse

import yaml
from latentqa.eval.reading_eval import ReadingEvalConfig, run_reading_eval


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, help="Path to config yaml path.")
    parser.add_argument(
        "--ckpt_names", type=str, nargs="+", help="Names of checkpoints to evaluate.", default=None
    )
    parser.add_argument("--batch_size", type=int, help="Batch size to evaluate.", default=None)
    parser.add_argument(
        "--num_batches", type=int, help="Number of batches to evaluate.", default=None
    )
    parser.add_argument(
        "--progress_bar", type=eval, default=True, choices=[True, False], help="Show progress bar."
    )
    args = parser.parse_args()

    with open(args.config_path, "r") as f:
        config_dict = yaml.load(f, Loader=yaml.FullLoader)
    config = ReadingEvalConfig(**config_dict)

    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.num_batches is not None:
        config.num_batches = args.num_batches

    if args.ckpt_names is not None:
        config.ckpt_names = args.ckpt_names

    run_reading_eval(config, progress_bar=args.progress_bar)


if __name__ == "__main__":
    main()
