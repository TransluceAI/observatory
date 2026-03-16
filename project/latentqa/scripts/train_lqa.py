import argparse

from latentqa.train import train
from latentqa.train_util import LQATrainConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, help="Path to config yaml path.")
    # Below are overrides to the config. Use default=None so they only override when explicitly set.
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--micro_batch_size", type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=None)
    parser.add_argument("--num_steps", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--use_wandb", type=eval, choices=[True, False], default=None)
    parser.add_argument("--wandb_group_name", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--eval_interval", type=int, default=None)
    parser.add_argument("--eval_num_steps", type=int, default=None)
    parser.add_argument("--save_interval", type=int, default=None)
    parser.add_argument("--log_interval", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    config = LQATrainConfig.from_yaml(args.config_path)

    # Apply selective overrides - only override if the argument was explicitly provided
    for arg_name, arg_value in vars(args).items():
        if arg_name != "config_path" and arg_value is not None:
            if hasattr(config, arg_name):
                setattr(config, arg_name, arg_value)
            else:
                print(f"Warning: {arg_name} is not a valid config parameter")

    train(config)


if __name__ == "__main__":
    main()
