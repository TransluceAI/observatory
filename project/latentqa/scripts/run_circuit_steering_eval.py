import argparse
import asyncio

import yaml
from latentqa.eval.circuit_based_steering_eval import (
    CircuitBasedSteeringEvalConfig,
    do_circuit_based_steering_eval,
)


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, help="Path to config yaml path.")
    parser.add_argument("--forward_batch_size", type=int, default=None)
    args = parser.parse_args()

    if args.config_path:
        with open(args.config_path, "r") as f:
            config_dict = yaml.load(f, Loader=yaml.FullLoader)

    config = CircuitBasedSteeringEvalConfig(**config_dict)

    if args.forward_batch_size is not None:
        config.forward_batch_size = args.forward_batch_size

    _ = await do_circuit_based_steering_eval(config)


if __name__ == "__main__":
    asyncio.run(main())
