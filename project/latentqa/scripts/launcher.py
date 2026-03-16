import argparse
import importlib
import os
import subprocess
import sys
from typing import Any

import yaml
from util.gpu import list_free_gpus


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file",
        type=str,
        required=True,
        help=(
            'python file that has a "get_configs()" function'
            "which returns a list of data classes."
        ),
    )
    parser.add_argument("-f", "--main_file", type=str)
    parser.add_argument("-g", "--num_gpus", type=int, default=1)
    parser.add_argument("--use_torchrun", type=eval, default=False, choices=[True, False])
    parser.add_argument("--launch", type=eval, default=True, choices=[True, False])
    args = parser.parse_args()

    # Import the get_configs function in the config file.
    # Convert the config file path to an absolute path first
    config_file_path = os.path.abspath(args.config_file)
    # Remove the .py extension
    if config_file_path.endswith(".py"):
        config_file_path = config_file_path[:-3]

    # Convert the path to a module path by replacing path separators with dots
    # We need to ensure this is a proper absolute import, not a relative one
    sys_path = list(map(os.path.abspath, sys.path))
    for path in sys_path:
        if config_file_path.startswith(path):
            module_name = (
                config_file_path[len(path) :].lstrip(os.path.sep).replace(os.path.sep, ".")
            )
            break
    else:
        # If not in sys.path, temporarily add the directory to sys.path
        dir_path = os.path.dirname(config_file_path)
        module_name = os.path.basename(config_file_path)
        sys.path.insert(0, dir_path)

    module = importlib.import_module(module_name)
    action = args.main_file.split("/")[-1].rstrip(".py")
    get_configs = getattr(module, f"{action}_configs")

    configs: list[dict[str, Any]] = get_configs()

    # Check that we have enough GPUs.
    free_gpus = list_free_gpus()
    print(f"Free GPUs: {free_gpus}")

    commands: list[str] = []
    for i, config in enumerate(configs):
        assert "save_path" in config, "`save_path` arg must always be provided."
        save_path = config["save_path"]
        os.makedirs(save_path, exist_ok=True)
        # Create and save yaml config files in the experiment folder.
        yaml_file = os.path.join(save_path, f"{action}_config.yaml")
        with open(yaml_file, "w") as f:
            yaml.dump(config, f)

        if args.use_torchrun:
            prefix = f"torchrun --nproc_per_node={args.num_gpus} --master_port={9999 - i}"
        else:
            prefix = "python"
        cmd = f"{prefix} {args.main_file} --config_path {yaml_file}"
        if args.num_gpus > 0:
            gpu_ids = free_gpus[i * args.num_gpus : (i + 1) * args.num_gpus]
            gpu_str = "CUDA_VISIBLE_DEVICES=" + ",".join(str(id) for id in gpu_ids)
            cmd = f"{gpu_str} {cmd}"
        commands.append(cmd)

    full_command = " \\\n  && ".join(commands)

    if args.launch:
        subprocess.call(full_command, shell=True)
    else:
        print("We will launch the following commands:")
        print(full_command)


if __name__ == "__main__":
    main()
