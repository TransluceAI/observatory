"""
Computes maximally and minimally activating exemplars for MLP neurons in a LM.
Can also save random sequences for each split and neuron.
"""

import argparse
from typing import Any

from activations.activations_computation import ActivationType
from activations.dataset import HFDatasetWrapperConfig, fineweb_dset_config, lmsys_dset_config
from activations.exemplars import ExemplarSplit
from activations.exemplars_computation import (
    compute_exemplars_for_layer,
    save_random_seqs_for_layer,
)
from activations.exemplars_wrapper import ExemplarConfig, ExemplarsWrapper
from util.subject import Subject, get_subject_config

parser = argparse.ArgumentParser()
parser.add_argument(
    "--activation_type",
    type=str,
    choices=[
        ActivationType.RESID,
        ActivationType.MLP_IN,
        ActivationType.MLP_OUT,
        ActivationType.ATTN_OUT,
        ActivationType.NEURONS,
    ],
    default="neurons",
    help="Type of activations from which we pick indices to compute exemplars for.",
)
parser.add_argument(
    "--layer_indices",
    type=int,
    nargs="+",
    default=None,
    help="Layers from which we pick indices to compute exemplars for.",
)
parser.add_argument(
    "--subject_hf_model_id",
    type=str,
    default="meta-llama/Meta-Llama-3.1-8B-Instruct",
    help="Huggingface model id for model to get activations for.",
)
parser.add_argument(
    "--split",
    type=str,
    choices=["train", "valid", "test", "random_train", "random_valid", "random_test"],
    help="Exemplar split to compute.",
)
parser.add_argument(
    "--hf_datasets",
    type=str,
    nargs="+",
    default=["fineweb", "lmsys"],
    help="Huggingface datasets to use.",
)
parser.add_argument(
    "--sampling_ratios",
    type=float,
    nargs="+",
    default=None,
    help="Sampling ratio for each dataset. If not specified, sample uniformly.",
)
parser.add_argument(
    "--data_dir",
    type=str,
    default="data/",
    help="Base data dir to create the results folder in.",
)
parser.add_argument(
    "--num_seqs",
    type=int,
    default=4_000_000,
    help="Number of sequences to consider when computing top activations.",
)
parser.add_argument(
    "--seq_len",
    type=int,
    default=64,
    help="Number of tokens per sequence.",
)
parser.add_argument("--k", type=int, default=32, help="Number of sequences to keep.")
parser.add_argument(
    "--num_top_acts_to_save",
    type=int,
    default=10_000,
    help="Number of top activations to save for each neuron from each end (min and max).",
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=1024,
    help="Batch size for getting model activations.",
)
parser.add_argument("--seed", type=int, default=64, help="Random seed for reproducibility.")
args = parser.parse_args()

assert args.seq_len >= 64, "It's probably best to use long-enough sequences."

subject_config = get_subject_config(args.subject_hf_model_id)
subject = Subject(subject_config, nnsight_lm_kwargs={"dispatch": True})

hf_dataset_configs: list[HFDatasetWrapperConfig] = []
for hf_dataset in args.hf_datasets:
    if hf_dataset == "fineweb":
        hf_dataset_configs.append(fineweb_dset_config)
    elif hf_dataset == "lmsys":
        hf_dataset_configs.append(lmsys_dset_config)
    else:
        raise ValueError(f"Unknown dataset: {hf_dataset}")

exemplar_config = ExemplarConfig(
    hf_model_id=args.subject_hf_model_id,
    hf_dataset_configs=tuple(hf_dataset_configs),
    sampling_ratios=args.sampling_ratios,
    num_seqs=args.num_seqs,
    seq_len=args.seq_len,
    k=args.k,
    num_top_acts_to_save=args.num_top_acts_to_save,
    batch_size=args.batch_size,
    seed=args.seed,
    activation_type=args.activation_type,
)
exemplars_wrapper = ExemplarsWrapper(args.data_dir, exemplar_config, subject)

layer_indices = args.layer_indices if args.layer_indices else range(subject.L)
for layer in layer_indices:
    print(f"============ Layer {layer} ============")
    kwargs: dict[str, Any] = {
        "exemplars_wrapper": exemplars_wrapper,
        "layer": layer,
        "split": ExemplarSplit(args.split),
    }
    if args.split.startswith("random"):
        save_random_seqs_for_layer(**kwargs)
    else:
        compute_exemplars_for_layer(**kwargs)
