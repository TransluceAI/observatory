import os
import pickle
from collections import defaultdict
from typing import Any, Dict, List, Literal, Optional, Sequence, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
from activations.activations import ActivationRecord
from activations.activations_computation import ActivationType
from activations.dataset import (
    ChatDataset,
    HFDatasetWrapper,
    HFDatasetWrapperConfig,
    NonChatDataset,
    fineweb_dset_config,
    lmsys_dset_config,
)
from activations.exemplars import ExemplarSplit, ExemplarType, NeuronExemplars
from IPython.display import HTML, display  # type: ignore
from pydantic import BaseModel
from torch.utils.data import Dataset, IterableDataset
from util.chat_input import IdsInput
from util.subject import Subject, get_subject_config
from util.types import NDFloatArray, NDIntArray

QUANTILE_KEYS = (
    1e-8,
    1e-7,
    1e-6,
    1e-5,
    1e-4,
    1 - 1e-4,
    1 - 1e-5,
    1 - 1e-6,
    1 - 1e-7,
    1 - 1e-8,
)


def get_color_str(act: float, cmap: bool = True) -> str:
    if act < 0:
        color = f"rgba(255, 0, 0, {abs(act)})"  # red
    else:
        color = f"rgba(0, 255, 0, {act})"  # green
    return color


# TODO(damichoi): Fold this function into the other one.
def generate_html_for_visualizing_neuron_exemplars_and_activations_single_sign(
    layer: int,
    neuron_idx: int,
    exemplars: List[ActivationRecord],
    unnorm_exemplars: List[ActivationRecord],
    dataset_names: List[str],
    ranks: List[int],
    extype: ExemplarType,
):
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Exemplar Visualization</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                line-height: 1.6;
                margin: 0;
                padding: 20px;
            }
            .container {
                display: flex;
                justify-content: center; /* Center the column */
            }
            .column {
                width: 100%; /* Full width */
            }
            .row {
                margin-bottom: 20px;
            }
            .rank {
                font-weight: bold;
                margin-right: 10px;
            }
            .textbox {
                border: 1px solid #ccc;
                padding: 10px;
                overflow-x: auto;
            }
            .token {
                white-space: pre-wrap;
                word-wrap: break-word;
            }
        </style>
    </head>
    """
    html_content += f"""
    <body>
        <span style="font-size: x-large; font-weight: bold;">Neuron ({layer}, {neuron_idx})</span>
        <div class="container">
    """

    if extype == "max":
        html_content += f'<div class="column"><h2>Maximally Activating</h2>'
    elif extype == "min":
        html_content += f'<div class="column"><h2>Minimally Activating</h2>'
    else:
        raise ValueError(f"Invalid ExemplarType: {extype}")

    assert len(ranks) == len(exemplars)
    assert len(ranks) == len(dataset_names)
    for rank, act_rec, unnorm_act_rec, dset_name in zip(
        ranks,
        exemplars,
        unnorm_exemplars,
        dataset_names,
    ):
        html_content += f"""
        <div class="row">
            <h3>{rank}. {dset_name}</h3>
            <div class="textbox">
        """

        tokens = act_rec.tokens
        activations = np.array(act_rec.activations) * (-1 if extype == ExemplarType.MIN else 1)
        unnorm_activations = np.array(unnorm_act_rec.activations)
        for token, act, unnorm_act in zip(tokens, activations, unnorm_activations):
            color = get_color_str(act, True)
            html_content += f'<span class="token" title="Activation: {unnorm_act:.2f}" style="background-color: {color};">{token}</span>'

        html_content += "</div></div>"

    html_content += """
        </div> <!-- Close column -->
    </div> <!-- Close container -->
    </body>
    </html>
    """

    return html_content


def generate_html_for_visualizing_neuron_exemplars_and_activations(
    layer: int,
    neuron_idx: int,
    exemplars: Dict[ExemplarType, List[ActivationRecord]],
    unnorm_exemplars: Dict[ExemplarType, List[ActivationRecord]],
    dataset_names: Dict[ExemplarType, List[str]],
    ranks: Dict[ExemplarType, List[int]],
):
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Exemplar Visualization</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                line-height: 1.6;
                margin: 0;
                padding: 20px;
            }
            .container {
                display: flex;
                justify-content: space-between;
            }
            .column {
                width: 48%;
            }
            .row {
                margin-bottom: 20px;
            }
            .rank {
                font-weight: bold;
                margin-right: 10px;
            }
            .textbox {
                border: 1px solid #ccc;
                padding: 10px;
                overflow-x: auto;
            }
            .token {
                white-space: pre-wrap;
                word-wrap: break-word;
            }
        </style>
    </head>
    """
    html_content += f"""
    <body>
        <span style="font-size: x-large; font-weight: bold;">Neuron ({layer}, {neuron_idx})</span>
        <div class="container">
    """

    for extype in exemplars:
        if extype.value == "max":
            html_content += f'<div class="column"><h2>Maximally Activating</h2>'
        elif extype.value == "min":
            html_content += f'<div class="column"><h2>Minimally Activating</h2>'
        else:
            raise ValueError(f"Invalid ExemplarType: {extype}")

        assert len(ranks[extype]) == len(exemplars[extype])
        assert len(ranks[extype]) == len(dataset_names[extype])
        for rank, act_rec, unnorm_act_rec, dset_name in zip(
            ranks[extype],
            exemplars[extype],
            unnorm_exemplars[extype],
            dataset_names[extype],
        ):
            html_content += f"""
            <div class="row">
                <h3>{rank}. {dset_name}</h3>
                <div class="textbox">
            """

            tokens = act_rec.tokens
            activations = np.array(act_rec.activations) * (-1 if extype == ExemplarType.MIN else 1)
            unnorm_activations = np.array(unnorm_act_rec.activations)
            for token, act, unnorm_act in zip(tokens, activations, unnorm_activations):
                color = get_color_str(act, True)
                html_content += f'<span class="token" title="Activation: {unnorm_act:.2f}" style="background-color: {color};">{token}</span>'

            html_content += "</div></div>"

        html_content += "</div>"

    html_content += """
        </div>
    </body>
    </html>
    """

    return html_content


def strip_padding(token_ids: Sequence[int], pad_token_id: int) -> Sequence[int]:
    first_non_pad_idx = 0
    while first_non_pad_idx < len(token_ids):
        if token_ids[first_non_pad_idx] != pad_token_id:
            break
        first_non_pad_idx += 1
    return token_ids[first_non_pad_idx:]


def approximate_quantile(
    q: float,
    N: int,
    k: int,
    bottom_k_values: NDFloatArray,
    top_k_values: NDFloatArray,
) -> NDFloatArray:
    """
    Approximate the q-quantile for each batch, given the bottom k and top k values.

    Parameters:
    - q: The desired quantile (cumulative probability).
    - N: The total number of data points.
    - k: The number of known bottom and top values.
    - bottom_k_values: Array of shape (batch_size, k) containing bottom k values.
    - top_k_values: Array of shape (batch_size, k) containing top k values.

    Returns:
    - approx_values: Array of shape (batch_size,) with the approximated quantile values.
    """
    batch_size = bottom_k_values.shape[0]
    approx_values = np.empty(batch_size, dtype=np.float64)

    # Known cumulative probabilities for bottom_k_values and top_k_values
    bottom_p = np.arange(1, k + 1) / N  # Shape: (k,)
    top_p = (N - k + np.arange(1, k + 1)) / N  # Shape: (k,)

    # Determine if q is in lower or upper quantile range
    if (1 / N) <= q <= (k / N):
        # Lower quantiles
        p = bottom_p
        values = bottom_k_values
    elif ((N - k + 1) / N) <= q <= 1:
        # Upper quantiles
        p = top_p
        values = top_k_values
    else:
        raise ValueError(f"q={q} is out of the known quantile ranges based on k={k} and N={N}.")

    # Find the indices for interpolation
    indices = np.searchsorted(p, q, side="right") - 1
    indices = np.clip(indices, 0, k - 2)  # Ensure indices are within valid range

    # Get the cumulative probabilities and values for interpolation
    p_lower = p[indices]  # Shape: (batch_size,)
    p_upper = p[indices + 1]  # Shape: (batch_size,)
    v_lower = values[:, indices]  # Shape: (batch_size,)
    v_upper = values[:, indices + 1]  # Shape: (batch_size,)

    # Compute the fraction for interpolation
    fraction = (v_upper - v_lower) / (p_upper - p_lower)

    # Handle cases where p_upper == p_lower to avoid division by zero
    zero_denominator = p_upper == p_lower
    approx_values[zero_denominator] = v_lower[zero_denominator]
    approx_values[~zero_denominator] = v_lower[~zero_denominator] + fraction * (
        q - p_lower[~zero_denominator]
    )

    return approx_values


class ExemplarConfig(BaseModel):
    hf_model_id: str
    hf_dataset_configs: Tuple[HFDatasetWrapperConfig, ...] = (
        fineweb_dset_config,
        lmsys_dset_config,
    )
    sampling_ratios: Optional[List[float]] = None
    num_seqs: int = 1_000_000
    seq_len: int = 64
    k: int = 100
    num_top_acts_to_save: int = 10_000
    batch_size: int = 512
    rand_seqs: int = 10
    seed: int = 64
    activation_type: ActivationType = ActivationType.NEURONS


class ExemplarsWrapper:
    def __init__(
        self,
        data_dir: str,
        config: ExemplarConfig,
        subject: Subject,
    ):
        # Check whether hf_model_id matches subject.
        assert config.hf_model_id == subject.lm_config.hf_model_id

        hf_datasets: Dict[str, HFDatasetWrapper] = {}
        for hf_dataset_config in config.hf_dataset_configs:
            hf_dataset = HFDatasetWrapper(config=hf_dataset_config, subject=subject)
            dataset_name = hf_dataset_config.hf_dataset_id.split("/")[-1]
            hf_datasets[dataset_name] = hf_dataset
        dataset_names: List[str] = sorted(hf_datasets.keys())

        folder_name_components = dataset_names.copy()
        model_name = config.hf_model_id.split("/")[-1].lower().replace("-", "_")
        folder_name_components.append(model_name)
        if subject.is_chat_model:
            folder_name_components.append("chat")
        folder_name_components.append(f"{config.seq_len}seqlen")
        if config.activation_type != "neurons":
            folder_name_components.append(config.activation_type)
        assert subject.tokenizer.padding_side == "left"

        folder_name = "_".join(folder_name_components)
        save_path = os.path.join(data_dir, folder_name)
        os.makedirs(save_path, exist_ok=True)

        # Check that data exists in save_path.
        config_file = os.path.join(save_path, "exemplar_config.json")
        if os.path.exists(config_file):
            with open(config_file, "r") as f:
                existing_config = ExemplarConfig.model_validate_json(f.read())
            # Check that the configs are the same.
            fields_to_exclude = set(["num_seqs"])
            existing_config_dict = existing_config.model_dump(exclude=fields_to_exclude)
            config_dict = config.model_dump(exclude=fields_to_exclude)
            for field in existing_config_dict:
                existing_val = existing_config_dict[field]
                curr_val = config_dict[field]
                assert existing_val == curr_val, (
                    f"Value of '{field}' for existing config is '{existing_val}', "
                    f"while the value given in the config is '{curr_val}')"
                )
        else:
            # If there's no data saved in save_path yet, save our config.
            with open(config_file, "w") as f:
                f.write(config.model_dump_json())

        # Define cache. This is useful if we want to load exemplars for many individual neurons
        # since exemplar information is saved by layer.
        self.layers_cache: Dict[
            int,
            Tuple[
                Dict[ExemplarSplit, Dict[ExemplarType, NDFloatArray]],
                Dict[ExemplarSplit, Dict[ExemplarType, NDIntArray]],
                Dict[ExemplarSplit, Dict[ExemplarType, NDIntArray]],
                Dict[float, NDFloatArray],
            ],
        ] = {}

        self.config: ExemplarConfig = config
        self.subject: Subject = subject
        self.hf_datasets: List[HFDatasetWrapper] = [hf_datasets[name] for name in dataset_names]
        self.dataset_names: List[str] = dataset_names
        self.save_path: str = save_path

    @classmethod
    def from_disk(cls, save_path: str, subject: Optional[Subject] = None):
        """Use this if you have the directory containing an exemplar wrapper config."""
        config_path = os.path.join(save_path, "exemplar_config.json")
        if not os.path.exists(config_path):
            raise ValueError(f"No exemplar config found at {config_path}")
        with open(config_path, "r") as f:
            exemplar_config = ExemplarConfig.model_validate_json(f.read())
        data_dir = os.path.dirname(save_path.removesuffix("/"))
        if subject is None:
            subject = Subject(get_subject_config(exemplar_config.hf_model_id))
        return cls(data_dir, exemplar_config, subject)

    def get_layer_dir(self, layer: int, split: ExemplarSplit) -> str:
        return os.path.join(self.save_path, split.value, f"layer{layer}")

    def get_layer_dir_for_neuron_level(self, layer: int, split: ExemplarSplit) -> str:
        return os.path.join(self.save_path, split.value, "neuron_level", f"layer{layer}")

    def load_layer_checkpoint(self, layer: int, split: ExemplarSplit) -> (
        tuple[
            Dict[ExemplarType, NDFloatArray],
            Dict[ExemplarType, NDFloatArray],
            Dict[ExemplarType, NDIntArray],
            Dict[ExemplarType, NDIntArray],
            int | None,
            int | None,
        ]
        | None
    ):
        # If we're loading for a random split, we don't load
        # top activations, step, and num_tokens_seen.
        random = split in (
            ExemplarSplit.RANDOM_TRAIN,
            ExemplarSplit.RANDOM_VALID,
            ExemplarSplit.RANDOM_TEST,
        )

        num_features = self.num_features
        num_top_feats_to_save = self.config.num_top_acts_to_save
        k, seq_len = self.config.k, self.config.seq_len

        layer_dir = self.get_layer_dir(layer, split)
        try:
            if not random:
                with open(os.path.join(layer_dir, "step.txt"), "r") as f:
                    step = int(f.read())
                with open(os.path.join(layer_dir, "num_tokens_seen.txt"), "r") as f:
                    num_tokens_seen = int(f.read())
            else:
                step, num_tokens_seen = None, None

            acts: Dict[ExemplarType, NDFloatArray] = {}
            seq_acts: Dict[ExemplarType, NDFloatArray] = {}
            ids: Dict[ExemplarType, NDIntArray] = {}
            dataset_ids: Dict[ExemplarType, NDIntArray] = {}
            for extype in ExemplarType:
                if not random:
                    acts[extype] = np.load(os.path.join(layer_dir, f"{extype.value}_acts.npy"))
                    assert (
                        acts[extype].shape[0] == num_features
                        and acts[extype].shape[1] <= num_top_feats_to_save
                    )

                seq_acts[extype] = np.load(os.path.join(layer_dir, f"{extype.value}_seq_acts.npy"))
                assert seq_acts[extype].shape == (num_features, k, seq_len)

                ids[extype] = np.load(os.path.join(layer_dir, f"{extype.value}_seq_ids.npy"))
                assert ids[extype].shape == (num_features, k, seq_len)

                dataset_ids[extype] = np.load(
                    os.path.join(layer_dir, f"{extype.value}_dataset_ids.npy")
                )
                assert dataset_ids[extype].shape == (num_features, k)

            return acts, seq_acts, ids, dataset_ids, step, num_tokens_seen
        except:
            return None

    def save_layer_checkpoint(
        self,
        layer: int,
        split: ExemplarSplit,
        seq_acts: Dict[ExemplarType, NDFloatArray],
        token_ids: Dict[ExemplarType, NDIntArray],
        dataset_ids: Dict[ExemplarType, NDIntArray],
        acts: Optional[Dict[ExemplarType, NDFloatArray]] = None,
        step: Optional[int] = None,
        num_tokens_seen: Optional[int] = None,
    ) -> None:
        # If we're loading for a random split, we don't load
        # top activations, step, and num_tokens_seen.
        random = split in (
            ExemplarSplit.RANDOM_TRAIN,
            ExemplarSplit.RANDOM_VALID,
            ExemplarSplit.RANDOM_TEST,
        )

        layer_dir = self.get_layer_dir(layer, split)
        os.makedirs(layer_dir, exist_ok=True)

        num_features = self.num_features
        num_top_feats_to_save = self.config.num_top_acts_to_save
        k, seq_len = self.config.k, self.config.seq_len

        # Save data and check shapes.
        for extype in ExemplarType:
            if not random:
                assert (
                    acts is not None
                    and acts[extype].shape[0] == num_features
                    and acts[extype].shape[1] <= num_top_feats_to_save
                )
                np.save(os.path.join(layer_dir, f"{extype.value}_acts.npy"), acts[extype])

            assert seq_acts[extype].shape == (num_features, k, seq_len)
            np.save(
                os.path.join(layer_dir, f"{extype.value}_seq_acts.npy"),
                seq_acts[extype],
            )

            assert token_ids[extype].shape == (num_features, k, seq_len)
            np.save(
                os.path.join(layer_dir, f"{extype.value}_seq_ids.npy"),
                token_ids[extype],
            )

            assert dataset_ids[extype].shape == (num_features, k)
            np.save(
                os.path.join(layer_dir, f"{extype.value}_dataset_ids.npy"),
                dataset_ids[extype],
            )

        if not random:
            with open(os.path.join(layer_dir, "step.txt"), "w") as f:
                f.write(str(step))
            with open(os.path.join(layer_dir, "num_tokens_seen.txt"), "w") as f:
                f.write(str(num_tokens_seen))

    def save_neuron_checkpoint(
        self,
        layer: int,
        neuron_idx: int,
        split: ExemplarSplit,
        seq_acts: Dict[ExemplarType, NDFloatArray],
        token_ids: Dict[ExemplarType, NDIntArray],
        dataset_ids: Dict[ExemplarType, NDIntArray],
        acts: Dict[ExemplarType, NDFloatArray],
        num_tokens_seen: Optional[int] = None,
    ) -> None:
        layer_dir = self.get_layer_dir_for_neuron_level(layer, split)
        os.makedirs(layer_dir, exist_ok=True)

        num_top_feats_to_save = self.config.num_top_acts_to_save
        k, seq_len = self.config.k, self.config.seq_len

        # Check shapes.
        for extype in ExemplarType:
            assert acts[extype].shape == (1, num_top_feats_to_save)
            assert seq_acts[extype].shape == (1, k, seq_len)
            assert token_ids[extype].shape == (1, k, seq_len)
            assert dataset_ids[extype].shape == (1, k)

        with open(os.path.join(layer_dir, f"{neuron_idx}.pkl"), "wb") as f:
            pickle.dump(
                {
                    "seq_acts": seq_acts,
                    "token_ids": token_ids,
                    "dataset_ids": dataset_ids,
                    "acts": acts,
                    "num_tokens_seen": num_tokens_seen,
                },
                f,
            )

    def get_datasets(
        self, dataset_split: Literal["train", "valid", "test"]
    ) -> List[Dataset[Any] | IterableDataset[Any]]:
        """Returns a list of PyTorch Datasets one for each HuggingFace dataset."""

        datasets: List[Dataset[Any] | IterableDataset[Any]] = []
        for hf_dataset in self.hf_datasets:
            if hf_dataset.is_chat_format:
                dataset = ChatDataset(
                    hf_dataset=hf_dataset,
                    dataset_split=dataset_split,
                    seq_len=self.config.seq_len,
                )
            else:
                dataset = NonChatDataset(
                    hf_dataset=hf_dataset,
                    dataset_split=dataset_split,
                    seq_len=self.config.seq_len,
                    use_chat_format=self.subject.is_chat_model,
                    seed=self.config.seed,
                )
            datasets.append(dataset)
        return datasets

    def get_layer_act_percs(self, layer: int) -> Dict[float, NDFloatArray]:
        """
        Computes the activation quantile information for a layer using top-activating sequences
        from the train, valid, and test splits.
        """
        act_percs_path = os.path.join(self.save_path, f"layer{layer}_act_percs.npy")
        if os.path.exists(act_percs_path):
            return np.load(act_percs_path, allow_pickle=True).item()

        layer_acts: Dict[ExemplarType, NDFloatArray] = {}
        splits: Set[ExemplarSplit] = set()
        for extype in ExemplarType:
            acts: Dict[ExemplarSplit, NDFloatArray] = {}
            for split in [ExemplarSplit.TRAIN, ExemplarSplit.VALID, ExemplarSplit.TEST]:
                layer_dir = self.get_layer_dir(layer, split)
                try:
                    acts[split] = np.load(
                        os.path.join(layer_dir, f"{extype.value}_acts.npy"),
                        mmap_mode="r",
                    )
                except:
                    continue
                splits.add(split)
            concatenated_acts = np.concatenate(list(acts.values()), axis=1)
            sorted_acts = np.sort(concatenated_acts, axis=1)
            layer_acts[extype] = sorted_acts

        num_tokens_seen = 0
        for split in splits:
            layer_dir = self.get_layer_dir(layer, split)
            with open(os.path.join(layer_dir, "num_tokens_seen.txt"), "r") as f:
                num_tokens_seen += int(f.read())

        act_percs: Dict[float, NDFloatArray] = {}
        for q in QUANTILE_KEYS:
            try:
                act_percs[q] = approximate_quantile(
                    q=q,
                    N=num_tokens_seen,
                    k=layer_acts[ExemplarType.MAX].shape[1],
                    bottom_k_values=layer_acts[ExemplarType.MIN],
                    top_k_values=layer_acts[ExemplarType.MAX],
                )
            except:
                continue
        np.save(act_percs_path, act_percs)  # type: ignore
        return act_percs

    def get_neuron_act_percs(
        self,
        top_acts: Dict[ExemplarSplit, Dict[ExemplarType, NDFloatArray]],
        num_tokens_seen: Dict[ExemplarSplit, int],
    ) -> Dict[float, float]:

        all_top_acts: Dict[ExemplarType, NDFloatArray] = {}
        splits: Set[ExemplarSplit] = set()
        for extype in ExemplarType:
            top_acts_list: List[NDFloatArray] = []
            for split in [ExemplarSplit.TRAIN, ExemplarSplit.VALID, ExemplarSplit.TEST]:
                if split not in top_acts:
                    continue
                top_acts_list.append(top_acts[split][extype])
                splits.add(split)
            concatenated_acts = np.concatenate(top_acts_list, axis=1)
            sorted_acts = np.sort(concatenated_acts, axis=1)
            all_top_acts[extype] = sorted_acts

        total_num_tokens_seen = sum(num_tokens_seen.values())

        act_percs: Dict[float, float] = {}
        for q in QUANTILE_KEYS:
            try:
                quantile = approximate_quantile(
                    q=q,
                    N=total_num_tokens_seen,
                    k=all_top_acts[ExemplarType.MAX].shape[1],
                    bottom_k_values=all_top_acts[ExemplarType.MIN],
                    top_k_values=all_top_acts[ExemplarType.MAX],
                )
                import pdb

                pdb.set_trace()
                act_percs[q] = float(quantile[0])
            except:
                continue
        return act_percs

    def get_layer_data(
        self,
        layer: int,
    ) -> Tuple[
        Dict[ExemplarSplit, Dict[ExemplarType, NDFloatArray]],
        Dict[ExemplarSplit, Dict[ExemplarType, NDIntArray]],
        Dict[ExemplarSplit, Dict[ExemplarType, NDIntArray]],
        Dict[float, NDFloatArray],
    ]:
        """
        Loads exemplar data for a layer.
        """
        if layer in self.layers_cache:
            return self.layers_cache[layer]

        num_neurons_per_layer = self.subject.I
        act_percs = self.get_layer_act_percs(layer)

        seq_acts: Dict[ExemplarSplit, Dict[ExemplarType, NDFloatArray]] = defaultdict(dict)
        token_ids: Dict[ExemplarSplit, Dict[ExemplarType, NDIntArray]] = defaultdict(dict)
        dataset_ids: Dict[ExemplarSplit, Dict[ExemplarType, NDIntArray]] = defaultdict(dict)
        for split in ExemplarSplit:
            layer_dir = self.get_layer_dir(layer, split)

            try:
                for extype in ExemplarType:
                    layer_seq_acts = np.load(
                        os.path.join(layer_dir, f"{extype.value}_seq_acts.npy"),
                        mmap_mode="r",
                    )
                    layer_seq_ids = np.load(
                        os.path.join(layer_dir, f"{extype.value}_seq_ids.npy"),
                        mmap_mode="r",
                    )
                    layer_dataset_ids = np.load(
                        os.path.join(layer_dir, f"{extype.value}_dataset_ids.npy")
                    )
                    seq_acts[split][extype] = layer_seq_acts[:num_neurons_per_layer]
                    token_ids[split][extype] = layer_seq_ids[:num_neurons_per_layer]
                    dataset_ids[split][extype] = layer_dataset_ids[:num_neurons_per_layer]
            except:
                continue
        # Save data in cache.
        self.layers_cache[layer] = (seq_acts, token_ids, dataset_ids, act_percs)
        return self.layers_cache[layer]

    def get_neuron_data(self, layer: int, neuron_idx: int) -> Tuple[
        Dict[ExemplarSplit, Dict[ExemplarType, NDFloatArray]],
        Dict[ExemplarSplit, Dict[ExemplarType, NDIntArray]],
        Dict[ExemplarSplit, Dict[ExemplarType, NDIntArray]],
        Dict[float, float],
    ]:
        seq_acts: Dict[ExemplarSplit, Dict[ExemplarType, NDFloatArray]] = {}
        token_ids: Dict[ExemplarSplit, Dict[ExemplarType, NDIntArray]] = {}
        dataset_ids: Dict[ExemplarSplit, Dict[ExemplarType, NDIntArray]] = {}
        top_acts: Dict[ExemplarSplit, Dict[ExemplarType, NDFloatArray]] = {}
        num_tokens_seen: Dict[ExemplarSplit, int] = {}
        for split in ExemplarSplit:
            layer_dir = self.get_layer_dir_for_neuron_level(layer, split)
            data_path = os.path.join(layer_dir, f"{neuron_idx}.pkl")
            if not os.path.exists(data_path):
                continue
            with open(data_path, "rb") as f:
                data: Dict[str, Any] = pickle.load(f)
            seq_acts[split] = data["seq_acts"]
            token_ids[split] = data["token_ids"]
            dataset_ids[split] = data["dataset_ids"]
            top_acts[split] = data["acts"]
            num_tokens_seen[split] = data["num_tokens_seen"]

        act_percs = self.get_neuron_act_percs(top_acts, num_tokens_seen)
        return seq_acts, token_ids, dataset_ids, act_percs

    def get_neuron_exemplars(
        self,
        layer: int,
        neuron_idx: int,
    ) -> NeuronExemplars:
        """Returns NeuronExemplars for a neuron, given a split."""
        (
            layer_acts,
            layer_token_ids,
            layer_dataset_ids,
            layer_act_percs,
        ) = self.get_layer_data(layer)

        pad_id = self.subject.tokenizer.pad_token_id
        assert pad_id is not None and isinstance(pad_id, int)

        act_records: Dict[ExemplarSplit, Dict[ExemplarType, List[ActivationRecord]]] = {}
        dset_names: Dict[ExemplarSplit, Dict[ExemplarType, List[str]]] = {}
        for split in layer_acts.keys():
            act_records[split] = {}
            dset_names[split] = {}
            for extype in ExemplarType:
                act_records[split][extype] = []
                neuron_acts = layer_acts[split][extype][neuron_idx]
                neuron_token_ids = layer_token_ids[split][extype][neuron_idx]

                for acts, ids in zip(neuron_acts, neuron_token_ids):
                    ids = strip_padding(ids, pad_id)
                    acts = acts[-len(ids) :]

                    tokens: List[str] = [self.subject.decode(id) for id in ids]
                    act_records[split][extype].append(
                        ActivationRecord(
                            tokens=tokens, token_ids=list(ids), activations=acts.tolist()
                        )
                    )

                dset_names[split][extype] = []
                for dataset_id in layer_dataset_ids[split][extype][neuron_idx]:
                    dataset_name = self.dataset_names[int(dataset_id)]
                    dset_names[split][extype].append(dataset_name)

        act_percs: Dict[float, float] = {q: perc[neuron_idx] for q, perc in layer_act_percs.items()}

        return NeuronExemplars(
            activation_records=act_records,
            activation_percentiles=act_percs,
            dataset_names=dset_names,
        )

    def check_exemplar_activation_correctness(
        self,
        layer: int,
        neuron_idx: int,
        extype: ExemplarType,
        exemplar_split: ExemplarSplit,
        rank: int,
        plot: bool = True,
    ):
        neuron_exemplars = self.get_neuron_exemplars(layer, neuron_idx)
        act_rec = neuron_exemplars.activation_records[exemplar_split][extype][rank]
        ids = act_rec.token_ids
        acts = np.array(act_rec.activations)

        cis = IdsInput(input_ids=ids)
        out = self.subject.collect_acts([cis], [layer], include=["neurons_BTI"])
        layer_acts = out.layers[layer]
        assert layer_acts.neurons_BTI is not None
        true_acts: NDFloatArray = layer_acts.neurons_BTI[0, :, neuron_idx].float().cpu().numpy()  # type: ignore

        diffs = np.abs(true_acts - acts)
        print("Biggest diff: ", diffs.max())

        if plot:
            _, ax = plt.subplots(figsize=(10, 5))  # type: ignore
            ax.plot(np.arange(len(true_acts)), true_acts, label="true")
            ax.plot(np.arange(len(acts)), acts, label="loaded")
            ax.legend()
            ax.show()

    def visualize_neuron_exemplars(
        self,
        layer: int,
        neuron_idx: int,
        exemplar_split: ExemplarSplit,
        exemplar_type: Optional[ExemplarType] = None,
        indices: Optional[Sequence[int]] = None,
    ):
        """Visualizes neuron exemplars and activations using HTML."""
        neuron_exemplars = self.get_neuron_exemplars(layer, neuron_idx)
        norm_neuron_act_recs = neuron_exemplars.get_normalized_act_records(exemplar_split)

        indices_to_viz = list(indices or range(len(norm_neuron_act_recs[ExemplarType.MAX])))
        norm_act_recs: Dict[ExemplarType, List[ActivationRecord]] = {}
        unnorm_act_recs: Dict[ExemplarType, List[ActivationRecord]] = {}
        dataset_names: Dict[ExemplarType, List[str]] = {}
        ranks: Dict[ExemplarType, List[int]] = {}
        for extype in ExemplarType:
            norm_act_recs[extype] = [norm_neuron_act_recs[extype][i] for i in indices_to_viz]
            unnorm_act_recs[extype] = [
                neuron_exemplars.activation_records[exemplar_split][extype][i]
                for i in indices_to_viz
            ]
            dataset_names[extype] = [
                neuron_exemplars.dataset_names[exemplar_split][extype][i] for i in indices_to_viz
            ]
            ranks[extype] = indices_to_viz

        if exemplar_type is None:
            html_content = generate_html_for_visualizing_neuron_exemplars_and_activations(
                layer, neuron_idx, norm_act_recs, unnorm_act_recs, dataset_names, ranks
            )
        else:
            html_content = (
                generate_html_for_visualizing_neuron_exemplars_and_activations_single_sign(
                    layer,
                    neuron_idx,
                    norm_act_recs[exemplar_type],
                    unnorm_act_recs[exemplar_type],
                    dataset_names[exemplar_type],
                    ranks[exemplar_type],
                    exemplar_type,
                )
            )
        display(HTML(html_content))  # type: ignore

    @property
    def num_features(self) -> int:
        if self.config.activation_type == ActivationType.NEURONS:
            return self.subject.I
        if self.config.activation_type in (
            ActivationType.RESID,
            ActivationType.MLP_IN,
            ActivationType.MLP_OUT,
            ActivationType.ATTN_OUT,
        ):
            return self.subject.D
        raise ValueError(f"Invalid activation type: {self.config.activation_type}")


###################
# Example Configs #
###################

fineweb_lmsys_llama31_8b_instruct_config = ExemplarConfig(
    hf_model_id="meta-llama/Meta-Llama-3.1-8B-Instruct",
    hf_dataset_configs=(fineweb_dset_config, lmsys_dset_config),
    sampling_ratios=[0.5, 0.5],
    num_seqs=1_000_000,
    seq_len=95,
    k=100,
    batch_size=512,
    seed=64,
)
