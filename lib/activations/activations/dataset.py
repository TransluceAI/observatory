import random
from typing import Any, Dict, Iterator, List, Literal, Optional, Tuple

import numpy as np
import torch
from datasets import load_dataset  # type: ignore
from pydantic import BaseModel
from torch.utils.data import Dataset, IterableDataset
from util.subject import Subject


class HFDatasetWrapperConfig(BaseModel):
    hf_dataset_id: str
    dataset_config_name: Optional[str] = None
    hf_split: str = "train"
    seed: int = 54


class HFSplitDatasetWrapper(Dataset[Any]):
    def __init__(self, dataset: Dataset[Any], is_chat_format: bool, subject: Subject):
        self.dataset = dataset
        self.is_chat_format = is_chat_format
        self.subject = subject

    def __len__(self) -> int:
        assert hasattr(self.dataset, "__len__")
        return len(self.dataset)

    def __getitem__(self, idx: int) -> List[int]:
        example = self.dataset[idx]["data_column"]
        if self.is_chat_format:
            ids = self.subject.tokenizer.apply_chat_template(
                example, add_generation_prompt=False, tokenize=True, return_dict=False
            )
        else:
            ids = self.subject.tokenizer.encode(example, add_special_tokens=True)
        ids: List[int] = list(ids)
        return ids

    def __iter__(self) -> Iterator[List[int]]:
        for idx in range(len(self)):
            yield self[idx]


class HFDatasetWrapper:
    def __init__(
        self,
        config: HFDatasetWrapperConfig,
        subject: Subject,
        num_proc: int = 16,
    ):
        if config.hf_dataset_id in ["HuggingFaceFW/fineweb", "HuggingFaceFW/fineweb-edu"]:
            assert config.hf_split == "train"
            dset_kwargs = {
                "path": config.hf_dataset_id,
                "name": config.dataset_config_name,
                "split": "train",
            }
            column_name = "text"
            is_chat_format = False
        elif config.hf_dataset_id == "lmsys/lmsys-chat-1m":
            assert config.hf_split == "train"
            dset_kwargs = {
                "path": config.hf_dataset_id,
                "split": "train",
            }
            column_name = "conversation"
            is_chat_format = True
        elif config.hf_dataset_id == "HuggingFaceH4/ultrachat_200k":
            assert config.hf_split == "train_sft"
            dset_kwargs = {
                "path": config.hf_dataset_id,
                "split": "train_sft",
            }
            column_name = "messages"
            is_chat_format = True
        else:
            raise ValueError(f'Unrecognized dataset name "{config.hf_dataset_id}"!')

        # If the dataset is chat data, but the subject is not a chat model, throw error.
        if is_chat_format and not subject.is_chat_model:
            raise ValueError(f"Dataset is chat data, but subject is not a chat model!")

        dataset = load_dataset(num_proc=num_proc, **dset_kwargs)  # type: ignore
        dataset = dataset.rename_column(column_name, "data_column")

        # Shuffle the dataset and split into train, valid, test.
        dataset = dataset.shuffle(seed=config.seed)  # type: ignore
        # This operation takes 1-2 minutes, but only needs to be done once since the results get
        # cached.
        dataset = dataset.flatten_indices(num_proc=num_proc)  # type: ignore
        split_datasets: Dict[str, Dataset[Any]] = {
            "train": dataset.select(range(len(dataset) // 3)),  # type: ignore
            "valid": dataset.select(range(len(dataset) // 3, 2 * len(dataset) // 3)),  # type: ignore
            "test": dataset.select(range(2 * len(dataset) // 3, len(dataset))),  # type: ignore
        }
        self.split_datasets = split_datasets
        self.subject = subject
        self.is_chat_format = is_chat_format  # Whether the dataset is inherently in chat format.

    def get_chat_prefix(self) -> Tuple[int, ...]:
        if self.subject.lm_config.hf_model_id in [
            "meta-llama/Meta-Llama-3.1-8B-Instruct",
            "meta-llama/Meta-Llama-3.1-70B-Instruct",
        ]:
            prefix_ids = (
                128000,
                128006,
                9125,
                128007,
                271,
                38766,
                1303,
                33025,
                2696,
                25,
                6790,
                220,
                2366,
                18,
                198,
                15724,
                2696,
                25,
                220,
                1627,
                10263,
                220,
                2366,
                19,
                271,
                128009,
                128006,
                882,
                128007,
                271,
            )
        else:
            raise ValueError(f"Unsupported model_name {self.subject.lm_config.hf_model_id}!")
        return prefix_ids

    def get_dataset_for_split(
        self, split: Literal["train", "valid", "test"]
    ) -> HFSplitDatasetWrapper:
        return HFSplitDatasetWrapper(self.split_datasets[split], self.is_chat_format, self.subject)


class ChatDataset(Dataset[Any]):
    """Wrapper around HFDatasetWrapper for chat data that takes care of sampling."""

    def __init__(
        self,
        hf_dataset: HFDatasetWrapper,
        dataset_split: Literal["train", "valid", "test"],
        seq_len: int,
    ):
        assert hf_dataset.is_chat_format

        self.dataset_for_split = hf_dataset.get_dataset_for_split(dataset_split)
        self.split = dataset_split
        self.seq_len = seq_len

    @staticmethod
    def get_seq(full_document: List[int], seq_len: int) -> List[int]:
        return full_document[:seq_len]

    def __len__(self) -> int:
        return len(self.dataset_for_split)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.tensor(self.get_seq(self.dataset_for_split[idx], self.seq_len))


class NonChatDataset(IterableDataset[Any]):
    """
    Wrapper around HFDatasetWrapper for non-chat data that takes care of sampling.
    Note that even though the underlying dataset is non-chat format, the output from getitem
    can be in chat format if use_chat_format is True.
    """

    def __init__(
        self,
        hf_dataset: HFDatasetWrapper,
        dataset_split: Literal["train", "valid", "test"],
        seq_len: int,
        use_chat_format: bool,
        seed: int,
    ):
        self.dataset_for_split = hf_dataset.get_dataset_for_split(dataset_split)
        self.split = dataset_split
        self.prefix_ids: Tuple[int, ...] = hf_dataset.get_chat_prefix() if use_chat_format else ()
        self.seq_len = seq_len - len(self.prefix_ids)
        self.rng = random.Random(seed)

    def sample_seq(self) -> torch.Tensor:
        while True:
            idx = self.rng.randint(a=0, b=len(self.dataset_for_split) - 1)
            full_document = self.dataset_for_split[idx]
            # Skip if the document is too short.
            if len(full_document) < self.seq_len:
                continue
            start = self.rng.randint(a=0, b=len(full_document) - self.seq_len)
            seq = full_document[start : start + self.seq_len]
            break

        if len(self.prefix_ids) > 0:
            seq = np.concatenate([self.prefix_ids, seq])
        return torch.tensor(seq)

    def __iter__(self) -> Iterator[torch.Tensor]:
        while True:
            yield self.sample_seq()


class MultiSourceDataset(IterableDataset[Any]):
    def __init__(
        self,
        datasets: List[Dataset[Any] | IterableDataset[Any]],
        weights: Optional[List[float]] = None,
        seed: Optional[int] = None,
    ):
        """
        Combines multiple Datasets and/or IterableDatasets, sampling based on weights.

        Args:
            datasets: List of datasets (can be mix of Dataset and IterableDataset)
            weights: Optional sampling weights for each dataset (will be normalized).
                    If None, uniform weights will be used.
            seed: Random seed for reproducibility
        """
        self.datasets = datasets
        self.rng = random.Random(seed)

        # Normalize weights or use uniform weights
        if weights is None:
            weights = [1.0] * len(datasets)
        total = sum(weights)
        self.weights = [w / total for w in weights]

        # Setup indices for regular datasets
        self.regular_indices: List[List[int] | None] = []
        for ds in datasets:
            if not isinstance(ds, IterableDataset):
                assert hasattr(ds, "__len__")
                self.regular_indices.append(list(range(len(ds))))  # type: ignore
            else:
                self.regular_indices.append(None)

    def _get_iterator(
        self, dataset: Dataset[Any] | IterableDataset[Any], indices: List[int] | None
    ) -> Iterator[torch.Tensor]:
        """Creates appropriate iterator for dataset type."""
        if not isinstance(dataset, IterableDataset):
            assert indices is not None
            while True:
                idx = self.rng.choice(indices)
                yield dataset[idx]
        else:
            while True:
                iterator = iter(dataset)
                try:
                    while True:
                        yield next(iterator)
                except StopIteration:
                    continue

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, int]]:
        # Create iterators for all datasets
        iterators = [
            self._get_iterator(ds, idx) for ds, idx in zip(self.datasets, self.regular_indices)
        ]

        while True:
            # Choose dataset according to weights
            dataset_idx = self.rng.choices(range(len(self.datasets)), weights=self.weights, k=1)[0]

            # Get next item from chosen dataset
            yield next(iterators[dataset_idx]), dataset_idx


####################################
# Example HFDatasetWrapperConfig's #
####################################

fineweb_dset_config = HFDatasetWrapperConfig(
    hf_dataset_id="HuggingFaceFW/fineweb",
    dataset_config_name="sample-10BT",
    hf_split="train",
    seed=54,
)

lmsys_dset_config = HFDatasetWrapperConfig(
    hf_dataset_id="lmsys/lmsys-chat-1m",
    hf_split="train",
    seed=54,
)

ultrachat_dset_config = HFDatasetWrapperConfig(
    hf_dataset_id="HuggingFaceH4/ultrachat_200k",
    hf_split="train_sft",
    seed=54,
)
