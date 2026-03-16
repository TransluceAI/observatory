from enum import Enum
from typing import Any

from torch.utils.data import Dataset


class SimpleDataset(Dataset[Any]):
    def __init__(self, data: list[Any]):
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: Any) -> Any:
        return self.data[index]


class Split(str, Enum):
    """The split of the dataset."""

    TRAIN = "train"
    VALID = "valid"
    TEST = "test"
