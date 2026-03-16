import pickle
from pathlib import Path
from typing import Sequence

from pydantic import BaseModel


def save_jsonl(data: Sequence[BaseModel], path: str | Path):
    with open(str(path), "w") as f:
        for d in data:
            f.write(d.model_dump_json() + "\n")


def load_legacy_pickle(data_path: str, alias: dict[str, str], cls: BaseModel | None = None):
    import importlib
    import sys

    for old, new in alias.items():
        if old not in sys.modules:
            sys.modules[old] = importlib.import_module(new)

    data = pickle.load(open(data_path, "rb"))
    if cls is not None:
        data = [cls.model_validate_json(d.model_dump()) for d in data]
    return data
