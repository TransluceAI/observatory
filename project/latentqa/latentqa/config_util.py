import json
import os
from pathlib import Path
from typing import ClassVar

from pydantic import BaseModel


class BaseConfig(BaseModel):
    save_path: str | None = None

    @property
    def fields_to_exclude(self) -> set[str]:
        raise NotImplementedError("Subclasses must implement this method")

    config_file_name: ClassVar[str] = "config.json"

    @property
    def config_path(self):
        assert self.save_path is not None, "save_path is not set"
        return os.path.join(self.save_path, self.config_file_name)

    @classmethod
    def from_config_path(cls, config_path: Path | os.PathLike):
        with open(config_path, "r") as f:
            return cls.model_validate_json(json.load(f))

    def save_config(self):
        assert self.save_path is not None, "save_path is not set"
        os.makedirs(self.save_path, exist_ok=True)
        if os.path.exists(self.config_path):
            # Make sure the configs match.
            with open(self.config_path, "r") as f:
                existing_config = self.__class__.model_validate_json(json.load(f))
                existing_config_dict = existing_config.model_dump()
            curr_config_dict = self.model_dump()
            for key, value in curr_config_dict.items():
                if key in self.fields_to_exclude:
                    continue
                try:
                    assert existing_config_dict[key] == value
                except AssertionError:
                    print(f"Config mismatch for key: {key}")
                    print(f"Existing value: {existing_config_dict[key]}")
                    print(f"Current value: {value}")
                    raise
        else:
            with open(self.config_path, "w") as f:
                json.dump(self.model_dump_json(), f)
