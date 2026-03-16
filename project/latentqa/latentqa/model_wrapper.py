import contextlib
from collections import defaultdict
from typing import Any, Callable, Generic, Iterator, Optional, Sequence, TypeVar, Union

import torch
from peft import PeftModel
from pydantic import BaseModel, ConfigDict
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers.generation.utils import GenerationMixin
from transformers.modeling_outputs import CausalLMOutput
from transformers.modeling_utils import PreTrainedModel

T = TypeVar("T")


class BaseModelWithGenerate(PreTrainedModel, GenerationMixin):
    pass


class LayerModuleKey(BaseModel):
    layer: int
    module: str

    def __hash__(self):
        return hash((self.layer, self.module))

    def __eq__(self, other):
        if not isinstance(other, LayerModuleKey):
            return False
        return self.layer == other.layer and self.module == other.module

    def __lt__(self, other):
        if not isinstance(other, LayerModuleKey):
            return NotImplemented
        return (self.layer, self.module) < (other.layer, other.module)


class LayerModuleDict(BaseModel, Generic[T]):
    """Generic container for layer -> module -> item mappings"""

    data: dict[LayerModuleKey, T]

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, data: Optional[dict[LayerModuleKey, T]] = None, **kwargs):
        super().__init__(data=data or {}, **kwargs)

    def add(self, key: LayerModuleKey, item: T) -> None:
        """Add an item for the given layer and module"""
        self.data[key] = item

    def get(self, key: LayerModuleKey) -> Optional[T]:
        """Get an item for the given layer and module"""
        return self.data.get(key, None)

    def get_layer(self, layer: int) -> dict[LayerModuleKey, T]:
        """Get all items for a given layer as a dict[module_name, item]"""
        return {key: item for key, item in self.data.items() if key.layer == layer}

    def layers(self) -> set[int]:
        """Get all layer indices"""
        return {key.layer for key in self.data.keys()}

    def modules_for_layer(self, layer: int) -> set[str]:
        """Get all module names for a given layer"""
        return {key.module for key in self.data.keys() if key.layer == layer}

    def keys(self) -> list[LayerModuleKey]:
        """Get all layer module keys"""
        return sorted(self.data.keys())

    def items(self) -> Iterator[tuple[LayerModuleKey, T]]:
        """Iterate over (key, item) tuples"""
        for key, item in self.data.items():
            yield key, item

    def to_nested_dict(self) -> dict[int, dict[str, T]]:
        """Convert to the old nested dict format for backwards compatibility"""
        result: dict[int, dict[str, T]] = {}
        for key, item in self.items():
            if key.layer not in result:
                result[key.layer] = {}
            result[key.layer][key.module] = item
        return result

    @classmethod
    def from_nested_dict(cls, nested_dict: dict[int, dict[str, T]]) -> "LayerModuleDict[T]":
        """Create from the old nested dict format"""
        data = {}
        for layer, modules in nested_dict.items():
            for module, item in modules.items():
                data[LayerModuleKey(layer=layer, module=module)] = item
        return cls(data=data)

    def __len__(self) -> int:
        return len(self.data)

    def __bool__(self) -> bool:
        return bool(self.data)


class Activations(LayerModuleDict[torch.Tensor]):
    """
    Container for batched activations.
    Each tensor should be 3D: [batch_size, seq_len, hidden_dim]
    """

    def __init__(self, data: Optional[dict[LayerModuleKey, torch.Tensor]] = None, **kwargs):
        super().__init__(data=data, **kwargs)
        # Validate dimensions
        for key, tensor in self.data.items():
            if not isinstance(tensor, torch.Tensor):
                raise ValueError(
                    f"All values must be torch.Tensor, got {type(tensor)} for key {key}"
                )
            if tensor.dim() != 3:
                raise ValueError(
                    f"All tensors must be 3D [batch_size, seq_len, hidden_dim], "
                    f"got shape {tensor.shape} for key {key}"
                )
        # Validate batch size consistency
        self.batch_size()  # Will throw if inconsistent

    def slice(
        self, start_idx: Optional[int] = None, end_idx: Optional[int] = None
    ) -> "Activations":
        """Slice the activations by the given start and end indices"""
        if start_idx is None:
            start_idx = 0
        if end_idx is None:
            end_idx = self.batch_size()
        return Activations(
            data={key: tensor[start_idx:end_idx] for key, tensor in self.data.items()}
        )

    def add(self, key: LayerModuleKey, item: torch.Tensor) -> None:
        """Add a batched tensor for the given layer and module"""
        if not isinstance(item, torch.Tensor):
            raise ValueError(f"Value must be torch.Tensor, got {type(item)}")
        if item.dim() != 3:
            raise ValueError(
                f"Tensor must be 3D [batch_size, seq_len, hidden_dim], got shape {item.shape}"
            )
        super().add(key, item)
        # Validate batch size consistency
        self.batch_size()  # Will throw if inconsistent

    def get_i(self, key: LayerModuleKey, i: int) -> Optional[torch.Tensor]:
        """Get the ith sample from the batched tensor for the given key"""
        tensor = self.get(key)
        if tensor is None:
            return None
        if i >= tensor.size(0):
            raise IndexError(f"Index {i} out of range for batch size {tensor.size(0)}")
        return tensor[i]  # Returns 2D tensor [seq_len, hidden_dim]

    def batch_size(self) -> int:
        """Get the batch size (validates consistency across all tensors)"""
        if not self.data:
            return 0
        batch_sizes = {tensor.size(0) for tensor in self.data.values()}
        if len(batch_sizes) > 1:
            raise ValueError(f"Inconsistent batch sizes: {batch_sizes}")
        return batch_sizes.pop()

    def to_activations_list(self) -> "ActivationsList":
        """Convert batched activations to list of individual activations"""
        batch_size = self.batch_size()

        # Create list of individual tensors for each key
        list_data = {}
        for key, tensor in self.data.items():
            # Split batch dimension into list of 2D tensors
            list_data[key] = [tensor[i] for i in range(batch_size)]

        return ActivationsList(data=list_data)


class ActivationsList(LayerModuleDict[list[Optional[torch.Tensor]]]):
    """
    Container for lists of individual activations.
    Each list contains 2D tensors: [seq_len, hidden_dim]
    Elements can be None for missing activations in batch, and can have different sequence lengths.
    """

    def __init__(
        self, data: Optional[dict[LayerModuleKey, list[Optional[torch.Tensor]]]] = None, **kwargs
    ):
        super().__init__(data=data, **kwargs)
        # Validate dimensions
        for key, tensor_list in self.data.items():
            if not isinstance(tensor_list, list):
                raise ValueError(
                    f"All values must be list[torch.Tensor | None], got {type(tensor_list)} for key {key}"
                )
            # Validate tensor dimensions
            for i, tensor in enumerate(tensor_list):
                if tensor is not None:
                    if not isinstance(tensor, torch.Tensor):
                        raise ValueError(
                            f"All list items must be torch.Tensor or None, "
                            f"got {type(tensor)} at index {i} for key {key}"
                        )
                    if tensor.dim() != 2:
                        raise ValueError(
                            f"All tensors must be 2D [seq_len, hidden_dim], "
                            f"got shape {tensor.shape} at index {i} for key {key}"
                        )
        # Validate batch size consistency
        self.batch_size()  # Will throw if inconsistent

    def add(self, key: LayerModuleKey, item: list[Optional[torch.Tensor]]) -> None:
        """Add a list of tensors for the given layer and module"""
        if not isinstance(item, list):
            raise ValueError(f"Value must be list[torch.Tensor | None], got {type(item)}")

        # Validate tensor dimensions
        for i, tensor in enumerate(item):
            if tensor is not None:
                if not isinstance(tensor, torch.Tensor):
                    raise ValueError(
                        f"All list items must be torch.Tensor or None, got {type(tensor)} at index {i}"
                    )
                if tensor.dim() != 2:
                    raise ValueError(
                        f"All tensors must be 2D [seq_len, hidden_dim], "
                        f"got shape {tensor.shape} at index {i}"
                    )

        super().add(key, item)
        # Validate batch size consistency
        self.batch_size()  # Will throw if inconsistent

    def get_i(self, key: LayerModuleKey, i: int) -> Optional[torch.Tensor]:
        """Get the ith tensor from the list for the given key"""
        tensor_list = self.get(key)
        if tensor_list is None:
            return None
        if i >= len(tensor_list):
            raise IndexError(f"Index {i} out of range for list length {len(tensor_list)}")
        return tensor_list[i]  # Returns 2D tensor [seq_len, hidden_dim] or None

    def batch_size(self) -> int:
        """Get the batch size (validates consistency across all lists)"""
        if not self.data:
            return 0
        batch_sizes = {len(tensor_list) for tensor_list in self.data.values()}
        if len(batch_sizes) > 1:
            raise ValueError(f"Inconsistent batch sizes: {batch_sizes}")
        return batch_sizes.pop()

    def has_none_elements(self) -> bool:
        """Check if any elements in any list are None"""
        for tensor_list in self.data.values():
            if any(t is None for t in tensor_list):
                return True
        return False

    def none_mask(self) -> dict[LayerModuleKey, list[bool]]:
        """Get a mask indicating which elements are None for each key"""
        return {key: [t is None for t in tensor_list] for key, tensor_list in self.data.items()}

    def to_activations(self) -> Activations:
        """Convert list of activations to batched activations.

        Note: This will fail if any elements are None.
        Use only when all batch elements have activations.
        """
        if not self.data:
            return Activations()

        # Stack individual tensors into batched tensors
        batched_data = {}
        for key, tensor_list in self.data.items():
            if tensor_list:  # Only stack if list is not empty
                # Check for None elements
                if any(t is None for t in tensor_list):
                    raise ValueError(
                        f"Cannot convert to batched activations: key {key} has None elements"
                    )

                batched_data[key] = torch.stack(
                    [t for t in tensor_list if t is not None], dim=0
                )  # [batch_size, seq_len, hidden_dim]

        return Activations(data=batched_data)

    @classmethod
    def concat(cls, activations_lists: list["ActivationsList"]) -> "ActivationsList":
        """Concatenate multiple ActivationsList into a single ActivationsList"""
        if not activations_lists:
            return cls()

        # First check whether all the activations lists have the same keys.
        all_keys = set()
        for acts_list in activations_lists:
            all_keys.update(acts_list.data.keys())
        for acts_list in activations_lists:
            if set(acts_list.data.keys()) != all_keys:
                raise ValueError(
                    f"All ActivationsList objects must have the same keys. "
                    f"Expected keys: {all_keys}, got keys: {set(acts_list.data.keys())}"
                )

        # Single pass: build concatenated data directly
        concat_data = defaultdict(list)
        for acts_list in activations_lists:
            for key, tensor_list in acts_list.data.items():
                concat_data[key].extend(tensor_list)

        return cls(data=concat_data)

    def batch_slice(self, start_idx: int, end_idx: int) -> "ActivationsList":
        """
        Convenience method for batch slicing with the pattern start_idx : end_idx

        Args:
            batch_idx: The batch index to slice
            eval_batch_size: The size of each evaluation batch

        Returns:
            ActivationsList containing only the activations for the specified batch slice
        """
        sliced_data = {}
        for key, tensor_list in self.data.items():
            sliced_data[key] = tensor_list[start_idx:end_idx]

        return ActivationsList(data=sliced_data)

    @classmethod
    def from_sample_dicts(
        cls, sample_dicts: list[Optional[dict[LayerModuleKey, torch.Tensor]]]
    ) -> "ActivationsList":
        """Initialize ActivationsList from a list of per-sample activation dictionaries

        Args:
            sample_dicts: List where each element is either:
                - dict[LayerModuleKey, torch.Tensor]: activations for one sample
                - None: missing activations for one sample

        Returns:
            ActivationsList with structure dict[LayerModuleKey, list[torch.Tensor | None]]
        """
        if not sample_dicts:
            return cls()

        # Collect all keys across all samples
        all_keys = set()
        for sample_dict in sample_dicts:
            if sample_dict is not None:
                all_keys.update(sample_dict.keys())

        # Build lists for each key
        result_data = {}
        for key in all_keys:
            tensor_list = []
            for sample_dict in sample_dicts:
                if sample_dict is None:
                    tensor_list.append(None)
                else:
                    tensor_list.append(sample_dict.get(key, None))
            result_data[key] = tensor_list

        return cls(data=result_data)

    def tile(self, n: int) -> "ActivationsList":
        """
        Repeat the entire activation lists n times consecutively.

        This is equivalent to the pattern:
        [param[i] for _ in range(n) for i in range(len(param))]

        Args:
            n: Number of times to repeat the entire list

        Returns:
            ActivationsList with the entire list repeated n times consecutively

        Example:
            If original list is [a, b, c] and n=2, result is [a, b, c, a, b, c]
        """
        expanded_data = {}
        for key, tensor_list in self.data.items():
            # Repeat the entire list n times
            expanded_data[key] = tensor_list * n

        return ActivationsList(data=expanded_data)

    def repeat_interleave(self, n: int) -> "ActivationsList":
        """
        Repeat each element in the activation lists n times before moving to the next element.

        This is equivalent to the pattern:
        [param[i] for i in range(len(param)) for _ in range(n)]

        Args:
            n: Number of times to repeat each element

        Returns:
            ActivationsList with each element repeated n times consecutively

        Example:
            If original list is [a, b, c] and n=2, result is [a, a, b, b, c, c]
        """
        expanded_data = {}
        for key, tensor_list in self.data.items():
            # Repeat each element n times before moving to the next
            expanded_data[key] = [tensor for tensor in tensor_list for _ in range(n)]

        return ActivationsList(data=expanded_data)


Modules = LayerModuleDict[torch.nn.Module]


class ModelWrapper:

    def __init__(
        self,
        model: Union[BaseModelWithGenerate, PeftModel, DDP, FSDP],
        hf_model_id: str,
        device: str,
    ):
        self.model = model
        self.hf_model_id = hf_model_id
        self.device = device

    def get_module_eval_str(self, key: LayerModuleKey) -> str:
        prefix = "self.model"

        # Handle DDP wrapper first
        if hasattr(self.model, "module"):
            prefix += ".module"

            # Handle PeftModel wrapper
            if isinstance(self.model.module, PeftModel):
                prefix += ".model"
        elif isinstance(self.model, PeftModel):
            prefix += ".model"

        if self.hf_model_id in [
            "meta-llama/Llama-3.1-8B-Instruct",
            "meta-llama/Llama-3.1-70B-Instruct",
        ]:
            if key.module == "residual":
                return f"{prefix}.model.layers[{key.layer}]"
            elif key.module in ["k_proj", "v_proj"]:
                return f"{prefix}.model.layers[{key.layer}].self_attn.{key.module}"
            elif key.module == "emb":
                assert key.layer == -1, "Embedding layer should be at layer -1"
                return f"{prefix}.model.embed_tokens"
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

    def get_modules(self, keys: Sequence[LayerModuleKey]) -> Modules:
        result = Modules()
        for key in keys:
            try:
                module = eval(self.get_module_eval_str(key))
            except Exception as e:
                print(self.get_module_eval_str(key))
                raise ValueError(f"Failed to get module for key {key}: {e}")
            result.add(key, module)
        return result

    def get_activations_and_logits(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        read_modules: Modules,
        device: Optional[str] = None,
        inference_mode: bool = True,
    ) -> tuple[Activations, torch.Tensor]:
        if device is None:
            device = self.device

        activations_by_layer = {key: None for key in read_modules.keys()}

        def get_hook_by_layer(key: LayerModuleKey):
            def _hook(module, input, output):
                if isinstance(output, tuple):
                    output = output[0]
                activations_by_layer[key] = output
                return None

            return _hook

        hook_handles = []
        for key, module in read_modules.items():
            hook_handles.append(module.register_forward_hook(get_hook_by_layer(key)))

        ctx = torch.no_grad() if inference_mode else contextlib.nullcontext()
        with ctx:
            out = self.model(
                input_ids=input_ids.to(self.device),
                attention_mask=attention_mask.to(self.device),
                use_cache=False,
            )

        for hook_handle in hook_handles:
            hook_handle.remove()

        activations = Activations()
        for key, act in activations_by_layer.items():
            assert act is not None
            activations.add(key, act.to(self.device))

        predictions = out.logits

        return activations, predictions

    def get_activations(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        read_modules: Modules,
        device: Optional[str] = None,
        inference_mode: bool = True,
    ) -> Activations:
        activations, _ = self.get_activations_and_logits(
            input_ids, attention_mask, read_modules, device, inference_mode=inference_mode
        )
        return activations

    def _execute_with_intervention(
        self,
        activations: Union[Activations, ActivationsList],
        token_range_from: list[tuple[int, int]],
        token_range_to: list[tuple[int, int]],
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        write_modules: Modules,
        layer_mapping: dict[int, list[int]],
        device: str,
        model_call: Callable[[dict], Any],
    ) -> Any:
        """Helper method to execute model with activation interventions."""

        def get_hook_by_layer(key: LayerModuleKey, acts: Union[torch.Tensor, list[torch.Tensor]]):
            def _hook(module, input, output):
                y = output[0] if isinstance(output, tuple) else output
                for i in range(y.size(0)):
                    y[i, token_range_to[i][0] : token_range_to[i][1]] = acts[i].to(y.device)
                return (y,) + output[1:] if isinstance(output, tuple) else y

            return _hook

        hook_handles = []

        for key, module in write_modules.items():
            read_layers = layer_mapping[key.layer]

            acts_dict = defaultdict(list)
            for read_layer in read_layers:
                read_key = LayerModuleKey(layer=read_layer, module=key.module)
                acts = activations.get(read_key)
                assert (
                    acts is not None
                ), f"No activations found for layer {read_layer}, module {key.module}"
                assert len(acts) == len(token_range_from)

                for i, acts_i in enumerate(acts):
                    assert acts_i is not None
                    start, end = token_range_from[i]
                    acts_dict[i].append(acts_i[start:end])

            acts_list = [torch.cat(acts_dict[i], dim=0) for i in range(len(acts_dict))]

            hook_handles.append(module.register_forward_hook(get_hook_by_layer(key, acts_list)))
        try:
            dec_in = {
                "input_ids": input_ids.to(device),
                "attention_mask": attention_mask.to(device),
            }
            return model_call(dec_in)
        finally:
            for hook_handle in hook_handles:
                hook_handle.remove()

    def forward_with_intervention(
        self,
        activations: Union[Activations, ActivationsList],
        token_range_from: list[tuple[int, int]],
        token_range_to: list[tuple[int, int]],
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        write_modules: Modules,
        layer_mapping: dict[int, list[int]],
        device: str,
    ) -> CausalLMOutput:
        def model_call(dec_in: dict) -> CausalLMOutput:
            return self.model(**dec_in, use_cache=False)

        return self._execute_with_intervention(
            activations,
            token_range_from,
            token_range_to,
            input_ids,
            attention_mask,
            write_modules,
            layer_mapping,
            device,
            model_call,
        )

    def generate_with_intervention(
        self,
        activations: Union[Activations, ActivationsList],
        token_range_from: list[tuple[int, int]],
        token_range_to: list[tuple[int, int]],
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        write_modules: Modules,
        layer_mapping: dict[int, list[int]],
        device: str,
        max_new_tokens: int = 128,
        pad_token_id: int = -100,
        temperature: float = 1.0,
        num_return_sequences: int = 1,
    ) -> torch.LongTensor:
        def model_call(dec_in: dict) -> torch.LongTensor:
            out = self.model.generate(
                **dec_in,
                use_cache=False,
                max_new_tokens=max_new_tokens,
                pad_token_id=pad_token_id,
                temperature=temperature,
                num_return_sequences=num_return_sequences,
            )
            return out

        return self._execute_with_intervention(
            activations,
            token_range_from,
            token_range_to,
            input_ids,
            attention_mask,
            write_modules,
            layer_mapping,
            device,
            model_call,
        )
