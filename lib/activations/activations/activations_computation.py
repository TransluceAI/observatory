from enum import Enum
from typing import Callable

import torch
from nnsight.envoy import Envoy  # type: ignore
from nnsight.intervention import InterventionProxy  # type: ignore
from util.subject import Subject


class ActivationType(str, Enum):
    RESID = "resid"
    MLP_IN = "mlp_in"
    MLP_OUT = "mlp_out"
    ATTN_OUT = "attn_out"
    NEURONS = "neurons"


def _get_activations_funcs(
    subject: Subject, activation_type: ActivationType, layer: int
) -> tuple[Callable[[], Envoy], Callable[[Envoy], InterventionProxy]]:
    if activation_type == ActivationType.RESID:
        return (
            lambda: subject.layers[layer],
            lambda component: component.output[0],
        )
    if activation_type == ActivationType.MLP_IN:
        return (
            lambda: subject.mlps[layer],
            lambda component: component.input,
        )
    if activation_type == ActivationType.MLP_OUT:
        return (
            lambda: subject.mlps[layer],
            lambda component: component.output,
        )
    if activation_type == ActivationType.ATTN_OUT:
        return (
            lambda: subject.attns[layer],
            lambda component: component.output[0],
        )
    if activation_type == ActivationType.NEURONS:
        return (
            lambda: subject.w_outs[layer],
            lambda component: component.input,
        )
    raise ValueError(f"Unknown activation type: {activation_type}")


def get_activations_computing_func(
    subject: Subject, activation_type: ActivationType, layer: int
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """
    Returns a function that computes activations for a given input:
    input_ids: torch.Tensor
    attn_mask: torch.Tensor

    """
    get_component, get_activations = _get_activations_funcs(subject, activation_type, layer)

    def activations_computing_func(
        input_ids: torch.Tensor, attn_mask: torch.Tensor
    ) -> torch.Tensor:
        with torch.no_grad():
            with subject.model.trace(
                {"input_ids": input_ids, "attention_mask": attn_mask}  # type: ignore
            ):
                acts: torch.Tensor = get_activations(get_component()).save()  # type: ignore
        return acts

    return activations_computing_func
