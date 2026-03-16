import gc
import json
import os
import pickle
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Any, Callable, Literal, cast

import numpy as np
import pandas as pd
import torch
from latentqa.config_util import BaseConfig
from latentqa.data_util import MinimalDatapoint, PatchFromRegion, prepare_input_for_subject
from latentqa.eval.steering_eval import evaluate_control, evaluate_reading_after_steering
from latentqa.eval.steering_util import (
    GenerationBasedControlEvalConfig,
    SteeringInput,
    get_test_mc_qas,
)
from latentqa.lqa_data import LQAData, load_lqa_data
from latentqa.model_wrapper import (
    Activations,
    ActivationsList,
    LayerModuleKey,
    ModelWrapper,
    Modules,
)
from latentqa.train_util import (
    LQATrainConfig,
    get_dataloader,
    get_decoder_model,
    get_subject_model,
    get_tokenizer,
)
from neurondb.filters import FILTER_QTILE_TYPE, QTILE_KEYS, QTILE_MAP, NeuronPolarity
from neurondb.postgres import DBManager
from neurondb.view import NeuronView
from peft import PeftModel
from torch import Tensor
from tqdm import tqdm
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizer
from user_modeling.self_describe import (
    ATTRIBUTE_PROMPTS,
    WIKIPEDIA_PROMPT_REQUEST,
    WIKIPEDIA_PROMPT_RESPONSE,
)
from user_modeling.user_attribute import UserAttribute, resolve_attr
from util.subject import Subject, llama31_8B_instruct_config


def get_self_description_from_conversation(
    conversation: list[dict],
    tokenizer: PreTrainedTokenizer,
    chat_template: str,
    input_ids: list[int],
    read_region: Literal["prefix", "entire_user", "wiki"],
) -> tuple[int, int]:
    assert len(conversation) <= 3

    if conversation[0]["role"] == "system":
        system_prompt = conversation[0]["content"]
        user_prompt = conversation[1]["content"]
    else:
        assert conversation[0]["role"] == "user"
        system_prompt = None
        user_prompt = conversation[0]["content"]
    if conversation[-1]["role"] == "assistant":
        conversation[-1]["content"]
    else:
        pass

    assert user_prompt.endswith(WIKIPEDIA_PROMPT_REQUEST)
    prefix = user_prompt.removesuffix(WIKIPEDIA_PROMPT_REQUEST).strip()

    conversation_read_range = (1, 2) if system_prompt else (0, 1)
    ids, read_range = prepare_input_for_subject(
        conversation=conversation,
        conversation_read_range=conversation_read_range,
        tokenizer=tokenizer,
        subject_chat_template=chat_template,
    )
    assert ids == input_ids

    if read_region == "prefix":
        up_till_prefix = tokenizer.apply_chat_template(
            conversation=conversation[: conversation_read_range[0]]
            + [{"role": "user", "content": prefix}],
            chat_template=chat_template,
            continue_final_message=True,
            return_dict=False,
        )
        prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
        read_range = (len(up_till_prefix) - len(prefix_ids), len(up_till_prefix))

        # Double check.
        prefix_decoded = tokenizer.decode(ids[read_range[0] : read_range[1]])
        assert prefix_decoded == prefix

    elif read_region == "wiki":
        up_till_prefix = tokenizer.apply_chat_template(
            conversation=conversation[: conversation_read_range[0]]
            + [{"role": "user", "content": prefix}],
            chat_template=chat_template,
            continue_final_message=True,
            return_dict=False,
        )
        prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
        read_range = (len(up_till_prefix), read_range[1])

    else:
        assert read_region == "entire_user"

    return read_range


class CircuitBasedSteeringEvalConfig(BaseConfig):
    subject_inputs_data_path: str
    neurons_path: str
    target_attr: str
    true_attrs: list[str]
    steer_region: Literal["prefix", "entire_user", "wiki"] = "prefix"
    read_region: Literal["prefix", "entire_user", "wiki"] = "wiki"
    seed_response_type: Literal["wiki", "none"]

    exp_path: str
    ckpt_name: str
    max_subject_inputs_per_attr_pair: int = 10

    steering_quantile: FILTER_QTILE_TYPE = "1e-7"
    steering_strengths: list[float]

    mc_stems_path: str

    generate_to_eval_control: bool = True
    forward_batch_size: int = 128
    gen_based_control_eval_config: GenerationBasedControlEvalConfig | None = (
        GenerationBasedControlEvalConfig(
            max_new_tokens=256,
            temperature=1.0,
            num_completions=5,
            llm_judge_provider="openai",
            llm_judge_model_name="gpt-4.1-mini",
            llm_judge_max_concurrency=100,
            llm_judge_timeout=100.0,
        )
    )
    seed: int = 42
    si_batch_size: int = 32

    @property
    def fields_to_exclude(self):
        return {
            "true_attrs",
            "max_subject_inputs_per_attr_pair",
            "generate_to_eval_control",
            "forward_batch_size",
            "control_eval_methods",
            "gen_based_control_eval_config",
            "si_batch_size",
        }

    def results_path(self, eval_type: Literal["reading", "control"]):
        if not self.save_path:
            return None
        if eval_type == "control":
            file_name = "gen_control" if self.generate_to_eval_control else "fixed_control"
        else:
            file_name = "reading"
        return os.path.join(self.save_path, f"{file_name}.pkl")

    def save_results(
        self,
        results: dict[SteeringInput, list[dict[str, Any]]],
        eval_type: Literal["reading", "control"],
    ):
        path = self.results_path(eval_type)
        if not path:
            print("save_path is not set, skipping saving results")
            return
        with open(path, "wb") as f:
            pickle.dump(results, f)

    def load_results(
        self, eval_type: Literal["reading", "control"]
    ) -> dict[SteeringInput, list[dict[str, Any]]] | None:
        path = self.results_path(eval_type)
        if not path or not os.path.exists(path):
            return None
        with open(path, "rb") as f:
            return pickle.load(f)

    def get_folder_name(self):
        components = [
            f"Q{self.steering_quantile}",
            f"steer_{self.steer_region}",
            f"read_{self.read_region}",
            f"seed_response_{self.seed_response_type}",
        ]
        return "_".join(components)

    def get_neurons(self) -> list[tuple[int, int, NeuronPolarity]]:
        with open(self.neurons_path, "rb") as f:
            neurons = pickle.load(f)
        return neurons


def get_neuron_interventions(
    neurons: list[tuple[int, int, NeuronPolarity]],
    token_ranges: list[tuple[int, int]],
    percentiles_PLI: dict[str, np.ndarray],
    strength: float = 0.0,
    steering_quantile: FILTER_QTILE_TYPE = "1e-7",
    flip_polarity: bool = False,
):
    interventions: dict[tuple[int, int, int], float] = {}
    for layer, neuron_idx, polarity in neurons:
        # Flip polarity if requested
        intervention_polarity = (
            NeuronPolarity.NEG
            if (flip_polarity and polarity == NeuronPolarity.POS)
            else (
                NeuronPolarity.POS
                if (flip_polarity and polarity == NeuronPolarity.NEG)
                else polarity
            )
        )

        quantile = percentiles_PLI[
            (
                QTILE_MAP[steering_quantile][0]
                if intervention_polarity == NeuronPolarity.POS
                else QTILE_MAP[steering_quantile][1]
            )
        ][layer, neuron_idx]
        # Check that the quantile matches the intervention polarity.
        # This check can fail for neurons that are active for one sign
        # (i.e. always positive or always negative).
        if intervention_polarity == NeuronPolarity.POS and quantile < 0:
            print(
                f"Skipping Neuron({layer}, {neuron_idx}, {polarity.value}) because quantile ({quantile}) is negative for POS polarity"
            )
            continue
        if intervention_polarity == NeuronPolarity.NEG and quantile > 0:
            print(
                f"Skipping Neuron({layer}, {neuron_idx}, {polarity.value}) because quantile ({quantile}) is positive for NEG polarity"
            )
            continue

        for ts, te in token_ranges:
            for t in range(ts, te):
                interventions[(layer, t, neuron_idx)] = quantile * strength
    return interventions


def do_neuron_interventions_and_get_activations(
    subject: PreTrainedModel,
    input_ids: Tensor,  # (batch_size, seq_len)
    attention_mask: Tensor,  # (batch_size, seq_len)
    intervention_tensors_by_layer: dict[int, tuple[Tensor, Tensor, Tensor, Tensor]],
    read_modules: Modules,
    device: str = "cuda:0",
) -> dict[LayerModuleKey, Tensor]:
    def get_intervention_hook_by_layer(layer: int):
        def _hook(module, input):
            if isinstance(input, tuple):
                input = input[0]

            input_idxs, tokens, neurons, values = intervention_tensors_by_layer[layer]
            input[input_idxs, tokens, neurons] = values.to(input.device)
            return input

        return _hook

    activations_by_layer: dict[LayerModuleKey, Tensor | None] = {
        key: None for key in read_modules.keys()
    }

    def get_reading_hook_by_layer(key: LayerModuleKey):
        def _hook(module, input, output):
            if isinstance(output, tuple):
                output = output[0]

            activations_by_layer[key] = output
            return None

        return _hook

    intervention_hook_handles = []
    for layer in intervention_tensors_by_layer:
        intervention_module = subject.model.layers[layer].mlp.down_proj
        intervention_hook_handles.append(
            intervention_module.register_forward_pre_hook(get_intervention_hook_by_layer(layer))
        )
    read_hook_handles = []
    for key, module in read_modules.items():
        read_hook_handles.append(module.register_forward_hook(get_reading_hook_by_layer(key)))

    with torch.no_grad():
        _ = subject.forward(
            input_ids=input_ids.to(device), attention_mask=attention_mask.to(device)
        )

    for hook_handle in intervention_hook_handles + read_hook_handles:
        hook_handle.remove()

    # Check that everything is not None.
    activations_by_layer_upd: dict[LayerModuleKey, Tensor] = {}
    for key, activation in activations_by_layer.items():
        assert activation is not None
        activations_by_layer_upd[key] = activation
    return activations_by_layer_upd


def get_activation_checkpoints_list_from_circuit_based_steering(
    steering_inputs: list[SteeringInput],
    neurons_list: list[tuple[int, int, NeuronPolarity]],
    read_region: Literal["prefix", "entire_user", "wiki"],
    exp_config: LQATrainConfig,
    subject: PreTrainedModel,
    read_modules: Modules,
    tokenizer: PreTrainedTokenizer,
    steering_quantile: FILTER_QTILE_TYPE,
    steering_strengths: list[float],
    percentiles_pli: dict[str, np.ndarray],
    custom_steer_range_fn: Callable[[list[dict]], tuple[int, int]],
    custom_read_range_fn: Callable[[list[dict]], tuple[int, int]],
    batch_size: int = 128,
    device: str = "cuda:0",
):
    # Extract only the part that pertains to the subject model input from steering inputs.
    subject_inputs = []
    for si in steering_inputs:
        subject_inputs.append(
            MinimalDatapoint(
                attr=si.target_attr,  # This is not used here, but just setting it.
                system_prompt=si.system_prompt,
                user_prompt=si.user_prompt,
                # The following are now used here.
                question="",
                answer="",
            )
        )

    common_kwargs = {
        "data": subject_inputs,
        "hf_model_id": exp_config.hf_model_id,
        "tokenizer": tokenizer,
        "generate": False,
        "shuffle": False,
        "patch_from_regions": [PatchFromRegion.CUSTOM],
        "batch_size": batch_size,
        "drop_last": False,
    }
    steer_dataloader = get_dataloader(**common_kwargs, custom_read_range_fn=custom_steer_range_fn)
    read_dataloader = get_dataloader(**common_kwargs, custom_read_range_fn=custom_read_range_fn)

    activations_by_strength: dict[float, list[tuple[dict[str, Any], tuple[int, int]]]] = (
        defaultdict(list)
    )
    for strength in tqdm(steering_strengths, desc="Getting activations by strength..."):
        for steer_batch, read_batch in zip(steer_dataloader, read_dataloader):
            input_ids = steer_batch.subject_input_ids
            attention_mask = steer_batch.subject_attention_mask

            # Group interventions by layer
            interventions_by_layer: dict[int, list[tuple[int, int, int, float]]] = defaultdict(list)
            for input_idx, input_read_range in enumerate(steer_batch.token_range_from):
                interventions_for_input = get_neuron_interventions(
                    neurons=neurons_list,
                    token_ranges=[input_read_range],
                    percentiles_PLI=percentiles_pli,
                    strength=strength,
                    steering_quantile=steering_quantile,
                    flip_polarity=False,
                )
                for (layer, token, neuron), value in interventions_for_input.items():
                    interventions_by_layer[layer].append((input_idx, token, neuron, value))
            # Convert to tensors for easier indexing.
            intervention_tensors_by_layer = {}
            for layer, intervs in interventions_by_layer.items():
                input_idxs, tokens, neurons, values = zip(*intervs)
                intervention_tensors_by_layer[layer] = (
                    torch.tensor(input_idxs),
                    torch.tensor(tokens),
                    torch.tensor(neurons),
                    torch.tensor(values, dtype=torch.bfloat16),
                )

            activations_by_layer = do_neuron_interventions_and_get_activations(
                subject=subject,
                input_ids=input_ids,
                attention_mask=attention_mask,
                intervention_tensors_by_layer=intervention_tensors_by_layer,
                read_modules=read_modules,
                device=device,
            )

            batch_size = input_ids.size(0)
            num_paddings = (1 - attention_mask).sum(dim=1)
            for input_idx in range(batch_size):
                num_padding = num_paddings[input_idx].item()

                start, end = read_batch.token_range_from[input_idx]
                input_read_range = (start - num_padding, end - num_padding)  # Get rid of padding.

                acts_for_input = Activations()
                for key, activation in activations_by_layer.items():
                    acts = activation[input_idx]
                    acts = acts[num_padding:]  # Get rid of padding.
                    acts_for_input.add(key, acts[None, :])

                activations_by_strength[strength].append(
                    ({"activations": acts_for_input, "strength": strength}, input_read_range)
                )

    # Gather checkpoints that share the same input.
    checkpoints_by_steering_input: dict[
        SteeringInput, tuple[list[dict[str, Any]], tuple[int, int]]
    ] = {}
    for idx, steering_input in enumerate(steering_inputs):
        checkpoints = []
        read_ranges = []
        for strength in steering_strengths:
            ckpt, read_range = activations_by_strength[strength][idx]
            checkpoints.append(ckpt)
            read_ranges.append(read_range)
        assert len(set(read_ranges)) == 1
        checkpoints_by_steering_input[steering_input] = (checkpoints, read_ranges[0])
    return checkpoints_by_steering_input


async def do_circuit_based_steering_on_batch(
    sis_to_get_activations_for: list[SteeringInput],
    config: CircuitBasedSteeringEvalConfig,
    exp_config: LQATrainConfig,
    wrapped_subject: ModelWrapper,
    wrapped_decoder: ModelWrapper,
    tokenizer: PreTrainedTokenizer,
    percentiles_pli: dict[str, np.ndarray],
    custom_steer_range_fn: Callable[[list[dict]], tuple[int, int]],
    custom_read_range_fn: Callable[[list[dict]], tuple[int, int]],
    save_dir: Path | None = None,
):
    checkpoints_by_steering_input = get_activation_checkpoints_list_from_circuit_based_steering(
        steering_inputs=sis_to_get_activations_for,
        neurons_list=config.get_neurons(),
        read_region=config.read_region,
        exp_config=exp_config,
        subject=wrapped_subject.model,
        read_modules=wrapped_subject.get_modules(exp_config.get_read_layer_module_keys()),
        tokenizer=tokenizer,
        steering_quantile=config.steering_quantile,
        steering_strengths=config.steering_strengths,
        percentiles_pli=percentiles_pli,
        custom_steer_range_fn=custom_steer_range_fn,
        custom_read_range_fn=custom_read_range_fn,
        batch_size=config.forward_batch_size,
        device="cuda:0",
    )

    #########################################################
    # Do steering eval.
    #########################################################

    for eval_type in ["reading", "control"]:
        existing_eval_results = config.load_results(eval_type)
        # Load pre-existing eval results.
        if existing_eval_results is None:
            existing_eval_results = {}

        for steering_input in tqdm(
            sis_to_get_activations_for,
            desc=f"Evaluating {eval_type}...",
        ):
            if steering_input in existing_eval_results:
                continue
            checkpoints, token_range_from = checkpoints_by_steering_input[steering_input]

            all_activations = ActivationsList.concat(
                [ckpt["activations"].to_activations_list() for ckpt in checkpoints]
            )
            assert all_activations.batch_size() == len(checkpoints)

            if eval_type == "reading":
                eval_results = evaluate_reading_after_steering(
                    steering_input=steering_input,
                    token_range_from=token_range_from,
                    all_activations=all_activations,
                    patch_from_region=PatchFromRegion.CUSTOM,
                    exp_config=exp_config,
                    wrapped_decoder=wrapped_decoder,
                    tokenizer=tokenizer,
                    custom_read_range_fn=custom_read_range_fn,
                    forward_batch_size=config.forward_batch_size,
                    device="cuda:0",
                )
            elif eval_type == "control":
                eval_results = await evaluate_control(
                    steering_input=steering_input,
                    all_activations=all_activations,
                    patch_from_region=PatchFromRegion.CUSTOM,
                    exp_config=exp_config,
                    wrapped_subject=wrapped_subject,
                    tokenizer=tokenizer,
                    generate_to_eval_control=config.generate_to_eval_control,
                    forward_batch_size=config.forward_batch_size,
                    gen_based_control_eval_config=config.gen_based_control_eval_config,
                    custom_read_range_fn=custom_read_range_fn,
                    device="cuda:0",
                    verbose=False,
                )
            else:
                raise ValueError(f"Invalid eval_type: {eval_type}")

            existing_eval_results[steering_input] = eval_results

            if save_dir is not None:
                config.save_results(existing_eval_results, eval_type)


async def do_circuit_based_steering_eval(
    config: CircuitBasedSteeringEvalConfig,
    subject: PreTrainedModel | None = None,
    decoder: PeftModel | None = None,
    percentiles_pli: dict[str, np.ndarray] | None = None,  # If None, load from DB.
):
    assert config.save_path is not None
    save_dir = Path(config.save_path)
    save_dir.mkdir(parents=True, exist_ok=True)

    print("saving config")
    config.save_config()
    print("#" * 100)
    print(f"Running steering for {config.exp_path}/{config.ckpt_name}...")
    print("#" * 100)

    np.random.seed(config.seed)

    # Load the experiment config.
    with open(Path(config.exp_path) / LQATrainConfig.config_file_name, "r") as f:
        exp_config = LQATrainConfig.model_validate_json(json.load(f))
    tokenizer = get_tokenizer(exp_config.hf_model_id, exp_config.new_context_tokens)

    #########################################################
    # Get steering inputs ready.
    #########################################################

    all_subject_inputs = load_lqa_data(config.subject_inputs_data_path, LQAData)
    subject_inputs_by_attr = defaultdict(set)
    for d in all_subject_inputs:
        subject_inputs_by_attr[d.attr].add((d.system_prompt, d.user_prompt))

    # This is required for getting the reading range for each steering input.
    custom_steer_range_fn = partial(
        get_self_description_from_conversation, read_region=config.steer_region  # type: ignore
    )
    custom_read_range_fn = partial(
        get_self_description_from_conversation, read_region=config.read_region  # type: ignore
    )

    # Get things ready for generating test qas.
    stems_by_attr_class = pickle.load(open(config.mc_stems_path, "rb"))

    # Collect all steering inputs.
    steering_inputs = []
    for true_attr_str in config.true_attrs:
        true_attr = resolve_attr(true_attr_str)
        target_attr = resolve_attr(config.target_attr)
        assert true_attr.__class__ == target_attr.__class__
        attr_class = true_attr.__class__

        subject_inputs = sorted(list(subject_inputs_by_attr[true_attr]))
        subject_inputs = np.random.permutation(subject_inputs)
        subject_inputs = subject_inputs[: config.max_subject_inputs_per_attr_pair]

        def get_answers(attr: UserAttribute) -> list[str]:
            return list(
                set(
                    [
                        " " + attr.value_name.capitalize(),
                        " " + attr.value_name.title(),
                        " " + attr.value_name.lower(),
                    ]
                )
            )

        if config.seed_response_type == "wiki":
            seed_response = (
                WIKIPEDIA_PROMPT_RESPONSE + ATTRIBUTE_PROMPTS[attr_class.__name__.lower()]
            )
            possible_subject_responses = [(attr, get_answers(attr)) for attr in attr_class]
        elif config.seed_response_type == "none":
            seed_response = None
            possible_subject_responses = None

        for system_prompt, user_prompt in subject_inputs:
            steering_input = SteeringInput(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                true_attr=true_attr,
                target_attr=target_attr,
                steering_qas=None,
                subject_seed_response=seed_response,
                expected_subject_responses=possible_subject_responses,
                # Fixing the seed here makes us always get the same same set of multiple choice
                # questions given a fixed attribute class.
                test_qas=tuple(
                    get_test_mc_qas(
                        true_attr,
                        target_attr,
                        stems_by_attr_class=stems_by_attr_class,
                        rng=np.random.default_rng(0),
                    )
                ),
            )

            steering_inputs.append(steering_input)

    #########################################################
    # Load models.
    #########################################################
    if subject is None:
        subject = cast(
            PreTrainedModel,
            get_subject_model(
                hf_model_id=exp_config.hf_model_id,
                use_fsdp=False,
                device="auto",
                inference_only=True,
                fsdp_config=None,
            ),
        )
    if decoder is None:
        decoder = cast(
            PeftModel,
            get_decoder_model(
                hf_model_id=exp_config.hf_model_id,
                tokenizer=tokenizer,
                use_peft=exp_config.use_peft,
                use_fsdp=False,
                device="auto",
                decoder_checkpoint_path=str(
                    Path(config.exp_path) / "checkpoints" / config.ckpt_name
                ),
                fsdp_config=None,
                lora_config=exp_config.lora_config,
            ),
        )

    wrapped_subject = ModelWrapper(subject, exp_config.hf_model_id, "cuda:0")
    wrapped_decoder = ModelWrapper(decoder, exp_config.hf_model_id, "cuda:0")

    db = DBManager.get_instance("neurons_vincent")
    subject = Subject(llama31_8B_instruct_config)
    percentiles_pli = NeuronView.load_percentiles(db, subject, QTILE_KEYS)

    #########################################################
    # Go through any existing eval results and collect new steering inputs to evaluate.
    #########################################################
    reading_eval_results = config.load_results("reading")
    control_eval_results = config.load_results("control")

    if reading_eval_results is not None and control_eval_results is not None:
        already_evaluated_sis = set(reading_eval_results.keys()).intersection(
            set(control_eval_results.keys())
        )
        sis_to_get_activations_for = list(set(steering_inputs) - already_evaluated_sis)
    else:
        sis_to_get_activations_for = steering_inputs

    print(f"Going to evaluate {len(sis_to_get_activations_for)} steering inputs.")

    #########################################################
    # Go through steering inputs in batches to save memory.
    #########################################################
    for start in tqdm(
        range(0, len(sis_to_get_activations_for), config.si_batch_size),
        desc="Going through steering inputs in batches...",
    ):
        sis = sis_to_get_activations_for[start : start + config.si_batch_size]
        await do_circuit_based_steering_on_batch(
            sis_to_get_activations_for=sis,
            config=config,
            exp_config=exp_config,
            wrapped_subject=wrapped_subject,
            wrapped_decoder=wrapped_decoder,
            tokenizer=tokenizer,
            percentiles_pli=percentiles_pli,
            custom_steer_range_fn=custom_steer_range_fn,
            custom_read_range_fn=custom_read_range_fn,
            save_dir=save_dir,
        )
        gc.collect()
        torch.cuda.empty_cache()

    del subject
    del decoder
    gc.collect()
    torch.cuda.empty_cache()


def parse_circuit_based_steering_eval_results(
    reading_results: dict[SteeringInput, list[dict[str, Any]]],
    fixed_control_results: dict[SteeringInput, list[dict[str, Any]]] | None,
    gen_control_results: dict[SteeringInput, list[dict[str, Any]]] | None = None,
) -> pd.DataFrame:
    steering_strengths = (np.arange(10) / 10).tolist()

    all_results = []
    for steering_input_idx, steering_input in enumerate(reading_results.keys()):
        attr_class = steering_input.attr_class
        true_attr = steering_input.true_attr
        target_attr = steering_input.target_attr

        true_attr_idx = list(attr_class).index(true_attr)
        target_attr_idx = list(attr_class).index(target_attr)

        reading_for_si = reading_results[steering_input]
        num_ckpts = len(reading_for_si)
        assert num_ckpts == len(steering_strengths)

        fixed_control_for_si = (
            fixed_control_results[steering_input]
            if fixed_control_results is not None
            else [None] * num_ckpts
        )
        gen_control_for_si = (
            gen_control_results[steering_input]
            if gen_control_results is not None
            else [None] * num_ckpts
        )

        for strength, reading, fixed_control, gen_control in zip(
            steering_strengths, reading_for_si, fixed_control_for_si, gen_control_for_si
        ):
            parsed_eval_result = {}
            parsed_eval_result.update(
                {
                    "tag": steering_input_idx,
                    "strength": strength,
                    "true_attr": true_attr,
                    "target_attr": target_attr,
                    # Reading
                    "reading_acc_target": reading["decoder_accs"][target_attr_idx],
                    "reading_acc_orig": reading["decoder_accs"][true_attr_idx],
                    "reading_prob_target": reading["decoder_probs"][target_attr_idx],
                    "reading_prob_orig": reading["decoder_probs"][true_attr_idx],
                    "reading_norm_prob_target": reading["decoder_norm_probs"][target_attr_idx],
                    "reading_norm_prob_orig": reading["decoder_norm_probs"][true_attr_idx],
                }
            )
            for control_type, control in [("fixed", fixed_control), ("gen", gen_control)]:
                if control is None:
                    continue
                parsed_eval_result.update(
                    {
                        f"{control_type}_acc_target": control["subject_accs"][target_attr_idx],
                        f"{control_type}_acc_orig": control["subject_accs"][true_attr_idx],
                        f"{control_type}_prob_target": control["subject_probs"][target_attr_idx],
                        f"{control_type}_prob_orig": control["subject_probs"][true_attr_idx],
                        f"{control_type}_norm_prob_target": control["subject_norm_probs"][
                            target_attr_idx
                        ],
                        f"{control_type}_norm_prob_orig": control["subject_norm_probs"][
                            true_attr_idx
                        ],
                    }
                )
                if "most_likely_first_token" in control:
                    parsed_eval_result["most_likely_first_token"] = control[
                        "most_likely_first_token"
                    ]
                if "subject_completions" in control:
                    parsed_eval_result["completions"] = control["subject_completions"]
            all_results.append(parsed_eval_result)
    return pd.DataFrame(all_results)
