import torch
from log_util import get_logger
from transformers import AutoModelForCausalLM, AutoTokenizer
from user_modeling.chat_templates import (
    STRIPPED_LLAMA31_CHAT_TEMPLATE,
    STRIPPED_LLAMA31_REFLECT_CHAT_TEMPLATE,
    STRIPPED_LLAMA31_RESPONSE_CHAT_TEMPLATE,
)

logger = get_logger(__name__)

MODEL_TYPES = {
    "meta-llama/Llama-3.1-8B-Instruct": "llama31",
    "meta-llama/Llama-3.1-70B-Instruct": "llama31",
}

CHAT_PREFIXES = {
    "llama31": (
        128000,
        128006,
        9125,
        128007,
        271,
        128009,
        128006,
        882,
        128007,
        271,
    ),
}

CHAT_TEMPLATES = {
    "llama31": {
        "default": STRIPPED_LLAMA31_CHAT_TEMPLATE,
        "reflect": STRIPPED_LLAMA31_REFLECT_CHAT_TEMPLATE,
        "response": STRIPPED_LLAMA31_RESPONSE_CHAT_TEMPLATE,
    },
}

OLD_LATENTQA_CHAT_TEMPLATES = {
    "llama31": {
        "subject": "{% set loop_messages = messages %}{% for message in loop_messages %}{% set role = message['role'] %}{% set content = '<|start_header_id|>' + role + '<|end_header_id|>\n\n' + message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}",
        "decoder": "{% set loop_messages = messages %}{% for message in loop_messages %}{% set role = message['role'] %}{% if role == 'assistant' %}{% set role = 'reflect' %}{% endif %}{% set content = '<|start_header_id|>' + role + '<|end_header_id|>\n\n' + message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>reflect<|end_header_id|>\n\n' }}{% endif %}",
    }
}


def shortened_hf_model_id(hf_model_id: str) -> str:
    if hf_model_id == "meta-llama/Llama-3.1-8B-Instruct":
        return "llama31_8b"
    elif hf_model_id == "meta-llama/Llama-3.1-70B-Instruct":
        return "llama31_70b"
    else:
        raise ValueError(f"Unknown model ID: {hf_model_id}")


def get_tokenizer(hf_model_id: str):
    tokenizer = AutoTokenizer.from_pretrained(hf_model_id)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    return tokenizer


def get_subject_model(hf_model_id: str, inference_only: bool = True) -> AutoModelForCausalLM:
    model = AutoModelForCausalLM.from_pretrained(
        hf_model_id,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        use_cache=None,
        device_map="auto",
    )
    if inference_only:
        model.eval()
        model.requires_grad_(False)

    return model
