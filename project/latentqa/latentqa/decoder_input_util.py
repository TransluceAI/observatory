from typing import Literal, cast

from transformers.tokenization_utils import PreTrainedTokenizer

ACT_IN_USER_Q_BEFORE_CONTEXT_USER_PROMPT = """
Question: {question}

# BEGIN CONTEXT
{context}
# END CONTEXT
""".strip()


ACT_IN_USER_Q_AFTER_CONTEXT_USER_PROMPT = """
# BEGIN CONTEXT
{context}
# END CONTEXT

Question: {question}
""".strip()

SYSTEM_PROMPT = "Your job is to answer questions based on the provided context."

ACT_IN_SYS_SYSTEM_PROMPT = """
{system_prompt}# BEGIN CONTEXT
{context}
# END CONTEXT
""".strip()


def find_subsequence_in_tokens(haystack: list[int], needle: list[int]) -> int:
    """Find the starting index of needle in haystack. Returns -1 if not found."""
    if len(needle) == 0:
        return 0
    if len(needle) > len(haystack):
        return -1

    for i in range(len(haystack) - len(needle) + 1):
        if haystack[i : i + len(needle)] == needle:
            return i
    return -1


def get_decoder_base_conversation_and_patch_range(
    format: str | Literal["default", "act_in_sys", "act_in_user"],
    num_patch_tokens: int,
    question: str,
    tokenizer: PreTrainedTokenizer,
    chat_template: str,
    assistant_role: str = "reflect",
    include_system_prompt: bool = False,
    question_before_context: bool = True,
) -> tuple[list[dict[str, str]], tuple[int, int]]:
    dummy_str = " ?" * num_patch_tokens + " "
    decoded_dummy_str = "?" * num_patch_tokens
    dummy_token_id = tokenizer.encode(" ?", add_special_tokens=False)[0]

    if format == "default":
        base_conversation = [
            {"role": "user", "content": dummy_str},
            {"role": "user", "content": question},
        ]
        dummy_token_ids = [30] + [dummy_token_id] * (num_patch_tokens - 1)

    elif format == "act_in_sys":
        prompt_before_context = f"{SYSTEM_PROMPT}\n" if include_system_prompt else ""
        system_prompt = ACT_IN_SYS_SYSTEM_PROMPT.format(
            system_prompt=prompt_before_context, context=dummy_str
        )
        base_conversation = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]
        dummy_token_ids = [dummy_token_id] * num_patch_tokens

    elif format == "act_in_user":
        template = (
            ACT_IN_USER_Q_BEFORE_CONTEXT_USER_PROMPT
            if question_before_context
            else ACT_IN_USER_Q_AFTER_CONTEXT_USER_PROMPT
        )
        base_conversation = [
            {
                "role": "user",
                "content": template.format(question=question, context=dummy_str),
            },
        ]
        if include_system_prompt:
            base_conversation.insert(0, {"role": "system", "content": SYSTEM_PROMPT})

        dummy_token_ids = [dummy_token_id] * num_patch_tokens

    elif format == "old_latentqa":
        base_conversation = [
            {"role": "user", "content": dummy_str},
            {"role": assistant_role, "content": "Sure, I've analyzed the assistant."},
            {"role": "user", "content": question},
        ]
        dummy_token_ids = [30] + [dummy_token_id] * (num_patch_tokens - 1)

    else:
        raise ValueError(f"Invalid format: {format}")

    # Find the patch range.
    base_conversation_ids = cast(
        list[int],
        tokenizer.apply_chat_template(
            conversation=base_conversation, chat_template=chat_template, return_dict=False
        ),
    )

    idx = find_subsequence_in_tokens(base_conversation_ids, dummy_token_ids)
    try:
        assert idx != -1, "Could not find dummy string in base conversation"
    except AssertionError:
        import pdb

        pdb.set_trace()

    patch_range = (idx, idx + num_patch_tokens)

    # Check that the patch_range is correct.
    assert (
        tokenizer.decode(base_conversation_ids[patch_range[0] : patch_range[1]])
        == decoded_dummy_str
    )

    return base_conversation, patch_range
