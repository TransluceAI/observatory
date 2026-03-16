import torch


def pad_sequence(
    sequences: list[list[int]] | list[list[bool]],
    pad_token_id: int | bool,
    padding_side: str = "right",
    max_length: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pad a list of sequences to the same length and return input_ids and attention_mask tensors.

    If `max_length` is provided, AND if the `max_length` is longer than the longest sequence,
    pad the sequences to the same length up to `max_length`.
    """
    seq_tensors = [torch.tensor(seq) for seq in sequences]
    ones_tensors = [torch.ones_like(seq, dtype=torch.int32) for seq in seq_tensors]
    input_ids = torch.nn.utils.rnn.pad_sequence(
        seq_tensors, batch_first=True, padding_value=pad_token_id, padding_side=padding_side
    )
    attention_mask = torch.nn.utils.rnn.pad_sequence(
        ones_tensors, batch_first=True, padding_value=0, padding_side=padding_side
    )
    if max_length is not None:
        cur_len = input_ids.shape[1]
        if cur_len < max_length:
            pad_len = max_length - cur_len
            if padding_side == "right":
                input_ids = torch.nn.functional.pad(input_ids, (0, pad_len), value=pad_token_id)
                attention_mask = torch.nn.functional.pad(attention_mask, (0, pad_len), value=0)
            else:
                input_ids = torch.nn.functional.pad(input_ids, (pad_len, 0), value=pad_token_id)
                attention_mask = torch.nn.functional.pad(attention_mask, (pad_len, 0), value=0)

    return input_ids, attention_mask


def strip_padding(token_ids: list[int], pad_token_id: int, padding_side: str = "left") -> list[int]:
    if padding_side == "left":
        first_non_pad_idx = 0
        while first_non_pad_idx < len(token_ids):
            if token_ids[first_non_pad_idx] != pad_token_id:
                break
            first_non_pad_idx += 1
        return token_ids[first_non_pad_idx:]
    elif padding_side == "right":
        last_non_pad_idx = len(token_ids) - 1
        while last_non_pad_idx >= 0:
            if token_ids[last_non_pad_idx] != pad_token_id:
                break
            last_non_pad_idx -= 1
        return token_ids[: last_non_pad_idx + 1]
    else:
        raise ValueError(f"Invalid padding side: {padding_side}")
