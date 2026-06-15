"""
tomi/utils/tokenizer.py
-----------------------
Tokenizer helpers that provide a model-agnostic interface over
HuggingFace tokenizers.
"""

from __future__ import annotations

from typing import List, Optional, Union

import torch
from transformers import PreTrainedTokenizerBase


def tokenize(
    tokenizer: PreTrainedTokenizerBase,
    text: Union[str, List[str]],
    device: Optional[torch.device] = None,
    padding: bool = True,
    truncation: bool = True,
    max_length: Optional[int] = None,
    return_tensors: str = "pt",
) -> dict[str, torch.Tensor]:
    """Tokenize *text* and move the result to *device*.

    Parameters
    ----------
    tokenizer:
        A HuggingFace tokenizer.
    text:
        A single string or list of strings to tokenise.
    device:
        Optional device to move the output tensors to.
    padding:
        Whether to pad to the longest sequence.
    truncation:
        Whether to truncate to *max_length*.
    max_length:
        Maximum number of tokens. ``None`` uses the model's maximum.
    return_tensors:
        Backend format, default ``"pt"`` (PyTorch).

    Returns
    -------
    dict[str, torch.Tensor]
        Dictionary with ``input_ids``, ``attention_mask``, etc.
    """
    kwargs: dict = dict(
        padding=padding,
        truncation=truncation,
        return_tensors=return_tensors,
    )
    if max_length is not None:
        kwargs["max_length"] = max_length

    encoding = tokenizer(text, **kwargs)
    if device is not None:
        encoding = {k: v.to(device) for k, v in encoding.items()}
    return encoding


def get_token_ids(
    tokenizer: PreTrainedTokenizerBase,
    tokens: Union[str, List[str]],
    add_prefix_space: bool = False,
) -> List[int]:
    """Return the single token ids for each string in *tokens*.

    Parameters
    ----------
    tokenizer:
        A HuggingFace tokenizer.
    tokens:
        A string or list of string tokens to look up.
    add_prefix_space:
        Prepend a space before each token (useful for GPT-style tokenizers).

    Returns
    -------
    List[int]
        A list of token ids.

    Raises
    ------
    ValueError
        If any string maps to more than one token.
    """
    if isinstance(tokens, str):
        tokens = [tokens]

    ids = []
    for tok in tokens:
        if add_prefix_space:
            tok = " " + tok
        encoded = tokenizer.encode(tok, add_special_tokens=False)
        if len(encoded) != 1:
            raise ValueError(
                f"Token string '{tok}' encodes to {len(encoded)} tokens: {encoded}. "
                "Please use a string that maps to exactly one token."
            )
        ids.append(encoded[0])
    return ids


def decode_tokens(
    tokenizer: PreTrainedTokenizerBase,
    token_ids: Union[torch.Tensor, List[int]],
    skip_special_tokens: bool = True,
) -> str:
    """Decode *token_ids* back to a string.

    Parameters
    ----------
    tokenizer:
        A HuggingFace tokenizer.
    token_ids:
        A 1-D tensor or list of token ids.
    skip_special_tokens:
        Whether to remove special tokens from the decoded string.

    Returns
    -------
    str
    """
    if isinstance(token_ids, torch.Tensor):
        token_ids = token_ids.tolist()
    return tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
