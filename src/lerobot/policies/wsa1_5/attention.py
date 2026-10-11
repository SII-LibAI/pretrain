"""Explicit WSA1.5 context/planning/action attention dependencies."""

from __future__ import annotations

import torch
from torch import Tensor


def _slices(context_len: int, group_lengths: tuple[int, int, int, int, int]) -> dict[str, slice]:
    names = ("subtask", "future_2d", "future_3d", "state", "action")
    result = {"context": slice(0, context_len)}
    start = context_len
    for name, length in zip(names, group_lengths, strict=True):
        result[name] = slice(start, start + length)
        start += length
    return result


def build_wsa_attention_mask(
    padding_mask: Tensor,
    context_len: int,
    group_lengths: tuple[int, int, int, int, int],
) -> Tensor:
    """Build the context/subtask/2D/3D/state/action dependency graph."""

    if padding_mask.ndim != 2:
        raise ValueError(f"padding_mask must be [B,L], got {tuple(padding_mask.shape)}")
    groups = _slices(context_len, group_lengths)
    expected = context_len + sum(group_lengths)
    if padding_mask.shape[1] != expected:
        raise ValueError(f"Mask length {padding_mask.shape[1]} does not match group total {expected}")

    batch, total = padding_mask.shape
    allowed = torch.zeros(batch, total, total, dtype=torch.bool, device=padding_mask.device)
    context = groups["context"]
    context_causal = torch.ones(context_len, context_len, dtype=torch.bool, device=padding_mask.device).tril()
    allowed[:, context, context] = context_causal

    dependencies = {
        "subtask": ("context", "subtask"),
        "future_2d": ("context", "subtask", "future_2d", "future_3d"),
        "future_3d": ("context", "subtask", "future_3d", "action"),
        "state": ("context", "subtask", "future_3d", "state"),
        "action": ("context", "subtask", "future_3d", "state", "action"),
    }
    for query_name, key_names in dependencies.items():
        for key_name in key_names:
            allowed[:, groups[query_name], groups[key_name]] = True

    valid = padding_mask.bool()
    return allowed & valid[:, :, None] & valid[:, None, :]


__all__ = ["build_wsa_attention_mask"]
