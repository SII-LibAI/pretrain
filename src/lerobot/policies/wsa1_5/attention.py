"""True means visible (PyTorch SDPA convention). Query rows, key columns."""
from __future__ import annotations

from collections.abc import Mapping
from contextlib import nullcontext

import torch
import torch.nn.functional as F

GROUPS = ("O", "R", "M", "G")
VISIBLE = {"O": ("O",), "R": ("O", "R"), "M": ("O", "M"), "G": GROUPS}


def build_causal_attention_mask(lengths: Mapping[str, int], *, valid_tokens=None, device=None):
    """Dynamic dense reference mask; production MoT uses equivalent block masks.

    valid_tokens maps groups to boolean [B,S] tensors. It masks both query and
    key padding, returning [B,1,N,N]; without padding the result is [N,N].
    """
    if set(lengths) - set(GROUPS) or any(not isinstance(n, int) or n < 0 for n in lengths.values()):
        raise ValueError("Expected non-negative integer O/R/M/G token lengths")
    offsets, cursor = {}, 0
    for group in GROUPS:
        n = lengths.get(group, 0)
        offsets[group] = slice(cursor, cursor + n)
        cursor += n
    mask = torch.zeros(cursor, cursor, dtype=torch.bool, device=device)
    for query in GROUPS:
        for key in VISIBLE[query]:
            mask[offsets[query], offsets[key]] = True
    if valid_tokens is None:
        return mask
    if not valid_tokens:
        raise ValueError("valid_tokens must include at least one batch-shaped mask")
    batch_size = next(iter(valid_tokens.values())).shape[0]
    flags = []
    for group in GROUPS:
        flag = valid_tokens.get(group)
        if flag is None:
            if lengths.get(group, 0):
                raise ValueError(f"Missing explicit padding mask for active group {group}")
            flag = torch.ones(batch_size, lengths.get(group, 0), dtype=torch.bool, device=mask.device)
        if flag.shape != (batch_size, lengths.get(group, 0)) or flag.dtype != torch.bool:
            raise ValueError(f"Invalid padding mask for {group}")
        flags.append(flag.to(mask.device))
    valid = torch.cat(flags, 1)
    return mask[None, None] & valid[:, None, :, None] & valid[:, None, None, :]


def masked_attention(q, k, v, mask, *, backend="sdpa"):
    """[B,H,S,D] attention with safe empty rows and explicit math fallback.

    No unmasked FlashAttention API is used. SDPA retains the boolean mask even
    when its dispatcher chooses an optimized kernel. backend='math' forces the
    portable PyTorch kernel at the cost of quadratic attention-score storage.
    """
    if backend not in {"sdpa", "math", "reference"}:
        raise ValueError(f"Unsupported masked backend {backend}")
    if mask.dtype != torch.bool:
        raise ValueError("mask must be boolean, True=visible")
    if k.shape[-2] == 0:
        raise ValueError("Attention requires at least one key; an empty key sequence is a structural error")
    empty = ~mask.any(-1, keepdim=True)
    # Give empty rows a numerically defined softmax and then exactly zero them.
    safe_mask = mask.clone()
    safe_mask[..., :1] |= empty
    if backend == "reference":
        scores = q.float() @ k.float().transpose(-1, -2) / q.shape[-1] ** 0.5
        scores = scores.masked_fill(~safe_mask, float("-inf"))
        result = (scores.softmax(-1) @ v.float()).to(q.dtype)
    else:
        if backend == "math":
            from torch.nn.attention import SDPBackend, sdpa_kernel
            context = sdpa_kernel(SDPBackend.MATH)
        else:
            context = nullcontext()
        with context:
            result = F.scaled_dot_product_attention(q, k, v, attn_mask=safe_mask, dropout_p=0.0)
    return result.masked_fill(empty, 0)
