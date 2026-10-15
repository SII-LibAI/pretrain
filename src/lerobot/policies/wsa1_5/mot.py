"""WSA-style heterogeneous experts with strictly block-causal mixed attention."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .attention import GROUPS, VISIBLE, masked_attention


@dataclass
class ExpertState:
    tokens: torch.Tensor
    modulation: torch.Tensor
    valid: torch.Tensor
    freqs: torch.Tensor | None = None
    context: torch.Tensor | None = None
    context_valid: torch.Tensor | None = None


def apply_rope(x, freqs, heads):
    if freqs is None:
        return x
    shaped = x.reshape(*x.shape[:2], heads, -1)
    complex_x = torch.view_as_complex(shaped.float().reshape(*shaped.shape[:-1], -1, 2))
    return torch.view_as_real(complex_x * freqs).flatten(2).to(x.dtype)


def split_modulation(block, t):
    modulation = block.modulation.to(dtype=t.dtype)
    if t.ndim == 4:
        return tuple(v.squeeze(2) for v in (t + modulation.unsqueeze(0)).chunk(6, 2))
    if t.ndim == 3:
        return (t + modulation).chunk(6, 1)
    raise ValueError("Time modulation must be [B,S,6,D] or [B,6,D]")


class CausalMoT(nn.Module):
    """Uses passed-in expert blocks without owning duplicate parameter references.

    Each group keeps its projections, RoPE, FFN and time modulation. Shared
    attention width is num_heads * head_dim, independent of expert hidden width.
    Only permitted key groups are gathered, avoiding a global NxN score matrix.
    """
    def __init__(self, *, backend="sdpa", gradient_checkpointing=False):
        super().__init__()
        self.backend = backend
        self.gradient_checkpointing = gradient_checkpointing

    def _layer(self, tokens, groups, states, blocks):
        qkv, mods = {}, {}
        for group, token in zip(groups, tokens, strict=True):
            state, block = states[group], blocks[group]
            token = token.masked_fill(~state.valid[:, :, None], 0)
            shift, scale, gate, fshift, fscale, fgate = split_modulation(block, state.modulation)
            attn_input = block.norm1(token) * (1 + scale) + shift
            attn = block.self_attn
            q = apply_rope(attn.norm_q(attn.q(attn_input)), state.freqs, attn.num_heads)
            k = apply_rope(attn.norm_k(attn.k(attn_input)), state.freqs, attn.num_heads)
            v = attn.v(attn_input)
            def heads(x):
                return x.reshape(x.shape[0], x.shape[1], attn.num_heads, -1).transpose(1, 2)
            qkv[group] = (heads(q), heads(k), heads(v))
            mods[group] = (gate, fshift, fscale, fgate)
        outputs = []
        for group, token in zip(groups, tokens, strict=True):
            block, state = blocks[group], states[group]
            sources = [source for source in VISIBLE[group] if source in qkv]
            k = torch.cat([qkv[source][1] for source in sources], 2)
            v = torch.cat([qkv[source][2] for source in sources], 2)
            key_valid = torch.cat([states[source].valid for source in sources], 1)
            mask = state.valid[:, None, :, None] & key_valid[:, None, None, :]
            mixed = masked_attention(qkv[group][0], k, v, mask, backend=self.backend)
            mixed = mixed.transpose(1, 2).flatten(2)
            gate, shift, scale, fgate = mods[group]
            x = token + gate * block.self_attn.o(mixed)
            if state.context is not None and hasattr(block, "cross_attn"):
                # Text context is immutable observed input, never a target field.
                cross = block.cross_attn
                cross_q = cross.norm_q(cross.q(block.norm3(x)))
                cross_k = cross.norm_k(cross.k(state.context))
                cross_v = cross.v(state.context)
                def cross_heads(value):
                    return value.reshape(value.shape[0], value.shape[1], cross.num_heads, -1).transpose(1, 2)
                context_valid = state.context_valid
                if context_valid is None:
                    raise ValueError("Text context requires an explicit padding mask")
                context_mask = state.valid[:, None, :, None] & context_valid[:, None, None, :]
                attended = masked_attention(cross_heads(cross_q), cross_heads(cross_k), cross_heads(cross_v),
                                            context_mask, backend=self.backend).transpose(1, 2).flatten(2)
                x = x + cross.o(attended)
            x = x + fgate * block.ffn(block.norm2(x) * (1 + scale) + shift)
            outputs.append(x.masked_fill(~state.valid[:, :, None], 0))
        return tuple(outputs)

    def forward(self, states, experts, *, capture_layers=()):
        if not states or "O" not in states:
            raise ValueError("CausalMoT requires observed O tokens")
        if set(states) - set(GROUPS) or set(states) - set(experts):
            raise ValueError("Every supplied group needs its own expert")
        groups = tuple(group for group in GROUPS if group in states)
        depths = {len(experts[group].blocks) for group in groups}
        widths = {(experts[group].blocks[0].self_attn.num_heads,
                   experts[group].blocks[0].self_attn.attn_head_dim) for group in groups}
        if len(depths) != 1 or len(widths) != 1:
            raise ValueError("Mixed experts must share depth and attention head geometry")
        depth = depths.pop()
        if any(layer < 0 or layer >= depth for layer in capture_layers):
            raise ValueError("Invalid captured layer index")
        for group in groups:
            state = states[group]
            if state.valid.shape != state.tokens.shape[:2] or state.valid.dtype != torch.bool:
                raise ValueError(f"Bad valid token mask for {group}")
            if state.context is not None and (state.context_valid is None
                    or state.context_valid.dtype != torch.bool or state.context_valid.shape != state.context.shape[:2]):
                raise ValueError(f"Bad/missing text context validity for {group}")
        tokens = tuple(states[group].tokens for group in groups)
        captured = {}
        for layer in range(depth):
            blocks = {group: experts[group].blocks[layer] for group in groups}
            # Bind each layer's blocks; checkpoint backward must not close over
            # a loop variable subsequently changed to the final layer.
            def run(*xs, current_blocks=blocks):
                return self._layer(xs, groups, states, current_blocks)
            if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
                tokens = checkpoint(run, *tokens, use_reentrant=False)
            else:
                tokens = run(*tokens)
            if layer in capture_layers:
                captured[layer] = dict(zip(groups, tokens, strict=True))
        return dict(zip(groups, tokens, strict=True)), captured
