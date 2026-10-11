"""WSA-B style three-expert transformer for the Qwen3.5 backbone.

The context, planning, and action branches own their transformer parameters.
At every Qwen3.5 full-attention layer their Q/K/V tensors are concatenated and
attended jointly. Qwen3.5 linear-attention layers keep independent recurrent
states; cross-expert exchange occurs at the interleaved full-attention layers.
"""

from __future__ import annotations

from copy import deepcopy

import torch
from torch import Tensor, nn
from transformers.models.qwen3_5 import Qwen3_5TextModel, modeling_qwen3_5


def _compute_three_expert_layer(
    layer_idx: int,
    inputs_embeds: list[Tensor],
    attention_mask: Tensor,
    position_ids: Tensor,
    qwen3_5: nn.Module,
    world_expert: nn.Module,
    action_expert: nn.Module,
    branch_lengths: tuple[int, int, int],
    knowledge_insulation: bool,
    linear_attn_mask: Tensor | None,
) -> list[Tensor]:
    models = [qwen3_5.language_model, world_expert, action_expert]
    layer_type = qwen3_5.language_model.layers[layer_idx].layer_type

    if layer_type == "linear_attention":
        outputs = []
        offset = 0
        for model, hidden_states, length in zip(models, inputs_embeds, branch_lengths, strict=True):
            layer = model.layers[layer_idx]
            residual = hidden_states
            normalized = layer.input_layernorm(hidden_states)
            branch_mask = None if linear_attn_mask is None else linear_attn_mask[:, offset : offset + length]
            normalized = layer.linear_attn(
                hidden_states=normalized,
                cache_params=None,
                cache_position=None,
                attention_mask=branch_mask,
            )
            hidden_states = residual + normalized
            residual = hidden_states
            hidden_states = layer.post_attention_layernorm(hidden_states)
            if layer.mlp.up_proj.weight.dtype == torch.bfloat16:
                hidden_states = hidden_states.to(torch.bfloat16)
            outputs.append(residual + layer.mlp(hidden_states))
            offset += length
        return outputs

    if layer_type != "full_attention":
        raise ValueError(f"Unsupported Qwen3.5 layer type: {layer_type}")

    queries, keys, values, gates = [], [], [], []
    for model, hidden_states in zip(models, inputs_embeds, strict=True):
        layer = model.layers[layer_idx]
        normalized = layer.input_layernorm(hidden_states)
        input_shape = normalized.shape[:-1]
        q_gate = layer.self_attn.q_proj(normalized).view(*input_shape, -1, layer.self_attn.head_dim * 2)
        query, gate = torch.chunk(q_gate, 2, dim=-1)
        hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)
        query = layer.self_attn.q_norm(query.view(hidden_shape)).transpose(1, 2)
        key = layer.self_attn.k_norm(layer.self_attn.k_proj(normalized).view(hidden_shape)).transpose(1, 2)
        value = layer.self_attn.v_proj(normalized).view(hidden_shape).transpose(1, 2)
        queries.append(query)
        keys.append(key)
        values.append(value)
        gates.append(gate.reshape(*input_shape, -1))

    joint_query = torch.cat(queries, dim=2)
    joint_key = torch.cat(keys, dim=2)
    dummy = torch.zeros(
        joint_query.shape[0],
        joint_query.shape[2],
        joint_query.shape[-1],
        dtype=joint_query.dtype,
        device=joint_query.device,
    )
    cos, sin = qwen3_5.language_model.rotary_emb(dummy, position_ids)
    joint_query, joint_key = modeling_qwen3_5.apply_rotary_pos_emb(
        joint_query, joint_key, cos, sin, unsqueeze_dim=1
    )
    joint_value = torch.cat(values, dim=2)

    attn_layer = qwen3_5.language_model.layers[layer_idx].self_attn
    branch_attention = []
    start = 0
    prefix_len = branch_lengths[0]
    for branch_index, length in enumerate(branch_lengths):
        stop = start + length
        query = joint_query[:, :, start:stop]
        if branch_index == 0:
            key = joint_key[:, :, :prefix_len]
            value = joint_value[:, :, :prefix_len]
            mask = attention_mask[:, :, start:stop, :prefix_len]
        else:
            prefix_key = joint_key[:, :, :prefix_len]
            prefix_value = joint_value[:, :, :prefix_len]
            if knowledge_insulation:
                prefix_key = prefix_key.detach()
                prefix_value = prefix_value.detach()
            key = torch.cat((prefix_key, joint_key[:, :, prefix_len:]), dim=2)
            value = torch.cat((prefix_value, joint_value[:, :, prefix_len:]), dim=2)
            mask = attention_mask[:, :, start:stop, :]
        attended, _ = modeling_qwen3_5.eager_attention_forward(
            attn_layer, query, key, value, mask, attn_layer.scaling
        )
        branch_attention.append(attended)
        start = stop

    outputs = []
    for model, hidden_states, attended, gate in zip(
        models, inputs_embeds, branch_attention, gates, strict=True
    ):
        layer = model.layers[layer_idx]
        attended = attended.reshape(
            attended.shape[0],
            attended.shape[1],
            attn_layer.config.num_attention_heads * attn_layer.head_dim,
        )
        attended = attended * torch.sigmoid(gate)
        attended = attended.to(layer.self_attn.o_proj.weight.dtype)
        hidden_states = hidden_states + layer.self_attn.o_proj(attended)
        residual = hidden_states
        hidden_states = layer.post_attention_layernorm(hidden_states)
        if layer.mlp.up_proj.weight.dtype == torch.bfloat16:
            hidden_states = hidden_states.to(torch.bfloat16)
        outputs.append(residual + layer.mlp(hidden_states))
    return outputs


class WSA15MixtureOfTransformers(nn.Module):
    """Qwen3.5 context expert plus full-depth planning and action experts."""

    def __init__(self, internvla_experts: nn.Module) -> None:
        super().__init__()
        self.qwen3_5 = internvla_experts.qwen3_5
        self.action_expert = internvla_experts.action_expert
        self.world_expert = Qwen3_5TextModel(config=deepcopy(self.action_expert.config))
        self.world_expert.embed_tokens = None

        target_dtype = next(self.action_expert.parameters()).dtype
        self.world_expert.to(dtype=target_dtype)
        for name, parameter in self.world_expert.named_parameters():
            if any(
                selector in name for selector in ("input_layernorm", "post_attention_layernorm", "model.norm")
            ):
                parameter.data = parameter.data.float()

    def forward(
        self,
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
        past_key_values=None,
        inputs_embeds: list[Tensor | None] | None = None,
        use_cache: bool | None = None,
        knowledge_insulation: bool = False,
        linear_attn_mask: Tensor | None = None,
        **_: object,
    ) -> tuple[list[Tensor | None], object | None]:
        if inputs_embeds is None or len(inputs_embeds) != 3:
            raise ValueError("WSA1.5 expects [context, planning, action] expert embeddings")
        context, planning, action = inputs_embeds
        if planning is None and action is None:
            output = self.qwen3_5.language_model(
                inputs_embeds=context,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
            )
            return [output.last_hidden_state, None, None], output.past_key_values
        if context is None or planning is None or action is None:
            raise ValueError("Joint WSA1.5 execution requires all three expert branches")
        if past_key_values is not None or use_cache:
            raise ValueError("Joint WSA1.5 execution does not use a prefix KV cache")

        values = [context, planning, action]
        lengths = (context.shape[1], planning.shape[1], action.shape[1])
        use_checkpoint = self.training and getattr(self.action_expert, "gradient_checkpointing", False)
        num_layers = self.qwen3_5.config.text_config.num_hidden_layers
        for layer_idx in range(num_layers):
            if use_checkpoint:
                values = torch.utils.checkpoint.checkpoint(
                    _compute_three_expert_layer,
                    layer_idx,
                    values,
                    attention_mask,
                    position_ids,
                    self.qwen3_5,
                    self.world_expert,
                    self.action_expert,
                    lengths,
                    knowledge_insulation,
                    linear_attn_mask,
                    use_reentrant=False,
                    preserve_rng_state=False,
                )
            else:
                values = _compute_three_expert_layer(
                    layer_idx,
                    values,
                    attention_mask,
                    position_ids,
                    self.qwen3_5,
                    self.world_expert,
                    self.action_expert,
                    lengths,
                    knowledge_insulation,
                    linear_attn_mask,
                )
        models = [self.qwen3_5.language_model, self.world_expert, self.action_expert]
        return [
            [model.norm(value) for model, value in zip(models, values, strict=True)],
            None,
        ]


__all__ = ["WSA15MixtureOfTransformers"]
