"""WSA-default slot-noise queries and multi-layer geometric distillation."""
import torch
from torch import nn
import torch.nn.functional as F

from .mot import ExpertState
from .validation import require_bool, require_finite
from .vendor.wan_video_dit import SelfAttention, precompute_freqs_cis, sinusoidal_embedding_1d


class Future3DBlock(nn.Module):
    def __init__(self, width, ffn_dim, heads, head_dim):
        super().__init__()
        self.self_attn = SelfAttention(width, head_dim, heads)
        self.norm1 = nn.LayerNorm(width, eps=1e-6, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(width, eps=1e-6, elementwise_affine=False)
        self.ffn = nn.Sequential(nn.Linear(width, ffn_dim), nn.GELU(approximate="tanh"), nn.Linear(ffn_dim, width))
        self.modulation = nn.Parameter(torch.randn(1, 6, width) / width ** 0.5)


class Future3DExpert(nn.Module):
    def __init__(self, hidden_dim=768, ffn_dim=3072, layers=30, heads=24, head_dim=128,
                 freq_dim=256, output_dim=2048, teacher_layers=(4, 11, 17, 23), student_layers=None):
        super().__init__()
        if hidden_dim % 2:
            raise ValueError("3D slot positional encoding requires an even hidden_dim")
        self.teacher_layers = tuple(teacher_layers)
        if len(self.teacher_layers) != 4 or len(set(self.teacher_layers)) != 4:
            raise ValueError("Four distinct teacher feature layers are required")
        self.student_layers = tuple(student_layers) if student_layers is not None else tuple(
            round(layer / 23 * (layers - 1)) for layer in self.teacher_layers)
        if len(self.student_layers) != 4 or any(layer < 0 or layer >= layers for layer in self.student_layers):
            raise ValueError("Invalid student feature layer mapping")
        self.hidden_dim, self.freq_dim = hidden_dim, freq_dim
        self.num_heads, self.attn_head_dim = heads, head_dim
        slots = sinusoidal_embedding_1d(hidden_dim, torch.arange(819, dtype=torch.float32)).unsqueeze(0)
        self.register_buffer("slot_pos_embed", slots, persistent=False)
        inference_noise = torch.empty(1, 819, hidden_dim, dtype=torch.float32)
        inference_noise.normal_(generator=torch.Generator().manual_seed(0))
        self.register_buffer("inference_noise_tokens", inference_noise, persistent=False)
        self.time_embedding = nn.Sequential(nn.Linear(freq_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, hidden_dim * 6))
        self.blocks = nn.ModuleList([Future3DBlock(hidden_dim, ffn_dim, heads, head_dim) for _ in range(layers)])
        self.feature_heads = nn.ModuleDict({str(layer): nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, output_dim))
                                          for layer in self.teacher_layers})

    def prepare(self, batch_size, view_valid, *, num_train_timesteps):
        if view_valid.dtype != torch.bool:
            raise TypeError("Observed view validity must be boolean")
        if view_valid.shape != (batch_size, 3):
            raise ValueError("3D query visibility requires the observed three-view mask")
        weight = self.time_embedding[0].weight
        slots = self.slot_pos_embed.expand(batch_size, -1, -1).to(device=weight.device, dtype=weight.dtype)
        if self.training and torch.is_grad_enabled():
            noise = torch.randn(batch_size, 819, self.hidden_dim, device=weight.device, dtype=weight.dtype)
        else:
            noise = self.inference_noise_tokens.expand(batch_size, -1, -1).to(device=weight.device, dtype=weight.dtype)
        # WSA slot_noise defaults: slot scale 0.5, constant sigma 0.5, noise scale 0.5.
        tokens = slots * 0.5 + noise * 0.25
        timestep = tokens.new_full((batch_size,), 0.5 * num_train_timesteps)
        t = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, timestep))
        modulation = self.time_projection(t).reshape(batch_size, 6, self.hidden_dim)
        freqs = precompute_freqs_cis(self.attn_head_dim, 819).to(tokens.device)[:, None]
        return ExpertState(tokens, modulation, view_valid.repeat_interleave(273, dim=1), freqs)

    def project(self, captured):
        result = {}
        for teacher_layer, student_layer in zip(self.teacher_layers, self.student_layers, strict=True):
            if student_layer not in captured or "G" not in captured[student_layer]:
                raise ValueError(f"Missing student feature layer {student_layer}")
            hidden = captured[student_layer]["G"]
            result[teacher_layer] = self.feature_heads[str(teacher_layer)](hidden).reshape(hidden.shape[0], 3, 273, -1)
        return result


def geometric_feature_loss(student, teacher, valid, *, mse_weight=1.0, cosine_weight=1.0):
    require_bool(valid, "3D feature validity")
    if not student:
        raise ValueError("3D feature supervision requires non-empty layers")
    if set(student) != set(teacher):
        raise ValueError("Student/teacher feature layer sets disagree")
    losses, metrics = [], {}
    for layer, prediction in student.items():
        target = teacher[layer]
        if prediction.shape != target.shape or valid.shape != target.shape[:-1]:
            raise ValueError(f"3D feature/mask shape mismatch at layer {layer}")
        require_finite(prediction, f"Student features at layer {layer}")
        require_finite(target, f"Teacher features at layer {layer}")
        target_norm = target.float().norm(dim=-1)
        require_finite(target_norm, f"Teacher feature norm at layer {layer}")
        require_finite(prediction.float().norm(dim=-1), f"Student feature norm at layer {layer}")
        if (target_norm[valid] <= 1e-6).any():
            raise ValueError(f"Valid teacher features have zero/degenerate norm at layer {layer}")
        p = F.normalize(prediction.float(), dim=-1, eps=1e-6)
        t = F.normalize(target.detach().float(), dim=-1, eps=1e-6)
        # Match WSA: LayerNorm MSE avoids the 1 / feature_dim scale of unit-vector MSE.
        p_ln = F.layer_norm(prediction.float(), (prediction.shape[-1],))
        t_ln = F.layer_norm(target.detach().float(), (target.shape[-1],))
        mse = ((p_ln - t_ln).square().mean(-1).masked_fill(~valid, 0)).sum() / valid.sum().clamp_min(1)
        cosine = ((1 - (p * t).sum(-1)).masked_fill(~valid, 0)).sum() / valid.sum().clamp_min(1)
        losses.append(mse_weight * mse + cosine_weight * cosine)
        metrics[f"3d/layer_{layer}_mse"] = mse.detach().item()
        metrics[f"3d/layer_{layer}_cosine"] = cosine.detach().item()
    return torch.stack(losses).mean(), metrics
