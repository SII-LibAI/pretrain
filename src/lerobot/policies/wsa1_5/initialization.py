"""Checkpoint loading, expert initialization and RGB LoRA adapters."""
from pathlib import Path
import logging
import math

import torch
from torch import nn
import torch.nn.functional as F


def read_checkpoint(path):
    path = Path(path)
    files = sorted(path.glob("diffusion_pytorch_model*.safetensors")) if path.is_dir() else [path]
    if not files:
        raise FileNotFoundError(f"No Wan safetensor shards found in {path}")
    state = {}
    for file in files:
        if not file.exists():
            raise FileNotFoundError(file)
        if file.suffix == ".safetensors":
            from safetensors.torch import load_file
            shard = load_file(str(file), device="cpu")
        else:
            shard = torch.load(file, map_location="cpu", weights_only=True)
        if isinstance(shard, dict):
            wrappers = [key for key in ["model_state", "state_dict", "model"] if key in shard]
            if len(wrappers) > 1:
                raise ValueError(f"Ambiguous checkpoint state containers in {file}: {wrappers}")
            if wrappers:
                shard = shard[wrappers[0]]
        if not isinstance(shard, dict):
            raise TypeError(f"Checkpoint {file} is not a state dictionary")
        duplicate = state.keys() & shard.keys()
        if duplicate:
            raise ValueError(f"Duplicate checkpoint shard keys: {sorted(duplicate)}")
        state.update(shard)
    return state


def load_stage1_policy(module, directory):
    """Load matching world weights; omit source UMT5 only for an offline-text target."""
    import json

    directory = Path(directory)
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    if config.get("enable_action", False):
        raise ValueError("init_checkpoint_path expects Stage-1; use policy.path/resume for Stage-2")
    state = read_checkpoint(directory / "model.safetensors")
    omitted_text = set()
    if config.get("load_text_encoder", False) is True and not module.config.load_text_encoder and module.text_encoder is None:
        omitted_text = {key for key in state if key.startswith("text_encoder.")}
        for key in omitted_text:
            del state[key]
    expected = module.state_dict()
    added = {key for key in expected if key.startswith("action_expert.") or ".lora_a" in key or ".lora_b" in key}
    missing, unexpected = set(expected) - set(state), set(state) - set(expected)
    if missing != added or unexpected:
        raise ValueError(f"Stage-1 initialization mismatch: missing={sorted(missing - added)}, unexpected={sorted(unexpected)}")
    for key, value in state.items():
        if value.shape != expected[key].shape:
            raise ValueError(f"Stage-1 shape mismatch for {key}: {tuple(value.shape)} != {tuple(expected[key].shape)}")
    report = load_report(module, state, name="Stage-1 -> Action post-training", strict=False)
    logging.info("[WSA init] loaded=%d new_action_or_lora=%d omitted_frozen_umt5=%d; "
                 "optimizer/scheduler/RNG/step start fresh", len(state), len(added), len(omitted_text))
    return report


def resize_axes(value, shape):
    """Linear interpolation along each changed axis, never silent slicing.

    Attention head geometry is preserved by the target shape; only expert width
    and FFN axes change. Modulation/time-projection groups are handled separately.
    """
    if value.ndim != len(shape):
        raise ValueError(f"Cannot interpolate different tensor ranks: {value.shape} -> {shape}")
    result = value.float()
    for axis, size in enumerate(shape):
        if result.shape[axis] == size:
            continue
        moved = result.movedim(axis, -1)
        prefix = moved.shape[:-1]
        moved = F.interpolate(moved.reshape(-1, 1, moved.shape[-1]), size=size, mode="linear",
                              align_corners=True).reshape(*prefix, size)
        result = moved.movedim(-1, axis)
    return result.to(value.dtype)


def interpolate_wan_state(source, target_state, *, apply_alpha_scaling=False):
    """Preserve modulation groups/head geometry; optionally apply WSA last-axis scaling."""
    converted, rules = {}, {}
    for key, target in target_state.items():
        if key not in source:
            continue
        value = source[key]
        if not torch.is_tensor(value) or not torch.isfinite(value).all():
            raise ValueError(f"Invalid/non-finite checkpoint tensor {key}")
        fixed_axes = []
        if key == "patch_embedding.weight":
            fixed_axes = list(range(1, target.ndim))  # latent channels and patch geometry
        elif key == "text_embedding.0.weight" or key == "time_embedding.0.weight":
            fixed_axes = [1]  # UMT5 width and sinusoidal frequency width
        elif ".self_attn." in key or ".cross_attn." in key:
            suffix = key.rsplit(".", 2)[-2:]
            if suffix[0] in {"q", "k", "v", "norm_q", "norm_k"}:
                fixed_axes = [0]  # head count * head dimension
            elif suffix == ["o", "weight"]:
                fixed_axes = [1]
        if value.ndim != target.ndim or any(value.shape[axis] != target.shape[axis] for axis in fixed_axes):
            raise ValueError(f"Fixed checkpoint geometry mismatch for {key}: {value.shape} -> {target.shape}")
        if value.shape == target.shape:
            converted[key], rules[key] = value, "copy"
        elif key == "time_projection.1.weight":
            converted[key] = resize_axes(value.reshape(6, value.shape[0] // 6, value.shape[1]),
                                         (6, target.shape[0] // 6, target.shape[1])).reshape(target.shape)
            rules[key] = "six separate modulation matrices"
        elif key == "time_projection.1.bias":
            converted[key] = resize_axes(value.reshape(6, -1), (6, target.numel() // 6)).reshape(target.shape)
            rules[key] = "six separate modulation vectors"
        elif key.startswith("head.head."):
            # The latent-channel/patch axes must never be resized.
            if value.shape[0] != target.shape[0]:
                raise ValueError(f"Output patch/channel mismatch for {key}")
            converted[key], rules[key] = resize_axes(value, target.shape), "preserve latent channels; interpolate hidden input"
        else:
            converted[key], rules[key] = resize_axes(value, target.shape), "axiswise linear width interpolation"
        if apply_alpha_scaling and value.ndim >= 2 and value.shape[-1] != target.shape[-1]:
            alpha = (value.shape[-1] / target.shape[-1]) ** 0.5
            converted[key] = (converted[key].float() * alpha).to(target.dtype)
            rules[key] += f"; alpha scaling {alpha:.6g}"
    return converted, rules


def load_report(module, state, *, name, strict=True):
    expected = module.state_dict()
    missing = sorted(expected.keys() - state.keys())
    unexpected = sorted(state.keys() - expected.keys())
    logging.info("[WSA load] %s missing=%s unexpected=%s", name, missing, unexpected)
    mismatched = [(key, tuple(value.shape), tuple(expected[key].shape)) for key, value in state.items()
                  if key in expected and torch.is_tensor(value) and value.shape != expected[key].shape]
    if mismatched:
        raise RuntimeError(f"{name} checkpoint shape mismatches: {mismatched}")
    if strict and (missing or unexpected):
        raise RuntimeError(f"Incomplete {name} checkpoint: missing={missing}, unexpected={unexpected}")
    for key, value in state.items():
        if key in expected and (not torch.is_tensor(value) or not torch.isfinite(value).all()):
            raise ValueError(f"Invalid/non-finite {name} checkpoint tensor {key}")
    report = module.load_state_dict(state, strict=False)
    if strict and (report.missing_keys or report.unexpected_keys):
        raise RuntimeError(f"Incomplete {name} checkpoint: missing={report.missing_keys}, unexpected={report.unexpected_keys}")
    return report


class LoRALinear(nn.Linear):
    def __init__(self, base, rank, alpha, dropout):
        super().__init__(base.in_features, base.out_features, bias=base.bias is not None,
                         device=base.weight.device, dtype=base.weight.dtype)
        self.weight, self.bias = base.weight, base.bias
        self.weight.requires_grad_(False)
        if self.bias is not None:
            self.bias.requires_grad_(False)
        self.lora_a = nn.Parameter(base.weight.new_empty(rank, base.in_features))
        self.lora_b = nn.Parameter(base.weight.new_zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        self.scale = alpha / rank
        self.dropout = nn.Dropout(dropout)

    def forward(self, value):
        return F.linear(value, self.weight, self.bias) + F.linear(F.linear(self.dropout(value), self.lora_a), self.lora_b) * self.scale


def configure_rgb(rgb, config):
    mode = config.rgb_finetune_mode
    rgb.requires_grad_(mode == "full")
    if mode != "lora":
        return []
    matches = []
    for target in config.rgb_lora_target_modules:
        found = [(name, module) for name, module in rgb.named_modules()
                 if name.startswith("blocks.") and name.endswith("." + target) and isinstance(module, nn.Linear)]
        if not found:
            raise ValueError(f"RGB LoRA target does not match any actual Linear: {target}")
        for name, module in found:
            if isinstance(module, LoRALinear):
                raise ValueError(f"Overlapping RGB LoRA targets: {name}")
            parent, attr = name.rsplit(".", 1)
            setattr(rgb.get_submodule(parent), attr, LoRALinear(module, config.rgb_lora_rank,
                                                               config.rgb_lora_alpha, config.rgb_lora_dropout))
            matches.append(name)
    return matches
