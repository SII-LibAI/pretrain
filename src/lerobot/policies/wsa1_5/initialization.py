"""Server-side checkpoint loading and explicit width interpolation rules."""
from pathlib import Path
import logging

import torch
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


def interpolate_wan_state(source, target_state):
    """Named rules preserve all six modulation groups and head output channels."""
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
            value = resize_axes(value.reshape(6, value.shape[0] // 6, value.shape[1]),
                                (6, target.shape[0] // 6, target.shape[1])).reshape(target.shape)
            converted[key], rules[key] = value, "six separate modulation matrices"
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
    for child_name, child in module.named_children():
        logging.info("[WSA parameters] %s.%s=%d", name, child_name, sum(p.numel() for p in child.parameters()))
    if hasattr(module, "blocks"):
        for index, block in enumerate(module.blocks):
            logging.info("[WSA layer parameters] %s.blocks.%d=%d", name, index, sum(p.numel() for p in block.parameters()))
    if strict and (report.missing_keys or report.unexpected_keys):
        raise RuntimeError(f"Incomplete {name} checkpoint: missing={report.missing_keys}, unexpected={report.unexpected_keys}")
    return report
