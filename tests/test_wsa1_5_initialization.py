"""Synthetic checkpoint tensors and tiny modules only; no pretrained model loading."""
from copy import deepcopy
from pathlib import Path

import pytest
import torch
from torch import nn

from lerobot.policies.wsa1_5.initialization import LoRALinear, interpolate_wan_state, load_world_policy
from lerobot.policies.wsa1_5.modeling_wsa1_5 import WSA1_5_Policy, WorldComponents
from test_wsa1_5_experts import small_flow
from test_wsa1_5_policy_contract import small_config


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("key,source_shape,target_shape,alpha", [
    ("blocks.0.self_attn.q.weight", (8, 16), (8, 4), 2.0),
    ("blocks.0.self_attn.k.weight", (8, 16), (8, 4), 2.0),
    ("blocks.0.self_attn.v.weight", (8, 16), (8, 4), 2.0),
    ("blocks.0.self_attn.q.weight", (8, 4), (8, 16), 0.5),
    ("blocks.0.self_attn.o.weight", (16, 8), (4, 8), 1.0),
    ("blocks.0.ffn.0.weight", (32, 16), (12, 4), 2.0),
    ("blocks.0.ffn.2.weight", (16, 32), (4, 12), (32 / 12) ** 0.5),
    ("time_embedding.0.weight", (16, 8), (4, 8), 1.0),
    ("time_embedding.2.weight", (16, 16), (4, 4), 2.0),
    ("time_projection.1.weight", (96, 16), (24, 4), 2.0),
    ("blocks.0.modulation", (1, 6, 16), (1, 6, 4), 2.0),
    ("blocks.0.ffn.0.bias", (32,), (12,), 1.0),
    ("blocks.0.self_attn.norm_q.weight", (8,), (8,), 1.0),
    ("blocks.0.self_attn.q.weight", (8, 16), (8, 16), 1.0),
])
def test_wsa_alpha_scaling_uses_last_axis_only(dtype, key, source_shape, target_shape, alpha):
    source = {key: torch.full(source_shape, 1.25, dtype=dtype)}
    target = {key: torch.zeros(target_shape, dtype=dtype)}
    converted, rules = interpolate_wan_state(source, target, apply_alpha_scaling=True)
    expected = (torch.full(target_shape, 1.25) * alpha).to(dtype)
    torch.testing.assert_close(converted[key], expected, atol=0, rtol=0)
    assert ("alpha scaling" in rules[key]) == (alpha != 1)
    assert torch.equal(source[key], torch.full(source_shape, 1.25, dtype=dtype))


def test_alpha_scaling_preserves_six_modulation_groups_and_target_dtype():
    groups = torch.arange(1, 7, dtype=torch.float32)
    source = {"time_projection.1.weight": groups[:, None, None].expand(6, 16, 16).reshape(96, 16),
              "time_projection.1.bias": groups.repeat_interleave(16),
              "blocks.0.modulation": groups[None, :, None].expand(1, 6, 16)}
    target = {"time_projection.1.weight": torch.zeros(24, 4, dtype=torch.bfloat16),
              "time_projection.1.bias": torch.zeros(24, dtype=torch.bfloat16),
              "blocks.0.modulation": torch.zeros(1, 6, 4, dtype=torch.bfloat16)}
    converted, _ = interpolate_wan_state(source, target, apply_alpha_scaling=True)
    torch.testing.assert_close(converted["time_projection.1.weight"].float().reshape(6, 4, 4),
                               (groups * 2)[:, None, None].expand(6, 4, 4), atol=0, rtol=0)
    torch.testing.assert_close(converted["blocks.0.modulation"].float(),
                               (groups * 2)[None, :, None].expand(1, 6, 4), atol=0, rtol=0)
    torch.testing.assert_close(converted["time_projection.1.bias"].reshape(6, 4),
                               groups[:, None].expand(6, 4), check_dtype=False, atol=0, rtol=0)
    assert converted["time_projection.1.weight"].dtype == torch.bfloat16
    assert converted["blocks.0.modulation"].dtype == torch.bfloat16


@pytest.mark.parametrize("initialization", ["wan_width_interpolation", "random"])
def test_policy_enables_alpha_for_3d_wan_transfer_only(tmp_path, monkeypatch, initialization):
    module = "lerobot.policies.wsa1_5.modeling_wsa1_5"
    source = small_flow().rgb.state_dict()
    wan_path, vae_path = tmp_path / "synthetic_wan", tmp_path / "synthetic_vae"
    wan_path.touch()
    vae_path.touch()

    def read_synthetic(path):
        if Path(path) == wan_path:
            return source
        assert Path(path) == vae_path
        return {"model.weight": torch.ones(2, 2), "model.bias": torch.zeros(2)}

    calls = []

    def interpolate(source_state, target, *, apply_alpha_scaling=False):
        calls.append((apply_alpha_scaling, set(target)))
        return interpolate_wan_state(source_state, target, apply_alpha_scaling=apply_alpha_scaling)

    monkeypatch.setattr(f"{module}.read_checkpoint", read_synthetic)
    monkeypatch.setattr(f"{module}.wan_video_dit_state_dict_converter", lambda state: state)
    monkeypatch.setattr(f"{module}.interpolate_wan_state", interpolate)
    monkeypatch.setattr("lerobot.policies.wsa1_5.vendor.wan_video_vae.WanVideoVAE38",
                        lambda: nn.ModuleDict({"model": nn.Linear(2, 2)}))
    config = small_config(wan_model_path=str(wan_path), vae_path=str(vae_path), goal_initialization=initialization)
    policy = WSA1_5_Policy(config)
    assert [enabled for enabled, _ in calls] == ([False, True] if initialization == "wan_width_interpolation" else [False])
    if initialization == "wan_width_interpolation":
        assert not any(key.startswith("feature_heads.") for key in calls[1][1])
        key = "blocks.0.self_attn.q.weight"
        shape = policy.future_3d.state_dict()[key].shape
        expected = torch.nn.functional.interpolate(source[key][:, None], size=shape[-1],
                                                     mode="linear", align_corners=True)[:, 0]
        expected *= (config.rgb_hidden_dim / config.goal_hidden_dim) ** 0.5
        torch.testing.assert_close(policy.future_3d.state_dict()[key], expected, atol=0, rtol=0)

    # The full-checkpoint construction path must never transfer/scale weights again.
    monkeypatch.setattr(f"{module}.read_checkpoint", lambda path: pytest.fail("Resume must not read Wan assets"))
    restored = WSA1_5_Policy(config, initialize_pretrained=False)
    restored.load_state_dict(policy.state_dict(), strict=True)
    assert len(calls) == (2 if initialization == "wan_width_interpolation" else 1)
    for key, value in policy.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[key], value, atol=0, rtol=0)


@pytest.mark.parametrize("mode", ["full", "frozen", "lora"])
def test_stage2_to_stage1_constructor_transfers_world_and_merges_lora(tmp_path, monkeypatch, caplog, mode):
    from dataclasses import replace
    from test_wsa1_5_action import action_policy

    source = action_policy(enable_motion=True, rgb_finetune_mode=mode, rgb_lora_rank=2, lambda_rgb=1)
    source.vae.register_buffer("checkpoint_marker", torch.tensor([3.0]))
    with torch.no_grad():
        for layer in source.flow.rgb.modules():
            if isinstance(layer, LoRALinear):
                layer.lora_b.fill_(0.025)
    source.eval().save_pretrained(tmp_path)
    template = action_policy(enable_action=False, enable_motion=True, lambda_rgb=1)
    template.vae.register_buffer("checkpoint_marker", torch.zeros(1))

    def components(self, initialize):
        assert not initialize, "Weight initialization must not reload the original Wan/VAE assets"
        return deepcopy(WorldComponents(template.flow, template.future_3d, template.vae))

    monkeypatch.setattr(WSA1_5_Policy, "_build_components", components)
    caplog.set_level("INFO")
    config = replace(template.config, init_checkpoint_path=str(tmp_path))
    restored = WSA1_5_Policy(config)
    assert restored.action_expert is None
    assert not any("lora_" in key or key.startswith("action_expert.") for key in restored.state_dict())
    for key, value in source.state_dict().items():
        if key.startswith(("action_expert.", "proprio_encoder.")) or ".lora_" in key:
            continue
        if key.endswith(".weight"):
            layer = source.get_submodule(key.removesuffix(".weight"))
            if isinstance(layer, LoRALinear):
                value = value + (layer.lora_b @ layer.lora_a) * layer.scale
        torch.testing.assert_close(restored.state_dict()[key], value, atol=0, rtol=0)
    probe = torch.randn(2, 16)
    torch.testing.assert_close(restored.flow.rgb.blocks[0].self_attn.q(probe),
                               source.flow.rgb.blocks[0].self_attn.q(probe))
    assert restored.flow.rgb.patch_embedding.weight.requires_grad
    assert "omitted_action=" in caplog.text and "optimizer/scheduler/RNG/step start fresh" in caplog.text

    # A full reload must remain independent of the initialization checkpoint.
    config.init_checkpoint_path = "unavailable/stage2"
    restored.save_pretrained(tmp_path / "recovered")
    reloaded = WSA1_5_Policy.from_pretrained(tmp_path / "recovered")
    for key, value in restored.state_dict().items():
        torch.testing.assert_close(reloaded.state_dict()[key], value, atol=0, rtol=0)


@pytest.mark.parametrize("source_online,target_online", [(False, True), (True, False), (True, True), (False, False)])
def test_stage2_to_stage1_frozen_text_encoder(tmp_path, source_online, target_online):
    from test_wsa1_5_action import action_policy

    source = action_policy(load_text_encoder=source_online)
    original_text = nn.Linear(8, 8).eval().requires_grad_(False)
    if source_online:
        source.text_encoder = original_text
    source.save_pretrained(tmp_path)
    original_path = tmp_path / "original_umt5.pth"
    torch.save(original_text.state_dict(), original_path)
    target = action_policy(enable_action=False, lambda_rgb=1, load_text_encoder=target_online,
                           text_encoder_path=str(original_path))
    if target_online:
        target.text_encoder = nn.Linear(8, 8).eval().requires_grad_(False)
    report = load_world_policy(target, tmp_path)
    assert not report.missing_keys and not report.unexpected_keys
    if target_online:
        for key, value in original_text.state_dict().items():
            torch.testing.assert_close(target.text_encoder.state_dict()[key], value, atol=0, rtol=0)


@pytest.mark.parametrize("damage", ["missing_world", "extra_world", "shape", "nonfinite",
                                   "missing_lora", "extra_lora", "lora_shape", "lora_nonfinite",
                                   "missing_all_lora", "undeclared_text", "missing_declared_text"])
def test_stage2_to_stage1_rejects_incomplete_or_invalid_world(tmp_path, damage):
    from safetensors.torch import save_file
    from test_wsa1_5_action import action_policy

    source = action_policy(rgb_finetune_mode="lora", rgb_lora_rank=2,
                           load_text_encoder=damage == "missing_declared_text")
    if source.config.load_text_encoder:
        source.text_encoder = nn.Linear(8, 8)
    source.save_pretrained(tmp_path)
    state = dict(source.state_dict())
    world_key = "flow.rgb.patch_embedding.weight"
    lora_key = "flow.rgb.blocks.0.self_attn.q.lora_a"
    if damage == "missing_world":
        del state[world_key]
    elif damage == "extra_world":
        state["flow.rgb.stray.weight"] = torch.zeros(1)
    elif damage in {"shape", "lora_shape"}:
        key = lora_key if damage == "lora_shape" else world_key
        state[key] = state[key][:1]
    elif damage in {"nonfinite", "lora_nonfinite"}:
        state[lora_key if damage == "lora_nonfinite" else world_key].flatten()[0] = float("nan")
    elif damage == "missing_lora":
        del state[lora_key]
    elif damage == "extra_lora":
        state["flow.rgb.stray.lora_a"] = torch.zeros(1)
    elif damage == "missing_all_lora":
        state = {key: value for key, value in state.items() if ".lora_" not in key}
    elif damage == "undeclared_text":
        state["text_encoder.weight"] = torch.zeros(8, 8)
    else:
        del state["text_encoder.weight"]
    save_file(state, tmp_path / "model.safetensors")
    target = action_policy(enable_action=False, lambda_rgb=1, load_text_encoder=source.config.load_text_encoder)
    if target.config.load_text_encoder:
        target.text_encoder = nn.Linear(8, 8)
    before = target.flow.rgb.patch_embedding.weight.detach().clone()
    with pytest.raises((ValueError, RuntimeError), match="mismatch|Incomplete|finite|undeclared"):
        load_world_policy(target, tmp_path)
    torch.testing.assert_close(target.flow.rgb.patch_embedding.weight, before, atol=0, rtol=0)


def test_stage2_to_stage1_offline_source_requires_original_umt5(tmp_path):
    from test_wsa1_5_action import action_policy

    action_policy().save_pretrained(tmp_path)
    target = action_policy(enable_action=False, lambda_rgb=1, load_text_encoder=True)
    target.text_encoder = nn.Linear(8, 8)
    with pytest.raises(ValueError, match="text_encoder_path"):
        load_world_policy(target, tmp_path)
