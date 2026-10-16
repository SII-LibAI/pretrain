"""Synthetic checkpoint tensors and tiny modules only; no pretrained model loading."""
from pathlib import Path

import pytest
import torch
from torch import nn

from lerobot.policies.wsa1_5.initialization import interpolate_wan_state
from lerobot.policies.wsa1_5.modeling_wsa1_5 import WSA1_5_Policy
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
