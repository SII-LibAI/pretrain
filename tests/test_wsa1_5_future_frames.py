"""Config-to-output horizon checks with synthetic video and tiny random experts only."""
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from lerobot.datasets.factory import _build_single_dataset
from lerobot.policies.wsa1_5.configuration_wsa1_5 import WSA1_5Config, WSA1_5DatasetConfig
from lerobot.policies.wsa1_5.modeling_wsa1_5 import WSA1_5_Policy, WorldComponents
from lerobot.policies.wsa1_5.utils.temporal import TemporalLayout
from test_wsa1_5_dataset import make_data
from test_wsa1_5_experts import small_flow
from test_wsa1_5_observation_encoding import RecordingVAE, make_policy
from test_wsa1_5_policy_contract import small_config


@pytest.mark.parametrize("frames", [0, -4, 1, 15, 17, True, 16.0])
def test_invalid_horizon_rejected_at_all_configuration_boundaries(frames):
    for constructor, kwargs in [(TemporalLayout, {}), (WSA1_5Config, {"device": "cpu"}),
                                (WSA1_5DatasetConfig, {"repo_id": "test"})]:
        with pytest.raises(ValueError, match="future_frames.*multiple of 4"):
            constructor(future_frames=frames, **kwargs)


@pytest.mark.parametrize("frames", [4, 12, 16, 32, 48])
@pytest.mark.parametrize("stride", [1, 3])
def test_config_controls_actual_decoded_frames_and_episode_padding(tmp_path, monkeypatch, frames, stride):
    dataset, decoded = make_data(tmp_path, monkeypatch, future_frames=frames, frame_stride=stride)
    for current in [0, 8, 39]:
        decoded.clear()
        sample = dataset[current]
        target = sample["targets"]
        requested = current + np.arange(1, frames + 1) * stride
        assert target["future_rgb"].shape == (frames, 3, 384, 320)
        assert target["future_motion"].shape == (frames, 3, 192, 160)
        np.testing.assert_allclose(sample["frame_timestamps"], np.minimum(requested, 39) / 30)
        torch.testing.assert_close(sample["future_frame_valid"], torch.from_numpy(requested < 40))
        torch.testing.assert_close(target["motion_valid"], sample["future_frame_valid"])
        # Endpoint stays tied to the current subtask, independently of the horizon.
        assert sample["goal_timestamp"] == pytest.approx((9 if current <= 9 else 39) / 30)
        goal_frame = 9 if current <= 9 else 39
        goal_in_window = goal_frame <= min(current + frames * stride, 39)
        assert len(decoded[0][1]) == frames + 1 + int(goal_in_window)
        assert len(decoded[1][1]) == frames
        if not goal_in_window:
            assert decoded[2][1] == pytest.approx([2 + goal_frame / 30])


@pytest.mark.parametrize("frames", [4, 12, 16, 32, 48])
def test_model_construction_and_saved_config_keep_horizon(tmp_path, monkeypatch, frames):
    monkeypatch.setattr("lerobot.policies.wsa1_5.vendor.wan_video_vae.WanVideoVAE38", RecordingVAE)
    config = small_config(future_frames=frames, enable_3d_queries=False)
    policy = WSA1_5_Policy(config, initialize_pretrained=False)
    assert policy.layout == policy.flow.layout == TemporalLayout(future_frames=frames)
    assert policy._shapes(2)[0][2] == 1 + frames // 4
    config._save_pretrained(tmp_path)
    assert json.loads((tmp_path / "config.json").read_text())["future_frames"] == frames


@pytest.mark.parametrize("frames", [4, 12, 16, 32, 48])
@pytest.mark.parametrize("queries", [False, True])
def test_horizon_reaches_vae_joint_forward_backward_and_prediction(monkeypatch, frames, queries):
    policy, vae = make_policy(monkeypatch, future_frames=frames, enable_3d_queries=queries)
    policy.config.sampler_steps = 2
    batch = {
        "observed_rgb": torch.randn(1, 3, 96, 32), "condition_text": ["move"],
        "observed_valid": torch.ones(1, 1, 96, 32, dtype=torch.bool),
        "targets": {"future_rgb": torch.randn(1, frames, 3, 96, 32),
                    "rgb_valid": torch.ones(1, frames, 1, 96, 32, dtype=torch.bool),
                    "future_motion": torch.randn(1, frames, 3, 32, 32),
                    "motion_valid": torch.ones(1, frames, dtype=torch.bool)},
    }
    if queries:
        batch["targets"].update(goal_rgb_views=[[torch.ones(3, 16, 16)] * 3],
                                 goal_view_valid=torch.ones(1, 3, dtype=torch.bool))
        def teacher_features(views, valid, device):
            feature = torch.arange(1, 2049, dtype=torch.float32, device=device).expand(1, 3, 273, -1)
            return ({layer: feature for layer in policy.config.teacher_layers},
                    valid[:, :, None].expand(-1, -1, 273))
        monkeypatch.setattr(policy.teacher, "features", teacher_features)

    seen = []
    def record_tokens(module, args):
        states = args[0]
        seen.append({name: state.tokens.shape[1] for name, state in states.items()})
    policy.flow.mot.register_forward_pre_hook(record_tokens)
    loss, metrics = policy.training_forward(batch)
    assert torch.isfinite(loss) and metrics["loss"] == loss.item()
    assert vae.calls == [(1, 3, frames + 1, 96, 32), (1, 3, frames + 1, 32, 32), (1, 3, 1, 32, 32)]
    assert len(seen) == 1 and seen[0]["R"] == 3 * frames // 4 and seen[0]["M"] == frames // 4
    loss.backward()
    assert policy.flow.rgb.patch_embedding.weight.grad.abs().sum() > 0
    assert policy.flow.motion.patch_embedding.weight.grad.abs().sum() > 0
    if queries:
        assert seen[0]["G"] == 819
        assert policy.future_3d.blocks[0].self_attn.q.weight.grad.abs().sum() > 0
    result = policy.predict_future(batch["observed_rgb"], batch["condition_text"],
                                   observed_valid=batch["observed_valid"])
    assert result["future_rgb"].shape == (1, frames, 3, 96, 32)
    assert result["future_motion"].shape == (1, frames, 3, 32, 32)
    assert result["future_rgb_valid"].shape == (1, frames, 1, 96, 32)
    assert len(seen) == 3  # One training forward and two inference steps.


@pytest.mark.parametrize("modality", ["rgb", "motion"])
def test_wrong_pixel_horizon_fails_before_encoding(monkeypatch, modality):
    policy, vae = make_policy(monkeypatch, future_frames=16, enable_rgb=modality == "rgb",
                               enable_motion=modality == "motion")
    size = (96, 32) if modality == "rgb" else (32, 32)
    batch = {"observed_rgb": torch.zeros(1, 3, 96, 32),
             "targets": {f"future_{modality}": torch.zeros(1, 32, 3, *size)}}
    with pytest.raises(ValueError, match=r"\[B,16,3,H,W\]"):
        policy.prepare_training_targets(batch)
    assert not vae.calls


def test_mismatched_dataset_policy_and_flow_horizons_are_rejected():
    config = small_config(future_frames=16)
    with pytest.raises(ValueError, match="temporal layout"):
        WSA1_5_Policy(config, components=WorldComponents(small_flow()))
    pipeline = SimpleNamespace(policy=config, dataset=WSA1_5DatasetConfig(repo_id="test", future_frames=32))
    with pytest.raises(ValueError, match="disagree on future_frames"):
        _build_single_dataset(pipeline, "test", None, 0)
    flow = small_flow(future_frames=16)
    wrong = torch.zeros(1, 4, 9, 2, 2)
    with pytest.raises(ValueError, match="5 latent timepoints"):
        flow.training_state(wrong, wrong[:, :, :1])
    with pytest.raises(ValueError, match="5 latent timepoints"):
        flow.sample_latents(wrong[:, :, :1], torch.zeros(1, 3, 8), torch.ones(1, 3, dtype=torch.bool),
                            rgb_shape=tuple(wrong.shape))
