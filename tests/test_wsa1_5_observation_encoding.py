"""Exercise encoding routes with small random experts and a recording VAE double."""
import pytest
import torch
from torch import nn
import torch.nn.functional as F

from lerobot.policies.wsa1_5.configuration_wsa1_5 import WSA1_5Config
from lerobot.policies.wsa1_5.future_3d import Future3DExpert
from lerobot.policies.wsa1_5.modeling_wsa1_5 import WSA1_5_Policy, WorldComponents
from test_wsa1_5_experts import small_flow


class RecordingVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    @torch.no_grad()
    def encode(self, video):
        self.calls.append(tuple(video.shape))
        first = video[:, :, :1]
        if video.shape[2] > 1:
            future = video[:, :, 1:].unflatten(2, (-1, 4)).mean(3)
            video = torch.cat([first, future], 2)
        pooled = F.avg_pool3d(video, (1, 16, 16))
        return torch.cat([pooled, pooled.mean(1, keepdim=True)], 1)

    @torch.no_grad()
    def decode(self, latent):
        return F.interpolate(latent[:, :3], size=(33, latent.shape[-2] * 16, latent.shape[-1] * 16), mode="nearest")


def make_policy(monkeypatch, *, enable_rgb=True, enable_motion=True, enable_3d_queries=False,
                **loss_weights):
    config = WSA1_5Config(
        device="cpu", dtype="float32", push_to_hub=False,
        rgb_hidden_dim=16, rgb_ffn_dim=32, motion_hidden_dim=12, motion_ffn_dim=24,
        goal_hidden_dim=10, goal_ffn_dim=20, num_layers=3, num_heads=2,
        head_dim=8, latent_channels=4, text_dim=8, freq_dim=8,
        rgb_height=96, rgb_width=32, motion_height=32, motion_width=32,
        enable_rgb=enable_rgb, enable_motion=enable_motion, enable_3d_queries=enable_3d_queries,
        **loss_weights,
    )
    vae = RecordingVAE()
    future = Future3DExpert(10, 20, 3, 2, 8, 8) if enable_3d_queries else None
    policy = WSA1_5_Policy(config, components=WorldComponents(small_flow(), future, vae=vae))
    monkeypatch.setattr(policy, "_encode_text", lambda texts: (
        torch.zeros(len(texts), 3, 8), torch.ones(len(texts), 3, dtype=torch.bool)))
    return policy, vae


@pytest.mark.parametrize("enable_rgb", [True, False])
def test_training_encoding_routes_and_backward(monkeypatch, enable_rgb):
    policy, vae = make_policy(monkeypatch, enable_rgb=enable_rgb)
    image = torch.randn(1, 3, 96, 32)
    pixel_valid = torch.ones(1, 1, 96, 32, dtype=torch.bool)
    pixel_valid[:, :, 64:, 16:] = False
    batch = {
        "observed_rgb": image, "condition_text": ["move the object"],
        "observed_valid": pixel_valid,
        "targets": {"future_rgb": torch.randn(1, 32, 3, 96, 32),
                    "rgb_valid": torch.ones(1, 32, dtype=torch.bool),
                    "future_motion": torch.randn(1, 32, 3, 32, 32),
                    "motion_valid": torch.ones(1, 32, dtype=torch.bool)},
    }
    conditions = []
    training_state = policy.flow.training_state

    def capture_condition(clean, condition):
        noisy, velocity, timestep = training_state(clean, condition)
        torch.testing.assert_close(noisy[:, :, :1], condition)
        conditions.append(condition.clone())
        return noisy, velocity, timestep

    monkeypatch.setattr(policy.flow, "training_state", capture_condition)
    velocity = policy.flow.velocity

    def capture_observation(observed, *args, **kwargs):
        expected = F.avg_pool3d(image[:, :, None], (1, 16, 16))
        expected = torch.cat([expected, expected.mean(1, keepdim=True)], 1)
        torch.testing.assert_close(observed, expected)
        expected_valid = F.interpolate(pixel_valid.float(), size=(6, 2), mode="area") >= 1 - 1e-6
        torch.testing.assert_close(kwargs["observed_valid"], expected_valid[:, :, None])
        assert not kwargs["motion_valid"][:, :, 0].any()
        assert kwargs["motion_valid"][:, :, 1:].all()
        return velocity(observed, *args, **kwargs)

    monkeypatch.setattr(policy.flow, "velocity", capture_observation)
    loss, metrics = policy.training_forward(batch)
    assert torch.isfinite(loss) and metrics["loss"] == loss.item()
    assert all(metrics[key] >= 0 for key in ["input_s", "mot_s", "pred_s"])
    loss.backward()
    assert policy.flow.motion.patch_embedding.weight.grad.abs().sum() > 0
    if enable_rgb:
        assert policy.flow.rgb.patch_embedding.weight.grad.abs().sum() > 0
        assert vae.calls == [(1, 3, 33, 96, 32), (1, 3, 33, 32, 32), (1, 3, 1, 32, 32)]
        assert conditions[0].shape == (1, 4, 1, 6, 2)
    else:
        assert vae.calls == [(1, 3, 33, 32, 32), (1, 3, 1, 96, 32), (1, 3, 1, 32, 32)]
    assert not conditions[-1].any()  # Motion still uses its fixed-zero placeholder.


def test_observation_without_training_targets_still_encodes_single_frame(monkeypatch):
    policy, vae = make_policy(monkeypatch)
    latent, _, _, valid, _ = policy.encode_observation(torch.randn(1, 3, 96, 32), ["move"])
    assert vae.calls == [(1, 3, 1, 96, 32)]
    assert latent.shape == (1, 4, 1, 6, 2) and valid.all()


def test_reused_observation_latent_is_validated(monkeypatch):
    policy, vae = make_policy(monkeypatch)
    image = torch.zeros(1, 3, 96, 32)
    with pytest.raises(ValueError, match="VAE output shape"):
        policy.encode_observation(image, ["move"], latent=torch.zeros(1, 4, 9, 6, 2))
    with pytest.raises(FloatingPointError, match="latent"):
        policy.encode_observation(image, ["move"], latent=torch.full((1, 4, 1, 6, 2), float("nan")))
    assert not vae.calls


@pytest.mark.parametrize("modalities", [(True, True), (True, False), (False, True), (False, False)])
def test_joint_training_has_one_mixed_forward_and_3d_gradients(monkeypatch, modalities):
    enable_rgb, enable_motion = modalities
    policy, _ = make_policy(monkeypatch, enable_rgb=enable_rgb, enable_motion=enable_motion,
                            enable_3d_queries=True, lambda_rgb=0, lambda_motion=0)
    image = torch.randn(1, 3, 96, 32)
    batch = {
        "observed_rgb": image, "condition_text": ["move"],
        "observed_valid": torch.ones(1, 1, 96, 32, dtype=torch.bool),
        "targets": {"future_rgb": torch.randn(1, 32, 3, 96, 32),
                    "rgb_valid": torch.ones(1, 32, dtype=torch.bool),
                    "future_motion": torch.randn(1, 32, 3, 32, 32),
                    "motion_valid": torch.ones(1, 32, dtype=torch.bool),
                    "goal_rgb_views": [[torch.ones(3, 16, 16)] * 3],
                    "goal_view_valid": torch.ones(1, 3, dtype=torch.bool)},
    }

    def teacher_features(views, valid, device):
        # Deterministic endpoint labels; never construct or load Omega.
        feature = torch.arange(2048, device=device).float().add(1)
        feature = feature + views[0][0].mean().to(device)
        return ({layer: feature.expand(1, 3, 273, -1) for layer in policy.config.teacher_layers},
                valid[:, :, None].expand(-1, -1, 273))

    monkeypatch.setattr(policy.teacher, "features", teacher_features)
    monkeypatch.setattr(policy, "sample_generated_latents",
                        lambda *args, **kwargs: pytest.fail("Training must not run a sampler"))
    monkeypatch.setattr(policy.flow, "sample_latents",
                        lambda *args, **kwargs: pytest.fail("Training must not run a sampler"))
    calls, representations = [], []

    def record_mixed_forward(module, args, kwargs, output):
        states = args[0]
        calls.append(set(states))
        representations.append({name: value.detach().clone() for name, value in output[0].items()})

    policy.flow.mot.register_forward_hook(record_mixed_forward, with_kwargs=True)
    torch.manual_seed(23)
    loss, _ = policy.training_forward(batch)
    expected_groups = {"O", "G"} | ({"R"} if enable_rgb else set()) | ({"M"} if enable_motion else set())
    assert calls == [expected_groups]
    loss.backward()
    grad = policy.future_3d.blocks[0].self_attn.q.weight.grad
    assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0
    for enabled, expert in [(enable_rgb, policy.flow.rgb), (enable_motion, policy.flow.motion)]:
        if enabled:
            grad = expert.blocks[0].self_attn.v.weight.grad
            assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0
    # Changing only endpoint labels cannot alter any student representation.
    batch["targets"]["goal_rgb_views"] = [[torch.zeros(3, 16, 16)] * 3]
    torch.manual_seed(23)
    policy.training_forward(batch)
    assert len(calls) == 2
    for group in expected_groups:
        torch.testing.assert_close(representations[0][group], representations[1][group], rtol=0, atol=0)
    assert policy.teacher.model is None
