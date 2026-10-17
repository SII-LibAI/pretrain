import pytest
import torch

from lerobot.policies.wsa1_5.experts import RGBVideoExpert, MotionDiTExpert, WorldFlow, wan_config
from lerobot.policies.wsa1_5.initialization import interpolate_wan_state
from lerobot.policies.wsa1_5.temporal import TemporalLayout


def small_flow(checkpointing=False, *, future_frames=32):
    torch.manual_seed(13)
    rgb = RGBVideoExpert(**wan_config(16, 32, layers=3, heads=2, head_dim=8, channels=4, text_dim=8, freq_dim=8))
    motion = MotionDiTExpert(**wan_config(12, 24, layers=3, heads=2, head_dim=8, channels=4, text_dim=8, freq_dim=8))
    return WorldFlow(rgb, motion, gradient_checkpointing=checkpointing, future_frames=future_frames)


def inputs():
    return (torch.randn(2, 4, 1, 4, 4), torch.randn(2, 3, 8), torch.ones(2, 3, dtype=torch.bool),
            torch.randn(2, 4, 9, 4, 4, requires_grad=True), torch.randn(2, 4, 9, 2, 2, requires_grad=True),
            torch.tensor([200.0, 500.0]), torch.tensor([300.0, 600.0]))


@pytest.mark.parametrize("mode", ["rgb", "motion", "both"])
def test_expert_backward_and_target_isolation(mode):
    model = small_flow(True)
    obs, text, text_valid, rgb, motion, rt, mt = inputs()
    out = model.velocity(obs, text, text_valid, rgb if mode != "motion" else None,
                         motion if mode != "rgb" else None, rt, mt)
    for name, target, t in [("rgb", rgb, rt), ("motion", motion, mt)]:
        if name not in out:
            continue
        valid = torch.ones(2, 1, *target.shape[2:], dtype=torch.bool)
        loss = model.flow_loss(out[name], torch.randn_like(target), t, valid)
        loss.backward(retain_graph=True)
        expert = model.rgb if name == "rgb" else model.motion
        assert expert.patch_embedding.weight.grad.abs().sum() > 0
    if mode == "both":
        rgrad = torch.autograd.grad(out["rgb"].square().sum(), motion, retain_graph=True, allow_unused=True)[0]
        mgrad = torch.autograd.grad(out["motion"].square().sum(), rgb, allow_unused=True)[0]
        assert rgrad is None or not rgrad.any()
        assert mgrad is None or not mgrad.any()


def test_rgb_motion_perturbation_and_observation_isolation():
    model = small_flow()
    obs, text, valid, rgb, motion, rt, mt = inputs()
    before = model.velocity(obs, text, valid, rgb, motion, rt, mt)
    changed_rgb = rgb.detach().clone()
    changed_rgb[:, :, 0] += 100  # target's first latent cannot become the condition
    first = model.velocity(obs, text, valid, changed_rgb, motion, rt, mt)
    torch.testing.assert_close(first["rgb"], before["rgb"], rtol=0, atol=0)
    only_rgb = model.velocity(obs, text, valid, rgb, motion + 100, rt, mt)
    only_motion = model.velocity(obs, text, valid, rgb + 100, motion, rt, mt)
    torch.testing.assert_close(only_rgb["rgb"], before["rgb"], rtol=0, atol=0)
    torch.testing.assert_close(only_motion["motion"], before["motion"], rtol=0, atol=0)


@pytest.mark.parametrize("backend", ["sdpa", "math", "reference"])
@pytest.mark.parametrize("checkpointing", [False, True])
def test_motion_placeholder_never_enters_attention(monkeypatch, backend, checkpointing):
    model = small_flow(checkpointing)
    model.mot.backend = backend
    obs, text, text_valid, rgb, _, rt, mt = inputs()
    motion = torch.randn(2, 4, 9, 4, 4, requires_grad=True)
    valid = torch.ones(2, 1, 9, 4, 4, dtype=torch.bool)
    valid[:, :, 4, :, :2] = False
    original_valid = valid.clone()
    states, _, pre, _ = model.prepare_states(obs, text, text_valid, rgb, motion, rt, mt, motion_valid=valid)
    prefix_n = pre["M"]["meta"]["tokens_per_frame"]
    assert prefix_n == 4 and states["M"].tokens.shape[1] == 8 * prefix_n
    torch.testing.assert_close(states["M"].freqs, pre["M"]["freqs"][prefix_n:])
    expected_valid = valid[:, :, 1:].reshape(2, 1, 8, 2, 2, 2, 2).all((4, 6)).flatten(1)
    torch.testing.assert_close(states["M"].valid, expected_valid)
    torch.testing.assert_close(valid, original_valid)
    calls = []

    def check_future_queries_only(module, args):
        assert args[0].shape[1] == 8 * prefix_n
        calls.append(1)

    for block in model.motion.blocks:
        for projection in [block.self_attn.q, block.self_attn.k, block.self_attn.v, block.cross_attn.q]:
            projection.register_forward_pre_hook(check_future_queries_only)
    before = model.velocity(obs, text, text_valid, rgb, motion, rt, mt, motion_valid=valid)
    assert len(calls) == 4 * len(model.motion.blocks)
    assert before["motion"].shape == motion.shape and not before["motion"][:, :, 0].any()
    changed = motion.detach().clone()
    changed[:, :, 0] += 1000
    after = model.velocity(obs, text, text_valid, rgb, changed, rt, mt, motion_valid=valid)
    for name in before:
        torch.testing.assert_close(before[name], after[name], rtol=0, atol=0)
    before["motion"][:, :, 1:].square().sum().backward()
    assert torch.isfinite(motion.grad).all()
    assert not motion.grad[:, :, 0].any() and motion.grad[:, :, 1:].abs().sum() > 0


def test_motion_sampling_without_motion_observation():
    model = small_flow().eval()
    obs, text, valid, *_ = inputs()
    kwargs = dict(rgb_shape=(2, 4, 9, 4, 4), motion_shape=(2, 4, 9, 2, 2),
                  motion_placeholder=torch.zeros(2, 4, 1, 2, 2), steps=2, seed=9)
    with torch.no_grad():
        a = model.sample_latents(obs, text, valid, **kwargs)
        b = model.sample_latents(obs, text, valid, **kwargs)
    for x, y in zip(a, b, strict=True):
        torch.testing.assert_close(x, y, atol=0, rtol=0)
    torch.testing.assert_close(a[0][:, :, :1], obs)
    assert not a[1][:, :, 0].any()
    # The codec retains t0, but its value must not affect future sampling.
    kwargs["motion_placeholder"] = torch.full_like(kwargs["motion_placeholder"], 100)
    with torch.no_grad():
        changed = model.sample_latents(obs, text, valid, **kwargs)
    torch.testing.assert_close(changed[0], a[0], rtol=0, atol=0)
    torch.testing.assert_close(changed[1][:, :, 1:], a[1][:, :, 1:], rtol=0, atol=0)
    torch.testing.assert_close(changed[1][:, :, :1], kwargs["motion_placeholder"])


def test_temporal_mapping_masks_and_no_observed_motion():
    layout = TemporalLayout()
    future = torch.randn(2, 32, 3, 4, 4)
    padded = layout.prepend_motion_placeholder(future)
    assert padded.shape == (2, 3, 33, 4, 4) and not padded[:, :, 0].any()
    torch.testing.assert_close(layout.future_from_decoded(padded), future)
    flags = torch.ones(2, 32, dtype=torch.bool)
    flags[0, 8] = False
    mask = layout.latent_validity(flags, (2, 2))
    assert mask.shape == (2, 1, 8, 2, 2)
    assert not mask[0, :, 2].any() and mask[0, :, 1].all()
    assert not mask[0, :, 2:].any()  # Causal encoder history stays invalid after a missing frame.
    with pytest.raises(ValueError):
        layout.future_from_decoded(padded[:, :, :-1])


def test_initialization_groups_not_blended():
    source = {"time_projection.1.bias": torch.arange(6.0).repeat_interleave(16),
              "blocks.0.modulation": torch.arange(6.0).reshape(1, 6, 1).expand(1, 6, 16)}
    target = {"time_projection.1.bias": torch.zeros(6 * 12), "blocks.0.modulation": torch.zeros(1, 6, 12)}
    converted, rules = interpolate_wan_state(source, target)
    torch.testing.assert_close(converted["time_projection.1.bias"].reshape(6, 12), torch.arange(6.0)[:, None].expand(6, 12))
    torch.testing.assert_close(converted["blocks.0.modulation"], target["blocks.0.modulation"] + torch.arange(6.0)[None, :, None])
    assert "six separate" in rules["time_projection.1.bias"]


def test_masked_flow_and_first_frame_exclusion():
    flow = small_flow()
    target = torch.randn(2, 4, 9, 2, 2)
    prediction = target.clone().requires_grad_()
    prediction.data[:, :, 0] = 100
    valid = torch.ones(2, 1, 9, 2, 2, dtype=torch.bool)
    t = torch.tensor([400.0, 600.0])
    assert flow.flow_loss(prediction, target, t, valid).item() == 0
    valid[:] = False
    loss = flow.flow_loss(prediction, target, t, valid)
    loss.backward()
    assert loss.item() == 0 and torch.isfinite(prediction.grad).all()


def test_placeholder_prediction_and_target_have_zero_loss_gradients():
    flow = small_flow()
    prediction = torch.randn(2, 4, 9, 2, 2, requires_grad=True)
    target = torch.randn_like(prediction, requires_grad=True)
    valid = torch.ones(2, 1, 9, 2, 2, dtype=torch.bool)
    t = torch.tensor([400.0, 600.0])
    loss = flow.flow_loss(prediction, target, t, valid)
    changed_prediction, changed_target = prediction.detach().clone(), target.detach().clone()
    changed_prediction[:, :, 0] += 1000
    changed_target[:, :, 0] -= 1000
    torch.testing.assert_close(loss, flow.flow_loss(changed_prediction, changed_target, t, valid), rtol=0, atol=0)
    loss.backward()
    for grad in [prediction.grad, target.grad]:
        assert torch.isfinite(grad).all() and not grad[:, :, 0].any()
        assert grad[:, :, 1:].abs().sum() > 0


def test_bf16_autocast_with_fp32_master_parameters():
    model = small_flow(True)
    obs, text, valid, rgb, motion, rt, mt = inputs()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = model.velocity(obs, text, valid, rgb, motion, rt, mt)
        loss = sum(value.float().square().mean() for value in out.values())
    loss.backward()
    assert all(p.dtype == torch.float32 for p in model.parameters())
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


def test_sampler_masks_missing_rgb_slots():
    model = small_flow().eval()
    obs, text, valid, *_ = inputs()
    observed_valid = torch.ones(2, 1, 1, 4, 4, dtype=torch.bool)
    observed_valid[..., 2:] = False
    with torch.no_grad():
        rgb, _ = model.sample_latents(obs, text, valid, rgb_shape=(2, 4, 9, 4, 4), steps=2,
                                     observed_valid=observed_valid)
    assert not rgb[:, :, 1:, :, 2:].any()
