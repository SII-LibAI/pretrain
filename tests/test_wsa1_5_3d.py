import pytest
import torch
import numpy as np
from PIL import Image

from lerobot.policies.wsa1_5.future_3d import Future3DExpert, geometric_feature_loss
from lerobot.policies.wsa1_5.teacher import FeatureAligner3D, OmegaTeacher, preprocess_omega_views
from lerobot.policies.wsa1_5.vendor.omega_preprocessing import load_and_preprocess_images
from lerobot.policies.wsa1_5.vendor.wan_video_dit import sinusoidal_embedding_1d
from test_wsa1_5_experts import small_flow, inputs


def test_slot_noise_training_matches_wsa_default_and_resamples():
    future = Future3DExpert(10, 20, 3, 2, 8, 8)
    view_valid = torch.tensor([[True, True, False], [True, False, False]])
    torch.manual_seed(27)
    rng = torch.random.get_rng_state()
    expected_noise = torch.randn(2, 819, 10)
    torch.random.set_rng_state(rng)
    first = future.prepare(2, view_valid, num_train_timesteps=1000)
    positions = torch.arange(819, dtype=torch.float32)
    slots = sinusoidal_embedding_1d(10, positions)[None]
    torch.testing.assert_close(first.tokens, slots * 0.5 + expected_noise * 0.25, rtol=0, atol=0)
    torch.testing.assert_close(first.valid, view_valid.repeat_interleave(273, 1))
    assert not first.tokens.requires_grad
    assert not torch.equal(first.tokens[0], first.tokens[1])
    second = future.prepare(2, view_valid, num_train_timesteps=1000)
    assert not torch.equal(first.tokens, second.tokens)
    assert "query_tokens" not in dict(future.named_parameters())


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("training", [True, False])
def test_slot_noise_fixed_inference_matches_wsa_and_preserves_rng(dtype, training):
    future = Future3DExpert(10, 20, 3, 2, 8, 8).to(dtype=dtype).train(training)
    view_valid = torch.ones(2, 3, dtype=torch.bool)
    rng = torch.random.get_rng_state().clone()
    with torch.no_grad():
        first = future.prepare(2, view_valid, num_train_timesteps=1000)
        second = future.prepare(2, view_valid, num_train_timesteps=1000)
    torch.testing.assert_close(torch.random.get_rng_state(), rng, rtol=0, atol=0)
    fixed_noise = torch.empty(1, 819, 10).normal_(generator=torch.Generator().manual_seed(0)).to(dtype)
    slots = sinusoidal_embedding_1d(10, torch.arange(819, dtype=torch.float32))[None].to(dtype)
    torch.testing.assert_close(first.tokens, (slots * 0.5 + fixed_noise * 0.25).expand(2, -1, -1), rtol=0, atol=0)
    torch.testing.assert_close(first.tokens, second.tokens, rtol=0, atol=0)
    if not training:
        # eval() alone must select fixed noise even when gradients remain enabled.
        third = future.prepare(2, view_valid, num_train_timesteps=1000)
        torch.testing.assert_close(first.tokens, third.tokens, rtol=0, atol=0)
    assert first.tokens.dtype == dtype
    assert not {"slot_pos_embed", "inference_noise_tokens", "query_tokens"} & set(future.state_dict())


@pytest.mark.parametrize("num_train_timesteps", [1000, 2000])
def test_slot_noise_time_modulation_uses_constant_sigma(num_train_timesteps):
    flow = small_flow()
    flow.scheduler.num_train_timesteps = num_train_timesteps
    future = Future3DExpert(10, 20, 3, 2, 8, 8)
    obs, text, valid, rgb, motion, rt, mt = inputs()
    recorded = []
    future.time_embedding.register_forward_pre_hook(lambda module, args: recorded.append(args[0].detach()))
    flow.velocity(obs, text, valid, rgb, motion, rt, mt, future_3d=future,
                  goal_view_valid=torch.ones(2, 3, dtype=torch.bool))
    assert len(recorded) == 1
    expected = sinusoidal_embedding_1d(8, torch.full((2,), num_train_timesteps * 0.5))
    torch.testing.assert_close(recorded[0], expected, rtol=0, atol=0)


def test_slot_noise_state_roundtrip_and_old_learned_query_rejection():
    future = Future3DExpert(10, 20, 3, 2, 8, 8).eval()
    restored = Future3DExpert(10, 20, 3, 2, 8, 8).eval()
    state = future.state_dict()
    restored.load_state_dict(state, strict=True)
    view_valid = torch.ones(1, 3, dtype=torch.bool)
    original = future.prepare(1, view_valid, num_train_timesteps=1000)
    reloaded = restored.prepare(1, view_valid, num_train_timesteps=1000)
    torch.testing.assert_close(original.tokens, reloaded.tokens, rtol=0, atol=0)
    assert not {id(buffer) for buffer in future.buffers()} & {id(param) for param in future.parameters()}
    with pytest.raises(RuntimeError, match="Unexpected key.*query_tokens"):
        restored.load_state_dict({**state, "query_tokens": torch.zeros(1, 819, 10)}, strict=True)


def test_slot_positional_encoding_rejects_odd_hidden_width():
    with pytest.raises(ValueError, match="even hidden_dim"):
        Future3DExpert(hidden_dim=9)


def test_official_preprocessing_geometry():
    images, geometry = preprocess_omega_views([torch.zeros(3, 40, 400), torch.ones(3, 300, 100)])
    assert images.shape == (2, 3, 256, 256)
    assert (geometry[0].height, geometry[0].width) == (128, 256)
    assert (geometry[1].height, geometry[1].width) == (256, 128)
    assert geometry[0].pad_top == 64 and geometry[1].pad_left == 64
    assert (images[0, :, :64] == 1).all()  # Official white padding.


def test_pixel_exact_official_preprocessing(tmp_path):
    rng = np.random.default_rng(31)
    tensors, files = [], []
    for index, shape in enumerate([(41, 401, 3), (301, 103, 3), (111, 113, 3)]):
        pixels = rng.integers(0, 256, shape, dtype=np.uint8)
        path = tmp_path / f"view{index}.png"
        Image.fromarray(pixels).save(path)
        files.append(path)
        tensors.append(torch.from_numpy(pixels.copy()).permute(2, 0, 1).float() / 255)
    actual, _ = preprocess_omega_views(tensors)
    expected = load_and_preprocess_images(files, mode="max_size", image_resolution=256)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_variable_teacher_tokens_and_special_token_order():
    images, geometry = preprocess_omega_views([torch.zeros(3, 80, 160)] * 3)
    assert images.shape[-2:] == (128, 256)
    tokens = 17 + 8 * 16
    features = torch.randn(2, 3, tokens, 20)
    aligned, valid = FeatureAligner3D()(features, geometry)
    assert aligned.shape == (2, 3, 273, 20) and valid.all()
    torch.testing.assert_close(aligned[:, :, :17], features[:, :, :17], atol=0, rtol=0)
    with pytest.raises(ValueError):
        FeatureAligner3D()(features[:, :, :-1], geometry)


def test_square_alignment_identity():
    _, geometry = preprocess_omega_views([torch.zeros(3, 100, 100)] * 3)
    x = torch.randn(1, 3, 273, 10)
    aligned, valid = FeatureAligner3D()(x, geometry)
    torch.testing.assert_close(aligned, x, atol=1e-6, rtol=1e-5)
    assert valid.all()


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_student_joint_states_backward(dtype):
    flow = small_flow(True)
    future = Future3DExpert(hidden_dim=10, ffn_dim=20, layers=3, heads=2, head_dim=8, freq_dim=8, output_dim=20)
    obs, text, valid, rgb, motion, rt, mt = inputs()
    # R/M may be noisy flow states; endpoint/teacher data is
    # never accepted by prepare_states or Future3DExpert.prepare.
    with torch.autocast("cpu", dtype=dtype, enabled=dtype != torch.float32):
        states, experts, _, _ = flow.prepare_states(obs, text, valid, rgb, motion, rt * 0, mt * 0)
        states["G"] = future.prepare(2, torch.tensor([[True, True, True], [True, False, False]]),
                                      num_train_timesteps=flow.scheduler.num_train_timesteps)
        experts["G"] = future
        _, captured = flow.mot(states, experts, capture_layers=future.student_layers)
        predicted = future.project(captured)
    assert all(value.shape == (2, 3, 273, 20) for value in predicted.values())
    targets = {layer: torch.randn_like(value) for layer, value in predicted.items()}
    supervision = torch.ones(2, 3, 273, dtype=torch.bool)
    supervision[1, 1:] = False
    loss, metrics = geometric_feature_loss(predicted, targets, supervision)
    loss.backward()
    assert "query_tokens" not in dict(future.named_parameters())
    grad = future.blocks[0].self_attn.q.weight.grad
    assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0
    assert flow.rgb.patch_embedding.weight.grad.abs().sum() > 0
    assert flow.motion.patch_embedding.weight.grad.abs().sum() > 0
    assert len(metrics) == 8


def test_missing_teacher_views_do_not_load_model_or_consume_rng():
    teacher = OmegaTeacher("")
    before = torch.random.get_rng_state().clone()
    targets, valid = teacher.features([[torch.zeros(3, 16, 16)] * 3], torch.zeros(1, 3, dtype=torch.bool), torch.device("cpu"))
    assert teacher.model is None and not valid.any()
    assert all(value.shape == (1, 3, 273, 2048) for value in targets.values())
    torch.testing.assert_close(before, torch.random.get_rng_state(), atol=0, rtol=0)


def test_3d_does_not_change_rgb_motion_outputs():
    flow = small_flow()
    obs, text, valid, rgb, motion, rt, mt = inputs()
    states, experts, _, _ = flow.prepare_states(obs, text, valid, rgb, motion, rt, mt)
    before, _ = flow.mot(states, experts)
    future = Future3DExpert(hidden_dim=10, ffn_dim=20, layers=3, heads=2, head_dim=8, freq_dim=8, output_dim=20)
    states["G"] = future.prepare(2, torch.ones(2, 3, dtype=torch.bool),
                                  num_train_timesteps=flow.scheduler.num_train_timesteps)
    experts["G"] = future
    after, _ = flow.mot(states, experts)
    for group in ["O", "R", "M"]:
        torch.testing.assert_close(before[group], after[group], atol=0, rtol=0)


def test_3d_cannot_read_motion_placeholder():
    flow = small_flow(True)
    future = Future3DExpert(hidden_dim=10, ffn_dim=20, layers=3, heads=2, head_dim=8, freq_dim=8, output_dim=20).eval()
    obs, text, valid, rgb, motion, rt, mt = inputs()

    def run(value):
        states, experts, _, _ = flow.prepare_states(obs, text, valid, rgb, value, rt, mt)
        states["G"] = future.prepare(2, torch.ones(2, 3, dtype=torch.bool),
                                      num_train_timesteps=flow.scheduler.num_train_timesteps)
        experts["G"] = future
        tokens, _ = flow.mot(states, experts)
        return tokens

    before = run(motion)
    changed = motion.detach().clone()
    changed[:, :, 0] += 1000
    after = run(changed)
    for group in before:
        torch.testing.assert_close(before[group], after[group], atol=0, rtol=0)
    grad = torch.autograd.grad(before["G"].square().sum(), motion)[0]
    assert torch.isfinite(grad).all() and not grad[:, :, 0].any()
    assert grad[:, :, 1:].abs().sum() > 0


def test_joint_sampler_predicts_3d_in_each_step_without_extra_forward():
    flow = small_flow().eval()
    future = Future3DExpert(hidden_dim=10, ffn_dim=20, layers=3, heads=2, head_dim=8, freq_dim=8, output_dim=20).eval()
    obs, text, valid, *_ = inputs()
    captures = []

    def record_joint_forward(module, args, output):
        assert set(args[0]) == {"O", "R", "M", "G"}
        assert args[0]["M"].tokens.shape[1] == 8
        captures.append(output[1])

    flow.mot.register_forward_hook(record_joint_forward)
    with torch.no_grad():
        rgb, motion, features = flow.sample_latents(
            obs, text, valid, rgb_shape=(2, 4, 9, 4, 4), motion_shape=(2, 4, 9, 2, 2),
            motion_placeholder=torch.zeros(2, 4, 1, 2, 2), steps=2, seed=9,
            future_3d=future, goal_view_valid=torch.ones(2, 3, dtype=torch.bool), return_features=True)
        expected = future.project(captures[-1])
    assert len(captures) == 2
    assert rgb.shape == (2, 4, 9, 4, 4) and motion.shape == (2, 4, 9, 2, 2)
    for layer in features:
        torch.testing.assert_close(features[layer], expected[layer], rtol=0, atol=0)


def test_missing_views_and_padding_excluded_from_loss():
    student = {4: torch.randn(1, 3, 273, 10, requires_grad=True)}
    teacher = {4: torch.randn(1, 3, 273, 10)}
    valid = torch.ones(1, 3, 273, dtype=torch.bool)
    valid[:, 1:] = False
    loss, _ = geometric_feature_loss(student, teacher, valid)
    changed = {4: teacher[4].clone()}
    changed[4][:, 1:] += 100
    altered, _ = geometric_feature_loss(student, changed, valid)
    torch.testing.assert_close(loss, altered, atol=0, rtol=0)
    loss.backward()
    assert not student[4].grad[:, 1:].any()
