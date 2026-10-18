import pytest
import torch

from lerobot.policies.wsa1_5.future_3d import Future3DExpert, geometric_feature_loss
from lerobot.policies.wsa1_5.teacher import OmegaTeacher, preprocess_omega_views
from lerobot.policies.wsa1_5.vendor.wan_video_dit import sinusoidal_embedding_1d
from test_wsa1_5_experts import small_flow, inputs


def test_3d_receives_text_only_through_visual_states_in_later_layers():
    flow = small_flow().eval()
    future = Future3DExpert(10, 20, 3, 2, 8, 8).eval()
    obs, text, valid, rgb, motion, rt, mt = inputs()
    text.requires_grad_()
    goal_valid = torch.ones(2, 3, dtype=torch.bool)

    def run(context):
        states, experts, _, _ = flow.prepare_states(obs, context, valid, rgb, motion, rt, mt)
        states["G"] = future.prepare(2, goal_valid, num_train_timesteps=1000)
        experts["G"] = future
        assert states["G"].context is None and states["G"].context_valid is None
        assert all(not hasattr(block, "cross_attn") for block in future.blocks)
        return flow.mot(states, experts, capture_layers=(0, 1))

    _, before = run(text)
    _, after = run(text + 1)
    torch.testing.assert_close(before[0]["G"], after[0]["G"], rtol=0, atol=0)
    assert not torch.allclose(before[1]["G"], after[1]["G"])
    before[1]["G"].square().mean().backward()
    assert text.grad is not None and torch.isfinite(text.grad).all() and text.grad.abs().sum() > 0


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


def test_fixed_square_preprocessing_preserves_full_image_without_padding():
    wide = torch.zeros(3, 40, 400)
    wide[:, :, :40] = 0.25
    wide[:, :, -40:] = 0.75
    tall = wide.transpose(-1, -2)
    images = preprocess_omega_views([wide, tall])
    assert images.shape == (2, 3, 256, 256)
    assert (images[0, :, :, :20] == 0.25).all()
    assert (images[0, :, :, -20:] == 0.75).all()
    assert (images[1, :, :20] == 0.25).all()
    assert (images[1, :, -20:] == 0.75).all()
    assert not images[:, :, 100:150, 100:150].any()


def test_preprocessing_matches_bilinear_without_pixel_quantization():
    view = torch.tensor([[[0.1, 0.2], [0.3, 0.4]]]).expand(3, -1, -1).clone().requires_grad_()
    actual = preprocess_omega_views([view])
    expected = torch.nn.functional.interpolate(view.detach()[None], size=(256, 256),
                                                mode="bilinear", align_corners=False)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert actual.dtype == torch.float32 and not actual.requires_grad
    torch.testing.assert_close(actual[0, :, 0, 0], torch.full((3,), 0.1), atol=0, rtol=0)


@pytest.mark.parametrize("height,width", [(80, 160), (160, 80), (100, 100), (150, 160)])
def test_single_or_identical_views_always_square(height, width):
    for count in [1, 3]:
        images = preprocess_omega_views([torch.zeros(3, height, width)] * count)
        assert images.shape == (count, 3, 256, 256)
        assert not images.any()


@pytest.mark.parametrize("pixel", [float("nan"), -0.1, 1.1])
def test_teacher_preprocessing_rejects_bad_pixels(pixel):
    with pytest.raises(ValueError):
        preprocess_omega_views([torch.full((3, 16, 16), pixel)])


def test_teacher_uses_native_fixed_tokens_without_feature_resampling(monkeypatch):
    from types import SimpleNamespace
    teacher = OmegaTeacher("unused", layers=(4, 11), dtype=torch.float32)
    views = [torch.zeros(3, 80, 160), torch.full((3, 16, 16), float("nan")), torch.zeros(3, 100, 100)]
    expected_images = preprocess_omega_views([views[0], views[0], views[2]])
    native = torch.arange(3 * 273 * 2048, dtype=torch.float32).reshape(1, 3, 273, 2048)
    calls = []

    def aggregator(images):
        torch.testing.assert_close(images, expected_images[None], atol=0, rtol=0)
        calls.append(images.shape)
        hidden = [None] * 24
        hidden[4], hidden[11] = native, native + 11
        return hidden, 17

    teacher.model = SimpleNamespace(aggregator=aggregator)
    monkeypatch.setattr(teacher, "_load", lambda device: None)
    targets, valid = teacher.features([views], torch.tensor([[True, False, True]]), torch.device("cpu"))
    assert calls == [torch.Size([1, 3, 3, 256, 256])]
    for layer, expected in [(4, native), (11, native + 11)]:
        torch.testing.assert_close(targets[layer][:, [0, 2]], expected[:, [0, 2]], atol=0, rtol=0)
        assert not targets[layer][:, 1].any() and not targets[layer].requires_grad
    assert valid.dtype == torch.bool and valid[0, [0, 2]].all()
    assert not valid[:, 1].any()
    student = {layer: torch.randn_like(value, requires_grad=True) for layer, value in targets.items()}
    loss, _ = geometric_feature_loss(student, targets, valid)
    loss.backward()
    for value in student.values():
        assert torch.isfinite(value.grad).all() and value.grad[valid].abs().sum() > 0
        assert not value.grad[~valid].any()


@pytest.mark.parametrize("tokens,dim,start,bad_value", [(272, 2048, 17, False), (273, 1024, 17, False),
                                                       (273, 2048, 5, False), (273, 2048, 17, True)])
def test_teacher_rejects_wrong_fixed_sequence_or_nonfinite_features(monkeypatch, tokens, dim, start, bad_value):
    from types import SimpleNamespace
    teacher = OmegaTeacher("unused", layers=(4,), dtype=torch.float32)
    feature = torch.ones(1, 3, tokens, dim)
    if bad_value:
        feature[..., 0, 0] = float("nan")
    teacher.model = SimpleNamespace(aggregator=lambda images: ({4: feature}, start))
    monkeypatch.setattr(teacher, "_load", lambda device: None)
    with pytest.raises(FloatingPointError if bad_value else ValueError):
        teacher.features([[torch.zeros(3, 80, 160)] * 3], torch.tensor([[True, False, False]]), torch.device("cpu"))


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


def test_geometric_mse_matches_wsa_layernorm_formula_and_scale():
    generator = torch.Generator().manual_seed(42)
    prediction = torch.randn(1, 3, 7, 2048, generator=generator, requires_grad=True)
    target = torch.randn(1, 3, 7, 2048, generator=generator, requires_grad=True)
    valid = torch.ones(1, 3, 7, dtype=torch.bool)
    loss, metrics = geometric_feature_loss({4: prediction}, {4: target}, valid, cosine_weight=0)
    expected = torch.nn.functional.mse_loss(torch.nn.functional.layer_norm(prediction, (2048,)),
                                            torch.nn.functional.layer_norm(target.detach(), (2048,)))
    torch.testing.assert_close(loss, expected)
    assert 1.8 < loss.item() < 2.2
    assert .9 < metrics["3d/layer_4_cosine"] < 1.1
    loss.backward()
    assert prediction.grad.abs().sum() > 0 and target.grad is None
