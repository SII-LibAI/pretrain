"""CPU contracts only: tiny random experts, synthetic frames, no pretrained weights."""
import json
from dataclasses import replace
from types import SimpleNamespace

import draccus
import pytest
import torch
import torch.nn.functional as functional
from safetensors.torch import save_file
from test_wsa1_5_dataset import make_data
from torch import nn

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.policies.wsa1_5.attention import GROUPS, VISIBLE, build_causal_attention_mask
from lerobot.policies.wsa1_5.configuration_wsa1_5 import WSA1_5Config, WSA1_5DatasetConfig
from lerobot.policies.wsa1_5.dataset_wsa1_5 import ActionCodec, WSA1_5Dataset, action_prompt, wsa1_5_collate
from lerobot.policies.wsa1_5.experts import MotionDiTExpert, RGBVideoExpert, WorldFlow, wan_config
from lerobot.policies.wsa1_5.future_3d import Future3DExpert
from lerobot.policies.wsa1_5.initialization import LoRALinear, load_stage1_policy
from lerobot.policies.wsa1_5.modeling_wsa1_5 import WorldComponents, WSA1_5_Policy
from lerobot.policies.wsa1_5.utils.temporal import ObservationHistory, history_indices
from lerobot.policies.wsa1_5.text_embeddings import text_embedding_path


def action_config(**overrides):
    args = {"device": "cpu", "dtype": "float32", "push_to_hub": False, "enable_action": True,
                "state_dim": 2, "action_dim": 2, "action_chunk_size": 4, "action_hidden_dim": 12, "action_ffn_dim": 24,
                "rgb_hidden_dim": 16, "rgb_ffn_dim": 32, "motion_hidden_dim": 12, "motion_ffn_dim": 24,
                "goal_hidden_dim": 10, "goal_ffn_dim": 20, "num_layers": 3, "num_heads": 2, "head_dim": 8,
                "latent_channels": 4, "text_dim": 8, "freq_dim": 8, "max_text_length": 4,
                "rgb_height": 96, "rgb_width": 64, "motion_height": 32, "motion_width": 32, "future_frames": 4,
                "enable_motion": False, "lambda_rgb": 0, "lambda_3d": 0, "num_history_frame": 4,
                "rgb_finetune_mode": "frozen", "action_sampler_steps": 2, "gradient_checkpointing": True}
    args.update(overrides)
    return WSA1_5Config(**args)


class TinyVAE(nn.Module):
    def encode(self, video):
        selected = video[:, :, ::4].mean(1)
        b, t = selected.shape[:2]
        small = functional.interpolate(selected.reshape(b * t, 1, *video.shape[-2:]),
                              size=(video.shape[-2] // 16, video.shape[-1] // 16), mode="area")
        return small.reshape(b, t, 1, *small.shape[-2:]).transpose(1, 2).repeat(1, 4, 1, 1, 1)

    def decode(self, latent):
        frames = latent[:, :3].repeat_interleave(4, 2)[:, :, 3:]
        b, c, t, h, w = frames.shape
        decoded = functional.interpolate(frames.transpose(1, 2).reshape(b * t, c, h, w),
                                         size=(h * 16, w * 16), mode="nearest")
        return decoded.reshape(b, t, c, h * 16, w * 16).transpose(1, 2)


def action_policy(**kwargs):
    cfg = action_config(**kwargs)
    rgb = RGBVideoExpert(**wan_config(16, 32, layers=3, heads=2, head_dim=8, channels=4, text_dim=8, freq_dim=8))
    motion = (MotionDiTExpert(**wan_config(12, 24, layers=3, heads=2, head_dim=8, channels=4, text_dim=8, freq_dim=8))
              if cfg.enable_motion else None)
    flow = WorldFlow(rgb, motion, future_frames=4, gradient_checkpointing=cfg.gradient_checkpointing)
    goal = Future3DExpert(10, 20, 3, 2, 8, 8, output_dim=8)
    return WSA1_5_Policy(cfg, components=WorldComponents(flow, goal, TinyVAE()))


def action_batch(cfg):
    out = {"observed_rgb": torch.rand(1, 3, cfg.rgb_height, cfg.rgb_width) * 2 - 1,
               "observed_valid": torch.ones(1, 1, cfg.rgb_height, cfg.rgb_width, dtype=torch.bool),
               "condition_text": ["Task: move"], "state": torch.rand(1, cfg.state_dim), "state_valid": torch.ones(1, dtype=torch.bool),
               "text_embedding": torch.rand(1, cfg.max_text_length, cfg.text_dim),
               "text_valid": torch.ones(1, cfg.max_text_length, dtype=torch.bool),
               "targets": {"action": torch.rand(1, cfg.action_chunk_size, cfg.action_dim),
                        "action_valid_mask": torch.ones(1, cfg.action_chunk_size, cfg.action_dim, dtype=torch.bool)}}
    if cfg.num_history_frame:
        out.update(history_rgb=torch.rand(1, cfg.num_history_frame, 3, cfg.rgb_height, cfg.rgb_width) * 2 - 1,
                   history_valid=torch.ones(1, cfg.num_history_frame, 1, cfg.rgb_height, cfg.rgb_width, dtype=torch.bool))
    out["targets"].update(future_rgb=torch.rand(1, cfg.future_frames, 3, cfg.rgb_height, cfg.rgb_width),
                          rgb_valid=torch.ones(1, cfg.future_frames, 1, cfg.rgb_height, cfg.rgb_width, dtype=torch.bool))
    return out


def test_five_group_mask_and_padding():
    actual = build_causal_attention_mask(dict.fromkeys(GROUPS, 1))
    expected = torch.tensor([[1, 0, 0, 0, 0], [1, 1, 0, 0, 0], [1, 0, 1, 0, 0],
                             [1, 1, 1, 1, 0], [1, 0, 0, 1, 1]], dtype=torch.bool)
    torch.testing.assert_close(actual, expected)
    valid = {g: torch.ones(2, 1, dtype=torch.bool) for g in GROUPS}
    valid["G"][0] = False
    padded = build_causal_attention_mask(dict.fromkeys(GROUPS, 1), valid_tokens=valid)
    assert not padded[0, :, 3].any() and not padded[0, :, :, 3].any()
    torch.testing.assert_close(padded[1, 0], expected)
    assert VISIBLE["A"] == ("O", "G", "A")


@pytest.mark.parametrize("frame,count,stride,expected,valid", [
    (10, 4, 2, [2, 4, 6, 8, 10], [1, 1, 1, 1, 1]),
    (3, 4, 2, [0, 0, 0, 1, 3], [0, 0, 0, 1, 1]),
    (0, 4, 2, [0, 0, 0, 0, 0], [0, 0, 0, 0, 1]),
    (2, 0, 2, [2], [1]),
])
def test_history_indices_and_online_buffer(frame, count, stride, expected, valid):
    indices, mask = history_indices(frame, count, stride)
    assert indices.tolist() == expected and mask.tolist() == valid
    buffer = ObservationHistory(count, stride)
    for i in range(frame + 1):
        buffer.append(torch.full((1, 3, 2, 2), float(i)), torch.ones(1, 1, 2, 2, dtype=torch.bool))
    frames, masks = buffer.sample()
    assert frames[0, :, 0, 0, 0].tolist() == expected
    assert masks[0, :, 0, 0, 0].tolist() == valid


def make_action_data(tmp_path, monkeypatch, *, mode="delta", use_subtask=False, allow_endframe=True, annotations=True):
    fixture, seen = make_data(tmp_path, monkeypatch, annotations=annotations, future_frames=4)
    base = fixture.base
    base.meta.robot_type = "aloha_robotwin"
    from lerobot.dataset_schemas import get_schema
    schema = get_schema("aloha_robotwin")
    # Real schema: xyz + quaternion + gripper per arm = 16.
    dimensions = [3, 4, 1, 3, 4, 1]
    stats = {}
    for keys, offset in [(schema.get_state_keys(), 10), (schema.get_action_keys(), 20)]:
        for key, dim in zip(keys, dimensions, strict=True):
            base.meta.features[key] = {"dtype": "float32", "shape": [dim]}
            base.hf_dataset[key] = [[float(i + offset)] * dim for i in range(40)]
            stats[key] = {"mean": [1.] * dim, "std": [2.] * dim}
    base.meta.stats = stats
    path = tmp_path / "action_stats.json"
    path.write_text(json.dumps(stats), encoding="utf-8")
    (tmp_path / "meta/info.json").write_text(json.dumps({"robot_type": "aloha_robotwin"}), encoding="utf-8")
    cfg = action_config(state_dim=16, action_dim=16, action_chunk_size=5, num_history_frame=4)
    cfg.action_mode = mode
    data_cfg = replace(fixture.config, use_subtask=use_subtask, allow_endframe_3d=allow_endframe,
                       action_mode=mode, use_external_stats=True, external_stats_path=str(path))
    ds = WSA1_5Dataset(base, data_cfg, action_config=cfg)
    return ds, seen


@pytest.mark.parametrize("mode", ["abs", "delta"])
def test_action_dataset_alignment_masks_normalization(tmp_path, monkeypatch, mode):
    ds, seen = make_action_data(tmp_path, monkeypatch, mode=mode)
    sample = ds[38]
    assert sample["state"].shape == (16,) and sample["state_valid"]
    assert sample["targets"]["action"].shape == (5, 16)
    assert sample["targets"]["action_valid_mask"].any(-1).tolist() == [True, True, False, False, False]
    assert sample["history_rgb"].shape == (4, 3, 384, 320)
    assert sample["history_valid"][:, :, :256].all()
    codec = ds.action_codec
    raw = {key: torch.tensor(ds.base.hf_dataset[key][38]) for key in codec.state_keys}
    decoded = codec.decode(sample["targets"]["action"], raw)
    torch.testing.assert_close(decoded[0], torch.full((16,), 58.))
    torch.testing.assert_close(decoded[1:], torch.full((4, 16), 59.))
    assert "Subtask" not in sample["condition_text"] and "home" not in sample["condition_text"]
    assert "Control Mode: endpose_effector" in sample["condition_text"]
    early = ds[3]
    assert early["history_valid"][:, 0, 0, 0].tolist() == [False, False, False, True]
    # History timestamps come from the current episode only, never its future.
    window = next(times for path, times in reversed(seen) if path.endswith("rgb") and len(times) > 5)
    assert max(window[5:9]) <= 2 + 3 / 30
    collated = wsa1_5_collate([early, sample])
    assert collated["targets"]["action"].shape == (2, 5, 16)


@pytest.mark.parametrize("subtask,endframe,annotation,expected", [
    (True, True, True, 9), (True, True, False, 39), (False, True, True, 39),
    (False, False, True, None), (True, False, False, None),
])
def test_goal_and_prompt_modes(tmp_path, monkeypatch, subtask, endframe, annotation, expected):
    ds, seen = make_action_data(tmp_path, monkeypatch, use_subtask=subtask,
                                allow_endframe=endframe, annotations=annotation)
    sample = ds[3]
    flags = sample["targets"]["goal_view_valid"]
    assert flags.any().item() == (expected is not None)
    if expected is not None:
        assert sample["goal_timestamp"] == pytest.approx(expected / 30)
    else:
        assert all(2 + 39 / 30 not in times for _, times in seen)
    if subtask and annotation:
        assert "Subtask: loop" in sample["condition_text"] and "both hands" in sample["condition_text"]
    else:
        assert "Subtask" not in sample["condition_text"] and "home" not in sample["condition_text"]
    ds.include_3d_targets = False
    assert "goal_rgb_views" not in ds[3]["targets"]


def test_prompt_hash_and_mode_validation():
    a = action_prompt("move", "joint", "delta")
    assert a.startswith("You are a robot control agent,follow the instruction.\nTask: move\n")
    b = action_prompt("move", "joint", "abs")
    assert text_embedding_path("cache", a) != text_embedding_path("cache", b)
    with pytest.raises(ValueError, match="control/action"):
        action_prompt("move", "eef", "delta")


def test_required_stats_and_missing_actions(tmp_path, monkeypatch):
    ds, _ = make_action_data(tmp_path, monkeypatch)
    missing = SimpleNamespace(**{**vars(ds.base.meta), "stats": {}})
    with pytest.raises(KeyError):
        ActionCodec(missing, ds.action_config)
    for key in ds.action_codec.action_keys:
        del ds.base.meta.features[key]
    rebuilt = WSA1_5Dataset(ds.base, ds.config, action_config=ds.action_config)
    assert not rebuilt[0]["targets"]["action_valid_mask"].any()
    assert rebuilt[0]["targets"]["rgb_valid"].any()


def test_action_stats_fit_existing_checkpoint_serializer(tmp_path, monkeypatch):
    from lerobot.datasets.utils import serialize_dict
    ds, _ = make_action_data(tmp_path, monkeypatch)
    stats = ds.action_codec.stats
    saved = serialize_dict({"aloha_robotwin": stats})
    ds.base.meta.stats = saved["aloha_robotwin"]
    codec = ActionCodec(ds.base.meta, ds.action_config)
    assert codec.config.action_mode == "delta"


def test_missing_action_rows_are_masked_and_empty_declared_source_errors(tmp_path, monkeypatch):
    ds, _ = make_action_data(tmp_path, monkeypatch)
    for key in ds.action_codec.action_keys:
        ds.base.hf_dataset[key][2] = None
    rebuilt = WSA1_5Dataset(ds.base, ds.config, action_config=ds.action_config)
    sample = rebuilt[0]
    assert sample["targets"]["action_valid_mask"].any(-1).tolist() == [True, True, False, True, True]
    for key in ds.action_codec.action_keys:
        ds.base.hf_dataset[key] = [None] * 40
    with pytest.raises(ValueError, match="no valid action rows"):
        WSA1_5Dataset(ds.base, ds.config, action_config=ds.action_config)


def test_action_flow_gradient_mask_and_seed():
    policy = action_policy()
    batch = action_batch(policy.config)
    loss, metrics = policy(batch)
    assert torch.isfinite(loss) and "loss_action_flow" in metrics
    loss.backward()
    assert policy.action_expert.state_encoder.weight.grad.abs().sum() > 0
    assert all(p.grad is not None for p in policy.action_expert.parameters() if p.requires_grad)
    assert all(p.grad is None for p in policy.flow.rgb.parameters())
    assert policy.future_3d.blocks[0].self_attn.k.weight.grad.abs().sum() > 0
    assert policy.teacher is None
    out = policy.predict_action_chunk(batch, seed=4, return_normalized=True)
    again = policy.predict_action_chunk(batch, seed=4, return_normalized=True)
    torch.testing.assert_close(out, again, rtol=0, atol=0)
    assert out.shape == (1, 4, 2)
    batch["targets"]["action_valid_mask"].zero_()
    zero, _ = policy(batch)
    assert zero.item() == 0
    zero.backward()


def test_action_scheduler_direction_and_element_reduction():
    expert = action_policy().action_expert
    target, noise = torch.ones(1, 4, 2), torch.full((1, 4, 2), 3.)
    noisy, velocity, time = expert.training_state(target, noise=noise, timestep=torch.tensor([500.]))
    torch.testing.assert_close(noisy, torch.full_like(noisy, 2))
    torch.testing.assert_close(expert.scheduler.step(velocity, torch.tensor(-1.), noise), target)
    mask = torch.zeros_like(target, dtype=torch.bool)
    mask[0, 0, 0] = True
    pred = velocity.clone()
    pred[0, 0, 0] += 2
    expected = 4 * expert.scheduler.training_weight(time)
    torch.testing.assert_close(expert.loss(pred, velocity, time, mask), expected)


def test_label_leakage_and_actual_goal_history_state_dependency():
    policy = action_policy()
    batch = action_batch(policy.config)
    # All future labels are deliberately absent from the observation-only API.
    output = policy.predict_action_chunk(batch, seed=9, return_normalized=True)
    changed = {**batch, "targets": {"future_rgb": torch.randn(1, 999), "future_motion": torch.randn(1, 77),
                                    "goal_rgb_views": torch.randn(1, 88), "teacher_features": torch.randn(1, 99),
                                    "action": torch.randn(1, 5)}}
    torch.testing.assert_close(policy.predict_action_chunk(changed, seed=9, return_normalized=True), output,
                               rtol=0, atol=0)
    for field in ["state", "observed_rgb", "history_rgb", "text_embedding"]:
        modified = {**batch, field: batch[field] + 0.5}
        assert not torch.allclose(policy.predict_action_chunk(modified, seed=9, return_normalized=True), output)
    with torch.no_grad():
        policy.future_3d.blocks[0].ffn[2].weight.add_(0.5)
    assert not torch.allclose(policy.predict_action_chunk(batch, seed=9, return_normalized=True), output)


@pytest.mark.parametrize("mode", ["frozen", "lora", "full"])
def test_rgb_modes_optimizer_and_lora_roundtrip(mode):
    policy = action_policy(rgb_finetune_mode=mode, lambda_rgb=1)
    trainable = [key for key, p in policy.flow.rgb.named_parameters() if p.requires_grad]
    if mode == "frozen":
        assert not trainable
    elif mode == "lora":
        assert trainable and all("lora_" in key for key in trainable)
        assert any(isinstance(m, LoRALinear) for m in policy.flow.rgb.modules())
        assert not any(isinstance(m, LoRALinear) for m in policy.future_3d.modules())
    else:
        assert "patch_embedding.weight" in trainable
    params = [p for g in policy.get_optim_params() for p in g["params"]]
    assert len(params) == len({id(p) for p in params})
    assert {id(p) for p in params} == {id(p) for p in policy.parameters() if p.requires_grad}
    restored = action_policy(rgb_finetune_mode=mode, lambda_rgb=1)
    restored.load_state_dict(policy.state_dict(), strict=True)
    batch = action_batch(policy.config)
    cfg = policy.config
    batch["targets"].update(future_rgb=torch.rand(1, 4, 3, cfg.rgb_height, cfg.rgb_width),
                             rgb_valid=torch.ones(1, 4, 1, cfg.rgb_height, cfg.rgb_width, dtype=torch.bool))
    loss, _ = policy(batch)
    loss.backward()
    assert all(p.grad is not None for p in params)
    if mode == "lora":
        assert any(p.grad.abs().sum() > 0 for key, p in policy.flow.rgb.named_parameters() if "lora_b" in key)


def test_stage1_initialization_and_old_keys(tmp_path):
    stage1 = action_policy(enable_action=False, lambda_rgb=1)
    assert not any("action_expert" in key or "lora_" in key for key in stage1.state_dict())
    (tmp_path / "config.json").write_text(json.dumps({"enable_action": False}))
    save_file(stage1.state_dict(), tmp_path / "model.safetensors")
    stage2 = action_policy(rgb_finetune_mode="lora", lambda_rgb=1)
    report = load_stage1_policy(stage2, tmp_path)
    assert report.missing_keys and not report.unexpected_keys
    for key, value in stage1.state_dict().items():
        torch.testing.assert_close(stage2.state_dict()[key], value)
    broken = dict(stage1.state_dict())
    key = "flow.rgb.patch_embedding.weight"
    broken[key] = broken[key][:1]
    save_file(broken, tmp_path / "model.safetensors")
    with pytest.raises(ValueError, match="shape mismatch"):
        load_stage1_policy(stage2, tmp_path)


@pytest.mark.parametrize("source_online,target_online", [(True, False), (True, True), (False, False)])
def test_stage1_text_encoder_transfer_respects_target_mode(tmp_path, source_online, target_online):
    stage1 = action_policy(enable_action=False, lambda_rgb=1, load_text_encoder=source_online)
    if source_online:
        stage1.text_encoder = nn.Linear(8, 8).eval().requires_grad_(False)
        assert not any(parameter.requires_grad for parameter in stage1.text_encoder.parameters())
    stage1.save_pretrained(tmp_path)
    stage2 = action_policy(rgb_finetune_mode="lora", lambda_rgb=1, load_text_encoder=target_online)
    if target_online:
        stage2.text_encoder = nn.Linear(8, 8).eval().requires_grad_(False)
    report = load_stage1_policy(stage2, tmp_path)
    assert not report.unexpected_keys
    assert all(key.startswith("action_expert.") or ".lora_" in key for key in report.missing_keys)
    for key, value in stage1.state_dict().items():
        if key.startswith("text_encoder.") and not target_online:
            continue
        torch.testing.assert_close(stage2.state_dict()[key], value)


@pytest.mark.parametrize("damage", ["extra_world_key", "missing_world_key", "world_shape", "undeclared_text_encoder"])
def test_dropping_stage1_umt5_does_not_relax_other_checkpoint_checks(tmp_path, damage):
    stage1 = action_policy(enable_action=False, lambda_rgb=1, load_text_encoder=True)
    stage1.text_encoder = nn.Linear(8, 8).eval().requires_grad_(False)
    stage1.save_pretrained(tmp_path)
    state = dict(stage1.state_dict())
    key = "flow.rgb.patch_embedding.weight"
    if damage == "extra_world_key":
        state["flow.rgb.stray.weight"] = torch.zeros(1)
    elif damage == "missing_world_key":
        del state[key]
    elif damage == "world_shape":
        state[key] = state[key][:1]
    else:
        config_path = tmp_path / "config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config["load_text_encoder"] = False
        config_path.write_text(json.dumps(config), encoding="utf-8")
    save_file(state, tmp_path / "model.safetensors")
    with pytest.raises(ValueError, match="mismatch"):
        load_stage1_policy(action_policy(rgb_finetune_mode="lora", lambda_rgb=1), tmp_path)


def test_offline_stage1_cannot_silently_initialize_an_online_text_encoder(tmp_path):
    stage1 = action_policy(enable_action=False, lambda_rgb=1)
    stage1.save_pretrained(tmp_path)
    stage2 = action_policy(load_text_encoder=True, lambda_rgb=1)
    stage2.text_encoder = nn.Linear(8, 8).eval().requires_grad_(False)
    with pytest.raises(ValueError, match="text_encoder"):
        load_stage1_policy(stage2, tmp_path)


def test_stage2_config_parse_and_old_stage1_defaults(tmp_path):
    with draccus.config_type("json"):
        parsed = draccus.parse(TrainPipelineConfig, "configs/wsa1_5_action_posttrain.json", args=[])
    assert parsed.policy.enable_action and parsed.dataset.use_subtask is False
    assert parsed.policy.sparse_frame_encoding == "joint"
    cfg = WSA1_5Config(device="cpu", push_to_hub=False)
    assert not cfg.enable_action and cfg.sparse_frame_encoding == "joint"
    assert "action" not in cfg.get_scheduler_preset().group_schedules
    with pytest.raises(ValueError, match="state_dim"):
        WSA1_5Config(device="cpu", push_to_hub=False, enable_action=True)


def test_action_depth_and_lora_bad_targets():
    with pytest.raises(ValueError, match="Parallel MoT"):
        action_policy(action_num_layers=2)
    with pytest.raises(ValueError, match="does not match"):
        action_policy(rgb_finetune_mode="lora", rgb_lora_target_modules=("nonexistent",))


def test_joint_world_output_reuses_rollout_and_masks_goal(monkeypatch):
    policy = action_policy()
    batch = action_batch(policy.config)
    batch["observed_valid"][:, :, 64:] = False
    sampler = policy.flow.sample_latents
    calls = []
    def record(*args, **kwargs):
        calls.append(kwargs)
        return sampler(*args, **kwargs)
    monkeypatch.setattr(policy.flow, "sample_latents", record)
    result = policy.predict_action_chunk(batch, seed=7, return_normalized=True, return_world=True)
    assert len(calls) == 1 and result["future_rgb"].shape == (1, 4, 3, 96, 64)
    assert not result["goal_3d_layer_features"][23][:, 1:].any()
    assert not result["future_rgb"][:, :, :, 64:].any()
    torch.testing.assert_close(result["actions"], policy.predict_action_chunk(batch, seed=7, return_normalized=True))


def test_single_joint_training_forward_and_teacher_only_labels(monkeypatch):
    policy = action_policy(lambda_rgb=1, lambda_motion=1, lambda_3d=1, enable_motion=True)
    cfg = policy.config
    batch = action_batch(cfg)
    batch["targets"].update(
        future_motion=torch.rand(1, 4, 3, cfg.motion_height, cfg.motion_width),
        motion_valid=torch.ones(1, 4, dtype=torch.bool),
        goal_rgb_views=[[torch.zeros(3, 8, 8) for _ in range(3)]],
        goal_view_valid=torch.ones(1, 3, dtype=torch.bool))
    class Teacher:
        def features(self, views, valid, device):
            value = views[0][0].mean() + 1
            return {layer: torch.full((1, 3, 273, 8), value, device=device) for layer in cfg.teacher_layers}, valid[..., None].expand(-1, -1, 273)
    policy.teacher = Teacher()
    captured = []
    def record(module, args, output):
        assert set(args[0]) == {"O", "R", "M", "G", "A"}
        captured.append(output[0]["A"].detach().clone())
    hook = policy.flow.mot.register_forward_hook(record)
    torch.manual_seed(53)
    loss, metrics = policy(batch)
    assert len(captured) == 1
    changed = {**batch, "targets": {**batch["targets"],
                                    "goal_rgb_views": [[torch.ones(3, 8, 8) for _ in range(3)]]}}
    torch.manual_seed(53)
    _, other_metrics = policy(changed)
    torch.testing.assert_close(captured[0], captured[1], rtol=0, atol=0)
    assert metrics["loss_action_flow"] == other_metrics["loss_action_flow"]
    changed["targets"]["future_rgb"] = batch["targets"]["future_rgb"] + .4
    torch.manual_seed(53)
    policy(changed)
    assert not torch.allclose(captured[0], captured[2])
    hook.remove()
    loss.backward()
    assert all(p.grad is not None for p in policy.parameters() if p.requires_grad)
    assert not {"action_grad_norm", "action_condition_s", "action_s"}.intersection(metrics)


def test_parallel_five_group_mot_has_no_reverse_dependency():
    from test_wsa1_5_attention import SmallBlock, fixture

    from lerobot.policies.wsa1_5.mot import CausalMoT, ExpertState
    states, experts = fixture()
    action_state = ExpertState(torch.randn(2, 4, 12), torch.zeros(2, 6, 12), torch.ones(2, 4, dtype=torch.bool))
    action_state.valid[0, -1] = False
    action = SimpleNamespace(blocks=nn.ModuleList([SmallBlock(12) for _ in range(3)]))
    mot = CausalMoT()
    original, _ = mot({**states, "A": action_state}, {**experts, "A": action})
    altered = replace(action_state, tokens=action_state.tokens + 100)
    changed, _ = mot({**states, "A": altered}, {**experts, "A": action})
    for group in states:
        torch.testing.assert_close(original[group], changed[group], rtol=0, atol=0)
    assert not original["A"][0, -1].any()


def test_online_queue_keeps_history_at_every_control_tick(monkeypatch):
    policy = action_policy()
    batch = action_batch(policy.config)
    plans = []
    def plan(prepared, **kwargs):
        plans.append(prepared)
        return torch.arange(8).reshape(1, 4, 2)
    monkeypatch.setattr(policy, "predict_action_chunk", plan)
    for tick in range(5):
        current = {k: v for k, v in batch.items() if k not in {"history_rgb", "history_valid", "targets"}}
        current["observed_rgb"] = torch.full_like(batch["observed_rgb"], float(tick))
        result = policy.select_action(current)
        torch.testing.assert_close(result, torch.arange(8).reshape(1, 4, 2)[:, tick % 4])
    assert len(plans) == 2
    assert plans[1]["history_rgb"][0, :, 0, 0, 0].tolist() == [0, 0, 0, 2]
    policy.reset()
    assert not policy._action_queue and policy._history is None


def test_stage2_prompt_precompute_matches_dataset(tmp_path, monkeypatch):
    from util_scripts.precompute_wsa1_5_text_embeddings import dataset_prompts
    for subtask in [False, True]:
        ds, _ = make_action_data(tmp_path, monkeypatch, use_subtask=subtask)
        prompts = dataset_prompts(tmp_path, enable_action=True, action_mode="delta", use_subtask=subtask)
        assert {s.text for s in ds.segments} <= set(prompts)


def test_history_and_action_windows_do_not_cross_episode(tmp_path, monkeypatch):
    ds, _ = make_action_data(tmp_path, monkeypatch)
    base = ds.base
    for key in base.hf_dataset:
        base.hf_dataset[key] = base.hf_dataset[key] * 2
    base.hf_dataset["episode_index"] = [0] * 40 + [7] * 40
    base.meta.episodes[7] = {**base.meta.episodes[0], "dataset_from_index": 40, "dataset_to_index": 80}
    rebuilt = WSA1_5Dataset(base, ds.config, action_config=ds.action_config)
    last, first = rebuilt[39], rebuilt[40]
    assert last["targets"]["action_valid_mask"].any(-1).tolist() == [True, False, False, False, False]
    assert not first["history_valid"].any()
    assert first["targets"]["action_valid_mask"].all()


def test_real_stage2_save_and_reload_uses_no_stage1_assets(tmp_path, monkeypatch):
    policy = action_policy(rgb_finetune_mode="lora", lambda_rgb=1, num_history_frame=0)
    policy.config.init_checkpoint_path = "unavailable/stage1"
    with torch.no_grad():
        for name, p in policy.named_parameters():
            if "lora_b" in name:
                p.fill_(0.01)
    policy.save_pretrained(tmp_path)
    def components(self, initialize):
        assert not initialize
        rgb = RGBVideoExpert(**wan_config(16, 32, layers=3, heads=2, head_dim=8, channels=4, text_dim=8, freq_dim=8))
        return WorldComponents(WorldFlow(rgb, future_frames=4), Future3DExpert(10, 20, 3, 2, 8, 8, output_dim=8), TinyVAE())
    monkeypatch.setattr(WSA1_5_Policy, "_build_components", components)
    restored = WSA1_5_Policy.from_pretrained(tmp_path)
    assert restored.teacher is None
    for key, value in policy.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[key], value, rtol=0, atol=0)


def test_factory_rejects_completely_actionless_sources(tmp_path):
    from lerobot.datasets.factory import make_dataset
    root = tmp_path / "fixture/meta"
    root.mkdir(parents=True)
    (root / "info.json").write_text(json.dumps({"total_frames": 40, "total_episodes": 1,
                                                "robot_type": "aloha_robotwin", "features": {}}))
    cfg = TrainPipelineConfig(policy=action_config(), dataset=WSA1_5DatasetConfig(
        root=str(tmp_path), repo_id="fixture", rgb_height=96, rgb_width=64,
        motion_height=32, motion_width=32, future_frames=4))
    with pytest.raises(ValueError, match="none of the selected"):
        make_dataset(cfg)


def test_action_launcher_dry_run_and_resume(tmp_path):
    import os
    import shlex
    import subprocess
    from pathlib import Path
    bash = Path(os.environ.get("PROGRAMFILES", "C:/Program Files")) / "Git/bin/bash.exe"
    if not bash.is_file():
        import shutil
        found = shutil.which("bash")
        if not found:
            pytest.skip("Bash unavailable")
        bash = Path(found)
    env = {**os.environ, "WSA_DRY_RUN": "1", "DATA_ROOT": "collection with spaces", "DATASET_REPO_ID": "task_a task_b",
           "INIT_CHECKPOINT_PATH": "stage1/pretrained_model", "PET_NNODES": "2", "PET_NPROC_PER_NODE": "2",
           "PET_NODE_RANK": "1", "PET_MASTER_ADDR": "10.0.0.1", "PET_MASTER_PORT": "29501"}
    env.pop("RESUME_CONFIG", None)
    env.pop("SPARSE_FRAME_ENCODING", None)
    for name in ["IMAGE_AUGMENT", "OBJECT_PASTE", "IMAGE_AUGMENTATION_CONFIG"]:
        env.pop(name, None)
    command = [str(bash), "launch/wsa_1_5_action_posttrain.sh"]
    result = subprocess.run(command, env=env, text=True, capture_output=True, check=True)
    args = shlex.split(result.stdout)
    assert "--dataset.repo_id=task_a task_b" in args and "--dataset.root=collection with spaces" in args
    assert "--num_processes=4" in args and "--machine_rank=1" in args
    assert "--dataset.use_external_stats=true" in args and "--dataset.action_mode=delta" in args
    assert "--policy.sparse_frame_encoding=joint" in args
    assert "--dataset.image_augment=false" in args and "--dataset.object_paste=false" in args
    assert not any("action_goal_steps" in arg or "action_stats_path" in arg for arg in args)
    env["SPARSE_FRAME_ENCODING"] = "independent"
    env.update(IMAGE_AUGMENT="true", OBJECT_PASTE="true", IMAGE_AUGMENTATION_CONFIG="custom recipe.yaml")
    result = subprocess.run(command, env=env, text=True, capture_output=True, check=True)
    args = shlex.split(result.stdout)
    assert "--policy.sparse_frame_encoding=independent" in args
    assert "--dataset.image_augment=true" in args and "--dataset.object_paste=true" in args
    assert "--dataset.image_augmentation_config=custom recipe.yaml" in args
    env["RESUME_CONFIG"] = "saved config/train_config.json"
    result = subprocess.run(command, env=env, text=True, capture_output=True, check=True)
    args = shlex.split(result.stdout)
    assert "--config_path=saved config/train_config.json" in args and "--resume=true" in args
    assert not any(arg.startswith("--policy.") or arg.startswith("--dataset.") for arg in args)


@pytest.mark.parametrize("mode,count,latent_frames,vae_calls", [
    ("joint", 4, 2, 1), ("independent", 4, 5, 5), ("independent", 2, 3, 3),
])
def test_prismwam_encoding_modes_and_temporal_rope(monkeypatch, mode, count, latent_frames, vae_calls):
    policy = action_policy(sparse_frame_encoding=mode, num_history_frame=count)
    batch = action_batch(policy.config)
    batch["history_valid"][:, 0] = False
    calls = []
    encode = policy.vae.encode
    def record(video):
        calls.append(video.shape)
        return encode(video)
    monkeypatch.setattr(policy.vae, "encode", record)
    history = policy.encode_history(batch)
    assert len(calls) == vae_calls
    assert all(shape[2] == (count + 1 if mode == "joint" else 1) for shape in calls)
    assert history["history_latent"].shape[2] == latent_frames
    assert history["history_valid"].shape[2] == latent_frames
    assert not history["history_valid"][:, :, 0].any()
    assert history["history_valid"][:, :, -1].all()
    observed, context, valid, spatial, _ = policy.encode_observation(
        batch["observed_rgb"], batch["condition_text"], batch["observed_valid"],
        text_embedding=batch["text_embedding"], text_valid=batch["text_valid"])
    rgb_latent = observed.expand(-1, -1, 2, -1, -1).clone()
    states, _, _, current_tokens = policy.flow.prepare_states(
        observed, context, valid, rgb_latent=rgb_latent, rgb_t=torch.tensor([500.]),
        observed_valid=spatial, **history)
    sequence = history["history_latent"]
    expected_tokens = policy.flow.rgb.patchify(sequence).flatten(2).transpose(1, 2)
    torch.testing.assert_close(states["O"].tokens, expected_tokens)
    assert states["O"].tokens.shape[1] == latent_frames * current_tokens
    assert not states["O"].valid[:, :current_tokens].any()
    temporal_width = policy.flow.rgb.freqs[0].shape[-1]
    for index in range(latent_frames):
        torch.testing.assert_close(states["O"].freqs[index * current_tokens, 0, :temporal_width],
                                   policy.flow.rgb.freqs[0][index])
    torch.testing.assert_close(states["R"].freqs[0, 0, :temporal_width], policy.flow.rgb.freqs[0][latent_frames])


def test_sparse_frame_encoding_validation_and_serialization(tmp_path):
    with pytest.raises(ValueError, match="sparse_frame_encoding"):
        action_config(sparse_frame_encoding="unknown")
    with pytest.raises(ValueError, match="multiple of 4"):
        action_config(num_history_frame=2)
    cfg = action_config(sparse_frame_encoding="independent", num_history_frame=2)
    cfg._save_pretrained(tmp_path)
    assert PreTrainedConfig.from_pretrained(tmp_path) == cfg


@pytest.mark.parametrize("mode", ["joint", "independent"])
def test_zero_history_skips_vae_for_both_modes(monkeypatch, mode):
    policy = action_policy(sparse_frame_encoding=mode, num_history_frame=0)
    def forbidden(*args):
        pytest.fail("Zero history must use the existing current-frame path")
    monkeypatch.setattr(policy.vae, "encode", forbidden)
    assert policy.encode_history({}) == {}


def test_independent_encoding_uses_shared_training_and_inference_path():
    policy = action_policy(sparse_frame_encoding="independent", num_history_frame=2)
    batch = action_batch(policy.config)
    loss, _ = policy(batch)
    loss.backward()
    assert torch.isfinite(loss) and policy.action_expert.head.weight.grad is not None
    first = policy.predict_action_chunk(batch, seed=8, return_normalized=True)
    second = policy.predict_action_chunk(batch, seed=8, return_normalized=True)
    assert first.shape == (1, policy.config.action_chunk_size, policy.config.action_dim)
    torch.testing.assert_close(first, second, rtol=0, atol=0)


@pytest.mark.parametrize("explicit_path", [False, True])
def test_original_plain_stats_lookup_by_embodiment_or_explicit_path(tmp_path, monkeypatch, explicit_path):
    from lerobot.datasets.factory import _apply_dataset_stats_overrides
    ds, _ = make_action_data(tmp_path / "dataset", monkeypatch)
    monkeypatch.setattr("lerobot.datasets.factory.HF_LEROBOT_HOME", tmp_path)
    path = (tmp_path / "single_dataset_stats.json" if explicit_path
            else tmp_path / "stats/aloha_robotwin/delta/stats.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    stats = {key: {"mean": [3.] * len(entry["mean"]), "std": [4.] * len(entry["std"])}
             for key, entry in ds.base.meta.stats.items()}
    path.write_text(json.dumps(stats), encoding="utf-8")
    data_cfg = replace(ds.config, external_stats_path=str(path) if explicit_path else None)
    cfg = TrainPipelineConfig(policy=ds.action_config, dataset=data_cfg)
    _apply_dataset_stats_overrides(cfg, ds.base)
    rebuilt = WSA1_5Dataset(ds.base, data_cfg, action_config=ds.action_config)
    torch.testing.assert_close(rebuilt[0]["state"], torch.full((16,), 7 / (4 + 1e-6)))
    assert set(rebuilt.action_codec.stats) == set(stats)


def test_action_loss_backpropagates_through_parallel_goal_and_rgb():
    policy = action_policy(rgb_finetune_mode="full", lambda_rgb=0, lambda_3d=0)
    loss, _ = policy(action_batch(policy.config))
    loss.backward()
    assert policy.flow.rgb.patch_embedding.weight.grad.abs().sum() > 0
    assert policy.future_3d.blocks[0].self_attn.k.weight.grad.abs().sum() > 0
    assert all(p.grad is not None for p in policy.parameters() if p.requires_grad)
