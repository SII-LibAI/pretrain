"""Small random tensors only: no pretrained assets, optimizer updates or training."""
import json

import pytest
import torch

from lerobot.datasets.configuration_wsa1_5 import WSA1_5DatasetConfig
from lerobot.datasets.wsa1_5 import WSA1_5Dataset
from lerobot.policies.wsa1_5.attention import build_causal_attention_mask, masked_attention
from lerobot.policies.wsa1_5.future_3d import geometric_feature_loss
from lerobot.policies.wsa1_5.initialization import interpolate_wan_state
from lerobot.policies.wsa1_5.modeling_wsa1_5 import WSA1_5_Policy
from lerobot.policies.wsa1_5.temporal import TemporalLayout
from lerobot.policies.wsa1_5.vendor.helpers.state_dict_converters import wan_video_dit_state_dict_converter
from lerobot.policies.wsa1_5.vendor.scheduler import WanContinuousFlowMatchScheduler
from test_wsa1_5_dataset import make_data
from test_wsa1_5_experts import small_flow
from test_wsa1_5_policy_contract import small_config, small_policy


@pytest.mark.parametrize("field", ["lambda_rgb", "lambda_motion", "lambda_3d", "rgb_lr", "flow_shift",
                                  "optimizer_weight_decay", "feature_cosine_weight"])
@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_nonfinite_config_rejected(field, value):
    with pytest.raises(ValueError, match="finite"):
        small_config(**{field: value})


@pytest.mark.parametrize("kwargs", [{"lambda_rgb": 0, "lambda_motion": 0, "lambda_3d": 0},
                                   {"enable_rgb": False, "enable_motion": False,
                                    "feature_mse_weight": 0, "feature_cosine_weight": 0},
                                   {"freq_dim": 7}, {"rgb_hidden_dim": 0}, {"goal_hidden_dim": 9},
                                   {"scheduler_warmup_steps": 600000}])
def test_invalid_config_contract(kwargs):
    # Avoid duplicate explicit arguments in the small_config convenience helper.
    config = small_config()
    for key, value in kwargs.items():
        setattr(config, key, value)
    with pytest.raises(ValueError):
        config.__post_init__()


def test_fp32_continuous_time_and_euler_under_bf16():
    scheduler = WanContinuousFlowMatchScheduler()
    torch.manual_seed(19)
    t = scheduler.sample_training_t(1000, torch.device("cpu"), torch.bfloat16)
    assert t.dtype == torch.float32 and (t != t.bfloat16().float()).any()
    times, deltas = scheduler.build_inference_schedule(20, torch.device("cpu"), torch.bfloat16)
    assert times.dtype == deltas.dtype == torch.float32
    assert (deltas < 0).all()
    torch.testing.assert_close(deltas.sum(), torch.tensor(-1.0))
    sample = torch.tensor([1.0], dtype=torch.bfloat16)
    delta = torch.tensor(-0.0003)
    result = scheduler.step(torch.ones_like(sample), delta, sample)
    assert result.dtype == torch.float32 and result.item() < 1
    mixed = scheduler.add_noise(sample, torch.zeros_like(sample), torch.tensor([0.3]))
    assert mixed.dtype == torch.float32 and mixed.item() < 1
    with pytest.raises(TypeError, match="FP32"):
        scheduler.training_weight(t.bfloat16())


@pytest.mark.parametrize("time", [float("nan"), -1.0, 1001.0])
def test_invalid_flow_time_rejected(time):
    with pytest.raises(ValueError, match="timestep"):
        WanContinuousFlowMatchScheduler().training_weight(torch.tensor([time]))


def test_bad_flow_targets_cannot_hide_behind_masks():
    flow = small_flow()
    clean = torch.zeros(1, 4, 9, 2, 2)
    mask = torch.zeros(1, 1, 9, 2, 2, dtype=torch.bool)
    bad = clean.clone()
    bad[:, :, -1] = float("nan")
    with pytest.raises(FloatingPointError):
        flow.flow_loss(bad, clean, torch.tensor([500.0]), mask)
    with pytest.raises(TypeError, match="boolean"):
        flow.flow_loss(clean, clean, torch.tensor([500.0]), mask.float())
    with pytest.raises(ValueError, match="noise"):
        flow.training_state(clean, clean[:, :, :1], noise=torch.zeros(1))
    with pytest.raises(ValueError, match="zero timestep weight"):
        flow.flow_loss(clean, clean, torch.tensor([1000.0]), ~mask)
    noisy, velocity, timestep = flow.training_state(clean.bfloat16(), clean[:, :, :1].bfloat16())
    assert noisy.dtype == velocity.dtype == timestep.dtype == torch.float32


def test_feature_targets_and_masks_cannot_hide_errors():
    student = {4: torch.randn(1, 3, 273, 8, requires_grad=True)}
    teacher = {4: torch.zeros_like(student[4])}
    mask = torch.ones(1, 3, 273, dtype=torch.bool)
    with pytest.raises(ValueError, match="norm"):
        geometric_feature_loss(student, teacher, mask)
    teacher[4][0, 0, 0, 0] = float("nan")
    with pytest.raises(FloatingPointError):
        geometric_feature_loss(student, teacher, ~mask)
    with pytest.raises(TypeError, match="boolean"):
        TemporalLayout().latent_validity(torch.ones(1, 32), (2, 2))
    with pytest.raises(FloatingPointError, match="norm"):
        geometric_feature_loss(student, {4: torch.full_like(student[4], 1e30)}, mask)


def test_all_invalid_batch_is_not_an_optimizer_update():
    policy = small_policy()
    assert policy.teacher.dtype == torch.float32
    empty = torch.zeros(1, 1, 9, 2, 2, dtype=torch.bool)
    goal = torch.zeros(1, 3, 273, dtype=torch.bool)
    with pytest.raises(ValueError, match="no valid positive-weight"):
        policy._require_supervision({"rgb": empty, "motion": empty}, goal)
    # The condition at t0 never counts as supervision.
    empty[:, :, 0] = True
    with pytest.raises(ValueError, match="no valid positive-weight"):
        policy._require_supervision({"rgb": empty}, goal)
    goal[:, 0] = True
    policy._require_supervision({"rgb": empty}, goal)
    with pytest.raises(ValueError, match="observed_valid"):
        policy.training_forward({"targets": {}})
    image = torch.zeros(1, 3, policy.config.rgb_height, policy.config.rgb_width)
    with pytest.raises(TypeError, match="boolean"):
        policy.encode_observation(image, ["test"], torch.ones_like(image[:, :1]))
    with pytest.raises(ValueError, match="valid observed RGB"):
        policy.encode_observation(image, ["test"], torch.zeros_like(image[:, :1], dtype=torch.bool))



@pytest.mark.parametrize("key,source_shape,target_shape", [
    ("patch_embedding.weight", (16, 4, 1, 2, 2), (12, 8, 1, 2, 2)),
    ("blocks.0.self_attn.q.weight", (16, 16), (8, 12)),
    ("text_embedding.0.weight", (16, 8), (12, 10)),
])
def test_initialization_cannot_resize_fixed_geometry(key, source_shape, target_shape):
    with pytest.raises(ValueError, match="geometry"):
        interpolate_wan_state({key: torch.zeros(source_shape)}, {key: torch.zeros(target_shape)})


def test_checkpoint_conversion_and_resume_are_strict():
    for state in [{"vace.foo": torch.zeros(1)}, {"x": torch.zeros(1), "model.x": torch.ones(1)}]:
        with pytest.raises(ValueError):
            wan_video_dit_state_dict_converter(state)
    with pytest.raises(ValueError, match="strict=True"):
        WSA1_5_Policy.from_pretrained("unused-no-model-loaded", strict=False)


def test_view_mapping_and_pixel_range_do_not_default(tmp_path, monkeypatch):
    ds, _ = make_data(tmp_path, monkeypatch)
    ds.base.meta.robot_type = "unknown-robot"
    with pytest.raises(ValueError, match="Unknown robot_type"):
        WSA1_5Dataset(ds.base, WSA1_5DatasetConfig(repo_id="x"))
    with pytest.raises(ValueError, match="absent from schema"):
        WSA1_5Dataset(ds.base, WSA1_5DatasetConfig(repo_id="x", view_keys=["rgb_typo", "", ""]))
    monkeypatch.setattr("lerobot.datasets.wsa1_5.decode_video_frames",
                        lambda *args: torch.full((34, 3, 16, 16), 1.5))
    with pytest.raises(RuntimeError, match="video decode failed"):
        ds[0]


@pytest.mark.parametrize("field,value", [("start_frame", 1.5), ("end_frame", 1000), ("subtask_id", "bad")])
def test_bad_sidecar_indices_do_not_round_or_mask(tmp_path, monkeypatch, field, value):
    ds, _ = make_data(tmp_path, monkeypatch)
    path = tmp_path / "meta/sidecar.json"
    annotations = json.loads(path.read_text())
    annotations[0]["subtasks"][0][field] = value
    path.write_text(json.dumps(annotations))
    with pytest.raises(ValueError):
        WSA1_5Dataset(ds.base, ds.config)


def test_attention_requires_complete_padding_contract():
    with pytest.raises(ValueError, match="Missing explicit"):
        build_causal_attention_mask({"O": 1, "R": 1}, valid_tokens={"O": torch.ones(1, 1, dtype=torch.bool)})
    with pytest.raises(ValueError, match="at least one key"):
        masked_attention(torch.zeros(1, 1, 2, 4), torch.zeros(1, 1, 0, 4), torch.zeros(1, 1, 0, 4),
                         torch.zeros(1, 1, 2, 0, dtype=torch.bool))


@pytest.mark.parametrize("ratio", [0.0, 0.02, 0.1, 1.0])
def test_wsa_scheduler_does_not_rescale_short_runs(ratio):
    config = small_config(scheduler_warmup_steps=2000, scheduler_decay_steps=600000,
                          scheduler_decay_lr_ratio=ratio)
    preset = config.get_scheduler_preset()
    assert not preset.auto_scale_to_training_steps
    # Build the scheduler only; no optimizer step or model training is performed.
    optimizer = torch.optim.SGD([{"params": [torch.nn.Parameter(torch.ones(1))], "name": "rgb", "lr": config.rgb_lr}])
    scheduler = preset.build(optimizer, num_training_steps=100)
    assert scheduler.lr_lambdas[0](0) == pytest.approx(1 / 2000)
    assert scheduler.lr_lambdas[0](2000) == pytest.approx(1)
    assert scheduler.lr_lambdas[0](600000) == pytest.approx(ratio)
    assert scheduler.lr_lambdas[0](700000) == pytest.approx(ratio)
