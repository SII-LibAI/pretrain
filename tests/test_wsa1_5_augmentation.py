"""LingBot RGB recipe replay at WSA's multi-view/video/teacher boundary."""

from dataclasses import replace
from pathlib import Path

import draccus
import pytest
import torch
import yaml
from test_wsa1_5_action import make_action_data

from lerobot.configs.default import DatasetConfig
from lerobot.policies.wsa1_5.configuration_wsa1_5 import WSA1_5DatasetConfig
from lerobot.policies.wsa1_5.dataset_wsa1_5 import WSA1_5Dataset, compose_rgb_views, wsa1_5_collate
from lerobot.policies.wsa1_5.utils import visual_augmentation as aug


def recipe(**overrides):
    options = {"keep_original_prob": 0, "color_prob": 0, "affine_prob": 0, "gaussian_noise_prob": 0,
                   "salt_pepper_prob": 0, "gaussian_blur_prob": 0, "motion_blur_prob": 0, "jpeg_prob": 0,
                   "occlusion_prob": 0, "paste_prob": 0}
    options.update(overrides)
    return options


def write_recipe(tmp_path, **overrides):
    path = tmp_path / "augmentation.yaml"
    path.write_text(yaml.safe_dump(recipe(**overrides)), encoding="utf-8")
    return str(path)


@pytest.mark.parametrize("image_augment,object_paste", [(False, False), (True, False), (False, True), (True, True)])
def test_independent_dataset_switches_and_config_roundtrip(tmp_path, monkeypatch, image_augment, object_paste):
    monkeypatch.chdir(tmp_path)
    cfg = WSA1_5DatasetConfig(repo_id="fixture", image_augment=image_augment, object_paste=object_paste)
    restored = draccus.decode(DatasetConfig, draccus.encode(cfg))
    assert isinstance(restored, WSA1_5DatasetConfig)
    assert (restored.image_augment, restored.object_paste, restored.image_augmentation_config) == (
        image_augment, object_paste, cfg.image_augmentation_config
    )
    loaded = aug.load_visual_augmentation(cfg.image_augmentation_config,
                                          image_augment=image_augment, object_paste=object_paste)
    if not image_augment and not object_paste:
        assert loaded is None
        return
    assert loaded.paste_prob == (0.5 if object_paste else 0)
    assert loaded.color_prob == (0.8 if image_augment else 0)
    assert loaded.occlusion_prob == (0.95 if image_augment else 0)
    assert loaded.keep_original_prob == 0.1


def test_disabled_paste_does_not_require_assets_and_bad_config_fails_early(tmp_path):
    path = write_recipe(tmp_path, paste_prob=1, crops_dir=str(tmp_path / "missing_crops"))
    assert aug.load_visual_augmentation(path, object_paste=False).paste_prob == 0
    with pytest.raises(FileNotFoundError, match="crop directory"):
        aug.load_visual_augmentation(path)
    assert aug.load_visual_augmentation("missing.yaml", image_augment=False, object_paste=False) is None
    for options in ({"image_augment": "true"}, {"object_paste": 1}):
        with pytest.raises(TypeError, match="boolean"):
            WSA1_5DatasetConfig(repo_id="fixture", **options)
    with pytest.raises(ValueError, match="YAML path"):
        WSA1_5DatasetConfig(repo_id="fixture", object_paste=True, image_augmentation_config="")
    for options in ({"paste_prob": 1.1}, {"occlusion_area": (0.1, 1.1)}, {"scale": (1.1, 0.9)},
                    {"gaussian_blur_prob": 0.9, "motion_blur_prob": 0.5}, {"jpeg_quality": (0, 90)}):
        with pytest.raises(ValueError):
            aug.VisualAugmentationConfig(**options)


@pytest.mark.parametrize("effect", ["color", "affine", "gaussian_noise", "salt_pepper", "gaussian_blur",
                                    "motion_blur", "jpeg", "occlusion", "paste"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.uint8])
def test_effects_replay_across_time_without_input_mutation(effect, dtype):
    torch.manual_seed(17)
    image = torch.randint(0, 256, (3, 48, 64), dtype=torch.uint8)
    if dtype == torch.float32:
        image = image.float() / 255
    original = image.clone()
    config = aug.VisualAugmentationConfig(**recipe(**{effect + "_prob": 1}))
    params = aug.sample_pixel_augmentation(config, {"camera_top": (48, 64)})
    result = aug.apply_pixel_augmentation(image, params, "camera_top")
    assert result.shape == image.shape and result.dtype == dtype
    assert torch.equal(image, original) and not torch.equal(result, original)
    assert result.min() >= 0 and result.max() <= (255 if dtype == torch.uint8 else 1)
    assert torch.equal(result, aug.apply_pixel_augmentation(image, params, "camera_top"))
    clip = aug.apply_pixel_augmentation(torch.stack([image] * 3), params, "camera_top")
    for frame in clip:
        torch.testing.assert_close(frame, result, rtol=0, atol=1e-7 if dtype == torch.float32 else 0)


def test_crop_assets_view_isolation_alpha_and_missing_views(monkeypatch):
    paths = aug._crop_bank("assets/RoboTwin_Clean_Object_Crops")
    assert len(paths) == 45 and len({Path(path).parent.name for path in paths}) == 15
    assert aug._load_crop(paths[0]) is aug._load_crop(paths[0])
    torch.manual_seed(17)
    config = aug.VisualAugmentationConfig(**recipe(paste_prob=1, occlusion_prob=1))
    shapes = dict.fromkeys(aug.CAMERA_WEIGHTS, (48, 64))
    params = aug.sample_pixel_augmentation(config, shapes)
    assert 1 <= len(params["pastes"]) <= 2
    assert len({paste["path"] for paste in params["pastes"]}) == len(params["pastes"])
    for paste in params["pastes"]:
        crop = aug._load_crop(paste["path"])
        x, y, w, h = aug._pixel_box(paste["box"], 48, 64)
        assert abs(w - h * crop.shape[2] / crop.shape[1]) <= 2
        assert 0 <= x < x + w <= 64 and 0 <= y < y + h <= 48
    image = torch.zeros(3, 48, 64)
    for name in shapes:
        result = aug.apply_pixel_augmentation(image, params, name)
        assert bool(result.any()) == (name in {params["paste_view"], params["occlusion_view"]})
    for views, expected in [({"camera_wrist_left": (30, 60)}, "camera_wrist_left"), ({}, None)]:
        sampled = aug.sample_pixel_augmentation(config, views)
        assert sampled["paste_view"] == sampled["occlusion_view"] == expected
    transparent_red = torch.zeros(4, 10, 20)
    transparent_red[0], transparent_red[3] = 1, 0.5
    monkeypatch.setattr(aug, "_load_crop", lambda path: transparent_red)
    params.update(occlusion_view=None, paste_view="camera_top",
                  pastes=[{"path": "test_alpha", "box": (0.1, 0.1, 0.3, 0.2)}])
    result = aug.apply_pixel_augmentation(image, params, "camera_top")
    x, y, w, h = aug._pixel_box(params["pastes"][0]["box"], 48, 64)
    torch.testing.assert_close(result[0, y:y+h, x:x+w], torch.full((h, w), 0.5))
    assert not result[1:].any() and not image.any()


def test_original_branch_and_camera_probabilities():
    torch.manual_seed(17)
    config = aug.VisualAugmentationConfig(**recipe(keep_original_prob=0.1, occlusion_prob=0.95, paste_prob=0.5))
    samples = [aug.sample_pixel_augmentation(config) for _ in range(2000)]
    kept = [params for params in samples if params["keep_original"]]
    assert abs(len(kept) / len(samples) - 0.1) < 0.03
    active = [params for params in samples if not params["keep_original"]]
    for field, probability in [("occlusion_view", 0.95), ("paste_view", 0.5)]:
        triggered = [params for params in active if params[field] is not None]
        assert abs(len(triggered) / len(active) - probability) < 0.04
        for name, weight in aug.CAMERA_WEIGHTS.items():
            assert abs(sum(params[field] == name for params in triggered) / len(triggered) - weight) < 0.05
    image = torch.rand(3, 48, 64)
    assert aug.apply_pixel_augmentation(image, kept[0]) is image


@pytest.mark.parametrize("index", [3, 8])  # Goal decoded separately / included in the future window.
@pytest.mark.parametrize("missing_wrist", [False, True])
@pytest.mark.parametrize("enable_action", [False, True])
def test_dataset_replays_one_draw_for_history_future_and_teacher(
    tmp_path, monkeypatch, index, missing_wrist, enable_action
):
    ds, _ = make_action_data(tmp_path, monkeypatch, use_subtask=True)
    action_config = ds.action_config if enable_action else None
    ds.base.meta.features.update({key: {"dtype": "video", "shape": [16, 16, 3]}
                                  for key in ["left_rgb", "right_rgb"]})
    ds.base.meta.episodes[0].update({f"videos/{key}/from_timestamp": 2.0
                                    for key in ["left_rgb", "right_rgb"]})
    cfg = replace(ds.config, view_keys=["rgb", "left_rgb", "" if missing_wrist else "right_rgb"],
                  rgb_height=48, rgb_width=32, motion_height=16, motion_width=16)
    decoded = []

    def decode(path, times, *args):
        raw = torch.full((len(times), 3, 16, 16), 0.4)
        decoded.append(raw)
        return raw

    monkeypatch.setattr("lerobot.policies.wsa1_5.dataset_wsa1_5.decode_video_frames", decode)
    baseline = WSA1_5Dataset(ds.base, cfg, action_config=action_config)[index]
    path = write_recipe(tmp_path, color_prob=1, affine_prob=1, scale=[0.9, 0.9], gaussian_noise_prob=1,
                        jpeg_prob=1, occlusion_prob=1, paste_prob=1, paste_area=[0.1, 0.2])
    cfg = replace(cfg, image_augment=True, object_paste=True, image_augmentation_config=path)
    dataset = WSA1_5Dataset(ds.base, cfg, action_config=action_config)
    draws = []
    sample = aug.sample_pixel_augmentation

    def record(config, shapes):
        params = sample(config, shapes)
        draws.append(params)
        return params

    monkeypatch.setattr("lerobot.policies.wsa1_5.dataset_wsa1_5.sample_pixel_augmentation", record)
    torch.manual_seed(17)
    result = dataset[index]
    assert len(draws) == 1
    assert not torch.equal(result["observed_rgb"], baseline["observed_rgb"])
    for frame in [*result.get("history_rgb", []), *result["targets"]["future_rgb"]]:
        torch.testing.assert_close(frame, result["observed_rgb"])
    targets = result["targets"]
    teacher_canvas, _ = compose_rgb_views([frame[None] for frame in targets["goal_rgb_views"]],
                                         targets["goal_view_valid"], cfg.rgb_height, cfg.rgb_width)
    torch.testing.assert_close(teacher_canvas[0] * 2 - 1, result["observed_rgb"])
    condition_keys = ["observed_valid", "future_frame_valid", "frame_timestamps"]
    if enable_action:
        condition_keys += ["state", "state_valid", "history_valid"]
    for key in condition_keys:
        torch.testing.assert_close(result[key], baseline[key])
    target_keys = ["future_motion", "motion_valid", "rgb_valid", "goal_view_valid"]
    if enable_action:
        target_keys += ["action", "action_valid_mask"]
    for key in target_keys:
        torch.testing.assert_close(targets[key], baseline["targets"][key])
    assert result["condition_text"] == baseline["condition_text"]
    if missing_wrist:
        assert not result["observed_valid"][:, 32:, 16:].any()
        assert (result["observed_rgb"][:, 32:, 16:] == -1).all()
        assert draws[0]["paste_view"] != "camera_wrist_right"
        assert draws[0]["occlusion_view"] != "camera_wrist_right"
    assert all((raw == 0.4).all() for raw in decoded)
    batch = wsa1_5_collate([result, result])
    if enable_action:
        assert batch["history_rgb"].shape == (2, 4, 3, 48, 32)
    assert len(batch["targets"]["goal_rgb_views"]) == 2


@pytest.mark.parametrize("training", [False, True])
def test_disabled_and_evaluation_paths_skip_yaml_assets_and_rng(tmp_path, monkeypatch, training):
    ds, _ = make_action_data(tmp_path, monkeypatch)
    expected = ds[3]
    cfg = replace(ds.config, image_augment=not training, object_paste=not training,
                  image_augmentation_config="missing_training_recipe.yaml")
    dataset = WSA1_5Dataset(ds.base, cfg, action_config=ds.action_config, training=training)
    rng = torch.get_rng_state().clone()
    result = dataset[3]
    assert torch.equal(rng, torch.get_rng_state())
    for key in ["observed_rgb", "history_rgb", "state"]:
        torch.testing.assert_close(result[key], expected[key], rtol=0, atol=0)
    for key in ["future_rgb", "future_motion", "action"]:
        torch.testing.assert_close(result["targets"][key], expected["targets"][key], rtol=0, atol=0)
