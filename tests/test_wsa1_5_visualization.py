"""Real PNG/text output from synthetic samples; no model construction or training."""
import json
import random

import numpy as np
import pytest
import torch
from PIL import Image

from lerobot.datasets.transformed_dataset import MultiLeRobotDataset
from lerobot.policies.wsa1_5.dataset_wsa1_5 import wsa1_5_collate
from lerobot.policies.wsa1_5.utils.visualization import save_training_sample
from test_wsa1_5_dataset import make_data
from test_wsa1_5_launch import bash, launch


def small_dataset(path, monkeypatch, **kwargs):
    dataset, seen = make_data(path, monkeypatch, future_frames=4, **kwargs)
    dataset.config.rgb_height, dataset.config.rgb_width = 96, 32
    dataset.config.motion_height = dataset.config.motion_width = 32
    return dataset, seen


@pytest.mark.parametrize("annotations", [False, True])
@pytest.mark.parametrize("motion_present", [False, True])
@pytest.mark.parametrize("include_3d_targets", [False, True])
def test_snapshot_saves_exact_prompt_images_and_masks_without_changing_rng(
        tmp_path, monkeypatch, annotations, motion_present, include_3d_targets):
    dataset, seen = small_dataset(tmp_path / "data", monkeypatch, annotations=annotations,
                                  include_3d_targets=include_3d_targets)
    if not motion_present:
        del dataset.meta.features[dataset.config.motion_key]
    batch = wsa1_5_collate([dataset[0], dataset[8]])
    observed = batch["observed_rgb"].clone()
    future = batch["targets"]["future_rgb"].clone()
    torch_rng, python_rng = torch.random.get_rng_state().clone(), random.getstate()
    seen.clear()
    directory = save_training_sample(batch, dataset, tmp_path / "validation", 20)
    assert directory.name == "step_00000020"
    assert {path.name for path in directory.iterdir()} == {
        "prompt.txt", "metadata.json", "rgb_current_future.png", "motion_current.png", "goal.png"}
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    index = metadata["batch_index"]
    assert 0 <= index < 2 and metadata["sample_id"] == batch["sample_id"][index]
    assert (directory / "prompt.txt").read_text(encoding="utf-8") == batch["condition_text"][index]
    assert metadata["future_frame_valid"] == batch["future_frame_valid"][index].tolist()
    assert metadata["observed_view_valid"] == [True, False, False]
    assert metadata["goal_view_valid"] == [annotations, False, False]
    assert metadata["current_motion_valid"] == motion_present
    rgb = np.asarray(Image.open(directory / "rgb_current_future.png"))
    assert rgb.shape == (2 * (96 + 2) + 2, 4 * (32 + 2) + 2, 3)
    expected_pixel = ((observed[index, :, 0, 0] + 1) * 127.5).round().to(torch.uint8).numpy()
    np.testing.assert_array_equal(rgb[2, 2], expected_pixel)
    assert not rgb[2 + 64:2 + 96, 2:2 + 32].any()  # Missing RGB views are black.
    motion = np.asarray(Image.open(directory / "motion_current.png"))
    assert motion.shape == (32, 32, 3)
    if motion_present:
        expected_motion = round((2 + metadata["start_timestamp"]) / 4 * 255)
        assert (motion == expected_motion).all()
        assert seen[0][1] == pytest.approx([2 + metadata["start_timestamp"]])
    else:
        assert not motion.any()
    goal = np.asarray(Image.open(directory / "goal.png"))
    assert goal.shape == (96, 32, 3)
    assert bool(goal.any()) == annotations
    assert not goal[64:].any()
    torch.testing.assert_close(torch.random.get_rng_state(), torch_rng, rtol=0, atol=0)
    assert random.getstate() == python_rng
    torch.testing.assert_close(batch["observed_rgb"], observed, rtol=0, atol=0)
    torch.testing.assert_close(batch["targets"]["future_rgb"], future, rtol=0, atol=0)


def test_snapshot_resolves_selected_repo_in_multi_dataset(tmp_path, monkeypatch):
    first, _ = small_dataset(tmp_path / "first", monkeypatch)
    second, seen = small_dataset(tmp_path / "second", monkeypatch)
    first.repo_id, second.repo_id = "first", "second"
    combined = MultiLeRobotDataset([first, second])
    batch = wsa1_5_collate([combined[len(first) + 8]])
    seen.clear()
    directory = save_training_sample(batch, combined, tmp_path / "validation", 1)
    metadata = json.loads((directory / "metadata.json").read_text())
    assert metadata["sample_id"] == "second:0:8"
    assert seen and all(path.startswith(str(second.root)) for path, _ in seen)


def test_snapshot_rejects_wrong_source_and_preserves_decode_errors(tmp_path, monkeypatch):
    dataset, _ = small_dataset(tmp_path / "data", monkeypatch)
    batch = wsa1_5_collate([dataset[0]])
    batch["sample_id"][0] = "unknown:0:0"
    with pytest.raises(LookupError, match="Cannot find snapshot source"):
        save_training_sample(batch, dataset, tmp_path / "validation", 1)
    batch["sample_id"][0] = "test:0:0"
    monkeypatch.setattr("lerobot.policies.wsa1_5.dataset_wsa1_5.decode_video_frames",
                        lambda *_: (_ for _ in ()).throw(RuntimeError("broken video")))
    with pytest.raises(RuntimeError, match="WSA video decode failed"):
        save_training_sample(batch, dataset, tmp_path / "validation", 1)


@pytest.mark.parametrize("frequency", [0, 500])
def test_launcher_exposes_snapshot_frequency(launch, frequency):
    args = launch(EVAL_FREQ=str(frequency), PET_NNODES="1", PET_NODE_RANK="0", PET_NPROC_PER_NODE="1",
                  PET_MASTER_ADDR="127.0.0.1", PET_MASTER_PORT="29500")
    assert args["eval_freq"] == str(frequency)


@pytest.mark.parametrize("overrides,expected", [({}, ("1", "2")),
    ({"VIDEO_DECODE_THREADS": "2", "PREFETCH_FACTOR": "1"}, ("2", "1"))])
def test_launcher_exposes_decoder_and_prefetch_limits(launch, overrides, expected):
    args = launch(PET_NNODES="1", PET_NODE_RANK="0", PET_NPROC_PER_NODE="1",
                  PET_MASTER_ADDR="127.0.0.1", PET_MASTER_PORT="29500", **overrides)
    assert (args["dataset.video_decode_threads"], args["dataset.prefetch_factor"]) == expected
