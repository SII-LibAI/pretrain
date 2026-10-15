import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from lerobot.datasets.configuration_wsa1_5 import WSA1_5DatasetConfig
from lerobot.datasets.wsa1_5 import (
    WSA1_5Dataset, build_inference_condition, compose_rgb_views, wsa1_5_collate,
)


def make_data(tmp_path, monkeypatch, annotations=True):
    tmp_path.mkdir(exist_ok=True)
    (tmp_path / "meta").mkdir(exist_ok=True)
    if annotations:
        (tmp_path / "meta/sidecar.json").write_text(json.dumps([{
            "episode_index": 0, "task_instruction": "knit", "scene": "home", "subtasks": [
                {"subtask_id": 0, "subtask_text": "loop", "start_frame": 0, "end_frame": 9, "hand_used": 2},
                {"subtask_id": 1, "subtask_text": "pull", "start_frame": 10, "end_frame": 39, "hand_used": 0}
            ]
        }]))
    features = {key: {"dtype": "video", "shape": [16, 16, 3]} for key in ["rgb", "observation.motion"]}
    meta = SimpleNamespace(features=features, episodes=[{
        **{f"videos/{key}/from_timestamp": 2.0 for key in features},
        "dataset_from_index": 0, "dataset_to_index": 40,
    }], robot_type="egodex_v", total_episodes=1, stats={})
    meta.get_video_file_path = lambda ep, key: key
    meta.episodes = {0: meta.episodes[0], "dataset_from_index": [0], "dataset_to_index": [40]}
    base = SimpleNamespace(meta=meta, root=tmp_path, repo_id="test", tolerance_s=1e-4, video_backend="pyav",
                           hf_dataset={"timestamp": [torch.tensor(i / 30, dtype=torch.float64) for i in range(40)],
                                       "frame_index": list(range(40)), "episode_index": [0] * 40})
    base._ensure_hf_dataset_loaded = lambda: None
    seen = []
    def decode(path, times, *args):
        seen.append((str(path), times))
        return torch.tensor(times)[:, None, None, None].expand(-1, 3, 16, 16) / 4
    monkeypatch.setattr("lerobot.datasets.wsa1_5.decode_video_frames", decode)
    config = WSA1_5DatasetConfig(repo_id="test", view_keys=["rgb", "", ""], motion_log_every=0)
    return WSA1_5Dataset(base, config), seen


def test_boundaries_targets_and_timestamps(tmp_path, monkeypatch):
    ds, seen = make_data(tmp_path, monkeypatch)
    sample = ds[8]
    assert sample["subtask_id"] == 0
    assert sample["goal_timestamp"] == pytest.approx(9 / 30)
    assert sample["targets"]["future_rgb"].shape == (32, 3, 384, 320)
    assert sample["targets"]["future_motion"].shape == (32, 3, 192, 160)
    assert sample["future_frame_valid"].sum() == 1
    assert sample["targets"]["motion_valid"].sum() == 1
    assert (sample["frame_timestamps"] <= 9 / 30 + 1e-8).all()
    np.testing.assert_allclose(seen[0][1][1:33], seen[1][1])
    assert sample["targets"]["goal_view_valid"].tolist() == [True, False, False]
    assert "both hands" in sample["condition_text"]
    assert "future_motion" not in sample
    assert "goal_rgb_views" not in sample
    assert ds[10]["subtask_id"] == 1
    assert "left hand" in ds[10]["condition_text"]


def test_layout():
    views = [torch.full((1, 3, 4, 4), float(i)) for i in [1, 2, 3]]
    canvas, mask = compose_rgb_views(views, [True, True, False], 12, 10)
    assert (canvas[:, :, :8] == 1).all()
    assert (canvas[:, :, 8:, :5] == 2).all()
    assert (canvas[:, :, 8:, 5:] == 0).all()
    assert not mask[:, :, 8:, 5:].any()


def test_missing_annotations_and_collate(tmp_path, monkeypatch):
    ds, _ = make_data(tmp_path, monkeypatch, False)
    sample = ds[0]
    assert sample["condition_text"] == ""
    assert not sample["metadata_valid"]
    assert not sample["targets"]["goal_view_valid"].any()
    batch = wsa1_5_collate([sample, ds[39]])
    assert batch["observed_rgb"].shape == (2, 3, 384, 320)
    assert len(batch["targets"]["goal_rgb_views"]) == 2
    assert not batch["targets"]["motion_valid"][1].any()


def test_declared_video_decode_failure_raises(tmp_path, monkeypatch):
    ds, _ = make_data(tmp_path, monkeypatch)
    def fail(*args):
        raise RuntimeError("bad media")
    monkeypatch.setattr("lerobot.datasets.wsa1_5.decode_video_frames", fail)
    with pytest.raises(RuntimeError, match="video decode failed"):
        ds[0]


def test_inference_signature():
    condition = build_inference_condition(torch.zeros(3, 12, 10), "test")
    assert set(condition) == {"observed_rgb", "condition_text"}
    with pytest.raises(TypeError):
        build_inference_condition(torch.zeros(3, 12, 10), "test", future_motion=torch.ones(1))


def test_duplicate_views_rejected():
    with pytest.raises(ValueError):
        WSA1_5DatasetConfig(repo_id="test", view_keys=["rgb", "rgb", ""])


@pytest.mark.parametrize("field", ["episode_index", "subtasks", "start_frame"])
def test_null_annotation_fields_clear_condition(tmp_path, monkeypatch, field):
    ds, _ = make_data(tmp_path, monkeypatch)
    path = tmp_path / "meta/sidecar.json"
    sidecar = json.loads(path.read_text())
    if field == "start_frame":
        sidecar[0]["subtasks"][0][field] = None
    else:
        sidecar[0][field] = None
    path.write_text(json.dumps(sidecar))
    rebuilt = WSA1_5Dataset(ds.base, ds.config)
    sample = rebuilt[rebuilt.sample_indices.index(0)]
    assert sample["condition_text"] == "" and not sample["metadata_valid"]
    assert not sample["targets"]["goal_view_valid"].any()
