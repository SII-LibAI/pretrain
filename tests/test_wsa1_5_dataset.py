import json
from dataclasses import replace
from types import SimpleNamespace

import datasets
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from lerobot.policies.wsa1_5.configuration_wsa1_5 import WSA1_5DatasetConfig
from lerobot.datasets.utils import load_nested_dataset
from lerobot.policies.wsa1_5.dataset_wsa1_5 import (
    WSA1_5Dataset, build_inference_condition, compose_rgb_views, condition_text, wsa1_5_collate,
)


class FrameTable(dict):
    """Mutable fixture columns with the real HF Arrow-format read interface."""
    def select_columns(self, columns):
        return datasets.Dataset.from_dict({key: self[key] for key in columns})


def make_data(tmp_path, monkeypatch, annotations=True, *, frame_stride=1, include_3d_targets=True,
              future_frames=32):
    tmp_path.mkdir(exist_ok=True)
    (tmp_path / "meta").mkdir(exist_ok=True)
    episode_dir = tmp_path / "meta/episodes/chunk-000"
    episode_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table({"episode_index": [0], "tasks": [["knit"]],
                             "dataset_from_index": [0], "dataset_to_index": [40]}), episode_dir / "file-000.parquet")
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
        "dataset_from_index": 0, "dataset_to_index": 40, "tasks": ["knit"],
    }], robot_type="egodex_v", total_episodes=1, stats={})
    meta.get_video_file_path = lambda ep, key: key
    meta.episodes = {0: meta.episodes[0], "dataset_from_index": [0], "dataset_to_index": [40]}
    base = SimpleNamespace(meta=meta, root=tmp_path, repo_id="test", tolerance_s=1e-4, video_backend="pyav",
                           hf_dataset=FrameTable(timestamp=[i / 30 for i in range(40)],
                                                 frame_index=list(range(40)), episode_index=[0] * 40))
    base._ensure_hf_dataset_loaded = lambda: None
    seen = []
    def decode(path, times, *args):
        seen.append((str(path), times))
        return torch.tensor(times)[:, None, None, None].expand(-1, 3, 16, 16) / 4
    monkeypatch.setattr("lerobot.policies.wsa1_5.dataset_wsa1_5.decode_video_frames", decode)
    config = WSA1_5DatasetConfig(repo_id="test", view_keys=["rgb", "", ""], motion_log_every=0,
                               frame_stride=frame_stride, future_frames=future_frames, use_subtask=True)
    return WSA1_5Dataset(base, config, include_3d_targets=include_3d_targets), seen


def test_boundaries_targets_and_timestamps(tmp_path, monkeypatch):
    ds, seen = make_data(tmp_path, monkeypatch)
    sample = ds[8]
    assert sample["subtask_id"] == 0
    assert sample["goal_timestamp"] == pytest.approx(9 / 30)
    assert sample["targets"]["future_rgb"].shape == (32, 3, 384, 320)
    assert sample["targets"]["future_motion"].shape == (32, 3, 192, 160)
    assert sample["future_frame_valid"].sum() == 31
    assert sample["targets"]["motion_valid"].sum() == 31
    torch.testing.assert_close(sample["frame_timestamps"], torch.tensor([*range(9, 40), 39]) / 30,
                               check_dtype=False)
    assert sample["targets"]["rgb_valid"][:31, :, :256].all()
    assert not sample["targets"]["rgb_valid"][31].any()
    assert (sample["targets"]["future_motion"][31] == -1).all()
    np.testing.assert_allclose(seen[0][1][1:33], seen[1][1])
    assert sample["targets"]["goal_view_valid"].tolist() == [True, False, False]
    assert "both hands" in sample["condition_text"]
    assert "future_motion" not in sample
    assert "goal_rgb_views" not in sample
    assert ds[10]["subtask_id"] == 1
    assert "left hand" in ds[10]["condition_text"]
    assert ds[0]["future_frame_valid"].all()


@pytest.mark.parametrize("pixel", [0.0, 0.5, 1.0])
def test_rgb_motion_use_wan_range_and_teacher_keeps_raw_pixels(tmp_path, monkeypatch, pixel):
    ds, _ = make_data(tmp_path, monkeypatch)
    monkeypatch.setattr("lerobot.policies.wsa1_5.dataset_wsa1_5.decode_video_frames",
                        lambda path, times, *args: torch.full((len(times), 3, 16, 16), pixel))
    sample = ds[8]
    expected = pixel * 2 - 1
    torch.testing.assert_close(sample["observed_rgb"][:, :256],
                               torch.full_like(sample["observed_rgb"][:, :256], expected), rtol=1e-6, atol=2e-7)
    assert (sample["observed_rgb"][:, 256:] == -1).all()
    assert not sample["observed_valid"][:, 256:].any()
    targets = sample["targets"]
    torch.testing.assert_close(targets["future_rgb"][:, :, :256],
                               torch.full_like(targets["future_rgb"][:, :, :256], expected), rtol=1e-6, atol=2e-7)
    assert (targets["future_rgb"][:, :, 256:] == -1).all()
    assert not targets["rgb_valid"][:, :, 256:].any()
    torch.testing.assert_close(targets["future_motion"][:31],
                               torch.full_like(targets["future_motion"][:31], expected), rtol=1e-6, atol=2e-7)
    assert (targets["future_motion"][31] == -1).all()
    assert not targets["motion_valid"][31]
    assert (targets["goal_rgb_views"][0] == pixel).all()


def test_missing_video_returns_explicit_invalid_placeholder(tmp_path, monkeypatch):
    ds, _ = make_data(tmp_path, monkeypatch)
    frames, valid = ds._decode("", [0.0], 0)
    assert frames.shape == (1, 3, 16, 16) and not frames.any()
    assert valid.dtype == torch.bool and not valid.any()


def test_missing_motion_is_zero_before_normalization_and_minus_one_after(tmp_path, monkeypatch):
    ds, _ = make_data(tmp_path, monkeypatch)
    del ds.meta.features[ds.config.motion_key]
    raw, valid = ds._decode(ds.config.motion_key, [0.0], 0)
    assert not raw.any() and not valid.any()
    sample = ds[0]
    assert (sample["targets"]["future_motion"] == -1).all()
    assert not sample["targets"]["motion_valid"].any()


@pytest.mark.parametrize("frame_stride", [1, 2, 5])
@pytest.mark.parametrize("include_3d_targets", [True, False])
def test_future_window_uses_episode_bounds(tmp_path, monkeypatch, frame_stride, include_3d_targets):
    ds, seen = make_data(tmp_path, monkeypatch, frame_stride=frame_stride,
                         include_3d_targets=include_3d_targets)
    requested = 8 + np.arange(1, 33) * frame_stride
    expected_valid = torch.from_numpy(requested < 40)
    sample = ds[8]
    torch.testing.assert_close(sample["future_frame_valid"], expected_valid)
    torch.testing.assert_close(sample["targets"]["motion_valid"], expected_valid)
    np.testing.assert_allclose(sample["frame_timestamps"], np.minimum(requested, 39) / 30)
    assert ("goal_rgb_views" in sample["targets"]) == include_3d_targets
    assert len(seen[0][1]) == (34 if include_3d_targets else 33)
    if include_3d_targets:
        assert seen[0][1][-1] == pytest.approx(2 + 9 / 30)
    assert sample["goal_timestamp"] == pytest.approx(9 / 30)
    assert "Subtask: loop" in sample["condition_text"]


def test_future_window_never_crosses_episodes_after_filtering(tmp_path, monkeypatch):
    ds, _ = make_data(tmp_path, monkeypatch)
    source = ds.base
    # Nonconsecutive, nonascending IDs with original global offsets must use local row bounds.
    entry = {**source.meta.episodes[0], "dataset_from_index": 1000, "dataset_to_index": 1040}
    source.meta.episodes = {7: entry, 2: {**entry, "dataset_from_index": 200, "dataset_to_index": 240}}
    source.hf_dataset["episode_index"] = [7] * 40 + [2] * 40
    source.hf_dataset["frame_index"] *= 2
    source.hf_dataset["timestamp"] *= 2
    path = tmp_path / "meta/sidecar.json"
    sidecar = json.loads(path.read_text())
    sidecar[0]["episode_index"] = 7
    path.write_text(json.dumps(sidecar))
    rebuilt = WSA1_5Dataset(source, ds.config)
    assert rebuilt.meta.episodes[7]["dataset_from_index"] == 0
    assert rebuilt.meta.episodes[2]["dataset_from_index"] == 40
    assert rebuilt.episode_indices[rebuilt.sample_indices[0]] == 7
    assert rebuilt.episode_indices[rebuilt.sample_indices[40]] == 2
    for ep in [7, 2]:
        for frame in [0, 8, 39]:
            row = (0 if ep == 7 else 40) + frame
            sample = rebuilt[np.flatnonzero(rebuilt.sample_indices == row)[0]]
            requested = np.arange(1, 33) + frame
            np.testing.assert_allclose(sample["frame_timestamps"], np.minimum(requested, 39) / 30)
            torch.testing.assert_close(sample["future_frame_valid"], torch.from_numpy(requested < 40))
            assert sample["episode_index"] == ep


def test_layout():
    views = [torch.full((1, 3, 4, 4), float(i)) for i in [1, 2, 3]]
    canvas, mask = compose_rgb_views(views, [True, True, False], 12, 10)
    assert (canvas[:, :, :8] == 1).all()
    assert (canvas[:, :, 8:, :5] == 2).all()
    assert (canvas[:, :, 8:, 5:] == 0).all()
    assert not mask[:, :, 8:, 5:].any()


def test_filtered_reader_handles_parquet_column_order(tmp_path):
    chunk = tmp_path / "chunk-000"
    chunk.mkdir()
    pq.write_table(pa.table({"timestamp": [0.0, 1.0, 2.0], "episode_index": [0, 1, 1]}),
                   chunk / "file-000.parquet")
    features = datasets.Features({"episode_index": datasets.Value("int64"),
                                  "timestamp": datasets.Value("float32")})
    filtered = load_nested_dataset(tmp_path, features=features, episodes=[1])
    assert filtered.column_names == list(features)
    assert filtered["episode_index"] == [1, 1]
    assert filtered["timestamp"] == [1.0, 2.0]
    assert filtered.features == features


@pytest.mark.parametrize("columns", [
    {"episode_index": [0]},
    {"episode_index": [0], "timestamp": [0.0], "unexpected": [1]},
])
def test_filtered_reader_rejects_missing_or_extra_columns(tmp_path, columns):
    chunk = tmp_path / "chunk-000"
    chunk.mkdir()
    pq.write_table(pa.table(columns), chunk / "file-000.parquet")
    features = datasets.Features({"episode_index": datasets.Value("int64"),
                                  "timestamp": datasets.Value("float32")})
    with pytest.raises(ValueError, match="do not match feature schema"):
        load_nested_dataset(tmp_path, features=features, episodes=[0])


def test_missing_annotations_and_collate(tmp_path, monkeypatch):
    ds, _ = make_data(tmp_path, monkeypatch, False)
    sample = ds[0]
    assert sample["condition_text"] == condition_text({"task_instruction": "knit"})
    assert not sample["metadata_valid"]
    assert not sample["targets"]["goal_view_valid"].any()
    batch = wsa1_5_collate([sample, ds[39]])
    assert batch["observed_rgb"].shape == (2, 3, 384, 320)
    assert len(batch["targets"]["goal_rgb_views"]) == 2
    assert not batch["targets"]["motion_valid"][1].any()


@pytest.mark.parametrize("annotations", [False, True])
@pytest.mark.parametrize("use_subtask", [False, True])
@pytest.mark.parametrize("allow_endframe", [False, True])
def test_stage1_subtask_and_episode_endpoints(tmp_path, monkeypatch, annotations, use_subtask, allow_endframe):
    fixture, _ = make_data(tmp_path, monkeypatch, annotations=annotations, future_frames=4)
    config = replace(fixture.config, use_subtask=use_subtask, allow_endframe_3d=allow_endframe)
    dataset = WSA1_5Dataset(fixture.base, config)
    assert dataset.action_config is None
    if not use_subtask:
        assert all(segment.subtask == -1 and not segment.metadata_valid for segment in dataset.segments)
    for index in (3, 8, 38):  # Separate endpoint decode, endpoint in window, and episode padding.
        sample = dataset[index]
        has_subtask = annotations and use_subtask
        endpoint = (9 if index < 10 else 39) if has_subtask else (39 if allow_endframe else None)
        assert sample["targets"]["goal_view_valid"].tolist() == [endpoint is not None, False, False]
        assert sample["metadata_valid"] == has_subtask
        assert ("Subtask:" in sample["condition_text"]) == has_subtask
        assert "Task: knit" in sample["condition_text"]
        assert "state" not in sample and "action" not in sample["targets"]
        if endpoint is not None:
            assert sample["goal_timestamp"] == pytest.approx(endpoint / 30)
            goal = sample["targets"]["goal_rgb_views"][0]
            torch.testing.assert_close(goal, torch.full_like(goal, (2 + endpoint / 30) / 4))
        else:
            assert not any(view.any() for view in sample["targets"]["goal_rgb_views"])
    dataset.include_3d_targets = False
    assert "goal_rgb_views" not in dataset[3]["targets"]


def test_stage1_task_only_ignores_subtask_bounds_but_keeps_original_task(tmp_path, monkeypatch):
    fixture, _ = make_data(tmp_path, monkeypatch, future_frames=4)
    sidecar = [{"episode_index": 0, "task_instruction": "original task",
                "subtasks": [{"start_frame": 99, "end_frame": 1, "subtask_id": 0}]}]
    (tmp_path / "meta/sidecar.json").write_text(json.dumps(sidecar), encoding="utf-8")
    dataset = WSA1_5Dataset(fixture.base, replace(fixture.config, use_subtask=False, allow_endframe_3d=True))
    assert dataset[3]["condition_text"] == condition_text({"task_instruction": "original task"})
    assert dataset[3]["goal_timestamp"] == pytest.approx(39 / 30)
    with pytest.raises(ValueError, match="bounds"):
        WSA1_5Dataset(fixture.base, fixture.config)


def test_video_adaptation_trains_rgb_and_endframe_3d_without_actions(tmp_path, monkeypatch):
    from test_wsa1_5_action import action_policy

    fixture, _ = make_data(tmp_path, monkeypatch, future_frames=4)
    config = replace(fixture.config, use_subtask=False, allow_endframe_3d=True,
                     rgb_height=96, rgb_width=64, motion_height=32, motion_width=32)
    dataset = WSA1_5Dataset(fixture.base, config)
    batch = wsa1_5_collate([dataset[3]])
    policy = action_policy(enable_action=False, lambda_rgb=1, lambda_3d=1)
    batch["text_embedding"] = torch.randn(1, 4, 8)
    batch["text_valid"] = torch.ones(1, 4, dtype=torch.bool)

    class Teacher:
        def features(self, views, valid, device):
            torch.testing.assert_close(views[0][0], torch.full_like(views[0][0], (2 + 39 / 30) / 4))
            return ({layer: torch.ones(1, 3, 273, 8, device=device) for layer in policy.config.teacher_layers},
                    valid[..., None].expand(-1, -1, 273))

    policy.teacher = Teacher()
    assert policy.action_expert is None and policy.proprio_encoder is None
    assert "action" not in {group["name"] for group in policy.get_optim_params()}
    loss, metrics = policy.training_forward(batch)
    assert metrics["loss_rgb_flow"] > 0 and metrics["loss_3d_feature"] > 0
    assert "loss_action_flow" not in metrics and torch.isfinite(loss)
    loss.backward()
    for weight in [policy.flow.rgb.patch_embedding.weight, policy.future_3d.feature_heads["4"][1].weight]:
        assert weight.grad is not None and torch.isfinite(weight.grad).all() and weight.grad.abs().sum() > 0


def test_declared_video_decode_failure_raises(tmp_path, monkeypatch):
    ds, _ = make_data(tmp_path, monkeypatch)
    def fail(*args):
        raise RuntimeError("bad media")
    monkeypatch.setattr("lerobot.policies.wsa1_5.dataset_wsa1_5.decode_video_frames", fail)
    with pytest.raises(RuntimeError, match="video decode failed"):
        ds[0]


@pytest.mark.parametrize("column", ["frame_index", "episode_index"])
def test_fractional_source_indices_are_not_truncated(tmp_path, monkeypatch, column):
    dataset, _ = make_data(tmp_path, monkeypatch)
    dataset.base.hf_dataset[column][0] = 0.5
    with pytest.raises(TypeError, match="integer"):
        WSA1_5Dataset(dataset.base, dataset.config)


def test_inference_signature():
    condition = build_inference_condition(torch.zeros(3, 12, 10), "test")
    assert set(condition) == {"observed_rgb", "condition_text"}
    with pytest.raises(TypeError):
        build_inference_condition(torch.zeros(3, 12, 10), "test", future_motion=torch.ones(1))


def test_duplicate_views_rejected():
    with pytest.raises(ValueError):
        WSA1_5DatasetConfig(repo_id="test", view_keys=["rgb", "rgb", ""])


@pytest.mark.parametrize("field", ["episode_index", "subtasks", "start_frame"])
def test_null_annotation_fields_retain_original_task(tmp_path, monkeypatch, field):
    ds, _ = make_data(tmp_path, monkeypatch)
    path = tmp_path / "meta/sidecar.json"
    sidecar = json.loads(path.read_text())
    if field == "start_frame":
        sidecar[0]["subtasks"][0][field] = None
    else:
        sidecar[0][field] = None
    path.write_text(json.dumps(sidecar))
    rebuilt = WSA1_5Dataset(ds.base, ds.config)
    sample = rebuilt[np.flatnonzero(rebuilt.sample_indices == 0)[0]]
    assert sample["condition_text"] == condition_text({"task_instruction": "knit"})
    assert not sample["metadata_valid"]
    assert not sample["targets"]["goal_view_valid"].any()


def test_subtask_gaps_keep_task_only_text_and_mask_3d(tmp_path, monkeypatch):
    ds, _ = make_data(tmp_path, monkeypatch)
    path = tmp_path / "meta/sidecar.json"
    sidecar = json.loads(path.read_text())
    sidecar[0]["subtasks"] = sidecar[0]["subtasks"][:1]
    path.write_text(json.dumps(sidecar))
    rebuilt = WSA1_5Dataset(ds.base, ds.config)
    prompt = "You are a robot control agent,follow the instruction.\nTask: knit"
    for index in (10, 20, 39):
        sample = rebuilt[index]
        assert sample["condition_text"] == prompt
        assert sample["subtask_id"] == -1 and not sample["metadata_valid"]
        assert not sample["targets"]["goal_view_valid"].any()
    assert "Subtask: loop" in rebuilt[0]["condition_text"]


def test_missing_original_task_is_not_replaced_by_empty_text(tmp_path, monkeypatch):
    ds, _ = make_data(tmp_path, monkeypatch, annotations=False)
    ds.base.meta.episodes[0]["tasks"] = []
    with pytest.raises(ValueError, match="original task"):
        WSA1_5Dataset(ds.base, ds.config)
