"""Synthetic metadata/video checks; no pretrained assets or optimizer updates."""
import datasets
import numpy as np
import pytest
import torch

from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import make_dataloader
from lerobot.policies.wsa1_5.configuration_wsa1_5 import WSA1_5Config, WSA1_5DatasetConfig
from lerobot.policies.wsa1_5.dataset_wsa1_5 import WSA1_5Dataset, wsa1_5_collate
from test_wsa1_5_dataset import make_data


def test_arrow_read_bypasses_torch_transform_preserves_precision_and_episode_order(tmp_path, monkeypatch):
    dataset, _ = make_data(tmp_path, monkeypatch)
    base = dataset.base
    base.meta.episodes = {7: base.meta.episodes[0], 2: base.meta.episodes[0]}
    table = datasets.Dataset.from_dict(
        {"timestamp": [1200 + i / 30 for i in range(40)] * 2,
         "frame_index": list(range(40)) * 2, "episode_index": [7] * 40 + [2] * 40},
        features=datasets.Features({"timestamp": datasets.Value("float64"),
                                    "frame_index": datasets.Value("int64"), "episode_index": datasets.Value("int64")}))
    # Select in reverse episode order through HF's row-index mapping.
    base.hf_dataset = table.select(list(range(40, 80)) + list(range(40)))
    base.hf_dataset.set_transform(lambda _: pytest.fail("Bulk init must not call the Torch transform"))
    (tmp_path / "meta/sidecar.json").unlink()
    rebuilt = WSA1_5Dataset(base, dataset.config)
    assert list(rebuilt.meta.entries) == [2, 7]
    np.testing.assert_array_equal(rebuilt.timestamps[:40], np.array([1200 + i / 30 for i in range(40)]))
    assert rebuilt.meta.episodes[7]["dataset_from_index"] == 40
    assert rebuilt.sample_indices.dtype == np.int64 and rebuilt.segment_for_sample.dtype == np.int32


def test_noncontiguous_episodes_are_still_rejected(tmp_path, monkeypatch):
    dataset, _ = make_data(tmp_path, monkeypatch, annotations=False)
    dataset.base.hf_dataset["episode_index"] = [0] * 10 + [1] * 10 + [0] * 20
    with pytest.raises(ValueError, match="not contiguous"):
        WSA1_5Dataset(dataset.base, dataset.config)


@pytest.mark.parametrize("value", [2**63, 2**64 - 1])
def test_unsigned_indices_cannot_wrap_to_negative(tmp_path, monkeypatch, value):
    dataset, _ = make_data(tmp_path, monkeypatch, annotations=False)
    dataset.base.hf_dataset["episode_index"] = np.full(40, value, dtype=np.uint64)
    with pytest.raises(ValueError, match="exceeds int64 range"):
        WSA1_5Dataset(dataset.base, dataset.config)


@pytest.mark.parametrize("annotations", [True, False])
def test_far_goal_is_a_separate_request_and_has_compact_storage(tmp_path, monkeypatch, annotations):
    dataset, seen = make_data(tmp_path, monkeypatch, annotations=annotations, future_frames=4)
    sample = dataset[0]
    # Window 0..4; annotated goal is 9, unannotated samples need no endpoint read.
    assert seen[0][1] == pytest.approx([2 + i / 30 for i in range(5)])
    assert seen[1][1] == pytest.approx([2 + i / 30 for i in range(1, 5)])
    assert len(seen) == (3 if annotations else 2)
    if annotations:
        assert seen[2][1] == pytest.approx([2 + 9 / 30])
    batch = wsa1_5_collate([sample])
    for view in batch["targets"]["goal_rgb_views"][0]:
        assert view.untyped_storage().nbytes() == view.numel() * view.element_size()
        assert view.storage_offset() == 0
    assert sample["targets"]["goal_view_valid"].tolist() == [annotations, False, False]


def test_near_goal_is_reused_without_exporting_video_storage(tmp_path, monkeypatch):
    dataset, seen = make_data(tmp_path, monkeypatch)
    sample = dataset[8]
    assert len(seen) == 2  # RGB window including goal, and motion window.
    goal = sample["targets"]["goal_rgb_views"][0]
    assert goal.untyped_storage().nbytes() == goal.numel() * goal.element_size()
    assert goal.storage_offset() == 0
    torch.testing.assert_close(goal, torch.full_like(goal, (2 + 9 / 30) / 4))


@pytest.mark.parametrize("workers", [0, 2])
@pytest.mark.parametrize("weighted", [False, True])
def test_wsa_loader_prefetch_and_persistent_workers(tmp_path, monkeypatch, workers, weighted):
    from lerobot.datasets.transformed_dataset import MultiLeRobotDataset

    dataset, _ = make_data(tmp_path, monkeypatch)
    if weighted:
        dataset = MultiLeRobotDataset([dataset], dataset_weights=[1.0])
    cfg = TrainPipelineConfig(policy=WSA1_5Config(device="cpu"),
                              dataset=WSA1_5DatasetConfig(repo_id="test", prefetch_factor=1),
                              num_workers=workers, batch_size=2)
    loader, _ = make_dataloader(cfg, dataset)
    assert loader.num_workers == workers and loader.persistent_workers == (workers > 0)
    assert loader.prefetch_factor == (1 if workers else None)


@pytest.mark.parametrize("name", ["prefetch_factor", "video_decode_threads"])
@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_invalid_resource_limits_are_rejected(name, value):
    with pytest.raises(ValueError, match="positive integer"):
        WSA1_5DatasetConfig(repo_id="test", **{name: value})
