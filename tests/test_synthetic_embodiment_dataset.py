from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from lerobot.dataset_schemas import DatasetSchema, get_schema, register_schema
from lerobot.datasets.synthetic_embodiment import (
    EmbodimentSpec,
    SyntheticEmbodimentDataset,
    load_embodiment_stats,
)
from util_scripts.compute_synthetic_embodiment_stats import (
    RunningVectorStats,
    compute_embodiment_stats,
    write_embodiment_stats,
)


class _FakeMeta:
    total_episodes = 1

    def __init__(self, robot_type: str = "egodex_v") -> None:
        self.robot_type = robot_type


class _FakeBaseDataset:
    def __init__(self, root: Path, length: int, robot_type: str = "egodex_v") -> None:
        self.root = root
        self.repo_id = "fake/egodex"
        self.meta = _FakeMeta(robot_type)
        self.fps = 30
        self.num_episodes = 1
        self.hf_dataset = {
            "index": list(range(length)),
            "episode_index": [0] * length,
            "frame_index": list(range(length)),
        }

    def __getitem__(self, idx: int) -> dict:
        return {
            "index": torch.tensor(idx),
            "observation.state": torch.full((8,), -1.0),
            "action": torch.full((8,), -1.0),
            "task": "fold the cloth",
        }


def _write_annotations(
    root: Path,
    body_values: dict[str, list[list[float]]],
    body_valid: dict[str, list[bool]],
    robot_type: str = "egodex_v",
) -> None:
    length = len(next(iter(body_values.values())))
    payload = {
        "index": list(range(length)),
        "episode_index": [0] * length,
        "frame_index": list(range(length)),
    }
    for name, values in body_values.items():
        payload[f"action.{name}"] = values
        payload[f"action.{name}_valid"] = body_valid[name]
    path = root / "meta" / "annotations" / "action.parquet"
    path.parent.mkdir(parents=True)
    pq.write_table(pa.table(payload), path)
    info_path = root / "meta" / "info.json"
    info_path.write_text(json.dumps({"robot_type": robot_type}), encoding="utf-8")


def _write_unit_stats(stats_dir: Path, dimensions: dict[str, int]) -> None:
    for name, dim in dimensions.items():
        body_dir = stats_dir / name
        body_dir.mkdir(parents=True)
        feature_stats = {
            "mean": [0.0] * dim,
            "std": [1.0] * dim,
            "min": [0.0] * dim,
            "max": [1.0] * dim,
            "count": 1,
        }
        (body_dir / "stats.json").write_text(
            json.dumps({"observation.state": feature_stats, "action": feature_stats}),
            encoding="utf-8",
        )


def _register_test_schema(
    name: str,
    dim: int,
    *,
    prefix: str = "egodex_synthetic_",
    action_mask_spec: list[int] | None = None,
    reorder: list[list[int]] | None = None,
) -> None:
    reorder = reorder or [[0, dim, 0, dim]]
    register_schema(
        DatasetSchema(
            robot_type=f"{prefix}{name}",
            state_dim=dim,
            action_dim=dim,
            action_mask_spec=action_mask_spec or [dim],
            action_reorder=reorder,
            state_reorder=reorder,
            action_layout=[{"name": "joints", "size": dim}],
            feature_mapping={"observation.state": ["observation.state"], "action": ["action"]},
        ),
        override=True,
    )


def test_adapter_filters_invalid_windows_and_shifts_raw_action(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    alpha = [[float(i), float(i) + 0.5] for i in range(6)]
    beta = [[float(i) + 10, float(i) + 20, float(i) + 30] for i in range(6)]
    _write_annotations(
        root,
        {"alpha": alpha, "beta": beta},
        {
            "alpha": [True, True, True, False, True, True],
            "beta": [True] * 6,
        },
    )
    stats_dir = tmp_path / "stats"
    _write_unit_stats(stats_dir, {"alpha": 2, "beta": 3})
    _register_test_schema("alpha", 2)
    _register_test_schema(
        "beta",
        3,
        action_mask_spec=[2, -1],
        reorder=[[0, 2, 1, 3], [2, 3, 0, 1]],
    )

    dataset = SyntheticEmbodimentDataset(
        _FakeBaseDataset(root, len(alpha)),
        stats_dir=stats_dir,
        chunk_size=2,
        max_state_dim=3,
        max_action_dim=3,
        transforms=[],
    )

    assert len(dataset) == 4
    assert dataset.candidate_embodiments(0) == ("alpha", "beta")
    assert dataset.candidate_embodiments(1) == ("beta",)

    sample = dataset[1]
    np.testing.assert_allclose(sample["observation.state"].numpy(), np.asarray(beta[1]))
    np.testing.assert_allclose(sample["action"].numpy(), np.asarray(beta[2:4]))
    assert get_schema("egodex_synthetic_beta").action_reorder == [
        [0, 2, 1, 3],
        [2, 3, 0, 1],
    ]
    assert sample["synthetic_embodiment"] == "beta"
    assert sample["task"].startswith("Current robot embodiment: beta; task: fold the cloth")
    assert sample["task"].endswith("action layout: joints[3]")

    alpha_only = SyntheticEmbodimentDataset(
        _FakeBaseDataset(root, len(alpha)),
        stats_dir=stats_dir,
        chunk_size=2,
        max_state_dim=3,
        max_action_dim=3,
        embodiments=["alpha"],
        transforms=[],
    )
    assert len(alpha_only) == 1
    alpha_sample = alpha_only[0]
    assert alpha_sample["observation.state"].shape == (2,)
    assert alpha_sample["action"].shape == (2, 2)

    beta_only = SyntheticEmbodimentDataset(
        _FakeBaseDataset(root, len(beta)),
        stats_dir=stats_dir,
        chunk_size=2,
        max_state_dim=3,
        max_action_dim=3,
        embodiments=["beta"],
        transforms=[],
    )
    beta_sample = beta_only[0]
    assert beta_sample["index"].item() == alpha_sample["index"].item()
    assert beta_sample["task"].startswith("Current robot embodiment: beta;")
    assert alpha_sample["task"].startswith("Current robot embodiment: alpha;")
    assert beta_sample["task"] != alpha_sample["task"]


def test_sources_use_separate_schemas_and_shared_stats(tmp_path: Path) -> None:
    stats_dir = tmp_path / "stats"
    _write_unit_stats(stats_dir, {"franka": 18})
    values = [[float(frame + joint) for joint in range(18)] for frame in range(3)]

    for source, expected_schema, camera_key in (
        ("egodex_v", "egodex_synthetic_franka", "observation.image"),
        (
            "egoverse_human_bimanual",
            "egoverse_synthetic_franka",
            "observation.images.front",
        ),
    ):
        assert get_schema(source).image_mapping == {camera_key: "observation.images.image0"}
        root = tmp_path / source
        _write_annotations(root, {"franka": values}, {"franka": [True] * 3}, source)
        dataset = SyntheticEmbodimentDataset(
            _FakeBaseDataset(root, 3, source),
            stats_dir=stats_dir,
            chunk_size=1,
            max_state_dim=18,
            max_action_dim=18,
            transforms=[],
        )
        assert dataset.embodiment_schemas[0].robot_type == expected_schema
        assert dataset.embodiment_schemas[0].image_mapping == {
            camera_key: "observation.images.image0"
        }
        np.testing.assert_allclose(dataset.embodiment_stats["franka"].state_std, 1)


def test_stats_are_aggregated_from_valid_rows_only(tmp_path: Path) -> None:
    pytest.importorskip("draccus")
    pytest.importorskip("datasets")
    pytest.importorskip("av")
    first = tmp_path / "collection" / "first"
    second = tmp_path / "collection" / "second"
    _write_annotations(
        first,
        {"arm": [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]},
        {"arm": [True, True, False]},
    )
    _write_annotations(
        second,
        {"arm": [[5.0, 6.0], [7.0, 8.0], [9.0, 10.0]]},
        {"arm": [True, True, True]},
        robot_type="egoverse_human_bimanual",
    )
    _register_test_schema(
        "arm",
        2,
        prefix="egodex_synthetic_",
        action_mask_spec=[1, -1],
    )
    _register_test_schema(
        "arm",
        2,
        prefix="egoverse_synthetic_",
        action_mask_spec=[1, -1],
    )

    roots = [first.resolve(), second.resolve()]
    stats, specs = compute_embodiment_stats(roots, chunk_size=1)
    output_dir = tmp_path / "stats"
    write_embodiment_stats(output_dir, stats)

    assert roots == [first.resolve(), second.resolve()]
    assert stats["arm"]["observation.state"]["count"] == 3
    assert stats["arm"]["action"]["count"] == 3
    np.testing.assert_allclose(stats["arm"]["observation.state"]["mean"], [13 / 3, 16 / 3])
    np.testing.assert_allclose(stats["arm"]["action"]["mean"], [2.0, 22 / 3])
    written = json.loads((output_dir / "arm" / "stats.json").read_text(encoding="utf-8"))
    assert written["observation.state"] != written["action"]
    assert not (output_dir / "manifest.json").exists()
    np.testing.assert_allclose(
        load_embodiment_stats(output_dir, specs)["arm"].action_mean,
        stats["arm"]["action"]["mean"],
    )


def test_parallel_stats_match_serial(tmp_path: Path) -> None:
    pytest.importorskip("draccus")
    pytest.importorskip("datasets")
    pytest.importorskip("av")
    roots = [tmp_path / "first", tmp_path / "second"]
    for offset, root in enumerate(roots):
        values = [[float(offset * 10 + frame + dim) for dim in range(18)] for frame in range(4)]
        _write_annotations(root, {"franka": values}, {"franka": [True] * 4})
    serial, specs = compute_embodiment_stats(roots, chunk_size=2, num_workers=1)
    parallel, parallel_specs = compute_embodiment_stats(roots, chunk_size=2, num_workers=2)
    assert specs == parallel_specs
    for key in ("observation.state", "action"):
        for stat in ("mean", "std", "min", "max"):
            np.testing.assert_allclose(
                parallel["franka"][key][stat], serial["franka"][key][stat]
            )
        assert parallel["franka"][key]["count"] == serial["franka"][key]["count"]


def test_adapter_uses_source_delta_normalize_and_reorder(tmp_path: Path) -> None:
    pytest.importorskip("draccus")
    pytest.importorskip("datasets")
    pytest.importorskip("av")
    from lerobot.transforms.core import (
        DeltaActionTransformFn,
        NormalizeTransformFn,
        PadStateAndActionTransformFn,
        ReorderStateActionTransform,
    )

    root = tmp_path / "dataset"
    _write_annotations(
        root,
        {"arm": [[1.0, 10.0], [3.0, 20.0], [5.0, 30.0]]},
        {"arm": [True] * 3},
    )
    _register_test_schema(
        "arm",
        2,
        action_mask_spec=[1, -1],
        reorder=[[1, 2, 0, 1], [0, 1, 1, 2]],
    )
    stats_dir = tmp_path / "stats"
    _write_unit_stats(stats_dir, {"arm": 2})
    dataset = SyntheticEmbodimentDataset(
        _FakeBaseDataset(root, 3),
        stats_dir=stats_dir,
        chunk_size=1,
        max_state_dim=3,
        max_action_dim=3,
        transforms=[
            DeltaActionTransformFn(),
            NormalizeTransformFn(),
            PadStateAndActionTransformFn(),
            ReorderStateActionTransform(),
        ],
    )
    sample = dataset[0]
    scale = 1.0 + 1e-6
    np.testing.assert_allclose(sample["observation.state"].numpy(), [10.0 / scale, 1.0 / scale, 0.0])
    np.testing.assert_allclose(sample["action"].numpy(), [[20.0 / scale, 2.0 / scale, 0.0]])
    assert "arm" in sample["task"]


def test_running_stats_merge_matches_single_pass() -> None:
    values = np.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 8.0]])
    whole = RunningVectorStats(2)
    whole.update(values)
    left = RunningVectorStats(2)
    right = RunningVectorStats(2)
    left.update(values[:1])
    right.update(values[1:])
    left.merge(right)
    for key in ("mean", "std", "min", "max"):
        np.testing.assert_allclose(left.as_dict()[key], whole.as_dict()[key])
    assert left.count == whole.count


def test_stats_files_work_without_manifest(tmp_path: Path) -> None:
    accumulator = RunningVectorStats(2)
    accumulator.update(np.asarray([[1.0, 2.0], [3.0, 4.0]]))
    stats = {"arm": {"observation.state": accumulator.as_dict(), "action": accumulator.as_dict()}}
    write_embodiment_stats(tmp_path, stats)

    assert not (tmp_path / "manifest.json").exists()
    spec = EmbodimentSpec("arm", "action.arm", "action.arm_valid", 2)
    loaded = load_embodiment_stats(tmp_path, [spec])["arm"]
    np.testing.assert_allclose(loaded.state_mean, [2.0, 3.0])
    np.testing.assert_allclose(loaded.action_std, [1.0, 1.0])


def test_source_schemas_share_action_layout_and_reorder() -> None:
    dimensions = {
        "piper_inspire": 40,
        "franka": 18,
        "arx_l5": 16,
        "xarm7": 14,
        "rm75": 14,
        "ur7e": 12,
        "ur7e_wuji": 52,
    }
    for name, dim in dimensions.items():
        egodex = get_schema(f"egodex_synthetic_{name}")
        egoverse = get_schema(f"egoverse_synthetic_{name}")
        assert egodex.action_mask_spec == egoverse.action_mask_spec
        assert egodex.action_layout == egoverse.action_layout
        assert egodex.state_reorder == egoverse.state_reorder
        assert egodex.action_reorder == egoverse.action_reorder
        source = np.arange(dim)
        expected = np.concatenate((source[dim // 2 :], source[: dim // 2]))
        for reorder in (egodex.state_reorder, egodex.action_reorder):
            actual = np.zeros(dim, dtype=np.int64)
            for src_start, src_end, dst_start, dst_end in reorder:
                actual[dst_start:dst_end] = source[src_start:src_end]
            np.testing.assert_array_equal(actual, expected, err_msg=name)
        assert egodex.action_layout[0]["name"].startswith("left_")
