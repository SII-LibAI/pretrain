from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from lerobot.dataset_schemas import DatasetSchema, get_schema, register_schema
from lerobot.datasets.synthetic_embodiment import (
    SyntheticEmbodimentDataset,
    load_stats_selection,
)
from tools.compute_synthetic_embodiment_stats import (
    compute_embodiment_stats,
    write_embodiment_stats,
)


class _FakeMeta:
    robot_type = "egodex_v"
    total_episodes = 1


class _FakeBaseDataset:
    def __init__(self, root: Path, length: int) -> None:
        self.root = root
        self.repo_id = "fake/egodex"
        self.meta = _FakeMeta()
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
    info_path.write_text("{}", encoding="utf-8")


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
    prefix: str = "test_synthetic_",
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
        ),
        override=True,
    )


def test_adapter_filters_invalid_windows_and_shifts_action(tmp_path: Path) -> None:
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
        schema_prefix="test_synthetic_",
        transforms=[],
    )

    assert len(dataset) == 4
    assert dataset.candidate_embodiments(0) == ("alpha", "beta")
    assert dataset.candidate_embodiments(1) == ("beta",)

    sample = dataset[1]
    scale = 1.0 + 1e-6
    np.testing.assert_allclose(
        sample["observation.state"].numpy(),
        np.asarray([31.0, 11.0, 21.0]) / scale,
    )
    expected_action = np.asarray([[32.0, 1.0, 1.0], [33.0, 2.0, 2.0]]) / scale
    np.testing.assert_allclose(sample["action"].numpy(), expected_action)
    assert get_schema("test_synthetic_beta").action_reorder == [
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
        schema_prefix="test_synthetic_",
        transforms=[],
    )
    assert len(alpha_only) == 1
    alpha_sample = alpha_only[0]
    assert alpha_sample["observation.state"][-1].item() == 0.0
    assert torch.equal(alpha_sample["action"][:, -1], torch.zeros(2))

    beta_only = SyntheticEmbodimentDataset(
        _FakeBaseDataset(root, len(beta)),
        stats_dir=stats_dir,
        chunk_size=2,
        max_state_dim=3,
        max_action_dim=3,
        embodiments=["beta"],
        schema_prefix="test_synthetic_",
        transforms=[],
    )
    beta_sample = beta_only[0]
    assert beta_sample["index"].item() == alpha_sample["index"].item()
    assert beta_sample["task"].startswith("Current robot embodiment: beta;")
    assert alpha_sample["task"].startswith("Current robot embodiment: alpha;")
    assert beta_sample["task"] != alpha_sample["task"]


def test_stats_are_aggregated_from_valid_rows_only(tmp_path: Path) -> None:
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
    )
    _register_test_schema(
        "arm",
        2,
        prefix="egodex_synthetic_",
        action_mask_spec=[1, -1],
    )

    roots = [first.resolve(), second.resolve()]
    stats, specs = compute_embodiment_stats(roots, chunk_size=1)
    output_dir = tmp_path / "stats"
    write_embodiment_stats(output_dir, roots, stats, specs, chunk_size=1)

    assert roots == [first.resolve(), second.resolve()]
    assert stats["arm"]["observation.state"]["count"] == 3
    assert stats["arm"]["action"]["count"] == 3
    np.testing.assert_allclose(stats["arm"]["observation.state"]["mean"], [13 / 3, 16 / 3])
    np.testing.assert_allclose(stats["arm"]["action"]["mean"], [2.0, 22 / 3])
    written = json.loads((output_dir / "arm" / "stats.json").read_text(encoding="utf-8"))
    assert written["observation.state"] != written["action"]
    assert load_stats_selection(
        output_dir,
        ["arm"],
        action_mode="delta",
        chunk_size=1,
    ) == ["arm"]


def test_egodex_reorder_moves_left_side_before_right_side() -> None:
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
        schema = get_schema(f"egodex_synthetic_{name}")
        source = np.arange(dim)
        expected = np.concatenate((source[dim // 2 :], source[: dim // 2]))
        for reorder in (schema.state_reorder, schema.action_reorder):
            actual = np.zeros(dim, dtype=np.int64)
            for src_start, src_end, dst_start, dst_end in reorder:
                actual[dst_start:dst_end] = source[src_start:src_end]
            np.testing.assert_array_equal(actual, expected, err_msg=name)
        assert schema.action_layout[0]["name"].startswith("left_")
