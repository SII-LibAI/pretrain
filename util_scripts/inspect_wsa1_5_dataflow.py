#!/usr/bin/env python
"""Build and inspect the WSA1.5 dataset flow without constructing a policy."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

from lerobot.dataset_schemas import get_schema
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.wsa1_5 import (
    ACTION_VALID_MASK_KEY,
    DEFAULT_HUMAN_ACTION_FILE,
    FUTURE_2D_TARGET_KEY,
    FUTURE_2D_VALID_KEY,
    FUTURE_3D_TARGET_KEY,
    FUTURE_3D_VALID_KEY,
    STATE_VALID_MASK_KEY,
    SUBTASK_TEXT_KEY,
    SUBTASK_VALID_KEY,
    build_wsa15_data_transforms,
    is_wsa15_human_robot_type,
    resolve_wsa15_delta_timestamps,
    resolve_wsa15_video_key_mapping,
    wrap_wsa15_dataset,
    wsa15_source_kind,
)
from lerobot.utils.constants import ACTION, OBS_STATE


def discover_repo_ids(collection_root: Path) -> list[str]:
    """Return dataset roots relative to a collection directory."""

    return sorted(
        str(info.parent.parent.relative_to(collection_root)).replace("\\", "/")
        for info in collection_root.rglob("meta/info.json")
    )


def build_dataflow_dataset(
    *,
    collection_root: Path,
    repo_id: str,
    human_stats_dir: Path | None,
    human_action_file: str,
    human_embodiments: list[str] | None,
    chunk_size: int,
    num_future_frames: int,
    action_mode: str,
    video_backend: str,
):
    """Construct the same robot/human branch used by the training factory."""

    dataset_root = collection_root / repo_id
    metadata = LeRobotDatasetMetadata(repo_id, root=dataset_root)
    human_source = is_wsa15_human_robot_type(metadata.robot_type)
    settings = SimpleNamespace(
        chunk_size=chunk_size,
        num_video_frames=num_future_frames,
        human_stats_dir=human_stats_dir,
        human_action_file=human_action_file,
        human_embodiments=human_embodiments,
        human_annotation_row_group_cache_size=0,
    )
    delta_timestamps = resolve_wsa15_delta_timestamps(
        settings,
        metadata,
        human_source=human_source,
    )
    video_key_mapping = resolve_wsa15_video_key_mapping(metadata)
    base = LeRobotDataset(
        repo_id,
        root=dataset_root,
        delta_timestamps=delta_timestamps,
        video_backend=video_backend,
        video_key_mapping=video_key_mapping,
    )

    transforms = build_wsa15_data_transforms(
        action_mode=action_mode,
        num_future_frames=num_future_frames,
    )
    return wrap_wsa15_dataset(base, settings, transforms), metadata


def _tensor_description(value: torch.Tensor) -> dict[str, Any]:
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype).removeprefix("torch."),
    }


def summarize_sample(sample: dict[str, Any], robot_type: str) -> dict[str, Any]:
    schema = get_schema(robot_type)
    image_key = next(iter(schema.image_mapping.values()), None)
    summary: dict[str, Any] = {
        "index": int(sample["index"]),
        "episode_index": int(sample["episode_index"]),
        "frame_index": int(sample["frame_index"]),
        "state": _tensor_description(sample[OBS_STATE]),
        "action": _tensor_description(sample[ACTION]),
        "subtask": sample.get(SUBTASK_TEXT_KEY, ""),
        "subtask_valid": bool(sample[SUBTASK_VALID_KEY]),
        "future_2d": _tensor_description(sample[FUTURE_2D_TARGET_KEY]),
        "future_2d_valid": bool(sample[FUTURE_2D_VALID_KEY]),
        "future_3d": _tensor_description(sample[FUTURE_3D_TARGET_KEY]),
        "future_3d_valid": bool(sample[FUTURE_3D_VALID_KEY]),
        "state_valid_dimensions": int(sample[STATE_VALID_MASK_KEY].sum()),
        "action_valid_dimensions": int(sample[ACTION_VALID_MASK_KEY].sum()),
    }
    if image_key is not None:
        summary["image"] = {
            "key": image_key,
            **_tensor_description(sample[image_key]),
        }
    if "synthetic_embodiment" in sample:
        summary["human_action_representation"] = sample["synthetic_embodiment"]
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--repo-id", action="append", dest="repo_ids")
    parser.add_argument("--human-stats-dir", type=Path)
    parser.add_argument("--human-action-file", default=DEFAULT_HUMAN_ACTION_FILE)
    parser.add_argument("--human-embodiment", action="append", dest="human_embodiments")
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument("--num-future-frames", type=int, default=4)
    parser.add_argument("--action-mode", choices=("abs", "delta"), default="abs")
    parser.add_argument("--video-backend", default="pyav")
    parser.add_argument("--samples-per-dataset", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    collection_root = args.dataset_root.resolve()
    repo_ids = args.repo_ids or discover_repo_ids(collection_root)
    if not repo_ids:
        raise ValueError(f"No LeRobotV3 dataset found below {collection_root}")
    if args.samples_per_dataset < 1:
        raise ValueError("--samples-per-dataset must be positive")

    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    for repo_id in repo_ids:
        dataset, metadata = build_dataflow_dataset(
            collection_root=collection_root,
            repo_id=repo_id,
            human_stats_dir=args.human_stats_dir,
            human_action_file=args.human_action_file,
            human_embodiments=args.human_embodiments,
            chunk_size=args.chunk_size,
            num_future_frames=args.num_future_frames,
            action_mode=args.action_mode,
            video_backend=args.video_backend,
        )
        report = {
            "repo_id": repo_id,
            "robot_type": metadata.robot_type,
            "source_kind": wsa15_source_kind(metadata.robot_type),
            "eligible_samples": len(dataset),
            "samples": [
                summarize_sample(dataset[rng.randrange(len(dataset))], metadata.robot_type)
                for _ in range(args.samples_per_dataset)
            ],
        }
        print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
