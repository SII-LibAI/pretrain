#!/usr/bin/env python
"""Build and inspect the WSA1.5 dataset flow without constructing a policy."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

from lerobot.dataset_schemas import get_schema
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.synthetic_embodiment import HUMAN_EEF_EMBODIMENT
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
    CanonicalStateActionTransformFn,
    build_wsa15_data_transforms,
    is_wsa15_human_robot_type,
    resolve_wsa15_delta_timestamps,
    resolve_wsa15_video_key_mapping,
    validate_wsa15_normalization_stats,
    wrap_wsa15_dataset,
    wsa15_source_kind,
)
from lerobot.transforms.core import ComposeFieldsTransform, DataDict, NormalizeTransformFn
from lerobot.utils.constants import ACTION, OBS_STATE

INSPECTION_KEY = "wsa.inspect.state_action"
NORMALIZATION_SNAPSHOT_KEY = "wsa.inspect.normalization_snapshot"


def _source_indices(raw_dim: int, reorder: list[list[int]] | None) -> list[int]:
    """Resolve logical dimensions independently of the codec's matrix math."""

    if not reorder:
        return list(range(raw_dim))
    indices = [-1] * max(entry[3] for entry in reorder)
    for src_start, src_end, dst_start, dst_end in reorder:
        indices[dst_start:dst_end] = list(range(src_start, src_end))
    return [index for index in indices if index >= 0]


@dataclass
class InspectWSA15NormalizationFn(NormalizeTransformFn):
    """Run the repository normalizer and retain values for slot inspection."""

    mapping: dict[str, list[str]] = field(default_factory=dict)

    def hydrate(self, dataset) -> InspectWSA15NormalizationFn:
        validate_wsa15_normalization_stats(dataset)
        schema = get_schema(dataset.meta.robot_type)
        return replace(
            super().hydrate(dataset),
            mapping={OBS_STATE: schema.get_state_keys(), ACTION: schema.get_action_keys()},
        )

    def __call__(self, data: DataDict) -> DataDict:
        original = {key: data[key].clone() for key in self.selected_keys}
        reference = NormalizeTransformFn(
            selected_keys=self.selected_keys, mode=self.mode, norm_stats=self.norm_stats
        )({key: value.clone() for key, value in original.items()})
        output = super().__call__(data)
        compose = ComposeFieldsTransform(mapping=self.mapping)
        original = compose(original)
        reference = compose(reference)
        output[NORMALIZATION_SNAPSHOT_KEY] = {
            key: {
                "input": original[key],
                "expected": reference[key],
                "mean": torch.cat([torch.as_tensor(self.norm_stats[source]["mean"]).flatten() for source in sources]),
                "std": torch.cat([torch.as_tensor(self.norm_stats[source]["std"]).flatten() for source in sources]),
                "mode": self.mode,
            }
            for key, sources in self.mapping.items()
        }
        return output


def _component_names(name: str, size: int) -> list[str]:
    if "eef_position" in name or "ee_pos" in name:
        return ["x", "y", "z"]
    if "quaternion" in name:
        return ["qx", "qy", "qz", "qw"]
    return [str(index) for index in range(size)]


@dataclass
class InspectWSA15StateActionFn(CanonicalStateActionTransformFn):
    """Audit the actual sampled embodiment immediately around canonicalization.

    NormalizeTransformFn already produced the model values. Compare each
    canonical slot to the repository-normalized reference retained upstream;
    this inspector implements no normalization formula or missing-stats fallback.
    """

    def __call__(self, data: DataDict) -> DataDict:
        snapshot = data.pop(NORMALIZATION_SNAPSHOT_KEY)
        output = super().__call__(data)
        report: dict[str, Any] = {"schema_robot_type": self.robot_type}

        for label, key, reorder, mask_key in (
            ("state", OBS_STATE, self.state_reorder, STATE_VALID_MASK_KEY),
            ("action", ACTION, self.action_reorder, ACTION_VALID_MASK_KEY),
        ):
            captured = snapshot[key]
            values = captured["input"]
            source_indices = _source_indices(values.shape[-1], reorder)
            mean, std = captured["mean"], captured["std"]
            dimensions = []
            expected_values = []
            actual_values = []
            offset = 0
            for block in self.action_layout:
                name, size = str(block["name"]), int(block["size"])
                slots = self._slot_indices(name, size)
                components = _component_names(name.lower(), size)
                for component, slot, source in zip(
                    components, slots, source_indices[offset : offset + size], strict=True
                ):
                    expected = captured["expected"][..., source]
                    actual = output[key][..., slot]
                    expected_values.append(expected)
                    actual_values.append(actual)
                    before = values[..., source]
                    dimensions.append(
                        {
                            "index": slot,
                            "semantic": f"{name}.{component}",
                            "source_index": source,
                            "input_value" if label == "state" else "input_value_t0": float(before.flatten()[0]),
                            "value" if label == "state" else "value_t0": float(actual.flatten()[0]),
                            "mean": float(mean[source]),
                            "std": float(std[source]),
                        }
                    )
                offset += size

            expected = torch.stack(expected_values, dim=-1)
            actual = torch.stack(actual_values, dim=-1)
            finite = bool(torch.isfinite(expected).all() and torch.isfinite(actual).all())
            formula_matches = finite and torch.allclose(actual, expected, rtol=1e-5, atol=1e-6)
            report[label] = {
                "valid_indices": output[mask_key].nonzero().flatten().tolist(),
                "dimensions": dimensions,
                "normalization": {
                    "mode": captured["mode"],
                    "reference_transform": "NormalizeTransformFn",
                    "mean_std_loaded": True,
                    "formula_matches": formula_matches,
                    "verified": formula_matches,
                    "nonidentity_parameter_dimensions": int(((mean != 0) | (std != 1)).sum()),
                    "zero_std_indices": (std == 0).nonzero().flatten().tolist(),
                    "max_abs_error": float((actual - expected).abs().max()) if finite else None,
                    "all_finite": finite,
                    "checked_values": actual.numel(),
                    "max_abs_normalized_value": float(actual.abs().max()) if finite else None,
                },
            }
        output[INSPECTION_KEY] = report
        return output


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
    for index, transform in enumerate(transforms):
        if isinstance(transform, NormalizeTransformFn):
            transforms[index] = InspectWSA15NormalizationFn(mode=transform.mode)
        elif isinstance(transform, CanonicalStateActionTransformFn):
            transforms[index] = InspectWSA15StateActionFn(normalization_mode=transform.normalization_mode)
    return wrap_wsa15_dataset(base, settings, transforms), metadata


def _tensor_description(value: torch.Tensor) -> dict[str, Any]:
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype).removeprefix("torch."),
    }


def summarize_sample(
    sample: dict[str, Any], robot_type: str, stats_file: Path
) -> dict[str, Any]:
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
        "state_action_inspection": {
            "stats_file": str(stats_file.resolve()),
            **sample[INSPECTION_KEY],
        },
    }
    if image_key is not None:
        summary["image"] = {
            "key": image_key,
            **_tensor_description(sample[image_key]),
        }
    if "synthetic_embodiment" in sample:
        summary["human_action_representation"] = sample["synthetic_embodiment"]
        summary["eligible_human_embodiments"] = sample["wsa.inspect.candidate_embodiments"]
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True, help="Parent directory of datasets")
    parser.add_argument(
        "--repo-id",
        action="append",
        dest="repo_ids",
        required=True,
        help="Dataset path relative to --dataset-root; repeat for multiple datasets",
    )
    parser.add_argument("--human-stats-dir", type=Path)
    parser.add_argument("--human-action-file", default=DEFAULT_HUMAN_ACTION_FILE)
    parser.add_argument(
        "--human-embodiment",
        action="append",
        dest="human_embodiments",
        help="Selected representation; repeat to select several (default: human_eef)",
    )
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument("--num-future-frames", type=int, default=4)
    parser.add_argument("--action-mode", choices=("abs", "delta"), default="abs")
    parser.add_argument("--video-backend", default="pyav")
    parser.add_argument("--samples-per-dataset", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.human_embodiments is None:
        args.human_embodiments = [HUMAN_EEF_EMBODIMENT]
    return args


def main() -> None:
    args = parse_args()
    collection_root = args.dataset_root.resolve()
    if args.samples_per_dataset < 1:
        raise ValueError("--samples-per-dataset must be positive")

    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    for repo_id in args.repo_ids:
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
        samples = []
        for _ in range(args.samples_per_dataset):
            sample_index = rng.randrange(len(dataset))
            sample = dataset[sample_index]
            if is_wsa15_human_robot_type(metadata.robot_type):
                sample["wsa.inspect.candidate_embodiments"] = dataset.candidate_embodiments(sample_index)
            stats_file = (
                args.human_stats_dir / sample["synthetic_embodiment"] / "stats.json"
                if is_wsa15_human_robot_type(metadata.robot_type)
                else dataset.root / "meta" / "stats.json"
            )
            samples.append(summarize_sample(sample, metadata.robot_type, stats_file))
        report = {
            "repo_id": repo_id,
            "action_mode": args.action_mode,
            "robot_type": metadata.robot_type,
            "source_kind": wsa15_source_kind(metadata.robot_type),
            "selected_human_embodiments": args.human_embodiments if is_wsa15_human_robot_type(metadata.robot_type) else [],
            "eligible_samples": len(dataset),
            "samples": samples,
        }
        print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
