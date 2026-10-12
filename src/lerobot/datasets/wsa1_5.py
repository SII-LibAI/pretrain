"""WSA1.5 unified data-flow transforms.

Standard LeRobot state/action fields are mapped into fixed semantic slots.  A
dataset may additionally provide subtask and motion annotations; missing
annotations produce explicit validity masks instead of making it unloadable.
"""

from __future__ import annotations

import bisect
import json
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor

from lerobot.dataset_schemas import get_schema
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.streaming_dataset import StreamingLeRobotDataset
from lerobot.transforms.core import (
    ComposeFieldsTransform,
    DataDict,
    DataTransformFn,
    DeltaActionTransformFn,
    LoadActionTextFromJsonlTransformFn,
    RemapImageKeyTransformFn,
    ResizeImagesWithPadFn,
)
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE, OBS_STR

MOTION_MASK_KEYS = ("observation.motion_mask", "motion_mask")
MOTION_MASK_ID_KEYS = ("observation.motion_mask_ids", "motion_mask_ids")
WSA15_DATAFLOW_NAME = "wsa1_5"
WSA15_SOURCE_ROBOT = "robot"
WSA15_SOURCE_HUMAN = "human"
DEFAULT_HUMAN_ACTION_FILE = "meta/annotations/action.parquet"
WSA15_VIDEO_STORAGE_ALIASES = {
    "observation.motion_mask": "motion_mask",
    "observation.motion_mask_ids": "motion_mask_ids",
}
FUTURE_2D_TARGET_KEY = "wsa.future_motion_mask"
FUTURE_2D_VALID_KEY = "wsa.future_motion_mask_valid"
FUTURE_3D_TARGET_KEY = "wsa.future_motion_mask_ids"
FUTURE_3D_VALID_KEY = "wsa.future_motion_mask_ids_valid"
PHYSICAL_TARGET_VALID_KEY = "physical_target_valid"
SUBTASK_TEXT_KEY = "wsa.subtask_text"
SUBTASK_VALID_KEY = "wsa.subtask_valid"
SUBTASK_TARGET_KEY = "wsa.subtask_token_ids"
SUBTASK_TARGET_MASK_KEY = "wsa.subtask_token_mask"


def is_wsa15_human_robot_type(robot_type: str) -> bool:
    """Human sources are identified solely by the source metadata robot_type."""

    return "ego" in robot_type.lower()


def wsa15_source_kind(robot_type: str) -> str:
    """Classify one LeRobotV3 source using its declared ``robot_type``."""

    return WSA15_SOURCE_HUMAN if is_wsa15_human_robot_type(robot_type) else WSA15_SOURCE_ROBOT


def wsa15_future_indices(chunk_size: int, num_future_frames: int) -> list[int]:
    """Return uniformly spaced future-frame offsets inside an action horizon."""

    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    if num_future_frames < 1:
        raise ValueError("num_future_frames must be positive")
    return [chunk_size * index // num_future_frames for index in range(1, num_future_frames + 1)]


def resolve_wsa15_video_key_mapping(dataset_metadata: Any) -> dict[str, str]:
    """Resolve the two documented supplemental-video storage aliases.

    EgoDex supplements expose logical feature names such as
    ``observation.motion_mask`` in ``info.json`` while their episode columns
    and video directories use ``motion_mask``. Standard LeRobot videos use the
    same key in both locations. Both layouts are explicit and supported; any
    third layout is a malformed dataset rather than a path to guess.
    """

    if not dataset_metadata.episodes:
        return {}
    episode = dataset_metadata.episodes[0]
    mapping: dict[str, str] = {}
    for logical_key, storage_key in WSA15_VIDEO_STORAGE_ALIASES.items():
        feature = dataset_metadata.features.get(logical_key)
        if feature is None or feature.get("dtype") != "video":
            continue
        logical_column = f"videos/{logical_key}/chunk_index"
        storage_column = f"videos/{storage_key}/chunk_index"
        if logical_column in episode:
            continue
        if storage_column not in episode:
            raise KeyError(
                f"{logical_key!r} is declared as video but episode metadata contains neither "
                f"{logical_column!r} nor {storage_column!r}"
            )
        mapping[logical_key] = storage_key
    return mapping


def resolve_wsa15_delta_timestamps(
    dataset_config: Any,
    dataset_metadata: Any,
    *,
    human_source: bool,
) -> dict[str, list[float]] | None:
    """Build the WSA1.5 query plan without depending on a policy config.

    Robot sources read the future action chunk from the standard LeRobotV3
    parquet. Human sources deliberately do not query that action column because
    :class:`SyntheticEmbodimentDataset` replaces it with ``action.parquet``.
    RGB inputs are current-frame observations. Optional motion videos are read
    only at the uniformly spaced future offsets used as supervision.
    """

    schema = get_schema(dataset_metadata.robot_type)
    features = dataset_metadata.features
    fps = dataset_metadata.fps
    chunk_size = int(dataset_config.chunk_size)
    num_future_frames = int(dataset_config.num_video_frames)
    action_indices = list(range(chunk_size))
    future_indices = wsa15_future_indices(chunk_size, num_future_frames)
    timestamps: dict[str, list[float]] = {}

    if not human_source:
        for key in schema.get_action_keys():
            if key in features:
                timestamps[key] = [index / fps for index in action_indices]

    for key in schema.image_mapping:
        if key in features:
            timestamps[key] = [0.0]

    for key in (*MOTION_MASK_KEYS, *MOTION_MASK_ID_KEYS):
        feature = features.get(key)
        if feature is not None and feature.get("dtype") == "video":
            timestamps[key] = [index / fps for index in future_indices]

    return timestamps or None

# Compact semantic state/action representation derived from Qwen-RobotManip's
# fixed-slot principle but adapted to this repository's embodiments.  The flat
# vector is [left 16 | right 16], and each side is
# [joint 7 | EEF xyz 3 | EEF quaternion xyzw 4 | gripper 2].
CANONICAL_DIM = 32
ARM_DIM = 16
JOINT_DIM = 7
EEF_POSITION_START = 7
EEF_QUATERNION_START = 10
GRIPPER_START = 14
GRIPPER_DIM = 2
STATE_VALID_MASK_KEY = "wsa.state_valid_mask"
ACTION_VALID_MASK_KEY = "wsa.action_valid_mask"
ACTION_DECODE_MATRIX_KEY = "wsa.action_decode_matrix"
ACTION_DECODE_BIAS_KEY = "wsa.action_decode_bias"
ACTION_RAW_DIM_KEY = "wsa.action_raw_dim"


def build_wsa15_data_transforms(
    *,
    action_mode: str = "abs",
    num_future_frames: int = 4,
    height: int = 224,
    width: int = 224,
) -> list[DataTransformFn]:
    """The shared, ordered preprocessing pipeline before tokenization.

    Training and the data inspector use this same list. ``hydrate(dataset)``
    binds schema mappings and statistics once; ``__call__(sample)`` applies
    those mappings to each sampled frame.
    """

    if action_mode not in {"abs", "delta"}:
        raise ValueError("action_mode must be 'abs' or 'delta'")
    transforms: list[DataTransformFn] = []
    if action_mode == "delta":
        transforms.append(DeltaActionTransformFn())
    return [
        *transforms,
        PrepareWSA15MotionTargetsFn(num_future_frames=num_future_frames, height=height, width=width),
        ResizeImagesWithPadFn(height=height, width=width),
        RemapImageKeyTransformFn(),
        SelectWSA15CurrentObservationFn(),
        ComposeFieldsTransform(),
        CanonicalStateActionTransformFn(),
        LoadActionTextFromJsonlTransformFn(),
        LoadWSA15PhysicalAnnotationsFn(),
    ]


def wrap_wsa15_dataset(
    base_dataset: LeRobotDataset,
    dataset_config: Any,
    transforms: Sequence[DataTransformFn],
):
    """Select one storage adapter for training and data inspection.

    Source classification always uses the base LeRobot metadata, before any
    synthetic embodiment schema replaces ``robot_type`` during hydration.
    """

    from lerobot.datasets.synthetic_embodiment import SyntheticEmbodimentDataset
    from lerobot.datasets.transformed_dataset import TransformedLeRobotDataset

    if not is_wsa15_human_robot_type(base_dataset.meta.robot_type):
        return TransformedLeRobotDataset.from_base(base_dataset, transforms)
    if dataset_config.human_stats_dir is None:
        raise ValueError(f"Human dataset {base_dataset.repo_id!r} requires human_stats_dir")
    return SyntheticEmbodimentDataset(
        base_dataset,
        annotation_file=dataset_config.human_action_file,
        stats_dir=dataset_config.human_stats_dir,
        chunk_size=dataset_config.chunk_size,
        max_state_dim=CANONICAL_DIM,
        max_action_dim=CANONICAL_DIM,
        embodiments=dataset_config.human_embodiments or None,
        row_group_cache_size=dataset_config.human_annotation_row_group_cache_size,
        transforms=transforms,
        require_wsa15_layout=True,
    )


def validate_wsa15_action_layout(action_layout: list[dict[str, Any]]) -> None:
    """Fail early when an embodiment cannot fit the compact WSA1.5 slots."""

    occupied: set[int] = set()
    for block in action_layout:
        name = str(block["name"])
        slots = CanonicalStateActionTransformFn._slot_indices(name, int(block["size"]))
        overlap = occupied.intersection(slots)
        if overlap:
            raise ValueError(f"Semantic action block {name!r} overlaps canonical slots {sorted(overlap)}")
        occupied.update(slots)


def _side_base(side: str) -> int:
    return 0 if side == "left" else ARM_DIM


def _layout_side(name: str) -> str | None:
    lowered = name.lower()
    if lowered.startswith("left") or ".left" in lowered:
        return "left"
    if lowered.startswith("right") or ".right" in lowered:
        return "right"
    return None


def _normalization_affine(
    stats: dict[str, Any] | None, mode: str, dim: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return raw = normalized * scale + bias for one source vector."""

    if not stats:
        return torch.ones(dim), torch.zeros(dim)
    if mode == "mean_std" and "mean" in stats and "std" in stats:
        return torch.as_tensor(stats["std"]).float().clamp_min(1e-6), torch.as_tensor(stats["mean"]).float()
    key_min, key_max = ("q01", "q99") if mode == "q01_q99" else ("min", "max")
    if key_min in stats and key_max in stats:
        minimum = torch.as_tensor(stats[key_min]).float()
        maximum = torch.as_tensor(stats[key_max]).float()
        return ((maximum - minimum) * 0.5).clamp_min(1e-6), (maximum + minimum) * 0.5
    return torch.ones(dim), torch.zeros(dim)


def _compose_stats(all_stats: dict[str, dict[str, Any]], source_keys: list[str]) -> dict[str, Tensor] | None:
    """Concatenate source-feature statistics in the same order as ComposeFields."""

    if not source_keys or any(key not in all_stats for key in source_keys):
        return None
    result: dict[str, Tensor] = {}
    for statistic in ("mean", "std", "min", "max", "q01", "q99"):
        values = [all_stats[key].get(statistic) for key in source_keys]
        if all(value is not None for value in values):
            result[statistic] = torch.cat([torch.as_tensor(value).flatten() for value in values], dim=0)
    return result or None


def _reorder_matrix(raw_dim: int, spec: list[list[int]] | None) -> torch.Tensor:
    """Map raw vectors to a compact semantic vector, removing legacy gap slots."""

    if not spec:
        return torch.eye(raw_dim)
    target_dim = max(int(item[3]) for item in spec)
    expanded = torch.zeros(target_dim, raw_dim)
    covered = torch.zeros(target_dim, dtype=torch.bool)
    for src_start, src_end, dst_start, dst_end in spec:
        if src_end - src_start != dst_end - dst_start:
            raise ValueError(f"Invalid reorder entry: {[src_start, src_end, dst_start, dst_end]}")
        expanded[dst_start:dst_end, src_start:src_end] = torch.eye(src_end - src_start)
        covered[dst_start:dst_end] = True
    return expanded[covered]


@DataTransformFn.register_subclass("wsa1_5_canonical_state_action")
@dataclass
class CanonicalStateActionTransformFn(DataTransformFn):
    """Encode heterogeneous embodiments into Qwen-RobotManip semantic slots.

    Unlike ``PadStateAndActionTransformFn``, zeros here mean *structurally
    absent* fields and are paired with binary masks.  They are never treated as
    action targets.  Absolute-versus-delta conversion is deliberately outside
    this codec and is controlled only by ``DeltaActionTransformFn`` plus the
    dataset schema's ``action_mask_spec``.  This transform only normalizes,
    reorders, fills semantic slots, and emits the inverse affine mapping.
    """

    normalization_mode: str = "mean_std"
    action_layout: list[dict[str, Any]] = field(default_factory=list)
    action_reorder: list[list[int]] | None = None
    state_reorder: list[list[int]] | None = None
    state_stats: dict[str, Any] | None = None
    action_stats: dict[str, Any] | None = None
    robot_type: str = ""

    def hydrate(self, dataset: LeRobotDataset | StreamingLeRobotDataset) -> CanonicalStateActionTransformFn:
        schema = get_schema(dataset.meta.robot_type)
        stats = getattr(dataset.meta, "stats", {})
        return replace(
            self,
            action_layout=list(schema.action_layout),
            action_reorder=schema.action_reorder,
            state_reorder=schema.state_reorder,
            state_stats=_compose_stats(stats, schema.feature_mapping.get(OBS_STATE, [OBS_STATE])),
            action_stats=_compose_stats(stats, schema.feature_mapping.get(ACTION, [ACTION])),
            robot_type=schema.robot_type,
        )

    @staticmethod
    def _slot_indices(name: str, size: int) -> list[int]:
        side = _layout_side(name)
        if side is None:
            raise ValueError(f"Semantic action block {name!r} must declare a left/right side")

        base = _side_base(side)
        lowered = name.lower()
        if "eef_position" in lowered or "ee_pos" in lowered:
            if size != 3:
                raise ValueError(f"EEF position block {name!r} must have 3 values")
            return list(range(base + EEF_POSITION_START, base + EEF_QUATERNION_START))
        if "quaternion" in lowered:
            if size != 4:
                raise ValueError(f"EEF quaternion block {name!r} must use xyzw with 4 values")
            return list(range(base + EEF_QUATERNION_START, base + GRIPPER_START))
        if "eef_rotation" in lowered or "ee_rot" in lowered:
            raise ValueError(f"EEF rotation block {name!r} is unsupported: WSA1.5 requires quaternion_xyzw")
        if "gripper" in lowered or "grasp" in lowered:
            if size > GRIPPER_DIM:
                raise ValueError(f"Gripper block {name!r} has {size} values; canonical capacity is 2")
            return list(range(base + GRIPPER_START, base + GRIPPER_START + size))
        if "hand" in lowered:
            raise ValueError(f"Dexterous-hand block {name!r} is not part of the compact WSA1.5 action space")
        if "arm" in lowered or "joint" in lowered:
            if size > JOINT_DIM:
                raise ValueError(f"Arm joint block {name!r} has {size} values; canonical capacity is 7")
            return list(range(base, base + size))
        raise ValueError(f"Unknown semantic action block {name!r}; add an explicit WSA1.5 mapping")

    def _logical_layout(self, raw_dim: int, reorder: list[list[int]] | None) -> tuple[torch.Tensor, int]:
        matrix = _reorder_matrix(raw_dim, reorder)
        declared = sum(int(block["size"]) for block in self.action_layout)
        if declared and matrix.shape[0] != declared:
            raise ValueError(
                f"{self.robot_type}: action_layout declares {declared} values but reorder yields "
                f"{matrix.shape[0]}"
            )
        return matrix, matrix.shape[0]

    def __call__(self, data: DataDict) -> DataDict:
        if not self.action_layout:
            raise ValueError(
                f"{self.robot_type or 'dataset'} has no action_layout. WSA1.5 requires semantic blocks; "
                "it deliberately refuses dimension-only padding."
            )
        state_raw = data[OBS_STATE].float()
        action_raw = data[ACTION].float()
        if state_raw.ndim != 1 or action_raw.ndim not in (1, 2):
            raise ValueError(
                f"Canonical transform expects state [D] and action [D] or [T,D], got "
                f"{tuple(state_raw.shape)} and {tuple(action_raw.shape)}"
            )
        action_raw = action_raw.unsqueeze(0) if action_raw.ndim == 1 else action_raw

        state_map, _ = self._logical_layout(state_raw.shape[-1], self.state_reorder)
        action_map, action_logical_dim = self._logical_layout(action_raw.shape[-1], self.action_reorder)
        state_map = state_map.to(state_raw)
        action_map = action_map.to(action_raw)
        state_logical = state_map @ state_raw
        action_logical = action_raw @ action_map.transpose(0, 1)

        state_scale_raw, state_bias_raw = _normalization_affine(
            self.state_stats, self.normalization_mode, state_raw.shape[-1]
        )
        action_scale_raw, action_bias_raw = _normalization_affine(
            self.action_stats, self.normalization_mode, action_raw.shape[-1]
        )
        state_scale = (state_map.abs() @ state_scale_raw.to(state_raw)).clamp_min(1e-6)
        state_bias = state_map @ state_bias_raw.to(state_raw)
        action_scale = (action_map.abs() @ action_scale_raw.to(action_raw)).clamp_min(1e-6)
        action_bias = action_map @ action_bias_raw.to(action_raw)
        state_normalized = (state_logical - state_bias) / state_scale
        action_normalized = (action_logical - action_bias) / action_scale

        state = state_raw.new_zeros(CANONICAL_DIM)
        actions = action_raw.new_zeros(action_raw.shape[0], CANONICAL_DIM)
        state_valid = torch.zeros(CANONICAL_DIM, dtype=torch.bool, device=state_raw.device)
        action_valid = torch.zeros(CANONICAL_DIM, dtype=torch.bool, device=action_raw.device)
        decode_matrix = action_raw.new_zeros(CANONICAL_DIM, CANONICAL_DIM)
        decode_bias = action_raw.new_zeros(CANONICAL_DIM)

        offset = 0
        for block in self.action_layout:
            name = str(block["name"])
            size = int(block["size"])
            logical_slice = slice(offset, offset + size)
            slots = self._slot_indices(name, size)
            if slots:
                if len(slots) != size:
                    raise RuntimeError(f"Internal canonical slot mismatch for {name!r}")
                slot_tensor = torch.tensor(slots, device=state_raw.device)
                state[slot_tensor] = state_normalized[logical_slice]
                actions[:, slot_tensor] = action_normalized[:, logical_slice]
                state_valid[slot_tensor] = True
                action_valid[slot_tensor] = True
                for logical_index, canonical_index in zip(range(offset, offset + size), slots, strict=True):
                    decode_matrix[logical_index, canonical_index] = action_scale[logical_index]
                    decode_bias[logical_index] = action_bias[logical_index]
            offset += size

        # Convert logical-order decode rows back to the repository's raw order.
        logical_to_raw = action_map.transpose(0, 1)
        full_logical_to_raw = action_raw.new_zeros(CANONICAL_DIM, CANONICAL_DIM)
        full_logical_to_raw[: action_raw.shape[-1], :action_logical_dim] = logical_to_raw
        decode_matrix = full_logical_to_raw @ decode_matrix
        decode_bias = full_logical_to_raw @ decode_bias

        data[OBS_STATE] = state
        data[ACTION] = actions
        data[STATE_VALID_MASK_KEY] = state_valid
        data[ACTION_VALID_MASK_KEY] = action_valid
        data[ACTION_DECODE_MATRIX_KEY] = decode_matrix
        data[ACTION_DECODE_BIAS_KEY] = decode_bias
        data[ACTION_RAW_DIM_KEY] = torch.tensor(action_raw.shape[-1], dtype=torch.long)
        return data


def decode_canonical_actions(
    canonical: Tensor,
    metadata: DataDict,
    fallback_output_dim: int = CANONICAL_DIM,
) -> Tensor:
    """Decode predicted canonical actions to the dataset's original interface."""

    matrix = metadata.get(ACTION_DECODE_MATRIX_KEY)
    if matrix is None:
        return canonical[..., :fallback_output_dim]
    if matrix.ndim == 2:
        matrix = matrix.unsqueeze(0).expand(canonical.shape[0], -1, -1)
    matrix = matrix.to(canonical)
    bias = metadata[ACTION_DECODE_BIAS_KEY]
    if bias.ndim == 1:
        bias = bias.unsqueeze(0).expand(canonical.shape[0], -1)
    bias = bias.to(canonical)
    decoded = torch.einsum("bij,btj->bti", matrix, canonical) + bias[:, None]

    raw_dims = metadata.get(ACTION_RAW_DIM_KEY)
    output_dim = (
        int(raw_dims.flatten()[0].item())
        if raw_dims is not None and int(raw_dims.flatten()[0].item()) > 0
        else fallback_output_dim
    )
    return decoded[..., :output_dim]


def _as_int(value: Any) -> int:
    return int(value.item()) if isinstance(value, torch.Tensor) else int(value)


@DataTransformFn.register_subclass("wsa1_5_physical_annotations")
@dataclass
class LoadWSA15PhysicalAnnotationsFn(DataTransformFn):
    """Attach the sidecar's discrete subtask target when available.

    WSA1.5 intentionally leaves episodic memory unspecified.  Pose annotations
    are therefore not serialized into language and cannot leak into context.
    """

    sidecar_file: str = ""
    output_key: str = "sub_task"
    _segments: dict[int, tuple[list[int], list[tuple[int, int, str]]]] = field(
        default_factory=dict, init=False, repr=False
    )
    _instructions: dict[int, str] = field(default_factory=dict, init=False, repr=False)

    def hydrate(self, dataset: LeRobotDataset | StreamingLeRobotDataset) -> LoadWSA15PhysicalAnnotationsFn:
        root = Path(str(getattr(dataset, "root", "")))
        sidecar = Path(self.sidecar_file) if self.sidecar_file else root / "meta" / "sidecar.json"
        hydrated = replace(self, sidecar_file=str(sidecar))
        hydrated._load_sidecar()
        return hydrated

    def _load_sidecar(self) -> None:
        path = Path(self.sidecar_file)
        if not path.is_file():
            return
        payload = json.loads(path.read_text(encoding="utf-8"))

        episodes = payload if isinstance(payload, list) else payload.get("episodes", [])
        for episode in episodes:
            episode_index = int(episode["episode_index"])
            instruction = str(episode.get("task_instruction", "")).strip()
            if instruction:
                self._instructions[episode_index] = instruction
            segments = sorted(
                (
                    int(segment["start_frame"]),
                    int(segment["end_frame"]),
                    str(segment.get("subtask_text", "")).strip(),
                )
                for segment in episode.get("subtasks", [])
            )
            if segments:
                self._segments[episode_index] = ([segment[0] for segment in segments], segments)

    def _subtask(self, episode_index: int, frame_index: int) -> str:
        cached = self._segments.get(episode_index)
        if cached is None:
            return ""
        starts, segments = cached
        position = bisect.bisect_right(starts, frame_index) - 1
        if position < 0:
            return ""
        start, end, text = segments[position]
        return text if start <= frame_index <= end else ""

    def __call__(self, data: DataDict) -> DataDict:
        if "episode_index" not in data or "frame_index" not in data:
            data[PHYSICAL_TARGET_VALID_KEY] = torch.tensor(False)
            data[SUBTASK_TEXT_KEY] = ""
            data[SUBTASK_VALID_KEY] = torch.tensor(False)
            return data

        episode_index = _as_int(data["episode_index"])
        frame_index = _as_int(data["frame_index"])
        subtask = self._subtask(episode_index, frame_index)

        existing = str(data.get(self.output_key, "")).strip()
        target = subtask or existing
        if target:
            data[self.output_key] = target
        instruction = self._instructions.get(episode_index)
        if instruction:
            data["task"] = instruction
        valid = torch.tensor(bool(target), dtype=torch.bool)
        data[PHYSICAL_TARGET_VALID_KEY] = valid
        data[SUBTASK_TEXT_KEY] = target
        data[SUBTASK_VALID_KEY] = valid
        return data


def _find_tensor(data: DataDict, keys: tuple[str, ...]) -> torch.Tensor | None:
    for key in keys:
        value = data.get(key)
        if isinstance(value, torch.Tensor):
            return value
    return None


@DataTransformFn.register_subclass("wsa1_5_select_current_observation")
@dataclass
class SelectWSA15CurrentObservationFn(DataTransformFn):
    """Keep current RGB for Qwen without discarding future motion targets."""

    num_views: int = 3

    def __call__(self, data: DataDict) -> DataDict:
        for index in range(self.num_views):
            key = f"{OBS_IMAGES}.image{index}"
            value = data.get(key)
            if isinstance(value, torch.Tensor) and value.ndim == 4:
                data[key] = value[0]
        return data


def _as_video(value: torch.Tensor) -> torch.Tensor:
    if value.ndim == 3:
        value = value.unsqueeze(0)
    if value.ndim != 4:
        raise ValueError(f"Motion target must be [C,H,W] or [T,C,H,W], got {tuple(value.shape)}")
    value = value.float()
    if value.numel() and value.max() > 1.0:
        value = value / 255.0
    return value.clamp(0.0, 1.0)


@DataTransformFn.register_subclass("wsa1_5_motion_targets")
@dataclass
class PrepareWSA15MotionTargetsFn(DataTransformFn):
    """Prepare independent future 2D-motion and 3D-ID image targets."""

    num_future_frames: int = 4
    height: int = 224
    width: int = 224

    def _prepare(self, value: torch.Tensor, mode: str) -> torch.Tensor:
        video = _as_video(value)
        if video.shape[0] != self.num_future_frames:
            raise ValueError(
                "WSA1.5 future video query returned "
                f"{video.shape[0]} frames, expected {self.num_future_frames}"
            )
        kwargs = {"align_corners": False} if mode != "nearest" else {}
        video = F.interpolate(video, size=(self.height, self.width), mode=mode, **kwargs)
        return video.mul(2.0).sub(1.0)

    def __call__(self, data: DataDict) -> DataDict:
        motion = _find_tensor(data, MOTION_MASK_KEYS)
        ids = _find_tensor(data, MOTION_MASK_ID_KEYS)

        def empty() -> torch.Tensor:
            return -torch.ones(self.num_future_frames, 3, self.height, self.width)

        data[FUTURE_2D_TARGET_KEY] = (
            self._prepare(motion, "bilinear") if motion is not None else empty()
        )
        data[FUTURE_2D_VALID_KEY] = torch.tensor(motion is not None, dtype=torch.bool)
        data[FUTURE_3D_TARGET_KEY] = (
            self._prepare(ids, "nearest") if ids is not None else empty()
        )
        data[FUTURE_3D_VALID_KEY] = torch.tensor(ids is not None, dtype=torch.bool)
        return data


@DataTransformFn.register_subclass("unify_wsa1_5_inputs")
@dataclass
class UnifyWSA15InputsFn(DataTransformFn):
    """Return a stable batch schema across fully and sparsely annotated data."""

    num_future_frames: int = 4
    height: int = 224
    width: int = 224
    num_subtask_tokens: int = 16

    def __call__(self, data: DataDict) -> DataDict:
        input_ids = data[f"{OBS_STR}.input_ids"]
        output = {
            OBS_STATE: data[OBS_STATE],
            ACTION: data[ACTION],
            f"{OBS_STR}.pixel_values": data[f"{OBS_STR}.pixel_values"],
            f"{OBS_STR}.image_grid_thw": data[f"{OBS_STR}.image_grid_thw"],
            f"{OBS_STR}.input_ids": input_ids,
            f"{OBS_STR}.attention_mask": data[f"{OBS_STR}.attention_mask"],
            f"{OBS_STR}.fast_token_mask": data.get(
                f"{OBS_STR}.fast_token_mask", torch.zeros_like(input_ids, dtype=torch.bool)
            ),
            "vqa_type": data.get("vqa_type", torch.tensor(0, dtype=torch.long)),
            "VQA.labels": data.get("VQA.labels", torch.full_like(input_ids, -100)),
            "label_mode": data.get("label_mode", torch.tensor(0, dtype=torch.long)),
            FUTURE_2D_TARGET_KEY: data[FUTURE_2D_TARGET_KEY],
            FUTURE_2D_VALID_KEY: data[FUTURE_2D_VALID_KEY],
            FUTURE_3D_TARGET_KEY: data[FUTURE_3D_TARGET_KEY],
            FUTURE_3D_VALID_KEY: data[FUTURE_3D_VALID_KEY],
            PHYSICAL_TARGET_VALID_KEY: data.get(PHYSICAL_TARGET_VALID_KEY, torch.tensor(False)),
            SUBTASK_TARGET_KEY: data.get(
                SUBTASK_TARGET_KEY, torch.full((self.num_subtask_tokens,), -100, dtype=torch.long)
            ),
            SUBTASK_TARGET_MASK_KEY: data.get(
                SUBTASK_TARGET_MASK_KEY, torch.zeros(self.num_subtask_tokens, dtype=torch.bool)
            ),
        }
        for key in (
            STATE_VALID_MASK_KEY,
            ACTION_VALID_MASK_KEY,
            ACTION_DECODE_MATRIX_KEY,
            ACTION_DECODE_BIAS_KEY,
            ACTION_RAW_DIM_KEY,
        ):
            output[key] = data[key]
        return output


@DataTransformFn.register_subclass("unify_wsa1_5_vqa_inputs")
@dataclass
class UnifyWSA15VQAInputsFn(DataTransformFn):
    """Match external VQA batches to the WSA1.5 robot batch schema."""

    num_future_frames: int = 4
    height: int = 224
    width: int = 224
    num_subtask_tokens: int = 16

    def __call__(self, data: DataDict) -> DataDict:
        input_ids = data[f"{OBS_STR}.input_ids"]
        return {
            OBS_STATE: torch.zeros(CANONICAL_DIM),
            ACTION: torch.zeros_like(data[ACTION]).new_zeros(data[ACTION].shape[:-1] + (CANONICAL_DIM,)),
            f"{OBS_STR}.pixel_values": data[f"{OBS_STR}.pixel_values"],
            f"{OBS_STR}.image_grid_thw": data[f"{OBS_STR}.image_grid_thw"],
            f"{OBS_STR}.input_ids": input_ids,
            f"{OBS_STR}.attention_mask": data[f"{OBS_STR}.attention_mask"],
            f"{OBS_STR}.fast_token_mask": data.get(
                f"{OBS_STR}.fast_token_mask", torch.zeros_like(input_ids, dtype=torch.bool)
            ),
            "vqa_type": torch.tensor(1, dtype=torch.long),
            "VQA.labels": data["VQA.labels"],
            "label_mode": data.get("label_mode", torch.tensor(1, dtype=torch.long)),
            FUTURE_2D_TARGET_KEY: -torch.ones(
                self.num_future_frames, 3, self.height, self.width
            ),
            FUTURE_2D_VALID_KEY: torch.tensor(False),
            FUTURE_3D_TARGET_KEY: -torch.ones(
                self.num_future_frames, 3, self.height, self.width
            ),
            FUTURE_3D_VALID_KEY: torch.tensor(False),
            PHYSICAL_TARGET_VALID_KEY: torch.tensor(False),
            SUBTASK_TARGET_KEY: torch.full((self.num_subtask_tokens,), -100, dtype=torch.long),
            SUBTASK_TARGET_MASK_KEY: torch.zeros(self.num_subtask_tokens, dtype=torch.bool),
            STATE_VALID_MASK_KEY: torch.zeros(CANONICAL_DIM, dtype=torch.bool),
            ACTION_VALID_MASK_KEY: torch.zeros(CANONICAL_DIM, dtype=torch.bool),
            ACTION_DECODE_MATRIX_KEY: torch.zeros(CANONICAL_DIM, CANONICAL_DIM),
            ACTION_DECODE_BIAS_KEY: torch.zeros(CANONICAL_DIM),
            ACTION_RAW_DIM_KEY: torch.tensor(0, dtype=torch.long),
        }
