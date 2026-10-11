"""Optional WSA1.5 supervision for augmented LeRobot v3 datasets.

Every dataset remains usable as an ordinary LeRobot dataset.  When subtask or
motion-mask annotations are present this module exposes them as WSA1.5
training targets; missing annotations produce explicit validity masks instead
of making the sample unloadable.  Episodic memory is intentionally left as a
future extension point.
"""

from __future__ import annotations

import bisect
import json
import logging
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor

from lerobot.dataset_schemas import get_schema
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.streaming_dataset import StreamingLeRobotDataset
from lerobot.transforms.core import DataDict, DataTransformFn
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE, OBS_STR

MOTION_MASK_KEYS = ("observation.motion_mask", "motion_mask")
MOTION_MASK_ID_KEYS = ("observation.motion_mask_ids", "motion_mask_ids")
WORLD_TARGET_KEY = "observation.video_frames"
WORLD_TARGET_VALID_KEY = "world_target_valid"
PHYSICAL_TARGET_VALID_KEY = "physical_target_valid"
SUBTASK_TARGET_KEY = "wsa.subtask_token_ids"
SUBTASK_TARGET_MASK_KEY = "wsa.subtask_token_mask"

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
ACTION_DECODE_EEF_KEY = "wsa.action_decode_eef"
ACTION_DECODE_CURRENT_KEY = "wsa.action_decode_current"
ACTION_DECODE_SCALE_KEY = "wsa.action_decode_scale"
ACTION_DECODE_REORDER_KEY = "wsa.action_decode_reorder"
ACTION_DECODE_EEF_INDEX_KEY = "wsa.action_decode_eef_index"
ACTION_DECODE_CAMERA_KEY = "wsa.action_decode_camera_rotation"
ACTION_RAW_DIM_KEY = "wsa.action_raw_dim"
CAMERA_FRAME_VALID_KEY = "wsa.camera_frame_valid"


def _side_base(side: str) -> int:
    return 0 if side == "left" else ARM_DIM


def _quaternion_xyzw_to_matrix(quaternion: torch.Tensor) -> torch.Tensor:
    quaternion = F.normalize(quaternion, dim=-1, eps=1e-8)
    x, y, z, w = quaternion.unbind(-1)
    two = quaternion.new_tensor(2.0)
    return torch.stack(
        (
            1 - two * (y * y + z * z),
            two * (x * y - z * w),
            two * (x * z + y * w),
            two * (x * y + z * w),
            1 - two * (x * x + z * z),
            two * (y * z - x * w),
            two * (x * z - y * w),
            two * (y * z + x * w),
            1 - two * (x * x + y * y),
        ),
        dim=-1,
    ).reshape(*quaternion.shape[:-1], 3, 3)


def _matrix_to_quaternion_xyzw(matrix: torch.Tensor) -> torch.Tensor:
    """Numerically stable matrix-to-quaternion conversion, xyzw convention."""

    m00, m01, m02 = matrix[..., 0, 0], matrix[..., 0, 1], matrix[..., 0, 2]
    m10, m11, m12 = matrix[..., 1, 0], matrix[..., 1, 1], matrix[..., 1, 2]
    m20, m21, m22 = matrix[..., 2, 0], matrix[..., 2, 1], matrix[..., 2, 2]
    candidates = (
        torch.stack(
            (
                1 + m00 - m11 - m22,
                1 - m00 + m11 - m22,
                1 - m00 - m11 + m22,
                1 + m00 + m11 + m22,
            ),
            dim=-1,
        )
        .clamp_min(0)
        .sqrt()
        * 0.5
    )
    index = candidates.argmax(dim=-1)
    eps = torch.finfo(matrix.dtype).eps
    x = torch.where(
        index == 0,
        candidates[..., 0],
        torch.where(
            index == 1,
            (m01 + m10) / (4 * candidates[..., 1] + eps),
            torch.where(
                index == 2,
                (m02 + m20) / (4 * candidates[..., 2] + eps),
                (m21 - m12) / (4 * candidates[..., 3] + eps),
            ),
        ),
    )
    y = torch.where(
        index == 1,
        candidates[..., 1],
        torch.where(
            index == 0,
            (m01 + m10) / (4 * candidates[..., 0] + eps),
            torch.where(
                index == 2,
                (m12 + m21) / (4 * candidates[..., 2] + eps),
                (m02 - m20) / (4 * candidates[..., 3] + eps),
            ),
        ),
    )
    z = torch.where(
        index == 2,
        candidates[..., 2],
        torch.where(
            index == 0,
            (m02 + m20) / (4 * candidates[..., 0] + eps),
            torch.where(
                index == 1,
                (m12 + m21) / (4 * candidates[..., 1] + eps),
                (m10 - m01) / (4 * candidates[..., 3] + eps),
            ),
        ),
    )
    w = torch.where(
        index == 3,
        candidates[..., 3],
        torch.where(
            index == 0,
            (m21 - m12) / (4 * candidates[..., 0] + eps),
            torch.where(
                index == 1,
                (m02 - m20) / (4 * candidates[..., 1] + eps),
                (m10 - m01) / (4 * candidates[..., 2] + eps),
            ),
        ),
    )
    return _canonicalize_quaternion(torch.stack((x, y, z, w), dim=-1))


def _canonicalize_quaternion(quaternion: torch.Tensor) -> torch.Tensor:
    """Normalize xyzw quaternions and choose the non-negative-w hemisphere."""

    quaternion = F.normalize(quaternion, dim=-1, eps=1e-8)
    sign = torch.where(quaternion[..., 3:4] < 0, -1.0, 1.0)
    return quaternion * sign


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
    action targets.  The transform also emits enough metadata for the policy to
    map canonical actions back to the repository's original action interface.
    """

    normalization_mode: str = "mean_std"
    require_camera_calibration: bool = False
    camera_extrinsics_key: str = "observation.camera_extrinsics"
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

    def _camera_rotation(self, data: DataDict, reference: torch.Tensor) -> tuple[torch.Tensor, bool]:
        extrinsics = data.get(self.camera_extrinsics_key)
        if not isinstance(extrinsics, torch.Tensor):
            if self.require_camera_calibration:
                raise ValueError(
                    f"{self.robot_type}: EEF actions require {self.camera_extrinsics_key!r} for camera-frame deltas"
                )
            return torch.eye(3, dtype=reference.dtype, device=reference.device), False
        if extrinsics.shape[-2:] == (4, 4):
            return extrinsics[..., :3, :3].to(reference), True
        if extrinsics.shape[-2:] == (3, 3):
            return extrinsics.to(reference), True
        raise ValueError(f"Camera extrinsics must end in [3,3] or [4,4], got {tuple(extrinsics.shape)}")

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

        state_map, state_logical_dim = self._logical_layout(state_raw.shape[-1], self.state_reorder)
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
        decode_scale = action_raw.new_ones(2, 3)
        decode_current = action_raw.new_zeros(2, 7)
        decode_eef_index = torch.full((2, 7), -1, dtype=torch.long, device=action_raw.device)

        offset = 0
        eef_blocks: dict[str, dict[str, slice]] = {"left": {}, "right": {}}
        for block in self.action_layout:
            name = str(block["name"])
            size = int(block["size"])
            logical_slice = slice(offset, offset + size)
            side = _layout_side(name)
            lowered = name.lower()
            if side is not None and ("eef_position" in lowered or "ee_pos" in lowered):
                eef_blocks[side]["position"] = logical_slice
            elif side is not None and "quaternion" in lowered:
                eef_blocks[side]["quaternion"] = logical_slice
            elif side is not None and ("gripper" in lowered or "grasp" in lowered):
                eef_blocks[side]["gripper"] = logical_slice

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

        has_eef = any("position" in blocks and "quaternion" in blocks for blocks in eef_blocks.values())
        camera_rotation = torch.eye(3, dtype=state_raw.dtype, device=state_raw.device)
        camera_valid = False
        if has_eef:
            camera_rotation, camera_valid = self._camera_rotation(data, state_raw)
            for side_index, side in enumerate(("left", "right")):
                blocks = eef_blocks[side]
                if "position" not in blocks or "quaternion" not in blocks:
                    continue
                pos_slice, quat_slice = blocks["position"], blocks["quaternion"]
                if quat_slice.stop - quat_slice.start != 4:
                    raise ValueError(f"{self.robot_type}: EEF quaternion must use xyzw with 4 values")
                base = _side_base(side)
                current_pos = state_logical[pos_slice]
                current_quat = _canonicalize_quaternion(state_logical[quat_slice])
                current_rotation = _quaternion_xyzw_to_matrix(current_quat)
                state[base + EEF_POSITION_START : base + EEF_QUATERNION_START] = (
                    current_pos - state_bias[pos_slice]
                ) / state_scale[pos_slice]
                state[base + EEF_QUATERNION_START : base + GRIPPER_START] = current_quat
                state_valid[base + EEF_POSITION_START : base + GRIPPER_START] = True

                desired_pos = action_logical[:, pos_slice]
                desired_rotation = _quaternion_xyzw_to_matrix(action_logical[:, quat_slice])
                delta_pos = desired_pos - current_pos
                delta_rotation = desired_rotation @ current_rotation.transpose(-1, -2)
                camera = camera_rotation
                delta_pos = torch.einsum("...ij,tj->ti", camera, delta_pos)
                delta_rotation = camera @ delta_rotation @ camera.transpose(-1, -2)
                pos_scale = action_scale[pos_slice]
                actions[:, base + EEF_POSITION_START : base + EEF_QUATERNION_START] = delta_pos / pos_scale
                actions[:, base + EEF_QUATERNION_START : base + GRIPPER_START] = _matrix_to_quaternion_xyzw(
                    delta_rotation
                )
                action_valid[base + EEF_POSITION_START : base + GRIPPER_START] = True
                decode_scale[side_index] = pos_scale
                decode_current[side_index, :3] = current_pos
                decode_current[side_index, 3:] = current_quat

        # Convert logical-order decode rows back to the repository's raw order.
        logical_to_raw = action_map.transpose(0, 1)
        full_logical_to_raw = action_raw.new_zeros(CANONICAL_DIM, CANONICAL_DIM)
        full_logical_to_raw[: action_raw.shape[-1], :action_logical_dim] = logical_to_raw
        if has_eef:
            for side_index, side in enumerate(("left", "right")):
                blocks = eef_blocks[side]
                if "position" not in blocks or "quaternion" not in blocks:
                    continue
                logical_indices = list(range(blocks["position"].start, blocks["position"].stop))
                logical_indices += list(range(blocks["quaternion"].start, blocks["quaternion"].stop))
                for index, logical_index in enumerate(logical_indices):
                    raw_matches = logical_to_raw[:, logical_index].nonzero(as_tuple=False).flatten()
                    if raw_matches.numel() != 1:
                        raise ValueError(
                            f"{self.robot_type}: EEF field at logical index {logical_index} is not invertible"
                        )
                    decode_eef_index[side_index, index] = raw_matches.item()
        decode_matrix = full_logical_to_raw @ decode_matrix
        decode_bias = full_logical_to_raw @ decode_bias

        data[OBS_STATE] = state
        data[ACTION] = actions
        data[STATE_VALID_MASK_KEY] = state_valid
        data[ACTION_VALID_MASK_KEY] = action_valid
        data[ACTION_DECODE_MATRIX_KEY] = decode_matrix
        data[ACTION_DECODE_BIAS_KEY] = decode_bias
        data[ACTION_DECODE_EEF_KEY] = torch.tensor(has_eef, dtype=torch.bool)
        data[ACTION_DECODE_CURRENT_KEY] = decode_current
        data[ACTION_DECODE_SCALE_KEY] = decode_scale
        data[ACTION_DECODE_REORDER_KEY] = full_logical_to_raw
        data[ACTION_DECODE_EEF_INDEX_KEY] = decode_eef_index
        data[ACTION_DECODE_CAMERA_KEY] = camera_rotation
        data[ACTION_RAW_DIM_KEY] = torch.tensor(action_raw.shape[-1], dtype=torch.long)
        data[CAMERA_FRAME_VALID_KEY] = torch.tensor(camera_valid, dtype=torch.bool)
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

    eef_rows = metadata[ACTION_DECODE_EEF_KEY]
    if eef_rows.ndim == 0:
        eef_rows = eef_rows[None]
    eef_rows = eef_rows.to(canonical.device).bool()
    if eef_rows.any():
        currents = metadata[ACTION_DECODE_CURRENT_KEY]
        scales = metadata[ACTION_DECODE_SCALE_KEY]
        indices = metadata[ACTION_DECODE_EEF_INDEX_KEY]
        cameras = metadata[ACTION_DECODE_CAMERA_KEY]
        if currents.ndim == 2:
            currents, scales, indices, cameras = (
                currents.unsqueeze(0),
                scales.unsqueeze(0),
                indices.unsqueeze(0),
                cameras.unsqueeze(0),
            )
        currents = currents.to(canonical)
        scales = scales.to(canonical)
        indices = indices.to(canonical.device)
        cameras = cameras.to(canonical)
        for row in eef_rows.nonzero(as_tuple=False).flatten().tolist():
            camera = cameras[row]
            for side_index, base in enumerate((0, ARM_DIM)):
                raw_indices = indices[row, side_index]
                if (raw_indices < 0).any():
                    continue
                delta_camera = (
                    canonical[
                        row,
                        :,
                        base + EEF_POSITION_START : base + EEF_QUATERNION_START,
                    ]
                    * scales[row, side_index]
                )
                delta_base = torch.einsum("ij,tj->ti", camera.transpose(-1, -2), delta_camera)
                desired_pos = currents[row, side_index, :3] + delta_base
                delta_rotation_camera = _quaternion_xyzw_to_matrix(
                    canonical[
                        row,
                        :,
                        base + EEF_QUATERNION_START : base + GRIPPER_START,
                    ]
                )
                delta_rotation_base = camera.transpose(-1, -2) @ delta_rotation_camera @ camera
                current_rotation = _quaternion_xyzw_to_matrix(currents[row, side_index, 3:])
                desired_quaternion = _matrix_to_quaternion_xyzw(delta_rotation_base @ current_rotation)
                decoded[row, :, raw_indices[:3]] = desired_pos
                decoded[row, :, raw_indices[3:]] = desired_quaternion

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
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            logging.warning("Could not read WSA1.5 sidecar %s: %s", path, error)
            return

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
        data[PHYSICAL_TARGET_VALID_KEY] = torch.tensor(bool(target), dtype=torch.bool)
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


def unpack_motion_canvas(canvas: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Split a WSA1.5 motion canvas into motion and semantic-ID images."""

    if canvas.shape[-1] % 2:
        raise ValueError("Motion canvas width must be even")
    return canvas.chunk(2, dim=-1)


@DataTransformFn.register_subclass("wsa1_5_motion_targets")
@dataclass
class PrepareWSA15MotionTargetsFn(DataTransformFn):
    """Pack motion-mask and ID videos side-by-side for the WAN image expert."""

    num_future_frames: int = 4
    height: int = 224
    width: int = 224
    output_key: str = WORLD_TARGET_KEY

    def _resize(self, video: torch.Tensor, width: int, mode: str) -> torch.Tensor:
        kwargs = {"align_corners": False} if mode != "nearest" else {}
        return F.interpolate(video, size=(self.height, width), mode=mode, **kwargs)

    def __call__(self, data: DataDict) -> DataDict:
        frame_count = self.num_future_frames + 1
        motion = _find_tensor(data, MOTION_MASK_KEYS)
        ids = _find_tensor(data, MOTION_MASK_ID_KEYS)
        valid = motion is not None and ids is not None

        if valid:
            motion_video = _as_video(motion)
            id_video = _as_video(ids)
            count = min(motion_video.shape[0], id_video.shape[0])
            valid = count > 0
            if valid:
                motion_video = motion_video[:count]
                id_video = id_video[:count]
                if count < frame_count:
                    pad = frame_count - count
                    motion_video = torch.cat([motion_video, motion_video[-1:].expand(pad, -1, -1, -1)], dim=0)
                    id_video = torch.cat([id_video, id_video[-1:].expand(pad, -1, -1, -1)], dim=0)
                else:
                    motion_video = motion_video[:frame_count]
                    id_video = id_video[:frame_count]

                left_width = self.width // 2
                right_width = self.width - left_width
                motion_video = self._resize(motion_video, left_width, "bilinear")
                id_video = self._resize(id_video, right_width, "nearest")
                canvas = torch.cat([motion_video, id_video], dim=-1)
                data[self.output_key] = canvas.mul(2.0).sub(1.0)
                # WAN treats frame zero as a fixed conditioning frame.  Keep it
                # blank so the target is predictable from the current RGB
                # observation rather than from an unavailable current motion map.
                data[self.output_key][0].fill_(-1.0)
        if not valid:
            data[self.output_key] = -torch.ones(frame_count, 3, self.height, self.width)

        data[WORLD_TARGET_VALID_KEY] = torch.tensor(valid, dtype=torch.bool)
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
            WORLD_TARGET_KEY: data.get(
                WORLD_TARGET_KEY,
                -torch.ones(self.num_future_frames + 1, 3, self.height, self.width),
            ),
            WORLD_TARGET_VALID_KEY: data.get(WORLD_TARGET_VALID_KEY, torch.tensor(False)),
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
            ACTION_DECODE_EEF_KEY,
            ACTION_DECODE_CURRENT_KEY,
            ACTION_DECODE_SCALE_KEY,
            ACTION_DECODE_REORDER_KEY,
            ACTION_DECODE_EEF_INDEX_KEY,
            ACTION_DECODE_CAMERA_KEY,
            ACTION_RAW_DIM_KEY,
            CAMERA_FRAME_VALID_KEY,
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
            WORLD_TARGET_KEY: -torch.ones(self.num_future_frames + 1, 3, self.height, self.width),
            WORLD_TARGET_VALID_KEY: torch.tensor(False),
            PHYSICAL_TARGET_VALID_KEY: torch.tensor(False),
            SUBTASK_TARGET_KEY: torch.full((self.num_subtask_tokens,), -100, dtype=torch.long),
            SUBTASK_TARGET_MASK_KEY: torch.zeros(self.num_subtask_tokens, dtype=torch.bool),
            STATE_VALID_MASK_KEY: torch.zeros(CANONICAL_DIM, dtype=torch.bool),
            ACTION_VALID_MASK_KEY: torch.zeros(CANONICAL_DIM, dtype=torch.bool),
            ACTION_DECODE_MATRIX_KEY: torch.zeros(CANONICAL_DIM, CANONICAL_DIM),
            ACTION_DECODE_BIAS_KEY: torch.zeros(CANONICAL_DIM),
            ACTION_DECODE_EEF_KEY: torch.tensor(False),
            ACTION_DECODE_CURRENT_KEY: torch.zeros(2, 7),
            ACTION_DECODE_SCALE_KEY: torch.ones(2, 3),
            ACTION_DECODE_REORDER_KEY: torch.zeros(CANONICAL_DIM, CANONICAL_DIM),
            ACTION_DECODE_EEF_INDEX_KEY: torch.full((2, 7), -1, dtype=torch.long),
            ACTION_DECODE_CAMERA_KEY: torch.eye(3),
            ACTION_RAW_DIM_KEY: torch.tensor(0, dtype=torch.long),
            CAMERA_FRAME_VALID_KEY: torch.tensor(False),
        }
