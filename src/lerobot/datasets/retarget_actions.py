"""Shared IK action windows for WSA training and offline normalization statistics."""
from dataclasses import replace
import json
import logging
import math
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from lerobot.datasets.synthetic_embodiment import (
    HEAD_EEF_KEY, HUMAN_EEF_EMBODIMENT, EmbodimentSpec, SyntheticActionTable,
    HUMAN_EEF_VALID_KEYS, _scalar_column_to_numpy, _validate_annotation_rows, _vector_column_to_numpy,
    dexterous_body_alias, embodiment_schema_prefix, load_embodiment_schemas,
)

HEAD_EEF_DIM = 7
HEAD_EEF_MASK = (3, -4)  # Delta translation, absolute quaternion, like the arm EEF.
IDLE_WINDOW = 32
MIN_ACTIVE_WINDOW_RATIO = 0.7
IDLE_ROTATION_THRESHOLD = 0.1  # rad/s
IDLE_MAD_MULTIPLIER = 6
WRIST_POSITION_KEYS = ("human_eef.right.pos", "human_eef.left.pos")
WRIST_ROTATION_KEYS = ("human_eef.right.rot", "human_eef.left.rot")


def action_offsets(chunk_size, stride=1, *, retarget=False):
    """Native controls start at t; retarget poses start at t+stride."""
    if chunk_size < 1 or stride < 1:
        raise ValueError("chunk_size and stride must be positive")
    return (np.arange(chunk_size, dtype=np.int64) + int(retarget)) * stride


def _pose_distance(a, b, angular):
    if not angular:
        return np.linalg.norm(a - b, axis=1)
    # Align quaternion signs before measuring the shortest relative rotation.
    b = np.where((a * b).sum(axis=1, keepdims=True) < 0, -b, b)
    return 4 * np.arctan2(np.linalg.norm(a - b, axis=1), np.linalg.norm(a + b, axis=1))


def _idle_windows(pose, valid, threshold, angular):
    windows = np.lib.stride_tricks.sliding_window_view
    visible = windows(valid, IDLE_WINDOW).all(axis=1)
    steps = _pose_distance(pose[:-1], pose[1:], angular)
    valid_steps = valid[:-1] & valid[1:]
    # Pauses must not drive the outlier cutoff below ordinary motion.
    min_step = threshold if angular else threshold / (IDLE_WINDOW - 1)
    moving = steps[valid_steps & (steps > min_step)]
    if moving.size:
        median = np.median(moving)
        mad = np.median(np.abs(moving - median))
        tolerance = np.finfo(np.float32).eps * max(1., median)
        outlier = valid_steps & (steps > median + IDLE_MAD_MULTIPLIER * mad + tolerance)
        if mad <= tolerance:
            # A lone out-and-back jump otherwise defines its own zero-MAD baseline.
            fast = valid_steps & (steps > min_step) & (steps >= median - tolerance)
            spike = fast[:-1] & fast[1:] & (_pose_distance(pose[:-2], pose[2:], angular) <= min_step)
            outlier[:-1] |= spike
            outlier[1:] |= spike
        visible &= ~windows(outlier, IDLE_WINDOW - 1).any(axis=1)
    if angular:
        score = windows(steps, IDLE_WINDOW - 1).mean(axis=1)
    else:
        score = np.linalg.norm(np.ptp(windows(pose, IDLE_WINDOW, axis=0), axis=-1), axis=1)
    return visible & (score > threshold), visible


def retarget_idle_bounds(root, path, threshold):
    """Return crops for assessed episodes; None values mark rejected episodes."""
    if threshold is None or not path.is_file():
        return None
    if not math.isfinite(threshold) or threshold <= 0:
        raise ValueError("idle_motion_threshold must be finite and positive")
    parquet = pq.ParquetFile(path)
    if parquet.metadata.num_rows == 0:
        return None
    keys = (*WRIST_POSITION_KEYS, *HUMAN_EEF_VALID_KEYS)
    present = set(keys).intersection(parquet.schema_arrow.names)
    if not present:
        logging.info("[WSA idle] %s has no human wrist positions, keeping original windows", path)
        return None
    if len(present) != len(keys):
        raise ValueError(f"{path}: incomplete wrist positions or validity columns for idle filtering")
    rotation_keys = set(WRIST_ROTATION_KEYS).intersection(parquet.schema_arrow.names)
    if rotation_keys:
        info = json.loads((root / "meta/info.json").read_text(encoding="utf-8"))
        fps = float(info.get("fps", 0))
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError(f"{root}: idle rotation filtering requires a positive dataset fps")
    metadata = parquet.read(columns=["index", "episode_index", "frame_index", *keys, *sorted(rotation_keys)])
    indices, episodes, frames = (_scalar_column_to_numpy(metadata[key], np.int64)
                                 for key in ("index", "episode_index", "frame_index"))
    _validate_annotation_rows(indices, episodes, frames, path, allow_sparse=True)
    episode_columns = ("episode_index", "dataset_from_index", "dataset_to_index")
    episode_table = pq.ParquetDataset(root / "meta/episodes").read(columns=list(episode_columns))
    episode_ranges = {ep: (lo, hi) for ep, lo, hi in zip(
        *(episode_table[key].to_pylist() for key in episode_columns), strict=True)}
    signals = []
    for pos_key, rot_key, valid_key in zip(WRIST_POSITION_KEYS, WRIST_ROTATION_KEYS, HUMAN_EEF_VALID_KEYS, strict=True):
        if not pa.types.is_boolean(metadata[valid_key].type):
            raise TypeError(f"{path}: {valid_key} must be boolean")
        annotated = _scalar_column_to_numpy(metadata[valid_key].fill_null(False), np.bool_)
        for key, dim in ((pos_key, 3), (rot_key, 4)):
            if key not in metadata.column_names:
                continue
            column = metadata[key]
            missing = pa.scalar([float("nan")] * dim, type=column.type)
            pose = _vector_column_to_numpy(column.fill_null(missing), expected_dim=dim).astype(np.float64)
            valid = annotated & np.isfinite(pose).all(axis=1)
            angular = dim == 4
            if angular:
                norm = np.linalg.norm(pose, axis=1)
                valid &= norm > 0
                pose[valid] /= norm[valid, None]
            pose[~valid] = 0
            signals.append((pose, valid, IDLE_ROTATION_THRESHOLD / fps if angular else threshold, angular))
    bounds = np.r_[0, np.flatnonzero(np.diff(episodes) != 0) + 1, len(episodes)]
    crops = {}
    for start, end in zip(bounds[:-1], bounds[1:], strict=True):
        ep = int(episodes[start])
        if ep not in episode_ranges:
            raise ValueError(f"{path}: annotation references unknown episode {ep}")
        source_start, source_stop = episode_ranges[ep]
        if (not np.array_equal(indices[start:end], source_start + frames[start:end])
                or frames[start] < 0 or indices[end - 1] >= source_stop):
            raise ValueError(f"{path}: annotation indices do not align with episode {ep}")
        length = end - start
        if length != source_stop - source_start or not any(valid[start:end].any() for _, valid, _, _ in signals):
            continue
        crops[ep] = None
        if length < IDLE_WINDOW:
            continue
        active = np.zeros(length - IDLE_WINDOW + 1, dtype=bool)
        valid_windows = np.zeros_like(active)
        for pose, valid, signal_threshold, angular in signals:
            moving_windows, visible = _idle_windows(pose[start:end], valid[start:end], signal_threshold, angular)
            active |= moving_windows
            valid_windows |= visible
        moving = np.flatnonzero(active)
        if not len(moving):
            continue
        first, last = moving[0], moving[-1]
        # Keep all frames between these windows, including pauses and invalid measurements.
        if len(moving) / valid_windows[first:last + 1].sum() >= MIN_ACTIVE_WINDOW_RATIO:
            lo = int(frames[start + first])
            hi = int(frames[start + last + IDLE_WINDOW - 1]) + 1
            crops[ep] = (lo, hi)
    logging.info("[WSA idle] %s kept %d/%d assessed episodes; skipped filtering for %d unassessed episodes",
                 path, sum(crop is not None for crop in crops.values()), len(crops), len(episode_ranges) - len(crops))
    return crops


class RetargetActionSource:
    """Read one embodiment plus its shared robot EEF, preserving source frame indices."""
    def __init__(self, root, chunk_size, *, stride=1, embodiments=None, action_space="unified72",
                 action_mode="delta", active_head=False, stats_dir=None, row_group_cache_size=2,
                 idle_motion_threshold=None,
                 annotation_file="meta/annotations/action.parquet"):
        if chunk_size < 1 or stride < 1:
            raise ValueError("chunk_size and stride must be positive")
        if action_space not in {"legacy", "unified72"} or action_mode not in {"abs", "delta"}:
            raise ValueError("Invalid action space or abs/delta mode")
        if active_head and action_space != "unified72":
            raise ValueError("active_head requires unified72")
        if idle_motion_threshold is not None and (
            not math.isfinite(idle_motion_threshold) or idle_motion_threshold <= 0
        ):
            raise ValueError("idle_motion_threshold must be finite and positive")
        self.root = Path(root)
        self.chunk_size, self.stride = chunk_size, stride
        self.action_space, self.action_mode = action_space, action_mode
        self.idle_motion_threshold = idle_motion_threshold
        self.offsets = np.concatenate(([0], action_offsets(chunk_size, stride, retarget=True)))
        self.table = SyntheticActionTable.from_parquet(
            self.root / annotation_file, row_group_cache_size=row_group_cache_size)
        available = self.table.names
        prefix = embodiment_schema_prefix(
            json.loads((self.root / "meta/info.json").read_text(encoding="utf-8"))["robot_type"])
        selected = [name.removeprefix(prefix) for name in embodiments] if embodiments else [
            name for name in available if action_space == "legacy" or name != HUMAN_EEF_EMBODIMENT]
        selected = [name if name in available else dexterous_body_alias(name) for name in selected]
        if action_space == "unified72" and HUMAN_EEF_EMBODIMENT in selected:
            raise ValueError("human_eef supplies the selected IK embodiment's EEF; select IK embodiment names")
        names = tuple(name for name in selected if name in available)
        if not names or len(set(names)) != len(names) or len(names) > 63:
            raise ValueError(f"Require 1..63 distinct available embodiments in {self.root}: {selected}")
        self.body_ids = [available.index(name) for name in names]
        self.eef_id = available.index(HUMAN_EEF_EMBODIMENT) if HUMAN_EEF_EMBODIMENT in available else None
        self.with_eef = [action_space == "unified72" and self.eef_id is not None
                         and name != HUMAN_EEF_EMBODIMENT for name in names]
        self.schemas = []
        self.specs = []
        specs = [self.table.specs[body_id] for body_id in self.body_ids]
        schemas = load_embodiment_schemas(specs, prefix)
        for spec, schema, include_eef in zip(specs, schemas, self.with_eef, strict=True):
            schema.for_action_space(action_space)
            if include_eef:
                eef_mapping = [[spec.dim, spec.dim + 7, 7, 14],
                               [spec.dim + 7, spec.dim + 14, 0, 7]]
                schema = replace(schema, state_dim=spec.dim + 14, action_dim=spec.dim + 14,
                                 action_mask_spec=[*schema.action_mask_spec, 3, -4, 3, -4],
                                 action_layout=[*schema.action_layout, {"name": "right_eef", "size": 7},
                                                {"name": "left_eef", "size": 7}],
                                 unified_action_reorder=[*schema.unified_action_reorder, *eef_mapping],
                                 unified_state_reorder=[*schema.unified_state_reorder, *eef_mapping])
                spec = replace(spec, dim=spec.dim + 14)
            if action_space == "unified72":
                spec = replace(spec, name=schema.robot_type)
            self.schemas.append(schema)
            self.specs.append(spec)
        self.names = tuple(spec.name for spec in self.specs)
        if len(set(self.names)) != len(self.names):
            raise ValueError(f"Multiple annotation columns resolve to the same embodiment in {self.root}")

        self.head_spec = None
        columns = ["index", "episode_index", "frame_index",
                   *dict.fromkeys(key for spec in self.table.specs for key in spec.valid_keys)]
        if active_head:
            fields = pq.ParquetFile(self.table.path).schema_arrow
            if HEAD_EEF_KEY in fields.names:
                dtype = fields.field(HEAD_EEF_KEY).type
                if (not (pa.types.is_list(dtype) or pa.types.is_large_list(dtype) or pa.types.is_fixed_size_list(dtype))
                        or pa.types.is_fixed_size_list(dtype) and dtype.list_size != HEAD_EEF_DIM):
                    raise ValueError(f"{HEAD_EEF_KEY} must contain {HEAD_EEF_DIM}D vectors")
                valid_key = f"{HEAD_EEF_KEY}_valid"
                self.head_spec = EmbodimentSpec(f"{prefix}head_eef", HEAD_EEF_KEY, valid_key, HEAD_EEF_DIM,
                                                allow_missing=True)
                self.head_reader = SyntheticActionTable(self.table.path, [self.head_spec], self.table.row_group_offsets,
                                                        row_group_cache_size=row_group_cache_size)
                if valid_key in fields.names:
                    if not pa.types.is_boolean(fields.field(valid_key).type):
                        raise TypeError(f"{valid_key} must be boolean")
                    columns.append(valid_key)
        metadata = pq.read_table(self.table.path, columns=columns)
        self.head_valid = None
        if self.head_spec is not None and self.head_spec.valid_key in columns:
            self.head_valid = _scalar_column_to_numpy(
                metadata[self.head_spec.valid_key].fill_null(False), np.bool_)
        self.indices, self.episodes, self.frames = [
            _scalar_column_to_numpy(metadata[key], np.int64) for key in ("index", "episode_index", "frame_index")]
        _validate_annotation_rows(self.indices, self.episodes, self.frames, self.table.path, allow_sparse=True)
        self.idle_bounds = retarget_idle_bounds(self.root, self.table.path, idle_motion_threshold)
        continuous = (np.diff(self.episodes) == 0) & (np.diff(self.frames) == 1)
        bounds = np.r_[0, np.flatnonzero(~continuous) + 1, len(self.episodes)]
        anchors = np.zeros(len(self.episodes), dtype=bool)
        for start, stop in zip(bounds[:-1], bounds[1:], strict=True):
            ep = int(self.episodes[start])
            if self.idle_bounds is not None and ep in self.idle_bounds:
                crop = self.idle_bounds[ep]
                if crop is None:
                    continue
                start, stop = (int(start - self.frames[start] + edge) for edge in crop)
            anchors[start:max(start, stop - self.offsets[-1])] = True
        self.candidate_masks = np.zeros(len(self.table), dtype=np.uint64)
        for body, (body_id, include_eef) in enumerate(zip(self.body_ids, self.with_eef, strict=True)):
            spec = self.table.specs[body_id]
            keys = list(spec.valid_keys)
            if include_eef:
                keys.extend(self.table.specs[self.eef_id].valid_keys)
            valid = np.logical_and.reduce([_scalar_column_to_numpy(metadata[key].fill_null(False), np.bool_) for key in keys])
            eligible = anchors & valid
            for offset in self.offsets[1:]:
                if offset >= len(eligible):
                    eligible[:] = False
                    break
                eligible[:-offset] &= valid[offset:]
            self.candidate_masks |= eligible.astype(np.uint64) << np.uint64(body)
        self.stats = {}
        if stats_dir is not None:
            entries = [(self.specs[body], self.schemas[body].action_mask_spec) for body in self.active_bodies]
            if self.head_spec is not None and entries:
                entries.append((self.head_spec, list(HEAD_EEF_MASK)))
            for spec, mask in entries:
                path = Path(stats_dir) / spec.name / "stats.json"
                payload = json.loads(path.read_text(encoding="utf-8"))
                if payload.get("training") != {**self.training_spec, "action_mask_spec": mask}:
                    raise ValueError(f"Statistics window/representation differs from training: {path}")
                for key in ("observation.state", "action"):
                    if any(len(payload[key][stat]) != spec.dim for stat in ("mean", "std")):
                        raise ValueError(f"Statistics dimension differs from {spec.name}: {path}")
                self.stats[spec.name] = payload

    @property
    def training_spec(self):
        spec = {"action_space": self.action_space, "action_mode": self.action_mode,
                "chunk_size": self.chunk_size, "stride": self.stride}
        if self.idle_motion_threshold is not None:
            spec["idle_filter"] = {"window": IDLE_WINDOW, "min_active_ratio": MIN_ACTIVE_WINDOW_RATIO,
                                   "motion_threshold": self.idle_motion_threshold,
                                   "rotation_threshold_rad_s": IDLE_ROTATION_THRESHOLD,
                                   "mad_multiplier": IDLE_MAD_MULTIPLIER}
        return spec

    @property
    def active_bodies(self):
        union = int(np.bitwise_or.reduce(self.candidate_masks, initial=np.uint64(0)))
        return [body for body in range(len(self.names)) if union & (1 << body)]

    def rows_for_base(self, base):
        metadata = base.hf_dataset.select_columns(["index", "episode_index", "frame_index"]).with_format("arrow")[:]
        indices, episodes, frames = [_scalar_column_to_numpy(metadata[key], np.int64)
                                     for key in ("index", "episode_index", "frame_index")]
        rows = np.minimum(np.searchsorted(self.indices, indices), len(self.indices) - 1)
        matched = self.indices[rows] == indices
        if (not np.array_equal(self.episodes[rows[matched]], episodes[matched])
                or not np.array_equal(self.frames[rows[matched]], frames[matched])):
            raise ValueError(f"Retarget annotations do not align with the loaded dataset: {self.root}")
        return np.where(matched, rows, -1)

    def candidates(self, row):
        mask = int(self.candidate_masks[row])
        return [body for body in range(len(self.names)) if mask & (1 << body)]

    def read_window(self, body, row):
        if not int(self.candidate_masks[row]) & (1 << body):
            raise ValueError(f"Invalid {self.names[body]} action window at annotation row {row}")
        stop = row + int(self.offsets[-1]) + 1
        values = self.table.read_values(self.body_ids[body], row, stop, offsets=self.offsets)
        if self.with_eef[body]:
            eef = self.table.read_values(self.eef_id, row, stop, offsets=self.offsets)
            values = np.concatenate((values, eef[:, :7], eef[:, 8:15]), axis=-1)
        return values[0].copy(), values[1:].copy()

    def read_head_window(self, row):
        """Use the same anchor and future offsets as the sampled arm embodiment."""
        stop = row + int(self.offsets[-1]) + 1
        poses = self.head_reader.read_values(0, row, stop, offsets=self.offsets)
        valid = np.isfinite(poses).all(axis=1)
        if self.head_valid is not None:
            valid &= self.head_valid[row + self.offsets]
        return np.where(valid[:, None], poses, 0), valid
