"""A thin, target-isolating adapter over the existing LeRobot V3 reader.

All timestamps come from the source frame table. Missing media is accompanied
by masks; decoding failures never become unmarked ground truth.
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate

from lerobot.dataset_schemas import get_registry, get_schema
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.transforms.core import DeltaActionTransformFn, NormalizeTransformFn

from .configuration_wsa1_5 import WSA1_5DatasetConfig
from .text_embeddings import PROMPT_PREFIX, PROMPT_VERSION, TextEmbeddingCache
from .utils.temporal import history_indices
from .utils.video import decode_video_frames
from .utils.visual_augmentation import (
    CAMERA_WEIGHTS, apply_pixel_augmentation, load_visual_augmentation, sample_pixel_augmentation,
)

HAND_NAMES = {0: "left hand", 1: "right hand", 2: "both hands"}


def compose_rgb_views(views, valid, height=384, width=320):
    """Official WSA robotwin layout: top 2/3, then left/right bottom 1/3."""
    if len(views) != 3 or len(valid) != 3 or height % 3 or width % 2:
        raise ValueError("Expected three ordered views and a valid WSA canvas size")
    sizes = [(height * 2 // 3, width), (height // 3, width // 2), (height // 3, width // 2)]
    length = views[0].shape[0]
    canvas = views[0].new_zeros((length, 3, height, width), dtype=torch.float32)
    pixels_valid = torch.zeros((length, 1, height, width), dtype=torch.bool, device=views[0].device)
    regions = [(slice(0, height * 2 // 3), slice(None)),
               (slice(height * 2 // 3, None), slice(0, width // 2)),
               (slice(height * 2 // 3, None), slice(width // 2, None))]
    for view, is_valid, size, (rows, cols) in zip(views, valid, sizes, regions, strict=True):
        if view.ndim != 4 or view.shape[1] != 3:
            raise ValueError("Each view must have shape [T,3,H,W]")
        mask = torch.as_tensor(is_valid, device=view.device)
        if mask.dtype != torch.bool:
            raise TypeError("RGB view validity must be boolean")
        mask = mask.expand(view.shape[0]).reshape(-1, 1, 1, 1)
        if mask.any():
            patch = F.interpolate(view.float(), size=size, mode="bilinear", align_corners=False)
            canvas[:, :, rows, cols] = patch.masked_fill(~mask, 0)
            pixels_valid[:, :, rows, cols] = mask
    return canvas, pixels_valid


def build_inference_condition(observed_rgb, text, *, observed_valid=None):
    """Explicit signature prevents accepting motion or target fields as conditions."""
    condition = {"observed_rgb": observed_rgb, "condition_text": text}
    if observed_valid is not None:
        condition["observed_valid"] = observed_valid
    return condition


@dataclass(frozen=True)
class Segment:
    episode: int
    first: int  # Relative index in the actual loaded HF table.
    last: int  # Inclusive endpoint.
    subtask: int
    text: str
    metadata_valid: bool


def condition_text(episode, subtask=None):
    # Only read fields verified in the real sample. Unknown optional fields are empty.
    task = episode.get("task_instruction", "") or ""
    prefix = f"{PROMPT_PREFIX}\nTask: {task}"
    if subtask is None:
        return prefix
    instruction = subtask.get("subtask_text", "") or ""
    hand = HAND_NAMES.get(subtask.get("hand_used"), "")
    scene = episode.get("scene", "") or ""
    return f"{prefix}\nSubtask: {instruction}\nHand: {hand}\nScene: {scene}"


def episode_task_texts(annotation, metadata):
    """Use the sidecar's original task, or the tasks recorded by LeRobot."""
    task = annotation.get("task_instruction")
    if task is not None and not isinstance(task, str):
        raise ValueError("Original task_instruction must be a string")
    tasks = [task] if task else metadata.get("tasks", [])
    if (not isinstance(tasks, list) or not tasks
            or any(not isinstance(value, str) or not value.strip() for value in tasks)):
        raise ValueError("Missing or invalid original task in sidecar and episode metadata")
    return list(dict.fromkeys(tasks))


def load_sidecar_annotations(path):
    path = Path(path)
    sidecar = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    if not isinstance(sidecar, list):
        raise ValueError("Verified sidecar schema is a list of episode objects")
    annotations = {}
    for row in sidecar:
        if not isinstance(row, dict) or row.get("episode_index") is None:
            logging.warning("[WSA sidecar] dropped annotation without episode_index")
            continue
        try:
            ep = int(row["episode_index"])
        except (TypeError, ValueError):
            raise ValueError("Invalid non-null sidecar episode_index") from None
        if isinstance(row["episode_index"], bool) or isinstance(row["episode_index"], float) and row["episode_index"] != ep:
            raise ValueError("Sidecar episode_index must be integral, never rounded")
        if ep in annotations:
            raise ValueError(f"Duplicate sidecar episode_index {ep}")
        annotations[ep] = row
    return annotations


def control_mode(schema):
    # The current registry names control space `action_mode`, not abs/delta.
    modes = {"joint": "joint", "end_effector": "endpose_effector", "endpose_effector": "endpose_effector"}
    if schema.action_mode not in modes:
        raise ValueError(f"Schema {schema.robot_type} requires control mode joint or endpose_effector")
    return modes[schema.action_mode]


def action_prompt(task, mode, action_mode, *, annotation=None, subtask=None):
    if not isinstance(task, str) or not task.strip():
        raise ValueError("Action prompt requires an actual task instruction")
    if mode not in {"joint", "endpose_effector"} or action_mode not in {"abs", "delta"}:
        raise ValueError("Invalid control/action mode")
    # Preserve verified Stage-1 subtask/hand/scene semantics, using ordinary UMT5 text.

    text = condition_text({**(annotation or {}), "task_instruction": task}, subtask)
    return f"{text}\nControl Mode: {mode}\nAction Mode: {action_mode}"


def _reorder(value, spec, dim):
    if spec is None:
        if value.shape[-1] != dim:
            raise ValueError(f"Schema vector dimension {value.shape[-1]} != configured {dim}")
        return value
    out = value.new_zeros(*value.shape[:-1], dim)
    used = set()
    for lo, hi, start, end in spec:
        if not (0 <= lo < hi <= value.shape[-1] and 0 <= start < end <= dim and hi - lo == end - start):
            raise ValueError("Invalid schema state/action reorder bounds")
        if used.intersection(range(start, end)):
            raise ValueError("Overlapping schema reorder destinations")
        used.update(range(start, end))
        out[..., start:end] = value[..., lo:hi]
    return out


class ActionCodec:
    """Use schema mappings and the stats already selected by the dataset factory."""
    def __init__(self, meta, config):
        self.schema = get_schema(meta.robot_type)
        self.control_mode = control_mode(self.schema)
        self.config = config
        self.state_keys, self.action_keys = self.schema.get_state_keys(), self.schema.get_action_keys()
        present = [key in meta.features for key in self.action_keys]
        if any(present) and not all(present):
            raise ValueError(f"Partial action schema for {meta.robot_type}: {self.action_keys}")
        self.has_action = all(present)
        self.has_state = all(key in meta.features for key in self.state_keys)
        if self.has_action and not self.has_state:
            raise ValueError("Action supervision requires the current state features declared by the schema")
        self.stats = {}
        if not self.has_state and not self.has_action:
            return  # Masked placeholders for world-only sources in a mixed dataset.
        stats = meta.stats
        for key in self.state_keys + (self.action_keys if self.has_action else []):
            dim = meta.features[key]["shape"]
            if len(dim) != 1:
                raise ValueError(f"State/action source {key} must be a vector feature")
            entry = {name: np.asarray(stats[key][name], dtype=np.float32) for name in ["mean", "std"]}
            if (any(value.shape != tuple(dim) or not np.isfinite(value).all() for value in entry.values())
                    or (entry["std"] < 0).any()):
                raise ValueError(f"Invalid mean/std statistics for {key}")
            self.stats[key] = entry
        self.normalize_state = NormalizeTransformFn(selected_keys=self.state_keys, mode="mean_std", norm_stats=self.stats)
        self.normalize_action = NormalizeTransformFn(selected_keys=self.action_keys, mode="mean_std", norm_stats=self.stats)
        state_size = sum(meta.features[k]["shape"][0] for k in self.state_keys)
        self.delta = self.schema.action_mask
        self.delta_transform = DeltaActionTransformFn(mask=self.delta, mapping=self.schema.feature_mapping,
                                                       robot_type=meta.robot_type)
        if self.has_action:
            action_size = sum(meta.features[k]["shape"][0] for k in self.action_keys)
            if config.action_mode == "delta" and (self.delta.numel() != action_size or state_size != action_size):
                raise ValueError("Delta action requires an explicit schema mask and aligned state/action coordinates")
            self.action_sizes = [meta.features[k]["shape"][0] for k in self.action_keys]
        _reorder(torch.zeros(state_size), self.schema.state_reorder, config.state_dim)
        if self.has_action:
            _reorder(torch.zeros(action_size), self.schema.action_reorder, config.action_dim)

    def encode(self, state, action=None):
        cfg = self.config
        raw = torch.cat([torch.as_tensor(state[k], dtype=torch.float32) for k in self.state_keys], -1)
        if raw.ndim != 1 or not torch.isfinite(raw).all():
            raise ValueError("Current state must be a finite vector")
        normalized_state = self.normalize_state({key: torch.as_tensor(state[key], dtype=torch.float32)
                                           for key in self.state_keys})
        result = {"state": _reorder(torch.cat([normalized_state[k] for k in self.state_keys], -1),
                                    self.schema.state_reorder, cfg.state_dim), "state_valid": torch.tensor(True)}
        if action is None:
            return result
        values = {key: torch.as_tensor(action[key], dtype=torch.float32) for key in self.action_keys}
        chunk = torch.cat([values[k] for k in self.action_keys], -1)
        if chunk.ndim != 2 or not torch.isfinite(chunk).all():
            raise ValueError("Actions must be a finite [chunk,source_action_dim] tensor")
        if cfg.action_mode == "delta":
            data = {**values, **{k: torch.as_tensor(state[k], dtype=torch.float32) for k in self.state_keys}}
            values = self.delta_transform(data)
            values = {key: values[key] for key in self.action_keys}
        norm = self.normalize_action(values)
        result["action"] = _reorder(torch.cat([norm[k] for k in self.action_keys], -1),
                                    self.schema.action_reorder, cfg.action_dim)
        return result

    def decode(self, action, state):
        """Return absolute actions in canonical schema order, in physical units."""
        if action.shape[-1] != self.config.action_dim or not torch.isfinite(action).all():
            raise ValueError("Invalid normalized action chunk")
        mean = torch.cat([torch.from_numpy(self.stats[k]["mean"]) for k in self.action_keys])
        std = torch.cat([torch.from_numpy(self.stats[k]["std"]) + 1e-6 for k in self.action_keys])
        mean = _reorder(mean, self.schema.action_reorder, self.config.action_dim).to(action)
        std = _reorder(std, self.schema.action_reorder, self.config.action_dim).to(action)
        out = action.float() * std + mean
        if self.config.action_mode == "delta":
            raw = torch.cat([torch.as_tensor(state[k], device=action.device, dtype=torch.float32)
                             for k in self.state_keys], -1)
            if action.ndim == 3 and raw.ndim == 1 and action.shape[0] == 1:
                raw = raw[None]
            if raw.shape[:-1] != action.shape[:-2] or not torch.isfinite(raw).all():
                raise ValueError("Raw current state batch must match the action batch exactly")
            offset = _reorder(raw * self.delta.to(raw.device), self.schema.action_reorder, self.config.action_dim)
            out = out + offset.unsqueeze(-2)
        return out


class LoadedEpisodeMetadata:
    """Expose loaded-sample offsets to existing mixing/sampling code.

    Video path methods stay bound to the original LeRobot metadata, preserving
    global episode IDs and chunk offsets after an episodes=[...] filter.
    """
    def __init__(self, source, episode_ids, starts, ends):
        self.source = source
        self.entries = {}
        for ep, start, end in zip(episode_ids, starts, ends, strict=True):
            self.entries[int(ep)] = {**source.episodes[int(ep)], "dataset_from_index": int(start),
                                     "dataset_to_index": int(end)}
        self.total_frames = int(ends[-1]) if len(ends) else 0
        self.total_episodes = len(self.entries)
        self.episodes = self

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "source"), name)

    def __getitem__(self, key):
        if isinstance(key, str):
            return [entry[key] for entry in self.entries.values()]
        return self.entries[int(key)]


class WSA1_5Dataset(Dataset):
    def __init__(self, base, config: WSA1_5DatasetConfig, *, include_3d_targets=True, text_embedding_shape=None,
                 action_config=None, training=True):
        self.base, self.config = base, config
        if type(training) is not bool:
            raise TypeError("training must be boolean")
        self.augmentation_config = (
            load_visual_augmentation(config.image_augmentation_config,
                                     image_augment=config.image_augment, object_paste=config.object_paste)
            if training else None
        )
        self.action_config = action_config
        self.action_codec = None
        if action_config is not None:
            if config.action_mode != action_config.action_mode:
                raise ValueError("dataset.action_mode and policy.action_mode must match")
            self.action_codec = ActionCodec(base.meta, action_config)
        if type(include_3d_targets) is not bool:
            raise TypeError("include_3d_targets must be boolean")
        self.include_3d_targets = include_3d_targets
        self.repo_id = base.repo_id
        self.meta = base.meta
        self.root = Path(base.root)
        base._ensure_hf_dataset_loaded()
        if self.action_codec is not None and self.action_codec.has_action:
            action_table = base.hf_dataset.select_columns(self.action_codec.action_keys).with_format("arrow")[:]
            if any(column.null_count for column in action_table.columns) and not action_table.drop_null().num_rows:
                raise ValueError("Source declares action features but has no valid action rows; mark world-only sources explicitly")
        # Arrow bypasses the HF per-scalar Torch transform and honors filtered row order.
        columns = ["timestamp", "frame_index", "episode_index"]
        table = base.hf_dataset.select_columns(columns).with_format("arrow")[:]
        for name in columns:
            if table.column(name).null_count:
                raise ValueError(f"Dataset column {name} contains null values")
        for name in ["frame_index", "episode_index"]:
            column = table.column(name)
            if not pa.types.is_integer(column.type):
                raise TypeError(f"Dataset {name} must contain integer indices")
            if (pa.types.is_unsigned_integer(column.type) and len(column)
                    and int(column.to_numpy().max()) > np.iinfo(np.int64).max):
                raise ValueError(f"Dataset {name} exceeds int64 range")
        self.timestamps = np.asarray(table.column("timestamp").to_numpy(), dtype=np.float64)
        if not np.isfinite(self.timestamps).all():
            raise ValueError("Dataset timestamps must be finite")
        self.frame_indices = np.asarray(table.column("frame_index").to_numpy(), dtype=np.int64)
        self.episode_indices = np.asarray(table.column("episode_index").to_numpy(), dtype=np.int64)
        changes = np.flatnonzero(self.episode_indices[1:] != self.episode_indices[:-1]) + 1
        bounds = np.concatenate(([0], changes, [len(self.episode_indices)]))
        self.episode_starts, self.episode_ends = bounds[:-1], bounds[1:]
        if not len(self.episode_indices):
            self.episode_starts, self.episode_ends = bounds[:0], bounds[:0]
        episode_ids = self.episode_indices[self.episode_starts]
        if len(np.unique(episode_ids)) != len(episode_ids):
            raise ValueError("Episode rows are not contiguous")
        self.meta = LoadedEpisodeMetadata(base.meta, episode_ids, self.episode_starts, self.episode_ends)
        if config.view_keys:
            self.view_keys = list(config.view_keys)
            absent = [key for key in self.view_keys if key and key not in self.meta.features]
            if absent:
                raise ValueError(f"Explicit RGB keys absent from schema: {absent}; use an empty slot for a missing view")
        else:
            if base.meta.robot_type not in get_registry().list_available():
                raise ValueError(f"Unknown robot_type {base.meta.robot_type!r}; specify ordered view_keys instead of a default mapping")
            schema = get_schema(base.meta.robot_type)
            self.view_keys = []
            for name in config.view_names:
                candidates = [source for source, target in schema.image_mapping.items()
                              if target == name and source in self.meta.features]
                if len(candidates) > 1:
                    raise ValueError(f"Ambiguous RGB slot {name}: {candidates}; specify view_keys")
                self.view_keys.append(candidates[0] if candidates else "")
        if not any(key and key in self.meta.features for key in self.view_keys):
            raise ValueError("WSA requires at least one declared RGB observation view")
        for key in self.view_keys:
            if key and key in base.meta.features and base.meta.features[key]["dtype"] != "video":
                raise ValueError(f"WSA V3 adapter expects a video feature for {key}")
        sidecar_path = self.root / config.sidecar_file
        annotations = ({} if action_config is not None and not config.use_subtask
                       else load_sidecar_annotations(sidecar_path))
        self.sidecar_digest = hashlib.sha256(sidecar_path.read_bytes() if sidecar_path.exists() else b"").hexdigest()
        self.segments = []
        self._build_segments(annotations)
        self.sample_indices = np.empty(len(self.timestamps), dtype=np.int64)
        self.segment_for_sample = np.empty(len(self.timestamps), dtype=np.int32)
        cursor = 0
        for sid, segment in enumerate(self.segments):
            length = segment.last - segment.first + 1
            self.sample_indices[cursor:cursor + length] = np.arange(segment.first, segment.last + 1)
            self.segment_for_sample[cursor:cursor + length] = sid
            cursor += length
        assert cursor == len(self.timestamps), "Segments must cover every loaded row exactly once"
        self.text_embeddings = None
        if text_embedding_shape is not None:
            self.text_embeddings = TextEmbeddingCache(self.root / config.text_embedding_dir, *text_embedding_shape)
            self.text_embeddings.require(segment.text for segment in self.segments)
        self.delta_indices = {config.motion_key: [i * config.frame_stride for i in range(1, config.future_frames + 1)]}
        self._motion_reads = 0
        logging.info("[WSA data] repo=%s samples=%d views=%s sidecar_sha256=%s",
                     self.repo_id, len(self), self.view_keys, self.sidecar_digest)

    def _build_segments(self, annotations):
        for start, end in zip(self.episode_starts, self.episode_ends, strict=True):
            ep = int(self.episode_indices[start])
            frames = self.frame_indices[start:end]
            if not np.array_equal(frames, np.arange(len(frames))):
                raise ValueError(f"Episode {ep} must contain contiguous frame indices starting at zero")
            if np.any(np.diff(self.timestamps[start:end]) <= 0):
                raise ValueError(f"Episode {ep} has non-increasing timestamps")
            annotation = annotations.get(int(ep), {})
            occupied = np.zeros(len(frames), dtype=bool)
            ids = set()
            subtasks = annotation.get("subtasks")
            if subtasks is not None and not isinstance(subtasks, list):
                raise ValueError(f"Episode {ep} subtasks must be a list or null")
            for subtask in subtasks or []:
                required = {"start_frame", "end_frame", "subtask_id"}
                if not isinstance(subtask, dict) or not required.issubset(subtask):
                    logging.warning("[WSA sidecar] ep=%s missing subtask fields; masked", ep)
                    continue
                if any(subtask[k] is None for k in required):
                    logging.warning("[WSA sidecar] ep=%s null subtask bounds; masked", ep)
                    continue
                try:
                    first, last, sid = (int(subtask[k]) for k in ["start_frame", "end_frame", "subtask_id"])
                except (TypeError, ValueError):
                    raise ValueError(f"Episode {ep} has invalid non-null subtask indices") from None
                for key, value in zip(["start_frame", "end_frame", "subtask_id"], [first, last, sid], strict=True):
                    raw = subtask[key]
                    if isinstance(raw, bool) or isinstance(raw, float) and raw != value:
                        raise ValueError(f"Episode {ep} {key} must be integral, never rounded")
                if first < 0 or last < first or last >= len(frames) or sid < 0:
                    raise ValueError(f"Episode {ep} subtask {sid} bounds [{first},{last}] exceed length {len(frames)}")
                if sid in ids or occupied[first:last + 1].any():
                    raise ValueError(f"Episode {ep} has duplicate/overlapping subtasks")
                ids.add(sid)
                occupied[first:last + 1] = True
                self._add_segment(Segment(int(ep), int(start + first), int(start + last),
                                          sid, self._condition_text(annotation, subtask), True))
            # Missing subtask labels retain the original task, without endpoint/3D supervision.
            missing = np.flatnonzero(~occupied)
            for run in np.split(missing, np.flatnonzero(np.diff(missing) > 1) + 1):
                if len(run):
                    self._add_task_segments(ep, start, run, annotation)

    def _add_task_segments(self, ep, start, run, annotation):
        tasks = episode_task_texts(annotation, self.meta.episodes[ep])
        if len(tasks) == 1:
            self._add_segment(Segment(ep, int(start + run[0]), int(start + run[-1]),
                                      -1, self._condition_text({"task_instruction": tasks[0]}), False))
            return
        # Multiple tasks in one episode must follow the frame's actual task_index.
        column = "task_index"
        if "galaxea" in self.repo_id and "coarse_task_index" in self.base.hf_dataset.column_names:
            column = "coarse_task_index"
        table = self.base.hf_dataset.select_columns([column]).with_format("arrow")[start + run[0]:start + run[-1] + 1]
        indices = table[column]
        if indices.null_count or not pa.types.is_integer(indices.type):
            raise ValueError(f"Episode {ep} has invalid {column}")
        indices = indices.to_numpy()
        names = self.base.meta.tasks.index
        if (indices < 0).any() or (indices >= len(names)).any():
            raise ValueError(f"Episode {ep} has out-of-range {column}")
        cuts = np.flatnonzero(indices[1:] != indices[:-1]) + 1
        for lo, hi in zip(np.r_[0, cuts], np.r_[cuts, len(indices)], strict=True):
            task = names[int(indices[lo])]
            if task not in tasks:
                raise ValueError(f"Episode {ep} frame task is absent from episode tasks")
            self._add_segment(Segment(ep, int(start + run[lo]), int(start + run[hi - 1]),
                                      -1, self._condition_text({"task_instruction": task}), False))

    def _condition_text(self, annotation, subtask=None):
        if self.action_config is None:
            return condition_text(annotation, subtask)
        return action_prompt(annotation["task_instruction"], self.action_codec.control_mode,
                             self.action_config.action_mode, annotation=annotation, subtask=subtask)

    def _add_segment(self, segment):
        self.segments.append(segment)

    @property
    def num_frames(self):
        return len(self.sample_indices)

    @property
    def num_episodes(self):
        return self.meta.total_episodes

    def __len__(self):
        return self.num_frames

    def _decode(self, key, timestamps, ep):
        feature = self.meta.features.get(key)
        if not key or feature is None:
            return torch.zeros(len(timestamps), 3, 16, 16), torch.zeros(len(timestamps), dtype=torch.bool)
        h, w, c = feature["shape"]
        if c != 3:
            raise ValueError(f"Expected 3-channel video: {key}")
        entry = self.meta.episodes[ep]
        query_ts = [float(t) + float(entry[f"videos/{key}/from_timestamp"]) for t in timestamps]
        path = self.root / self.meta.get_video_file_path(ep, key)
        try:
            frames = decode_video_frames(path, query_ts, 0.001, self.base.video_backend,
                                         self.config.video_decode_threads)
            if (frames.shape != (len(timestamps), 3, h, w) or not torch.isfinite(frames).all()
                    or (frames < 0).any() or (frames > 1).any()):
                raise ValueError(f"Bad decoded shape/values for {key}: {tuple(frames.shape)}")
            return frames, torch.ones(len(timestamps), dtype=torch.bool)
        except (OSError, RuntimeError, ValueError, AssertionError) as error:
            raise RuntimeError(f"WSA video decode failed: ep={ep} key={key} path={path}") from error

    def __getitem__(self, index):
        idx = int(self.sample_indices[index])
        segment = self.segments[self.segment_for_sample[index]]
        # Use loaded-table episode offsets, including when the base dataset filters episodes.
        query_indices, padding = LeRobotDataset._get_query_indices(self, idx, segment.episode)
        future_indices = query_indices[self.config.motion_key]
        valid_time = ~padding[f"{self.config.motion_key}_is_pad"]
        future_ts = self.timestamps[future_indices]
        all_ts = np.concatenate(([self.timestamps[idx]], future_ts))
        need_goal = self.include_3d_targets and segment.metadata_valid
        goal_index = segment.last
        history_count = 0
        if self.action_config is not None:
            cfg = self.action_config
            entry = self.meta.episodes[segment.episode]
            first, end = entry["dataset_from_index"], entry["dataset_to_index"]
            history, history_valid = history_indices(idx - first, cfg.num_history_frame, cfg.history_frame_stride)
            history_count = cfg.num_history_frame
            all_ts = np.concatenate((all_ts, self.timestamps[first + history[:-1]]))
            if not segment.metadata_valid:
                need_goal = self.include_3d_targets and self.config.allow_endframe_3d
                goal_index = end - 1
        goal_in_window = need_goal and goal_index <= future_indices[-1]
        if goal_in_window:
            all_ts = np.concatenate((all_ts, [self.timestamps[goal_index]]))
        views, view_valid = zip(*(self._decode(key, all_ts, segment.episode) for key in self.view_keys), strict=True)
        augment_params = None
        if self.augmentation_config is not None:
            shapes = {name: view.shape[-2:] for name, view, valid in
                      zip(CAMERA_WEIGHTS, views, view_valid, strict=True) if valid.any()}
            augment_params = sample_pixel_augmentation(self.augmentation_config, shapes)
            views = self._augment_views(views, view_valid, augment_params)
        window_length = 1 + self.config.future_frames
        canvas, pixels_valid = compose_rgb_views([view[:window_length] for view in views],
                                                 [mask[:window_length] for mask in view_valid],
                                                 self.config.rgb_height, self.config.rgb_width)
        normalized = canvas * 2 - 1
        motion, motion_valid = self._decode(self.config.motion_key, future_ts, segment.episode)
        # self._motion_reads += 1
        # if self.config.motion_log_every > 0 and self._motion_reads % self.config.motion_log_every == 1:
        #     logging.info("[WSA motion] repo=%s ep=%s decoded=[%.6f,%.6f]",self.repo_id, segment.episode, motion.min(), motion.max())
        motion = F.interpolate(motion, size=(self.config.motion_height, self.config.motion_width),
                               mode="bilinear", align_corners=False)
        motion_valid &= valid_time
        motion = motion.masked_fill(~motion_valid[:, None, None, None], 0)
        motion = motion * 2 - 1
        targets = {
            "future_rgb": normalized[1:1 + self.config.future_frames], "future_motion": motion,
            "rgb_valid": pixels_valid[1:1 + self.config.future_frames] & valid_time[:, None, None, None],
            "motion_valid": motion_valid,
        }
        if self.include_3d_targets:
            if goal_in_window:
                goal_views = [view[-1].clone() for view in views]
                goal_valid = [bool(mask[-1]) for mask in view_valid]
            elif need_goal:
                goals = [self._decode(key, [self.timestamps[goal_index]], segment.episode) for key in self.view_keys]
                if augment_params is not None:
                    frames, masks = zip(*goals, strict=True)
                    goals = list(zip(self._augment_views(frames, masks, augment_params), masks, strict=True))
                goal_views = [frames[0].clone() for frames, _ in goals]
                goal_valid = [bool(mask[0]) for _, mask in goals]
            else:
                goal_views = [torch.zeros_like(view[0]) for view in views]
                goal_valid = [False] * 3
            targets.update({
                "goal_rgb_views": goal_views,  # Compact raw [0,1] frames, not views of a video storage.
                "goal_view_valid": torch.tensor(goal_valid),
            })
        condition = build_inference_condition(normalized[0], segment.text, observed_valid=pixels_valid[0])
        if self.action_config is not None:
            if history_count:
                history_canvas, history_pixels = compose_rgb_views(
                    [view[window_length:window_length + history_count] for view in views],
                    [mask[window_length:window_length + history_count] for mask in view_valid],
                    self.config.rgb_height, self.config.rgb_width)
                condition["history_rgb"] = history_canvas * 2 - 1
                condition["history_valid"] = history_pixels & history_valid[:-1, None, None, None]
            condition.update(self._action_sample(idx, segment.episode, targets))
        if self.text_embeddings is not None:
            condition["text_embedding"], condition["text_valid"] = self.text_embeddings.get(segment.text)
        return {
            **condition,
            "targets": targets, "sample_id": f"{self.repo_id}:{segment.episode}:{int(self.frame_indices[idx])}",
            "episode_index": segment.episode, "subtask_id": segment.subtask,
            "start_timestamp": float(self.timestamps[idx]), "goal_timestamp": float(self.timestamps[goal_index]),
            "frame_timestamps": torch.from_numpy(future_ts.copy()), "future_frame_valid": valid_time,
            "metadata_valid": segment.metadata_valid, "sidecar_digest": self.sidecar_digest,
            "prompt_version": PROMPT_VERSION,
        }

    @staticmethod
    def _augment_views(views, masks, params):
        return [apply_pixel_augmentation(frames, params, name).masked_fill(~mask[:, None, None, None], 0)
                if mask.any() else frames
                for name, frames, mask in zip(CAMERA_WEIGHTS, views, masks, strict=True)]

    def _action_sample(self, idx, ep, targets):
        cfg, codec = self.action_config, self.action_codec
        entry = self.meta.episodes[ep]
        requested = idx + np.arange(cfg.action_chunk_size) * cfg.action_frame_stride
        valid = torch.from_numpy(requested < entry["dataset_to_index"])
        indices = np.minimum(requested, entry["dataset_to_index"] - 1)
        action = torch.zeros(cfg.action_chunk_size, cfg.action_dim)
        mask = torch.zeros_like(action, dtype=torch.bool)
        state = {"state": torch.zeros(cfg.state_dim), "state_valid": torch.tensor(False)}
        if codec.has_state:
            keys = codec.state_keys + (codec.action_keys if codec.has_action else [])
            table = self.base.hf_dataset.select_columns(keys).with_format("arrow")[idx:int(indices[-1]) + 1]
            raw_state = {key: table[key][0].as_py() for key in codec.state_keys}
            raw_action = ({key: [table[key][int(i) - idx].as_py() for i in indices] for key in codec.action_keys}
                          if codec.has_action else None)
            if raw_action is not None:
                for step in range(cfg.action_chunk_size):
                    if any(raw_action[key][step] is None for key in codec.action_keys):
                        valid[step] = False
                        for key, size in zip(codec.action_keys, codec.action_sizes, strict=True):
                            raw_action[key][step] = [0.0] * size  # Explicitly masked missing label, never supervision.
            state = codec.encode(raw_state, raw_action)
            if codec.has_action:
                action = state.pop("action")
                dims = _reorder(torch.ones(sum(codec.action_sizes)), codec.schema.action_reorder, cfg.action_dim).bool()
                mask = valid[:, None] & dims[None]
        targets.update(action=action, action_valid_mask=mask)
        return state


def wsa1_5_collate(samples):
    """Native teacher view shapes may differ between cameras and datasets."""
    if not samples:
        raise ValueError("Cannot collate an empty WSA batch")
    target_keys = set(samples[0]["targets"])
    if any(set(sample["targets"]) != target_keys for sample in samples):
        raise ValueError("Cannot collate samples with different 3D supervision settings")
    out = default_collate([{k: v for k, v in sample.items() if k not in {"targets", "repo_id", "dataset_index"}}
                           for sample in samples])
    out["targets"] = default_collate([{k: v for k, v in sample["targets"].items() if k != "goal_rgb_views"}
                                      for sample in samples])
    if "goal_rgb_views" in target_keys:
        out["targets"]["goal_rgb_views"] = [sample["targets"]["goal_rgb_views"] for sample in samples]
    return out
