"""A thin, target-isolating adapter over the existing LeRobot V3 reader.

All timestamps come from the source frame table. Missing media is accompanied
by masks; decoding failures never become unmarked ground truth.
"""
from __future__ import annotations

import hashlib
import json
import logging
import operator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate

from lerobot.dataset_schemas import get_registry, get_schema
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.video_utils import decode_video_frames

from .configuration_wsa1_5 import WSA1_5DatasetConfig
from .text_embeddings import PROMPT_VERSION, TextEmbeddingCache

HAND_NAMES = {0: "left hand", 1: "right hand", 2: "both hands"}


def compose_rgb_views(views, valid, height=384, width=320):
    """Official WSA robotwin layout: top 2/3, then left/right bottom 1/3."""
    if len(views) != 3 or len(valid) != 3 or height % 3 or width % 2:
        raise ValueError("Expected three ordered views and a valid WSA canvas size")
    sizes = [(height * 2 // 3, width), (height // 3, width // 2), (height // 3, width // 2)]
    patches, masks = [], []
    for view, is_valid, size in zip(views, valid, sizes, strict=True):
        if view.ndim != 4 or view.shape[1] != 3:
            raise ValueError("Each view must have shape [T,3,H,W]")
        patch = F.interpolate(view.float(), size=size, mode="bilinear", align_corners=False)
        mask = torch.as_tensor(is_valid, device=view.device)
        if mask.dtype != torch.bool:
            raise TypeError("RGB view validity must be boolean")
        mask = mask.expand(view.shape[0]).reshape(-1, 1, 1, 1)
        patches.append(patch.masked_fill(~mask, 0))
        masks.append(mask.expand(-1, 1, *size))
    return (torch.cat([patches[0], torch.cat(patches[1:], -1)], -2),
            torch.cat([masks[0], torch.cat(masks[1:], -1)], -2))


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


def condition_text(episode, subtask):
    # Only read fields verified in the real sample. Unknown optional fields are empty.
    task = episode.get("task_instruction", "") or ""
    instruction = subtask.get("subtask_text", "") or ""
    hand = HAND_NAMES.get(subtask.get("hand_used"), "")
    scene = episode.get("scene", "") or ""
    return f"You are a world modeling agent observing an interacting through multi-view video observations.:Task: {task}\nSubtask: {instruction}\nHand: {hand}\nScene: {scene}"


class LoadedEpisodeMetadata:
    """Expose loaded-sample offsets to existing mixing/sampling code.

    Video path methods stay bound to the original LeRobot metadata, preserving
    global episode IDs and chunk offsets after an episodes=[...] filter.
    """
    def __init__(self, source, episode_indices):
        self.source = source
        self.entries = {}
        cursor = 0
        for ep in dict.fromkeys(episode_indices.tolist()):
            count = int((episode_indices == ep).sum())
            self.entries[int(ep)] = {**source.episodes[int(ep)], "dataset_from_index": cursor,
                                     "dataset_to_index": cursor + count}
            cursor += count
        self.total_frames = cursor
        self.total_episodes = len(self.entries)
        self.episodes = self

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "source"), name)

    def __getitem__(self, key):
        if isinstance(key, str):
            return [entry[key] for entry in self.entries.values()]
        return self.entries[int(key)]


class WSA1_5Dataset(Dataset):
    def __init__(self, base, config: WSA1_5DatasetConfig, *, include_3d_targets=True, text_embedding_shape=None):
        self.base, self.config = base, config
        if type(include_3d_targets) is not bool:
            raise TypeError("include_3d_targets must be boolean")
        self.include_3d_targets = include_3d_targets
        self.repo_id = base.repo_id
        self.meta = base.meta
        self.root = Path(base.root)
        base._ensure_hf_dataset_loaded()
        self.timestamps = np.asarray([float(v) for v in base.hf_dataset["timestamp"]], dtype=np.float64)
        if not np.isfinite(self.timestamps).all():
            raise ValueError("Dataset timestamps must be finite")
        self.frame_indices = np.asarray([operator.index(v) for v in base.hf_dataset["frame_index"]], dtype=np.int64)
        self.episode_indices = np.asarray([operator.index(v) for v in base.hf_dataset["episode_index"]], dtype=np.int64)
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
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8")) if sidecar_path.exists() else []
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
        self.sidecar_digest = hashlib.sha256(sidecar_path.read_bytes() if sidecar_path.exists() else b"").hexdigest()
        self.segments = []
        self.sample_indices = []
        self.segment_for_sample = []
        self._build_segments(annotations)
        self.text_embeddings = None
        if text_embedding_shape is not None:
            self.text_embeddings = TextEmbeddingCache(self.root / config.text_embedding_dir, *text_embedding_shape)
            self.text_embeddings.require(segment.text for segment in self.segments)
        self.meta = LoadedEpisodeMetadata(base.meta, self.episode_indices)
        self.delta_indices = {config.motion_key: [i * config.frame_stride for i in range(1, config.future_frames + 1)]}
        self._motion_reads = 0
        logging.info("[WSA data] repo=%s samples=%d views=%s sidecar_sha256=%s",
                     self.repo_id, len(self), self.view_keys, self.sidecar_digest)

    def _build_segments(self, annotations):
        for ep in dict.fromkeys(self.episode_indices.tolist()):
            positions = np.flatnonzero(self.episode_indices == ep)
            if not np.array_equal(positions, np.arange(positions[0], positions[-1] + 1)):
                raise ValueError(f"Episode {ep} rows are not contiguous")
            frames = self.frame_indices[positions]
            if not np.array_equal(frames, np.arange(len(frames))):
                raise ValueError(f"Episode {ep} must contain contiguous frame indices starting at zero")
            if np.any(np.diff(self.timestamps[positions]) <= 0):
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
                self._add_segment(Segment(int(ep), int(positions[first]), int(positions[last]),
                                          sid, condition_text(annotation, subtask), True))
            # Missing labels: blank condition, bounded interval, no endpoint/3D supervision.
            missing = np.flatnonzero(~occupied)
            for run in np.split(missing, np.flatnonzero(np.diff(missing) > 1) + 1):
                if len(run):
                    self._add_segment(Segment(int(ep), int(positions[run[0]]), int(positions[run[-1]]),
                                              -1, "", False))

    def _add_segment(self, segment):
        sid = len(self.segments)
        self.segments.append(segment)
        self.sample_indices.extend(range(segment.first, segment.last + 1))
        self.segment_for_sample.extend([sid] * (segment.last - segment.first + 1))

    @property
    def num_frames(self):
        return len(self.sample_indices)

    @property
    def num_episodes(self):
        return len(np.unique(self.episode_indices))

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
            frames = decode_video_frames(path, query_ts, 0.001, self.base.video_backend)
            if (frames.shape != (len(timestamps), 3, h, w) or not torch.isfinite(frames).all()
                    or (frames < 0).any() or (frames > 1).any()):
                raise ValueError(f"Bad decoded shape/values for {key}: {tuple(frames.shape)}")
            return frames, torch.ones(len(timestamps), dtype=torch.bool)
        except (OSError, RuntimeError, ValueError, AssertionError) as error:
            raise RuntimeError(f"WSA video decode failed: ep={ep} key={key} path={path}") from error

    def __getitem__(self, index):
        idx = self.sample_indices[index]
        segment = self.segments[self.segment_for_sample[index]]
        # Use loaded-table episode offsets, including when the base dataset filters episodes.
        query_indices, padding = LeRobotDataset._get_query_indices(self, idx, segment.episode)
        future_indices = query_indices[self.config.motion_key]
        valid_time = ~padding[f"{self.config.motion_key}_is_pad"]
        future_ts = self.timestamps[future_indices]
        all_ts = np.concatenate(([self.timestamps[idx]], future_ts))
        if self.include_3d_targets:
            all_ts = np.concatenate((all_ts, [self.timestamps[segment.last]]))
        views, view_valid = zip(*(self._decode(key, all_ts, segment.episode) for key in self.view_keys), strict=True)
        canvas, pixels_valid = compose_rgb_views(views, view_valid, self.config.rgb_height, self.config.rgb_width)
        normalized = (canvas * 2 - 1).masked_fill(~pixels_valid, 0)
        motion, motion_valid = self._decode(self.config.motion_key, future_ts, segment.episode)
        # self._motion_reads += 1
        # if self.config.motion_log_every > 0 and self._motion_reads % self.config.motion_log_every == 1:
        #     logging.info("[WSA motion] repo=%s ep=%s decoded=[%.6f,%.6f]",self.repo_id, segment.episode, motion.min(), motion.max())
        motion = F.interpolate(motion, size=(self.config.motion_height, self.config.motion_width),
                               mode="bilinear", align_corners=False)
        motion_valid &= valid_time
        motion = (motion * 2 - 1).masked_fill(~motion_valid[:, None, None, None], 0)
        targets = {
            "future_rgb": normalized[1:1 + self.config.future_frames], "future_motion": motion,
            "rgb_valid": pixels_valid[1:1 + self.config.future_frames] & valid_time[:, None, None, None],
            "motion_valid": motion_valid,
        }
        if self.include_3d_targets:
            targets.update({
                "goal_rgb_views": [view[-1] for view in views],  # Raw [0,1], native sizes.
                "goal_view_valid": torch.tensor([bool(mask[-1]) and segment.metadata_valid for mask in view_valid]),
            })
        condition = build_inference_condition(normalized[0], segment.text, observed_valid=pixels_valid[0])
        if self.text_embeddings is not None:
            condition["text_embedding"], condition["text_valid"] = self.text_embeddings.get(segment.text)
        return {
            **condition,
            "targets": targets, "sample_id": f"{self.repo_id}:{segment.episode}:{int(self.frame_indices[idx])}",
            "episode_index": segment.episode, "subtask_id": segment.subtask,
            "start_timestamp": float(self.timestamps[idx]), "goal_timestamp": float(self.timestamps[segment.last]),
            "frame_timestamps": torch.from_numpy(future_ts.copy()), "future_frame_valid": valid_time,
            "metadata_valid": segment.metadata_valid, "sidecar_digest": self.sidecar_digest,
            "prompt_version": PROMPT_VERSION,
        }


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
