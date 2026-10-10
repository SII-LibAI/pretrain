"""Dataset adapter for frame-aligned synthetic embodiments."""

from __future__ import annotations

import json
import os
from collections import OrderedDict
from collections.abc import Sequence
from copy import copy
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset

from lerobot.utils.constants import ACTION, OBS_STATE

if TYPE_CHECKING:
    from lerobot.transforms.core import DataTransformFn

DEFAULT_ANNOTATION_FILE = "meta/annotations/action.parquet"
HUMAN_EEF_EMBODIMENT = "human_eef"
HUMAN_EEF_COMPONENTS = (
    ("human_eef.right.pos", 3),
    ("human_eef.right.rot", 4),
    ("human_eef.right.grasp", 1),
    ("human_eef.left.pos", 3),
    ("human_eef.left.rot", 4),
    ("human_eef.left.grasp", 1),
)
HUMAN_EEF_VALID_KEYS = ("human_eef.right.valid", "human_eef.left.valid")
EMBODIMENT_SCHEMA_PREFIXES = {
    "egodex_v": "egodex_synthetic_",
    "egoverse_human_bimanual": "egoverse_synthetic_",
}


def embodiment_schema_prefix(source_robot_type: str) -> str:
    return EMBODIMENT_SCHEMA_PREFIXES[source_robot_type]


@dataclass(frozen=True)
class EmbodimentSpec:
    """Schema of one synthetic embodiment in an annotation parquet."""

    name: str
    action_key: str
    valid_key: str
    dim: int
    component_keys: tuple[str, ...] = ()
    component_valid_keys: tuple[str, ...] = ()

    @property
    def value_keys(self) -> tuple[str, ...]:
        return self.component_keys or (self.action_key,)

    @property
    def valid_keys(self) -> tuple[str, ...]:
        return self.component_valid_keys or (self.valid_key,)


@dataclass(frozen=True)
class EmbodimentStats:
    """Mean/std statistics used by the adapter for one embodiment."""

    state_mean: np.ndarray
    state_std: np.ndarray
    action_mean: np.ndarray
    action_std: np.ndarray


class SyntheticActionTable:
    """Row-group-backed synthetic annotations with bounded per-process caching."""

    def __init__(
        self,
        path: Path,
        specs: Sequence[EmbodimentSpec],
        row_group_offsets: np.ndarray,
        *,
        row_group_cache_size: int = 0,
    ) -> None:
        self.path = path
        self.specs = tuple(specs)
        self.row_group_offsets = row_group_offsets
        self.num_rows = int(row_group_offsets[-1])
        self.row_group_cache_size = row_group_cache_size
        self._reader: pq.ParquetFile | None = None
        self._reader_pid: int | None = None
        self._value_cache: OrderedDict[tuple[int, int], np.ndarray] = OrderedDict()
        self._eager_data: dict[str, Any] | None = None

    @classmethod
    def from_parquet(
        cls,
        path: str | Path,
        embodiments: Sequence[str] | None = None,
        *,
        row_group_cache_size: int = 0,
    ) -> SyntheticActionTable:
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Synthetic action annotations not found: {path}")

        parquet = pq.ParquetFile(path)
        specs = discover_embodiments(parquet, embodiments)
        if row_group_cache_size < 0:
            raise ValueError("row_group_cache_size must be non-negative")
        row_group_offsets = np.zeros(parquet.num_row_groups + 1, dtype=np.int64)
        for row_group in range(parquet.num_row_groups):
            row_group_offsets[row_group + 1] = (
                row_group_offsets[row_group] + parquet.metadata.row_group(row_group).num_rows
            )
        return cls(
            path,
            specs,
            row_group_offsets,
            row_group_cache_size=row_group_cache_size,
        )

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(spec.name for spec in self.specs)

    def _load_eager_data(self) -> dict[str, Any]:
        """Compatibility path for the offline stats tool; training does not call this."""
        if self._eager_data is not None:
            return self._eager_data
        columns = ["index", "episode_index", "frame_index"]
        columns.extend(key for spec in self.specs for key in spec.value_keys)
        columns.extend(key for spec in self.specs for key in spec.valid_keys)
        table = self._parquet().read(columns=list(dict.fromkeys(columns)))
        values = []
        valid = []
        for spec in self.specs:
            components = [_feature_column_to_numpy(table[key]) for key in spec.value_keys]
            body_values = np.concatenate(components, axis=-1)
            if body_values.shape[1] != spec.dim:
                raise ValueError(
                    f"{self.path}: concatenated values for {spec.name!r} have dimension "
                    f"{body_values.shape[1]}, expected {spec.dim}"
                )
            body_valid = np.logical_and.reduce(
                [_scalar_column_to_numpy(table[key], np.bool_) for key in spec.valid_keys]
            )
            if not np.isfinite(body_values[body_valid]).all():
                raise ValueError(f"{self.path}: valid values for {spec.name!r} are not finite")
            values.append(body_values)
            valid.append(body_valid)
        self._eager_data = {
            "indices": _scalar_column_to_numpy(table["index"], np.int64),
            "episode_indices": _scalar_column_to_numpy(table["episode_index"], np.int64),
            "frame_indices": _scalar_column_to_numpy(table["frame_index"], np.int64),
            "values": tuple(values),
            "valid": tuple(valid),
        }
        _validate_annotation_rows(
            self._eager_data["indices"],
            self._eager_data["episode_indices"],
            self._eager_data["frame_indices"],
            self.path,
        )
        return self._eager_data

    @property
    def indices(self) -> np.ndarray:
        return self._load_eager_data()["indices"]

    @property
    def episode_indices(self) -> np.ndarray:
        return self._load_eager_data()["episode_indices"]

    @property
    def frame_indices(self) -> np.ndarray:
        return self._load_eager_data()["frame_indices"]

    @property
    def values(self) -> tuple[np.ndarray, ...]:
        return self._load_eager_data()["values"]

    @property
    def valid(self) -> tuple[np.ndarray, ...]:
        return self._load_eager_data()["valid"]

    def __len__(self) -> int:
        return self.num_rows

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_reader"] = None
        state["_reader_pid"] = None
        state["_value_cache"] = OrderedDict()
        state["_eager_data"] = None
        return state

    def _parquet(self) -> pq.ParquetFile:
        pid = os.getpid()
        if self._reader is None or self._reader_pid != pid:
            self._reader = pq.ParquetFile(self.path)
            self._reader_pid = pid
            self._value_cache.clear()
        return self._reader

    def _read_body_row_group(self, body_id: int, row_group: int) -> np.ndarray:
        cache_key = (body_id, row_group)
        cached = self._value_cache.get(cache_key)
        if cached is not None:
            self._value_cache.move_to_end(cache_key)
            return cached

        spec = self.specs[body_id]
        table = self._parquet().read_row_group(row_group, columns=list(spec.value_keys))
        components = [_feature_column_to_numpy(table[key]) for key in spec.value_keys]
        values = np.concatenate(components, axis=-1)
        if values.shape[1] != spec.dim:
            raise ValueError(
                f"{self.path}: concatenated values for {spec.name!r} have dimension "
                f"{values.shape[1]}, expected {spec.dim}"
            )
        if self.row_group_cache_size > 0:
            self._value_cache[cache_key] = values
            self._value_cache.move_to_end(cache_key)
            while len(self._value_cache) > self.row_group_cache_size:
                self._value_cache.popitem(last=False)
        return values

    def read_values(self, body_id: int, start: int, stop: int) -> np.ndarray:
        """Read ``[start:stop]`` without materializing the complete parquet column."""
        if start < 0 or stop < start or stop > self.num_rows:
            raise IndexError(f"Invalid annotation slice [{start}:{stop}] for {self.num_rows} rows")
        if start == stop:
            return np.empty((0, self.specs[body_id].dim), dtype=np.float32)

        pieces = []
        row_group = int(np.searchsorted(self.row_group_offsets, start, side="right") - 1)
        cursor = start
        while cursor < stop:
            group_start = int(self.row_group_offsets[row_group])
            group_stop = int(self.row_group_offsets[row_group + 1])
            values = self._read_body_row_group(body_id, row_group)
            local_start = cursor - group_start
            local_stop = min(stop, group_stop) - group_start
            pieces.append(values[local_start:local_stop])
            cursor = group_start + local_stop
            row_group += 1
        result = pieces[0] if len(pieces) == 1 else np.concatenate(pieces, axis=0)
        if not np.isfinite(result).all():
            raise ValueError(
                f"{self.path}: non-finite values for {self.specs[body_id].name!r} "
                f"in rows [{start}:{stop}]"
            )
        return result


class SyntheticDatasetMetadataView:
    """Expose episode ranges after invalid anchors have been removed."""

    def __init__(
        self,
        base_meta,
        episode_indices: np.ndarray | None = None,
        *,
        episodes: dict[str, list[int]] | None = None,
        total_frames: int | None = None,
        robot_type: str | None = None,
        stats: dict[str, dict[str, np.ndarray]] | None = None,
    ) -> None:
        self._base_meta = base_meta
        self.robot_type = robot_type or base_meta.robot_type
        if stats is not None:
            self.stats = stats
        if episodes is None:
            if episode_indices is None:
                raise ValueError("episode_indices or episodes must be provided")
            boundaries = np.flatnonzero(np.diff(episode_indices) != 0) + 1
            starts = np.concatenate(([0], boundaries)).tolist()
            ends = np.concatenate((boundaries, [len(episode_indices)])).tolist()
            episodes = {
                "dataset_from_index": starts,
                "dataset_to_index": ends,
            }
            total_frames = len(episode_indices)
        elif total_frames is None:
            raise ValueError("total_frames is required with precomputed episodes")
        self.episodes = episodes
        self.total_frames = int(total_frames)
        self.total_episodes = len(episodes["dataset_from_index"])

    def __getattr__(self, name: str) -> Any:
        return getattr(self._base_meta, name)


class SyntheticDatasetHydrationView:
    """Minimal dataset view used to hydrate one embodiment's transforms."""

    def __init__(self, dataset: SyntheticEmbodimentDataset, meta: SyntheticDatasetMetadataView) -> None:
        self.root = dataset.root
        self.repo_id = dataset.repo_id
        self.meta = meta


class SyntheticEmbodimentDataset(Dataset):
    """Adapt frame-aligned robot poses into state and future action chunks.

    A sample at source frame ``t`` contains state ``q[t]`` and action chunk
    ``q[t + 1 : t + chunk_size + 1]``. Only embodiment/frame pairs for which
    the complete interval is valid are eligible. This is the dataset-side
    equivalent of a temporal loss mask for policies that do not accept one.
    """

    def __init__(
        self,
        base_dataset,
        *,
        annotation_file: str | Path = DEFAULT_ANNOTATION_FILE,
        stats_dir: str | Path,
        chunk_size: int,
        max_state_dim: int,
        max_action_dim: int,
        embodiments: Sequence[str] | None = None,
        row_group_cache_size: int = 0,
        transforms: Sequence[DataTransformFn] | None = None,
    ) -> None:
        if chunk_size < 1:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")

        self.base_dataset = base_dataset
        self.repo_id = base_dataset.repo_id
        self.root = Path(base_dataset.root)
        self.fps = base_dataset.fps
        self.chunk_size = chunk_size
        self.max_state_dim = max_state_dim
        self.max_action_dim = max_action_dim
        annotation_path = Path(annotation_file)
        if not annotation_path.is_absolute():
            annotation_path = self.root / annotation_path
        self.annotations = SyntheticActionTable.from_parquet(
            annotation_path,
            embodiments,
            row_group_cache_size=row_group_cache_size,
        )
        schema_prefix = embodiment_schema_prefix(base_dataset.meta.robot_type)
        self.embodiment_schemas = load_embodiment_schemas(self.annotations.specs, schema_prefix)
        self.embodiment_stats = load_embodiment_stats(stats_dir, self.annotations.specs)

        required_state_dim = max(
            _reorder_output_dim(schema.state_reorder, spec.dim)
            for spec, schema in zip(self.annotations.specs, self.embodiment_schemas)
        )
        required_action_dim = max(
            _reorder_output_dim(schema.action_reorder, spec.dim)
            for spec, schema in zip(self.annotations.specs, self.embodiment_schemas)
        )
        if required_state_dim > max_state_dim or required_action_dim > max_action_dim:
            raise ValueError(
                "Synthetic embodiment dimensions exceed the configured policy dimensions: "
                f"required state/action={required_state_dim}/{required_action_dim}, "
                f"configured state/action={max_state_dim}/{max_action_dim}"
            )

        self._rows, self._candidate_masks, episodes = _build_streaming_index(
            self.annotations,
            chunk_size,
            base_dataset=base_dataset,
        )
        if len(self._rows) == 0:
            raise ValueError("No samples have a fully valid synthetic action chunk")
        self.meta = SyntheticDatasetMetadataView(
            base_dataset.meta,
            episodes=episodes,
            total_frames=len(self._rows),
        )

        unique_masks = np.unique(self._candidate_masks)
        self._candidates_by_mask = {
            int(mask): tuple(i for i in range(len(self.annotations.specs)) if int(mask) & (1 << i))
            for mask in unique_masks
        }

        source_transforms = list(transforms or [])
        if source_transforms:
            from lerobot.transforms.core import (
                DeltaActionTransformFn,
                NormalizeTransformFn,
                PadStateAndActionTransformFn,
                ReorderStateActionTransform,
                compose,
            )

            chat_transform_type = None
            if any(type(t).__name__ == "InternVLAA15ChatProcessorTransformFn" for t in source_transforms):
                from lerobot.policies.internvla_a1_5.transform_internvla_a1_5 import (
                    InternVLAA15ChatProcessorTransformFn,
                )

                chat_transform_type = InternVLAA15ChatProcessorTransformFn

            pose_types = (
                DeltaActionTransformFn,
                NormalizeTransformFn,
                PadStateAndActionTransformFn,
                ReorderStateActionTransform,
            )
            pose_sources = []
            for transform_type in pose_types:
                matches = [t for t in source_transforms if isinstance(t, transform_type)]
                if len(matches) != 1:
                    raise ValueError(f"Expected one {transform_type.__name__} in dataset transforms")
                pose_sources.append(matches[0])
            remaining_transforms = [
                t for t in source_transforms if not isinstance(t, pose_types)
            ]
            self._pose_transforms = []
            self._transforms = []
            for spec, schema in zip(self.annotations.specs, self.embodiment_schemas):
                stats = self.embodiment_stats[spec.name]
                body_meta = SyntheticDatasetMetadataView(
                    base_dataset.meta,
                    episodes=self.meta.episodes,
                    total_frames=len(self),
                    robot_type=schema.robot_type,
                    stats={
                        OBS_STATE: {"mean": stats.state_mean, "std": stats.state_std},
                        ACTION: {"mean": stats.action_mean, "std": stats.action_std},
                    },
                )
                view = SyntheticDatasetHydrationView(self, body_meta)
                delta, normalize, pad, reorder = pose_sources
                self._pose_transforms.append(
                    compose(
                        [
                            delta.hydrate(view),
                            normalize.hydrate(view),
                            replace(
                                pad,
                                max_state_dim=max_state_dim,
                                max_action_dim=max_action_dim,
                            ),
                            reorder.hydrate(view),
                        ]
                    )
                )
                hydrated = []
                for transform in remaining_transforms:
                    if chat_transform_type is not None and isinstance(transform, chat_transform_type):
                        body_transform = copy(transform)
                        body_transform.action_mode = schema.action_mode
                    else:
                        body_transform = transform.hydrate(view)
                    hydrated.append(body_transform)
                self._transforms.append(compose(hydrated))
        else:
            self._pose_transforms = [_identity] * len(self.annotations.specs)
            self._transforms = [_identity] * len(self.annotations.specs)

    @property
    def num_frames(self) -> int:
        return len(self)

    @property
    def num_episodes(self) -> int:
        return self.meta.total_episodes

    @property
    def checkpoint_stats(self) -> dict[str, Any]:
        return {
            "synthetic_embodiments": {
                spec.name: {
                    OBS_STATE: _stats_to_json_dict(self.embodiment_stats[spec.name], state=True),
                    ACTION: _stats_to_json_dict(self.embodiment_stats[spec.name], state=False),
                }
                for spec in self.annotations.specs
            }
        }

    def candidate_embodiments(self, idx: int) -> tuple[str, ...]:
        """Return the valid embodiment choices for a public dataset index."""
        candidate_ids = self._candidates_by_mask[int(self._candidate_masks[idx])]
        return tuple(self.annotations.specs[body_id].name for body_id in candidate_ids)

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if idx < 0:
            idx += len(self)
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)

        base_row = int(self._rows[idx])
        annotation_row = base_row
        mask = int(self._candidate_masks[idx])
        candidate_ids = self._candidates_by_mask[mask]
        body_id = candidate_ids[torch.randint(len(candidate_ids), ()).item()]
        spec = self.annotations.specs[body_id]
        schema = self.embodiment_schemas[body_id]
        poses = self.annotations.read_values(
            body_id,
            annotation_row,
            annotation_row + self.chunk_size + 1,
        )

        sample = self.base_dataset[base_row]
        sample[OBS_STATE] = torch.from_numpy(poses[0].copy())
        sample[ACTION] = torch.from_numpy(poses[1:].copy())
        sample = self._pose_transforms[body_id](sample)
        layout = ", ".join(
            f"{block['name']}[{block['size']}]" for block in schema.action_layout
        )
        sample["task"] = (
            f"Current robot embodiment: {spec.name}; task: {sample['task']}; "
            # f"action layout: {layout}"
        )
        sample["synthetic_embodiment"] = spec.name
        sample = self._transforms[body_id](sample)
        if isinstance(sample, dict):
            sample["repo_id"] = self.repo_id
        return sample


def discover_embodiments(
    parquet: pq.ParquetFile,
    embodiments: Sequence[str] | None = None,
) -> tuple[EmbodimentSpec, ...]:
    """Discover ``action.<name>`` and paired ``_valid`` fields."""
    schema = parquet.schema_arrow
    field_names = set(schema.names)
    discovered = sorted(
        name.removeprefix("action.").removesuffix("_valid")
        for name in field_names
        if name.startswith("action.") and name.endswith("_valid")
    )
    eef_columns = {key for key, _ in HUMAN_EEF_COMPONENTS} | set(HUMAN_EEF_VALID_KEYS)
    present_eef_columns = eef_columns & field_names
    if present_eef_columns and present_eef_columns != eef_columns:
        missing = sorted(eef_columns - field_names)
        raise KeyError(f"Incomplete {HUMAN_EEF_EMBODIMENT!r} annotation; missing columns: {missing}")
    if present_eef_columns:
        discovered.append(HUMAN_EEF_EMBODIMENT)
    selected = list(embodiments) if embodiments is not None else discovered
    if not selected:
        raise ValueError("No synthetic embodiments were selected")
    if len(set(selected)) != len(selected):
        raise ValueError("Synthetic embodiment selection contains duplicates")

    specs: list[EmbodimentSpec] = []
    for name in selected:
        if name == HUMAN_EEF_EMBODIMENT:
            missing = sorted(eef_columns - field_names)
            if missing:
                raise KeyError(f"Missing annotation columns for {name!r}: {missing}")
            for key, expected_dim in HUMAN_EEF_COMPONENTS:
                dim = _parquet_feature_dim(parquet, key)
                if dim != expected_dim:
                    raise ValueError(f"{key} has dimension {dim}, expected {expected_dim}")
            for key in HUMAN_EEF_VALID_KEYS:
                if not pa.types.is_boolean(schema.field(key).type):
                    raise TypeError(f"{key} must be a boolean column")
            specs.append(
                EmbodimentSpec(
                    name=HUMAN_EEF_EMBODIMENT,
                    action_key=HUMAN_EEF_EMBODIMENT,
                    valid_key=HUMAN_EEF_VALID_KEYS[0],
                    dim=sum(dim for _, dim in HUMAN_EEF_COMPONENTS),
                    component_keys=tuple(key for key, _ in HUMAN_EEF_COMPONENTS),
                    component_valid_keys=HUMAN_EEF_VALID_KEYS,
                )
            )
            continue
        action_key = f"action.{name}"
        valid_key = f"{action_key}_valid"
        if action_key not in field_names or valid_key not in field_names:
            raise KeyError(f"Missing annotation columns for embodiment {name!r}")
        if not pa.types.is_boolean(schema.field(valid_key).type):
            raise TypeError(f"{valid_key} must be a boolean column")
        dim = _parquet_vector_dim(parquet, action_key)
        specs.append(EmbodimentSpec(name, action_key, valid_key, dim))
    return tuple(specs)


def load_embodiment_stats(
    stats_dir: str | Path,
    specs: Sequence[EmbodimentSpec],
) -> dict[str, EmbodimentStats]:
    """Load per-embodiment files produced by the offline statistics tool."""
    stats_dir = Path(stats_dir)
    loaded: dict[str, EmbodimentStats] = {}
    for spec in specs:
        path = stats_dir / spec.name / "stats.json"
        with path.open(encoding="utf-8") as stream:
            payload = json.load(stream)
        state = payload[OBS_STATE]
        action = payload[ACTION]
        body_stats = EmbodimentStats(
            state_mean=np.asarray(state["mean"], dtype=np.float32),
            state_std=np.asarray(state["std"], dtype=np.float32),
            action_mean=np.asarray(action["mean"], dtype=np.float32),
            action_std=np.asarray(action["std"], dtype=np.float32),
        )
        for field_name, values in (
            ("observation.state.mean", body_stats.state_mean),
            ("observation.state.std", body_stats.state_std),
            ("action.mean", body_stats.action_mean),
            ("action.std", body_stats.action_std),
        ):
            if values.shape != (spec.dim,):
                raise ValueError(
                    f"{path}: {field_name} has shape {values.shape}, expected ({spec.dim},)"
                )
        loaded[spec.name] = body_stats
    return loaded


def load_embodiment_schemas(
    specs: Sequence[EmbodimentSpec],
    schema_prefix: str,
) -> tuple[Any, ...]:
    """Load exact per-embodiment schemas and validate their vector layouts."""
    from lerobot.dataset_schemas import get_registry

    registry = get_registry()
    available = set(registry.list_available())
    schemas = []
    for spec in specs:
        schema_name = f"{schema_prefix}{spec.name}"
        if schema_name not in available:
            raise KeyError(f"No dataset schema registered for synthetic embodiment {spec.name!r}")
        schema = registry.get(schema_name)
        if schema.state_dim != spec.dim or schema.action_dim != spec.dim:
            raise ValueError(
                f"Schema {schema_name!r} declares state/action dimensions "
                f"{schema.state_dim}/{schema.action_dim}, annotation has {spec.dim}"
            )
        layout_dim = sum(int(block["size"]) for block in schema.action_layout)
        if layout_dim != spec.dim:
            raise ValueError(
                f"Schema {schema_name!r} action_layout totals {layout_dim}, expected {spec.dim}"
            )
        _validate_reorder(schema.action_reorder, spec, "action_reorder")
        _validate_reorder(schema.state_reorder, spec, "state_reorder")
        _schema_delta_mask(schema, spec)
        schemas.append(schema)
    return tuple(schemas)


def _candidate_mask_dtype(num_embodiments: int) -> np.dtype:
    if num_embodiments <= 8:
        return np.dtype(np.uint8)
    if num_embodiments <= 16:
        return np.dtype(np.uint16)
    if num_embodiments <= 32:
        return np.dtype(np.uint32)
    if num_embodiments <= 63:
        return np.dtype(np.uint64)
    raise ValueError("At most 63 embodiments are supported")


def _dataset_num_rows(dataset) -> int:
    hf_dataset = dataset.hf_dataset
    if isinstance(hf_dataset, dict):
        return len(hf_dataset["index"])
    return len(hf_dataset)


def _scalar_values_to_numpy(values: Sequence[Any], count: int) -> np.ndarray:
    return np.fromiter(
        (int(value.item()) if isinstance(value, torch.Tensor) else int(value) for value in values),
        dtype=np.int64,
        count=count,
    )


def _dataset_alignment_slice(dataset, start: int, stop: int) -> dict[str, np.ndarray]:
    keys = ("index", "episode_index", "frame_index")
    hf_dataset = dataset.hf_dataset
    if isinstance(hf_dataset, dict):
        values = {key: hf_dataset[key][start:stop] for key in keys}
    else:
        # Restrict the Arrow view before slicing so image/state/action columns
        # are never materialized just to validate annotation alignment.
        values = hf_dataset.select_columns(list(keys))[start:stop]
    return {
        key: _scalar_values_to_numpy(values[key], stop - start)
        for key in keys
    }


def _row_group_valid_matrix(
    table: pa.Table,
    specs: Sequence[EmbodimentSpec],
) -> np.ndarray:
    columns = []
    for spec in specs:
        body_valid = np.logical_and.reduce(
            [_scalar_column_to_numpy(table[key], np.bool_) for key in spec.valid_keys]
        )
        columns.append(body_valid)
    return np.stack(columns, axis=-1)


def _validate_stream_rows(
    indices: np.ndarray,
    episode_indices: np.ndarray,
    frame_indices: np.ndarray,
    path: Path,
    previous: tuple[int, int, int] | None,
) -> tuple[int, int, int]:
    if len(indices) == 0:
        return previous if previous is not None else (0, 0, 0)
    if previous is not None:
        prev_index, prev_episode, prev_frame = previous
        if indices[0] <= prev_index:
            raise ValueError(f"Annotation indices must be strictly increasing: {path}")
        if episode_indices[0] < prev_episode:
            raise ValueError(f"Annotation episode indices must be non-decreasing: {path}")
        if episode_indices[0] == prev_episode and frame_indices[0] != prev_frame + 1:
            raise ValueError(f"Frame indices must be contiguous within every episode: {path}")
    if len(indices) > 1:
        if (np.diff(indices) <= 0).any():
            raise ValueError(f"Annotation indices must be strictly increasing: {path}")
        episode_diff = np.diff(episode_indices)
        if (episode_diff < 0).any():
            raise ValueError(f"Annotation episode indices must be non-decreasing: {path}")
        same_episode = episode_diff == 0
        if (np.diff(frame_indices)[same_episode] != 1).any():
            raise ValueError(f"Frame indices must be contiguous within every episode: {path}")
    return int(indices[-1]), int(episode_indices[-1]), int(frame_indices[-1])


def _scan_candidate_masks(
    table: SyntheticActionTable,
    chunk_size: int,
    *,
    base_dataset=None,
) -> np.ndarray:
    """Scan only metadata/valid columns and build a compact per-row candidate mask."""
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    if base_dataset is not None and _dataset_num_rows(base_dataset) != len(table):
        raise ValueError(
            "Row-group synthetic loading requires annotation rows to align one-to-one "
            f"with the base dataset: annotations={len(table)}, base={_dataset_num_rows(base_dataset)}"
        )

    masks = np.zeros(len(table), dtype=_candidate_mask_dtype(len(table.specs)))
    valid_keys = list(dict.fromkeys(key for spec in table.specs for key in spec.valid_keys))
    columns = ["index", "episode_index", "frame_index", *valid_keys]
    tail_episode = np.empty(0, dtype=np.int64)
    tail_frame = np.empty(0, dtype=np.int64)
    tail_valid = np.empty((0, len(table.specs)), dtype=np.bool_)
    previous = None

    parquet = pq.ParquetFile(table.path)
    for row_group in range(parquet.num_row_groups):
        offset = int(table.row_group_offsets[row_group])
        row_group_table = parquet.read_row_group(row_group, columns=columns)
        indices = _scalar_column_to_numpy(row_group_table["index"], np.int64)
        episodes = _scalar_column_to_numpy(row_group_table["episode_index"], np.int64)
        frames = _scalar_column_to_numpy(row_group_table["frame_index"], np.int64)
        previous = _validate_stream_rows(indices, episodes, frames, table.path, previous)

        if base_dataset is not None:
            stop = offset + len(indices)
            base_values = _dataset_alignment_slice(base_dataset, offset, stop)
            for key, annotation_values in (
                ("index", indices),
                ("episode_index", episodes),
                ("frame_index", frames),
            ):
                if not np.array_equal(annotation_values, base_values[key]):
                    raise ValueError(
                        f"Synthetic annotation {key} values do not match the base dataset "
                        f"in rows [{offset}:{stop}]"
                    )

        valid = _row_group_valid_matrix(row_group_table, table.specs)
        combined_episode = np.concatenate((tail_episode, episodes))
        combined_frame = np.concatenate((tail_frame, frames))
        combined_valid = np.concatenate((tail_valid, valid), axis=0)
        num_windows = max(0, len(combined_episode) - chunk_size)
        if num_windows:
            global_start = offset - len(tail_episode)
            same_episode = (
                combined_episode[:num_windows]
                == combined_episode[chunk_size : chunk_size + num_windows]
            )
            contiguous = (
                combined_frame[chunk_size : chunk_size + num_windows]
                - combined_frame[:num_windows]
                == chunk_size
            )
            target = masks[global_start : global_start + num_windows]
            window = chunk_size + 1
            for body_id in range(len(table.specs)):
                invalid = np.logical_not(combined_valid[:, body_id]).astype(np.int32)
                prefix = np.empty(len(invalid) + 1, dtype=np.int32)
                prefix[0] = 0
                np.cumsum(invalid, dtype=np.int32, out=prefix[1:])
                valid_window = prefix[window:] == prefix[:-window]
                eligible = valid_window & same_episode & contiguous
                target[eligible] |= np.array(1 << body_id, dtype=masks.dtype)

        keep = min(chunk_size, len(combined_episode))
        tail_episode = combined_episode[-keep:]
        tail_frame = combined_frame[-keep:]
        tail_valid = combined_valid[-keep:]

    if previous is None:
        raise ValueError(f"Synthetic annotation table is empty: {table.path}")
    return masks


def _episodes_from_candidate_masks(
    table: SyntheticActionTable,
    masks: np.ndarray,
) -> dict[str, list[int]]:
    counts: list[int] = []
    current_episode: int | None = None
    current_count = 0
    parquet = pq.ParquetFile(table.path)
    for row_group in range(parquet.num_row_groups):
        start = int(table.row_group_offsets[row_group])
        stop = int(table.row_group_offsets[row_group + 1])
        local_eligible = masks[start:stop] != 0
        if not local_eligible.any():
            continue
        episode_table = parquet.read_row_group(row_group, columns=["episode_index"])
        episode_values = _scalar_column_to_numpy(episode_table["episode_index"], np.int64)
        eligible_episodes = episode_values[local_eligible]
        boundaries = np.flatnonzero(np.diff(eligible_episodes) != 0) + 1
        run_starts = np.concatenate(([0], boundaries))
        run_stops = np.concatenate((boundaries, [len(eligible_episodes)]))
        for run_start, run_stop in zip(run_starts, run_stops):
            episode = int(eligible_episodes[run_start])
            count = int(run_stop - run_start)
            if current_episode == episode:
                current_count += count
            else:
                if current_episode is not None:
                    counts.append(current_count)
                current_episode = episode
                current_count = count
    if current_episode is not None:
        counts.append(current_count)
    ends = np.cumsum(counts, dtype=np.int64)
    starts = np.concatenate(([0], ends[:-1])) if len(ends) else np.empty(0, dtype=np.int64)
    return {
        "dataset_from_index": starts.tolist(),
        "dataset_to_index": ends.tolist(),
    }


def _build_streaming_index(
    table: SyntheticActionTable,
    chunk_size: int,
    *,
    base_dataset=None,
) -> tuple[np.ndarray, np.ndarray, dict[str, list[int]]]:
    full_masks = _scan_candidate_masks(table, chunk_size, base_dataset=base_dataset)
    episodes = _episodes_from_candidate_masks(table, full_masks)
    eligible_count = int(np.count_nonzero(full_masks))
    row_dtype = np.uint32 if len(table) <= np.iinfo(np.uint32).max else np.uint64
    rows = np.empty(eligible_count, dtype=row_dtype)
    masks = np.empty(eligible_count, dtype=full_masks.dtype)
    cursor = 0
    block_size = 1_000_000
    for start in range(0, len(table), block_size):
        stop = min(start + block_size, len(table))
        local_rows = np.flatnonzero(full_masks[start:stop])
        count = len(local_rows)
        rows[cursor : cursor + count] = local_rows + start
        masks[cursor : cursor + count] = full_masks[start + local_rows]
        cursor += count
    return rows, masks, episodes


def _build_candidate_masks(table: SyntheticActionTable, chunk_size: int) -> np.ndarray:
    if len(table.specs) > 63:
        raise ValueError("At most 63 embodiments are supported")
    masks = np.zeros(len(table.indices), dtype=np.uint64)
    boundaries = np.flatnonzero(np.diff(table.episode_indices) != 0) + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [len(table.indices)]))
    window = chunk_size + 1

    for start, end in zip(starts, ends):
        if end - start < window:
            continue
        for body_id, body_valid in enumerate(table.valid):
            invalid = np.logical_not(body_valid[start:end]).astype(np.int64)
            prefix = np.concatenate(([0], np.cumsum(invalid)))
            full_window_valid = prefix[window:] == prefix[:-window]
            rows = start + np.flatnonzero(full_window_valid)
            masks[rows] |= np.uint64(1) << np.uint64(body_id)
    return masks


def _dataset_indices(dataset) -> np.ndarray:
    return _dataset_scalar_column(dataset, "index")


def _dataset_scalar_column(dataset, key: str) -> np.ndarray:
    values = dataset.hf_dataset[key]
    return np.fromiter(
        (int(value.item()) if isinstance(value, torch.Tensor) else int(value) for value in values),
        dtype=np.int64,
        count=len(values),
    )


def _validate_annotation_rows(
    indices: np.ndarray,
    episode_indices: np.ndarray,
    frame_indices: np.ndarray,
    path: Path,
) -> None:
    if len(indices) == 0:
        raise ValueError(f"Synthetic annotation table is empty: {path}")
    if not (len(indices) == len(episode_indices) == len(frame_indices)):
        raise ValueError(f"Annotation index columns have different lengths: {path}")
    if (np.diff(indices) <= 0).any():
        raise ValueError(f"Annotation indices must be strictly increasing: {path}")
    boundaries = np.flatnonzero(np.diff(episode_indices) != 0) + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [len(indices)]))
    segment_episodes = episode_indices[starts]
    if len(np.unique(segment_episodes)) != len(segment_episodes):
        raise ValueError(f"Episode rows must be contiguous: {path}")
    for start, end in zip(starts, ends):
        if not np.array_equal(np.diff(frame_indices[start:end]), np.ones(end - start - 1, dtype=np.int64)):
            raise ValueError(f"Frame indices must be contiguous within every episode: {path}")


def _parquet_vector_dim(parquet: pq.ParquetFile, column_name: str) -> int:
    field_type = parquet.schema_arrow.field(column_name).type
    if pa.types.is_fixed_size_list(field_type):
        return field_type.list_size
    if not (pa.types.is_list(field_type) or pa.types.is_large_list(field_type)):
        raise TypeError(f"{column_name} must be a list column, got {field_type}")
    first_batch = next(parquet.iter_batches(batch_size=1, columns=[column_name]))
    first_value = first_batch.column(0)[0].as_py()
    if not first_value:
        raise ValueError(f"Cannot determine vector dimension from an empty value in {column_name}")
    return len(first_value)


def _parquet_feature_dim(parquet: pq.ParquetFile, column_name: str) -> int:
    field_type = parquet.schema_arrow.field(column_name).type
    if pa.types.is_floating(field_type) or pa.types.is_integer(field_type):
        return 1
    return _parquet_vector_dim(parquet, column_name)


def _feature_column_to_numpy(column: pa.ChunkedArray) -> np.ndarray:
    if pa.types.is_floating(column.type) or pa.types.is_integer(column.type):
        return _scalar_column_to_numpy(column, np.float32)[:, None]
    return _vector_column_to_numpy(column, expected_dim=None)


def _vector_column_to_numpy(column: pa.ChunkedArray, expected_dim: int | None) -> np.ndarray:
    array = column.combine_chunks()
    if array.null_count:
        raise ValueError("Synthetic annotation vector column contains null values")
    if pa.types.is_fixed_size_list(array.type):
        dim = array.type.list_size
        flat_values = array.values
    elif pa.types.is_list(array.type) or pa.types.is_large_list(array.type):
        offsets = np.asarray(array.offsets.to_numpy(zero_copy_only=False), dtype=np.int64)
        lengths = np.diff(offsets)
        if len(lengths) and not np.all(lengths == lengths[0]):
            raise ValueError("Synthetic annotation column must contain fixed-size vectors")
        dim = int(lengths[0]) if len(lengths) else (expected_dim or 0)
        flat_values = array.values.slice(int(offsets[0]), int(offsets[-1] - offsets[0]))
    else:
        raise TypeError(f"Synthetic annotation values must be a list column, got {array.type}")

    if flat_values.null_count:
        raise ValueError("Synthetic annotation vectors contain null elements")
    if expected_dim is not None and dim != expected_dim:
        raise ValueError(f"Synthetic annotation dimension is {dim}, expected {expected_dim}")
    values = np.asarray(flat_values.to_numpy(zero_copy_only=False), dtype=np.float32)
    return values.reshape(len(array), dim)


def _scalar_column_to_numpy(column: pa.ChunkedArray, dtype: np.dtype) -> np.ndarray:
    array = column.combine_chunks()
    if array.null_count:
        raise ValueError("Synthetic annotation scalar column contains null values")
    return np.asarray(array.to_numpy(zero_copy_only=False), dtype=dtype)


def _schema_delta_mask(schema: Any, spec: EmbodimentSpec) -> np.ndarray:
    mask = schema.action_mask.detach().cpu().numpy().astype(np.bool_)
    if mask.shape != (spec.dim,):
        raise ValueError(
            f"Schema {schema.robot_type!r} action_mask has dimension {len(mask)}, "
            f"expected {spec.dim}"
        )
    return mask


def _validate_reorder(
    reorder: Sequence[Sequence[int]] | None,
    spec: EmbodimentSpec,
    field_name: str,
) -> None:
    if reorder is None:
        return
    for item in reorder:
        if len(item) != 4:
            raise ValueError(f"Schema {field_name} entries must contain four indices: {item}")
        src_start, src_end, dst_start, dst_end = item
        if min(item) < 0 or src_end > spec.dim or src_start > src_end or dst_start > dst_end:
            raise ValueError(f"Invalid {field_name} entry for {spec.name!r}: {item}")
        if src_end - src_start != dst_end - dst_start:
            raise ValueError(f"Source/destination lengths differ in {field_name}: {item}")


def _reorder_output_dim(reorder: Sequence[Sequence[int]] | None, source_dim: int) -> int:
    if not reorder:
        return source_dim
    return max(source_dim, max(item[3] for item in reorder))


def _identity(sample: dict[str, Any]) -> dict[str, Any]:
    return sample


def _stats_to_json_dict(stats: EmbodimentStats, *, state: bool) -> dict[str, list[float]]:
    mean = stats.state_mean if state else stats.action_mean
    std = stats.state_std if state else stats.action_std
    return {"mean": mean.tolist(), "std": std.tolist()}
