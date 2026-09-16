"""Dataset adapter for frame-aligned synthetic embodiments."""

from __future__ import annotations

import json
from collections.abc import Sequence
from copy import copy
from dataclasses import dataclass
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

STATS_FORMAT_VERSION = 2
DEFAULT_ANNOTATION_FILE = "meta/annotations/action.parquet"
DEFAULT_EMBODIMENT_SCHEMA_PREFIX = "egodex_synthetic_"
NORMALIZATION_EPS = 1e-6


@dataclass(frozen=True)
class EmbodimentSpec:
    """Schema of one synthetic embodiment in an annotation parquet."""

    name: str
    action_key: str
    valid_key: str
    dim: int


@dataclass(frozen=True)
class EmbodimentStats:
    """Mean/std statistics used by the adapter for one embodiment."""

    state_mean: np.ndarray
    state_std: np.ndarray
    action_mean: np.ndarray
    action_std: np.ndarray


class SyntheticActionTable:
    """In-memory synthetic pose annotations for one LeRobot dataset."""

    def __init__(
        self,
        indices: np.ndarray,
        episode_indices: np.ndarray,
        frame_indices: np.ndarray,
        specs: Sequence[EmbodimentSpec],
        values: Sequence[np.ndarray],
        valid: Sequence[np.ndarray],
    ) -> None:
        self.indices = indices
        self.episode_indices = episode_indices
        self.frame_indices = frame_indices
        self.specs = tuple(specs)
        self.values = tuple(values)
        self.valid = tuple(valid)

    @classmethod
    def from_parquet(
        cls,
        path: str | Path,
        embodiments: Sequence[str] | None = None,
    ) -> SyntheticActionTable:
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Synthetic action annotations not found: {path}")

        parquet = pq.ParquetFile(path)
        specs = discover_embodiments(parquet, embodiments)
        columns = ["index", "episode_index", "frame_index"]
        columns.extend(spec.action_key for spec in specs)
        columns.extend(spec.valid_key for spec in specs)
        table = parquet.read(columns=columns)

        indices = _scalar_column_to_numpy(table["index"], np.int64)
        episode_indices = _scalar_column_to_numpy(table["episode_index"], np.int64)
        frame_indices = _scalar_column_to_numpy(table["frame_index"], np.int64)
        _validate_annotation_rows(indices, episode_indices, frame_indices, path)

        values: list[np.ndarray] = []
        valid: list[np.ndarray] = []
        for spec in specs:
            body_values = _vector_column_to_numpy(table[spec.action_key], spec.dim)
            body_valid = _scalar_column_to_numpy(table[spec.valid_key], np.bool_)
            if not np.isfinite(body_values[body_valid]).all():
                raise ValueError(f"{path}: valid values in {spec.action_key} are not finite")
            values.append(body_values)
            valid.append(body_valid)

        return cls(indices, episode_indices, frame_indices, specs, values, valid)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(spec.name for spec in self.specs)


class SyntheticDatasetMetadataView:
    """Expose episode ranges after invalid anchors have been removed."""

    def __init__(
        self,
        base_meta,
        episode_indices: np.ndarray,
        *,
        robot_type: str | None = None,
    ) -> None:
        self._base_meta = base_meta
        self.robot_type = robot_type or base_meta.robot_type
        boundaries = np.flatnonzero(np.diff(episode_indices) != 0) + 1
        starts = np.concatenate(([0], boundaries)).tolist()
        ends = np.concatenate((boundaries, [len(episode_indices)])).tolist()
        self.episodes = {
            "dataset_from_index": starts,
            "dataset_to_index": ends,
        }
        self.total_frames = len(episode_indices)
        self.total_episodes = len(starts)

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
        schema_prefix: str = DEFAULT_EMBODIMENT_SCHEMA_PREFIX,
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
        self.schema_prefix = schema_prefix

        annotation_path = Path(annotation_file)
        if not annotation_path.is_absolute():
            annotation_path = self.root / annotation_path
        self.annotations = SyntheticActionTable.from_parquet(annotation_path, embodiments)
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

        self._candidate_masks = _build_candidate_masks(self.annotations, chunk_size)
        source_indices = _dataset_indices(base_dataset)
        annotation_rows = np.searchsorted(self.annotations.indices, source_indices)
        aligned = annotation_rows < len(self.annotations.indices)
        safe_rows = np.minimum(annotation_rows, len(self.annotations.indices) - 1)
        aligned &= self.annotations.indices[safe_rows] == source_indices
        if not aligned.all():
            missing = source_indices[~aligned][:5].tolist()
            raise ValueError(f"Synthetic annotations are not aligned with dataset indices: {missing}")
        base_episodes = _dataset_scalar_column(base_dataset, "episode_index")
        base_frames = _dataset_scalar_column(base_dataset, "frame_index")
        if not np.array_equal(self.annotations.episode_indices[annotation_rows], base_episodes):
            raise ValueError("Synthetic annotation episode_index values do not match the base dataset")
        if not np.array_equal(self.annotations.frame_indices[annotation_rows], base_frames):
            raise ValueError("Synthetic annotation frame_index values do not match the base dataset")

        eligible = self._candidate_masks[annotation_rows] != 0
        self._base_rows = np.flatnonzero(eligible).astype(np.int64)
        self._annotation_rows = annotation_rows[eligible].astype(np.int64)
        if len(self._base_rows) == 0:
            raise ValueError("No samples have a fully valid synthetic action chunk")
        eligible_episodes = self.annotations.episode_indices[self._annotation_rows]
        self.meta = SyntheticDatasetMetadataView(base_dataset.meta, eligible_episodes)

        unique_masks = np.unique(self._candidate_masks[self._annotation_rows])
        self._candidates_by_mask = {
            int(mask): tuple(i for i in range(len(self.annotations.specs)) if int(mask) & (1 << i))
            for mask in unique_masks
        }

        source_transforms = list(transforms or [])
        if source_transforms:
            from lerobot.policies.internvla_a1_5.transform_internvla_a1_5 import (
                InternVLAA15ChatProcessorTransformFn,
            )
            from lerobot.transforms.core import (
                DeltaActionTransformFn,
                NormalizeTransformFn,
                ReorderStateActionTransform,
                compose,
            )

            source_transforms = [
                transform
                for transform in source_transforms
                if not isinstance(
                    transform,
                    (DeltaActionTransformFn, NormalizeTransformFn, ReorderStateActionTransform),
                )
            ]
            self._transforms = []
            for schema in self.embodiment_schemas:
                body_meta = SyntheticDatasetMetadataView(
                    base_dataset.meta,
                    eligible_episodes,
                    robot_type=schema.robot_type,
                )
                view = SyntheticDatasetHydrationView(self, body_meta)
                hydrated = []
                for transform in source_transforms:
                    if isinstance(transform, InternVLAA15ChatProcessorTransformFn):
                        body_transform = copy(transform)
                        body_transform.action_mode = schema.action_mode
                    else:
                        body_transform = transform.hydrate(view)
                    hydrated.append(body_transform)
                self._transforms.append(compose(hydrated))
        else:
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
        annotation_row = int(self._annotation_rows[idx])
        candidate_ids = self._candidates_by_mask[int(self._candidate_masks[annotation_row])]
        return tuple(self.annotations.specs[body_id].name for body_id in candidate_ids)

    def __len__(self) -> int:
        return len(self._base_rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if idx < 0:
            idx += len(self)
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)

        base_row = int(self._base_rows[idx])
        annotation_row = int(self._annotation_rows[idx])
        mask = int(self._candidate_masks[annotation_row])
        candidate_ids = self._candidates_by_mask[mask]
        body_id = candidate_ids[torch.randint(len(candidate_ids), ()).item()]
        spec = self.annotations.specs[body_id]
        schema = self.embodiment_schemas[body_id]
        stats = self.embodiment_stats[spec.name]
        poses = self.annotations.values[body_id]

        raw_state = poses[annotation_row]
        raw_action = poses[annotation_row + 1 : annotation_row + self.chunk_size + 1]
        delta_mask = _schema_delta_mask(schema, spec)
        training_action = _to_delta_action(raw_action, raw_state, delta_mask)

        state = _normalize(raw_state, stats.state_mean, stats.state_std)
        action = _normalize(
            training_action,
            stats.action_mean,
            stats.action_std,
        )

        padded_state = np.zeros(self.max_state_dim, dtype=np.float32)
        padded_action = np.zeros((self.chunk_size, self.max_action_dim), dtype=np.float32)
        padded_state[: spec.dim] = state
        padded_action[:, : spec.dim] = action

        sample = self.base_dataset[base_row]
        # Reorder before the chat processor tokenizes state; the default pipeline
        # would reorder only after that textual state representation was built.
        sample[OBS_STATE] = torch.from_numpy(_reorder_vector(padded_state, schema.state_reorder))
        sample[ACTION] = torch.from_numpy(_reorder_vector(padded_action, schema.action_reorder))
        layout = ", ".join(
            f"{block['name']}[{block['size']}]" for block in schema.action_layout
        )
        sample["task"] = (
            f"Current robot embodiment: {spec.name}; task: {sample['task']}; "
            f"action layout: {layout}"
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
    selected = list(embodiments) if embodiments is not None else discovered
    if not selected:
        raise ValueError("No synthetic embodiments were selected")
    if len(set(selected)) != len(selected):
        raise ValueError("Synthetic embodiment selection contains duplicates")

    specs: list[EmbodimentSpec] = []
    for name in selected:
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
        if not path.is_file():
            raise FileNotFoundError(f"Normalization statistics not found: {path}")
        with path.open(encoding="utf-8") as stream:
            payload = json.load(stream)
        state = payload[OBS_STATE]
        action = payload[ACTION]
        body_stats = EmbodimentStats(
            state_mean=_stat_vector(state, "mean", spec, path),
            state_std=_stat_vector(state, "std", spec, path),
            action_mean=_stat_vector(action, "mean", spec, path),
            action_std=_stat_vector(action, "std", spec, path),
        )
        if (body_stats.state_std < 0).any() or (body_stats.action_std < 0).any():
            raise ValueError(f"Negative standard deviation in {path}")
        loaded[spec.name] = body_stats
    return loaded


def load_embodiment_schemas(
    specs: Sequence[EmbodimentSpec],
    schema_prefix: str = DEFAULT_EMBODIMENT_SCHEMA_PREFIX,
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


def load_stats_selection(
    stats_dir: str | Path,
    embodiments: Sequence[str] | None,
    *,
    action_mode: str,
    chunk_size: int,
    schema_prefix: str = DEFAULT_EMBODIMENT_SCHEMA_PREFIX,
) -> list[str]:
    """Resolve selected embodiments and verify that stats match the training target."""
    manifest_path = Path(stats_dir) / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Synthetic embodiment stats manifest not found: {manifest_path}")
    with manifest_path.open(encoding="utf-8") as stream:
        manifest = json.load(stream)

    if manifest.get("format_version") != STATS_FORMAT_VERSION:
        raise ValueError(
            f"Unsupported synthetic stats format in {manifest_path}: "
            f"{manifest.get('format_version')!r}"
        )
    if manifest.get("action_mode") != action_mode:
        raise ValueError(
            f"Stats action_mode is {manifest.get('action_mode')!r}, expected {action_mode!r}"
        )
    if manifest.get("chunk_size") != chunk_size:
        raise ValueError(
            f"Stats chunk_size is {manifest.get('chunk_size')!r}, expected {chunk_size}"
        )

    dimensions = {}
    for item in manifest["embodiments"]:
        name = item["name"]
        expected_schema = f"{schema_prefix}{name}"
        if item.get("schema") != expected_schema:
            raise ValueError(
                f"Stats manifest schema for {name!r} is {item.get('schema')!r}, "
                f"expected {expected_schema!r}"
            )
        dimensions[name] = int(item["dimension"])
    selected = list(embodiments) if embodiments else list(dimensions)
    missing = [name for name in selected if name not in dimensions]
    if missing:
        raise KeyError(f"Embodiments missing from stats manifest: {missing}")
    if not selected:
        raise ValueError("Stats manifest contains no embodiments")
    return selected


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


def _stat_vector(payload: dict[str, Any], key: str, spec: EmbodimentSpec, path: Path) -> np.ndarray:
    vector = np.asarray(payload[key], dtype=np.float32)
    if vector.shape != (spec.dim,) or not np.isfinite(vector).all():
        raise ValueError(f"Invalid {key} for {spec.name!r} in {path}")
    return vector


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


def _reorder_vector(
    values: np.ndarray,
    reorder: Sequence[Sequence[int]] | None,
) -> np.ndarray:
    if reorder is None:
        return values
    result = np.zeros_like(values)
    for src_start, src_end, dst_start, dst_end in reorder:
        result[..., dst_start:dst_end] = values[..., src_start:src_end]
    return result


def _to_delta_action(
    future_poses: np.ndarray,
    current_pose: np.ndarray,
    delta_mask: np.ndarray,
) -> np.ndarray:
    action = np.asarray(future_poses, dtype=np.float32).copy()
    action[..., delta_mask] -= np.asarray(current_pose, dtype=np.float32)[..., delta_mask]
    return action


def _normalize(values: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return np.asarray((values - mean) / (std + NORMALIZATION_EPS), dtype=np.float32)


def _identity(sample: dict[str, Any]) -> dict[str, Any]:
    return sample


def _stats_to_json_dict(stats: EmbodimentStats, *, state: bool) -> dict[str, list[float]]:
    mean = stats.state_mean if state else stats.action_mean
    std = stats.state_std if state else stats.action_std
    return {"mean": mean.tolist(), "std": std.tolist()}
