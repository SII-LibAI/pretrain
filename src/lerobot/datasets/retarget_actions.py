"""Shared IK action windows for WSA training and offline normalization statistics."""
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from lerobot.datasets.synthetic_embodiment import (
    HUMAN_EEF_EMBODIMENT, SyntheticActionTable, _scalar_column_to_numpy, _validate_annotation_rows,
    embodiment_schema_prefix, load_embodiment_schemas,
)


def action_offsets(chunk_size, stride=1, *, retarget=False):
    """Native controls start at t; retarget poses start at t+stride."""
    if chunk_size < 1 or stride < 1:
        raise ValueError("chunk_size and stride must be positive")
    return (np.arange(chunk_size, dtype=np.int64) + int(retarget)) * stride


class RetargetActionSource:
    """Read one embodiment plus its shared robot EEF, preserving source frame indices."""
    def __init__(self, root, chunk_size, *, stride=1, embodiments=None, action_space="unified72",
                 action_mode="delta", stats_dir=None, row_group_cache_size=2,
                 annotation_file="meta/annotations/action.parquet"):
        if chunk_size < 1 or stride < 1:
            raise ValueError("chunk_size and stride must be positive")
        if action_space not in {"legacy", "unified72"} or action_mode not in {"abs", "delta"}:
            raise ValueError("Invalid action space or abs/delta mode")
        self.root = Path(root)
        self.chunk_size, self.stride = chunk_size, stride
        self.action_space, self.action_mode = action_space, action_mode
        self.offsets = np.concatenate(([0], action_offsets(chunk_size, stride, retarget=True)))
        self.table = SyntheticActionTable.from_parquet(
            self.root / annotation_file, row_group_cache_size=row_group_cache_size)
        available = self.table.names
        prefix = embodiment_schema_prefix(
            json.loads((self.root / "meta/info.json").read_text(encoding="utf-8"))["robot_type"])
        selected = [name.removeprefix(prefix) for name in embodiments] if embodiments else [
            name for name in available if action_space == "legacy" or name != HUMAN_EEF_EMBODIMENT]
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
                schema = replace(schema, embodiment_name=spec.name)
            self.schemas.append(schema)
            self.specs.append(spec)
        self.names = tuple(spec.name for spec in self.specs)

        valid_keys = list(dict.fromkeys(key for spec in self.table.specs for key in spec.valid_keys))
        metadata = pq.read_table(self.table.path, columns=["index", "episode_index", "frame_index", *valid_keys])
        self.indices, self.episodes, self.frames = [
            _scalar_column_to_numpy(metadata[key], np.int64) for key in ("index", "episode_index", "frame_index")]
        _validate_annotation_rows(self.indices, self.episodes, self.frames, self.table.path)
        self.candidate_masks = np.zeros(len(self.table), dtype=np.uint64)
        for body, (body_id, include_eef) in enumerate(zip(self.body_ids, self.with_eef, strict=True)):
            spec = self.table.specs[body_id]
            keys = list(spec.valid_keys)
            if include_eef:
                keys.extend(self.table.specs[self.eef_id].valid_keys)
            valid = np.logical_and.reduce([_scalar_column_to_numpy(metadata[key], np.bool_) for key in keys])
            eligible = valid.copy()
            for offset in self.offsets[1:]:
                if offset >= len(eligible):
                    eligible[:] = False
                    break
                eligible[:-offset] &= valid[offset:] & (self.episodes[:-offset] == self.episodes[offset:])
                eligible[-offset:] = False
            self.candidate_masks |= eligible.astype(np.uint64) << np.uint64(body)
        self.stats = {}
        if stats_dir is not None:
            for body in self.active_bodies:
                name, spec, schema = self.names[body], self.specs[body], self.schemas[body]
                path = Path(stats_dir) / name / "stats.json"
                payload = json.loads(path.read_text(encoding="utf-8"))
                if payload.get("training") != {**self.training_spec, "action_mask_spec": schema.action_mask_spec}:
                    raise ValueError(f"Statistics window/representation differs from training: {path}")
                for key in ("observation.state", "action"):
                    if any(len(payload[key][stat]) != spec.dim for stat in ("mean", "std")):
                        raise ValueError(f"Statistics dimension differs from {name}: {path}")
                self.stats[name] = payload

    @property
    def training_spec(self):
        return {"action_space": self.action_space, "action_mode": self.action_mode,
                "chunk_size": self.chunk_size, "stride": self.stride}

    @property
    def active_bodies(self):
        union = int(np.bitwise_or.reduce(self.candidate_masks, initial=np.uint64(0)))
        return [body for body in range(len(self.names)) if union & (1 << body)]

    def rows_for_base(self, base):
        metadata = base.hf_dataset.select_columns(["index", "episode_index", "frame_index"]).with_format("arrow")[:]
        indices, episodes, frames = [_scalar_column_to_numpy(metadata[key], np.int64)
                                     for key in ("index", "episode_index", "frame_index")]
        rows = np.searchsorted(self.indices, indices)
        if ((rows >= len(self.indices)).any() or not np.array_equal(self.indices[rows], indices)
                or not np.array_equal(self.episodes[rows], episodes) or not np.array_equal(self.frames[rows], frames)):
            raise ValueError(f"Retarget annotations do not align with the loaded dataset: {self.root}")
        return rows

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
