#!/usr/bin/env python
"""Compute per-embodiment delta-action normalization statistics offline."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from lerobot.datasets.synthetic_embodiment import (
    DEFAULT_ANNOTATION_FILE,
    DEFAULT_EMBODIMENT_SCHEMA_PREFIX,
    STATS_FORMAT_VERSION,
    EmbodimentSpec,
    SyntheticActionTable,
    _build_candidate_masks,
    _schema_delta_mask,
    _to_delta_action,
    discover_embodiments,
    load_embodiment_schemas,
)
from lerobot.utils.constants import ACTION, OBS_STATE


def discover_dataset_roots(
    search_roots: list[str],
    annotation_file: str,
) -> list[Path]:
    """Find annotated LeRobot roots for this offline preprocessing command."""
    annotation_path = Path(annotation_file)
    roots: set[Path] = set()
    for search_root_value in search_roots:
        search_root = Path(search_root_value).resolve()
        if (search_root / "meta" / "info.json").is_file():
            if (search_root / annotation_path).is_file():
                roots.add(search_root)
            continue
        for directory, _, files in os.walk(search_root, followlinks=True):
            directory = Path(directory)
            if directory.name == "meta" and "info.json" in files:
                dataset_root = directory.parent
                if (dataset_root / annotation_path).is_file():
                    roots.add(dataset_root)
    if not roots:
        raise FileNotFoundError(f"No LeRobot datasets with {annotation_file} were found")
    return sorted(roots, key=str)


class RunningVectorStats:
    """Streaming population statistics for fixed-size vectors."""

    def __init__(self, dim: int) -> None:
        self.count = 0
        self.mean = np.zeros(dim, dtype=np.float64)
        self.m2 = np.zeros(dim, dtype=np.float64)
        self.minimum = np.full(dim, np.inf, dtype=np.float64)
        self.maximum = np.full(dim, -np.inf, dtype=np.float64)

    def update(self, values: np.ndarray) -> None:
        if len(values) == 0:
            return
        values = np.asarray(values, dtype=np.float64)
        batch_count = len(values)
        batch_mean = values.mean(axis=0)
        batch_m2 = np.square(values - batch_mean).sum(axis=0)
        delta = batch_mean - self.mean
        total = self.count + batch_count
        self.mean += delta * batch_count / total
        self.m2 += batch_m2 + np.square(delta) * self.count * batch_count / total
        self.count = total
        self.minimum = np.minimum(self.minimum, values.min(axis=0))
        self.maximum = np.maximum(self.maximum, values.max(axis=0))

    def as_dict(self) -> dict[str, Any]:
        if self.count == 0:
            raise ValueError("Cannot serialize empty statistics")
        return {
            "mean": self.mean.tolist(),
            "std": np.sqrt(self.m2 / self.count).tolist(),
            "min": self.minimum.tolist(),
            "max": self.maximum.tolist(),
            "count": self.count,
        }


def compute_embodiment_stats(
    dataset_roots: Sequence[str | Path],
    *,
    annotation_file: str | Path = DEFAULT_ANNOTATION_FILE,
    embodiments: Sequence[str] | None = None,
    chunk_size: int = 50,
    batch_size: int = 8192,
    schema_prefix: str = DEFAULT_EMBODIMENT_SCHEMA_PREFIX,
) -> tuple[dict[str, dict[str, dict[str, Any]]], tuple[EmbodimentSpec, ...]]:
    """Compute state and delta-action statistics over eligible training samples."""
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    if batch_size < 1:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    roots = [Path(root).resolve() for root in dataset_roots]
    reference_specs: tuple[EmbodimentSpec, ...] | None = None
    schemas: dict[str, Any] = {}
    state_accumulators: dict[str, RunningVectorStats] = {}
    action_accumulators: dict[str, RunningVectorStats] = {}

    for root in roots:
        path = root / annotation_file
        parquet = pq.ParquetFile(path)
        specs = discover_embodiments(parquet, embodiments)
        if reference_specs is None:
            reference_specs = specs
            loaded_schemas = load_embodiment_schemas(specs, schema_prefix)
            schemas = dict(zip((spec.name for spec in specs), loaded_schemas))
            state_accumulators = {spec.name: RunningVectorStats(spec.dim) for spec in specs}
            action_accumulators = {spec.name: RunningVectorStats(spec.dim) for spec in specs}
        elif [(spec.name, spec.dim) for spec in specs] != [
            (spec.name, spec.dim) for spec in reference_specs
        ]:
            raise ValueError(f"Embodiment schema differs in {path}")

        # Read one embodiment at a time so peak memory is bounded by its vector width.
        for spec in specs:
            table = SyntheticActionTable.from_parquet(path, [spec.name])
            anchors = np.flatnonzero(_build_candidate_masks(table, chunk_size)).astype(np.int64)
            values = table.values[0]
            delta_mask = _schema_delta_mask(schemas[spec.name], spec)
            for start in range(0, len(anchors), batch_size):
                batch_anchors = anchors[start : start + batch_size]
                states = values[batch_anchors]
                state_accumulators[spec.name].update(states)
                for offset in range(1, chunk_size + 1):
                    actions = _to_delta_action(
                        values[batch_anchors + offset],
                        states,
                        delta_mask,
                    )
                    action_accumulators[spec.name].update(actions)

    if reference_specs is None:
        raise ValueError("dataset_roots is empty")
    return {
        spec.name: {
            OBS_STATE: state_accumulators[spec.name].as_dict(),
            ACTION: action_accumulators[spec.name].as_dict(),
        }
        for spec in reference_specs
    }, reference_specs


def write_embodiment_stats(
    output_dir: str | Path,
    dataset_roots: Sequence[str | Path],
    stats: dict[str, dict[str, dict[str, Any]]],
    specs: Sequence[EmbodimentSpec],
    *,
    annotation_file: str | Path = DEFAULT_ANNOTATION_FILE,
    chunk_size: int,
    schema_prefix: str = DEFAULT_EMBODIMENT_SCHEMA_PREFIX,
) -> None:
    """Write one normalization file per embodiment and a collection manifest."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for spec in specs:
        body_dir = output_dir / spec.name
        body_dir.mkdir(parents=True, exist_ok=True)
        _write_json(body_dir / "stats.json", stats[spec.name])

    manifest = {
        "format_version": STATS_FORMAT_VERSION,
        "annotation_file": Path(annotation_file).as_posix(),
        "action_offset": 1,
        "action_mode": "delta",
        "chunk_size": chunk_size,
        "normalization": "mean_std",
        "dataset_roots": [str(Path(root).resolve()) for root in dataset_roots],
        "embodiments": [
            {
                "name": spec.name,
                "schema": f"{schema_prefix}{spec.name}",
                "dimension": spec.dim,
                "state_count": stats[spec.name][OBS_STATE]["count"],
                "action_count": stats[spec.name][ACTION]["count"],
            }
            for spec in specs
        ],
    }
    _write_json(output_dir / "manifest.json", manifest)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        action="append",
        required=True,
        help="A LeRobot dataset root or a directory recursively containing dataset roots. Repeat as needed.",
    )
    parser.add_argument(
        "--output-dir",
        help="Directory for stats. Defaults to <dataset-root>/synthetic_embodiment_stats.",
    )
    parser.add_argument(
        "--annotation-file",
        default=DEFAULT_ANNOTATION_FILE,
        help="Annotation parquet path relative to each LeRobot root.",
    )
    parser.add_argument(
        "--embodiment",
        action="append",
        dest="embodiments",
        help="Embodiment to include. Repeat to select several; omit to include every discovered embodiment.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=50,
        help="Training action chunk length. Delta-action stats are computed over the same windows.",
    )
    parser.add_argument("--batch-size", type=int, default=8192)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    roots = discover_dataset_roots(args.dataset_root, args.annotation_file)
    stats, specs = compute_embodiment_stats(
        roots,
        annotation_file=args.annotation_file,
        embodiments=args.embodiments,
        chunk_size=args.chunk_size,
        batch_size=args.batch_size,
    )
    output_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else Path(args.dataset_root[0]).resolve() / "synthetic_embodiment_stats"
    )
    write_embodiment_stats(
        output_dir,
        roots,
        stats,
        specs,
        annotation_file=args.annotation_file,
        chunk_size=args.chunk_size,
    )
    print(f"Wrote statistics for {len(specs)} embodiments from {len(roots)} datasets to {output_dir}")


if __name__ == "__main__":
    main()
