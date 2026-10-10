#!/usr/bin/env python
"""Compute per-representation normalization statistics for synthetic actions."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch

from lerobot.datasets.synthetic_embodiment import (
    DEFAULT_ANNOTATION_FILE,
    EmbodimentSpec,
    SyntheticActionTable,
    _build_candidate_masks,
    discover_embodiments,
    embodiment_schema_prefix,
    load_embodiment_schemas,
)
from lerobot.utils.constants import ACTION, OBS_STATE


def load_dataset_roots(list_file: str | Path, annotation_file: str | Path) -> list[Path]:
    """Read one LeRobot root per line; relative paths use the list file's directory."""
    list_file = Path(list_file).resolve()
    annotation_file = Path(annotation_file)
    if annotation_file.is_absolute():
        raise ValueError("annotation_file must be relative to each dataset root")
    roots: list[Path] = []
    seen: set[Path] = set()
    for line_number, line in enumerate(list_file.read_text(encoding="utf-8-sig").splitlines(), 1):
        entry = line.strip()
        if not entry or entry.startswith("#"):
            continue
        root = Path(entry).expanduser()
        if not root.is_absolute():
            root = list_file.parent / root
        root = root.resolve()
        if root in seen:
            raise ValueError(f"Duplicate dataset root on line {line_number}: {root}")
        if not (root / "meta" / "info.json").is_file() or not (root / "data").is_dir():
            raise FileNotFoundError(f"Line {line_number} is not a LeRobot dataset root: {root}")
        if not (root / annotation_file).is_file():
            raise FileNotFoundError(f"Line {line_number} has no {annotation_file}: {root}")
        seen.add(root)
        roots.append(root)
    if not roots:
        raise ValueError(f"No dataset roots in {list_file}")
    return roots


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

    def merge(self, other: RunningVectorStats) -> None:
        """Merge one worker's Welford accumulator without losing variance."""
        if other.count == 0:
            return
        if self.count == 0:
            self.count = other.count
            self.mean = other.mean.copy()
            self.m2 = other.m2.copy()
            self.minimum = other.minimum.copy()
            self.maximum = other.maximum.copy()
            return
        total = self.count + other.count
        delta = other.mean - self.mean
        self.m2 += other.m2 + np.square(delta) * self.count * other.count / total
        self.mean += delta * other.count / total
        self.count = total
        self.minimum = np.minimum(self.minimum, other.minimum)
        self.maximum = np.maximum(self.maximum, other.maximum)

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


def _compute_one_root(
    root: Path,
    annotation_file: str | Path,
    embodiments: Sequence[str] | None,
    chunk_size: int,
    batch_size: int,
) -> tuple[tuple[EmbodimentSpec, ...], dict[str, tuple[RunningVectorStats, RunningVectorStats]]]:
    """Worker: accumulate each embodiment's trainable state/action vectors."""
    from lerobot.transforms.core import DeltaActionTransformFn

    path = root / annotation_file
    parquet = pq.ParquetFile(path)
    specs = discover_embodiments(parquet, embodiments)
    with (root / "meta" / "info.json").open(encoding="utf-8") as stream:
        source_robot_type = json.load(stream)["robot_type"]
    schema_prefix = embodiment_schema_prefix(source_robot_type)
    schemas = load_embodiment_schemas(specs, schema_prefix)
    accumulators = {
        spec.name: (RunningVectorStats(spec.dim), RunningVectorStats(spec.dim))
        for spec in specs
    }

    for spec, schema in zip(specs, schemas):
        table = SyntheticActionTable.from_parquet(path, [spec.name])
        anchors = np.flatnonzero(_build_candidate_masks(table, chunk_size))
        values = table.values[0]
        state_stats, action_stats = accumulators[spec.name]
        view = SimpleNamespace(meta=SimpleNamespace(robot_type=schema.robot_type))
        delta_transform = DeltaActionTransformFn().hydrate(view)

        for start in range(0, len(anchors), batch_size):
            batch_anchors = anchors[start : start + batch_size]
            states = values[batch_anchors]
            state_stats.update(states)
            actions = np.empty((len(batch_anchors) * chunk_size, spec.dim), dtype=np.float32)
            for i, anchor in enumerate(batch_anchors):
                sample = {
                    OBS_STATE: torch.from_numpy(states[i]),
                    ACTION: torch.from_numpy(values[anchor + 1 : anchor + chunk_size + 1].copy()),
                }
                delta_transform(sample)
                actions[i * chunk_size : (i + 1) * chunk_size] = sample[ACTION].numpy()
            action_stats.update(actions)

    return specs, accumulators


def compute_embodiment_stats(
    dataset_roots: Sequence[str | Path],
    *,
    annotation_file: str | Path = DEFAULT_ANNOTATION_FILE,
    embodiments: Sequence[str] | None = None,
    chunk_size: int = 50,
    batch_size: int = 1024,
    num_workers: int = 1,
) -> tuple[dict[str, dict[str, dict[str, Any]]], tuple[EmbodimentSpec, ...]]:
    """Aggregate worker-local statistics, parallelized across dataset roots."""
    if chunk_size < 1 or batch_size < 1 or num_workers < 1:
        raise ValueError("chunk_size, batch_size, and num_workers must be positive")
    roots = [Path(root).resolve() for root in dataset_roots]
    if not roots:
        raise ValueError("dataset_roots is empty")
    jobs = [
        (root, annotation_file, embodiments, chunk_size, batch_size)
        for root in roots
    ]
    if num_workers == 1:
        results = [_compute_one_root(*job) for job in jobs]
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=min(num_workers, len(roots))) as pool:
            results = pool.starmap(_compute_one_root, jobs)

    reference_specs = results[0][0]
    accumulators = {
        spec.name: (RunningVectorStats(spec.dim), RunningVectorStats(spec.dim))
        for spec in reference_specs
    }
    for root, (specs, worker_stats) in zip(roots, results):
        if [(spec.name, spec.dim) for spec in specs] != [
            (spec.name, spec.dim) for spec in reference_specs
        ]:
            raise ValueError(f"Embodiment schema differs in {root / annotation_file}")
        for spec in reference_specs:
            state_stats, action_stats = accumulators[spec.name]
            worker_state, worker_action = worker_stats[spec.name]
            state_stats.merge(worker_state)
            action_stats.merge(worker_action)

    return {
        spec.name: {
            OBS_STATE: accumulators[spec.name][0].as_dict(),
            ACTION: accumulators[spec.name][1].as_dict(),
        }
        for spec in reference_specs
    }, reference_specs


def write_embodiment_stats(
    output_dir: str | Path,
    stats: dict[str, dict[str, dict[str, Any]]],
) -> None:
    """Write one normalization file per embodiment."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, body_stats in stats.items():
        body_dir = output_dir / name
        body_dir.mkdir(parents=True, exist_ok=True)
        _write_json(body_dir / "stats.json", body_stats)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-roots-file",
        type=Path,
        required=True,
        help="UTF-8 text file containing one LeRobot dataset root per line.",
    )
    parser.add_argument(
        "--output-dir",
        help="Directory for stats. Defaults to synthetic_embodiment_stats beside the roots file.",
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
        help=(
            "Training action chunk length. Joint delta and absolute human EEF statistics "
            "are computed over the same valid windows."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Number of spawned processes computing dataset roots in parallel.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    roots = load_dataset_roots(args.dataset_roots_file, args.annotation_file)
    stats, _ = compute_embodiment_stats(
        roots,
        annotation_file=args.annotation_file,
        embodiments=args.embodiments,
        chunk_size=args.chunk_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    output_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else args.dataset_roots_file.resolve().parent / "synthetic_embodiment_stats"
    )
    write_embodiment_stats(
        output_dir,
        stats,
    )
    print(f"Wrote statistics for {len(stats)} embodiments from {len(roots)} datasets to {output_dir}")


if __name__ == "__main__":
    main()
