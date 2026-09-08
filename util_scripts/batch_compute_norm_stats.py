#!/usr/bin/env python

"""Batch-compute normalization statistics for directories of LeRobot v3 datasets.

The input directory is recursively scanned for dataset roots containing both
``meta/info.json`` and ``data/``. Datasets are grouped by ``robot_type`` and each
group is passed to ``compute_norm_stats_multi``. Results are written as::

    <output_dir>/<robot_type>/<action_mode>/stats.json
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Discover LeRobot v3 datasets and aggregate norm stats per robot type."
    )
    parser.add_argument(
        "--dataset_root",
        required=True,
        help="Directory to scan recursively for LeRobot v3 datasets.",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Output root. Results are written to <robot_type>/<action_mode>/stats.json.",
    )
    parser.add_argument(
        "--action_mode",
        choices=["abs", "delta"],
        required=True,
        help="Action representation used when computing statistics.",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=50,
        help="Action chunk size. Episodes shorter than this are skipped (default: 50).",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of repo-level worker processes per robot type (default: 8).",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Only print discovered datasets and groups; do not compute statistics.",
    )
    return parser.parse_args()


def _read_robot_type(info_path: Path) -> str:
    try:
        with info_path.open("r", encoding="utf-8") as file:
            info = json.load(file)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Failed to read LeRobot metadata: {info_path}") from exc

    robot_type = info.get("robot_type")
    if not isinstance(robot_type, str) or not robot_type.strip():
        raise ValueError(f"Missing non-empty robot_type in {info_path}")
    return robot_type.strip()


def discover_datasets(dataset_root: str) -> dict[str, list[Path]]:
    """Return unique dataset roots grouped by metadata robot_type."""
    grouped: dict[str, list[Path]] = defaultdict(list)
    seen: set[Path] = set()

    scan_root = Path(dataset_root).expanduser().resolve()
    if not scan_root.is_dir():
        raise NotADirectoryError(f"Dataset root does not exist or is not a directory: {scan_root}")

    for info_path in scan_root.glob("**/meta/info.json"):
        discovered_root = info_path.parent.parent.resolve()
        if discovered_root in seen or not (discovered_root / "data").is_dir():
            continue
        seen.add(discovered_root)
        grouped[_read_robot_type(info_path)].append(discovered_root)

    for datasets in grouped.values():
        datasets.sort(key=lambda path: str(path).lower())
    return dict(sorted(grouped.items()))


def main() -> None:
    args = parse_args()
    if args.chunk_size <= 0:
        raise ValueError("--chunk_size must be positive")
    if args.num_workers <= 0:
        raise ValueError("--num_workers must be positive")

    grouped = discover_datasets(args.dataset_root)
    if not grouped:
        raise FileNotFoundError(
            "No LeRobot v3 datasets found. Expected dataset roots containing meta/info.json and data/."
        )

    print(f"Discovered {sum(len(paths) for paths in grouped.values())} datasets "
          f"across {len(grouped)} robot types:")
    for robot_type, dataset_paths in grouped.items():
        print(f"\n[{robot_type}] ({len(dataset_paths)} datasets)")
        for dataset_path in dataset_paths:
            print(f"  - {dataset_path}")
        print(f"  -> {Path(args.output_dir).resolve() / robot_type / args.action_mode / 'stats.json'}")

    if args.dry_run:
        print("\nDry run complete; no statistics were computed.")
        return

    from compute_norm_stats_multi import _make_group_name, compute_norm_stats_multi

    for robot_type, dataset_paths in grouped.items():
        print(f"\n========== robot_type: {robot_type} ==========")
        repo_ids = [str(path) for path in dataset_paths]
        with tempfile.TemporaryDirectory(prefix="wsa_norm_stats_") as temporary_dir:
            compute_norm_stats_multi(
                SimpleNamespace(
                    repo_ids=repo_ids,
                    action_mode=args.action_mode,
                    chunk_size=args.chunk_size,
                    num_workers=min(args.num_workers, len(dataset_paths)),
                    output_dir=temporary_dir,
                )
            )
            generated_stats = Path(temporary_dir) / _make_group_name(repo_ids) / "stats.json"
            if not generated_stats.is_file():
                raise FileNotFoundError(f"Expected statistics output was not created: {generated_stats}")

            target_dir = Path(args.output_dir).expanduser().resolve() / robot_type / args.action_mode
            target_dir.mkdir(parents=True, exist_ok=True)
            target_stats = target_dir / "stats.json"
            shutil.copy2(generated_stats, target_stats)
            print(f"final output: {target_stats}")


if __name__ == "__main__":
    main()
