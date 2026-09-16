#!/usr/bin/env python
"""List annotated LeRobot dataset roots under a parent directory."""

from __future__ import annotations

import argparse
from pathlib import Path


def discover_lerobot_roots(parent_dir: str | Path, annotation_file: str | Path) -> list[Path]:
    parent_dir = Path(parent_dir).resolve()
    annotation_file = Path(annotation_file)
    if not parent_dir.is_dir():
        raise NotADirectoryError(parent_dir)
    if annotation_file.is_absolute():
        raise ValueError("annotation_file must be relative to each dataset root")

    roots: set[Path] = set()
    for info_path in parent_dir.rglob("meta/info.json"):
        root = info_path.parent.parent.resolve()
        if (root / "data").is_dir() and (root / annotation_file).is_file():
            roots.add(root)
    if not roots:
        raise FileNotFoundError(f"No annotated LeRobot datasets found under {parent_dir}")
    return sorted(roots, key=str)


def write_dataset_roots(output_file: str | Path, roots: list[Path]) -> None:
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text("".join(f"{root}\n" for root in roots), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-dir", type=Path, required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    parser.add_argument(
        "--annotation-file",
        default="meta/annotations/action.parquet",
        help="Annotation path relative to each dataset root.",
    )
    args = parser.parse_args()
    roots = discover_lerobot_roots(args.parent_dir, args.annotation_file)
    write_dataset_roots(args.output_file, roots)
    print(f"Wrote {len(roots)} dataset roots to {args.output_file}")


if __name__ == "__main__":
    main()
