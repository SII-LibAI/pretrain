from pathlib import Path

import pytest

from util_scripts.compute_synthetic_embodiment_stats import load_dataset_roots
from util_scripts.discover_lerobot_root import discover_lerobot_roots, write_dataset_roots


def _make_dataset_root(root: Path) -> None:
    (root / "data").mkdir(parents=True)
    (root / "meta" / "annotations").mkdir(parents=True)
    (root / "meta" / "info.json").write_text("{}", encoding="utf-8")
    (root / "meta" / "annotations" / "action.parquet").touch()


def test_discover_roots_and_read_list(tmp_path: Path) -> None:
    parent = tmp_path / "collection"
    first = parent / "task one" / "demo"
    second = parent / "task two" / "demo"
    _make_dataset_root(first)
    _make_dataset_root(second)
    (parent / "unannotated" / "meta").mkdir(parents=True)
    (parent / "unannotated" / "meta" / "info.json").touch()

    roots = discover_lerobot_roots(parent, "meta/annotations/action.parquet")
    assert roots == [first.resolve(), second.resolve()]

    list_file = tmp_path / "dataset_roots.txt"
    write_dataset_roots(list_file, roots)
    assert list_file.read_text(encoding="utf-8").splitlines() == [str(root) for root in roots]
    assert load_dataset_roots(list_file, "meta/annotations/action.parquet") == roots


def test_roots_file_accepts_relative_paths_and_rejects_duplicates(tmp_path: Path) -> None:
    root = tmp_path / "demo"
    _make_dataset_root(root)
    list_file = tmp_path / "dataset_roots.txt"
    list_file.write_text("# selected datasets\ndemo\n", encoding="utf-8")
    assert load_dataset_roots(list_file, "meta/annotations/action.parquet") == [root.resolve()]

    list_file.write_text("demo\ndemo\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate dataset root"):
        load_dataset_roots(list_file, "meta/annotations/action.parquet")

    list_file.write_text(".\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="not a LeRobot dataset root"):
        load_dataset_roots(list_file, "meta/annotations/action.parquet")
