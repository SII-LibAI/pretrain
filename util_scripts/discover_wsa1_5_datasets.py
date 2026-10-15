"""Print a JSON repo-id list for local LeRobot V3 datasets, including unlabeled ones."""
import argparse
import json
from pathlib import Path


def discover(root):
    root = Path(root).resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)
    repos = sorted({path.parent.parent.relative_to(root).as_posix()
                    for path in root.rglob("meta/info.json") if (path.parent.parent / "data").is_dir()})
    if not repos:
        raise FileNotFoundError(f"No LeRobot datasets under {root}")
    return repos


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    print(json.dumps(discover(args.root), ensure_ascii=False))
