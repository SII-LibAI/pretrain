"""Crash-safe progress records for LIBERO-plus evaluations.

Task records live outside logs/{suite}/*.json so the existing shard aggregator
only sees complete shards. A record is committed after its optional video.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path


def write_json_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as file:
            json.dump(value, file, indent=2)
            file.flush()
            os.fsync(file.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def archive_shard_summary(path: Path) -> None:
    """Exclude an incomplete shard from aggregation without deleting its old result."""
    if path.exists():
        path.replace(path.with_name(f"{path.name}.previous.{os.getpid()}.{time.time_ns()}"))


def progress_path(root: Path, suite: str, task_id: int) -> Path:
    return root / "progress" / suite / f"{task_id}.json"


def video_path(video_dir: Path, tag: str, episode_idx: int, success: bool) -> Path:
    suffix = "success" if success else "failure"
    return video_dir / f"rollout_{tag}_episode{episode_idx}_{suffix}.mp4"


def load_progress(path: Path, signature: dict, video_dir: Path) -> list[bool]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as file:
        record = json.load(file)
    if not isinstance(record, dict):
        raise ValueError(f"Invalid task progress: {path}")
    if record.get("signature") != signature:
        raise ValueError(
            f"Resume metadata differs at {path}. Use the original evaluation settings "
            "and checkpoint, or choose a new EVAL_LOG_DIR."
        )
    successes = record.get("successes")
    expected = signature["num_episodes"]
    if not isinstance(successes, list) or len(successes) > expected or any(type(x) is not bool for x in successes):
        raise ValueError(f"Invalid task progress: {path}")
    if signature["save_videos"]:
        for index, success in enumerate(successes):
            video = video_path(video_dir, signature["video_tag"], index, success)
            if not video.is_file() or video.stat().st_size == 0:
                return successes[:index]
    return successes


def save_progress(path: Path, signature: dict, successes: list[bool]) -> None:
    write_json_atomic(path, {"signature": signature, "successes": successes})


def shard_complete(
    root: Path,
    suite: str,
    start: int,
    end: int,
    classification: dict[int, dict],
    run_id: str,
    requested_trials: int,
    save_videos: bool,
) -> bool:
    """Skip a shard only when every task and its category summary are complete."""
    shard_file = root / "logs" / suite / f"{start}_to_{end}.json"
    if not shard_file.is_file():
        return False
    # eval_libero_plus.py emits every suite category, including zero-count
    # categories outside this particular shard.
    counts: dict[str, dict[str, int]] = {
        item["category"]: {"total_count": 0, "success_count": 0}
        for item in classification.values()
    }
    video_dir = root / "videos" / suite
    try:
        for task_id in range(start, end):
            item = classification[task_id]
            path = progress_path(root, suite, task_id)
            with path.open(encoding="utf-8") as file:
                record = json.load(file)
            if not isinstance(record, dict):
                return False
            signature = record["signature"]
            if (
                signature["run_id"] != run_id
                or signature["suite"] != suite
                or signature["task_id"] != task_id
                or signature["category"] != item["category"]
                or signature["video_tag"] != item["name"]
                or signature["save_videos"] is not save_videos
                or not 0 < signature["num_episodes"] <= requested_trials
            ):
                return False
            successes = load_progress(path, signature, video_dir)
            if len(successes) != signature["num_episodes"]:
                return False
            bucket = counts.setdefault(item["category"], {"total_count": 0, "success_count": 0})
            bucket["total_count"] += len(successes)
            bucket["success_count"] += sum(successes)
        with shard_file.open(encoding="utf-8") as file:
            return json.load(file) == counts
    except (OSError, ValueError, KeyError, TypeError, IndexError, json.JSONDecodeError):
        return False


def read_run_id(root: Path) -> str | None:
    manifest = root / "run_config.json"
    if not manifest.exists():
        return None
    with manifest.open(encoding="utf-8") as file:
        return json.load(file)["run_id"]


def ensure_run_manifest(root: Path, settings: dict[str, str]) -> str:
    """Keep a resumed output directory tied to one model and eval setup."""
    classification_path = settings.get("classification_path")
    if classification_path:
        settings = {
            **settings,
            "classification_sha256": hashlib.sha256(Path(classification_path).read_bytes()).hexdigest(),
        }
    normalized = {"format_version": 1, **settings}
    run_id = hashlib.sha256(json.dumps(normalized, sort_keys=True).encode("utf-8")).hexdigest()
    manifest_path = root / "run_config.json"
    manifest = {"run_id": run_id, "settings": normalized}
    if manifest_path.exists():
        with manifest_path.open(encoding="utf-8") as file:
            previous = json.load(file)
        if previous != manifest:
            raise ValueError(
                f"Evaluation settings differ from {manifest_path}. "
                "Resume with the original settings or choose a new EVAL_LOG_DIR."
            )
    else:
        write_json_atomic(manifest_path, manifest)
    return run_id


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["manifest"])
    parser.add_argument("root", type=Path)
    # REMAINDER also accepts values like "--action_loss_only" passed by the
    # launcher as ordinary setting values, not CLI options.
    parser.add_argument("settings", nargs=argparse.REMAINDER, help="Alternating key value pairs")
    args = parser.parse_args()
    if len(args.settings) % 2:
        parser.error("settings must be alternating key value pairs")
    settings = dict(zip(args.settings[::2], args.settings[1::2]))
    print(ensure_run_manifest(args.root, settings))


if __name__ == "__main__":
    main()
