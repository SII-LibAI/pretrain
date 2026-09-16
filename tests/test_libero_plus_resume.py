from __future__ import annotations

import importlib.util
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


MODULE_PATH = Path(__file__).resolve().parents[1] / "evaluation" / "LIBERO-plus" / "resume_utils.py"
SPEC = importlib.util.spec_from_file_location("libero_plus_resume_utils", MODULE_PATH)
resume = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(resume)


class LiberoPlusResumeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.video_dir = self.root / "videos" / "libero_goal"
        self.signature = {
            "run_id": "run-a",
            "suite": "libero_goal",
            "task_id": 0,
            "category": "Camera Viewpoints",
            "video_tag": "camera_variant_0",
            "num_episodes": 2,
            "save_videos": True,
        }

    def test_partial_task_and_missing_video_resume_from_first_missing_episode(self) -> None:
        path = resume.progress_path(self.root, "libero_goal", 0)
        self.assertEqual(resume.load_progress(path, self.signature, self.video_dir), [])
        self.video_dir.mkdir(parents=True)
        resume.video_path(self.video_dir, "camera_variant_0", 0, True).write_bytes(b"video")
        resume.save_progress(path, self.signature, [True, False])
        self.assertEqual(resume.load_progress(path, self.signature, self.video_dir), [True])
        resume.video_path(self.video_dir, "camera_variant_0", 1, False).write_bytes(b"video")
        self.assertEqual(resume.load_progress(path, self.signature, self.video_dir), [True, False])

    def test_changed_signature_is_rejected(self) -> None:
        path = resume.progress_path(self.root, "libero_goal", 0)
        resume.save_progress(path, self.signature, [True])
        changed = {**self.signature, "run_id": "run-b"}
        with self.assertRaisesRegex(ValueError, "Resume metadata differs"):
            resume.load_progress(path, changed, self.video_dir)

    def test_shard_requires_all_task_records_and_matching_summary(self) -> None:
        signature = {**self.signature, "num_episodes": 1, "save_videos": False}
        record = resume.progress_path(self.root, "libero_goal", 0)
        summary = self.root / "logs" / "libero_goal" / "0_to_1.json"
        classification = {
            0: {"id": 1, "name": "camera_variant_0", "category": "Camera Viewpoints"},
            1: {"id": 2, "name": "light_variant_0", "category": "Light Conditions"},
        }
        resume.save_progress(record, signature, [False])
        self.assertFalse(resume.shard_complete(self.root, "libero_goal", 0, 1, classification, "run-a", 1, False))
        expected = {
            "Camera Viewpoints": {"total_count": 1, "success_count": 0},
            "Light Conditions": {"total_count": 0, "success_count": 0},
        }
        resume.write_json_atomic(summary, expected)
        self.assertTrue(resume.shard_complete(self.root, "libero_goal", 0, 1, classification, "run-a", 1, False))
        expected["Camera Viewpoints"]["success_count"] = 1
        resume.write_json_atomic(summary, expected)
        self.assertFalse(resume.shard_complete(self.root, "libero_goal", 0, 1, classification, "run-a", 1, False))

    def test_run_manifest_detects_changed_settings_and_classification(self) -> None:
        classification = self.root / "classification.json"
        classification.write_text(json.dumps({"libero_goal": []}), encoding="utf-8")
        settings = {"checkpoint": "checkpoint-a", "classification_path": str(classification)}
        first = resume.ensure_run_manifest(self.root, settings)
        self.assertEqual(resume.ensure_run_manifest(self.root, settings), first)
        with self.assertRaisesRegex(ValueError, "settings differ"):
            resume.ensure_run_manifest(self.root, {**settings, "checkpoint": "checkpoint-b"})
        classification.write_text(json.dumps({"libero_goal": [1]}), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "settings differ"):
            resume.ensure_run_manifest(self.root, settings)

    def test_incomplete_shard_summary_is_archived_outside_aggregation_glob(self) -> None:
        summary = self.root / "logs" / "libero_goal" / "0_to_1.json"
        resume.write_json_atomic(summary, {"Camera Viewpoints": {"total_count": 1, "success_count": 1}})
        resume.archive_shard_summary(summary)
        self.assertFalse(summary.exists())
        archived = list(summary.parent.glob("0_to_1.json.previous.*"))
        self.assertEqual(len(archived), 1)
        self.assertEqual(list(summary.parent.glob("*.json")), [])

    def test_manifest_cli_accepts_dash_prefixed_setting_values(self) -> None:
        argv = ["resume_utils.py", "manifest", str(self.root), "action_loss_only", "--action_loss_only", "no_video", "--no-save_videos"]
        with patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()):
            resume.main()
        with (self.root / "run_config.json").open(encoding="utf-8") as file:
            settings = json.load(file)["settings"]
        self.assertEqual(settings["action_loss_only"], "--action_loss_only")
        self.assertEqual(settings["no_video"], "--no-save_videos")


if __name__ == "__main__":
    unittest.main()
