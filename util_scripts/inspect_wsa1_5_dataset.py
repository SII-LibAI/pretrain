"""Read-only real-data check. This program never constructs or loads a model."""
import argparse
import json
import logging
from pathlib import Path

import torch
import av
from torch.utils.data import DataLoader

from lerobot.datasets.configuration_wsa1_5 import WSA1_5DatasetConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.wsa1_5 import WSA1_5Dataset, wsa1_5_collate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    torch.set_num_threads(2)
    base = LeRobotDataset(args.root.name, root=args.root, video_backend="pyav")
    ds = WSA1_5Dataset(base, WSA1_5DatasetConfig(repo_id=args.root.name))
    batch = next(iter(DataLoader(ds, batch_size=1, num_workers=args.workers, collate_fn=wsa1_5_collate)))
    report = {"samples": len(ds), "episodes": ds.num_episodes, "view_keys": ds.view_keys,
              "observed_shape": list(batch["observed_rgb"].shape),
              "rgb_shape": list(batch["targets"]["future_rgb"].shape),
              "motion_shape": list(batch["targets"]["future_motion"].shape),
              "goal_valid": batch["targets"]["goal_view_valid"].tolist(),
              "rgb_valid_pixels": int(batch["targets"]["rgb_valid"].sum()),
              "motion_valid_frames": int(batch["targets"]["motion_valid"].sum()),
              "subtask_id": batch["subtask_id"].tolist(), "text": batch["condition_text"],
              "workers": args.workers, "sidecar_digest": ds.sidecar_digest}
    ep = int(batch["episode_index"][0])
    timestamps = batch["frame_timestamps"][0].tolist()
    frames, valid = ds._decode(ds.config.motion_key, timestamps, ep)
    if valid.all():
        video_path = ds.root / ds.meta.get_video_file_path(ep, ds.config.motion_key)
        offset = float(ds.meta.episodes[ep][f"videos/{ds.config.motion_key}/from_timestamp"])
        query = torch.tensor([timestamp + offset for timestamp in timestamps], dtype=torch.float64)
        native, pts = [], []
        with av.open(str(video_path)) as container:
            for frame in container.decode(video=0):
                time = float(frame.pts * frame.time_base)
                if time >= float(query.min()) - 1 / ds.meta.fps:
                    pts.append(time)
                    native.append(torch.from_numpy(frame.to_ndarray(format="rgb24").copy()).permute(2, 0, 1).float() / 255)
                if time >= float(query.max()):
                    break
        distance = (query[:, None] - torch.tensor(pts, dtype=torch.float64)[None]).abs()
        errors, indices = distance.min(1)
        other = torch.stack(native)[indices]
        report["motion_codec_range"] = [float(frames.min()), float(frames.max())]
        report["motion_decoder_max_difference"] = float((frames - other).abs().max())
        report["motion_decoder_mean_difference"] = float((frames - other).abs().mean())
        report["motion_max_timestamp_error_seconds"] = float(errors.max())
        report["motion_physical_decode_error"] = "unknown: original XYZ and scale/offset metadata are unavailable"
    if report["motion_valid_frames"] != 32 or report["rgb_valid_pixels"] == 0:
        raise RuntimeError(f"Real first sample failed media checks: {report}")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
