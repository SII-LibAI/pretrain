"""Save one real training sample without running model inference."""
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.utils import make_grid

from ..dataset_wsa1_5 import WSA1_5Dataset, compose_rgb_views


def _save_image(image, path):
    if image.ndim != 3 or image.shape[0] != 3 or not torch.isfinite(image).all():
        raise ValueError("Snapshot image must be finite [3,H,W]")
    pixels = ((image.detach().float().cpu() + 1) * 127.5).clamp(0, 255).round().to(torch.uint8)
    Image.fromarray(pixels.permute(1, 2, 0).numpy()).save(path)


@torch.no_grad()
def save_training_sample(batch, dataset, output_dir, step):
    """Save the main rank's current batch; selection does not change training RNGs."""
    index = random.Random(step).randrange(len(batch["sample_id"]))
    sample_id = batch["sample_id"][index]
    repo_id, episode, _ = sample_id.rsplit(":", 2)
    episode = int(episode)
    pending = [dataset]
    while pending:
        source = pending.pop()
        if isinstance(source, WSA1_5Dataset):
            if source.repo_id == repo_id:
                break
        else:
            pending.extend(getattr(source, "datasets", ()))
    else:
        raise LookupError(f"Cannot find snapshot source dataset for {sample_id}")

    config = source.config
    observed = batch["observed_rgb"][index].detach().cpu()
    future = batch["targets"]["future_rgb"][index].detach().cpu()
    start = float(batch["start_timestamp"][index])
    goal_time = float(batch["goal_timestamp"][index])
    metadata_valid = bool(batch["metadata_valid"][index])

    # Current motion is diagnostic data only; never add it to model conditions.
    motion, motion_valid = source._decode(config.motion_key, [start], episode)
    motion = F.interpolate(motion, size=(config.motion_height, config.motion_width),
                           mode="bilinear", align_corners=False)
    motion = motion.masked_fill(~motion_valid[:, None, None, None], 0) * 2 - 1

    goal = torch.full_like(observed, -1)
    goal_valid = [False, False, False]
    if metadata_valid:
        views, flags = zip(*(source._decode(key, [goal_time], episode) for key in source.view_keys), strict=True)
        canvas, _ = compose_rgb_views(views, flags, config.rgb_height, config.rgb_width)
        goal = canvas[0] * 2 - 1
        goal_valid = [bool(flag[0]) for flag in flags]

    directory = Path(output_dir) / f"step_{step:08d}"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "prompt.txt").write_text(batch["condition_text"][index], encoding="utf-8")
    rgb = torch.cat([observed[None], future], 0)
    _save_image(make_grid(rgb, nrow=4, padding=2, pad_value=-1), directory / "rgb_current_future.png")
    _save_image(motion[0], directory / "motion_current.png")
    _save_image(goal, directory / "goal.png")
    metadata = {
        "step": step, "batch_index": index, "sample_id": sample_id,
        "start_timestamp": start, "goal_timestamp": goal_time,
        "subtask_id": int(batch["subtask_id"][index]), "metadata_valid": metadata_valid,
        "rgb_grid_columns": 4, "rgb_grid_order": "current, then future frames in time order; row-major",
        "frame_timestamps": batch["frame_timestamps"][index].detach().cpu().tolist(),
        "future_frame_valid": batch["future_frame_valid"][index].detach().cpu().tolist(),
        "view_keys": source.view_keys,
        "observed_view_valid": [bool(key and key in source.meta.features) for key in source.view_keys],
        "current_motion_valid": bool(motion_valid[0]), "goal_view_valid": goal_valid,
        "future_motion_valid": batch["targets"]["motion_valid"][index].detach().cpu().tolist(),
    }
    (directory / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return directory
