"""Bounded-memory PyAV decoding for WSA; other policies keep their existing decoder."""
import av
import numpy as np
import torch

from lerobot.datasets.video_utils import decode_video_frames as decode_other_backend


def decode_video_frames(path, timestamps, tolerance_s, backend, num_threads=1):
    if backend != "pyav":
        return decode_other_backend(path, timestamps, tolerance_s, backend)
    query = np.asarray(timestamps, dtype=np.float64)
    if query.ndim != 1 or not len(query) or not np.isfinite(query).all():
        raise ValueError("Video queries must be a nonempty list of finite timestamps")
    if num_threads <= 0:
        raise ValueError("Video decoder threads must be positive")
    unique, inverse = np.unique(query, return_inverse=True)
    selected, actual = [], []
    converted = {}

    def keep(frame, timestamp):
        if timestamp not in converted:
            pixels = frame.to_ndarray(format="rgb24")
            converted[timestamp] = torch.from_numpy(pixels).permute(2, 0, 1)
        selected.append(converted[timestamp])
        actual.append(timestamp)

    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        # Each GPU has its own DataLoader workers; FFmpeg auto-threading oversubscribes the node.
        stream.codec_context.thread_count = num_threads
        container.seek(int(unique[0] / float(stream.time_base)), stream=stream, backward=True, any_frame=False)
        previous, previous_ts = None, None
        for frame in container.decode(stream):
            if frame.pts is None:
                raise ValueError(f"Video frame has no presentation timestamp: {path}")
            timestamp = float(frame.pts * frame.time_base)
            while len(selected) < len(unique) and unique[len(selected)] <= timestamp:
                requested = unique[len(selected)]
                if previous is not None and abs(previous_ts - requested) <= abs(timestamp - requested):
                    keep(previous, previous_ts)
                else:
                    keep(frame, timestamp)
            previous, previous_ts = frame, timestamp
            if len(selected) == len(unique):
                break
        if previous is None:
            raise ValueError(f"No video frames decoded: {path}")
        while len(selected) < len(unique):
            keep(previous, previous_ts)
    distances = np.abs(unique - np.asarray(actual))
    if not (distances <= tolerance_s).all():
        raise ValueError(f"Video timestamps exceed tolerance {tolerance_s}: "
                         f"query={unique.tolist()} decoded={actual}; path={path}")
    return torch.stack(selected)[torch.from_numpy(inverse)].float().div_(255)
