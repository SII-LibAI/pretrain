"""Short synthetic videos and fake long streams; no model assets are used."""
from fractions import Fraction
from types import SimpleNamespace
import weakref

import av
import numpy as np
import pytest
import torch

from lerobot.policies.wsa1_5.video import decode_video_frames


def test_streaming_decode_matches_full_reference_for_unsorted_duplicate_queries(tmp_path):
    path = tmp_path / "video.mp4"
    with av.open(str(path), "w") as container:
        stream = container.add_stream("mpeg4", rate=30)
        stream.width = stream.height = 32
        stream.pix_fmt = "yuv420p"
        stream.codec_context.gop_size = 30
        for index in range(90):
            pixels = np.full((32, 32, 3), (index * 7) % 256, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            frame.pts, frame.time_base = index, Fraction(1, 30)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    with av.open(str(path)) as container:
        frames = list(container.decode(video=0))
        pts = np.array([float(frame.pts * frame.time_base) for frame in frames])
        reference = torch.stack([torch.from_numpy(frame.to_ndarray(format="rgb24")).permute(2, 0, 1)
                                 for frame in frames]).float() / 255
    query = [pts[70] + 0.00001, pts[3], pts[70] + 0.00001, pts[30], pts[-1]]
    expected = reference[np.abs(np.asarray(query)[:, None] - pts).argmin(1)]
    actual = decode_video_frames(path, query, 0.001, "pyav", 1)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


class FakeContainer:
    def __init__(self, length, offset=0):
        self.length, self.offset = length, offset
        self.streams = SimpleNamespace(video=[SimpleNamespace(time_base=Fraction(1, 30),
                                                              codec_context=SimpleNamespace(thread_count=None))])
        self.converted, self.live, self.peak, self.closed = [], weakref.WeakSet(), 0, False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.closed = True

    def seek(self, offset, **kwargs):
        self.seek_offset = offset
        assert kwargs["backward"] and not kwargs["any_frame"]

    def decode(self, stream):
        owner = self
        class Frame:
            time_base = Fraction(1, 30)
            def __init__(self, index):
                self.pts, self.index = index + owner.offset, index
                owner.live.add(self)
                owner.peak = max(owner.peak, len(owner.live))
            def to_ndarray(self, format):
                assert format == "rgb24"
                owner.converted.append(self.index)
                return np.full((8, 8, 3), self.index % 256, dtype=np.uint8)
        for index in range(self.length):
            yield Frame(index)


def test_only_selected_frames_are_converted_and_intermediate_frames_are_released(monkeypatch):
    container = FakeContainer(4000)
    monkeypatch.setattr("lerobot.policies.wsa1_5.video.av.open", lambda _: container)
    actual = decode_video_frames("unused", [100.0, 0.0, 100.0], 0.001, "pyav", 1)
    assert container.converted == [0, 3000]
    assert container.peak <= 3 and container.closed
    assert container.streams.video[0].codec_context.thread_count == 1
    torch.testing.assert_close(actual[:, 0, 0, 0], torch.tensor([3000 % 256, 0, 3000 % 256]) / 255)


def test_long_offsets_keep_float64_timestamp_precision(monkeypatch):
    container = FakeContainer(4, offset=1200 * 30)
    monkeypatch.setattr("lerobot.policies.wsa1_5.video.av.open", lambda _: container)
    actual = decode_video_frames("unused", [1200 + 1 / 30 + 0.00005], 0.0001, "pyav", 1)
    torch.testing.assert_close(actual[0], torch.full_like(actual[0], 1 / 255))
    assert container.closed


def test_bad_timestamp_still_raises_and_closes_decoder(monkeypatch):
    container = FakeContainer(3)
    monkeypatch.setattr("lerobot.policies.wsa1_5.video.av.open", lambda _: container)
    with pytest.raises(ValueError, match="exceed tolerance"):
        decode_video_frames("unused", [1.0], 0.001, "pyav", 1)
    assert container.closed


def test_other_backends_keep_the_existing_decoder(monkeypatch):
    calls = []
    def decode(*args):
        calls.append(args)
        return torch.ones(1, 3, 8, 8)
    monkeypatch.setattr("lerobot.policies.wsa1_5.video.decode_other_backend", decode)
    result = decode_video_frames("unused", [0.0], 0.001, "torchcodec", 2)
    assert result.shape == (1, 3, 8, 8)
    assert calls == [("unused", [0.0], 0.001, "torchcodec")]
