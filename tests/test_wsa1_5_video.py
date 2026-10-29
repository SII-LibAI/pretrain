"""Short synthetic videos and fake long streams; no model assets are used."""
from fractions import Fraction
from types import SimpleNamespace
import weakref

import av
import numpy as np
import pytest
import torch

from lerobot.policies.wsa1_5.utils.video import decode_video_frames


@pytest.mark.parametrize("start_offset", [Fraction(0), Fraction(507, 15360)])
def test_streaming_decode_matches_full_reference_for_unsorted_duplicate_queries(tmp_path, start_offset):
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
    if start_offset:
        shifted_path = tmp_path / "shifted.mp4"
        with av.open(str(path)) as source, av.open(str(shifted_path), "w",
                                                 options={"avoid_negative_ts": "disabled"}) as target:
            output = target.add_stream_from_template(source.streams.video[0])
            for packet in source.demux(video=0):
                if packet.dts is None:
                    continue
                offset = int(start_offset / packet.time_base)
                packet.pts += offset
                packet.dts += offset
                packet.stream = output
                target.mux(packet)
        path = shifted_path
    with av.open(str(path)) as container:
        frames = list(container.decode(video=0))
        pts = np.array([float(frame.pts * frame.time_base) for frame in frames])
        assert pts[0] == pytest.approx(float(start_offset), abs=0.001)
        pts -= float(start_offset)
        reference = torch.stack([torch.from_numpy(frame.to_ndarray(format="rgb24")).permute(2, 0, 1)
                                 for frame in frames]).float() / 255
    query = [pts[70] + 0.00001, pts[3], pts[70] + 0.00001, pts[30], pts[-1]]
    if start_offset:
        query.insert(1, 0.0)
    expected = reference[np.abs(np.asarray(query)[:, None] - pts).argmin(1)]
    actual = decode_video_frames(path, query, 0.001, "pyav", 1)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    goal = decode_video_frames(path, [pts[70]], 0.001, "pyav", 1)
    torch.testing.assert_close(goal[0], reference[70], rtol=0, atol=0)


class FakeContainer:
    def __init__(self, length, offset=0, *, time_base=Fraction(1, 30), step=1, start_time=0):
        self.length, self.offset = length, offset
        self.time_base, self.step = time_base, step
        self.streams = SimpleNamespace(video=[SimpleNamespace(time_base=time_base,
                                                              start_time=start_time,
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
            time_base = owner.time_base
            def __init__(self, index):
                self.pts, self.index = index * owner.step + owner.offset, index
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
    monkeypatch.setattr("lerobot.policies.wsa1_5.utils.video.av.open", lambda _: container)
    actual = decode_video_frames("unused", [100.0, 0.0, 100.0], 0.001, "pyav", 1)
    assert container.converted == [0, 3000]
    assert container.peak <= 3 and container.closed
    assert container.streams.video[0].codec_context.thread_count == 1
    torch.testing.assert_close(actual[:, 0, 0, 0], torch.tensor([3000 % 256, 0, 3000 % 256]) / 255)


def test_long_offsets_keep_float64_timestamp_precision(monkeypatch):
    container = FakeContainer(4, offset=1200 * 30)
    monkeypatch.setattr("lerobot.policies.wsa1_5.utils.video.av.open", lambda _: container)
    actual = decode_video_frames("unused", [1200 + 1 / 30 + 0.00005], 0.0001, "pyav", 1)
    torch.testing.assert_close(actual[0], torch.full_like(actual[0], 1 / 255))
    assert container.closed


def test_bad_timestamp_still_raises_and_closes_decoder(monkeypatch):
    container = FakeContainer(3)
    monkeypatch.setattr("lerobot.policies.wsa1_5.utils.video.av.open", lambda _: container)
    with pytest.raises(ValueError, match="exceed tolerance"):
        decode_video_frames("unused", [1.0], 0.001, "pyav", 1)
    assert container.closed


def test_start_pts_rebases_all_queries_without_duplicating_first_frame(monkeypatch):
    container = FakeContainer(40, offset=507, time_base=Fraction(1, 15360), step=512, start_time=507)
    monkeypatch.setattr("lerobot.policies.wsa1_5.utils.video.av.open", lambda _: container)
    query = [4 / 30, 0.0, 8 / 30, 0.0, 1 / 30]
    actual = decode_video_frames("unused", query, 0.001, "pyav", 1)
    torch.testing.assert_close(actual[:, 0, 0, 0], torch.tensor([4, 0, 8, 0, 1]) / 255)
    assert container.converted == [0, 1, 4, 8] and container.closed
    assert container.seek_offset == 507


@pytest.mark.parametrize("query", [0.0, 8 / 30])
def test_single_goal_query_initializes_previous_for_the_first_decoded_frame(monkeypatch, query):
    container = FakeContainer(40, offset=507, time_base=Fraction(1, 15360), step=512, start_time=507)
    decode = container.decode
    monkeypatch.setattr(container, "decode", lambda stream: (
        frame for frame in decode(stream) if frame.pts >= container.seek_offset))
    monkeypatch.setattr("lerobot.policies.wsa1_5.utils.video.av.open", lambda _: container)
    actual = decode_video_frames("unused", [query], 0.001, "pyav", 1)
    torch.testing.assert_close(actual[0], torch.full_like(actual[0], round(query * 30) / 255))
    assert container.converted == [round(query * 30)] and container.closed


@pytest.mark.parametrize("offset,query", [(2, [0.0]), (1, [-1 / 30]), (1, [0.0, 0.5])])
def test_timeline_alignment_does_not_hide_larger_or_other_timestamp_errors(monkeypatch, offset, query):
    container = FakeContainer(4, offset=offset)
    monkeypatch.setattr("lerobot.policies.wsa1_5.utils.video.av.open", lambda _: container)
    with pytest.raises(ValueError, match="exceed tolerance"):
        decode_video_frames("unused", query, 0.001, "pyav", 1)
    assert container.closed


def test_interior_timestamp_mismatch_is_still_rejected(monkeypatch):
    container = FakeContainer(4, offset=30 * 1200 + 1)
    monkeypatch.setattr("lerobot.policies.wsa1_5.utils.video.av.open", lambda _: container)
    with pytest.raises(ValueError, match="exceed tolerance"):
        decode_video_frames("unused", [1200.0], 0.001, "pyav", 1)
    assert container.seek_offset == 1200 * 30 and container.closed


def test_missing_start_timestamp_is_not_silently_replaced_by_zero(monkeypatch):
    container = FakeContainer(4, start_time=None)
    monkeypatch.setattr("lerobot.policies.wsa1_5.utils.video.av.open", lambda _: container)
    with pytest.raises(ValueError, match="no start timestamp"):
        decode_video_frames("unused", [0.0], 0.001, "pyav", 1)
    assert container.closed


def test_empty_stream_raises_a_decode_error_instead_of_an_uninitialized_variable(monkeypatch):
    container = FakeContainer(0)
    monkeypatch.setattr("lerobot.policies.wsa1_5.utils.video.av.open", lambda _: container)
    with pytest.raises(ValueError, match="No video frames decoded"):
        decode_video_frames("unused", [0.0], 0.001, "pyav", 1)
    assert container.closed


def test_other_backends_keep_the_existing_decoder(monkeypatch):
    calls = []
    def decode(*args):
        calls.append(args)
        return torch.ones(1, 3, 8, 8)
    monkeypatch.setattr("lerobot.policies.wsa1_5.utils.video.decode_other_backend", decode)
    result = decode_video_frames("unused", [0.0], 0.001, "torchcodec", 2)
    assert result.shape == (1, 3, 8, 8)
    assert calls == [("unused", [0.0], 0.001, "torchcodec")]
