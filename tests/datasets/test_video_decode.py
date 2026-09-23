#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for video frame decoding in ``lerobot.datasets.video_utils``.

Covers three things:
- each backend (pyav, torchcodec) selects the correct frames, in order, verified against an
  oracle independent of the decoder: committed indexed-clip artifacts whose frame ``i`` is
  painted the constant value ``i``, so decoded content equals the known index;
- the pyav backend decodes only the frames it needs: it seeks to the keyframe at or before the
  target and stops at the target, without an extra GOP or an extra frame;
- ``VideoDecoderCache`` LRU bounding + file-handle release (torchcodec decode path).
"""

import importlib.util
import logging
import shutil
from pathlib import Path

import pytest

pytest.importorskip("av", reason="av is required (install lerobot[dataset])")

import av  # noqa: E402
import torch  # noqa: E402

from lerobot.datasets.video_utils import (  # noqa: E402
    VideoDecoderCache,
    decode_video_frames,
    decode_video_frames_pyav,
)

FPS = 30

TEST_ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts" / "encoded_videos"
SRC_CLIP = TEST_ARTIFACTS_DIR / "clip_4frames.mp4"
# 60-frame 16x16 lossless-RGB clip (libx264rgb, qp=0, gop=2) with frame ``i`` painted the constant
# value ``i``. Lossless RGB avoids the YUV<->RGB range shift, so decoded pixels equal the frame index exactly.
# Every frame changes completely, so x264 scene-cut detection codes each one as a keyframe.
INDEXED_CLIP = TEST_ARTIFACTS_DIR / "indexed_clip_60frames.mp4"
# 90-frame 16x16 lossless-RGB HEVC clip with frame ``i`` painted the constant value ``i``, keyframes
# at 0, 30 and 60, and open GOPs: the B-frames displayed just before keyframes 30 and 60 are decoded
# after them and reference them, so they cannot be decoded by seeking to that keyframe. Generated with:
#   ffmpeg -f lavfi -i "color=black:s=16x16:r=30,format=gbrp,geq=r=N:g=N:b=N" -frames:v 90 \
#     -c:v libx265 -pix_fmt gbrp -x265-params \
#     "lossless=1:keyint=30:min-keyint=30:scenecut=0:open-gop=1:bframes=4:b-adapt=0:info=0" \
#     indexed_clip_hevc_open_gop_90frames.mp4
OPEN_GOP_CLIP = TEST_ARTIFACTS_DIR / "indexed_clip_hevc_open_gop_90frames.mp4"
# Open-GOP leading pictures of ``OPEN_GOP_CLIP``: displayed before a keyframe, decoded after it.
OPEN_GOP_LEADING_PICTURES = frozenset({26, 27, 28, 29, 56, 57, 58, 59})

torchcodec_required = pytest.mark.skipif(
    importlib.util.find_spec("torchcodec") is None,
    reason="torchcodec not available",
)


def _make_distinct_clips(tmp_path: Path, n: int) -> list[Path]:
    """Copy the small reference mp4 to ``n`` distinct paths.

    The cache keys on absolute path, so distinct paths force distinct cache entries
    even though the file contents are identical.
    """
    assert SRC_CLIP.exists(), f"missing test artifact {SRC_CLIP}"
    paths = []
    for i in range(n):
        dst = tmp_path / f"clip_{i:04d}.mp4"
        shutil.copyfile(SRC_CLIP, dst)
        paths.append(dst)
    return paths


def test_open_gop_clip_has_leading_pictures():
    """Guard the artifact: ``OPEN_GOP_CLIP`` must keep the structure the regression cases rely on.

    Frames may be reordered (``has_b_frames`` is set), and the leading pictures of each keyframe
    after the first are exactly ``OPEN_GOP_LEADING_PICTURES``.
    """
    with av.open(str(OPEN_GOP_CLIP)) as container:
        stream = container.streams.video[0]
        assert stream.codec_context.has_b_frames
        packets = [
            (round(float(packet.pts * stream.time_base) * FPS), packet.is_keyframe)
            for packet in container.demux(stream)
            if packet.pts is not None
        ]
    leading = set()
    for position, (keyframe_index, is_keyframe) in enumerate(packets):
        if is_keyframe and keyframe_index > 0:
            leading.update(index for index, _ in packets[position + 1 :] if index < keyframe_index)
    assert leading == OPEN_GOP_LEADING_PICTURES


_FRAME_SELECTION_INDICES = [
    [3, 50, 20, 58, 10],  # far apart, unsorted -> a seek per keyframe cluster
    list(range(20, 31)),  # contiguous window -> single forward pass
    [5, 6, 7, 40, 41, 42],  # two contiguous clusters far apart
    [58, 40, 20, 0],  # descending
    [30, 30, 31],  # duplicates
    [42],  # single frame
]
# Each set requests an open-GOP leading picture, which seeking to the target lands after.
_OPEN_GOP_INDICES = [
    [0, 58],  # far-apart clusters, the second ends on a leading picture
    [10, 59],
    [28, 29],  # contiguous leading pictures
    [29, 59],  # leading pictures of both keyframes
    [58],  # a leading picture alone
]
# torchcodec with LeRobot's seek_mode="approximate" returns wrong frames when it has to seek to an
# open-GOP leading picture (seek_mode="exact" does not). A window decoded forward from frame 20
# reaches leading pictures 26-29 correctly, so it is not listed.
_TORCHCODEC_OPEN_GOP_FAILURES = {
    tuple(indices) for indices in _OPEN_GOP_INDICES + [[3, 50, 20, 58, 10], [58, 40, 20, 0]]
}


@pytest.mark.parametrize(
    "backend",
    ["pyav", pytest.param("torchcodec", marks=torchcodec_required)],
)
@pytest.mark.parametrize(
    ("clip", "indices"),
    [
        pytest.param(INDEXED_CLIP, indices, id=f"intra-{'_'.join(map(str, indices))}")
        for indices in _FRAME_SELECTION_INDICES
    ]
    + [
        pytest.param(OPEN_GOP_CLIP, indices, id=f"open_gop-{'_'.join(map(str, indices))}")
        for indices in _FRAME_SELECTION_INDICES + _OPEN_GOP_INDICES
    ],
)
def test_decode_selects_correct_frames(request, backend, clip, indices):
    """Each backend returns exactly the requested frames, in order.

    Both clips paint frame ``i`` the constant value ``i``, so the expected content is the frame
    index itself -- an oracle independent of the decoder under test.
    """
    if backend == "torchcodec" and clip == OPEN_GOP_CLIP and tuple(indices) in _TORCHCODEC_OPEN_GOP_FAILURES:
        request.applymarker(
            pytest.mark.xfail(
                strict=True,
                reason="torchcodec seek_mode='approximate' returns wrong frames when seeking to open-GOP leading pictures",
            )
        )
    tolerance_s = 1.0 / FPS
    frames = decode_video_frames(clip, [i / FPS for i in indices], tolerance_s, backend, return_uint8=True)
    expected = torch.tensor([i % 256 for i in indices], dtype=torch.uint8).view(-1, 1, 1, 1).expand_as(frames)
    assert frames.shape[0] == len(indices)
    assert torch.equal(frames, expected)


@pytest.mark.parametrize(
    ("clip", "index", "query_offset_s", "tolerance_s", "expected_decoded"),
    [
        # A keyframe target decodes only itself, not the previous keyframe's GOP too.
        pytest.param(INDEXED_CLIP, 42, 0.0, 1e-4, [42], id="keyframe_target"),
        # A non-keyframe target decodes from its keyframe up to itself.
        pytest.param(OPEN_GOP_CLIP, 31, 0.0, 1e-4, [30, 31], id="non_keyframe_target"),
        # Query timestamps carry float error. A query slightly above the frame's pts still stops at
        # that frame instead of decoding the next one.
        pytest.param(OPEN_GOP_CLIP, 31, 1e-6, 1e-4, [30, 31], id="query_above_frame_pts"),
        # With a one-frame tolerance, the stop margin stays capped at half a frame, so decoding does
        # not stop one frame early and return the previous frame.
        pytest.param(OPEN_GOP_CLIP, 31, 0.0, 1.0 / FPS, [30, 31], id="one_frame_tolerance"),
    ],
)
def test_pyav_decodes_only_needed_frames(caplog, clip, index, query_offset_s, tolerance_s, expected_decoded):
    """pyav decodes from the keyframe at or before the target up to the target, and no further.

    Decoded frames are observed through ``log_loaded_timestamps``, which logs one record per frame.
    """
    caplog.set_level(logging.INFO, logger="lerobot.datasets.video_utils")
    frames = decode_video_frames_pyav(
        clip,
        [index / FPS + query_offset_s],
        tolerance_s,
        log_loaded_timestamps=True,
        return_uint8=True,
    )
    prefix = "frame loaded at timestamp="
    decoded = [
        round(float(record.getMessage().removeprefix(prefix)) * FPS)
        for record in caplog.records
        if record.getMessage().startswith(prefix)
    ]
    assert decoded == expected_decoded
    assert torch.equal(frames, torch.full_like(frames, index))


@torchcodec_required
class TestVideoDecoderCacheBounded:
    """LRU bounding + file-handle release, added to prevent unbounded growth when iterating over
    datasets with many distinct video files (observed: ~35 GB anon-rss per DataLoader worker on an
    8 k-file dataset)."""

    def test_default_cache_is_bounded(self):
        """The default cache must have a finite ``max_size`` to bound RSS growth."""
        cache = VideoDecoderCache()
        assert cache.max_size is not None, "default cache must be bounded"
        assert cache.max_size > 0

    def test_size_capped_at_max_size(self, tmp_path):
        """``get_decoder`` for >``max_size`` distinct paths must NOT grow without bound."""
        paths = _make_distinct_clips(tmp_path, n=5)
        cache = VideoDecoderCache(max_size=2)
        for p in paths:
            cache.get_decoder(p)
        assert cache.size() == 2

    def test_evicts_least_recently_used(self, tmp_path):
        """Re-accessing an entry must promote it; the LRU entry is the one evicted."""
        paths = _make_distinct_clips(tmp_path, n=3)
        cache = VideoDecoderCache(max_size=2)

        cache.get_decoder(paths[0])
        cache.get_decoder(paths[1])
        cache.get_decoder(paths[0])  # promote paths[0] to MRU; paths[1] is now LRU
        cache.get_decoder(paths[2])  # should evict paths[1]

        assert str(paths[0]) in cache  # MRU stays
        assert str(paths[1]) not in cache  # LRU evicted
        assert str(paths[2]) in cache  # newest stays

    def test_eviction_closes_file_handle(self, tmp_path):
        """Evicting an entry must close its fsspec file handle (otherwise we leak FDs)."""
        paths = _make_distinct_clips(tmp_path, n=2)
        cache = VideoDecoderCache(max_size=1)

        cache.get_decoder(paths[0])
        # Reach into the cache to capture the handle before it is evicted. This is
        # the only assertion in the suite that touches a private attribute, and it
        # is the most direct way to prove the file descriptor is actually released.
        evicted_handle = cache._cache[str(paths[0])][1]
        assert evicted_handle.closed is False

        cache.get_decoder(paths[1])  # forces eviction of paths[0]

        assert evicted_handle.closed is True

    def test_clear_closes_all_file_handles(self, tmp_path):
        """``clear()`` must close every cached file handle."""
        paths = _make_distinct_clips(tmp_path, n=3)
        cache = VideoDecoderCache(max_size=10)

        for p in paths:
            cache.get_decoder(p)
        handles = [entry[1] for entry in cache._cache.values()]
        assert all(not h.closed for h in handles)

        cache.clear()

        assert cache.size() == 0
        assert all(h.closed for h in handles)

    def test_hit_does_not_reopen_or_evict(self, tmp_path):
        """A cache hit must return the same decoder instance without touching the cap."""
        paths = _make_distinct_clips(tmp_path, n=1)
        cache = VideoDecoderCache(max_size=2)

        first = cache.get_decoder(paths[0])
        second = cache.get_decoder(paths[0])

        assert first is second
        assert cache.size() == 1

    def test_unbounded_when_max_size_none(self, tmp_path):
        """``max_size=None`` preserves the legacy unbounded behaviour."""
        paths = _make_distinct_clips(tmp_path, n=4)
        cache = VideoDecoderCache(max_size=None)
        for p in paths:
            cache.get_decoder(p)
        assert cache.size() == 4

    def test_env_var_overrides_default(self, tmp_path, monkeypatch):
        """``LEROBOT_VIDEO_DECODER_CACHE_SIZE`` env var sets the default ``max_size``."""
        monkeypatch.setenv("LEROBOT_VIDEO_DECODER_CACHE_SIZE", "3")
        cache = VideoDecoderCache()
        assert cache.max_size == 3

        paths = _make_distinct_clips(tmp_path, n=5)
        for p in paths:
            cache.get_decoder(p)
        assert cache.size() == 3
