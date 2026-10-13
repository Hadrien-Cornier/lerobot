# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from lerobot.streaming.manifest import EpisodeVideoManifest, VideoFileRecord
from lerobot.streaming.mp4 import (
    _compact_integer_array,
    parse_mp4_index,
    synthesize_mp4,
    synthesized_mp4_size,
)
from lerobot.streaming.sidecar import SidecarSpec
from tests.datasets.test_episode_video_streaming import _minimal_mp4


@pytest.mark.parametrize(
    "values,dtype",
    [
        ([0, 255], "uint8"),
        ([0, 256, 65535], "uint16"),
        ([0, 65536, 2**32 - 1], "uint32"),
        ([0, 2**32, 2**63 - 1], "int64"),
        ([-128, 127], "int8"),
        ([-129, 32767], "int16"),
        ([-32769, 2**31 - 1], "int32"),
        ([-(2**31) - 1, 2**63 - 1], "int64"),
    ],
)
def test_integer_storage_preserves_width_boundaries(values: list[int], dtype: str) -> None:
    source = np.array(values, dtype=np.int64)
    compact = _compact_integer_array(source)
    assert compact.dtype == np.dtype(dtype)
    np.testing.assert_array_equal(compact, source)
    assert compact.nbytes <= source.nbytes


def test_empty_integer_storage_preserves_the_array() -> None:
    source = np.empty(0, dtype=np.int64)
    assert _compact_integer_array(source) is source


def test_compact_offsets_add_sizes_without_unsigned_overflow() -> None:
    index = parse_mp4_index("boundary.mp4", _minimal_mp4([65520, 65525, 65530]))
    assert index.sample_offsets.dtype == np.uint16
    span = index.sample_slice(0, 2, keyframe_pad_s=0, keyframe_pad_fraction=0)
    assert span.byte_offset == 65520
    assert span.byte_length == 20
    mini = synthesize_mp4(index, span, b"x" * span.byte_length)
    assert len(mini) == synthesized_mp4_size(index, span)
    round_trip = parse_mp4_index("mini.mp4", mini)
    np.testing.assert_array_equal(round_trip.sample_offsets - round_trip.mdat_payload_offset, [0, 5, 10])


def test_compact_offsets_preserve_co64_and_reordered_chunks() -> None:
    offsets = [2**32 - 5, 2**32 + 40, 2**32 + 15]
    index = parse_mp4_index("large.mp4", _minimal_mp4(offsets, use_co64=True))
    np.testing.assert_array_equal(index.sample_offsets, offsets)
    assert index.sample_offsets.dtype == np.int64
    span = index.sample_slice(0, 2, keyframe_pad_s=0, keyframe_pad_fraction=0)
    assert span.byte_offset == offsets[0]
    assert span.byte_length == 55


def test_compact_sync_samples_do_not_wrap_one_based_indices() -> None:
    prototype = parse_mp4_index("prototype.mp4", _minimal_mp4([10000, 10010, 10020]))
    index = replace(
        prototype,
        sample_pts=np.arange(256, dtype=np.float64),
        sample_durations=np.full(256, 1, dtype=np.uint8),
        sample_composition_offsets=np.zeros(256, dtype=np.uint8),
        sample_sizes=np.ones(256, dtype=np.uint8),
        sample_offsets=np.arange(10000, 10256, dtype=np.uint16),
        sync_samples=np.array([0, 255], dtype=np.uint8),
    )
    span = index.sample_slice(0, 255, keyframe_pad_s=0, keyframe_pad_fraction=0)
    mini = synthesize_mp4(index, span, b"x" * span.byte_length)
    assert len(mini) == synthesized_mp4_size(index, span)
    round_trip = parse_mp4_index("mini.mp4", mini)
    np.testing.assert_array_equal(round_trip.sync_samples, [0, 255])


def test_compact_integer_slices_preserve_negative_composition_times() -> None:
    prototype = parse_mp4_index("prototype.mp4", _minimal_mp4([10000, 10010, 10020]))
    index = replace(
        prototype,
        sample_pts=np.array([0.0, 0.5, 1.0]),
        sample_durations=np.array([250, 250, 250], dtype=np.uint8),
        sample_composition_offsets=np.array([-128, 127, -128], dtype=np.int8),
        sync_samples=np.array([0, 1, 2], dtype=np.uint8),
    )
    wide = replace(
        index,
        sample_durations=index.sample_durations.astype(np.int64),
        sample_composition_offsets=index.sample_composition_offsets.astype(np.int64),
        sync_samples=index.sync_samples.astype(np.int64),
    )
    span = index.sample_slice(0, 1, keyframe_pad_s=0, keyframe_pad_fraction=0)
    payload = b"x" * span.byte_length
    assert synthesize_mp4(index, span, payload) == synthesize_mp4(wide, span, payload)
    assert synthesized_mp4_size(index, span) == synthesized_mp4_size(wide, span)


def test_sidecar_keeps_compact_arrays_and_exact_float_timestamps(tmp_path: Path) -> None:
    index = parse_mp4_index("test.mp4", _minimal_mp4([10000, 10050, 10025]))
    index = replace(index, sample_pts=np.array([0.0, np.nextafter(0.1, 1.0), -0.0]))
    record = VideoFileRecord("test.mp4", 10128, index)
    spec = SidecarSpec("test/compact", "rev", str(tmp_path), ((record.file_path, record.file_size),))
    output = tmp_path / "compact.npz"
    EpisodeVideoManifest.save_file_sidecar(output, [record], spec=spec)
    assert EpisodeVideoManifest.validate_file_sidecar(output, spec)
    loaded = EpisodeVideoManifest.load_file_sidecar(output)[record.file_path].mp4
    assert loaded.sample_pts.tobytes() == index.sample_pts.tobytes()
    for name in (
        "sample_durations",
        "sample_composition_offsets",
        "sample_sizes",
        "sample_offsets",
        "sync_samples",
    ):
        actual, expected = getattr(loaded, name), getattr(index, name)
        assert actual.dtype == expected.dtype
        assert not actual.flags.writeable
        np.testing.assert_array_equal(actual, expected)
