# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np

from granite_speech.segmenter import FixedWindowSegmenter, VadSegmenter


def mono(seconds: float, sample_rate: int = 100) -> np.ndarray:
    return np.zeros((1, int(seconds * sample_rate)), dtype=np.float32)


def add_tone(wav: np.ndarray, start: float, end: float, *, sample_rate: int = 100) -> None:
    wav[:, int(start * sample_rate) : int(end * sample_rate)] = 1.0


def test_fixed_window_segmenter_uses_waveform_length():
    segments = FixedWindowSegmenter(3.0).segment(mono(8.0), 100)

    assert [(segment.start, segment.end) for segment in segments] == [
        (0.0, 3.0),
        (3.0, 6.0),
        (6.0, 8.0),
    ]


def test_fixed_window_segmenter_keeps_total_sample_call_compatibility():
    segments = FixedWindowSegmenter(3.0).segment(800, 100)

    assert [(segment.start, segment.end) for segment in segments] == [
        (0.0, 3.0),
        (3.0, 6.0),
        (6.0, 8.0),
    ]


def test_vad_segmenter_splits_speech_regions_and_applies_padding():
    wav = mono(10.0)
    add_tone(wav, 1.0, 3.0)
    add_tone(wav, 6.0, 8.0)

    segments = VadSegmenter(
        chunk_length=30.0,
        threshold=0.5,
        min_speech_duration=0.1,
        min_silence_duration=0.5,
        speech_pad=0.1,
        frame_length=0.1,
        frame_step=0.1,
    ).segment(wav, 100)

    assert [(segment.start, segment.end) for segment in segments] == [
        (0.9, 3.1),
        (5.9, 8.1),
    ]


def test_vad_segmenter_merges_short_silence_between_speech_regions():
    wav = mono(5.0)
    add_tone(wav, 1.0, 2.0)
    add_tone(wav, 2.3, 3.0)

    segments = VadSegmenter(
        chunk_length=30.0,
        threshold=0.5,
        min_speech_duration=0.1,
        min_silence_duration=0.5,
        speech_pad=0.0,
        frame_length=0.1,
        frame_step=0.1,
    ).segment(wav, 100)

    assert [(segment.start, segment.end) for segment in segments] == [(1.0, 3.0)]


def test_vad_segmenter_splits_long_speech_at_chunk_length():
    wav = mono(70.0)
    add_tone(wav, 0.0, 70.0)

    segments = VadSegmenter(
        chunk_length=30.0,
        threshold=0.5,
        min_speech_duration=0.1,
        min_silence_duration=0.5,
        speech_pad=0.0,
        frame_length=0.1,
        frame_step=0.1,
    ).segment(wav, 100)

    assert [(segment.start, segment.end) for segment in segments] == [
        (0.0, 30.0),
        (30.0, 60.0),
        (60.0, 70.0),
    ]


def test_vad_segmenter_returns_no_segments_for_silence():
    segments = VadSegmenter(chunk_length=30.0).segment(mono(10.0), 100)

    assert segments == []
