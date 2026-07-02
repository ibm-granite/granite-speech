from __future__ import annotations

import numpy as np
import pytest

from granite_speech._backends import BackendCapabilities
from granite_speech._backends.fake import FakeBackend
from granite_speech.errors import InvalidArgumentError
from granite_speech.chunking import ChunkingOptions, resolve_max_new_tokens, transcribe_chunks


def options(**overrides) -> ChunkingOptions:
    values = {
        "task": "transcribe",
        "language": None,
        "target_language": None,
        "prompt": "<|audio|>transcribe the speech.",
        "instruction": "transcribe the speech.",
        "keyword_biases": (),
        "max_new_tokens": None,
        "num_beams": 1,
        "temperature": 0.0,
        "chunk_length": 30.0,
        "chunk_overlap": 0.0,
        "verbose": None,
    }
    values.update(overrides)
    return ChunkingOptions(**values)


def capabilities(max_reliable_audio_seconds: float | None = None) -> BackendCapabilities:
    return BackendCapabilities(
        max_reliable_audio_seconds=max_reliable_audio_seconds,
        supports_word_timing_output=False,
        supports_speaker_attribution_output=False,
        supports_batch=False,
        supports_translation=True,
    )


def test_thirty_minute_long_form_audio_transcribes_in_order():
    backend = FakeBackend(lambda _req, index: f"chunk-{index:02d}")
    wav = np.zeros((1, 30 * 60), dtype=np.float32)

    result = transcribe_chunks(
        wav,
        backend=backend,
        sample_rate=1,
        options=options(chunk_length=30.0),
    )

    assert result["warnings"] == []
    assert len(result["segments"]) == 60
    assert len(backend.calls) == 60
    assert result["text"].startswith("chunk-00 chunk-01")
    assert result["text"].endswith("chunk-58 chunk-59")
    assert result["segments"][0] == {"start": 0.0, "end": 30.0, "text": "chunk-00"}
    assert result["segments"][-1] == {"start": 1770.0, "end": 1800.0, "text": "chunk-59"}
    assert all(call.wav.shape == (1, 30) for call in backend.calls)
    assert all(call.max_new_tokens == 200 for call in backend.calls)


def test_auto_max_new_tokens_scales_with_window_length():
    assert resolve_max_new_tokens(None, 15.0) == 200
    assert resolve_max_new_tokens(None, 30.0) == 200
    assert resolve_max_new_tokens(None, 45.0) == 300
    assert resolve_max_new_tokens(None, 60.0) == 400
    assert resolve_max_new_tokens(64, 60.0) == 64


def test_auto_max_new_tokens_uses_backend_clamped_chunk_length():
    backend = FakeBackend(
        lambda _req, index: f"chunk-{index}",
        capabilities=capabilities(max_reliable_audio_seconds=60.0),
    )

    result = transcribe_chunks(
        np.zeros((1, 180), dtype=np.float32),
        backend=backend,
        sample_rate=1,
        options=options(chunk_length=120.0),
    )

    assert [call.max_new_tokens for call in backend.calls] == [400, 400, 400]
    assert [(segment["start"], segment["end"]) for segment in result["segments"]] == [
        (0.0, 60.0),
        (60.0, 120.0),
        (120.0, 180.0),
    ]
    assert result["warnings"] == [
        {
            "type": "chunk_clamp",
            "message": (
                "requested chunk_length 120.000s exceeds backend "
                "'fake' reliable limit 60.000s; clamped"
            ),
            "requested": 120.0,
            "applied": 60.0,
        }
    ]


def test_word_timestamp_parsing_uses_segment_offsets():
    backend = FakeBackend(["first [T:50]", "second [T:25]"])

    result = transcribe_chunks(
        np.zeros((1, 2), dtype=np.float32),
        backend=backend,
        sample_rate=1,
        options=options(chunk_length=1.0, prompt_mode="word_timestamps"),
    )

    assert result["text"] == "first second"
    assert result["words"] == [
        {"word": "first", "start": 0.0, "end": 0.5},
        {"word": "second", "start": 1.0, "end": 1.25},
    ]
    assert result["segments"][1]["words"] == [
        {"word": "second", "start": 1.0, "end": 1.25}
    ]


def test_vad_segmentation_skips_silence_between_speech_regions():
    backend = FakeBackend(lambda _req, index: f"speech-{index}")
    wav = np.zeros((1, 1000), dtype=np.float32)
    wav[:, 100:300] = 1.0
    wav[:, 700:900] = 1.0

    result = transcribe_chunks(
        wav,
        backend=backend,
        sample_rate=100,
        options=options(
            segmentation="vad",
            chunk_length=30.0,
            vad_threshold=0.5,
            vad_min_speech_duration=0.1,
            vad_min_silence_duration=0.5,
            vad_speech_pad=0.1,
        ),
    )

    assert result["text"] == "speech-0 speech-1"
    assert result["warnings"] == []
    assert [(segment["start"], segment["end"]) for segment in result["segments"]] == [
        (0.88, 3.12),
        (6.88, 9.12),
    ]
    assert len(backend.calls) == 2
    assert all(call.max_new_tokens == 200 for call in backend.calls)


def test_vad_segmentation_rejects_chunk_overlap():
    with pytest.raises(InvalidArgumentError, match="chunk_overlap"):
        transcribe_chunks(
            np.zeros((1, 100), dtype=np.float32),
            backend=FakeBackend([]),
            sample_rate=100,
            options=options(segmentation="vad", chunk_overlap=1.0),
        )
