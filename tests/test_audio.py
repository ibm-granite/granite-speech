from __future__ import annotations

import numpy as np
import pytest

from granite_speech.audio import load_audio
from granite_speech.errors import InvalidArgumentError


def test_load_audio_accepts_channel_last_and_downmixes():
    left = np.ones(8000, dtype=np.float32)
    right = np.zeros(8000, dtype=np.float32)
    stereo = np.stack([left, right], axis=1)

    audio = load_audio(stereo, sample_rate=8000)

    assert audio.sample_rate == 16000
    assert audio.wav.shape == (1, 16000)
    assert np.allclose(audio.wav.mean(), 0.5, atol=0.05)


def test_load_audio_rejects_ambiguous_2d_shape():
    with pytest.raises(InvalidArgumentError, match="ambiguous"):
        load_audio(np.zeros((10, 10), dtype=np.float32), sample_rate=16000)


def test_load_audio_rejects_non_finite_samples():
    with pytest.raises(InvalidArgumentError, match="NaN"):
        load_audio(np.array([0.0, np.nan], dtype=np.float32), sample_rate=16000)
