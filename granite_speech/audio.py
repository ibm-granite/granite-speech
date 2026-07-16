# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeAlias

import numpy as np

from .errors import AudioDecodeError, InvalidArgumentError

# The rate load_audio resamples every input to. Deliberately separate from
# _backends.SAMPLE_RATE (the backend-contract constant): this is the audio-layer
# target, that is the interface backends validate against. They hold the same
# value but belong to different layers, so the audio module does not import from
# _backends (which would invert the dependency direction).
SAMPLE_RATE = 16000
DEFAULT_MAX_AUDIO_SECONDS = 4 * 60 * 60
ENV_MAX_AUDIO_SECONDS = "GRANITE_SPEECH_MAX_AUDIO_SECONDS"

# Accepted audio inputs: a filesystem path (str/Path) or an in-memory waveform.
# np.ndarray is the concrete array type; array-likes that convert cleanly via
# np.asarray (e.g. torch tensors) are also accepted at runtime without a torch
# import. Kept as a named alias so all public signatures stay in sync.
AudioInput: TypeAlias = str | Path | np.ndarray


@dataclass(frozen=True)
class AudioData:
    wav: np.ndarray
    sample_rate: int

    @property
    def duration(self) -> float:
        return self.wav.shape[-1] / self.sample_rate if self.sample_rate else 0.0


def load_audio(
    audio: AudioInput,
    *,
    sample_rate: int | None = None,
    max_audio_seconds: float | None = None,
) -> AudioData:
    """Load and normalize audio to mono, 16 kHz, float32.

    Path inputs infer their sample rate from the file, so passing ``sample_rate``
    with a path is an error; raw array/tensor inputs require ``sample_rate``. The
    waveform is validated against ``max_audio_seconds`` (a decompression-bomb
    guard, checked both before and after resampling), converted to float32,
    downmixed to mono, and resampled to :data:`SAMPLE_RATE` if needed. Returns an
    :class:`AudioData` with a contiguous float32 waveform.
    """
    if isinstance(audio, (str, Path)):
        if sample_rate is not None:
            raise InvalidArgumentError(
                "sample_rate is inferred for path inputs and must not be set"
            )
        wav, sr = _load_audio_path(Path(audio))
    else:
        if sample_rate is None:
            raise InvalidArgumentError(
                "sample_rate is required for raw array or tensor audio input"
            )
        wav = _array_to_channels_first(audio)
        sr = sample_rate

    _validate_duration(wav, sr, max_audio_seconds)
    wav = _to_float32(wav)
    wav = _downmix_to_mono(wav)
    if sr != SAMPLE_RATE:
        wav = _resample(wav, sr, SAMPLE_RATE)
        sr = SAMPLE_RATE
    _validate_duration(wav, sr, max_audio_seconds)
    return AudioData(wav=np.ascontiguousarray(wav, dtype=np.float32), sample_rate=sr)


def _load_audio_path(path: Path) -> tuple[np.ndarray, int]:
    expanded = path.expanduser()
    if not expanded.exists():
        raise AudioDecodeError(f"audio file does not exist: {expanded}")

    soundfile_error: Exception | None = None
    try:
        import soundfile as sf

        data, sr = sf.read(str(expanded), dtype="float32", always_2d=True)
        return np.asarray(data, dtype=np.float32).T, int(sr)
    except Exception as exc:  # pragma: no cover - exercised only when codecs are unavailable
        soundfile_error = exc

    try:
        import torchaudio

        tensor, sr = torchaudio.load(str(expanded), normalize=True)
        return tensor.detach().cpu().numpy().astype(np.float32, copy=False), int(sr)
    except Exception as exc:
        raise AudioDecodeError(
            f"could not decode audio file {expanded}; tried soundfile and torchaudio. "
            "For container formats such as M4A/AAC, transcode to WAV or FLAC first. "
            f"soundfile error: {soundfile_error}; torchaudio error: {exc}"
        ) from exc


def _array_to_channels_first(audio: Any) -> np.ndarray:
    if _looks_like_torch_tensor(audio):
        audio = audio.detach().cpu().numpy()
    arr = np.asarray(audio)
    if arr.ndim == 0:
        raise InvalidArgumentError("audio array must have at least one sample")
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    if arr.ndim != 2:
        raise InvalidArgumentError(
            f"audio array must be mono or 2-D channel-first/channel-last, got shape {arr.shape}"
        )

    rows, cols = arr.shape
    if rows == 1:
        return arr
    if cols == 1:
        return arr.T
    if rows <= 8 and cols > rows:
        return arr
    if cols <= 8 and rows > cols:
        return arr.T
    raise InvalidArgumentError(
        f"ambiguous 2-D audio shape {arr.shape}; pass mono, channel-first, or channel-last audio"
    )


def _looks_like_torch_tensor(value: Any) -> bool:
    return hasattr(value, "detach") and hasattr(value, "cpu") and hasattr(value, "numpy")


def _to_float32(wav: np.ndarray) -> np.ndarray:
    if np.issubdtype(wav.dtype, np.floating):
        out = wav.astype(np.float32, copy=False)
    elif np.issubdtype(wav.dtype, np.integer):
        info = np.iinfo(wav.dtype)
        scale = max(abs(info.min), abs(info.max))
        out = wav.astype(np.float32) / float(scale)
    else:
        raise InvalidArgumentError(f"unsupported audio dtype {wav.dtype}")

    if not np.all(np.isfinite(out)):
        raise InvalidArgumentError("audio contains NaN or infinite samples")
    return out


def _downmix_to_mono(wav: np.ndarray) -> np.ndarray:
    if wav.shape[0] == 1:
        return wav
    return wav.mean(axis=0, keepdims=True, dtype=np.float32)


def _resample(wav: np.ndarray, source_sr: int, target_sr: int) -> np.ndarray:
    if source_sr <= 0:
        raise InvalidArgumentError(f"sample_rate must be positive, got {source_sr}")
    if wav.shape[-1] == 0:
        return wav.astype(np.float32, copy=False)

    try:
        import torch
        import torchaudio.functional as F

        tensor = torch.from_numpy(wav.astype(np.float32, copy=False))
        return F.resample(tensor, source_sr, target_sr).numpy().astype(np.float32, copy=False)
    except Exception:
        old_len = wav.shape[-1]
        new_len = int(round(old_len * target_sr / source_sr))
        if new_len <= 0:
            return np.zeros((1, 0), dtype=np.float32)
        old_x = np.linspace(0.0, 1.0, num=old_len, endpoint=False)
        new_x = np.linspace(0.0, 1.0, num=new_len, endpoint=False)
        resampled = np.interp(new_x, old_x, wav[0]).astype(np.float32)
        return resampled.reshape(1, -1)


def _validate_duration(
    wav: np.ndarray,
    sample_rate: int,
    max_audio_seconds: float | None,
) -> None:
    limit = _resolved_max_audio_seconds(max_audio_seconds)
    if limit is None:
        return
    if limit <= 0:
        raise InvalidArgumentError("max audio duration must be positive when set")
    duration = wav.shape[-1] / sample_rate if sample_rate else 0.0
    if duration > limit:
        raise AudioDecodeError(
            f"decoded audio duration {duration:.1f}s exceeds granite-speech limit "
            f"{limit:.1f}s; set {ENV_MAX_AUDIO_SECONDS}=0 to disable or raise the cap"
        )


def _resolved_max_audio_seconds(value: float | None) -> float | None:
    if value is not None:
        return value
    raw = os.environ.get(ENV_MAX_AUDIO_SECONDS)
    if raw is None:
        return float(DEFAULT_MAX_AUDIO_SECONDS)
    try:
        parsed = float(raw)
    except ValueError as exc:
        raise InvalidArgumentError(f"{ENV_MAX_AUDIO_SECONDS} must be a number") from exc
    if parsed == 0:
        return None
    return parsed
