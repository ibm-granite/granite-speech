from __future__ import annotations

import logging
import threading
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

from ._models import DEFAULT_MODEL, available_models
from .errors import (
    AudioDecodeError,
    GraniteSpeechError,
    InvalidArgumentError,
    ModelLoadError,
    TranscriptionError,
    TransformersVersionError,
)
from .loader import load_model
from .model import GraniteSpeechModel
from .version import __version__

logging.getLogger(__name__).addHandler(logging.NullHandler())

_MODEL_CACHE: dict[str, GraniteSpeechModel] = {}
_MODEL_CACHE_LOCK = threading.Lock()


def transcribe(
    audio: str | Path | np.ndarray | Any,
    *,
    model: str = DEFAULT_MODEL,
    sample_rate: int | None = None,
    task: str = "transcribe",
    language: str | None = None,
    target_language: str | None = None,
    prompt: str | None = None,
    prompt_mode: str = "default",
    prefix_text: str | None = None,
    keyword_biases: Iterable[str] | str | None = None,
    max_new_tokens: int | None = None,
    num_beams: int = 1,
    temperature: float | tuple[float, ...] | list[float] = 0.0,
    verbose: bool | None = None,
    chunk_length: float = 30.0,
    chunk_overlap: float = 0.0,
    segmentation: str = "fixed",
    vad_threshold: float = 0.2,
    vad_min_speech_duration: float = 0.2,
    vad_min_silence_duration: float = 0.5,
    vad_speech_pad: float = 0.2,
    initial_prompt: str | None = None,
    word_timestamps: bool | None = None,
    fp16: bool | None = None,
    clip_timestamps: str | Iterable[float] | Iterable[tuple[float, float]] | None = None,
    **whisper_options: Any,
) -> dict:
    loaded = _get_cached_model(model)
    return loaded.transcribe(
        audio,
        sample_rate=sample_rate,
        task=task,
        language=language,
        target_language=target_language,
        prompt=prompt,
        prompt_mode=prompt_mode,
        prefix_text=prefix_text,
        keyword_biases=keyword_biases,
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
        temperature=temperature,
        verbose=verbose,
        chunk_length=chunk_length,
        chunk_overlap=chunk_overlap,
        segmentation=segmentation,
        vad_threshold=vad_threshold,
        vad_min_speech_duration=vad_min_speech_duration,
        vad_min_silence_duration=vad_min_silence_duration,
        vad_speech_pad=vad_speech_pad,
        initial_prompt=initial_prompt,
        word_timestamps=word_timestamps,
        fp16=fp16,
        clip_timestamps=clip_timestamps,
        **whisper_options,
    )


def _get_cached_model(model: str) -> GraniteSpeechModel:
    with _MODEL_CACHE_LOCK:
        loaded = _MODEL_CACHE.get(model)
        if loaded is None:
            loaded = load_model(model)
            _MODEL_CACHE[model] = loaded
        return loaded


__all__ = [
    "__version__",
    "available_models",
    "load_model",
    "transcribe",
    "GraniteSpeechModel",
    "GraniteSpeechError",
    "ModelLoadError",
    "TransformersVersionError",
    "AudioDecodeError",
    "InvalidArgumentError",
    "TranscriptionError",
]
