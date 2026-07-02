from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class GenerateRequest:
    prompt: str
    wav: np.ndarray
    sample_rate: int = 16000
    instruction: str | None = None
    task: str = "transcribe"
    language: str | None = None
    target_language: str | None = None
    max_new_tokens: int = 200
    num_beams: int = 1
    temperature: float = 0.0
    keyword_biases: tuple[str, ...] = ()


@dataclass(frozen=True)
class GenerateResult:
    text: str
    words: list[dict] | None = None
    speakers: list[dict] | None = None
    tokens: list[int] | None = None


@dataclass(frozen=True)
class BackendCapabilities:
    max_reliable_audio_seconds: float | None
    supports_word_timing_output: bool
    supports_speaker_attribution_output: bool
    supports_batch: bool
    supports_translation: bool


class Backend(Protocol):
    name: str
    capabilities: BackendCapabilities

    def generate(self, req: GenerateRequest) -> GenerateResult: ...


__all__ = [
    "Backend",
    "BackendCapabilities",
    "GenerateRequest",
    "GenerateResult",
]
