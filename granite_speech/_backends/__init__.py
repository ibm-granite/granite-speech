from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

# Every Granite Speech backend consumes 16 kHz mono audio; load_audio resamples
# to this rate before a GenerateRequest is ever built.
SAMPLE_RATE = 16000


@dataclass(frozen=True)
class GenerateRequest:
    prompt: str
    wav: np.ndarray
    sample_rate: int = SAMPLE_RATE
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


def require_sample_rate(sample_rate: int) -> None:
    """Guard that a request uses the audio rate every backend expects."""
    if sample_rate != SAMPLE_RATE:
        raise ValueError(f"GenerateRequest.sample_rate must be {SAMPLE_RATE}, got {sample_rate}")


__all__ = [
    "Backend",
    "BackendCapabilities",
    "GenerateRequest",
    "GenerateResult",
    "SAMPLE_RATE",
    "require_sample_rate",
]
