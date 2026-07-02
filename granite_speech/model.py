from __future__ import annotations

import warnings
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ._backends import Backend
from ._models import (
    ModelSpec,
    validate_source_language,
    validate_translation_pair,
)
from .audio import load_audio
from .chunking import ChunkingOptions, transcribe_chunks
from .errors import InvalidArgumentError
from .prompts import PROMPT_MODES, build_prompt, normalize_keyword_biases
from .segmenter import validate_chunk_geometry, validate_vad_options

_WHISPER_DEFAULT_ONLY_OPTIONS = {
    "compression_ratio_threshold": 2.4,
    "logprob_threshold": -1.0,
    "no_speech_threshold": 0.6,
    "clip_timestamps": "0",
    "hallucination_silence_threshold": None,
    "best_of": None,
    "patience": None,
    "length_penalty": None,
    "suppress_tokens": None,
    "suppress_blank": None,
    "without_timestamps": None,
    "max_initial_timestamp": None,
}

_WHISPER_WARN_ONLY_OPTIONS = {
    "condition_on_previous_text": (
        "Granite Speech does not implement Whisper previous-text conditioning; "
        "ignoring condition_on_previous_text."
    ),
    "carry_initial_prompt": (
        "Granite Speech applies its prompt independently to each audio window; "
        "ignoring carry_initial_prompt."
    ),
    "prepend_punctuations": (
        "Granite Speech word timestamp output does not use Whisper punctuation merge controls; "
        "ignoring prepend_punctuations."
    ),
    "append_punctuations": (
        "Granite Speech word timestamp output does not use Whisper punctuation merge controls; "
        "ignoring append_punctuations."
    ),
}


@dataclass
class GraniteSpeechModel:
    backend: Backend
    processor: Any
    model: Any
    tokenizer: Any
    device: Any
    spec: ModelSpec

    def transcribe(
        self,
        audio: str | Path | np.ndarray | Any,
        *,
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
        **whisper_options: Any,
    ) -> dict:
        prompt, prompt_mode, num_beams, temperature = _resolve_whisper_compat_options(
            prompt=prompt,
            prompt_mode=prompt_mode,
            num_beams=num_beams,
            temperature=temperature,
            initial_prompt=initial_prompt,
            word_timestamps=word_timestamps,
            fp16=fp16,
            whisper_options=whisper_options,
        )
        target_language = self._validate_request(
            task=task,
            language=language,
            target_language=target_language,
            prompt_mode=prompt_mode,
            prefix_text=prefix_text,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            temperature=temperature,
            chunk_length=chunk_length,
            chunk_overlap=chunk_overlap,
            segmentation=segmentation,
            vad_threshold=vad_threshold,
            vad_min_speech_duration=vad_min_speech_duration,
            vad_min_silence_duration=vad_min_silence_duration,
            vad_speech_pad=vad_speech_pad,
        )
        normalized_keyword_biases = normalize_keyword_biases(keyword_biases)

        audio_data = load_audio(audio, sample_rate=sample_rate)
        prompt_parts = build_prompt(
            self.tokenizer,
            task=task,
            language=language,
            target_language=target_language,
            prompt=prompt,
            keyword_biases=normalized_keyword_biases,
            model_spec=self.spec,
            prompt_mode=prompt_mode,
            prefix_text=prefix_text,
        )

        chunk_result = transcribe_chunks(
            audio_data.wav,
            backend=self.backend,
            sample_rate=audio_data.sample_rate,
            options=ChunkingOptions(
                task=task,
                language=language,
                target_language=target_language,
                prompt=prompt_parts.prompt,
                instruction=prompt_parts.instruction,
                keyword_biases=normalized_keyword_biases,
                prompt_mode=prompt_mode,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                temperature=temperature,
                chunk_length=chunk_length,
                chunk_overlap=chunk_overlap,
                verbose=verbose,
                segmentation=segmentation,
                vad_threshold=vad_threshold,
                vad_min_speech_duration=vad_min_speech_duration,
                vad_min_silence_duration=vad_min_silence_duration,
                vad_speech_pad=vad_speech_pad,
            ),
        )
        result = {
            "text": chunk_result["text"],
            "segments": chunk_result["segments"],
            "language": language,
            "target_language": target_language if task == "translate" else None,
            "warnings": chunk_result["warnings"],
        }
        if "words" in chunk_result:
            result["words"] = chunk_result["words"]
        if "speakers" in chunk_result:
            result["speakers"] = chunk_result["speakers"]
        return result

    def _validate_request(
        self,
        *,
        task: str,
        language: str | None,
        target_language: str | None,
        prompt_mode: str,
        prefix_text: str | None,
        max_new_tokens: int | None,
        num_beams: int,
        temperature: float,
        chunk_length: float,
        chunk_overlap: float,
        segmentation: str,
        vad_threshold: float,
        vad_min_speech_duration: float,
        vad_min_silence_duration: float,
        vad_speech_pad: float,
    ) -> str | None:
        if task not in {"transcribe", "translate"}:
            raise InvalidArgumentError("task must be either 'transcribe' or 'translate'")
        if prompt_mode not in PROMPT_MODES:
            supported = ", ".join(sorted(PROMPT_MODES))
            raise InvalidArgumentError(f"prompt_mode must be one of: {supported}")
        if prefix_text is not None and not isinstance(prefix_text, str):
            raise InvalidArgumentError("prefix_text must be a string")
        if segmentation not in {"fixed", "vad"}:
            raise InvalidArgumentError("segmentation must be either 'fixed' or 'vad'")
        validate_chunk_geometry(chunk_length, chunk_overlap)
        if segmentation == "vad":
            if chunk_overlap > 0:
                raise InvalidArgumentError("chunk_overlap is only supported with segmentation='fixed'")
            validate_vad_options(
                chunk_length=chunk_length,
                threshold=vad_threshold,
                min_speech_duration=vad_min_speech_duration,
                min_silence_duration=vad_min_silence_duration,
                speech_pad=vad_speech_pad,
            )
        if max_new_tokens is not None and max_new_tokens <= 0:
            raise InvalidArgumentError("max_new_tokens must be greater than 0")
        if num_beams < 1:
            raise InvalidArgumentError("num_beams must be greater than or equal to 1")
        if self.backend.name == "llama.cpp" and num_beams != 1:
            raise InvalidArgumentError("llama.cpp backend does not support num_beams; use num_beams=1")
        if temperature < 0.0:
            raise InvalidArgumentError("temperature must be greater than or equal to 0")
        if temperature > 0.0 and num_beams > 1:
            raise InvalidArgumentError(
                "temperature > 0 with num_beams > 1 is unsupported; use sampling with "
                "num_beams=1 or beam search with temperature=0"
            )

        if task == "transcribe":
            if target_language is not None:
                raise InvalidArgumentError(
                    "target_language is only valid with task='translate'; "
                    "do not pass it for transcription"
                )
            if (
                prompt_mode == "speaker_attributed"
                and not self.spec.supports_speaker_attribution_output
            ):
                raise InvalidArgumentError(
                    f"model {self.spec.name!r} does not support speaker-attributed prompts"
                )
            if prompt_mode == "word_timestamps" and not self.spec.supports_word_timing_output:
                raise InvalidArgumentError(
                    f"model {self.spec.name!r} does not support word timestamp prompts"
                )
            if prefix_text is not None and self.spec.prompt_profile != "plus":
                raise InvalidArgumentError(
                    f"model {self.spec.name!r} does not support prefix_text incremental decoding"
                )
            validate_source_language(self.spec, language)
            return None

        if prompt_mode != "default":
            raise InvalidArgumentError("prompt_mode is only supported with task='transcribe'")
        if prefix_text is not None:
            raise InvalidArgumentError("prefix_text is only supported with task='transcribe'")
        if language is None:
            raise InvalidArgumentError("task='translate' requires an explicit source language")
        resolved_target = target_language if target_language is not None else "en"
        validate_translation_pair(self.spec, language, resolved_target)
        return resolved_target


def _resolve_whisper_compat_options(
    *,
    prompt: str | None,
    prompt_mode: str,
    num_beams: int,
    temperature: float | tuple[float, ...] | list[float],
    initial_prompt: str | None,
    word_timestamps: bool | None,
    fp16: bool | None,
    whisper_options: dict[str, Any],
) -> tuple[str | None, str, int, float]:
    _handle_unsupported_whisper_options(whisper_options)

    if initial_prompt is not None:
        if prompt is not None:
            raise InvalidArgumentError("pass either prompt or initial_prompt, not both")
        if not isinstance(initial_prompt, str):
            raise InvalidArgumentError("initial_prompt must be a string")
        prompt = initial_prompt

    if word_timestamps is not None:
        if not isinstance(word_timestamps, bool):
            raise InvalidArgumentError("word_timestamps must be a boolean")
        if word_timestamps:
            if prompt_mode not in {"default", "word_timestamps"}:
                raise InvalidArgumentError(
                    "word_timestamps=True cannot be combined with a different prompt_mode"
                )
            if prompt_mode != "word_timestamps":
                _warn_whisper_compat(
                    "word_timestamps=True maps to prompt_mode='word_timestamps' on capable "
                    "Granite Speech plus models; this is model-generated timestamp output, "
                    "not Whisper DTW alignment."
                )
            prompt_mode = "word_timestamps"

    if fp16 is not None:
        if not isinstance(fp16, bool):
            raise InvalidArgumentError("fp16 must be a boolean")
        _warn_whisper_compat(
            "Granite Speech does not support Whisper's per-transcription fp16 option; "
            "ignoring fp16."
        )

    temperature = _normalize_temperature(temperature)
    return prompt, prompt_mode, num_beams, temperature


def _handle_unsupported_whisper_options(options: dict[str, Any]) -> None:
    for name, value in sorted(options.items()):
        if name == "beam_size":
            raise InvalidArgumentError(
                "unsupported Whisper option 'beam_size'; Granite Speech does not expose "
                "beam_size as a compatibility alias because the default llama.cpp backend "
                "does not support beam search. Use the default greedy decoding."
            )

        if name in _WHISPER_WARN_ONLY_OPTIONS:
            _warn_whisper_compat(_WHISPER_WARN_ONLY_OPTIONS[name])
            continue

        if name in _WHISPER_DEFAULT_ONLY_OPTIONS:
            default = _WHISPER_DEFAULT_ONLY_OPTIONS[name]
            if value == default:
                _warn_whisper_compat(
                    f"Granite Speech does not implement Whisper option {name!r}; "
                    "the default value was provided and will be ignored."
                )
                continue
            raise InvalidArgumentError(
                f"unsupported Whisper option {name!r}; Granite Speech cannot honor "
                "non-default values for this transcript-affecting option"
            )

        raise InvalidArgumentError(f"unsupported transcribe option {name!r}")


def _normalize_temperature(temperature: Any) -> float:
    if isinstance(temperature, (list, tuple)):
        if not temperature:
            raise InvalidArgumentError("temperature fallback schedule must not be empty")
        selected = temperature[0]
        _warn_whisper_compat(
            "Granite Speech does not implement Whisper temperature fallback schedules; "
            f"using the first temperature value {selected!r}."
        )
    else:
        selected = temperature

    if isinstance(selected, bool):
        raise InvalidArgumentError("temperature must be a number")
    try:
        return float(selected)
    except (TypeError, ValueError) as exc:
        raise InvalidArgumentError("temperature must be a number") from exc


def _warn_whisper_compat(message: str) -> None:
    warnings.warn(message, UserWarning, stacklevel=3)
