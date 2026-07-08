from __future__ import annotations

import sys
from dataclasses import dataclass
from math import ceil

import numpy as np

from ._backends import Backend, GenerateRequest
from .errors import InvalidArgumentError, TranscriptionError
from .plus_output import join_text, parse_plus_output
from .reconciliation import reconcile_overlapping_chunks
from .segmenter import (
    AudioSegment,
    FixedWindowSegmenter,
    VadSegmenter,
    segment_boundaries,
)

DEFAULT_MAX_NEW_TOKENS = 200
DEFAULT_TOKEN_BUDGET_SECONDS = 30.0


@dataclass(frozen=True)
class ChunkingOptions:
    task: str
    language: str | None
    target_language: str | None
    prompt: str
    instruction: str | None
    keyword_biases: tuple[str, ...]
    max_new_tokens: int | None
    num_beams: int
    temperature: float
    chunk_length: float
    chunk_overlap: float
    verbose: bool | None = None
    segmentation: str = "fixed"
    vad_threshold: float = 0.2
    vad_min_speech_duration: float = 0.2
    vad_min_silence_duration: float = 0.5
    vad_speech_pad: float = 0.2
    prompt_mode: str = "default"


def transcribe_chunks(
    wav: np.ndarray,
    *,
    backend: Backend,
    sample_rate: int,
    options: ChunkingOptions,
) -> dict:
    """Transcribe a waveform by splitting it into windows and reconciling them.

    ``chunk_length`` is first clamped to the backend's reliable limit (recording
    a ``chunk_clamp`` warning if it was exceeded). The audio is then segmented —
    fixed overlapping windows, or VAD-detected speech spans when
    ``segmentation='vad'`` (VAD forbids ``chunk_overlap``) — and each window is
    generated independently. A window that fails is captured as a per-window
    warning rather than aborting the run. Overlapping fixed windows are
    de-duplicated at their seams via the reconciliation pass, and the surviving
    text plus per-segment timing is assembled into the result dict
    (``text`` / ``segments`` / ``warnings``, and ``words`` / ``speakers`` for the
    plus prompt modes).
    """
    warnings: list[dict] = []
    chunk_length = options.chunk_length

    cap = backend.capabilities.max_reliable_audio_seconds
    if cap is not None and chunk_length > cap:
        warnings.append(
            {
                "type": "chunk_clamp",
                "message": (
                    f"requested chunk_length {chunk_length:.3f}s exceeds backend "
                    f"{backend.name!r} reliable limit {cap:.3f}s; clamped"
                ),
                "requested": chunk_length,
                "applied": cap,
            }
        )
        chunk_length = cap

    windows, boundary_overlap = _select_windows(wav, sample_rate, chunk_length, options)

    duration = wav.shape[-1] / sample_rate if sample_rate else 0.0
    boundaries = segment_boundaries(windows, chunk_overlap=boundary_overlap, duration=duration)

    segments = _generate_segments(
        wav,
        windows=windows,
        boundaries=boundaries,
        backend=backend,
        sample_rate=sample_rate,
        chunk_length=chunk_length,
        options=options,
        warnings=warnings,
    )

    if segments and all("error" in segment for segment in segments):
        raise TranscriptionError("every audio window failed during transcription")

    if boundary_overlap > 0:
        reconcile_successful_segments(
            segments,
            overlap_fraction=options.chunk_overlap / chunk_length,
        )

    all_words, all_speakers = apply_structured_output_parsing(
        segments,
        prompt_mode=options.prompt_mode,
    )

    if options.verbose:
        for segment in segments:
            if "error" not in segment:
                print(segment["text"], file=sys.stderr, flush=True)

    text = join_text(segment["text"] for segment in segments if "error" not in segment)
    result = {
        "text": text,
        "segments": segments,
        "warnings": warnings,
    }
    if options.prompt_mode == "word_timestamps" or all_words:
        result["words"] = all_words
    if options.prompt_mode == "speaker_attributed" or all_speakers:
        result["speakers"] = all_speakers
    return result


def _select_windows(
    wav: np.ndarray,
    sample_rate: int,
    chunk_length: float,
    options: ChunkingOptions,
) -> tuple[list[AudioSegment], float]:
    """Segment the waveform per ``options.segmentation`` and report the boundary overlap.

    Returns the windows plus the overlap (in seconds) used when computing segment
    boundaries: ``chunk_overlap`` for fixed windows, ``0.0`` for VAD (which forbids
    ``chunk_overlap``). Raises ``InvalidArgumentError`` for an unknown mode.
    """
    if options.segmentation == "fixed":
        segmenter: FixedWindowSegmenter | VadSegmenter = FixedWindowSegmenter(
            chunk_length, options.chunk_overlap
        )
        return segmenter.segment(wav, sample_rate), options.chunk_overlap
    if options.segmentation == "vad":
        if options.chunk_overlap > 0:
            raise InvalidArgumentError("chunk_overlap is only supported with segmentation='fixed'")
        segmenter = VadSegmenter(
            chunk_length,
            threshold=options.vad_threshold,
            min_speech_duration=options.vad_min_speech_duration,
            min_silence_duration=options.vad_min_silence_duration,
            speech_pad=options.vad_speech_pad,
        )
        return segmenter.segment(wav, sample_rate), 0.0
    raise InvalidArgumentError("segmentation must be either 'fixed' or 'vad'")


def _generate_segments(
    wav: np.ndarray,
    *,
    windows: list[AudioSegment],
    boundaries: list[tuple[float, float]],
    backend: Backend,
    sample_rate: int,
    chunk_length: float,
    options: ChunkingOptions,
    warnings: list[dict],
) -> list[dict]:
    """Generate one segment per window, capturing failures as per-window warnings.

    Each window is generated independently; a window that raises becomes a segment
    with an ``error`` field and a matching ``window_error`` entry appended to
    ``warnings`` (mutated in place) rather than aborting the whole run.
    """
    segments: list[dict] = []
    for segment_id, (window, (seg_start, seg_end)) in enumerate(
        zip(windows, boundaries, strict=True)
    ):
        window_wav = wav[:, window.start_sample : window.end_sample]
        window_duration = window_wav.shape[-1] / sample_rate if sample_rate else chunk_length
        try:
            req = GenerateRequest(
                prompt=options.prompt,
                wav=window_wav,
                sample_rate=sample_rate,
                instruction=options.instruction,
                keyword_biases=options.keyword_biases,
                task=options.task,
                language=options.language,
                target_language=options.target_language,
                max_new_tokens=resolve_max_new_tokens(options.max_new_tokens, window_duration),
                num_beams=options.num_beams,
                temperature=options.temperature,
            )
            generated = backend.generate(req)
            segment = {
                "id": segment_id,
                "start": seg_start,
                "end": seg_end,
                "text": generated.text.strip(),
                "temperature": options.temperature,
            }
            if generated.tokens is not None:
                segment["tokens"] = list(generated.tokens)
            if generated.words is not None:
                segment["_backend_words"] = generated.words
            if generated.speakers is not None:
                segment["_backend_speakers"] = generated.speakers
            segments.append(segment)
        except Exception as exc:
            error = str(exc).strip() or exc.__class__.__name__
            segment = {
                "id": segment_id,
                "start": seg_start,
                "end": seg_end,
                "text": "",
                "temperature": options.temperature,
                "error": error,
            }
            segments.append(segment)
            warning = {"type": "window_error", **segment}
            warnings.append(warning)
    return segments


def resolve_max_new_tokens(max_new_tokens: int | None, chunk_length: float) -> int:
    if max_new_tokens is not None:
        return max_new_tokens
    scale = max(1.0, chunk_length / DEFAULT_TOKEN_BUDGET_SECONDS)
    return int(ceil(DEFAULT_MAX_NEW_TOKENS * scale))


def reconcile_successful_segments(segments: list[dict], *, overlap_fraction: float) -> None:
    group: list[dict] = []
    for segment in segments:
        if "error" not in segment and segment.get("text", "").strip():
            group.append(segment)
            continue
        _rewrite_reconciled_group(group, overlap_fraction=overlap_fraction)
        group = []
    _rewrite_reconciled_group(group, overlap_fraction=overlap_fraction)


def _rewrite_reconciled_group(group: list[dict], *, overlap_fraction: float) -> None:
    if len(group) < 2:
        return
    reconciled = reconcile_overlapping_chunks(
        [segment["text"] for segment in group],
        overlap_fraction=overlap_fraction,
    )
    for segment, text in zip(group, reconciled, strict=True):
        segment["text"] = text


def apply_structured_output_parsing(
    segments: list[dict],
    *,
    prompt_mode: str,
) -> tuple[list[dict], list[dict]]:
    all_words: list[dict] = []
    all_speakers: list[dict] = []

    for segment in segments:
        if "error" in segment:
            continue
        backend_words = segment.pop("_backend_words", None)
        backend_speakers = segment.pop("_backend_speakers", None)
        parsed = parse_plus_output(
            segment.get("text", ""),
            prompt_mode=prompt_mode,
            segment_start=segment["start"],
        )
        segment["text"] = parsed.text
        _add_parsed_fields(
            segment,
            raw_text=parsed.raw_text,
            words=backend_words if backend_words is not None else parsed.words,
            speakers=backend_speakers if backend_speakers is not None else parsed.speakers,
            include_words=prompt_mode == "word_timestamps" or backend_words is not None,
            include_speakers=prompt_mode == "speaker_attributed" or backend_speakers is not None,
        )
        if "words" in segment:
            all_words.extend(segment["words"])
        if "speakers" in segment:
            all_speakers.extend(segment["speakers"])

    return all_words, all_speakers


def _add_parsed_fields(
    segment: dict,
    *,
    raw_text: str,
    words: list[dict] | None,
    speakers: list[dict] | None,
    include_words: bool,
    include_speakers: bool,
) -> None:
    if raw_text.strip() != segment["text"]:
        segment["raw_text"] = raw_text.strip()
    if include_words:
        segment["words"] = list(words or [])
    if include_speakers:
        segment["speakers"] = list(speakers or [])
