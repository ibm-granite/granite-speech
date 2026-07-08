from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .errors import InvalidArgumentError


@dataclass(frozen=True)
class AudioSegment:
    start: float
    end: float
    start_sample: int
    end_sample: int


@dataclass(frozen=True)
class FixedWindowSegmenter:
    chunk_length: float
    chunk_overlap: float = 0.0

    def __post_init__(self) -> None:
        validate_chunk_geometry(self.chunk_length, self.chunk_overlap)

    def segment(self, wav: np.ndarray | int, sample_rate: int) -> list[AudioSegment]:
        total_samples = int(wav) if isinstance(wav, int) else wav.shape[-1]
        if total_samples <= 0:
            return []
        chunk_samples = max(1, int(round(self.chunk_length * sample_rate)))
        overlap_samples = int(round(self.chunk_overlap * sample_rate))
        hop_samples = chunk_samples - overlap_samples
        if hop_samples <= 0:
            raise InvalidArgumentError("chunk geometry produced a non-positive hop")

        segments: list[AudioSegment] = []
        start_sample = 0
        while start_sample < total_samples:
            end_sample = min(start_sample + chunk_samples, total_samples)
            segments.append(
                AudioSegment(
                    start=start_sample / sample_rate,
                    end=end_sample / sample_rate,
                    start_sample=start_sample,
                    end_sample=end_sample,
                )
            )
            start_sample += hop_samples
        return segments


@dataclass(frozen=True)
class VadSegmenter:
    chunk_length: float
    threshold: float = 0.2
    min_speech_duration: float = 0.2
    min_silence_duration: float = 0.5
    speech_pad: float = 0.2
    frame_length: float = 0.03
    frame_step: float = 0.01

    def __post_init__(self) -> None:
        validate_vad_options(
            chunk_length=self.chunk_length,
            threshold=self.threshold,
            min_speech_duration=self.min_speech_duration,
            min_silence_duration=self.min_silence_duration,
            speech_pad=self.speech_pad,
            frame_length=self.frame_length,
            frame_step=self.frame_step,
        )

    def segment(self, wav: np.ndarray, sample_rate: int) -> list[AudioSegment]:
        samples = _mono_samples(wav)
        total_samples = samples.shape[-1]
        if total_samples <= 0:
            return []

        speech_spans = self._detect_speech_spans(samples, sample_rate)
        if not speech_spans:
            return []

        padded = self._pad_spans(speech_spans, total_samples, sample_rate)
        merged = self._merge_close_spans(padded, sample_rate)
        return self._split_long_spans(merged, sample_rate)

    def _detect_speech_spans(
        self,
        samples: np.ndarray,
        sample_rate: int,
    ) -> list[tuple[int, int]]:
        """Find speech spans by relative frame energy (RMS).

        Slides a frame over the audio and computes each frame's RMS. The
        threshold is *relative* to the loudest frame (``peak_rms * threshold``),
        not an absolute level, so it adapts to overall recording gain; silent
        audio (peak RMS 0) yields no spans. Contiguous above-threshold frames
        become a span, and spans shorter than ``min_speech_duration`` are
        dropped as noise. Returns ``(start_sample, end_sample)`` pairs; padding,
        merging, and length-splitting happen in the caller.
        """
        frame_samples = max(1, int(round(self.frame_length * sample_rate)))
        step_samples = max(1, int(round(self.frame_step * sample_rate)))
        total_samples = samples.shape[-1]
        starts = list(range(0, total_samples, step_samples))
        rms_values = np.asarray(
            [
                _rms(samples[start : min(start + frame_samples, total_samples)])
                for start in starts
            ],
            dtype=np.float32,
        )
        peak_rms = float(rms_values.max(initial=0.0))
        if peak_rms <= 0.0:
            return []

        speech_mask = rms_values >= peak_rms * self.threshold
        min_speech_samples = int(round(self.min_speech_duration * sample_rate))
        spans: list[tuple[int, int]] = []
        span_start: int | None = None
        span_end = 0
        for is_speech, start in zip(speech_mask, starts, strict=True):
            if is_speech:
                if span_start is None:
                    span_start = start
                span_end = min(start + frame_samples, total_samples)
            elif span_start is not None:
                if span_end - span_start >= min_speech_samples:
                    spans.append((span_start, span_end))
                span_start = None
        if span_start is not None and span_end - span_start >= min_speech_samples:
            spans.append((span_start, span_end))
        return spans

    def _pad_spans(
        self,
        spans: list[tuple[int, int]],
        total_samples: int,
        sample_rate: int,
    ) -> list[tuple[int, int]]:
        pad_samples = int(round(self.speech_pad * sample_rate))
        return [
            (max(0, start - pad_samples), min(total_samples, end + pad_samples))
            for start, end in spans
        ]

    def _merge_close_spans(
        self,
        spans: list[tuple[int, int]],
        sample_rate: int,
    ) -> list[tuple[int, int]]:
        if not spans:
            return []
        max_gap_samples = int(round(self.min_silence_duration * sample_rate))
        merged: list[tuple[int, int]] = [spans[0]]
        for start, end in spans[1:]:
            prev_start, prev_end = merged[-1]
            if start - prev_end <= max_gap_samples:
                merged[-1] = (prev_start, max(prev_end, end))
            else:
                merged.append((start, end))
        return merged

    def _split_long_spans(
        self,
        spans: list[tuple[int, int]],
        sample_rate: int,
    ) -> list[AudioSegment]:
        max_samples = max(1, int(round(self.chunk_length * sample_rate)))
        segments: list[AudioSegment] = []
        for span_start, span_end in spans:
            start = span_start
            while start < span_end:
                end = min(start + max_samples, span_end)
                segments.append(
                    AudioSegment(
                        start=start / sample_rate,
                        end=end / sample_rate,
                        start_sample=start,
                        end_sample=end,
                    )
                )
                start = end
        return segments


def validate_chunk_geometry(chunk_length: float, chunk_overlap: float) -> None:
    if chunk_length <= 0:
        raise InvalidArgumentError("chunk_length must be greater than 0")
    if chunk_overlap < 0:
        raise InvalidArgumentError("chunk_overlap must be greater than or equal to 0")
    if chunk_overlap >= chunk_length:
        raise InvalidArgumentError("chunk_overlap must be smaller than chunk_length")


def validate_vad_options(
    *,
    chunk_length: float,
    threshold: float,
    min_speech_duration: float,
    min_silence_duration: float,
    speech_pad: float,
    frame_length: float = 0.03,
    frame_step: float = 0.01,
) -> None:
    validate_chunk_geometry(chunk_length, 0.0)
    if not 0.0 < threshold <= 1.0:
        raise InvalidArgumentError("vad_threshold must be greater than 0 and at most 1")
    if min_speech_duration < 0:
        raise InvalidArgumentError("vad_min_speech_duration must be greater than or equal to 0")
    if min_silence_duration < 0:
        raise InvalidArgumentError("vad_min_silence_duration must be greater than or equal to 0")
    if speech_pad < 0:
        raise InvalidArgumentError("vad_speech_pad must be greater than or equal to 0")
    if frame_length <= 0:
        raise InvalidArgumentError("vad frame_length must be greater than 0")
    if frame_step <= 0:
        raise InvalidArgumentError("vad frame_step must be greater than 0")


def _mono_samples(wav: np.ndarray) -> np.ndarray:
    audio = np.asarray(wav, dtype=np.float32)
    if audio.ndim == 1:
        return audio
    if audio.ndim == 2:
        if audio.shape[0] == 1:
            return audio[0]
        return audio.mean(axis=0, dtype=np.float32)
    raise InvalidArgumentError(
        f"audio window must be mono or channel-first, got shape {audio.shape}"
    )


def _rms(frame: np.ndarray) -> float:
    if frame.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(frame.astype(np.float32, copy=False) ** 2)))


def segment_boundaries(
    windows: list[AudioSegment],
    *,
    chunk_overlap: float,
    duration: float,
) -> list[tuple[float, float]]:
    if not windows:
        return []
    if chunk_overlap == 0:
        return [(window.start, window.end) for window in windows]

    boundaries: list[tuple[float, float]] = []
    for index, window in enumerate(windows):
        if index == 0:
            start = window.start
        else:
            start = windows[index].start + chunk_overlap / 2.0

        if index + 1 < len(windows):
            end = windows[index + 1].start + chunk_overlap / 2.0
        else:
            end = duration

        boundaries.append((max(0.0, start), max(start, min(end, duration))))
    return boundaries
