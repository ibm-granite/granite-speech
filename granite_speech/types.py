"""Typed shapes for the public transcription result.

These ``TypedDict``s document the result returned by :func:`granite_speech.transcribe`
and :meth:`GraniteSpeechModel.transcribe`. They are a static-typing layer only: at
runtime every result is a plain ``dict`` (``TypedDict`` is erased), so the wire shape
is byte-identical to what the library has always returned. The keys here are the
"stable keys" promised in the README; the shape is deliberately Whisper-*familiar*,
not identical (see ``docs/porting-from-whisper.md``).

Optional keys use the base-class + ``total=False`` subclass pattern rather than
``typing.NotRequired`` so the types work on the supported floor (Python 3.10).
"""

from __future__ import annotations

from typing import TypedDict


class Word(TypedDict):
    """A single word with model-generated timing (``word_timestamps`` mode)."""

    word: str
    start: float
    end: float


class Speaker(TypedDict):
    """One speaker turn's text (``speaker_attributed`` mode)."""

    speaker: str
    text: str


class _SegmentBase(TypedDict):
    id: int  # Whisper-familiar compatibility metadata
    start: float
    end: float
    text: str
    temperature: float


class Segment(_SegmentBase, total=False):
    """One transcribed window.

    ``id`` / ``start`` / ``end`` / ``text`` / ``temperature`` are always present; the
    keys below appear conditionally.
    """

    tokens: list[int]  # only when the backend provides token IDs
    error: str  # only on a failed window
    raw_text: str  # plus modes, when the parsed text differs from raw
    words: list[Word]  # word_timestamps mode
    speakers: list[Speaker]  # speaker_attributed mode


class Warning(TypedDict, total=False):
    """A non-fatal warning attached to the result.

    ``type`` is always present (``"chunk_clamp"`` or ``"window_error"``); the rest of
    the payload differs per variant, so every key is optional here:

    - ``chunk_clamp``: ``message`` / ``requested`` / ``applied``
    - ``window_error``: the failed segment's fields (``id`` / ``start`` / ``end`` /
      ``text`` / ``temperature`` / ``error``) plus ``message`` where set.
    """

    type: str
    message: str
    requested: float
    applied: float
    id: int
    start: float
    end: float
    text: str
    temperature: float
    error: str


class _TranscriptionResultBase(TypedDict):
    text: str
    segments: list[Segment]
    language: str | None
    target_language: str | None
    warnings: list[Warning]


class TranscriptionResult(_TranscriptionResultBase, total=False):
    """The result dict returned by the public ``transcribe`` entry points.

    ``text`` / ``segments`` / ``language`` / ``target_language`` / ``warnings`` are
    always present; ``words`` / ``speakers`` appear in the corresponding plus prompt
    modes (or whenever such items were produced).
    """

    words: list[Word]  # word_timestamps mode, or any words present
    speakers: list[Speaker]  # speaker_attributed mode, or any present


# NOTE: this shape intentionally OMITS Whisper's per-segment ``seek`` /
# ``avg_logprob`` / ``compression_ratio`` / ``no_speech_prob`` and top-level
# ``duration``. Granite Speech does not fabricate confidence metrics — see the
# README and docs/porting-from-whisper.md. Do not add these to chase strict Whisper
# parity; the shape is deliberately Whisper-*familiar*, not identical.

__all__ = ["Word", "Speaker", "Segment", "Warning", "TranscriptionResult"]
