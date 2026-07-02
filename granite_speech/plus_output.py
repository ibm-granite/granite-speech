from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class ParsedPlusOutput:
    text: str
    raw_text: str
    words: list[dict] | None = None
    speakers: list[dict] | None = None


_SPEAKER_TAG_RE = re.compile(r"\[Speaker\s+([^\]]+)\]\s*:?", re.IGNORECASE)
_TIMESTAMP_TAG_RE = re.compile(
    r"(?P<word>\S+)\s*\[T:\s*(?P<centiseconds>[+-]?\d+(?:\.\d+)?)\s*\]",
    re.IGNORECASE,
)
_TIMESTAMP_STRIP_RE = re.compile(r"\s*\[T:\s*[+-]?\d+(?:\.\d+)?\s*\]", re.IGNORECASE)


def parse_plus_output(
    text: str,
    *,
    prompt_mode: str,
    segment_start: float = 0.0,
) -> ParsedPlusOutput:
    if prompt_mode == "speaker_attributed":
        return parse_speaker_attributed_output(text)
    if prompt_mode == "word_timestamps":
        return parse_word_timestamp_output(text, segment_start=segment_start)
    return ParsedPlusOutput(text=text.strip(), raw_text=text)


def parse_speaker_attributed_output(text: str) -> ParsedPlusOutput:
    matches = list(_SPEAKER_TAG_RE.finditer(text))
    if not matches:
        cleaned = _normalize_text(text)
        return ParsedPlusOutput(text=cleaned, raw_text=text, speakers=[])

    speakers: list[dict] = []
    pieces: list[str] = []

    leading = _normalize_text(text[: matches[0].start()])
    if leading:
        pieces.append(leading)

    for index, match in enumerate(matches):
        turn_start = match.end()
        turn_end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        turn_text = _normalize_text(text[turn_start:turn_end])
        if not turn_text:
            continue
        speaker = f"Speaker {match.group(1).strip()}"
        speakers.append({"speaker": speaker, "text": turn_text})
        pieces.append(turn_text)

    return ParsedPlusOutput(text=join_text(pieces), raw_text=text, speakers=speakers)


def parse_word_timestamp_output(text: str, *, segment_start: float = 0.0) -> ParsedPlusOutput:
    words: list[dict] = []
    previous_end = float(segment_start)

    for match in _TIMESTAMP_TAG_RE.finditer(text):
        word = match.group("word").strip()
        if not word:
            continue
        try:
            relative_end = float(match.group("centiseconds")) / 100.0
        except ValueError:
            continue
        end = max(float(segment_start), float(segment_start) + relative_end)
        start = min(previous_end, end)
        words.append({"word": word, "start": start, "end": end})
        previous_end = max(previous_end, end)

    cleaned = _normalize_text(_TIMESTAMP_STRIP_RE.sub(" ", text))
    return ParsedPlusOutput(text=cleaned, raw_text=text, words=words)


def join_text(parts) -> str:
    return " ".join(part.strip() for part in parts if part and part.strip()).strip()


def _normalize_text(text: str) -> str:
    return " ".join(text.split())
