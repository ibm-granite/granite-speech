from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import Callable

from .errors import InvalidArgumentError

OUTPUT_FORMATS = {"txt", "srt", "vtt", "tsv", "json", "all"}


def get_writer(
    output_format: str,
    output_dir: str | Path,
    *,
    max_line_width: int | None = None,
    max_line_count: int | None = None,
) -> Callable[[dict, str | Path], list[Path]]:
    if output_format not in OUTPUT_FORMATS:
        raise InvalidArgumentError(
            f"output_format must be one of {', '.join(sorted(OUTPUT_FORMATS))}"
        )
    _validate_subtitle_layout(max_line_width=max_line_width, max_line_count=max_line_count)

    def writer(result: dict, audio_path: str | Path) -> list[Path]:
        return write_result(
            result,
            audio_path,
            output_dir=output_dir,
            output_format=output_format,
            max_line_width=max_line_width,
            max_line_count=max_line_count,
        )

    return writer


def write_result(
    result: dict,
    audio_path: str | Path,
    *,
    output_dir: str | Path,
    output_format: str,
    max_line_width: int | None = None,
    max_line_count: int | None = None,
) -> list[Path]:
    if output_format not in OUTPUT_FORMATS:
        raise InvalidArgumentError(
            f"output_format must be one of {', '.join(sorted(OUTPUT_FORMATS))}"
        )
    _validate_subtitle_layout(max_line_width=max_line_width, max_line_count=max_line_count)
    formats = ["txt", "srt", "vtt", "tsv", "json"] if output_format == "all" else [output_format]
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(audio_path).stem
    paths: list[Path] = []
    for fmt in formats:
        path = out_dir / f"{stem}.{fmt}"
        _write_one(
            result,
            path,
            fmt,
            max_line_width=max_line_width,
            max_line_count=max_line_count,
        )
        paths.append(path)
    return paths


def _write_one(
    result: dict,
    path: Path,
    fmt: str,
    *,
    max_line_width: int | None,
    max_line_count: int | None,
) -> None:
    if fmt == "txt":
        path.write_text(_format_txt(result), encoding="utf-8")
    elif fmt == "srt":
        path.write_text(
            _format_srt(
                result,
                max_line_width=max_line_width,
                max_line_count=max_line_count,
            ),
            encoding="utf-8",
        )
    elif fmt == "vtt":
        path.write_text(
            _format_vtt(
                result,
                max_line_width=max_line_width,
                max_line_count=max_line_count,
            ),
            encoding="utf-8",
        )
    elif fmt == "tsv":
        path.write_text(_format_tsv(result), encoding="utf-8")
    elif fmt == "json":
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    else:  # pragma: no cover - guarded by caller
        raise InvalidArgumentError(f"unsupported output format {fmt!r}")


def _format_txt(result: dict) -> str:
    text = result.get("text", "")
    return text + ("\n" if text else "")


def _format_srt(
    result: dict,
    *,
    max_line_width: int | None = None,
    max_line_count: int | None = None,
) -> str:
    cues = []
    for idx, segment in enumerate(_successful_text_segments(result), start=1):
        text = _format_subtitle_text(
            segment["text"],
            max_line_width=max_line_width,
            max_line_count=max_line_count,
        )
        cues.append(
            f"{idx}\n"
            f"{format_timestamp(segment['start'], decimal_marker=',')} --> "
            f"{format_timestamp(segment['end'], decimal_marker=',')}\n"
            f"{text}\n"
        )
    return "\n".join(cues)


def _format_vtt(
    result: dict,
    *,
    max_line_width: int | None = None,
    max_line_count: int | None = None,
) -> str:
    cues = ["WEBVTT\n"]
    for segment in _successful_text_segments(result):
        text = _format_subtitle_text(
            segment["text"],
            max_line_width=max_line_width,
            max_line_count=max_line_count,
        )
        cues.append(
            f"{format_timestamp(segment['start'])} --> {format_timestamp(segment['end'])}\n"
            f"{text}\n"
        )
    return "\n".join(cues)


def _format_tsv(result: dict) -> str:
    rows = ["start\tend\ttext"]
    for segment in _successful_text_segments(result):
        rows.append(f"{segment['start']:.3f}\t{segment['end']:.3f}\t{segment['text'].strip()}")
    return "\n".join(rows) + "\n"


def _successful_text_segments(result: dict):
    for segment in result.get("segments", []):
        if "error" in segment:
            continue
        if not segment.get("text", "").strip():
            continue
        yield segment


def _format_subtitle_text(
    text: str,
    *,
    max_line_width: int | None,
    max_line_count: int | None,
) -> str:
    stripped = text.strip()
    if max_line_width is None and max_line_count is None:
        return stripped

    normalized = " ".join(stripped.split())
    if not normalized:
        return ""

    if max_line_width is None:
        lines = [normalized]
    else:
        lines = textwrap.wrap(
            normalized,
            width=max_line_width,
            break_long_words=False,
            break_on_hyphens=False,
        )
        if not lines:
            lines = [normalized]

    if max_line_count is not None and len(lines) > max_line_count:
        if max_line_count == 1:
            lines = [" ".join(lines)]
        else:
            lines = lines[: max_line_count - 1] + [" ".join(lines[max_line_count - 1 :])]
    return "\n".join(lines)


def _validate_subtitle_layout(
    *,
    max_line_width: int | None,
    max_line_count: int | None,
) -> None:
    if max_line_width is not None and (
        isinstance(max_line_width, bool) or max_line_width <= 0
    ):
        raise InvalidArgumentError("max_line_width must be greater than 0")
    if max_line_count is not None and (
        isinstance(max_line_count, bool) or max_line_count <= 0
    ):
        raise InvalidArgumentError("max_line_count must be greater than 0")


def format_timestamp(seconds: float, *, decimal_marker: str = ".") -> str:
    milliseconds = round(max(seconds, 0.0) * 1000)
    hours = milliseconds // 3_600_000
    milliseconds %= 3_600_000
    minutes = milliseconds // 60_000
    milliseconds %= 60_000
    secs = milliseconds // 1000
    millis = milliseconds % 1000
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{decimal_marker}{millis:03d}"
